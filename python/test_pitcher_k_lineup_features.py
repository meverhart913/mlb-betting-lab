"""Test whether posted/actual starting-lineup K tendency improves pitcher K projections.

Player K rates use only prior game-days, so same-day doubleheaders never leak an
earlier game result into the later game's pregame feature. Historical starting
lineups are used as a posted-lineup research scenario. Nothing is promoted unless
walk-forward count fit improves on the identical eligible-start subset.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline

from build_pitcher_k_model import build_table, feature_cols

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
BATTERS = DERIVED / "mlb_batter_game_logs.csv"
STATUS = OUT / "pitcher_k_lineup_feature_status.csv"
METRICS = OUT / "pitcher_k_lineup_feature_metrics.csv"
SUMMARY = OUT / "pitcher_k_lineup_feature_summary.csv"
WINDOWS = (20, 50)


def hgb():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.04,
            max_leaf_nodes=15, l2_regularization=2.5, random_state=913,
        )),
    ])


def build_lineup_features() -> pd.DataFrame:
    b = pd.read_csv(BATTERS, low_memory=False)
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    for c in ["player_id","strikeouts","approx_plate_appearances","in_starting_lineup","batting_order"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[b["date"].notna() & b["player_id"].notna()].copy()

    # Aggregate by player-day first. This makes every game in a doubleheader use
    # the same prior-day-only player feature and removes ambiguous intra-day order.
    daily = b.groupby(["player_id","date"], as_index=False).agg(
        k=("strikeouts","sum"), pa=("approx_plate_appearances","sum")
    ).sort_values(["player_id","date"])
    grp = daily.groupby("player_id", group_keys=False)
    feature_names = []
    for w in WINDOWS:
        prior_k = grp["k"].transform(lambda x: x.shift(1).rolling(w, min_periods=5).sum())
        prior_pa = grp["pa"].transform(lambda x: x.shift(1).rolling(w, min_periods=5).sum())
        name = f"player_k_rate_{w}d"
        daily[name] = prior_k / prior_pa.replace(0, np.nan)
        daily[f"player_prior_pa_{w}d"] = prior_pa
        feature_names.append(name)

    starters = b[b["in_starting_lineup"].eq(1)].copy()
    starters = starters.merge(
        daily[["player_id","date"] + feature_names + [f"player_prior_pa_{w}d" for w in WINDOWS]],
        on=["player_id","date"], how="left"
    )
    aggs = []
    for (game_id, side), g in starters.groupby(["game_id","side"], dropna=False):
        row = {"game_id": game_id, "opponent_side": side, "lineup_starter_count": int(g["player_id"].nunique())}
        for w in WINDOWS:
            rate = f"player_k_rate_{w}d"
            pa = f"player_prior_pa_{w}d"
            known = g[rate].notna()
            row[f"lineup_known_{w}d"] = int(known.sum())
            row[f"lineup_k_rate_{w}d"] = float(g.loc[known, rate].mean()) if known.any() else np.nan
            weights = pd.to_numeric(g.loc[known, pa], errors="coerce").clip(lower=1)
            row[f"lineup_k_rate_pa_weighted_{w}d"] = (
                float(np.average(g.loc[known, rate], weights=weights))
                if known.any() and weights.notna().all() and weights.sum() > 0 else np.nan
            )
        aggs.append(row)
    return pd.DataFrame(aggs)


def main() -> None:
    if not BATTERS.exists():
        pd.DataFrame([{"status":"waiting_for_batter_history","eligible_starts":0,"promote":False}]).to_csv(STATUS,index=False)
        print("Batter history not available yet; lineup feature test deferred.")
        return

    lineup = build_lineup_features()
    if lineup.empty:
        pd.DataFrame([{"status":"waiting_for_batter_history","eligible_starts":0,"promote":False}]).to_csv(STATUS,index=False)
        return

    z = build_table()
    z["opponent_side"] = np.where(z["side"].eq("home"), "away", "home")
    z = z.merge(lineup, on=["game_id","opponent_side"], how="left")
    z["season"] = pd.to_numeric(z["season"], errors="coerce")
    lineup_feats = [
        "lineup_k_rate_20d","lineup_k_rate_pa_weighted_20d",
        "lineup_k_rate_50d","lineup_k_rate_pa_weighted_50d",
    ]
    eligible = z[
        z["lineup_starter_count"].ge(9) & z["lineup_known_20d"].ge(7) & z["lineup_known_50d"].ge(7)
    ].copy()
    max_season = int(eligible["season"].max()) if eligible["season"].notna().any() else 0
    if len(eligible) < 2000 or max_season < 2022:
        pd.DataFrame([{
            "status":"insufficient_historical_coverage","eligible_starts":len(eligible),
            "max_season":max_season,"promote":False,
        }]).to_csv(STATUS,index=False)
        print(f"Lineup feature test deferred: {len(eligible):,} eligible starts through {max_season}.")
        return

    base = feature_cols(eligible)
    rows=[]
    for year in sorted(eligible["season"].dropna().astype(int).unique()):
        if year < 2022: continue
        train = eligible["season"] < year
        test = eligible["season"] == year
        if train.sum() < 1500 or test.sum() < 250: continue
        y = eligible.loc[test,"strikeouts"].to_numpy(float)
        for name, feats in {
            "team_level": base,
            "team_plus_lineup": base + lineup_feats,
        }.items():
            m=hgb(); m.fit(eligible.loc[train,feats], eligible.loc[train,"strikeouts"])
            mu=np.clip(m.predict(eligible.loc[test,feats]),0.05,None)
            rows.append({
                "season":year,"model":name,"starts":int(test.sum()),
                "mae":mean_absolute_error(y,mu),"rmse":root_mean_squared_error(y,mu),
                "poisson_deviance":mean_poisson_deviance(y,mu),
            })

    metrics=pd.DataFrame(rows)
    metrics.to_csv(METRICS,index=False)
    if metrics.empty:
        pd.DataFrame([{"status":"insufficient_walkforward_seasons","eligible_starts":len(eligible),"promote":False}]).to_csv(STATUS,index=False)
        return
    summary=metrics.groupby("model",as_index=False).agg(
        seasons=("season","count"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),
        mean_poisson_deviance=("poisson_deviance","mean"),
    )
    base_row=summary[summary.model.eq("team_level")].iloc[0]
    line_row=summary[summary.model.eq("team_plus_lineup")].iloc[0]
    dev_gain=float(base_row.mean_poisson_deviance-line_row.mean_poisson_deviance)
    mae_gain=float(base_row.mean_mae-line_row.mean_mae)
    # Require both metrics to improve and enough held-out seasons to avoid an
    # automatic promotion from one lucky slice.
    promote=bool(line_row.seasons >= 3 and dev_gain > 0 and mae_gain > 0)
    summary["dev_improvement_vs_team"] = np.where(summary.model.eq("team_plus_lineup"),dev_gain,0.0)
    summary["mae_improvement_vs_team"] = np.where(summary.model.eq("team_plus_lineup"),mae_gain,0.0)
    summary["promote"] = np.where(summary.model.eq("team_plus_lineup"),promote,False)
    summary.to_csv(SUMMARY,index=False)
    pd.DataFrame([{
        "status":"evaluated","eligible_starts":len(eligible),"max_season":max_season,
        "heldout_seasons":int(line_row.seasons),"dev_improvement":dev_gain,
        "mae_improvement":mae_gain,"promote":promote,
    }]).to_csv(STATUS,index=False)
    print(summary.round(6).to_string(index=False))
    print(f"Lineup feature promotion gate: {promote}")

if __name__=="__main__":
    main()
