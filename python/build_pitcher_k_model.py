"""Build and walk-forward test a starting-pitcher strikeout projection model.

Target: strikeouts recorded by the starting pitcher. Features are strictly
pregame: prior starter workload/performance plus opponent offensive form from
games before the target game.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PITCHER_WINDOWS = (3, 5, 10)
TEAM_WINDOWS = (10, 30)


def ip_to_outs(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        s = str(v)
        if "." not in s:
            return float(int(s) * 3)
        a, b = s.split(".", 1)
        b = int(b)
        return float(int(a) * 3 + b) if b in (0, 1, 2) else np.nan
    except Exception:
        return np.nan


def build_table() -> pd.DataFrame:
    p = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    t = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    p["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(int)
    p = p[p["is_starter"].eq(1)].copy()
    p["outs"] = p["innings_pitched"].map(ip_to_outs)
    p["ip"] = p["outs"] / 3.0
    for c in ["strikeouts", "walks", "hits", "earned_runs", "home_runs", "batters_faced", "pitches"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.sort_values(["pitcher_id", "date", "game_id"])
    p["days_rest"] = p.groupby("pitcher_id")["date"].diff().dt.days

    for w in PITCHER_WINDOWS:
        grp = p.groupby("pitcher_id", group_keys=False)
        minp = max(2, w // 2)
        for c in ["strikeouts", "walks", "hits", "earned_runs", "home_runs", "batters_faced", "pitches", "outs"]:
            p[f"sp_{c}_{w}"] = grp[c].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).mean())
        prev_k = grp["strikeouts"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        prev_bf = grp["batters_faced"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        prev_pitches = grp["pitches"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        prev_outs = grp["outs"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        p[f"sp_k_rate_{w}"] = prev_k / prev_bf.replace(0, np.nan)
        p[f"sp_k_per_100_pitches_{w}"] = 100.0 * prev_k / prev_pitches.replace(0, np.nan)
        p[f"sp_pitches_per_bf_{w}"] = prev_pitches / prev_bf.replace(0, np.nan)
        p[f"sp_outs_per_bf_{w}"] = prev_outs / prev_bf.replace(0, np.nan)

    for c in ["strikeouts", "at_bats", "walks", "runs", "hits", "home_runs"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce")
    t = t.sort_values(["team_id", "date", "game_id"])
    created_team_features = []
    for w in TEAM_WINDOWS:
        grp = t.groupby("team_id", group_keys=False)
        minp = max(5, w // 2)
        k = grp["strikeouts"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        ab = grp["at_bats"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        bb = grp["walks"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        hits = grp["hits"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).sum())
        pa_game = grp["at_bats"].transform(lambda x: (x + t.loc[x.index, "walks"]).shift(1).rolling(w, min_periods=minp).mean())
        names = [
            f"team_k_per_ab_{w}", f"team_k_per_pa_{w}", f"team_runs_{w}",
            f"team_walk_per_pa_{w}", f"team_hits_per_ab_{w}", f"team_pa_per_game_{w}",
        ]
        t[names[0]] = k / ab.replace(0, np.nan)
        t[names[1]] = k / (ab + bb).replace(0, np.nan)
        t[names[2]] = grp["runs"].transform(lambda x: x.shift(1).rolling(w, min_periods=minp).mean())
        t[names[3]] = bb / (ab + bb).replace(0, np.nan)
        t[names[4]] = hits / ab.replace(0, np.nan)
        t[names[5]] = pa_game
        created_team_features.extend(names)

    ids = t[["game_id", "side", "team_id"]].drop_duplicates(["game_id", "side"])
    home_ids = ids[ids.side.eq("home")][["game_id", "team_id"]].rename(columns={"team_id": "home_team_id"})
    away_ids = ids[ids.side.eq("away")][["game_id", "team_id"]].rename(columns={"team_id": "away_team_id"})
    p = p.merge(home_ids, on="game_id", how="left").merge(away_ids, on="game_id", how="left")
    p["opponent_team_id"] = np.where(p["side"].eq("home"), p["away_team_id"], p["home_team_id"])

    opp = t[["game_id", "team_id"] + created_team_features].copy()
    opp = opp.rename(columns={c: f"opp_{c}" for c in created_team_features})
    z = p.merge(opp, left_on=["game_id", "opponent_team_id"], right_on=["game_id", "team_id"], how="left", suffixes=("", "_opp"))
    z["season"] = z["date"].dt.year
    return z[z["strikeouts"].notna() & z["date"].notna()].copy()


def feature_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("sp_") or c.startswith("opp_team_")] + ["days_rest"])


def candidate_models():
    return {
        "poisson_glm": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.5, max_iter=2000)),
        ]),
        "hist_poisson": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=913)),
        ]),
    }


def main() -> None:
    z = build_table()
    feats = feature_cols(z)
    rows, pred_frames = [], []
    for year in sorted(z["season"].dropna().astype(int).unique()):
        if year < 2022:
            continue
        train = z["season"] < year
        test = z["season"] == year
        if train.sum() < 1500 or test.sum() < 300:
            continue
        for name, model in candidate_models().items():
            model.fit(z.loc[train, feats], z.loc[train, "strikeouts"])
            mu = np.clip(model.predict(z.loc[test, feats]), 0.05, None)
            y = z.loc[test, "strikeouts"].to_numpy(float)
            rows.append({
                "season": year, "model": name, "starts": int(test.sum()),
                "mae": mean_absolute_error(y, mu),
                "rmse": root_mean_squared_error(y, mu),
                "poisson_deviance": mean_poisson_deviance(y, mu),
                "mean_actual_k": float(np.mean(y)), "mean_projected_k": float(np.mean(mu)),
            })
            q = z.loc[test, ["game_id", "date", "pitcher_id", "pitcher_name", "side", "strikeouts", "batters_faced", "pitches"]].copy()
            q["model"] = name
            q["projected_k"] = mu
            pred_frames.append(q)

    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    summary = metrics.groupby("model", as_index=False).agg(
        seasons=("season", "count"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
        mean_poisson_deviance=("poisson_deviance", "mean"),
    ).sort_values("mean_poisson_deviance")
    metrics.to_csv(OUT / "pitcher_k_walkforward_metrics.csv", index=False)
    preds.to_csv(OUT / "pitcher_k_walkforward_predictions.csv", index=False)
    summary.to_csv(OUT / "pitcher_k_model_summary.csv", index=False)
    z.to_csv(OUT / "pitcher_k_modeling_table.csv", index=False)
    print("PITCHER K WALK-FORWARD SUMMARY")
    print(summary.round(5).to_string(index=False))

if __name__ == "__main__":
    main()
