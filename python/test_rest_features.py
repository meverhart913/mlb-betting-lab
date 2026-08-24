"""Test whether schedule/fatigue information adds out-of-sample signal.

Rest features are known before first pitch and are calculated only from prior game
dates. The script compares the existing baseball model with and without rest, and
then repeats the comparison on top of the validated sportsbook market baseline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(max_iter=3000, C=0.3)),
    ])


def implied(v: pd.Series) -> np.ndarray:
    x = pd.to_numeric(v, errors="coerce").astype(float)
    return np.where(x > 0, 100.0 / (x + 100.0), np.where(x < 0, -x / (-x + 100.0), np.nan))


def logit(p: pd.Series) -> pd.Series:
    q = p.clip(0.01, 0.99)
    return np.log(q / (1.0 - q))


def build_rest() -> pd.DataFrame:
    teams = pd.read_csv(DATA / "mlb_team_game_logs.csv", usecols=["game_id", "date", "side", "team_id"])
    teams["date"] = pd.to_datetime(teams["date"], errors="coerce")
    teams = teams.sort_values(["team_id", "date", "game_id"])
    teams["team_rest_days"] = teams.groupby("team_id")["date"].diff().dt.days.clip(lower=0, upper=14)
    tw = teams[["game_id", "side", "team_rest_days"]].drop_duplicates(["game_id", "side"], keep="last").pivot(index="game_id", columns="side")
    tw.columns = [f"{side}_team_rest_days" for _, side in tw.columns]
    tw = tw.reset_index()

    pitchers = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", usecols=["game_id", "date", "side", "pitcher_id", "is_starter"])
    pitchers["date"] = pd.to_datetime(pitchers["date"], errors="coerce")
    pitchers["is_starter"] = pd.to_numeric(pitchers["is_starter"], errors="coerce").fillna(0).astype(int)
    starters = pitchers[pitchers["is_starter"] == 1].sort_values(["pitcher_id", "date", "game_id"]).copy()
    starters["starter_rest_days"] = starters.groupby("pitcher_id")["date"].diff().dt.days.clip(lower=0, upper=30)
    sw = starters[["game_id", "side", "starter_rest_days"]].drop_duplicates(["game_id", "side"], keep="last").pivot(index="game_id", columns="side")
    sw.columns = [f"{side}_starter_rest_days" for _, side in sw.columns]
    sw = sw.reset_index()

    r = tw.merge(sw, on="game_id", how="outer")
    r["diff_team_rest_days"] = r.get("home_team_rest_days") - r.get("away_team_rest_days")
    r["diff_starter_rest_days"] = r.get("home_starter_rest_days") - r.get("away_starter_rest_days")
    r["home_team_short_rest"] = (r.get("home_team_rest_days") <= 1).astype(float)
    r["away_team_short_rest"] = (r.get("away_team_rest_days") <= 1).astype(float)
    r["home_starter_short_rest"] = (r.get("home_starter_rest_days") <= 4).astype(float)
    r["away_starter_short_rest"] = (r.get("away_starter_rest_days") <= 4).astype(float)
    return r


def main():
    t = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["season"] = t["date"].dt.year
    t = t.merge(build_rest(), on="game_id", how="left")
    t = t[t["home_win"].notna() & t["date"].notna()].copy()
    t["home_win"] = t["home_win"].astype(int)

    base = [c for c in t.columns if c.startswith("diff_sp_") or c.startswith("diff_team_") and not c.endswith("rest_days")]
    base = [c for c in base if c not in {"diff_team_rest_days", "diff_starter_rest_days"}]
    rest = [
        "home_team_rest_days", "away_team_rest_days", "diff_team_rest_days",
        "home_starter_rest_days", "away_starter_rest_days", "diff_starter_rest_days",
        "home_team_short_rest", "away_team_short_rest", "home_starter_short_rest", "away_starter_short_rest",
    ]

    rows = []
    pred_rows = []
    for season in range(max(2021, int(t["season"].min()) + 2), int(t["season"].max()) + 1):
        tr = t["season"] < season
        te = t["season"] == season
        if tr.sum() < 500 or te.sum() == 0:
            continue
        ytr = t.loc[tr, "home_win"]
        yte = t.loc[te, "home_win"]
        for name, features in {"baseball_base": base, "baseball_plus_rest": base + rest}.items():
            m = model(); m.fit(t.loc[tr, features], ytr); p = m.predict_proba(t.loc[te, features])[:, 1]
            rows.append({"season": season, "model": name, "games": int(te.sum()), "log_loss": log_loss(yte, p, labels=[0,1]), "brier": brier_score_loss(yte, p), "roc_auc": roc_auc_score(yte, p)})

        valid_market = (
            t.loc[te, "close_home_odds"].abs().between(100, 5000, inclusive="both")
            & t.loc[te, "close_away_odds"].abs().between(100, 5000, inclusive="both")
        ) if {"close_home_odds", "close_away_odds"}.issubset(t.columns) else pd.Series(False, index=t.index[te])
        idx = t.index[te][valid_market]
        if len(idx) < 100:
            continue
        ph = implied(t.loc[idx, "close_home_odds"]); pa = implied(t.loc[idx, "close_away_odds"])
        market_prob = pd.Series(ph / (ph + pa), index=idx)
        t.loc[idx, "market_logit_rest_test"] = logit(market_prob)
        ym = t.loc[idx, "home_win"]
        rows.append({"season": season, "model": "market_only", "games": len(idx), "log_loss": log_loss(ym, market_prob, labels=[0,1]), "brier": brier_score_loss(ym, market_prob), "roc_auc": roc_auc_score(ym, market_prob)})

        train_market = (
            tr
            & t["close_home_odds"].abs().between(100, 5000, inclusive="both")
            & t["close_away_odds"].abs().between(100, 5000, inclusive="both")
        )
        train_idx = t.index[train_market]
        if len(train_idx) < 500:
            continue
        phtr = implied(t.loc[train_idx, "close_home_odds"]); patr = implied(t.loc[train_idx, "close_away_odds"])
        train_mp = pd.Series(phtr / (phtr + patr), index=train_idx)
        t.loc[train_idx, "market_logit_rest_test"] = logit(train_mp)
        for name, features in {
            "market_plus_baseball": ["market_logit_rest_test"] + base,
            "market_plus_baseball_rest": ["market_logit_rest_test"] + base + rest,
        }.items():
            m = model(); m.fit(t.loc[train_idx, features], t.loc[train_idx, "home_win"]); p = m.predict_proba(t.loc[idx, features])[:, 1]
            rows.append({"season": season, "model": name, "games": len(idx), "log_loss": log_loss(ym, p, labels=[0,1]), "brier": brier_score_loss(ym, p), "roc_auc": roc_auc_score(ym, p)})
            z = t.loc[idx, ["game_id", "date", "home_team", "away_team", "home_win"]].copy(); z["model"] = name; z["prob"] = p; z["market_prob"] = market_prob; pred_rows.append(z)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "rest_feature_walkforward.csv", index=False)
    summary = results.groupby("model", as_index=False).agg(seasons=("season","count"), mean_log_loss=("log_loss","mean"), mean_brier=("brier","mean"), mean_roc_auc=("roc_auc","mean")).sort_values("mean_log_loss")
    summary.to_csv(OUT / "rest_feature_summary.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "rest_feature_predictions.csv", index=False)
    print("\nREST FEATURE SUMMARY")
    print(summary.round(5).to_string(index=False))
    print("\nREST FEATURE WALK-FORWARD")
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
