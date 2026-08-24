"""Test whether baseball features add value on top of validated market prices.

This is the relevant benchmark for a betting model: sportsbook probability is the
baseline, and baseball features must improve out-of-sample scoring or ROI beyond
that baseline. Closing lines are used here for historical research only.
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
OUT = ROOT / "outputs"


def implied(v: pd.Series) -> np.ndarray:
    x = pd.to_numeric(v, errors="coerce").astype(float)
    return np.where(x > 0, 100.0 / (x + 100.0), np.where(x < 0, -x / (-x + 100.0), np.nan))


def win_profit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    return np.where(x > 0, x / 100.0, np.where(x < 0, 100.0 / -x, np.nan))


def logit(p: pd.Series) -> pd.Series:
    q = p.clip(0.01, 0.99)
    return np.log(q / (1.0 - q))


def main() -> None:
    t = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["season"] = t["date"].dt.year
    t["close_home_odds"] = pd.to_numeric(t["close_home_odds"], errors="coerce")
    t["close_away_odds"] = pd.to_numeric(t["close_away_odds"], errors="coerce")

    valid = (
        t["close_home_odds"].abs().between(100, 5000, inclusive="both")
        & t["close_away_odds"].abs().between(100, 5000, inclusive="both")
        & t["home_win"].notna()
        & t["date"].notna()
    )
    t = t[valid].copy()
    ph = implied(t["close_home_odds"])
    pa = implied(t["close_away_odds"])
    t["market_prob"] = ph / (ph + pa)
    t["market_logit"] = logit(t["market_prob"])
    t["home_win"] = t["home_win"].astype(int)

    baseball = [c for c in t.columns if c.startswith("diff_sp_") or c.startswith("diff_team_")]
    team = [c for c in t.columns if c.startswith("diff_team_")]
    pitcher = [c for c in t.columns if c.startswith("diff_sp_")]
    feature_sets = {
        "market_plus_team": ["market_logit"] + team,
        "market_plus_pitcher": ["market_logit"] + pitcher,
        "market_plus_all": ["market_logit"] + baseball,
    }

    metric_rows = []
    bet_rows = []
    pred_rows = []
    for season in sorted(t["season"].dropna().astype(int).unique()):
        if season < 2022:
            continue
        train = t["season"] < season
        test = t["season"] == season
        if train.sum() < 1000 or test.sum() < 100:
            continue
        y_train = t.loc[train, "home_win"]
        y_test = t.loc[test, "home_win"]
        market = t.loc[test, "market_prob"]

        metric_rows.append({
            "season": season,
            "model": "market_only",
            "games": int(test.sum()),
            "log_loss": log_loss(y_test, market, labels=[0, 1]),
            "brier": brier_score_loss(y_test, market),
            "roc_auc": roc_auc_score(y_test, market),
        })

        for name, features in feature_sets.items():
            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("m", LogisticRegression(max_iter=3000, C=0.2)),
            ])
            model.fit(t.loc[train, features], y_train)
            prob = model.predict_proba(t.loc[test, features])[:, 1]
            metric_rows.append({
                "season": season,
                "model": name,
                "games": int(test.sum()),
                "log_loss": log_loss(y_test, prob, labels=[0, 1]),
                "brier": brier_score_loss(y_test, prob),
                "roc_auc": roc_auc_score(y_test, prob),
            })

            z = t.loc[test, ["game_id", "date", "home_team", "away_team", "home_win", "market_prob", "close_home_odds", "close_away_odds"]].copy()
            z["model"] = name
            z["hybrid_prob"] = prob
            pred_rows.append(z)

            for threshold in (0.01, 0.02, 0.03, 0.05):
                edge = z["hybrid_prob"] - z["market_prob"]
                b = z[(edge >= threshold) | (edge <= -threshold)].copy()
                if b.empty:
                    continue
                be = b["hybrid_prob"] - b["market_prob"]
                b["side"] = np.where(be >= threshold, "home", "away")
                b["won"] = np.where(b["side"] == "home", b["home_win"] == 1, b["home_win"] == 0)
                price = np.where(b["side"] == "home", b["close_home_odds"], b["close_away_odds"])
                profit = np.where(b["won"], win_profit(price), -1.0)
                bet_rows.append({
                    "season": season,
                    "model": name,
                    "edge_threshold": threshold,
                    "bets": int(len(b)),
                    "wins": int(b["won"].sum()),
                    "hit_rate": float(b["won"].mean()),
                    "roi": float(profit.mean()),
                })

    metrics = pd.DataFrame(metric_rows)
    bets = pd.DataFrame(bet_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics.to_csv(OUT / "hybrid_market_metrics.csv", index=False)
    bets.to_csv(OUT / "hybrid_market_edge_backtest.csv", index=False)
    preds.to_csv(OUT / "hybrid_market_predictions.csv", index=False)

    summary = metrics.groupby("model", as_index=False).agg(
        seasons=("season", "count"),
        mean_log_loss=("log_loss", "mean"),
        mean_brier=("brier", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
    ).sort_values("mean_log_loss")
    summary.to_csv(OUT / "hybrid_market_summary.csv", index=False)
    print("\nHYBRID MARKET SUMMARY")
    print(summary.round(5).to_string(index=False))
    print("\nHYBRID EDGE BACKTEST")
    print(bets.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
