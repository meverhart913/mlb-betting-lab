"""Walk-forward challenger: does leakage-safe Statcast improve pitcher K projection?"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features" / "statcast_pitcher_pregame.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit("Missing data/features/statcast_pitcher_pregame.csv; build Statcast cache first.")

    hist = build_table().copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["pitcher_id"] = pd.to_numeric(hist["pitcher_id"], errors="coerce")
    hist = hist[hist["date"].notna() & hist["pitcher_id"].notna()].copy()

    sc = pd.read_csv(FEATURES, low_memory=False)
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")
    sc["pitcher_id"] = pd.to_numeric(sc["pitcher_id"], errors="coerce")
    sc = sc.rename(columns={"game_date": "date"})
    stat_cols = [c for c in sc.columns if c.startswith("statcast_")]

    hist = hist.merge(sc[["date", "pitcher_id", *stat_cols]], on=["date", "pitcher_id"], how="left")
    _, _, base_feats = specialized_features(hist)
    challenger_feats = base_feats + stat_cols

    rows = []
    preds = []
    for season in range(2022, 2027):
        train = hist[hist["date"].dt.year < season].copy()
        test = hist[hist["date"].dt.year == season].copy()
        if len(train) < 1500 or len(test) < 500:
            continue

        for label, feats in (("v1_features", base_feats), ("v21_statcast", challenger_feats)):
            model = hgb("poisson", leaves=15, l2=2.0)
            model.fit(train[feats], train["strikeouts"])
            mu = np.clip(model.predict(test[feats]), 0.05, None)
            y = test["strikeouts"].to_numpy()
            rows.append({
                "season": season,
                "model": label,
                "starts": len(test),
                "statcast_feature_count": len(stat_cols) if label == "v21_statcast" else 0,
                "statcast_row_coverage": float(test[stat_cols].notna().any(axis=1).mean()) if stat_cols else 0.0,
                "mae": mean_absolute_error(y, mu),
                "rmse": mean_squared_error(y, mu) ** 0.5,
                "poisson_deviance": mean_poisson_deviance(y, mu),
            })
            q = test[["date", "game_id", "pitcher_id", "pitcher_name", "strikeouts"]].copy()
            q["season"] = season
            q["model"] = label
            q["projected_k"] = mu
            preds.append(q)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise SystemExit("No walk-forward seasons available for Statcast challenger.")
    metrics.to_csv(OUT / "pitcher_k_statcast_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "pitcher_k_statcast_predictions.csv", index=False)

    summary = metrics.groupby("model", as_index=False).agg(
        seasons=("season", "nunique"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_poisson_deviance=("poisson_deviance", "mean"),
        mean_statcast_coverage=("statcast_row_coverage", "mean"),
    ).sort_values("mean_poisson_deviance")
    summary.to_csv(OUT / "pitcher_k_statcast_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
