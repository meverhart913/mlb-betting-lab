"""Measure incremental predictive value of pitcher and team feature families."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    table = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table = table[table["home_win"].notna() & table["date"].notna()].copy()
    table["home_win"] = table["home_win"].astype(int)
    table["season"] = table["date"].dt.year

    pitcher = [c for c in table.columns if c.startswith("diff_sp_")]
    team = [c for c in table.columns if c.startswith("diff_team_")]
    sets = {"pitcher_only": pitcher, "team_only": team, "combined": pitcher + team}
    if not pitcher or not team:
        raise ValueError(f"expected both pitcher and team features; found {len(pitcher)} pitcher and {len(team)} team")

    model_factories = {
        "logistic": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("m", LogisticRegression(max_iter=2500, C=0.5)),
        ]),
        "hist_gb": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=2.0, random_state=42)),
        ]),
    }

    rows = []
    years = table["season"]
    for season in range(max(2021, int(years.min()) + 2), int(years.max()) + 1):
        train = years < season
        test = years == season
        if train.sum() < 500 or test.sum() == 0:
            continue
        actual = table.loc[test, "home_win"]
        for set_name, features in sets.items():
            for model_name, factory in model_factories.items():
                model = factory()
                model.fit(table.loc[train, features], table.loc[train, "home_win"])
                prob = model.predict_proba(table.loc[test, features])[:, 1]
                rows.append({
                    "season": season,
                    "feature_set": set_name,
                    "model": model_name,
                    "features": len(features),
                    "games": int(test.sum()),
                    "accuracy": accuracy_score(actual, prob >= 0.5),
                    "log_loss": log_loss(actual, prob, labels=[0, 1]),
                    "brier": brier_score_loss(actual, prob),
                    "roc_auc": roc_auc_score(actual, prob) if actual.nunique() > 1 else np.nan,
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "feature_set_walkforward_comparison.csv", index=False)

    summary = results.groupby(["feature_set", "model"], as_index=False).agg(
        seasons=("season", "count"),
        mean_accuracy=("accuracy", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_brier=("brier", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
    )
    summary.to_csv(OUT / "feature_set_summary.csv", index=False)
    print("\nFEATURE SET WALK-FORWARD COMPARISON")
    print(results.round(4).to_string(index=False))
    print("\nFEATURE SET SUMMARY")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
