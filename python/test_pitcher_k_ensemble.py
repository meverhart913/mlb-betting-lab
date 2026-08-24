"""Walk-forward test specialized BF/K-rate components and direct-model blends."""
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
OUT = ROOT / "outputs"


def hgb(loss="squared_error", leaves=15, l2=3.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(
            loss=loss, max_iter=300, learning_rate=0.04, max_leaf_nodes=leaves,
            l2_regularization=l2, random_state=913,
        )),
    ])


def specialized_features(z: pd.DataFrame):
    all_feats = feature_cols(z)
    # Opportunity/leash: prior BF, pitch count, outs, effectiveness, rest, opponent run environment.
    bf = [c for c in all_feats if any(k in c for k in (
        "batters_faced", "pitches_", "outs_", "earned_runs_", "walks_", "hits_", "home_runs_", "days_rest", "opp_team_runs_"
    ))]
    # Strikeout rate: prior K skill plus opponent strikeout tendency. Keep workload variables out.
    kr = [c for c in all_feats if any(k in c for k in (
        "k_rate_", "k_per_100_pitches_", "strikeouts_", "opp_team_k_per_"
    ))]
    return sorted(set(bf)), sorted(set(kr)), all_feats


def main() -> None:
    z = build_table()
    z = z[pd.to_numeric(z["batters_faced"], errors="coerce").gt(0)].copy()
    z["k_rate_target"] = (z["strikeouts"] / z["batters_faced"]).clip(0, 0.7)
    bf_feats, kr_feats, all_feats = specialized_features(z)
    if not bf_feats or not kr_feats:
        raise SystemExit("Specialized feature sets are empty")

    rows = []
    pred_frames = []
    for year in sorted(z["season"].dropna().astype(int).unique()):
        if year < 2022:
            continue
        train = z["season"] < year
        test = z["season"] == year
        if train.sum() < 1500 or test.sum() < 300:
            continue

        bf = hgb("poisson", leaves=12, l2=4.0)
        kr = hgb("squared_error", leaves=12, l2=4.0)
        direct = hgb("poisson", leaves=15, l2=2.0)
        bf.fit(z.loc[train, bf_feats], z.loc[train, "batters_faced"])
        kr.fit(z.loc[train, kr_feats], z.loc[train, "k_rate_target"], m__sample_weight=z.loc[train, "batters_faced"])
        direct.fit(z.loc[train, all_feats], z.loc[train, "strikeouts"])

        bf_hat = np.clip(bf.predict(z.loc[test, bf_feats]), 5, 40)
        kr_hat = np.clip(kr.predict(z.loc[test, kr_feats]), 0.02, 0.55)
        component = np.clip(bf_hat * kr_hat, 0.05, None)
        direct_hat = np.clip(direct.predict(z.loc[test, all_feats]), 0.05, None)
        y = z.loc[test, "strikeouts"].to_numpy(float)

        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            # w=1 is pure specialized component; w=0 is pure direct model.
            mu = np.clip(w * component + (1.0 - w) * direct_hat, 0.05, None)
            rows.append({
                "season": year, "component_weight": w, "starts": int(test.sum()),
                "mae": mean_absolute_error(y, mu),
                "rmse": root_mean_squared_error(y, mu),
                "poisson_deviance": mean_poisson_deviance(y, mu),
                "mean_actual_k": float(y.mean()), "mean_projected_k": float(mu.mean()),
            })
            q = z.loc[test, ["game_id","date","pitcher_id","pitcher_name","strikeouts","batters_faced"]].copy()
            q["season"] = year
            q["component_weight"] = w
            q["projected_bf"] = bf_hat
            q["projected_k_rate"] = kr_hat
            q["component_k"] = component
            q["direct_k"] = direct_hat
            q["projected_k"] = mu
            pred_frames.append(q)

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("component_weight", as_index=False).agg(
        seasons=("season","count"), mean_mae=("mae","mean"), mean_rmse=("rmse","mean"),
        mean_poisson_deviance=("poisson_deviance","mean"),
    ).sort_values(["mean_poisson_deviance","mean_mae"])
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics.to_csv(OUT / "pitcher_k_ensemble_metrics.csv", index=False)
    summary.to_csv(OUT / "pitcher_k_ensemble_summary.csv", index=False)
    preds.to_csv(OUT / "pitcher_k_ensemble_predictions.csv", index=False)
    print("SPECIALIZED BF x K-RATE + DIRECT MODEL BLEND")
    print(summary.round(6).to_string(index=False))
    print(f"BF features: {len(bf_feats)}; K-rate features: {len(kr_feats)}; direct features: {len(all_feats)}")

if __name__ == "__main__":
    main()
