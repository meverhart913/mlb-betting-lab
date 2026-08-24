"""Test each bullpen workload feature individually against the current baseline."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from test_bullpen_features import bullpen_features, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    base = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base[base["home_win"].notna() & base["date"].notna()].copy()
    z = base.merge(bullpen_features(), on="game_id", how="left")
    baseline = sorted([c for c in z.columns if c.startswith("diff_sp_") or c.startswith("diff_team_")])
    bp_cols = sorted([c for c in z.columns if c.startswith("diff_bp_")])

    base_rows = pd.DataFrame(evaluate(z, baseline, "baseline"))
    base_mean = base_rows["log_loss"].mean()
    rows = []
    for c in bp_cols:
        test = pd.DataFrame(evaluate(z, baseline + [c], c))
        joined = test.merge(base_rows[["season", "log_loss"]], on="season", suffixes=("_test", "_base"))
        rows.append({
            "feature": c,
            "mean_log_loss": test["log_loss"].mean(),
            "delta_log_loss_vs_baseline": test["log_loss"].mean() - base_mean,
            "seasons_improved": int((joined["log_loss_test"] < joined["log_loss_base"]).sum()),
            "seasons_tested": int(len(joined)),
            "mean_auc": test["auc"].mean(),
        })
    out = pd.DataFrame(rows).sort_values(["delta_log_loss_vs_baseline", "feature"])
    out.to_csv(OUT / "bullpen_feature_ablation.csv", index=False)
    print(f"Baseline mean log loss: {base_mean:.6f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
