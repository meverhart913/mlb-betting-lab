"""Grade completed Pitcher-K V2 projections without altering V1 history."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"
HIST = CURRENT / "pitcher_k_v2_projection_history.csv"
GRADED = OUT / "pitcher_k_v2_prospective_graded.csv"
SUMMARY = OUT / "pitcher_k_v2_prospective_summary.csv"


def main() -> None:
    if not HIST.exists():
        print("No V2 projection history yet.")
        return
    h = pd.read_csv(HIST, low_memory=False)
    logs = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs["game_id"] = pd.to_numeric(logs["game_id"], errors="coerce")
    logs["is_starter"] = pd.to_numeric(logs["is_starter"], errors="coerce").fillna(0).astype(int)
    logs["strikeouts"] = pd.to_numeric(logs["strikeouts"], errors="coerce")
    s = logs[logs["is_starter"].eq(1)][["game_id", "pitcher_id", "strikeouts"]].drop_duplicates(["game_id", "pitcher_id"], keep="last")
    h["game_id"] = pd.to_numeric(h["game_id"], errors="coerce")
    h["pitcher_id"] = pd.to_numeric(h["pitcher_id"], errors="coerce")
    g = h.merge(s, on=["game_id", "pitcher_id"], how="left")
    g["actual_k"] = pd.to_numeric(g["strikeouts"], errors="coerce")
    g = g[g["actual_k"].notna()].copy()
    OUT.mkdir(exist_ok=True)
    if g.empty:
        print("V2 history archived; no completed starts gradeable yet.")
        return
    g["v2_error"] = pd.to_numeric(g["v2_projected_k"], errors="coerce") - g["actual_k"]
    g["v1_error"] = pd.to_numeric(g["v1_projected_k"], errors="coerce") - g["actual_k"]
    g["v2_abs_error"] = g["v2_error"].abs()
    g["v1_abs_error"] = g["v1_error"].abs()
    g["actual_over"] = (g["actual_k"] > pd.to_numeric(g["line"], errors="coerce")).astype(float)
    g["v2_over_brier"] = (pd.to_numeric(g["v2_fair_over_prob"], errors="coerce") - g["actual_over"]) ** 2
    g["v1_over_brier"] = (pd.to_numeric(g["fair_over_prob"], errors="coerce") - g["actual_over"]) ** 2
    g.to_csv(GRADED, index=False)
    summary = pd.DataFrame([{
        "graded_rows": len(g),
        "unique_starts": int(g[["game_id", "pitcher_id"]].drop_duplicates().shape[0]),
        "v1_mae": float(g["v1_abs_error"].mean()),
        "v2_mae": float(g["v2_abs_error"].mean()),
        "mae_improvement_v2_minus_v1": float(g["v2_abs_error"].mean() - g["v1_abs_error"].mean()),
        "v1_over_brier": float(g["v1_over_brier"].mean()),
        "v2_over_brier": float(g["v2_over_brier"].mean()),
        "brier_improvement_v2_minus_v1": float(g["v2_over_brier"].mean() - g["v1_over_brier"].mean()),
    }])
    summary.to_csv(SUMMARY, index=False)
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
