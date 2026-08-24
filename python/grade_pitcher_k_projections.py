"""Archive and grade prospective pitcher strikeout projections.

This is an evaluation layer, not a betting executor. Each timestamped projection
snapshot is preserved independently so first-vs-latest market comparisons and
future CLV analysis remain possible. Completed starts are graded automatically.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"
PRED = OUT / "pitcher_k_prop_predictions.csv"
HISTORY = CURRENT / "pitcher_k_projection_history.csv"
GRADED = OUT / "pitcher_k_prospective_graded.csv"
SUMMARY = OUT / "pitcher_k_prospective_summary.csv"


def append_current() -> None:
    if not PRED.exists():
        return
    fresh = pd.read_csv(PRED, low_memory=False)
    if fresh.empty:
        return
    if HISTORY.exists():
        old = pd.read_csv(HISTORY, low_memory=False)
        h = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        h = fresh.copy()
    keys = [c for c in ["date", "event_id", "pitcher_id", "line", "projection_model", "snapshot_time_et"] if c in h.columns]
    if keys:
        h = h.drop_duplicates(keys, keep="last")
    CURRENT.mkdir(parents=True, exist_ok=True)
    h.to_csv(HISTORY, index=False)


def main() -> None:
    append_current()
    if not HISTORY.exists():
        print("No prospective pitcher K projection history yet.")
        return
    h = pd.read_csv(HISTORY, low_memory=False)
    logs = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    logs["is_starter"] = pd.to_numeric(logs["is_starter"], errors="coerce").fillna(0).astype(int)
    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs["strikeouts"] = pd.to_numeric(logs["strikeouts"], errors="coerce")
    starters = logs[logs["is_starter"].eq(1)][["game_id", "pitcher_id", "strikeouts", "batters_faced", "pitches"]].drop_duplicates(["game_id", "pitcher_id"], keep="last")
    h["game_id"] = pd.to_numeric(h.get("game_id"), errors="coerce")
    h["pitcher_id"] = pd.to_numeric(h.get("pitcher_id"), errors="coerce")
    g = h.merge(starters, on=["game_id", "pitcher_id"], how="left", suffixes=("", "_actual"))
    g["actual_k"] = pd.to_numeric(g["strikeouts"], errors="coerce")
    g["projection_error"] = g["projected_k"] - g["actual_k"]
    g["absolute_error"] = g["projection_error"].abs()
    g["squared_error"] = g["projection_error"] ** 2
    g["actual_over"] = np.where(g["actual_k"].notna(), (g["actual_k"] > g["line"]).astype(float), np.nan)
    g["actual_under"] = np.where(g["actual_k"].notna(), (g["actual_k"] < g["line"]).astype(float), np.nan)
    g["actual_push"] = np.where(g["actual_k"].notna(), (g["actual_k"] == g["line"]).astype(float), np.nan)
    g["over_brier"] = np.where(g["actual_k"].notna(), (g["fair_over_prob"] - g["actual_over"]) ** 2, np.nan)
    graded = g[g["actual_k"].notna()].copy()
    OUT.mkdir(exist_ok=True)
    graded.to_csv(GRADED, index=False)
    if graded.empty:
        print("Projection history archived; no completed projected starts are gradeable yet.")
        return

    snap = graded.copy()
    if "snapshot_time_et" in snap.columns:
        snap["snapshot_time_et"] = pd.to_datetime(snap["snapshot_time_et"], errors="coerce")
    start_keys = ["game_id", "pitcher_id", "line"]
    first = snap.sort_values("snapshot_time_et").drop_duplicates(start_keys, keep="first") if "snapshot_time_et" in snap.columns else snap.drop_duplicates(start_keys)
    latest = snap.sort_values("snapshot_time_et").drop_duplicates(start_keys, keep="last") if "snapshot_time_et" in snap.columns else snap.drop_duplicates(start_keys)

    summary = pd.DataFrame([{
        "graded_snapshots": int(len(graded)),
        "unique_starts": int(graded[["game_id", "pitcher_id"]].drop_duplicates().shape[0]),
        "unique_prop_lines": int(graded[start_keys].drop_duplicates().shape[0]),
        "all_snapshot_mae_k": float(graded["absolute_error"].mean()),
        "first_snapshot_mae_k": float(first["absolute_error"].mean()),
        "latest_snapshot_mae_k": float(latest["absolute_error"].mean()),
        "all_snapshot_rmse_k": float(np.sqrt(graded["squared_error"].mean())),
        "first_snapshot_over_brier": float(first["over_brier"].mean()),
        "latest_snapshot_over_brier": float(latest["over_brier"].mean()),
        "mean_projection_error": float(graded["projection_error"].mean()),
        "mean_model_market_edge": float(pd.to_numeric(graded.get("model_market_edge"), errors="coerce").mean()),
    }])
    summary.to_csv(SUMMARY, index=False)
    print(summary.round(5).to_string(index=False))

if __name__ == "__main__":
    main()
