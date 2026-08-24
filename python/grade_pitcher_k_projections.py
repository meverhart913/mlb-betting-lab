"""Archive and grade prospective pitcher strikeout projections.

This is an evaluation layer, not a betting executor. Current projection rows are
appended to a persistent history. Once final pitcher logs are available, each
row is graded against actual strikeouts. Aggregate calibration/error summaries
are written for ongoing research.
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
    keys = [c for c in ["date", "event_id", "pitcher_id", "line", "projection_model"] if c in h.columns]
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
    summary = pd.DataFrame([{
        "graded_props": int(len(graded)),
        "unique_starts": int(graded[["game_id", "pitcher_id"]].drop_duplicates().shape[0]),
        "mae_k": float(graded["absolute_error"].mean()),
        "rmse_k": float(np.sqrt(graded["squared_error"].mean())),
        "mean_projection_error": float(graded["projection_error"].mean()),
        "over_brier": float(graded["over_brier"].mean()),
        "mean_model_market_edge": float(pd.to_numeric(graded.get("model_market_edge"), errors="coerce").mean()),
    }])
    summary.to_csv(SUMMARY, index=False)
    print(summary.round(5).to_string(index=False))

if __name__ == "__main__":
    main()
