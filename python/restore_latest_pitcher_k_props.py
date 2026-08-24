"""Restore the latest cached pitcher-K consensus snapshot to the current file.

Used by V2 so a late-afternoon lineup run can reuse the noon sportsbook snapshot
without spending more Odds API credits.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
HISTORY = CURRENT / "pitcher_k_props_history.csv"
OUT = CURRENT / "pitcher_k_props.csv"


def main() -> None:
    if not HISTORY.exists():
        raise SystemExit("No cached pitcher_k_props_history.csv is available.")
    h = pd.read_csv(HISTORY, low_memory=False)
    if h.empty:
        raise SystemExit("Cached pitcher K prop history is empty.")
    h["snapshot_time_et"] = pd.to_datetime(h["snapshot_time_et"], errors="coerce")
    h = h[h["snapshot_time_et"].notna()].copy()
    if h.empty:
        raise SystemExit("Cached pitcher K prop history has no valid snapshot timestamps.")
    target_date = str(h.sort_values("snapshot_time_et").iloc[-1]["date"])
    d = h[h["date"].astype(str).eq(target_date)].copy()
    latest = d["snapshot_time_et"].max()
    out = d[d["snapshot_time_et"].eq(latest)].copy()
    out.to_csv(OUT, index=False)
    print(f"Restored {len(out)} pitcher-K consensus rows from {latest} for {target_date}; no Odds API call made.")


if __name__ == "__main__":
    main()
