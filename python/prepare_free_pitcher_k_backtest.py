"""Prepare the accumulated free pitcher-K archive for the V2.2 market backtester."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "market" / "free_pitcher_k_archive_graded.csv"
OUT = ROOT / "data" / "market" / "pitcher_k_historical_raw.csv"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}")
    z = pd.read_csv(SRC, low_memory=False)
    if "matched_mlb_date" in z.columns:
        matched = pd.to_datetime(z["matched_mlb_date"], errors="coerce")
        original = pd.to_datetime(z["date"], errors="coerce")
        z["date"] = matched.fillna(original).dt.date
    starter = pd.to_numeric(z.get("mlb_is_starter"), errors="coerce").eq(1)
    actual = pd.to_numeric(z.get("actual_k"), errors="coerce").notna()
    z = z[starter & actual].copy()
    keep = [c for c in ["date", "pitcher_name", "line", "side", "price", "sportsbook", "event_id", "snapshot_time_et", "source"] if c in z.columns]
    z = z[keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(OUT, index=False)
    print(f"Prepared {len(z):,} graded starter quote rows for V2.2 market backtest")


if __name__ == "__main__":
    main()
