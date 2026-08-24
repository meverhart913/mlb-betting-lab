"""Create pregame rolling Statcast pitcher features from cached daily aggregates.

For each pitcher/date, every feature is shifted by one appearance before rolling,
so the current game's pitches can never enter its own pregame feature vector.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features"
SRC = DATA / "statcast_pitcher_daily.csv"
OUT = DATA / "statcast_pitcher_pregame.csv"

BASE_COLS = ["pitches", "mean_velocity", "max_velocity", "whiff_per_swing", "pitch_types"]
WINDOWS = (3, 5, 10)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}; run backfill_statcast_pitch_features.py first.")
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        df.to_csv(OUT, index=False)
        return
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    df = df[df.game_date.notna() & df.pitcher_id.notna()].sort_values(["pitcher_id", "game_date"])

    numeric = [c for c in df.columns if c not in {"game_date", "pitcher_id"}]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = df[["game_date", "pitcher_id"]].copy()
    for c in numeric:
        shifted = df.groupby("pitcher_id", sort=False)[c].shift(1)
        for w in WINDOWS:
            keep[f"statcast_{c}_{w}"] = (
                shifted.groupby(df["pitcher_id"], sort=False)
                .rolling(w, min_periods=max(2, w // 2))
                .mean()
                .reset_index(level=0, drop=True)
            )
    # Useful trend features: recent velocity/whiff relative to a longer baseline.
    if "statcast_mean_velocity_3" in keep and "statcast_mean_velocity_10" in keep:
        keep["statcast_velocity_trend_3v10"] = keep["statcast_mean_velocity_3"] - keep["statcast_mean_velocity_10"]
    if "statcast_whiff_per_swing_3" in keep and "statcast_whiff_per_swing_10" in keep:
        keep["statcast_whiff_trend_3v10"] = keep["statcast_whiff_per_swing_3"] - keep["statcast_whiff_per_swing_10"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT, index=False)
    print(f"Wrote {len(keep):,} leakage-safe pregame Statcast rows to {OUT}.")


if __name__ == "__main__":
    main()
