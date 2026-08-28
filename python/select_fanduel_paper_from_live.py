"""Price the current FanDuel board using already-built live projections.

The projections must predate the quote collection. This script enforces that
relationship before any paper selection can be frozen.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from run_v22_fanduel_paper import MKT, freeze, pair_fanduel, select_candidates

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"


def main() -> None:
    day = date.today().isoformat()
    if not PROJ.exists() or PROJ.stat().st_size == 0:
        print("No prebuilt live projections; nothing to price.")
        return
    try:
        projections = pd.read_csv(PROJ, low_memory=False)
    except pd.errors.EmptyDataError:
        print("Live projection file is empty; nothing to price.")
        return
    if projections.empty:
        print("No live projections; nothing to price.")
        return

    if not MKT.exists() or MKT.stat().st_size == 0:
        print("No current FanDuel market file; nothing to price.")
        return
    try:
        raw = pd.read_csv(MKT, low_memory=False)
    except pd.errors.EmptyDataError:
        print("Current FanDuel market file is empty; nothing to price.")
        return
    market = pair_fanduel(raw, day)
    if market.empty:
        print(f"No FanDuel pitcher-K quotes found for {day}.")
        return

    projections["model_generated_at_et"] = pd.to_datetime(
        projections["model_generated_at_et"], errors="coerce", utc=True
    )
    market["collected_at_utc"] = pd.to_datetime(market["collected_at_utc"], errors="coerce", utc=True)
    latest_projection = projections["model_generated_at_et"].max()
    earliest_quote = market["collected_at_utc"].min()
    if pd.isna(latest_projection) or pd.isna(earliest_quote):
        raise ValueError("Missing model or quote timestamps; cannot guarantee timing integrity.")
    if latest_projection > earliest_quote:
        raise ValueError(
            f"Timing integrity failure: latest model timestamp {latest_projection} is after "
            f"earliest decision quote {earliest_quote}. Rebuild projections before recollecting market."
        )

    # select_candidates expects the display timestamp as text, but accepts the
    # parsed value too; preserve a consistent ISO representation in outputs.
    projections["model_generated_at_et"] = projections["model_generated_at_et"].dt.tz_convert("America/New_York").astype(str)
    candidates = select_candidates(market, projections)
    freeze(candidates)


if __name__ == "__main__":
    main()
