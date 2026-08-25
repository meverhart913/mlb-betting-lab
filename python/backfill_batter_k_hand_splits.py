"""Backfill batter strikeout outcomes by opposing pitcher handedness from Statcast.

Produces compact batter/day/handedness aggregates for historical matchup research.
Only completed plate appearances are counted. Research-only V2.1 data layer.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features"
DATA.mkdir(parents=True, exist_ok=True)
OUT = DATA / "batter_k_by_pitcher_hand_daily.csv"
BASE = "https://baseballsavant.mlb.com/statcast_search/csv"


def chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        hi = min(end, cur + timedelta(days=days - 1))
        yield cur, hi
        cur = hi + timedelta(days=1)


def fetch_chunk(start: date, end: date, retries: int = 5) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "batter",
        "player_type": "batter",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "hfGT": "R|",
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(BASE, params=params, timeout=120)
            r.raise_for_status()
            text = r.text.strip()
            return pd.read_csv(StringIO(text), low_memory=False) if text else pd.DataFrame()
        except (requests.RequestException, OSError) as exc:
            last = exc
            if attempt >= retries:
                break
            wait = min(30, 2 ** attempt)
            print(f"WARN Statcast fetch failed for {start}..{end} attempt {attempt}/{retries}: {exc}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Statcast fetch failed after {retries} attempts for {start}..{end}: {last}")


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    required = {"game_date", "batter", "p_throws", "events", "at_bat_number"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Statcast response missing columns: " + ", ".join(missing))

    z = raw.copy()
    z["game_date"] = pd.to_datetime(z["game_date"], errors="coerce").dt.date
    z["batter"] = pd.to_numeric(z["batter"], errors="coerce")
    z["at_bat_number"] = pd.to_numeric(z["at_bat_number"], errors="coerce")
    z["p_throws"] = z["p_throws"].astype(str).str.upper()
    z = z[z["game_date"].notna() & z["batter"].notna() & z["p_throws"].isin(["L", "R"])].copy()

    pa = z[z["events"].notna() & z["at_bat_number"].notna()].copy()
    pa = pa.drop_duplicates(["game_date", "batter", "at_bat_number"], keep="last")
    pa["is_k"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)

    out = (
        pa.groupby(["game_date", "batter", "p_throws"], dropna=False)
        .agg(plate_appearances=("at_bat_number", "size"), strikeouts=("is_k", "sum"))
        .reset_index()
        .rename(columns={"batter": "batter_id", "p_throws": "pitcher_hand"})
    )
    out["k_per_pa"] = out["strikeouts"] / out["plate_appearances"].replace(0, pd.NA)
    return out


def merge_save(fresh: pd.DataFrame) -> None:
    if fresh.empty:
        return
    if OUT.exists():
        old = pd.read_csv(OUT, low_memory=False)
        out = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        out = fresh.copy()
    out = out.drop_duplicates(["game_date", "batter_id", "pitcher_hand"], keep="last")
    out = out.sort_values(["game_date", "batter_id", "pitcher_hand"])
    out.to_csv(OUT, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--chunk-days", type=int, default=14)
    ap.add_argument("--sleep-seconds", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    parts = []
    for lo, hi in chunks(start, end, args.chunk_days):
        print(f"Fetching batter handedness Statcast {lo} through {hi}...", flush=True)
        part = aggregate(fetch_chunk(lo, hi, retries=max(1, args.retries)))
        if not part.empty:
            parts.append(part)
        time.sleep(max(args.sleep_seconds, 0))
    fresh = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    merge_save(fresh)
    print(f"Wrote {len(fresh):,} batter/day/hand rows into {OUT}.")


if __name__ == "__main__":
    main()
