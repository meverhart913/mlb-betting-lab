"""Backfill leakage-safe daily Statcast pitcher features for Pitcher K V2.1.

The script downloads pitch-level CSV data from Baseball Savant in date chunks,
then aggregates only information available before each game date. It writes a
compact daily feature table rather than storing every pitch in the repository.

Research only. This collector is separate from V1/V2 prospective histories.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features"
DATA.mkdir(parents=True, exist_ok=True)
OUT = DATA / "statcast_pitcher_daily.csv"
BASE = "https://baseballsavant.mlb.com/statcast_search/csv"

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}


def chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        hi = min(end, cur + timedelta(days=days - 1))
        yield cur, hi
        cur = hi + timedelta(days=1)


def fetch_chunk(start: date, end: date) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "pitcher",
        "player_type": "pitcher",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "hfGT": "R|",
    }
    r = requests.get(BASE, params=params, timeout=120)
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(text), low_memory=False)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    required = {"game_date", "pitcher", "pitch_type", "release_speed", "description"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Statcast response missing columns: " + ", ".join(missing))

    z = raw.copy()
    z["game_date"] = pd.to_datetime(z["game_date"], errors="coerce").dt.date
    z["pitcher"] = pd.to_numeric(z["pitcher"], errors="coerce")
    z["release_speed"] = pd.to_numeric(z["release_speed"], errors="coerce")
    z["description"] = z["description"].astype(str)
    z["is_swing"] = z["description"].isin(SWING_DESCRIPTIONS).astype(int)
    z["is_whiff"] = z["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    z = z[z["game_date"].notna() & z["pitcher"].notna()].copy()

    g = z.groupby(["game_date", "pitcher"], dropna=False)
    base = g.agg(
        pitches=("pitcher", "size"),
        mean_velocity=("release_speed", "mean"),
        max_velocity=("release_speed", "max"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        pitch_types=("pitch_type", "nunique"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    base["whiff_per_swing"] = np.where(base["swings"] > 0, base["whiffs"] / base["swings"], np.nan)

    mix = (
        z.assign(n=1)
        .pivot_table(index=["game_date", "pitcher"], columns="pitch_type", values="n", aggfunc="sum", fill_value=0)
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
    )
    pitch_cols = [c for c in mix.columns if c not in {"game_date", "pitcher_id"}]
    if pitch_cols:
        denom = mix[pitch_cols].sum(axis=1).replace(0, np.nan)
        for c in pitch_cols:
            mix[f"pitch_share_{c}"] = mix[c] / denom
        mix = mix.drop(columns=pitch_cols)
    return base.merge(mix, on=["game_date", "pitcher_id"], how="left")


def merge_save(fresh: pd.DataFrame) -> None:
    if fresh.empty:
        return
    if OUT.exists():
        old = pd.read_csv(OUT, low_memory=False)
        out = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        out = fresh
    out = out.drop_duplicates(["game_date", "pitcher_id"], keep="last")
    out = out.sort_values(["game_date", "pitcher_id"])
    out.to_csv(OUT, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--sleep-seconds", type=float, default=1.0)
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    parts = []
    for lo, hi in chunks(start, end, args.chunk_days):
        print(f"Fetching Statcast {lo} through {hi}...")
        raw = fetch_chunk(lo, hi)
        part = aggregate(raw)
        if not part.empty:
            parts.append(part)
        time.sleep(max(args.sleep_seconds, 0))
    fresh = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    merge_save(fresh)
    print(f"Wrote {len(fresh):,} daily pitcher rows into {OUT}.")


if __name__ == "__main__":
    main()
