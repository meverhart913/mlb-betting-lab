"""Persist the current FanDuel decision-cycle quote into the CLV archive.

Every non-empty same-day FanDuel snapshot is stored immutably under
`data/market/free_archive/YYYY-MM-DD/`. Later snapshots can therefore measure
CLV against a frozen paper selection even when the market came from the Odds API
fallback rather than PropLine.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/market/pitcher_k_historical_raw.csv'
ARCHIVE=ROOT/'data/market/free_archive'


def main():
    if not SRC.exists() or SRC.stat().st_size==0:
        print('No current market snapshot to archive.')
        return
    try:
        z=pd.read_csv(SRC,low_memory=False)
    except pd.errors.EmptyDataError:
        print('Current market snapshot is empty.')
        return
    needed={'date','sportsbook','pitcher_name','side','line','price','collected_at_utc','commence_time_utc'}
    if z.empty or not needed.issubset(z.columns):
        print('Current snapshot has no timing-complete FanDuel rows to archive.')
        return
    now=datetime.now(ZoneInfo('America/New_York'))
    day=now.date().isoformat()
    q=z[(z.date.astype(str)==day)&z.sportsbook.astype(str).str.casefold().eq('fanduel')].copy()
    q=q.dropna(subset=['collected_at_utc','commence_time_utc','pitcher_name','side','line','price'])
    if q.empty:
        print(f'No same-day FanDuel rows to archive for {day}.')
        return
    # Use actual collection time from the rows where possible, not workflow time.
    ts=pd.to_datetime(q.collected_at_utc,errors='coerce',utc=True).max()
    if pd.isna(ts):
        print('No valid collection timestamp; refusing to create CLV archive snapshot.')
        return
    stamp=ts.tz_convert('America/New_York').strftime('%H%M%S')
    folder=ARCHIVE/day
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'decision-{stamp}.csv'
    q.sort_values(['event_id','pitcher_name','line','side'] if 'event_id' in q.columns else ['pitcher_name','line','side']).to_csv(out,index=False)
    print(f'Archived {len(q)} FanDuel decision-cycle quote rows -> {out.relative_to(ROOT)}')

if __name__=='__main__': main()
