"""Download PropLine's free seven-day MLB pitcher-strikeout sample.

No API key required. Normalizes the CSV into the exact historical market schema
used by backtest_pitcher_k_market.py so V2.2 pricing/grading can be smoke-tested
without consuming paid historical-odds credits.
"""
from pathlib import Path
import io
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'market' / 'pitcher_k_historical_raw.csv'
URL = 'https://api.prop-line.com/v1/exports/sample'


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def main():
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    market = pick(raw, 'market', 'market_key').astype(str)
    raw = raw[market.str.contains('pitcher_strikeouts', case=False, na=False)].copy()
    side = pick(raw, 'outcome_name', 'name', 'selection_side').astype(str).str.lower()

    commence_utc = pd.to_datetime(pick(raw, 'commence_time', 'date'), errors='coerce', utc=True)
    # MLB schedule dates are U.S. calendar dates. Using the raw UTC date shifts
    # late-evening games (especially West Coast starts) into the following day.
    # Normalize to Eastern before deriving the event date so it aligns with the
    # MLB Stats API game-date field used by our result and model histories.
    commence_et = commence_utc.dt.tz_convert('America/New_York')

    z = pd.DataFrame({
        'date': commence_et.dt.date,
        'commence_time_utc': commence_utc.astype('string'),
        'pitcher_name': pick(raw, 'player_name', 'description', 'player'),
        'line': pd.to_numeric(pick(raw, 'point', 'line'), errors='coerce'),
        'side': side,
        'price': pd.to_numeric(pick(raw, 'price_american', 'american_odds', 'price'), errors='coerce'),
        'sportsbook': pick(raw, 'bookmaker', 'bookmaker_title', 'sportsbook'),
        'event_id': pick(raw, 'event_id'),
        'home_team': pick(raw, 'home_team'),
        'away_team': pick(raw, 'away_team'),
        'snapshot_time_et': pick(raw, 'closing_at', 'recorded_at', 'snapshot_time'),
        'source': 'propline_free_sample',
    })
    z = z[z.side.isin(['over', 'under'])].dropna(subset=['date', 'pitcher_name', 'line', 'price', 'sportsbook'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(OUT, index=False)
    print(f'PropLine sample: {len(raw):,} pitcher-K outcome rows -> {len(z):,} normalized rows at {OUT}')
    print(f'Dates: {z.date.min()} through {z.date.max()}' if len(z) else 'No usable pitcher-K rows found')


if __name__ == '__main__':
    main()
