"""Fetch current MLB moneylines automatically from The Odds API.

Requires THE_ODDS_API_KEY in the environment. The free plan currently supports
current MLB odds, so this removes daily manual odds entry from the workflow.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "current" / "morning_odds.csv"
URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def main() -> None:
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        raise SystemExit("THE_ODDS_API_KEY is not configured. Add it as a GitHub Actions repository secret.")

    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    snapshot = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes")

    rows = []
    for event in payload:
        home = event.get("home_team")
        away = event.get("away_team")
        commence = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        game_date = commence.tz_convert("America/New_York").date().isoformat() if pd.notna(commence) else None
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {x.get("name"): x.get("price") for x in market.get("outcomes", [])}
                if home not in prices or away not in prices:
                    continue
                rows.append({
                    "date": game_date,
                    "sportsbook": book.get("title") or book.get("key"),
                    "away_team": away,
                    "home_team": home,
                    "away_moneyline": prices[away],
                    "home_moneyline": prices[home],
                    "snapshot_time_et": snapshot,
                    "source": "the-odds-api",
                    "source_last_update": book.get("last_update"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("The Odds API returned no MLB h2h moneylines.")

    # Keep every sportsbook quote. The morning model can form a consensus and
    # preserve individual books for price shopping later.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["date", "away_team", "home_team", "sportsbook"]).to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} sportsbook quotes for {df[['date','away_team','home_team']].drop_duplicates().shape[0]:,} MLB games to {OUT}")


if __name__ == "__main__":
    main()
