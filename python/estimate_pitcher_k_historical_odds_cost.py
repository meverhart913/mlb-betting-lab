"""Estimate The Odds API credits needed for historical MLB pitcher-K props.

No Odds API request is made and no API key is required. The estimate uses MLB's
public schedule to count regular-season games, then applies The Odds API's
published historical event-odds cost of 10 credits per region, market, event.
Player-prop history is only available from 2023-05-03 onward.
"""
from __future__ import annotations

import argparse
from datetime import date
import requests

MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
PROP_HISTORY_START = date(2023, 5, 3)
CREDITS_PER_EVENT = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-05-03")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()
    start = max(date.fromisoformat(args.start), PROP_HISTORY_START)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    r = requests.get(
        MLB_SCHEDULE,
        params={
            "sportId": 1,
            "gameType": "R",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        },
        timeout=60,
    )
    r.raise_for_status()
    games = sum(len(d.get("games", [])) for d in r.json().get("dates", []))
    minimum = games * CREDITS_PER_EVENT
    print(f"Date range: {start} through {end}")
    print(f"Regular-season MLB games: {games:,}")
    print(f"Historical pitcher_strikeouts event-odds calls: up to {games:,}")
    print(f"Published event-odds credit cost: {CREDITS_PER_EVENT} per event for 1 region x 1 market")
    print(f"Minimum event-odds credits for full coverage: {minimum:,}")
    print("This excludes any separate quota cost, if applicable, for historical event-ID discovery calls.")
    print("Use a sampled date range first if the account does not have enough paid historical credits.")


if __name__ == "__main__":
    main()
