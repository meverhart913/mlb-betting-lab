"""Print true when the target date has at least one MLB regular-season game."""
from __future__ import annotations

from datetime import date
import argparse
import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": args.date, "gameType": "R"},
        timeout=30,
    )
    r.raise_for_status()
    count = sum(len(d.get("games", [])) for d in r.json().get("dates", []))
    print("true" if count else "false")


if __name__ == "__main__":
    main()
