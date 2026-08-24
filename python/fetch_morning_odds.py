"""Fetch current MLB moneylines automatically from The Odds API.

Requires THE_ODDS_API_KEY in the environment. Every individual sportsbook quote
is retained for audit/price shopping, while morning_odds.csv contains one
consensus row per game using median prices. Each successful pull is also
appended to snapshot-history files so morning-to-later line movement can be
measured prospectively without manual collection.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
OUT = CURRENT / "morning_odds.csv"
RAW = CURRENT / "morning_odds_raw.csv"
CONSENSUS_HISTORY = CURRENT / "odds_consensus_history.csv"
RAW_HISTORY = CURRENT / "odds_raw_history.csv"
URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def best_positive_value(group: pd.DataFrame, column: str) -> tuple[float, str | None]:
    vals = pd.to_numeric(group[column], errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan"), None
    idx = vals.idxmax()
    return float(vals.loc[idx]), str(group.loc[idx, "sportsbook"])


def append_history(path: Path, fresh: pd.DataFrame, keys: list[str]) -> None:
    if fresh.empty:
        return
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        both = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        both = fresh.copy()
    both = both.drop_duplicates(keys, keep="last")
    both.to_csv(path, index=False)


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
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")

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

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise SystemExit("The Odds API returned no MLB h2h moneylines.")

    CURRENT.mkdir(parents=True, exist_ok=True)
    raw = raw.sort_values(["date", "away_team", "home_team", "sportsbook"])
    raw.to_csv(RAW, index=False)
    append_history(RAW_HISTORY, raw, ["date", "away_team", "home_team", "sportsbook", "snapshot_time_et"])

    consensus_rows = []
    keys = ["date", "away_team", "home_team"]
    for key_vals, g in raw.groupby(keys, dropna=False):
        best_home, best_home_book = best_positive_value(g, "home_moneyline")
        best_away, best_away_book = best_positive_value(g, "away_moneyline")
        consensus_rows.append({
            "date": key_vals[0],
            "sportsbook": "CONSENSUS_MEDIAN",
            "away_team": key_vals[1],
            "home_team": key_vals[2],
            "away_moneyline": pd.to_numeric(g["away_moneyline"], errors="coerce").median(),
            "home_moneyline": pd.to_numeric(g["home_moneyline"], errors="coerce").median(),
            "snapshot_time_et": snapshot,
            "quote_count": int(len(g)),
            "best_away_moneyline": best_away,
            "best_away_sportsbook": best_away_book,
            "best_home_moneyline": best_home,
            "best_home_sportsbook": best_home_book,
            "source": "the-odds-api",
        })

    consensus = pd.DataFrame(consensus_rows).sort_values(keys)
    consensus.to_csv(OUT, index=False)
    append_history(CONSENSUS_HISTORY, consensus, keys + ["snapshot_time_et"])
    quota = f" Odds API quota used={used}, remaining={remaining}." if used or remaining else ""
    print(
        f"Wrote {len(raw):,} raw sportsbook quotes to {RAW} and "
        f"{len(consensus):,} consensus MLB game rows to {OUT}. Snapshot history updated.{quota}"
    )


if __name__ == "__main__":
    main()
