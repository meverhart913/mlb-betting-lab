"""Fetch live/upcoming FanDuel MLB pitcher-strikeout prices from PropLine.

Requires PROPLINE_API_KEY. The free tier supports live events/odds; this script
uses the API key only in the X-API-Key header so it is never written to logs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import re

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/market/pitcher_k_live_fanduel.csv"
ALL = ROOT / "data/market/pitcher_k_live_fanduel_all_outcomes.csv"
BASE = "https://api.prop-line.com/v1"
SPORT = "baseball_mlb"


def get_json(url, key, params=None):
    r = requests.get(url, headers={"X-API-Key": key}, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers


def milestone_threshold(name: str):
    s = str(name or "").lower()
    m = re.search(r"(?:^|\s)(\d+)\+", s)
    return int(m.group(1)) if m else None


def main():
    key = os.environ.get("PROPLINE_API_KEY", "").strip()
    if not key:
        raise SystemExit("PROPLINE_API_KEY is not set. Add the free PropLine API key as a GitHub Actions secret.")

    now = pd.Timestamp(datetime.now(timezone.utc))
    events, headers = get_json(f"{BASE}/sports/{SPORT}/events", key)
    if isinstance(events, dict):
        events = events.get("events", events.get("data", []))
    if not isinstance(events, list):
        raise ValueError("Unexpected PropLine events response shape")

    selected = []
    for ev in events:
        commence = pd.to_datetime(ev.get("commence_time"), errors="coerce", utc=True)
        if pd.isna(commence):
            continue
        minutes = (commence - now).total_seconds() / 60
        # Include the next day and a small post-start buffer for diagnostics, but
        # paper selection later requires a strictly pregame decision window.
        if minutes < -15 or minutes > 30 * 60:
            continue
        selected.append(ev)

    rows = []
    for ev in selected:
        event_id = str(ev.get("id"))
        payload, h = get_json(
            f"{BASE}/sports/{SPORT}/events/{event_id}/odds",
            key,
            params={"markets": "pitcher_strikeouts", "bookmakers": "fanduel"},
        )
        if isinstance(payload, list):
            items = payload
        else:
            items = [payload]
        for item in items:
            commence = item.get("commence_time") or ev.get("commence_time")
            for book in item.get("bookmakers", []) or []:
                if str(book.get("key", "")).lower() != "fanduel" and str(book.get("title", "")).lower() != "fanduel":
                    continue
                for market in book.get("markets", []) or []:
                    if market.get("key") != "pitcher_strikeouts":
                        continue
                    for outcome in market.get("outcomes", []) or []:
                        name = str(outcome.get("name", ""))
                        side = name.lower()
                        if side not in {"over", "under"}:
                            threshold = milestone_threshold(name)
                            side_norm = "at_least" if threshold is not None else "other"
                        else:
                            threshold = None
                            side_norm = side
                        point = outcome.get("point")
                        if point is None and threshold is not None:
                            point = threshold
                        rows.append({
                            "date": pd.to_datetime(commence, utc=True).tz_convert("America/New_York").date() if commence else None,
                            "event_id": event_id,
                            "commence_time_utc": commence,
                            "home_team": item.get("home_team") or ev.get("home_team"),
                            "away_team": item.get("away_team") or ev.get("away_team"),
                            "pitcher_name": outcome.get("description"),
                            "outcome_name": name,
                            "side": side_norm,
                            "line": point,
                            "price": outcome.get("price"),
                            "sportsbook": "fanduel",
                            "book_last_update": book.get("last_update"),
                            "market_last_update": market.get("last_update"),
                            "last_change_at": outcome.get("last_change_at"),
                            "collected_at_utc": now.isoformat(),
                            "collected_at_et": now.tz_convert("America/New_York").isoformat(),
                            "snapshot_time_et": now.tz_convert("America/New_York").isoformat(),
                            "source": "propline_live_api",
                        })

    z = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(ALL, index=False)
    usable = z[
        z.side.isin(["over", "under"])
        & pd.to_numeric(z.line, errors="coerce").notna()
        & pd.to_numeric(z.price, errors="coerce").notna()
        & z.pitcher_name.notna()
    ].copy() if not z.empty else z.copy()
    usable.to_csv(OUT, index=False)

    remain = headers.get("X-Daily-Remaining") or headers.get("X-RateLimit-Remaining") or "unknown"
    milestones = int(z.side.eq("at_least").sum()) if not z.empty else 0
    print(f"PropLine live: events={len(selected)}, FanDuel outcomes={len(z)}, usable O/U rows={len(usable)}, milestone rows={milestones}")
    print(f"Collected {now.isoformat()}; API requests remaining header={remain}")


if __name__ == "__main__":
    main()
