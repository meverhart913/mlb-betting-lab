"""Fetch current MLB starting-pitcher strikeout props from The Odds API.

Current player props are queried one event at a time. Raw sportsbook quotes are
retained; a consensus table keeps median and best prices. The collector protects
a configurable API-credit reserve and records collection status so a low quota
produces partial research output instead of exhausting the account.
"""
from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
RAW = CURRENT / "pitcher_k_props_raw.csv"
CONSENSUS = CURRENT / "pitcher_k_props.csv"
RAW_HISTORY = CURRENT / "pitcher_k_props_raw_history.csv"
CONS_HISTORY = CURRENT / "pitcher_k_props_history.csv"
STATUS = CURRENT / "pitcher_k_collection_status.csv"
BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
DEFAULT_RESERVE = 30


def append_history(path: Path, fresh: pd.DataFrame, keys: list[str]) -> None:
    if fresh.empty:
        return
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        out = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        out = fresh.copy()
    out = out.drop_duplicates(keys, keep="last")
    out.to_csv(path, index=False)


def int_header(resp, name: str):
    try:
        return int(resp.headers.get(name, ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        raise SystemExit("THE_ODDS_API_KEY is not configured.")
    reserve = int(os.getenv("ODDS_API_CREDIT_RESERVE", str(DEFAULT_RESERVE)))
    target = date.today().isoformat()
    common = {"apiKey": key, "dateFormat": "iso"}
    r = requests.get(f"{BASE}/events", params=common, timeout=30)
    r.raise_for_status()

    events = []
    for e in r.json():
        ts = pd.to_datetime(e.get("commence_time"), utc=True, errors="coerce")
        local_date = ts.tz_convert("America/New_York").date().isoformat() if pd.notna(ts) else None
        if local_date == target:
            events.append((ts, e))
    events.sort(key=lambda x: x[0] if pd.notna(x[0]) else pd.Timestamp.max.tz_localize("UTC"))
    events = [e for _, e in events]

    remaining_start = int_header(r, "x-requests-remaining")
    budget_events = len(events)
    if remaining_start is not None:
        budget_events = max(0, min(len(events), remaining_start - reserve))
    selected_events = events[:budget_events]
    skipped_budget = len(events) - len(selected_events)

    snapshot = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes")
    rows = []
    credits = 0
    last_remaining = remaining_start
    checked = 0
    for event in selected_events:
        eid = event.get("id")
        params = {"apiKey": key, "regions": "us", "markets": "pitcher_strikeouts", "oddsFormat": "american", "dateFormat": "iso"}
        q = requests.get(f"{BASE}/events/{eid}/odds", params=params, timeout=30)
        checked += 1
        if q.status_code == 422:
            print(f"WARN no pitcher strikeout market for event {eid}")
            continue
        q.raise_for_status()
        used = int_header(q, "x-requests-last")
        if used is not None:
            credits += used
        q_remaining = int_header(q, "x-requests-remaining")
        if q_remaining is not None:
            last_remaining = q_remaining
        payload = q.json()
        for book in payload.get("bookmakers", []):
            sportsbook = book.get("title") or book.get("key")
            for market in book.get("markets", []):
                if market.get("key") != "pitcher_strikeouts":
                    continue
                for o in market.get("outcomes", []):
                    pitcher = o.get("description")
                    side = str(o.get("name") or "").lower()
                    if not pitcher or side not in {"over", "under"}:
                        continue
                    rows.append({
                        "date": target, "event_id": eid,
                        "away_team": payload.get("away_team"), "home_team": payload.get("home_team"),
                        "pitcher_name": pitcher, "side": side, "line": o.get("point"), "price": o.get("price"),
                        "sportsbook": sportsbook, "bookmaker_key": book.get("key"),
                        "source_last_update": market.get("last_update") or book.get("last_update"),
                        "snapshot_time_et": snapshot, "source": "the-odds-api",
                    })

    CURRENT.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame([{
        "date": target, "snapshot_time_et": snapshot, "events_on_slate": len(events),
        "events_checked": checked, "events_skipped_for_quota": skipped_budget,
        "credit_reserve": reserve, "credits_used_event_calls": credits,
        "credits_remaining_start": remaining_start, "credits_remaining_end": last_remaining,
        "raw_quotes": len(rows),
    }])
    status.to_csv(STATUS, index=False)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raw.to_csv(RAW, index=False)
        pd.DataFrame().to_csv(CONSENSUS, index=False)
        print(f"No pitcher strikeout props returned for {target}; checked {checked}/{len(events)} MLB events; skipped for quota: {skipped_budget}.")
        return

    raw["line"] = pd.to_numeric(raw["line"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.sort_values(["event_id", "pitcher_name", "line", "sportsbook", "side"])
    raw.to_csv(RAW, index=False)
    append_history(RAW_HISTORY, raw, ["event_id", "pitcher_name", "line", "sportsbook", "side", "snapshot_time_et"])

    wide = raw.pivot_table(
        index=["date","event_id","away_team","home_team","pitcher_name","line","sportsbook","snapshot_time_et"],
        columns="side", values="price", aggfunc="last"
    ).reset_index()
    rows2 = []
    for keys, g in wide.groupby(["date","event_id","away_team","home_team","pitcher_name","line"], dropna=False):
        def best(col):
            x = pd.to_numeric(g.get(col), errors="coerce")
            if x.notna().sum() == 0:
                return (float("nan"), None)
            idx = x.idxmax()
            return float(x.loc[idx]), str(g.loc[idx, "sportsbook"])
        bo, bob = best("over"); bu, bub = best("under")
        rows2.append({
            "date": keys[0], "event_id": keys[1], "away_team": keys[2], "home_team": keys[3],
            "pitcher_name": keys[4], "line": keys[5],
            "over_price_median": pd.to_numeric(g.get("over"), errors="coerce").median(),
            "under_price_median": pd.to_numeric(g.get("under"), errors="coerce").median(),
            "best_over_price": bo, "best_over_sportsbook": bob,
            "best_under_price": bu, "best_under_sportsbook": bub,
            "sportsbook_count": int(g["sportsbook"].nunique()), "snapshot_time_et": snapshot,
        })
    cons = pd.DataFrame(rows2).sort_values(["event_id","pitcher_name","line"])
    cons.to_csv(CONSENSUS, index=False)
    append_history(CONS_HISTORY, cons, ["event_id", "pitcher_name", "line", "snapshot_time_et"])
    print(
        f"Wrote {len(raw):,} raw K quotes and {len(cons):,} consensus rows; checked {checked}/{len(events)} events; "
        f"skipped for quota: {skipped_budget}; credits used: {credits}; remaining: {last_remaining}."
    )

if __name__ == "__main__":
    main()
