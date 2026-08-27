"""Fetch current FanDuel MLB pitcher-K main + alternate markets from The Odds API.

Used only when the free PropLine sample does not contain a usable same-day
FanDuel board. Outputs the normalized schema consumed by run_v22_fanduel_paper.py.
Credit-aware: only events beginning within the configurable horizon are queried
and a hard reserve is preserved.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/market/pitcher_k_live_fanduel.csv"
ALL = ROOT / "data/market/pitcher_k_live_fanduel_all_outcomes.csv"
STATUS = ROOT / "data/current/fanduel_oddsapi_status.csv"
BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
MARKETS = "pitcher_strikeouts,pitcher_strikeouts_alternate"
RESERVE = int(os.getenv("ODDS_API_CREDIT_RESERVE", "30"))
HORIZON_HOURS = float(os.getenv("FANDUEL_ODDS_HORIZON_HOURS", "4.0"))


def ih(resp, name):
    try:
        return int(resp.headers.get(name, ""))
    except (TypeError, ValueError):
        return None


def main():
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        raise SystemExit("THE_ODDS_API_KEY is not configured")

    now = pd.Timestamp.now(tz="UTC")
    today_et = now.tz_convert("America/New_York").date().isoformat()
    ev = requests.get(f"{BASE}/events", params={"apiKey":key,"dateFormat":"iso"}, timeout=30)
    ev.raise_for_status()
    remaining = ih(ev, "x-requests-remaining")

    events=[]
    for e in ev.json():
        ct=pd.to_datetime(e.get("commence_time"),errors="coerce",utc=True)
        if pd.isna(ct): continue
        if ct.tz_convert("America/New_York").date().isoformat()!=today_et: continue
        hours=(ct-now).total_seconds()/3600
        if -0.25 <= hours <= HORIZON_HOURS:
            events.append((ct,e))
    events.sort(key=lambda x:x[0])

    # Two requested markets cost at least two market credits per event in one US region.
    affordable=len(events)
    if remaining is not None:
        affordable=max(0,min(len(events),(remaining-RESERVE)//2))
    events=events[:affordable]

    rows=[]; used_total=0; checked=0; last_remaining=remaining
    collected=datetime.now(timezone.utc).isoformat()
    for ct,e in events:
        eid=e.get("id")
        q=requests.get(
            f"{BASE}/events/{eid}/odds",
            params={"apiKey":key,"regions":"us","markets":MARKETS,"oddsFormat":"american","dateFormat":"iso","bookmakers":"fanduel"},
            timeout=30,
        )
        checked+=1
        if q.status_code in (404,422):
            print(f"WARN no FanDuel pitcher-K markets for {eid}")
            continue
        q.raise_for_status()
        last=ih(q,"x-requests-last")
        if last is not None: used_total+=last
        rem=ih(q,"x-requests-remaining")
        if rem is not None: last_remaining=rem
        payload=q.json()
        for book in payload.get("bookmakers",[]):
            if str(book.get("key") or "").lower()!="fanduel" and str(book.get("title") or "").lower()!="fanduel":
                continue
            for market in book.get("markets",[]):
                mk=market.get("key")
                if mk not in {"pitcher_strikeouts","pitcher_strikeouts_alternate"}: continue
                for o in market.get("outcomes",[]):
                    side=str(o.get("name") or "").lower()
                    pitcher=o.get("description")
                    if side not in {"over","under"} or not pitcher: continue
                    rows.append({
                        "date":ct.tz_convert("America/New_York").date().isoformat(),
                        "event_id":eid,
                        "away_team":payload.get("away_team") or e.get("away_team"),
                        "home_team":payload.get("home_team") or e.get("home_team"),
                        "pitcher_name":pitcher,
                        "side":side,
                        "line":o.get("point"),
                        "price":o.get("price"),
                        "sportsbook":"FanDuel",
                        "market_key":mk,
                        "commence_time_utc":ct.isoformat(),
                        "collected_at_utc":collected,
                        "snapshot_time_et":pd.Timestamp(collected).tz_convert("America/New_York").isoformat(),
                        "source_last_update":market.get("last_update") or book.get("last_update"),
                        "source":"the_odds_api_fanduel_fallback",
                    })

    z=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True); STATUS.parent.mkdir(parents=True,exist_ok=True)
    if not z.empty:
        z["line"]=pd.to_numeric(z.line,errors="coerce"); z["price"]=pd.to_numeric(z.price,errors="coerce")
        z=z.dropna(subset=["pitcher_name","line","price"]).drop_duplicates(["event_id","pitcher_name","line","side","price"],keep="last")
    z.to_csv(ALL,index=False)
    z.to_csv(OUT,index=False)
    pd.DataFrame([{
        "collected_at_utc":collected,"date_et":today_et,"eligible_events":len(events),"events_checked":checked,
        "markets":MARKETS,"fanduel_outcome_rows":len(z),"credits_used":used_total,
        "credits_remaining_start":remaining,"credits_remaining_end":last_remaining,"reserve":RESERVE,
        "horizon_hours":HORIZON_HOURS,
    }]).to_csv(STATUS,index=False)
    print(f"Odds API FanDuel fallback: checked {checked} events, {len(z)} main/alt K outcomes, credits used={used_total}, remaining={last_remaining}")

if __name__=="__main__": main()
