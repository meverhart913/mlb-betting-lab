"""Capture morning-of-game venue and weather context without manual input.

Venue coordinates/roof metadata come from MLB Stats API. Hourly forecast data
comes from Open-Meteo. These fields are captured now for audit and future
historical validation; they are not yet authorized production model features.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "current" / "morning_context.csv"


def schedule(day: str) -> pd.DataFrame:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": day, "gameType": "R", "hydrate": "probablePitcher"},
        timeout=30,
    )
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            rows.append({
                "game_id": g.get("gamePk"),
                "date": d.get("date"),
                "game_datetime_utc": g.get("gameDate"),
                "venue_id": (g.get("venue") or {}).get("id"),
                "venue_name": (g.get("venue") or {}).get("name"),
                "away_team": (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name"),
                "home_team": (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name"),
            })
    return pd.DataFrame(rows)


def venue(venue_id: int) -> dict:
    r = requests.get(
        f"https://statsapi.mlb.com/api/v1/venues/{int(venue_id)}",
        params={"hydrate": "location,fieldInfo,timezone"},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("venues", [])
    if not items:
        return {}
    v = items[0]
    loc = v.get("location") or {}
    field = v.get("fieldInfo") or {}
    tz = v.get("timeZone") or {}
    return {
        "latitude": pd.to_numeric(loc.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(loc.get("longitude"), errors="coerce"),
        "venue_timezone": tz.get("id"),
        "roof_type": field.get("roofType"),
        "turf_type": field.get("turfType"),
    }


def forecast(lat: float, lon: float, game_time_utc: pd.Timestamp) -> dict:
    if pd.isna(lat) or pd.isna(lon) or pd.isna(game_time_utc):
        return {}
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": "temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "forecast_days": 7,
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json().get("hourly") or {}
    times = pd.to_datetime(h.get("time", []), utc=True, errors="coerce")
    if len(times) == 0:
        return {}
    target = pd.Timestamp(game_time_utc)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    idx = int(abs(times - target).argmin())

    def get(name):
        vals = h.get(name, [])
        return vals[idx] if idx < len(vals) else None

    return {
        "forecast_hour_utc": times[idx].isoformat(),
        "temperature_f": get("temperature_2m"),
        "precip_probability_pct": get("precipitation_probability"),
        "precipitation_in": get("precipitation"),
        "wind_mph": get("wind_speed_10m"),
        "wind_gust_mph": get("wind_gusts_10m"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    games = schedule(args.date)
    if games.empty:
        print(f"No MLB regular-season games for {args.date}.")
        return

    venue_cache = {}
    rows = []
    for g in games.to_dict("records"):
        vid = g.get("venue_id")
        if vid not in venue_cache:
            venue_cache[vid] = venue(vid) if pd.notna(vid) else {}
        row = dict(g)
        row.update(venue_cache.get(vid, {}))
        game_time = pd.to_datetime(row.get("game_datetime_utc"), utc=True, errors="coerce")
        row.update(forecast(row.get("latitude"), row.get("longitude"), game_time))
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote automated venue/weather context for {len(out):,} games to {OUT}.")


if __name__ == "__main__":
    main()
