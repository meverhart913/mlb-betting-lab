"""Incrementally refresh recent MLB games, team logs, and pitcher logs.

Designed for the scheduled morning workflow. It revisits the last few calendar
days so postponed games and late corrections are naturally repaired by upsert.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import argparse
import time

import pandas as pd
import requests

from download_mlb_enrichment import extract

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"


def upsert(path: Path, fresh: pd.DataFrame, keys: list[str]) -> None:
    if fresh.empty:
        return
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        combined = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        combined = fresh.copy()
    combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(path, index=False)


def schedule(start: str, end: str) -> list[dict]:
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "gameType": "R",
        "hydrate": "probablePitcher,linescore",
    }
    r = requests.get(SCHEDULE_URL, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            h = g.get("teams", {}).get("home", {})
            a = g.get("teams", {}).get("away", {})
            hs, aws = h.get("score"), a.get("score")
            row = {
                "season": pd.to_numeric(str(d.get("date", ""))[:4], errors="coerce"),
                "game_id": g.get("gamePk"),
                "date": d.get("date"),
                "game_datetime": g.get("gameDate"),
                "away_team": (a.get("team") or {}).get("name"),
                "home_team": (h.get("team") or {}).get("name"),
                "away_score": aws,
                "home_score": hs,
                "away_wins": (a.get("leagueRecord") or {}).get("wins"),
                "away_losses": (a.get("leagueRecord") or {}).get("losses"),
                "home_wins": (h.get("leagueRecord") or {}).get("wins"),
                "home_losses": (h.get("leagueRecord") or {}).get("losses"),
                "away_probable_pitcher": (a.get("probablePitcher") or {}).get("fullName"),
                "home_probable_pitcher": (h.get("probablePitcher") or {}).get("fullName"),
                "venue": (g.get("venue") or {}).get("name"),
                "status": (g.get("status") or {}).get("detailedState"),
                "series": g.get("seriesDescription"),
                "game_number": g.get("gameNumber"),
                "winner": None,
                "home_win": None,
            }
            if hs is not None and aws is not None and hs != aws:
                row["home_win"] = int(hs > aws)
                row["winner"] = row["home_team"] if hs > aws else row["away_team"]
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=4)
    args = ap.parse_args()
    end = date.today()
    start = end - timedelta(days=max(args.lookback_days, 1))

    game_rows = schedule(start.isoformat(), end.isoformat())
    games = pd.DataFrame(game_rows)
    upsert(DATA / "mlb_games_2018_present.csv", games, ["game_id"])

    enrich_rows, pitcher_rows, team_rows = [], [], []
    completed = games[games["home_score"].notna() & games["away_score"].notna()] if not games.empty else games
    for row in completed.itertuples(index=False):
        try:
            r = requests.get(FEED_URL.format(int(row.game_id)), timeout=40)
            r.raise_for_status()
            e, p, t = extract(int(row.game_id), r.json())
            enrich_rows.append(e)
            pitcher_rows.extend(p)
            team_rows.extend(t)
        except Exception as exc:
            print(f"WARN game {row.game_id}: {exc!r}")
        time.sleep(0.05)

    upsert(DATA / "mlb_game_enrichment.csv", pd.DataFrame(enrich_rows), ["game_id"])
    upsert(DATA / "mlb_pitcher_game_logs.csv", pd.DataFrame(pitcher_rows), ["game_id", "side", "pitcher_id"])
    upsert(DATA / "mlb_team_game_logs.csv", pd.DataFrame(team_rows), ["game_id", "side"])
    print(f"Refreshed {len(games):,} recent schedule rows and {len(enrich_rows):,} completed game feeds ({start} through {end}).")


if __name__ == "__main__":
    main()
