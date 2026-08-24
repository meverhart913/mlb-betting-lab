"""Capture batting-order status from MLB game feeds at the morning snapshot.

This does not project a lineup. It records the lineup MLB has actually published
at capture time and explicitly marks games where it is not yet available.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
SUMMARY_OUT = CURRENT / "lineup_context.csv"
PLAYERS_OUT = CURRENT / "lineup_players.csv"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"


def schedule(day: str) -> list[dict]:
    r = requests.get(SCHEDULE_URL, params={"sportId": 1, "date": day, "gameType": "R"}, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            rows.append({"date": d.get("date"), "game_id": g.get("gamePk")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    snapshot = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes")
    summary_rows, player_rows = [], []

    for game in schedule(args.date):
        r = requests.get(FEED_URL.format(int(game["game_id"])), timeout=30)
        r.raise_for_status()
        j = r.json()
        box = (j.get("liveData") or {}).get("boxscore") or {}
        teams = box.get("teams") or {}
        for side in ("away", "home"):
            tb = teams.get(side) or {}
            team = tb.get("team") or {}
            order = tb.get("battingOrder") or []
            players = tb.get("players") or {}
            # Some pregame feeds expose battingOrder on each player before the
            # top-level battingOrder list is populated. Use that only to recover
            # an explicitly published order, never to project one.
            if not order:
                ordered = []
                for key, rec in players.items():
                    bo = rec.get("battingOrder")
                    pid = (rec.get("person") or {}).get("id")
                    if bo and pid:
                        try:
                            ordered.append((int(bo), int(pid)))
                        except (TypeError, ValueError):
                            pass
                order = [pid for _, pid in sorted(ordered)]

            for slot, pid in enumerate(order, 1):
                rec = players.get(f"ID{pid}") or {}
                person = rec.get("person") or {}
                pos = rec.get("position") or {}
                player_rows.append({
                    "date": args.date, "game_id": game["game_id"], "side": side,
                    "team_id": team.get("id"), "team_name": team.get("name"),
                    "batting_slot": slot, "player_id": pid,
                    "player_name": person.get("fullName"), "position": pos.get("abbreviation"),
                    "snapshot_time_et": snapshot,
                })
            count = len(order)
            summary_rows.append({
                "date": args.date, "game_id": game["game_id"], "side": side,
                "team_id": team.get("id"), "team_name": team.get("name"),
                "lineup_player_count": count,
                "lineup_status": "posted" if count >= 9 else ("partial" if count > 0 else "not_posted"),
                "snapshot_time_et": snapshot,
                "source": "mlb-game-feed",
            })

    CURRENT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
    pd.DataFrame(player_rows).to_csv(PLAYERS_OUT, index=False)
    posted = sum(1 for x in summary_rows if x["lineup_status"] == "posted")
    print(f"Captured lineup status for {len(summary_rows):,} team-game rows; {posted:,} posted lineups.")


if __name__ == "__main__":
    main()
