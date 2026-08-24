"""Capture current active-roster and injured-list context from MLB Stats API.

The active roster is authoritative for who is currently active. IL state is
reconstructed conservatively from each player's latest injury-list transaction
in the current season. Raw transaction descriptions are retained for audit.
These fields are context-only until historically validated.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
SUMMARY_OUT = CURRENT / "roster_context.csv"
IL_OUT = CURRENT / "injured_list_players.csv"
TX_OUT = CURRENT / "recent_roster_transactions.csv"
BASE = "https://statsapi.mlb.com/api/v1"


def schedule_teams(day: str) -> pd.DataFrame:
    r = requests.get(f"{BASE}/schedule", params={"sportId": 1, "date": day, "gameType": "R"}, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            for side in ("away", "home"):
                team = ((g.get("teams") or {}).get(side) or {}).get("team") or {}
                rows.append({"date": d.get("date"), "game_id": g.get("gamePk"), "side": side,
                             "team_id": team.get("id"), "team_name": team.get("name")})
    return pd.DataFrame(rows)


def active_roster(team_id: int) -> list[dict]:
    r = requests.get(f"{BASE}/teams/{team_id}/roster", params={"rosterType": "active"}, timeout=30)
    r.raise_for_status()
    return r.json().get("roster", []) or []


def transactions(team_id: int, start: str, end: str) -> list[dict]:
    r = requests.get(f"{BASE}/transactions", params={"teamId": team_id, "startDate": start, "endDate": end}, timeout=30)
    r.raise_for_status()
    return r.json().get("transactions", []) or []


def classify_il(description: str) -> str | None:
    d = (description or "").lower()
    if not any(x in d for x in ("injured list", "disabled list", "day il", "-day il")):
        return None
    if any(x in d for x in ("reinstated", "activated from", "returned from")):
        return "off_il"
    if any(x in d for x in ("placed on", "transferred to", "selected to the injured list")):
        return "on_il"
    return "il_event"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    target = pd.Timestamp(args.date)
    season_start = pd.Timestamp(year=target.year, month=2, day=1).date().isoformat()
    teams = schedule_teams(args.date)
    if teams.empty:
        CURRENT.mkdir(parents=True, exist_ok=True)
        teams.to_csv(SUMMARY_OUT, index=False)
        print(f"No regular-season teams found for {args.date}.")
        return

    summary_rows, il_rows, tx_rows = [], [], []
    for team in teams[["team_id", "team_name"]].drop_duplicates().itertuples(index=False):
        roster = active_roster(int(team.team_id))
        tx = transactions(int(team.team_id), season_start, args.date)
        active_ids = set()
        pitcher_count = 0
        for item in roster:
            person = item.get("person") or {}
            pid = person.get("id")
            if pid is not None:
                active_ids.add(pid)
            pos = item.get("position") or {}
            if (pos.get("type") or "").lower() == "pitcher":
                pitcher_count += 1

        latest_il = {}
        for item in tx:
            person = item.get("person") or {}
            pid = person.get("id")
            desc = item.get("description") or ""
            event = classify_il(desc)
            if event:
                dt = pd.to_datetime(item.get("date") or item.get("effectiveDate"), errors="coerce")
                rec = {"team_id": team.team_id, "team_name": team.team_name, "player_id": pid,
                       "player_name": person.get("fullName"), "date": str(item.get("date") or item.get("effectiveDate") or ""),
                       "event": event, "description": desc}
                tx_rows.append(rec)
                key = pid or person.get("fullName")
                old = latest_il.get(key)
                if old is None or (pd.notna(dt) and dt >= old[0]):
                    latest_il[key] = (dt, rec)

        current_il = []
        for _, rec in latest_il.values():
            if rec["event"] == "on_il" and rec["player_id"] not in active_ids:
                current_il.append(rec)
                il_rows.append(rec)

        summary_rows.append({
            "date": args.date,
            "team_id": team.team_id,
            "team_name": team.team_name,
            "active_roster_count": len(active_ids),
            "active_pitchers": pitcher_count,
            "active_position_players": max(len(active_ids) - pitcher_count, 0),
            "inferred_il_count": len(current_il),
            "source": "mlb-stats-api",
        })

    CURRENT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
    pd.DataFrame(il_rows).to_csv(IL_OUT, index=False)
    pd.DataFrame(tx_rows).to_csv(TX_OUT, index=False)
    print(f"Wrote roster context for {len(summary_rows):,} teams; inferred {len(il_rows):,} current IL players.")


if __name__ == "__main__":
    main()
