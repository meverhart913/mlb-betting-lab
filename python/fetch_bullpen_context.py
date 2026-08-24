"""Build current bullpen workload/availability proxies from existing MLB logs.

No external credential is required. Pitcher appearances are joined to team IDs
through the team game logs, then only relief appearances strictly before the
target date are used. Output is audit/context data until historical tests show
stable predictive value.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = CURRENT / "bullpen_context.csv"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def schedule_teams(day: str) -> pd.DataFrame:
    r = requests.get(SCHEDULE_URL, params={"sportId": 1, "date": day, "gameType": "R"}, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            for side in ("away", "home"):
                team = ((g.get("teams") or {}).get(side) or {}).get("team") or {}
                rows.append({
                    "date": d.get("date"),
                    "game_id": g.get("gamePk"),
                    "side": side,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                })
    return pd.DataFrame(rows)


def build_context(target_day: str) -> pd.DataFrame:
    target = pd.Timestamp(target_day)
    pitchers = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    teams = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    pitchers["date"] = pd.to_datetime(pitchers["date"], errors="coerce").dt.normalize()
    teams["date"] = pd.to_datetime(teams["date"], errors="coerce").dt.normalize()
    pitchers["is_starter"] = pd.to_numeric(pitchers["is_starter"], errors="coerce").fillna(0).astype(int)
    pitchers["pitches"] = pd.to_numeric(pitchers["pitches"], errors="coerce")
    rel = pitchers[pitchers["is_starter"] == 0].merge(
        teams[["game_id", "side", "team_id", "team_name"]].drop_duplicates(["game_id", "side"]),
        on=["game_id", "side"], how="left",
    )
    rel = rel[(rel["date"].notna()) & (rel["date"] < target)].copy()
    today = schedule_teams(target_day)
    if today.empty:
        return today

    rows = []
    for t in today.itertuples(index=False):
        z = rel[rel["team_id"] == t.team_id].copy()
        row = t._asdict()
        for days in (1, 2, 3):
            start = target - pd.Timedelta(days=days)
            q = z[(z["date"] >= start) & (z["date"] < target)]
            row[f"bullpen_pitches_{days}d"] = float(q["pitches"].sum(min_count=1)) if not q.empty else 0.0
            row[f"relief_appearances_{days}d"] = int(len(q))
            row[f"unique_relievers_{days}d"] = int(q["pitcher_id"].nunique())
        q1 = z[z["date"] == target - pd.Timedelta(days=1)]
        q2 = z[z["date"] == target - pd.Timedelta(days=2)]
        q3 = z[z["date"] == target - pd.Timedelta(days=3)]
        row["relievers_20plus_1d"] = int((q1.groupby("pitcher_id")["pitches"].sum() >= 20).sum()) if not q1.empty else 0
        q_2d = z[(z["date"] >= target - pd.Timedelta(days=2)) & (z["date"] < target)]
        row["relievers_30plus_2d"] = int((q_2d.groupby("pitcher_id")["pitches"].sum() >= 30).sum()) if not q_2d.empty else 0
        s1, s2, s3 = set(q1["pitcher_id"].dropna()), set(q2["pitcher_id"].dropna()), set(q3["pitcher_id"].dropna())
        row["back_to_back_relievers"] = len(s1 & s2)
        row["three_straight_relievers"] = len(s1 & s2 & s3)
        row["max_reliever_pitches_1d"] = float(q1.groupby("pitcher_id")["pitches"].sum().max()) if not q1.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    out = build_context(args.date)
    CURRENT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote bullpen context for {len(out):,} team-game rows to {OUT}.")


if __name__ == "__main__":
    main()
