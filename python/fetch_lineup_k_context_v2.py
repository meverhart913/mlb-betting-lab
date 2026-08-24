"""Collect posted MLB lineups and batter strikeout context for Pitcher-K V2.

V1 is untouched. V2 only uses an actual batting order exposed by MLB's live game
feed. If a lineup is not posted, the game is recorded as unavailable rather than
inventing a projected lineup. Current-season batter K/PA and batting side are
captured prospectively so later V2 models can be validated without hindsight.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
CURRENT.mkdir(parents=True, exist_ok=True)
OUT = CURRENT / "pitcher_k_v2_lineup_context.csv"
HISTORY = CURRENT / "pitcher_k_v2_lineup_context_history.csv"
STATUS = CURRENT / "pitcher_k_v2_lineup_status.csv"
API = "https://statsapi.mlb.com/api"


def get_json(url: str, params=None, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i + 1 < tries:
                time.sleep(1.0 + i)
    raise last


def season_hitting(player_id: int, season: int, cache: dict) -> dict:
    if player_id in cache:
        return cache[player_id]
    payload = get_json(
        f"{API}/v1/people/{player_id}/stats",
        {"stats": "season", "group": "hitting", "season": season},
    )
    stat = {}
    for block in payload.get("stats", []):
        splits = block.get("splits") or []
        if splits:
            stat = splits[0].get("stat") or {}
            break
    so = pd.to_numeric(stat.get("strikeOuts"), errors="coerce")
    pa = pd.to_numeric(stat.get("plateAppearances"), errors="coerce")
    ab = pd.to_numeric(stat.get("atBats"), errors="coerce")
    bb = pd.to_numeric(stat.get("baseOnBalls"), errors="coerce")
    if pd.isna(pa):
        pa = (0 if pd.isna(ab) else ab) + (0 if pd.isna(bb) else bb)
    out = {
        "season_pa": float(pa) if pd.notna(pa) else np.nan,
        "season_so": float(so) if pd.notna(so) else np.nan,
        "season_k_per_pa": float(so / pa) if pd.notna(so) and pd.notna(pa) and pa > 0 else np.nan,
    }
    cache[player_id] = out
    return out


def append_history(fresh: pd.DataFrame) -> None:
    if fresh.empty:
        return
    if HISTORY.exists():
        old = pd.read_csv(HISTORY, low_memory=False)
        out = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        out = fresh.copy()
    keys = ["game_id", "pitcher_id", "batter_id", "snapshot_time_et"]
    out = out.drop_duplicates(keys, keep="last")
    out.to_csv(HISTORY, index=False)


def main() -> None:
    day = date.today().isoformat()
    season = date.today().year
    snapshot = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes")
    sched = get_json(
        f"{API}/v1/schedule",
        {"sportId": 1, "date": day, "gameType": "R", "hydrate": "probablePitcher"},
    )
    games = [g for d in sched.get("dates", []) for g in d.get("games", [])]
    rows = []
    game_status = []
    stat_cache: dict[int, dict] = {}

    for g in games:
        game_id = int(g["gamePk"])
        feed = get_json(f"{API}/v1.1/game/{game_id}/feed/live")
        box = (feed.get("liveData") or {}).get("boxscore") or {}
        teams_box = box.get("teams") or {}
        gd = feed.get("gameData") or {}
        people = gd.get("players") or {}

        for pitcher_side, lineup_side in (("away", "home"), ("home", "away")):
            sched_team = ((g.get("teams") or {}).get(pitcher_side) or {})
            pp = sched_team.get("probablePitcher") or {}
            if not pp.get("id"):
                continue
            pitcher_id = int(pp["id"])
            pitcher_name = pp.get("fullName")
            pitcher_key = people.get(f"ID{pitcher_id}") or {}
            pitch_hand = ((pitcher_key.get("pitchHand") or {}).get("code"))

            side_box = teams_box.get(lineup_side) or {}
            order = side_box.get("battingOrder") or []
            posted = len(order) >= 9
            game_status.append({
                "date": day, "game_id": game_id, "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name, "opponent_side": lineup_side,
                "lineup_posted": int(posted), "lineup_count": len(order),
                "snapshot_time_et": snapshot,
            })
            if not posted:
                continue

            for slot, batter_id in enumerate(order[:9], start=1):
                batter_id = int(batter_id)
                p = people.get(f"ID{batter_id}") or {}
                stats = season_hitting(batter_id, season, stat_cache)
                rows.append({
                    "date": day, "game_id": game_id,
                    "pitcher_id": pitcher_id, "pitcher_name": pitcher_name,
                    "pitcher_hand": pitch_hand,
                    "batter_id": batter_id, "batter_name": p.get("fullName"),
                    "batter_side": ((p.get("batSide") or {}).get("code")),
                    "batting_order": slot,
                    **stats,
                    "snapshot_time_et": snapshot,
                    "source": "MLB Stats API live feed + season stats",
                })

    status = pd.DataFrame(game_status)
    status.to_csv(STATUS, index=False)
    out = pd.DataFrame(rows)
    if not out.empty:
        # PA-weighted lineup K rate, capped to prevent one tiny-sample player dominating.
        out["weight_pa"] = pd.to_numeric(out["season_pa"], errors="coerce").clip(lower=25, upper=500)
        out["weighted_k"] = out["season_k_per_pa"] * out["weight_pa"]
        grp = out.groupby(["game_id", "pitcher_id"], dropna=False)
        summary = grp.agg(
            lineup_batters=("batter_id", "nunique"),
            lineup_mean_k_per_pa=("season_k_per_pa", "mean"),
            lineup_total_pa=("season_pa", "sum"),
            weighted_k_sum=("weighted_k", "sum"),
            weight_sum=("weight_pa", "sum"),
        ).reset_index()
        summary["lineup_weighted_k_per_pa"] = summary["weighted_k_sum"] / summary["weight_sum"]
        out = out.merge(
            summary[["game_id", "pitcher_id", "lineup_batters", "lineup_mean_k_per_pa", "lineup_weighted_k_per_pa"]],
            on=["game_id", "pitcher_id"], how="left",
        )
    out.to_csv(OUT, index=False)
    append_history(out)

    posted_pairs = int(status["lineup_posted"].sum()) if not status.empty else 0
    total_pairs = len(status)
    print(f"V2 lineup capture: {posted_pairs}/{total_pairs} probable-starter opponent lineups posted; {len(out)} batter rows captured.")


if __name__ == "__main__":
    main()
