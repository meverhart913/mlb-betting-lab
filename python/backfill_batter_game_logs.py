"""Incrementally backfill batter-level MLB boxscore history.

The job is intentionally resumable. It reads the canonical game list, skips game
IDs already present in the batter log, fetches a bounded batch from MLB StatsAPI,
and appends/deduplicates the results. This is research data collection only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived"
CURRENT = DATA / "current"
GAMES = DATA / "mlb_games_2018_present.csv"
OUT = DERIVED / "mlb_batter_game_logs.csv"
STATE = CURRENT / "batter_backfill_state.csv"
BASE = "https://statsapi.mlb.com/api/v1/game"


def get_with_retry(url: str, attempts: int = 4) -> dict:
    delay = 0.75
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"retryable HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed {url}: {last}")


def side_rows(payload: dict, game_id: int, game_date: str, side: str) -> list[dict]:
    teams = payload.get("teams") or {}
    team = teams.get(side) or {}
    opp_side = "home" if side == "away" else "away"
    opp = teams.get(opp_side) or {}
    team_id = (team.get("team") or {}).get("id")
    opp_id = (opp.get("team") or {}).get("id")
    out = []
    for player in (team.get("players") or {}).values():
        person = player.get("person") or {}
        batting = ((player.get("stats") or {}).get("batting") or {})
        order_raw = player.get("battingOrder")
        try:
            batting_order = int(order_raw) // 100 if order_raw not in (None, "") else None
        except (TypeError, ValueError):
            batting_order = None
        # A batting order indicates the player appeared in the offensive lineup;
        # retain bench/substitute appearances too when batting stats exist.
        has_batting = bool(batting) or batting_order is not None
        if not has_batting:
            continue
        ab = pd.to_numeric(batting.get("atBats"), errors="coerce")
        bb = pd.to_numeric(batting.get("baseOnBalls"), errors="coerce")
        hbp = pd.to_numeric(batting.get("hitByPitch"), errors="coerce")
        sf = pd.to_numeric(batting.get("sacFlies"), errors="coerce")
        sh = pd.to_numeric(batting.get("sacBunts"), errors="coerce")
        # MLB boxscore payloads do not always expose plateAppearances directly.
        # This approximation is sufficient for K-rate and is explicitly stored.
        vals = [0 if pd.isna(x) else float(x) for x in (ab, bb, hbp, sf, sh)]
        approx_pa = sum(vals)
        out.append({
            "game_id": game_id,
            "date": game_date,
            "side": side,
            "team_id": team_id,
            "opponent_team_id": opp_id,
            "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "batting_order": batting_order,
            "in_starting_lineup": int(batting_order is not None and batting_order > 0),
            "at_bats": batting.get("atBats"),
            "approx_plate_appearances": approx_pa,
            "hits": batting.get("hits"),
            "doubles": batting.get("doubles"),
            "triples": batting.get("triples"),
            "home_runs": batting.get("homeRuns"),
            "walks": batting.get("baseOnBalls"),
            "strikeouts": batting.get("strikeOuts"),
            "hit_by_pitch": batting.get("hitByPitch"),
            "sac_flies": batting.get("sacFlies"),
            "sac_bunts": batting.get("sacBunts"),
            "runs": batting.get("runs"),
            "rbi": batting.get("rbi"),
        })
    return out


def fetch_game(row: tuple[int, str]) -> tuple[int, list[dict], str | None]:
    game_id, game_date = row
    try:
        payload = get_with_retry(f"{BASE}/{int(game_id)}/boxscore")
        rows = side_rows(payload, int(game_id), game_date, "away")
        rows += side_rows(payload, int(game_id), game_date, "home")
        return int(game_id), rows, None
    except Exception as exc:
        return int(game_id), [], str(exc)


def load_games() -> pd.DataFrame:
    g = pd.read_csv(GAMES, low_memory=False)
    if "game_id" not in g.columns:
        raise ValueError("mlb_games_2018_present.csv is missing game_id")
    date_col = "date" if "date" in g.columns else "game_date" if "game_date" in g.columns else None
    if date_col is None:
        raise ValueError("mlb_games_2018_present.csv is missing date/game_date")
    g["game_id"] = pd.to_numeric(g["game_id"], errors="coerce")
    g["date"] = pd.to_datetime(g[date_col], errors="coerce").dt.date.astype("string")
    g = g[g["game_id"].notna() & g["date"].notna()][["game_id", "date"]].drop_duplicates("game_id")
    g["game_id"] = g["game_id"].astype(int)
    return g.sort_values(["date", "game_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=350)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.workers > 12:
        raise SystemExit("Invalid batch-size/workers")

    DERIVED.mkdir(parents=True, exist_ok=True)
    CURRENT.mkdir(parents=True, exist_ok=True)
    games = load_games()
    existing = pd.read_csv(OUT, low_memory=False) if OUT.exists() else pd.DataFrame()
    done = set(pd.to_numeric(existing.get("game_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    pending = games[~games["game_id"].isin(done)].head(args.batch_size)

    if pending.empty:
        pd.DataFrame([{
            "total_games": len(games), "completed_games": len(done), "remaining_games": 0,
            "last_batch_requested": 0, "last_batch_succeeded": 0, "last_batch_failed": 0,
            "complete": True,
        }]).to_csv(STATE, index=False)
        print(f"Batter history complete: {len(done):,}/{len(games):,} games.")
        return

    results: list[dict] = []
    failed: list[tuple[int, str]] = []
    work = [(int(r.game_id), str(r.date)) for r in pending.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_game, item): item for item in work}
        for fut in as_completed(futures):
            gid, rows, err = fut.result()
            if err:
                failed.append((gid, err))
            else:
                results.extend(rows)

    fresh = pd.DataFrame(results)
    if not fresh.empty:
        if existing.empty:
            combined = fresh
        else:
            combined = pd.concat([existing, fresh], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(["game_id", "team_id", "player_id"], keep="last")
        combined = combined.sort_values(["date", "game_id", "side", "batting_order", "player_id"], na_position="last")
        combined.to_csv(OUT, index=False)
    else:
        combined = existing

    completed_now = set(pd.to_numeric(combined.get("game_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    state = pd.DataFrame([{
        "total_games": len(games),
        "completed_games": len(completed_now),
        "remaining_games": max(len(games) - len(completed_now), 0),
        "last_batch_requested": len(work),
        "last_batch_succeeded": len(work) - len(failed),
        "last_batch_failed": len(failed),
        "complete": len(completed_now) >= len(games),
    }])
    state.to_csv(STATE, index=False)
    if failed:
        fail_path = CURRENT / "batter_backfill_failures.csv"
        pd.DataFrame(failed, columns=["game_id", "error"]).to_csv(fail_path, index=False)
    print(state.to_string(index=False))
    print(f"Rows stored: {len(combined):,}; failed game requests this batch: {len(failed)}")

if __name__ == "__main__":
    main()
