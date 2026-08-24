"""Summarize consensus moneyline movement from stored snapshots.

Both first-seen movement and game-day morning-to-latest movement are retained so
the selected morning decision point is not conflated with an earlier opening-ish
price captured days before the game.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
INFILE = CURRENT / "odds_consensus_history.csv"
OUTFILE = CURRENT / "line_movement.csv"


def american_prob(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    p = pd.Series(np.nan, index=v.index, dtype=float)
    pos, neg = v > 0, v < 0
    p.loc[pos] = 100.0 / (v.loc[pos] + 100.0)
    p.loc[neg] = -v.loc[neg] / (-v.loc[neg] + 100.0)
    return p


def main() -> None:
    if not INFILE.exists():
        raise SystemExit(f"Missing {INFILE}; collect at least one odds snapshot first.")
    h = pd.read_csv(INFILE, low_memory=False)
    if h.empty:
        raise SystemExit("Odds consensus history is empty.")
    h["snapshot_time_et"] = pd.to_datetime(h["snapshot_time_et"], errors="coerce")
    h["game_date"] = pd.to_datetime(h["date"], errors="coerce").dt.date
    h["snapshot_date_et"] = h["snapshot_time_et"].dt.date
    h["home_raw_prob"] = american_prob(h["home_moneyline"])
    h["away_raw_prob"] = american_prob(h["away_moneyline"])
    denom = h["home_raw_prob"] + h["away_raw_prob"]
    h["home_no_vig_prob"] = h["home_raw_prob"] / denom.replace(0, np.nan)
    keys = ["date", "away_team", "home_team"]
    rows = []
    for key, g in h.sort_values("snapshot_time_et").groupby(keys, dropna=False):
        g = g.dropna(subset=["snapshot_time_et"])
        if g.empty:
            continue
        first_seen, latest = g.iloc[0], g.iloc[-1]
        same_day = g[g["snapshot_date_et"] == g["game_date"]]
        morning = same_day.iloc[0] if not same_day.empty else None
        row = {
            "date": key[0], "away_team": key[1], "home_team": key[2],
            "snapshot_count": len(g),
            "first_seen_time_et": first_seen["snapshot_time_et"],
            "latest_snapshot_time_et": latest["snapshot_time_et"],
            "first_seen_home_moneyline": first_seen.get("home_moneyline"),
            "first_seen_home_no_vig_prob": first_seen.get("home_no_vig_prob"),
            "latest_home_moneyline": latest.get("home_moneyline"),
            "latest_home_no_vig_prob": latest.get("home_no_vig_prob"),
            "first_seen_to_latest_home_prob_move_pp": 100.0 * (latest.get("home_no_vig_prob") - first_seen.get("home_no_vig_prob")),
            "game_day_snapshot_count": int(len(same_day)),
        }
        if morning is not None:
            row.update({
                "morning_snapshot_time_et": morning["snapshot_time_et"],
                "morning_home_moneyline": morning.get("home_moneyline"),
                "morning_away_moneyline": morning.get("away_moneyline"),
                "morning_home_no_vig_prob": morning.get("home_no_vig_prob"),
                "morning_to_latest_home_prob_move_pp": 100.0 * (latest.get("home_no_vig_prob") - morning.get("home_no_vig_prob")),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    CURRENT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTFILE, index=False)
    print(f"Wrote line movement summaries for {len(out):,} games to {OUTFILE}.")


if __name__ == "__main__":
    main()
