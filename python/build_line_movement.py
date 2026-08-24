"""Summarize first-to-latest consensus moneyline movement from stored snapshots."""
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
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            "date": key[0], "away_team": key[1], "home_team": key[2],
            "snapshot_count": len(g),
            "first_snapshot_time_et": first["snapshot_time_et"],
            "latest_snapshot_time_et": last["snapshot_time_et"],
            "first_home_moneyline": first.get("home_moneyline"),
            "latest_home_moneyline": last.get("home_moneyline"),
            "first_away_moneyline": first.get("away_moneyline"),
            "latest_away_moneyline": last.get("away_moneyline"),
            "first_home_no_vig_prob": first.get("home_no_vig_prob"),
            "latest_home_no_vig_prob": last.get("home_no_vig_prob"),
            "home_prob_move_pp": 100.0 * (last.get("home_no_vig_prob") - first.get("home_no_vig_prob")),
        })
    out = pd.DataFrame(rows)
    CURRENT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTFILE, index=False)
    print(f"Wrote line movement summaries for {len(out):,} games to {OUTFILE}.")


if __name__ == "__main__":
    main()
