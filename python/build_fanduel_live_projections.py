"""Build today's pitcher-K projections before collecting the decision quote.

This ordering protects timing integrity: confirmed-lineup/Statcast inputs and the
model projection are created first, then a later workflow step collects the
FanDuel quote used for the paper decision.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from run_fanduel_hybrid_paper import hybrid_predictions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"


def main() -> None:
    day = date.today().isoformat()
    projections = hybrid_predictions(day)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if projections.empty:
        pd.DataFrame().to_csv(OUT, index=False)
        print(f"No live pitcher-K projections available for {day}.")
        return
    cols = [c for c in [
        "date", "game_id", "pitcher_id", "pitcher_name", "away_team", "home_team",
        "name_key", "projected_k", "projected_bf", "projected_k_rate",
        "lineup_match_coverage", "model_version", "model_generated_at_et"
    ] if c in projections.columns]
    projections[cols].to_csv(OUT, index=False)
    print(projections[["pitcher_name","projected_k","model_version","model_generated_at_et"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
