"""Test a leakage-safe pregame Elo strength feature against the current model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from test_bullpen_features import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

BASE_ELO = 1500.0
K = 20.0
HOME_ADVANTAGE = 35.0
SEASON_CARRY = 2.0 / 3.0


def expected_home(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((home_elo + HOME_ADVANTAGE) - away_elo) / 400.0))


def build_elo() -> pd.DataFrame:
    games = pd.read_csv(DATA / "mlb_games_2018_present.csv", low_memory=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games[games["date"].notna()].sort_values(["date", "game_id"]).copy()
    ratings: dict[str, float] = {}
    rows = []
    current_year = None
    for g in games.itertuples(index=False):
        year = g.date.year
        if current_year is None:
            current_year = year
        elif year != current_year:
            ratings = {team: BASE_ELO + SEASON_CARRY * (rating - BASE_ELO) for team, rating in ratings.items()}
            current_year = year
        h = ratings.get(g.home_team, BASE_ELO)
        a = ratings.get(g.away_team, BASE_ELO)
        ph = expected_home(h, a)
        rows.append({
            "game_id": g.game_id,
            "pregame_home_elo": h,
            "pregame_away_elo": a,
            "diff_elo": h - a,
            "elo_expected_home": ph,
        })
        if pd.notna(g.home_score) and pd.notna(g.away_score) and g.home_score != g.away_score:
            actual = 1.0 if g.home_score > g.away_score else 0.0
            delta = K * (actual - ph)
            ratings[g.home_team] = h + delta
            ratings[g.away_team] = a - delta
    return pd.DataFrame(rows).drop_duplicates("game_id", keep="last")


def main() -> None:
    base = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base[base["home_win"].notna() & base["date"].notna()].copy()
    z = base.merge(build_elo(), on="game_id", how="left")
    baseline = sorted([c for c in z.columns if c.startswith("diff_sp_") or c.startswith("diff_team_")])
    rows = evaluate(z, ["diff_elo"], "elo_only")
    rows += evaluate(z, baseline, "team_plus_pitcher")
    rows += evaluate(z, baseline + ["diff_elo"], "team_plus_pitcher_plus_elo")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "elo_feature_comparison.csv", index=False)
    summary = out.groupby("feature_set")[["log_loss", "brier", "auc", "accuracy"]].mean().reset_index()
    summary.to_csv(OUT / "elo_feature_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
