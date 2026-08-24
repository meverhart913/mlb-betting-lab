import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from run_morning_model_elo import build_elo_history


class EloProductionTests(unittest.TestCase):
    def test_current_game_result_does_not_enter_pregame_rating(self):
        games = pd.DataFrame([
            {"game_id": 1, "date": "2026-04-01", "home_team": "A", "away_team": "B", "home_score": 5, "away_score": 1},
            {"game_id": 2, "date": "2026-04-02", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 8},
        ])
        games["date"] = pd.to_datetime(games["date"])
        elo, _, _ = build_elo_history(games)
        first = elo.loc[elo.game_id == 1].iloc[0]
        second = elo.loc[elo.game_id == 2].iloc[0]
        self.assertAlmostEqual(first.elo_diff, 0.0)
        self.assertGreater(second.elo_diff, 0.0)

    def test_unfinished_game_does_not_update_rating(self):
        games = pd.DataFrame([
            {"game_id": 1, "date": "2026-04-01", "home_team": "A", "away_team": "B", "home_score": None, "away_score": None},
            {"game_id": 2, "date": "2026-04-02", "home_team": "A", "away_team": "B", "home_score": 4, "away_score": 3},
        ])
        games["date"] = pd.to_datetime(games["date"])
        elo, _, _ = build_elo_history(games)
        second = elo.loc[elo.game_id == 2].iloc[0]
        self.assertAlmostEqual(second.elo_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
