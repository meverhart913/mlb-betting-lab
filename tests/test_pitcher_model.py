import unittest

import numpy as np
import pandas as pd

from python.build_pitcher_model import american_prob, ip_to_outs, make_game_labels, pitcher_features


class PitcherModelTests(unittest.TestCase):
    def test_innings_notation_to_outs(self):
        self.assertEqual(ip_to_outs("5.2"), 17.0)
        self.assertEqual(ip_to_outs("6.0"), 18.0)
        self.assertTrue(np.isnan(ip_to_outs("4.3")))

    def test_american_probability(self):
        p = american_prob(pd.Series([100, -200]))
        self.assertAlmostEqual(float(p[0]), 0.5)
        self.assertAlmostEqual(float(p[1]), 2 / 3)

    def test_unfinished_game_is_not_labeled_loss(self):
        games = pd.DataFrame({"home_score": [5, None, 2], "away_score": [3, None, 2]})
        labeled = make_game_labels(games)
        self.assertEqual(labeled.loc[0, "home_win"], 1.0)
        self.assertTrue(np.isnan(labeled.loc[1, "home_win"]))
        self.assertTrue(np.isnan(labeled.loc[2, "home_win"]))

    def test_pitcher_features_are_shifted(self):
        p = pd.DataFrame([
            {"game_id": 1, "date": pd.Timestamp("2024-04-01"), "side": "home", "pitcher_id": 9, "is_starter": 1, "innings_pitched": "5.0", "earned_runs": 1, "walks": 1, "strikeouts": 5, "home_runs": 0, "hits": 4, "batters_faced": 20, "pitches": 80},
            {"game_id": 2, "date": pd.Timestamp("2024-04-07"), "side": "away", "pitcher_id": 9, "is_starter": 1, "innings_pitched": "6.0", "earned_runs": 4, "walks": 2, "strikeouts": 6, "home_runs": 1, "hits": 7, "batters_faced": 25, "pitches": 95},
        ])
        wide, _ = pitcher_features(p)
        first = wide.loc[wide.game_id == 1, "home_sp_earned_runs_3"].iloc[0]
        second = wide.loc[wide.game_id == 2, "away_sp_earned_runs_3"].iloc[0]
        self.assertTrue(np.isnan(first))
        self.assertEqual(second, 1.0)


if __name__ == "__main__":
    unittest.main()
