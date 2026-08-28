from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import run_v22_fanduel_paper as v22  # noqa: E402
import run_fanduel_hybrid_paper as hybrid  # noqa: E402


class FanDuelPaperTests(unittest.TestCase):
    def test_integer_line_probabilities_include_push(self):
        over, under, push = v22.fair_probs(5.0, 5.0)
        self.assertAlmostEqual(over + under + push, 1.0, places=10)
        self.assertGreater(push, 0.0)

    def test_half_line_has_no_push(self):
        over, under, push = v22.fair_probs(5.5, 5.0)
        self.assertAlmostEqual(over + under, 1.0, places=10)
        self.assertEqual(push, 0.0)

    def test_american_odds_profit(self):
        self.assertAlmostEqual(v22.win_profit(150), 1.5)
        self.assertAlmostEqual(v22.win_profit(-200), 0.5)
        self.assertAlmostEqual(v22.implied_prob(100), 0.5)
        self.assertAlmostEqual(v22.implied_prob(-100), 0.5)

    def test_pair_fanduel_preserves_alt_lines(self):
        raw = pd.DataFrame([
            {"date":"2026-08-28","event_id":"e1","pitcher_name":"Test Pitcher","side":"over","line":4.5,"price":-110,"sportsbook":"FanDuel","commence_time_utc":"2026-08-28T23:00:00Z","collected_at_utc":"2026-08-28T21:00:00Z"},
            {"date":"2026-08-28","event_id":"e1","pitcher_name":"Test Pitcher","side":"under","line":4.5,"price":-110,"sportsbook":"FanDuel","commence_time_utc":"2026-08-28T23:00:00Z","collected_at_utc":"2026-08-28T21:00:00Z"},
            {"date":"2026-08-28","event_id":"e1","pitcher_name":"Test Pitcher","side":"over","line":5.5,"price":120,"sportsbook":"FanDuel","commence_time_utc":"2026-08-28T23:00:00Z","collected_at_utc":"2026-08-28T21:00:00Z"},
            {"date":"2026-08-28","event_id":"e1","pitcher_name":"Test Pitcher","side":"under","line":5.5,"price":-145,"sportsbook":"FanDuel","commence_time_utc":"2026-08-28T23:00:00Z","collected_at_utc":"2026-08-28T21:00:00Z"},
            {"date":"2026-08-28","event_id":"e1","pitcher_name":"Test Pitcher","side":"over","line":5.5,"price":125,"sportsbook":"OtherBook","commence_time_utc":"2026-08-28T23:00:00Z","collected_at_utc":"2026-08-28T21:00:00Z"},
        ])
        paired = v22.pair_fanduel(raw, "2026-08-28")
        self.assertEqual(set(paired.line.astype(float)), {4.5, 5.5})
        self.assertEqual(len(paired), 2)

    def test_candidate_ev_uses_actual_price(self):
        market = pd.DataFrame([{
            "date":"2026-08-28","event_id":"e1","name_key":"testpitcher","pitcher_name":"Test Pitcher","line":4.5,
            "commence_time_utc":pd.Timestamp("2026-08-28T23:00:00Z"),"collected_at_utc":pd.Timestamp("2026-08-28T21:00:00Z"),
            "over_price":150.0,"under_price":-190.0,
        }])
        projection = pd.DataFrame([{
            "game_id":1,"pitcher_id":10,"pitcher_name":"Test Pitcher","name_key":"testpitcher","projected_bf":24.0,
            "projected_k_rate":0.22,"projected_k":5.3,"lineup_match_coverage":1.0,"model_version":"v22_lineup_all_live",
            "model_generated_at_et":"2026-08-28T17:00:00-04:00",
        }])
        c = v22.select_candidates(market, projection)
        over = c[c.side.eq("OVER")].iloc[0]
        expected = over.model_win_prob * 1.5 - (1.0 - over.model_win_prob)
        self.assertAlmostEqual(over.expected_profit_per_unit, expected, places=10)

    @patch.object(hybrid, "fit_v21_and_predict")
    @patch.object(hybrid, "fit_v22_and_predict")
    def test_hybrid_uses_v21_only_for_missing_v22_starts(self, m22, m21):
        m22.return_value = pd.DataFrame([
            {"game_id":1,"pitcher_id":10,"pitcher_name":"A","model_version":"v22_lineup_all_live"}
        ])
        m21.return_value = pd.DataFrame([
            {"game_id":1,"pitcher_id":10,"pitcher_name":"A","model_version":"v21_statcast_fallback_live"},
            {"game_id":2,"pitcher_id":20,"pitcher_name":"B","model_version":"v21_statcast_fallback_live"},
        ])
        out = hybrid.hybrid_predictions("2026-08-28")
        self.assertEqual(len(out), 2)
        versions = dict(zip(out.pitcher_name, out.model_version))
        self.assertEqual(versions["A"], "v22_lineup_all_live")
        self.assertEqual(versions["B"], "v21_statcast_fallback_live")


if __name__ == "__main__":
    unittest.main()
