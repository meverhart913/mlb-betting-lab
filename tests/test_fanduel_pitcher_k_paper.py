from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import run_v22_fanduel_paper as v22  # noqa: E402
import run_fanduel_hybrid_paper as hybrid  # noqa: E402
import select_fanduel_paper_from_live as selector  # noqa: E402
import grade_fanduel_pitcher_k_paper as grader  # noqa: E402


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

    def test_timing_guard_allows_model_before_quote(self):
        latest, earliest = selector.assert_projection_before_quote(
            ["2026-08-28T20:59:00Z"], ["2026-08-28T21:00:00Z"]
        )
        self.assertLessEqual(latest, earliest)

    def test_timing_guard_rejects_model_after_quote(self):
        with self.assertRaises(ValueError):
            selector.assert_projection_before_quote(
                ["2026-08-28T21:01:00Z"], ["2026-08-28T21:00:00Z"]
            )

    def test_timing_guard_rejects_missing_timestamp(self):
        with self.assertRaises(ValueError):
            selector.assert_projection_before_quote([None], ["2026-08-28T21:00:00Z"])

    def test_candidate_timing_guard_allows_each_model_before_own_quote(self):
        candidates = pd.DataFrame([
            {"pitcher_name":"A", "line":5.5, "side":"OVER", "model_generated_at_et":"2026-08-28T16:59:00-04:00", "collected_at_utc":"2026-08-28T21:00:00Z"},
            {"pitcher_name":"B", "line":4.5, "side":"UNDER", "model_generated_at_et":"2026-08-28T17:10:00-04:00", "collected_at_utc":"2026-08-28T21:15:00Z"},
        ])
        selector.assert_candidate_timing(candidates)

    def test_candidate_timing_guard_rejects_late_model_row(self):
        candidates = pd.DataFrame([{
            "pitcher_name":"A", "line":5.5, "side":"OVER",
            "model_generated_at_et":"2026-08-28T17:01:00-04:00", "collected_at_utc":"2026-08-28T21:00:00Z",
        }])
        with self.assertRaises(ValueError):
            selector.assert_candidate_timing(candidates)

    def test_paper_eligibility_requires_timing_nonnegative_edge_and_positive_ev(self):
        x = pd.DataFrame([
            {"timing_eligible":True, "model_market_edge":0.03, "expected_profit_per_unit":0.02},
            {"timing_eligible":True, "model_market_edge":-0.01, "expected_profit_per_unit":0.02},
            {"timing_eligible":True, "model_market_edge":0.03, "expected_profit_per_unit":0.0},
            {"timing_eligible":False, "model_market_edge":0.03, "expected_profit_per_unit":0.02},
        ])
        got = selector.mark_paper_eligibility(x)
        self.assertEqual(got.paper_eligible.tolist(), [True, False, False, False])
        self.assertEqual(got.loc[0, "paper_rejection_reason"], "ELIGIBLE")
        self.assertIn("NEGATIVE_MARKET_EDGE", got.loc[1, "paper_rejection_reason"])
        self.assertIn("NONPOSITIVE_EV", got.loc[2, "paper_rejection_reason"])
        self.assertIn("OUTSIDE_DECISION_WINDOW", got.loc[3, "paper_rejection_reason"])

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

    def test_freeze_keeps_one_best_alt_line_and_no_duplicate_history(self):
        with tempfile.TemporaryDirectory() as td:
            t=Path(td); current=t/'current'; out=t/'out'; current.mkdir(); out.mkdir()
            history=current/'history.csv'; today=out/'today.csv'; audit=out/'audit.csv'
            base={
                'date':'2026-08-28','game_id':1,'event_id':'e1','pitcher_id':10,'pitcher_name':'Test Pitcher',
                'fanduel_price':-110,'fanduel_implied_prob':0.5238,'fanduel_no_vig_prob':0.50,'model_win_prob':0.60,
                'push_prob':0.0,'model_market_edge':0.10,'projected_k':5.3,'projected_bf':24.0,'projected_k_rate':0.22,
                'lineup_match_coverage':1.0,'commence_time_utc':'2026-08-28T23:00:00Z','collected_at_utc':'2026-08-28T21:00:00Z',
                'minutes_to_start':120.0,'timing_eligible':True,'model_version':'v22_lineup_all_live',
                'model_generated_at_et':'2026-08-28T16:59:00-04:00'
            }
            candidates=pd.DataFrame([
                {**base,'line':4.5,'side':'OVER','expected_profit_per_unit':0.08},
                {**base,'line':5.5,'side':'OVER','expected_profit_per_unit':0.15},
                {**base,'line':6.5,'side':'UNDER','expected_profit_per_unit':0.05},
            ])
            with patch.object(v22,'CURRENT',current), patch.object(v22,'OUT',out), patch.object(v22,'HISTORY',history), \
                 patch.object(v22,'TODAY_OUT',today), patch.object(v22,'AUDIT_OUT',audit):
                first=v22.freeze(candidates)
                second=v22.freeze(candidates)
            self.assertEqual(len(first),1)
            self.assertEqual(float(first.iloc[0].line),5.5)
            self.assertEqual(float(first.iloc[0].expected_profit_per_unit),0.15)
            self.assertEqual(len(second),1)
            hist=pd.read_csv(history)
            self.assertEqual(len(hist),1)
            self.assertEqual(float(hist.loc[0,'line']),5.5)

    def test_completed_game_with_different_starter_is_void(self):
        with tempfile.TemporaryDirectory() as td:
            t=Path(td); hist=t/'history.csv'; logs=t/'logs.csv'; archive=t/'archive'; out=t/'out'; archive.mkdir(); out.mkdir()
            pd.DataFrame([{
                'date':'2026-08-28','game_id':1,'event_id':'e1','pitcher_id':10,'pitcher_name':'Scratched Pitcher',
                'line':5.5,'side':'OVER','fanduel_price':-110,'model_win_prob':0.60,'model_market_edge':0.08,
                'collected_at_utc':'2026-08-28T20:00:00Z','commence_time_utc':'2026-08-28T22:00:00Z',
                'model_version':'v22_lineup_all_live'
            }]).to_csv(hist,index=False)
            pd.DataFrame([{
                'date':'2026-08-28','game_id':1,'pitcher_id':20,'pitcher_name':'Replacement Starter',
                'strikeouts':6,'is_starter':1
            }]).to_csv(logs,index=False)
            with patch.object(grader,'HISTORY',hist), patch.object(grader,'LOG',logs), patch.object(grader,'ARCHIVE',archive), \
                 patch.object(grader,'OUT',out), patch.object(grader,'GRADED',out/'graded.csv'), patch.object(grader,'SUMMARY',out/'summary.csv'), \
                 patch.object(grader,'CALIB',out/'calib.csv'), patch.object(grader,'MODEL_SUMMARY',out/'models.csv'):
                grader.grade()
            got=pd.read_csv(hist)
            self.assertEqual(got.loc[0,'result'],'VOID_STARTER_CHANGE')
            self.assertEqual(float(got.loc[0,'flat_profit_units']),0.0)


if __name__ == "__main__":
    unittest.main()
