from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import grade_fanduel_pitcher_k_paper as grader


class FanDuelGradingEdgeCaseTests(unittest.TestCase):
    def test_push_pending_and_same_side_clv_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            history = t / "history.csv"
            logs = t / "logs.csv"
            games = t / "games.csv"
            archive = t / "archive"
            out = t / "out"
            day = archive / "2026-08-29"
            day.mkdir(parents=True)
            out.mkdir()

            pd.DataFrame([
                {
                    "date": "2026-08-29", "game_id": 1, "event_id": "e1", "pitcher_id": 11,
                    "pitcher_name": "Over Push", "line": 5.0, "side": "OVER", "fanduel_price": -110,
                    "model_win_prob": 0.55, "model_market_edge": 0.03,
                    "collected_at_utc": "2026-08-29T18:00:00Z", "commence_time_utc": "2026-08-29T20:00:00Z",
                    "model_version": "v22_lineup_all_live",
                },
                {
                    "date": "2026-08-29", "game_id": 2, "event_id": "e2", "pitcher_id": 22,
                    "pitcher_name": "Under Win", "line": 4.5, "side": "UNDER", "fanduel_price": 100,
                    "model_win_prob": 0.60, "model_market_edge": 0.07,
                    "collected_at_utc": "2026-08-29T18:10:00Z", "commence_time_utc": "2026-08-29T20:10:00Z",
                    "model_version": "v21_statcast_fallback_live",
                },
                {
                    "date": "2026-08-29", "game_id": 3, "event_id": "e3", "pitcher_id": 33,
                    "pitcher_name": "Pending Pitcher", "line": 6.5, "side": "OVER", "fanduel_price": -105,
                    "model_win_prob": 0.54, "model_market_edge": 0.02,
                    "collected_at_utc": "2026-08-29T21:00:00Z", "commence_time_utc": "2026-08-30T01:00:00Z",
                    "model_version": "v22_lineup_all_live",
                },
            ]).to_csv(history, index=False)

            # Game 3 deliberately contains a stale/partial pitcher line. The grader
            # must ignore it until the authoritative game table says the exact game
            # is final.
            pd.DataFrame([
                {"date": "2026-08-29", "game_id": 1, "pitcher_id": 11, "strikeouts": 5, "is_starter": 1},
                {"date": "2026-08-29", "game_id": 2, "pitcher_id": 22, "strikeouts": 3, "is_starter": 1},
                {"date": "2026-08-29", "game_id": 3, "pitcher_id": 33, "strikeouts": 8, "is_starter": 1},
            ]).to_csv(logs, index=False)

            pd.DataFrame([
                {"game_id": 1, "status": "Final"},
                {"game_id": 2, "status": "Completed Early: Rain"},
                {"game_id": 3, "status": "In Progress"},
            ]).to_csv(games, index=False)

            pd.DataFrame([
                {
                    "date": "2026-08-29", "event_id": "e1", "sportsbook": "FanDuel", "pitcher_name": "Over Push",
                    "side": "OVER", "line": 5.0, "price": -130,
                    "collected_at_utc": "2026-08-29T19:30:00Z", "commence_time_utc": "2026-08-29T20:00:00Z",
                },
                {
                    "date": "2026-08-29", "event_id": "e2", "sportsbook": "FanDuel", "pitcher_name": "Under Win",
                    "side": "UNDER", "line": 4.5, "price": -120,
                    "collected_at_utc": "2026-08-29T19:40:00Z", "commence_time_utc": "2026-08-29T20:10:00Z",
                },
            ]).to_csv(day / "decision-154000.csv", index=False)

            with (
                patch.object(grader, "HISTORY", history),
                patch.object(grader, "LOG", logs),
                patch.object(grader, "GAMES", games),
                patch.object(grader, "ARCHIVE", archive),
                patch.object(grader, "OUT", out),
                patch.object(grader, "GRADED", out / "graded.csv"),
                patch.object(grader, "SUMMARY", out / "summary.csv"),
                patch.object(grader, "CALIB", out / "calibration.csv"),
                patch.object(grader, "MODEL_SUMMARY", out / "models.csv"),
            ):
                grader.grade()

            got = pd.read_csv(history).set_index("pitcher_name")
            self.assertEqual(got.loc["Over Push", "result"], "PUSH")
            self.assertEqual(float(got.loc["Over Push", "flat_profit_units"]), 0.0)

            self.assertEqual(got.loc["Under Win", "result"], "WIN")
            self.assertEqual(float(got.loc["Under Win", "flat_profit_units"]), 1.0)

            self.assertEqual(got.loc["Pending Pitcher", "result"], "PENDING")
            self.assertTrue(pd.isna(got.loc["Pending Pitcher", "flat_profit_units"]))
            self.assertEqual(float(got.loc["Pending Pitcher", "actual_k"]), 8.0)

            # Positive CLV means the same selected side became more expensive / more
            # implied-likely before first pitch. The sign must work identically for
            # OVER and UNDER selections.
            self.assertGreater(float(got.loc["Over Push", "clv_implied_prob"]), 0.0)
            self.assertGreater(float(got.loc["Under Win", "clv_implied_prob"]), 0.0)

            summary = pd.read_csv(out / "summary.csv")
            row0 = summary[summary.min_edge.eq(0.0)].iloc[0]
            self.assertEqual(int(row0.independent_bets), 1)
            self.assertEqual(int(row0.pushes), 1)
            self.assertEqual(int(row0.wins), 1)
            self.assertEqual(int(row0.losses), 0)


if __name__ == "__main__":
    unittest.main()
