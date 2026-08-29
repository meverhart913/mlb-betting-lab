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


class FanDuelClvArchiveIntegrityTests(unittest.TestCase):
    @staticmethod
    def _row(price: int) -> dict:
        return {
            "date": "2026-08-29",
            "event_id": "event-1",
            "sportsbook": "FanDuel",
            "pitcher_name": "Test Pitcher",
            "side": "OVER",
            "line": 5.5,
            "price": price,
            "collected_at_utc": "2026-08-29T20:00:00Z",
            "commence_time_utc": "2026-08-29T22:00:00Z",
        }

    def test_public_propline_sample_is_excluded_from_prospective_clv(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            day = archive / "2026-08-29"
            day.mkdir()
            pd.DataFrame([self._row(-140)]).to_csv(
                day / "propline_pitcher_k_20260829T200000Z.csv", index=False
            )

            with patch.object(grader, "ARCHIVE", archive):
                got = grader.load_archive()

            self.assertTrue(got.empty)

    def test_verified_decision_snapshot_is_available_for_prospective_clv(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            day = archive / "2026-08-29"
            day.mkdir()
            pd.DataFrame([self._row(-140)]).to_csv(
                day / "propline_pitcher_k_20260829T200000Z.csv", index=False
            )
            pd.DataFrame([self._row(-125)]).to_csv(
                day / "decision-160000.csv", index=False
            )

            with patch.object(grader, "ARCHIVE", archive):
                got = grader.load_archive()

            self.assertEqual(len(got), 1)
            self.assertEqual(float(got.iloc[0]["price"]), -125.0)
            self.assertEqual(str(got.iloc[0]["sportsbook"]), "fanduel")

    def test_wrong_event_id_cannot_supply_clv_for_same_pitcher_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            history = t / "history.csv"
            logs = t / "logs.csv"
            archive = t / "archive"
            out = t / "out"
            day = archive / "2026-08-29"
            day.mkdir(parents=True)
            out.mkdir()

            pd.DataFrame([{
                "date": "2026-08-29", "game_id": 1, "event_id": "correct-event", "pitcher_id": 10,
                "pitcher_name": "Test Pitcher", "line": 5.5, "side": "OVER", "fanduel_price": -110,
                "model_win_prob": 0.58, "model_market_edge": 0.05,
                "collected_at_utc": "2026-08-29T18:00:00Z", "commence_time_utc": "2026-08-29T22:00:00Z",
                "model_version": "v22_lineup_all_live",
            }]).to_csv(history, index=False)
            pd.DataFrame([{
                "date": "2026-08-29", "game_id": 1, "pitcher_id": 10,
                "pitcher_name": "Test Pitcher", "strikeouts": 6, "is_starter": 1,
            }]).to_csv(logs, index=False)
            wrong = self._row(-150)
            wrong["event_id"] = "different-event"
            wrong["collected_at_utc"] = "2026-08-29T21:00:00Z"
            pd.DataFrame([wrong]).to_csv(day / "decision-170000.csv", index=False)

            with (
                patch.object(grader, "HISTORY", history),
                patch.object(grader, "LOG", logs),
                patch.object(grader, "ARCHIVE", archive),
                patch.object(grader, "OUT", out),
                patch.object(grader, "GRADED", out / "graded.csv"),
                patch.object(grader, "SUMMARY", out / "summary.csv"),
                patch.object(grader, "CALIB", out / "calibration.csv"),
                patch.object(grader, "MODEL_SUMMARY", out / "models.csv"),
            ):
                grader.grade()

            got = pd.read_csv(history)
            self.assertTrue(pd.isna(got.loc[0, "clv_implied_prob"]))
            self.assertTrue(pd.isna(got.loc[0, "closing_same_line_price"]))


if __name__ == "__main__":
    unittest.main()
