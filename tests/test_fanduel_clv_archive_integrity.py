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


if __name__ == "__main__":
    unittest.main()
