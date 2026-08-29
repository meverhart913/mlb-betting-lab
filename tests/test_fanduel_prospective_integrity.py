from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import run_v22_fanduel_paper as paper


class FrozenLedgerImmutabilityTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, side: str, line: float, ev: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-08-29",
                    "game_id": 777001,
                    "event_id": "fd-event-1",
                    "pitcher_id": 555,
                    "pitcher_name": "Frozen Pitcher",
                    "line": line,
                    "side": side,
                    "fanduel_price": -110,
                    "fanduel_implied_prob": 0.5238,
                    "fanduel_no_vig_prob": 0.50,
                    "model_win_prob": 0.58,
                    "push_prob": 0.0,
                    "model_market_edge": 0.08,
                    "expected_profit_per_unit": ev,
                    "projected_k": 6.2,
                    "projected_bf": 24.0,
                    "projected_k_rate": 0.2583,
                    "lineup_match_coverage": 1.0,
                    "commence_time_utc": "2026-08-29T19:00:00Z",
                    "collected_at_utc": "2026-08-29T17:00:00Z",
                    "minutes_to_start": 120.0,
                    "timing_eligible": True,
                    "model_version": "test",
                    "model_generated_at_et": "2026-08-29T12:55:00-04:00",
                }
            ]
        )

    def test_later_cycle_cannot_replace_first_frozen_line_or_side(self):
        """First write wins even when a later cycle offers a different, higher-EV bet."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            current = base / "current"
            out = base / "outputs"
            history = current / "history.csv"

            with (
                patch.object(paper, "CURRENT", current),
                patch.object(paper, "OUT", out),
                patch.object(paper, "HISTORY", history),
                patch.object(paper, "TODAY_OUT", out / "today.csv"),
                patch.object(paper, "AUDIT_OUT", out / "audit.csv"),
            ):
                paper.freeze(self._candidate(side="OVER", line=5.5, ev=0.12))
                paper.freeze(self._candidate(side="UNDER", line=6.5, ev=0.20))

            frozen = pd.read_csv(history)
            self.assertEqual(len(frozen), 1)
            self.assertEqual(str(frozen.loc[0, "side"]), "OVER")
            self.assertAlmostEqual(float(frozen.loc[0, "line"]), 5.5)
            self.assertAlmostEqual(float(frozen.loc[0, "expected_profit_per_unit"]), 0.12)


if __name__ == "__main__":
    unittest.main()
