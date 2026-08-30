from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import select_fanduel_paper_from_live as selector


class FanDuelCycleAuditTests(unittest.TestCase):
    def test_cycle_audit_is_immutable_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            with patch.object(selector, "ROOT", archive), patch.object(selector, "CYCLE_ARCHIVE", archive):
                first = selector.write_cycle_audit(
                    "2026-08-29",
                    "NO_PAPER",
                    no_paper_reason="NO_ELIGIBLE_CANDIDATES",
                    rejection_reason_counts={"NONPOSITIVE_EV": 3},
                )
                second = selector.write_cycle_audit(
                    "2026-08-29",
                    "NO_PAPER",
                    no_paper_reason="NO_CURRENT_FANDUEL_MARKET",
                )

            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            with first.open(encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["status"], "NO_PAPER")
            self.assertEqual(payload["no_paper_reason"], "NO_ELIGIBLE_CANDIDATES")
            self.assertEqual(payload["rejection_reason_counts"]["NONPOSITIVE_EV"], 3)

    def test_rejection_component_counts_split_combined_reasons(self):
        audited = pd.DataFrame(
            [
                {"paper_eligible": False, "paper_rejection_reason": "OUTSIDE_DECISION_WINDOW,NONPOSITIVE_EV"},
                {"paper_eligible": False, "paper_rejection_reason": "NONPOSITIVE_EV"},
                {"paper_eligible": False, "paper_rejection_reason": "NEGATIVE_MARKET_EDGE,NONPOSITIVE_EV"},
                {"paper_eligible": True, "paper_rejection_reason": "ELIGIBLE"},
            ]
        )
        self.assertEqual(
            selector.rejection_component_counts(audited),
            {
                "NEGATIVE_MARKET_EDGE": 1,
                "NONPOSITIVE_EV": 3,
                "OUTSIDE_DECISION_WINDOW": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
