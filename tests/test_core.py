import csv
import tempfile
import unittest
from pathlib import Path

from mlb_lab.core import Observation, analyze, fair_implied_probability, load_csv, wilson_interval


class CoreTests(unittest.TestCase):
    def test_no_vig_probability(self):
        self.assertAlmostEqual(fair_implied_probability(1.91, 1.91), 0.5)
        self.assertAlmostEqual(fair_implied_probability(2.0, None), 0.5)

    def test_wilson_interval(self):
        low, high = wilson_interval(70, 100)
        self.assertLess(low, 0.70)
        self.assertGreater(high, 0.70)

    def test_rejects_missing_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("date,market,outcome,decimal_odds\n2026-01-01,ml,1,1.9\n")
            with self.assertRaisesRegex(ValueError, "feature"):
                load_csv(path, ["feature"])

    def test_walk_forward_finds_signal_and_clears_seventy_percent_gate(self):
        rows = []
        for index in range(360):
            feature = 2.0 if index % 5 else -2.0
            outcome = int(feature > 0)
            rows.append(Observation(f"2026-{1 + index // 30:02d}-{1 + index % 28:02d}", "moneyline", outcome, 1.9, 1.9, (feature,)))
        result = analyze(rows, warmup=80, min_edge=0.02, min_bets=100)[0]
        self.assertGreaterEqual(result.hit_rate, 0.70)
        self.assertTrue(result.qualified)


if __name__ == "__main__":
    unittest.main()
