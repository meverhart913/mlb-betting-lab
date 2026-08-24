import unittest

from python.fetch_roster_context import classify_il


class RosterContextTests(unittest.TestCase):
    def test_il_placement(self):
        self.assertEqual(classify_il("Placed John Doe on the 10-day injured list."), "on_il")

    def test_il_transfer_stays_on_il(self):
        self.assertEqual(classify_il("Transferred John Doe to the 60-day injured list."), "on_il")

    def test_reinstatement_clears_il(self):
        self.assertEqual(classify_il("Reinstated John Doe from the 10-day injured list."), "off_il")

    def test_non_injury_transaction_ignored(self):
        self.assertIsNone(classify_il("Optioned John Doe to Triple-A."))


if __name__ == "__main__":
    unittest.main()
