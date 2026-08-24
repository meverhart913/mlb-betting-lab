import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from backfill_batter_game_logs import side_rows


class BatterBackfillTests(unittest.TestCase):
    def test_side_rows_extracts_lineup_and_batting_stats(self):
        payload = {
            "teams": {
                "away": {
                    "team": {"id": 10},
                    "players": {
                        "ID1": {
                            "person": {"id": 1, "fullName": "Test Hitter"},
                            "battingOrder": "300",
                            "stats": {"batting": {
                                "atBats": 4, "hits": 1, "doubles": 0, "triples": 0,
                                "homeRuns": 0, "baseOnBalls": 1, "strikeOuts": 2,
                                "hitByPitch": 0, "sacFlies": 0, "sacBunts": 0,
                                "runs": 1, "rbi": 0,
                            }},
                        }
                    },
                },
                "home": {"team": {"id": 20}, "players": {}},
            }
        }
        rows = side_rows(payload, 123, "2026-08-24", "away")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["game_id"], 123)
        self.assertEqual(r["team_id"], 10)
        self.assertEqual(r["opponent_team_id"], 20)
        self.assertEqual(r["player_id"], 1)
        self.assertEqual(r["batting_order_code"], 300)
        self.assertEqual(r["batting_order"], 3)
        self.assertEqual(r["in_starting_lineup"], 1)
        self.assertEqual(r["strikeouts"], 2)
        self.assertEqual(r["approx_plate_appearances"], 5.0)

    def test_substitute_retains_slot_but_is_not_starter(self):
        payload = {
            "teams": {
                "away": {"team": {"id": 10}, "players": {
                    "ID2": {
                        "person": {"id": 2, "fullName": "Pinch Hitter"},
                        "battingOrder": "301",
                        "stats": {"batting": {"atBats": 1, "strikeOuts": 1}},
                    }
                }},
                "home": {"team": {"id": 20}, "players": {}},
            }
        }
        r = side_rows(payload, 123, "2026-08-24", "away")[0]
        self.assertEqual(r["batting_order_code"], 301)
        self.assertEqual(r["batting_order"], 3)
        self.assertEqual(r["in_starting_lineup"], 0)

    def test_non_batter_is_ignored(self):
        payload = {
            "teams": {
                "away": {"team": {"id": 10}, "players": {
                    "ID9": {"person": {"id": 9, "fullName": "Pitcher Only"}, "stats": {"pitching": {"inningsPitched": "1.0"}}}
                }},
                "home": {"team": {"id": 20}, "players": {}},
            }
        }
        self.assertEqual(side_rows(payload, 123, "2026-08-24", "away"), [])


if __name__ == "__main__":
    unittest.main()
