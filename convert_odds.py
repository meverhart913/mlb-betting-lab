import json
import csv

INPUT_FILE = "mlb_odds_dataset.json"
OUTPUT_FILE = "mlb_odds_2021_2025.csv"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for date, games in data.items():
    for game in games:

        gv = game.get("gameView", {})
        odds = game.get("odds", {})

        base = {
            "date": date,
            "start_datetime": gv.get("startDate"),
            "away_team": gv.get("awayTeam", {}).get("fullName"),
            "home_team": gv.get("homeTeam", {}).get("fullName"),
            "away_score": gv.get("awayTeamScore"),
            "home_score": gv.get("homeTeamScore"),
            "status": gv.get("gameStatusText"),
            "venue": gv.get("venueName"),
            "game_type": gv.get("gameType"),
        }

        # MONEYLINE
        for book in odds.get("moneyline", []):
            opening = book.get("openingLine", {})
            current = book.get("currentLine", {})

            row = base.copy()
            row.update({
                "sportsbook": book.get("sportsbook"),
                "market": "moneyline",

                "open_home_odds": opening.get("homeOdds"),
                "open_away_odds": opening.get("awayOdds"),

                "close_home_odds": current.get("homeOdds"),
                "close_away_odds": current.get("awayOdds"),
            })

            rows.append(row)

        # POINT SPREAD
        for book in odds.get("pointspread", []):
            opening = book.get("openingLine", {})
            current = book.get("currentLine", {})

            row = base.copy()
            row.update({
                "sportsbook": book.get("sportsbook"),
                "market": "pointspread",

                "open_home_odds": opening.get("homeOdds"),
                "open_away_odds": opening.get("awayOdds"),
                "open_home_spread": opening.get("homeSpread"),
                "open_away_spread": opening.get("awaySpread"),

                "close_home_odds": current.get("homeOdds"),
                "close_away_odds": current.get("awayOdds"),
                "close_home_spread": current.get("homeSpread"),
                "close_away_spread": current.get("awaySpread"),
            })

            rows.append(row)

        # TOTALS
        for book in odds.get("totals", []):
            opening = book.get("openingLine", {})
            current = book.get("currentLine", {})

            row = base.copy()
            row.update({
                "sportsbook": book.get("sportsbook"),
                "market": "totals",

                "open_over_odds": opening.get("overOdds"),
                "open_under_odds": opening.get("underOdds"),
                "open_total": opening.get("total"),

                "close_over_odds": current.get("overOdds"),
                "close_under_odds": current.get("underOdds"),
                "close_total": current.get("total"),
            })

            rows.append(row)

fieldnames = sorted(
    {key for row in rows for key in row.keys()}
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Created {OUTPUT_FILE}")
print(f"{len(rows):,} sportsbook-market records")
print(f"{len(data):,} dates processed")