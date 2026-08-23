import json
import csv
import math

INPUT_FILE = "mlb_odds_dataset.json"
MAX_ROWS_PER_FILE = 100000

# Load JSON
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


# Build complete column list
fieldnames = sorted(
    {key for row in rows for key in row.keys()}
)

# Split output into manageable files
num_files = math.ceil(
    len(rows) / MAX_ROWS_PER_FILE
)

print()
print(f"Total records: {len(rows):,}")
print(f"Creating {num_files} files...")
print()

for i in range(num_files):

    start = i * MAX_ROWS_PER_FILE
    end = start + MAX_ROWS_PER_FILE

    chunk = rows[start:end]

    filename = f"mlb_odds_part_{i + 1}.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(chunk)

    print(
        f"Created {filename}: "
        f"{len(chunk):,} rows"
    )


print()
print("Finished.")
print(f"{len(data):,} dates processed.")
print(f"{len(rows):,} sportsbook-market records exported.")