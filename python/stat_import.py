from pathlib import Path
from datetime import date
import time

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://statsapi.mlb.com/api/v1/schedule"
START_YEAR = 2018
END_YEAR = date.today().year
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "mlb_games_2018_present.csv"

all_games = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Downloading {year}...")
    params = {
        "sportId": 1,
        "season": year,
        "gameType": "R",
        "hydrate": "linescore,probablePitcher",
    }
    response = requests.get(BASE_URL, params=params, timeout=60, verify=False)
    response.raise_for_status()
    data = response.json()

    for day in data.get("dates", []):
        for game in day.get("games", []):
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            row = {
                "season": year,
                "game_id": game["gamePk"],
                "date": day["date"],
                "game_datetime": game["gameDate"],
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "away_score": away.get("score"),
                "home_score": home.get("score"),
                "away_wins": away.get("leagueRecord", {}).get("wins"),
                "away_losses": away.get("leagueRecord", {}).get("losses"),
                "home_wins": home.get("leagueRecord", {}).get("wins"),
                "home_losses": home.get("leagueRecord", {}).get("losses"),
                "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName"),
                "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName"),
                "venue": game.get("venue", {}).get("name"),
                "status": game.get("status", {}).get("detailedState"),
                "series": game.get("seriesDescription"),
                "game_number": game.get("gameNumber"),
            }
            if home.get("score") is not None and away.get("score") is not None:
                if home["score"] > away["score"]:
                    row["winner"] = row["home_team"]
                    row["home_win"] = 1
                elif away["score"] > home["score"]:
                    row["winner"] = row["away_team"]
                    row["home_win"] = 0
                else:
                    row["winner"] = None
                    row["home_win"] = None
            else:
                row["winner"] = None
                row["home_win"] = None
            all_games.append(row)
    time.sleep(0.25)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(all_games).sort_values(["date", "game_datetime"]).reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Downloaded {len(df):,} games.")
print(f"Saved as {OUTPUT_FILE.relative_to(ROOT)}")
print()
print(df.head())
