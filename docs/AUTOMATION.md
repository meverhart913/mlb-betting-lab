# MLB Betting Lab automation map

## Fully automated daily inputs

| Data | Source | Credential | Output |
|---|---|---|---|
| Schedule/results | MLB Stats API | none | `data/mlb_games_2018_present.csv` |
| Team box scores | MLB Stats API game feeds | none | `data/mlb_team_game_logs.csv` |
| Pitcher game logs | MLB Stats API game feeds | none | `data/mlb_pitcher_game_logs.csv` |
| Probable starters | MLB Stats API schedule | none | morning prediction input |
| Sportsbook moneylines | The Odds API | `THE_ODDS_API_KEY` | `data/current/morning_odds_raw.csv`, `morning_odds.csv` |
| Venue/location/roof/turf | MLB Stats API venues | none | `data/current/morning_context.csv` |
| Weather forecast | Open-Meteo | none | `data/current/morning_context.csv` |

## Derived automatically

- rolling team form;
- rolling starter performance/workload;
- market no-vig probability;
- median sportsbook consensus;
- best available home/away price and sportsbook;
- model probability and model-vs-market difference;
- starter availability status;
- workflow artifact and job summary.

## Intentionally not automated as betting actions

The system does not place wagers and currently forces every row to `NO BET`. Research has not demonstrated a stable out-of-sample profitable decision rule.

## Remaining data automation targets

1. bullpen workload/availability derived from recent reliever appearances;
2. MLB roster/transaction and injured-list state;
3. projected/confirmed batting lineups from a stable programmatic source;
4. repeated odds snapshots for morning-to-close line movement;
5. historical backfills of any new feature before it can enter the production decision model.

Manual entry is fallback-only and should not be the primary source for any field with a stable API or derivable history.
