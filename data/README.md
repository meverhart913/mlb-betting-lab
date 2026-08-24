# Data

Canonical inputs used by `python/build_pitcher_model.py`:

- `mlb_games_2018_present.csv`: one row per scheduled MLB regular-season game with scores/results.
- `mlb_game_enrichment.csv`: one row per game with starter IDs and game metadata.
- `mlb_pitcher_game_logs.csv`: pitcher-level game box-score records, including starter flag.
- `mlb_team_game_logs.csv`: team-level batting, pitching, and fielding box-score records.
- `mlb_odds_part_*.csv`: sportsbook market snapshots split into manageable files.

`legacy/` contains older odds exports retained for provenance only. The current pitcher-aware model does not read them.

Historical box-score features must always be shifted before rolling aggregation. Never merge same-game performance values directly into a pregame feature row.
