# Repository structure

- `data/` canonical raw and derived CSV datasets used by the current model.
- `data/legacy/` older odds exports kept only for provenance/reference.
- `mlb_lab/` reusable backtesting package and CLI.
- `python/` ingestion, conversion, feature engineering, modeling, and evaluation scripts.
- `outputs/` generated model tables, predictions, metrics, diagnostics, and reports.
- `tests/` automated tests.
- `docs/` design notes and operating instructions.
- `archive/` obsolete snapshots and placeholders that are not part of the active pipeline.

## Active data

The pitcher-aware model expects these under `data/`:

- `mlb_games_2018_present.csv`
- `mlb_game_enrichment.csv`
- `mlb_pitcher_game_logs.csv`
- `mlb_team_game_logs.csv`
- `mlb_odds_part_1.csv`
- `mlb_odds_part_2.csv`
- `mlb_odds_part_3.csv`

Older `oddsData.csv` and `oddsDataMLB.csv` exports live under `data/legacy/` and are not read by the current model.

## Active code

The reusable backtester is under `mlb_lab/`. Standalone data/model scripts are under `python/`. `tests/` imports `mlb_lab.core` directly and also tests leakage-sensitive pitcher feature logic from `python.build_pitcher_model`.

## Model outputs

`python/build_pitcher_model.py` writes the baseball-model diagnostics and walk-forward results to `outputs/`. `python/evaluate_clean_market.py` validates historical American moneyline prices before producing the trusted market-comparison and ROI files.
