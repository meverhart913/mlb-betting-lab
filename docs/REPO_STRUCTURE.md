# Repository structure

- `data/` raw and derived CSV datasets used by the model.
- `python/` ingestion, conversion, feature engineering, modeling, and backtest scripts.
- `outputs/` generated model tables, predictions, metrics, and reports.
- `tests/` automated tests.
- `docs/` design notes and operating instructions.
- `archive/` obsolete snapshots or one-off artifacts that should not be part of the active pipeline.

## Recommended moves from repository root

Move these into `data/`:
- `mlb_games_2018_present.csv`
- `mlb_odds_part_1.csv`
- `mlb_odds_part_2.csv`
- `mlb_odds_part_3.csv`
- `oddsData.csv`
- `oddsDataMLB.csv`

Move these into `python/`:
- `download_mlb_enrichment.py`
- `convert_odds.py`
- `stat_import.py`
- `core.py`
- `__init__.py`
- `__main__.py`

Move `test_core.py` into `tests/`.
Move `archive.zip` and the empty `mlb-odds-scraper-main/` placeholder into `archive/` or remove them after verification.

The new `python/build_pitcher_model.py` expects the canonical datasets under `data/` and writes generated artifacts to `outputs/`.
