# MLB Betting Lab

Leakage-resistant MLB moneyline research pipeline using historical game results, team box-score form, starting-pitcher game logs, and sportsbook closing prices.

## Repository layout

- `data/` canonical model inputs.
- `data/legacy/` older odds exports retained for reference but not used by the current model.
- `mlb_lab/` reusable backtesting package and CLI.
- `python/` ingestion, conversion, feature engineering, and model scripts.
- `tests/` unit tests.
- `outputs/` generated diagnostics, predictions, walk-forward metrics, and betting simulations.
- `docs/` operating notes.
- `archive/` obsolete snapshots and placeholders.

## Current pitcher-aware model

Run:

```bash
pip install -r requirements.txt
python python/build_pitcher_model.py
```

The model builds pregame matchup differences from:

- starter rolling 3-, 5-, and 10-appearance workload/performance statistics;
- team rolling 10- and 30-game batting, pitching, and fielding form;
- no current-game box-score information.

Two models are tested independently with expanding-window season walk-forwards:

- regularized logistic regression;
- histogram gradient boosting.

Sportsbook closing moneylines are not used as model features. They are retained as an external benchmark and as the price for flat-$1 edge simulations.

## Leakage controls

Pitcher features are shifted one appearance and team features are shifted one game before rolling calculations. Unfinished/tied games are excluded from the target rather than silently labeled as losses. Tests cover both behaviors.

## Outputs

`python/build_pitcher_model.py` creates:

- `outputs/pitcher_model_data_diagnostics.csv`
- `outputs/pitcher_model_walkforward_metrics.csv`
- `outputs/pitcher_model_walkforward_predictions.csv`
- `outputs/pitcher_model_edge_backtest.csv`
- `outputs/pitcher_modeling_table.csv`

## Testing

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit tests and the full historical pitcher-aware backtest on pushes and pull requests.

## Data refresh

The current enrichment files are generated from MLB StatsAPI game feeds. The model is designed to be rerun after those inputs are refreshed. Generated outputs should be treated as research artifacts, not evidence of future profitability without out-of-sample validation.
