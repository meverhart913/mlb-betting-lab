# MLB Betting Lab

Leakage-resistant MLB moneyline research pipeline using historical game results, team box-score form, starting-pitcher game logs, and sportsbook prices.

## Repository layout

- `data/` canonical model inputs.
- `data/current/` current-day operating inputs such as morning sportsbook odds.
- `data/legacy/` older odds exports retained for reference but not used by the current model.
- `mlb_lab/` reusable backtesting package and CLI.
- `python/` ingestion, conversion, feature engineering, modeling, and operating scripts.
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

The model builds pregame matchup differences from starter rolling 3-, 5-, and 10-appearance workload/performance statistics and team rolling 10- and 30-game batting, pitching, and fielding form. No current-game box-score information is used.

Regularized logistic regression and histogram gradient boosting are tested independently with expanding-window season walk-forwards. Cleaned sportsbook closing moneylines are the benchmark rather than a trusted source of model profit.

## Morning-of-game operating point

The selected production decision time is morning-of-game.

1. Copy `data/current/morning_odds_template.csv` to `data/current/morning_odds.csv`.
2. Enter the morning moneyline for each game and the sportsbook/time captured.
3. Rebuild the historical modeling table if the game/team/pitcher logs were refreshed:

```bash
python python/build_pitcher_model.py
```

4. Run:

```bash
python python/run_morning_model.py
```

The runner pulls that day's MLB schedule and probable starters from MLB Stats API, creates team and starter features using only games before the target date, trains the current historical logistic model, joins the supplied morning market prices, and writes `outputs/morning_model_predictions.csv`.

The morning output includes market probability, model probability, estimated model-vs-market difference, probable-starter status, and a research signal. **Every row remains `NO BET` until a profitable decision rule is demonstrated out of sample.**

A different date can be evaluated with:

```bash
python python/run_morning_model.py --date YYYY-MM-DD
```

## Leakage controls

Pitcher features are shifted one appearance and team features are shifted one game before rolling calculations. Unfinished/tied games are excluded from the target rather than silently labeled as losses. Morning predictions explicitly select historical rows dated before the game date.

## Research findings so far

- Team form is more predictive than pitcher history by itself.
- Pitcher history adds a small incremental improvement to the baseball-only logistic model.
- The cleaned sportsbook closing market outperforms the baseball-only model on probability scoring.
- Adding the current baseball feature set on top of the market worsens average out-of-sample scoring.
- Team/starter rest features do not provide a stable improvement.
- No tested edge rule has shown sufficiently stable out-of-sample profitability to authorize betting.

## Testing

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit tests and historical research pipeline on pushes and pull requests.

## Data refresh

The enrichment files are generated from MLB Stats API game feeds. Generated outputs are research artifacts, not evidence of future profitability. The next meaningful model improvements require additional information known before the morning decision point, especially bullpen availability/workload, lineup information, injuries/roster changes, confirmed starter status, weather/park context, and historical morning line movement.
