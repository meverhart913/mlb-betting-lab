# MLB Betting Lab

Leakage-resistant MLB moneyline research pipeline using historical game results, team box-score form, starting-pitcher game logs, sportsbook prices, and automatically captured pregame context.

## Repository layout

- `data/` canonical model inputs.
- `data/current/` automated current-day odds and weather/context snapshots.
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

## Automated morning-of-game operation

The operating decision time is morning-of-game. The normal workflow requires no daily data entry.

`.github/workflows/morning-mlb.yml` runs at 12:00 UTC (8:00 AM Eastern during the MLB daylight-saving season) from April through October and can also be started manually with `workflow_dispatch`.

The scheduled job automatically:

1. refreshes the previous four days of MLB schedule/results plus team and pitcher logs from MLB Stats API;
2. rebuilds the leakage-safe historical modeling table;
3. downloads current U.S. MLB moneylines from The Odds API;
4. retains every raw sportsbook quote in `data/current/morning_odds_raw.csv`;
5. creates one median-price consensus row per game in `data/current/morning_odds.csv`, while also recording the best available home and away prices/books;
6. captures venue coordinates, roof/turf metadata, game time, temperature, precipitation probability, precipitation, wind speed, and wind gust forecasts in `data/current/morning_context.csv` using MLB Stats API plus Open-Meteo;
7. pulls the day's MLB schedule and probable starters;
8. creates team and starter features using only games before the target date;
9. trains the current historical logistic model;
10. writes `outputs/morning_model_predictions.csv`;
11. uploads the predictions, raw/consensus odds, and context snapshot as a GitHub Actions artifact and writes a workflow job summary.

### One-time credential setup

The automated odds step requires a repository Actions secret named `THE_ODDS_API_KEY`. Current MLB odds are available on The Odds API free plan. This is a one-time credential setup, not a daily data-input step.

Local/manual execution remains available only as a fallback:

```bash
python python/refresh_recent_mlb.py
python python/build_pitcher_model.py
python python/fetch_morning_odds.py
python python/fetch_morning_context.py
python python/run_morning_model.py
```

The morning output includes market probability, model probability, estimated model-vs-market difference, probable-starter status, and a research signal. **Every row remains `NO BET` until a profitable decision rule is demonstrated out of sample.**

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

GitHub Actions runs the unit tests and historical research pipeline on pushes and pull requests. The morning workflow handles live daily operation separately.

## Next data priorities

Daily schedule, recent results, probable starters, pitcher/team logs, sportsbook moneylines, venue context, and weather forecasts are now automated. The next meaningful model work is to automate and historically validate bullpen workload/availability, confirmed/projected lineups, injuries/roster changes, and morning-to-close line movement. Manual data entry should be used only when no stable automated source exists.
