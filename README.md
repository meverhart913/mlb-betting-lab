# MLB Betting Lab

Leakage-resistant MLB moneyline research and automated morning-of-game operating pipeline.

## Repository layout

- `data/` canonical historical model inputs.
- `data/current/` automated current-day odds and pregame context snapshots.
- `data/legacy/` older odds exports retained for provenance/reference.
- `mlb_lab/` reusable backtesting package and CLI.
- `python/` ingestion, feature engineering, modeling, research, and operating scripts.
- `tests/` unit/leakage tests.
- `outputs/` generated diagnostics, predictions, comparisons, and reports.
- `docs/` operating/design notes.
- `archive/` obsolete snapshots and placeholders.

## Current production research model

The current baseball-only probability model uses leakage-safe differences in:

- starting-pitcher rolling 3-, 5-, and 10-appearance workload/performance statistics;
- team rolling 10- and 30-game batting, pitching, and fielding form.

Regularized logistic regression and histogram gradient boosting are evaluated with expanding-season walk-forward tests. Sportsbook closing prices remain the external benchmark. Every live row remains `NO BET` until an out-of-sample profitable decision rule is demonstrated.

## Automated morning operation

The selected operating decision time is morning-of-game. `.github/workflows/morning-mlb.yml` runs at 12:00 UTC (8:00 AM Eastern during the MLB daylight-saving season) from April through October and can also be dispatched manually.

On game days it automatically:

1. checks that a regular-season slate exists before using quota-limited services;
2. refreshes recent MLB results, team logs, pitcher logs, and probable starters;
3. rebuilds the leakage-safe historical modeling table;
4. pulls U.S. MLB moneylines from The Odds API;
5. stores raw sportsbook quotes plus a median consensus and separately records best available prices/books;
6. captures bullpen workload/availability proxies from recent reliever appearances;
7. captures active-roster and conservatively reconstructed injured-list context from MLB transactions;
8. captures any batting order actually posted in the MLB game feed and explicitly labels lineups not yet posted;
9. captures venue, roof/turf, game time, and weather forecast context using MLB Stats API plus Open-Meteo;
10. scores the slate with the currently authorized historical feature set;
11. assembles `outputs/morning_report.csv` with model, market, best-price, weather, bullpen, roster/IL, and lineup context;
12. uploads the complete morning artifact bundle.

If there are no games, the workflow exits cleanly and does not spend an Odds API request.

### Odds history and line movement

The morning odds pull appends raw and consensus snapshots to a rolling GitHub Actions cache so history survives ephemeral runners. `.github/workflows/afternoon-odds.yml` adds a second quota-conscious snapshot at 2:00 PM Eastern on game days.

`python/build_line_movement.py` separates:

- first-seen-to-latest movement; and
- game-day morning-to-latest movement, matching the selected morning decision point.

This dataset is prospective. Line movement will not enter the model until enough snapshots exist for leakage-safe validation.

### One-time credential

The odds workflows require the repository Actions secret `THE_ODDS_API_KEY`. No daily manual odds entry is required.

## Feature promotion discipline

Automated capture does not imply model inclusion. New context is first captured/audited, then historically or prospectively tested, and only then considered for production features.

Current findings:

- Team form is more predictive than pitcher history by itself.
- Pitcher history adds a small incremental improvement to the baseball-only logistic model.
- The cleaned sportsbook closing market still outperforms the baseball-only model on probability scoring.
- Adding the current baseball feature set on top of the market worsens average out-of-sample scoring.
- Team/starter rest features do not provide stable improvement.
- The full bullpen workload bundle also fails promotion: mean walk-forward log loss worsened from about `0.68153` to `0.68191`, with AUC also declining. Bullpen therefore remains context-only while individual-feature ablation is tested.
- No tested betting-edge rule has shown sufficiently stable out-of-sample profitability to authorize betting.

## Testing

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python python/build_pitcher_model.py
python python/test_bullpen_features.py
python python/test_bullpen_ablation.py
```

GitHub Actions runs the complete historical research suite on pull requests and relevant pushes.

## Local fallback

Manual/local execution remains available for troubleshooting, not normal operation:

```bash
python python/refresh_recent_mlb.py
python python/build_pitcher_model.py
python python/fetch_morning_odds.py
python python/fetch_bullpen_context.py
python python/fetch_roster_context.py
python python/fetch_lineup_context.py
python python/fetch_morning_context.py
python python/run_morning_model.py
python python/build_morning_report.py
```

The guiding rule is automation first; manual data entry is fallback-only when no stable automated or derivable source exists.
