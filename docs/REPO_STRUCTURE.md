# Repository structure

- `data/` canonical raw and derived datasets used by the model.
- `data/current/` machine-generated morning odds and venue/weather snapshots; manual templates are fallback-only.
- `data/legacy/` older odds exports retained for reference and data-quality investigation.
- `python/` ingestion, conversion, feature engineering, model, backtest, research-comparison, and daily operating scripts.
- `mlb_lab/` reusable package code and CLI.
- `outputs/` generated model tables, predictions, metrics, reports, and research comparisons.
- `tests/` automated unit/leakage tests.
- `docs/` design and operating notes.
- `archive/` obsolete snapshots and one-off artifacts.

## Automated morning workflow

`.github/workflows/morning-mlb.yml` is the normal operating path. During MLB season it runs each morning, incrementally refreshes recent MLB results/team/pitcher data, rebuilds the model table, pulls current U.S. moneylines, captures venue/weather context, scores the current slate, and uploads the morning artifacts.

The odds pull preserves both raw sportsbook quotes and a one-row-per-game median consensus with separately recorded best available home/away prices. This avoids contaminating fair-probability estimation with an impossible best-of-both-books synthetic market.

Manual CSV entry is retained only as a fallback if an automated provider is unavailable.
