# Pitcher K V2.1 research notes

Goal: test whether pitch-level Statcast-derived pitcher features and handedness-aware batter strikeout tendencies improve the existing pitcher-strikeout model without contaminating V1/V2 prospective histories.

Current V2.1 research components:

- Baseball Savant pitch-level backfill aggregated to pitcher-day rows.
- Leakage-safe rolling pitcher velocity, whiff, and pitch-mix features.
- Historical batter strikeout rates split by opposing pitcher handedness.
- Walk-forward challenger versus the current K feature set.
- Explicit promotion gate in `docs/pitcher_k_v21_promotion_gate.md`.

Research principle: every historical feature must be reproducible from information available strictly before the game being predicted.
