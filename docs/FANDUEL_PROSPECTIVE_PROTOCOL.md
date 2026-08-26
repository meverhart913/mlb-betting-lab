# FanDuel Pitcher-K Prospective Protocol

Frozen: 2026-08-26

Purpose: prevent outcome-driven rule changes while prospective validation accumulates.

## Primary market
- FanDuel pitcher strikeouts only.
- Preserve the main line and every available alternate strikeout line with quote timestamp and American price.
- Other regulated books are diagnostic/control data only and do not determine the primary result.
- PrizePicks, Kalshi, pick'em products, and exchanges are excluded from sportsbook ROI.

## Unit of evaluation
- Maximum one selected wager per pitcher start.
- All FanDuel lines and both sides may be evaluated, but only the single highest expected-value eligible wager is the headline selection.
- Multiple alt lines for one pitcher are correlated candidates, not independent bets.

## Frozen threshold ladder
Track all of these prospectively without selecting the winner after outcomes are known:
- edge >= 0%
- edge >= 2.5%
- edge >= 5%
- edge >= 7.5%
- edge >= 10%

The 5% threshold is a research candidate, not a promoted live-betting rule.

## Timing integrity
- A recommendation may use only information timestamped at or before the FanDuel quote snapshot used for that recommendation.
- Never pair a later confirmed lineup, injury update, result, or model input with an earlier FanDuel price.
- Preserve quote timestamp and model-generation timestamp.
- Pitcher scratches or starter changes void the research recommendation rather than being silently removed.

## Selection and grading
- Calculate model probability for OVER and UNDER for every available FanDuel line.
- Calculate expected value using the actual FanDuel American price.
- Select the eligible side/line with highest expected value for that pitcher.
- Grade WIN/LOSS/PUSH from official completed-game pitcher strikeouts.
- Preserve every frozen recommendation, including losses and voids.

## Validation gates
- 50 independent pitcher starts: preliminary diagnostic checkpoint only.
- 100 independent starts: first point at which tiny live stakes may be considered if calibration, CLV, ROI, and timing-integrity checks all survive.
- 200+ independent starts: first meaningful promotion decision.
- No threshold/model promotion based only on the initial 20-start sample.

## Bankroll rule before promotion
- Paper units only by default.
- No Martingale, loss chasing, or outcome-dependent stake changes.
- If later promoted to live testing, initial flat stake target is 0.25%-0.50% of dedicated bankroll per qualifying wager; no Kelly sizing until probability calibration is demonstrated.

## Required reporting
Track at minimum:
- independent bets
- W/L/P
- flat profit units and ROI
- FanDuel price and line at recommendation
- later/closing FanDuel price and line when available
- closing-line value (CLV)
- model probability calibration
- model edge and expected value
- projection error
- performance by uncertainty/failure regime

Any future change to this protocol must be versioned and evaluated prospectively from the change date rather than retroactively applied to make historical results look better.
