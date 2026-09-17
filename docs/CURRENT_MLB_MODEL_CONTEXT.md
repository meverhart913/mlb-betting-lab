# Current MLB Model Canonical Context

**Canonical status date:** 2026-09-17 19:25 ET  
**Repository:** `meverhart913/mlb-betting-lab`  
**Primary live research lane:** FanDuel MLB pitcher strikeouts  
**Mode:** Prospective paper validation / maintenance only  
**Real wagering:** NOT AUTHORIZED

This file is the current handoff/source-of-truth summary for the MLB pitcher-K model. Read it before making model, workflow, grading, or promotion decisions. The frozen prospective protocol remains authoritative for rules that must not be changed retroactively: `docs/FANDUEL_PROSPECTIVE_PROTOCOL.md`.

## 1. Current model

- Primary model: **V2.2 lineup-handedness + Statcast** when a confirmed lineup is available.
- Fallback: **V2.1 Statcast-only** when the lineup is not confirmed.
- Primary sportsbook: **FanDuel**.
- Unit of evaluation: maximum **one frozen wager per independent pitcher start**.
- Timing window: **45-195 minutes before first pitch**.
- Selection uses only information available at or before the quoted FanDuel snapshot.
- Frozen threshold ladder: edge >= 0%, 2.5%, 5%, 7.5%, 10%.
- The 5% threshold is a research candidate, **not** a promoted betting rule.

## 2. Current prospective evidence

Current ledger: `data/current/fanduel_pitcher_k_paper_history.csv`

As of the current repository state:

| Minimum model edge | Bets | W-L | Flat units | ROI |
| --- | ---: | ---: | ---: | ---: |
| 0% | 115 | 35-80 | -24.8605 | -21.62% |
| 2.5% | 99 | 35-64 | -8.8605 | -8.95% |
| 5% | 64 | 23-41 | -4.1541 | -6.49% |
| 7.5% | 41 | 15-26 | -5.6583 | -13.80% |
| 10% | 29 | 11-18 | -8.7877 | -30.30% |

Additional integrity status:

- **115 total frozen pitcher starts**
- **115 unique pitcher starts**
- **115 settled**
- **0 pending**
- **0 pushes**
- Current ledger ends with selections dated **2026-09-12**.
- The 100-start validation gate has been crossed, but the gate does **not** authorize promotion by sample size alone.
- Current ROI evidence is negative at every frozen edge threshold. The model therefore **fails the current promotion condition** and remains paper-only.
- Earlier positive small-sample checkpoints are superseded by this larger prospective sample. Do not quote the earlier 67-bet positive checkpoint as the current performance state.

## 3. Active blocker: FanDuel market capture is empty

The primary FanDuel workflow is completing successfully, but it has not frozen a new selection after 2026-09-12 because the live FanDuel market file is empty.

Every stored decision-cycle audit from **2026-09-13 through 2026-09-17** currently reports:

- `status: NO_PAPER`
- `no_paper_reason: EMPTY_FANDUEL_MARKET_FILE`

The model is still generating projections (roughly 14-30 projection rows in the audited cycles), so this is **not equivalent to a healthy NO BET decision**. It is a **market-data capture blocker**.

### 2026-09-17 pull status

- FanDuel paper run **#91** started 2026-09-17 14:34 ET and completed successfully.
  - Audit: 17 projections.
  - Result: `EMPTY_FANDUEL_MARKET_FILE`.
- FanDuel paper run **#92** started 2026-09-17 19:06 ET and completed successfully.
  - All workflow steps completed, including V2.2 artifact download, recent-log refresh, projection generation, market-pull step, selection, grading, and history commit.
  - Audit recorded at 19:08 ET: 18 projections.
  - Result: `EMPTY_FANDUEL_MARKET_FILE`.
- No valid new FanDuel market rows or frozen starts were added by either run.
- The configured schedule remains 10:45 AM, 1:45 PM, 4:45 PM, and 7:45 PM ET during daylight time, but GitHub scheduled execution has continued to be delayed/irregular. Do not infer exact capture time from the cron slot; use the cycle audit timestamp.

### Market-source logic that needs investigation

The live workflow currently prefers sources in this order:

1. If `PROPLINE_API_KEY` is configured, use `python/fetch_propline_live_fanduel.py`.
2. Otherwise, if `THE_ODDS_API_KEY` is configured, use `python/fetch_oddsapi_live_fanduel.py`.
3. Otherwise, create an empty market file.

Because the job is green while the resulting FanDuel file is empty, determine which source is active and why it has produced no FanDuel pitcher-K rows since September 13. Do not treat these empty-market cycles as evidence that the model found no edge.

## 4. Grading and ledger integrity

The most recent FanDuel grading run checked during this update was **run #31** on 2026-09-17 at approximately 14:09 ET and completed successfully.

The permanent reconciliation tooling is:

- `python/grade_fanduel_pitcher_k_paper.py`
- `python/reconcile_fanduel_paper_ledger.py`
- `.github/workflows/fanduel-pitcher-k-grade.yml`

The reconciled ledger currently has one row per independent pitcher start and all 115 rows are settled.

Authoritative grading rules remain:

- Use official completed-game pitcher strikeouts.
- Do not grade from live/in-progress partial logs.
- Starter changes/scratches are voids.
- Preserve losses, pushes, and voids.
- Never backfill a missed prospective capture after the game and count it as prospective evidence.

## 5. Artifact dependency status

The earlier missing-V2.2-artifact failure was repaired. The primary FanDuel run #92 successfully completed the step:

`Download V2.2 research and freshest live feature data`

Therefore the old expired-artifact defect is **not the current blocker**.

The primary FanDuel path can currently restore its V2.2 research inputs and build live projections.

## 6. Secondary issue: Pitcher K V2 lineups workflow

The separate workflow **Pitcher K V2 lineups** is currently unhealthy.

Latest checked run:

- Run **#25**, 2026-09-17 ~19:16 ET: **FAILED**
- Failure step: `Restore noon prop snapshot without API call`
- Earlier runs #21-24 also failed.

This workflow is a secondary research/lineup-comparison lane. Its failure did **not** block FanDuel paper run #92, which completed its own V2.2 artifact restore and projection path successfully. Still, the repeated failure should be repaired so the research comparison remains current.

## 7. Current interpretation

The correct current status sentence is:

> **The MLB pitcher-K pipeline has 115 clean settled prospective starts, but the larger sample is currently unprofitable and new validation is blocked by an empty FanDuel market feed. Profitability is not established, no threshold is promoted, and real wagering remains unauthorized.**

Do not describe the model as validated or profitable.

Do not redesign V2.2 merely because the current sample is losing. Preserve the frozen protocol and separate:
- pipeline/data defects,
- prospective validation evidence,
- future V2.3 research.

Any model or threshold change must be versioned prospectively from its change date.

## 8. Priority next actions

### P0 — Restore FanDuel market capture
1. Identify whether the live run is using PropLine or The Odds API.
2. Inspect the raw response/normalized output that produces the empty FanDuel file.
3. Verify FanDuel `pitcher_strikeouts` / alternate market availability and parser compatibility.
4. Distinguish **DATA_SOURCE_FAILURE** from a legitimate **NO_PAPER/NO_BET** state in the cycle audit.
5. Add a guard/alert so repeated empty boards cannot remain silently green for multiple days.
6. Resume prospective accumulation only after real timestamped FanDuel rows are being archived again.

### P1 — Repair V2 lineups workflow
Fix `Restore noon prop snapshot without API call` in `Pitcher K V2 lineups`.

### P2 — Continue frozen validation
After market capture is restored:
- continue to 200+ independent starts without outcome-driven threshold changes;
- monitor ROI, calibration, CLV, projection error, and failure regimes;
- keep all activity paper-only unless the frozen promotion requirements are satisfied.

## 9. Key source-of-truth files

- `docs/CURRENT_MLB_MODEL_CONTEXT.md` — current canonical status/handoff
- `docs/FANDUEL_PROSPECTIVE_PROTOCOL.md` — frozen prospective rules
- `data/current/fanduel_pitcher_k_paper_history.csv` — prospective ledger
- `data/market/free_archive/YYYY-MM-DD/cycle-audit-*.json` — immutable decision-cycle evidence
- `.github/workflows/fanduel-pitcher-k-paper.yml` — primary prospective capture workflow
- `.github/workflows/fanduel-pitcher-k-grade.yml` — grading/reconciliation workflow
- `python/select_fanduel_paper_from_live.py` — prospective selection logic
- `python/grade_fanduel_pitcher_k_paper.py` — grader
- `python/reconcile_fanduel_paper_ledger.py` — ledger integrity audit
- `python/fetch_propline_live_fanduel.py` — preferred live source when configured
- `python/fetch_oddsapi_live_fanduel.py` — credit-aware Odds API fallback

## 10. Latest verification

Verified against GitHub Actions at **2026-09-17 19:25 ET**. The latest primary FanDuel workflow is **run #92** (`schedule`), started at 19:06 ET and completed successfully at 19:08 ET. The prior primary run **#91** also completed successfully. The live V2.2 feature refresh **#22** and free-market grading **#29** completed successfully today. Despite green workflow execution, the active P0 remains the empty FanDuel market feed described above; successful workflow status must not be interpreted as successful market capture.

## 11. Handoff rule

At the start of any future MLB-model conversation:

1. Read this file.
2. Read the frozen prospective protocol before proposing model/threshold changes.
3. Refresh GitHub workflow history and the current ledger before quoting performance.
4. Treat cycle-audit timestamps—not nominal cron slots—as the actual capture times.
5. Never combine retrospective diagnostics with prospective performance.
6. Never call missed/backfilled selections prospective.
