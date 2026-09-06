from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path('data/current/fanduel_pitcher_k_paper_history.csv')
OUT_JSON = Path('outputs/fanduel_pitcher_k_ledger_reconciliation.json')
OUT_CSV = Path('outputs/fanduel_pitcher_k_ledger_reconciliation_issues.csv')


def american_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)


def main() -> None:
    if not LEDGER.exists() or LEDGER.stat().st_size == 0:
        raise SystemExit('Prospective ledger missing or empty')

    df = pd.read_csv(LEDGER, low_memory=False)
    issues: list[dict] = []

    def add(mask, code, detail):
        idx = df.index[mask.fillna(False)] if hasattr(mask, 'fillna') else df.index[mask]
        for i in idx:
            r = df.loc[i]
            issues.append({
                'row': int(i) + 2,
                'date': r.get('date'),
                'game_id': r.get('game_id'),
                'pitcher_id': r.get('pitcher_id'),
                'pitcher_name': r.get('pitcher_name'),
                'code': code,
                'detail': detail,
            })

    required = ['date','game_id','pitcher_id','pitcher_name','line','side','fanduel_price',
                'model_market_edge','expected_profit_per_unit','commence_time_utc','collected_at_utc',
                'timing_eligible','model_generated_at_et','paper_eligible','paper_status','result',
                'flat_profit_units','actual_k','actual_is_starter']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing required ledger columns: {missing}')

    # Immutable one-wager-per-independent-pitcher-start key.
    dup = df.duplicated(['date','game_id','pitcher_id'], keep=False)
    add(dup, 'DUPLICATE_PITCHER_START', 'More than one frozen wager for the same pitcher start')

    # All rows in the ledger must be frozen, timing eligible, and paper eligible.
    add(df['paper_status'].astype(str) != 'FROZEN_PAPER_SELECTION', 'BAD_PAPER_STATUS', 'Ledger row is not frozen')
    add(~as_bool(df['timing_eligible']), 'TIMING_NOT_ELIGIBLE', 'Frozen row has timing_eligible != True')
    add(~as_bool(df['paper_eligible']), 'PAPER_NOT_ELIGIBLE', 'Frozen row has paper_eligible != True')

    collected = pd.to_datetime(df['collected_at_utc'], utc=True, errors='coerce')
    commence = pd.to_datetime(df['commence_time_utc'], utc=True, errors='coerce')
    model_ts = pd.to_datetime(df['model_generated_at_et'], utc=True, errors='coerce')
    mins = (commence - collected).dt.total_seconds() / 60.0
    add(collected.isna() | commence.isna() | model_ts.isna(), 'BAD_TIMESTAMP', 'Required timestamp failed to parse')
    add(model_ts > collected, 'MODEL_AFTER_QUOTE', 'Model timestamp is after quote timestamp')
    add((mins < 45) | (mins > 195), 'OUTSIDE_DECISION_WINDOW', 'Frozen quote is outside the 45-195 minute window')
    if 'minutes_to_start' in df.columns:
        stored = pd.to_numeric(df['minutes_to_start'], errors='coerce')
        add((stored - mins).abs() > 0.2, 'MINUTES_TO_START_MISMATCH', 'Stored minutes_to_start disagrees with timestamps')

    # Threshold flags must be deterministic functions of frozen edge.
    edge = pd.to_numeric(df['model_market_edge'], errors='coerce')
    for col, threshold in [('edge_ge_0',0),('edge_ge_025',0.025),('edge_ge_05',0.05),('edge_ge_075',0.075),('edge_ge_10',0.10)]:
        if col in df.columns:
            expected = edge >= threshold
            actual = as_bool(df[col])
            add(actual != expected, 'THRESHOLD_FLAG_MISMATCH', f'{col} does not match edge >= {threshold}')

    # Selection EV must be positive/nonnegative-edge according to frozen protocol.
    ev = pd.to_numeric(df['expected_profit_per_unit'], errors='coerce')
    add(edge < -1e-12, 'NEGATIVE_EDGE_FROZEN', 'Frozen row has negative model-market edge')
    add(ev <= 0, 'NONPOSITIVE_EV_FROZEN', 'Frozen row has nonpositive expected profit')

    result = df['result'].fillna('PENDING').astype(str).str.upper()
    actual_k = pd.to_numeric(df['actual_k'], errors='coerce')
    line = pd.to_numeric(df['line'], errors='coerce')
    starter = pd.to_numeric(df['actual_is_starter'], errors='coerce')
    odds = pd.to_numeric(df['fanduel_price'], errors='coerce')
    profit = pd.to_numeric(df['flat_profit_units'], errors='coerce')
    side = df['side'].astype(str).str.upper()

    settled = result.isin(['WIN','LOSS','PUSH'])
    add(settled & actual_k.isna(), 'SETTLED_WITHOUT_ACTUAL_K', 'Settled row has no actual K')
    add(settled & (starter != 1), 'SETTLED_NONSTARTER', 'Settled row is not marked as starter')

    expected_result = pd.Series(index=df.index, dtype='object')
    over = side.eq('OVER')
    under = side.eq('UNDER')
    expected_result.loc[actual_k == line] = 'PUSH'
    expected_result.loc[over & (actual_k > line)] = 'WIN'
    expected_result.loc[over & (actual_k < line)] = 'LOSS'
    expected_result.loc[under & (actual_k < line)] = 'WIN'
    expected_result.loc[under & (actual_k > line)] = 'LOSS'
    add(settled & expected_result.notna() & (result != expected_result), 'RESULT_ACTUAL_K_MISMATCH', 'WIN/LOSS/PUSH disagrees with side, line, and actual K')

    expected_profit = pd.Series(np.nan, index=df.index)
    expected_profit.loc[result.eq('LOSS')] = -1.0
    expected_profit.loc[result.eq('PUSH')] = 0.0
    expected_profit.loc[result.eq('WIN')] = odds[result.eq('WIN')].map(american_profit)
    add(settled & ((profit - expected_profit).abs() > 1e-9), 'PROFIT_MISMATCH', 'flat_profit_units disagrees with frozen American odds/result')

    void = result.str.startswith('VOID')
    pending = result.eq('PENDING') | result.eq('')
    add(void & profit.notna() & (profit.abs() > 1e-12), 'VOID_NONZERO_PROFIT', 'Void row has nonzero profit')
    add(pending & profit.notna(), 'PENDING_HAS_PROFIT', 'Pending row already has flat profit')

    # CLV sanity: when same-line closing price is absent, no CLV value should be asserted.
    if 'closing_same_line_price' in df.columns and 'clv_implied_prob' in df.columns:
        closing = pd.to_numeric(df['closing_same_line_price'], errors='coerce')
        clv = pd.to_numeric(df['clv_implied_prob'], errors='coerce')
        add(closing.isna() & clv.notna(), 'CLV_WITHOUT_CLOSING_LINE', 'CLV exists without same-line closing price')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues).to_csv(OUT_CSV, index=False)
    summary = {
        'ledger_rows': int(len(df)),
        'unique_pitcher_starts': int(df[['date','game_id','pitcher_id']].drop_duplicates().shape[0]),
        'settled_rows': int(settled.sum()),
        'wins': int(result.eq('WIN').sum()),
        'losses': int(result.eq('LOSS').sum()),
        'pushes': int(result.eq('PUSH').sum()),
        'voids': int(void.sum()),
        'pending': int(pending.sum()),
        'flat_profit_units_settled': float(profit[settled].sum()),
        'critical_issue_count': int(len(issues)),
        'issue_codes': pd.Series([x['code'] for x in issues]).value_counts().to_dict() if issues else {},
        'status': 'PASS' if not issues else 'FAIL',
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    if issues:
        print(pd.DataFrame(issues).to_string(index=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
