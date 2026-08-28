"""Summarize settled FanDuel paper performance by frozen diagnostic regime."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
GRADED=ROOT/'outputs/fanduel_pitcher_k_paper_graded.csv'
OUT=ROOT/'outputs/fanduel_pitcher_k_failure_regime_summary.csv'


def binary_scores(x):
    q=x[x.result.isin(['WIN','LOSS']) & pd.to_numeric(x.model_win_prob,errors='coerce').notna()].copy()
    if q.empty:return np.nan,np.nan
    y=q.result.eq('WIN').astype(float).to_numpy(); p=np.clip(pd.to_numeric(q.model_win_prob,errors='coerce').to_numpy(float),1e-6,1-1e-6)
    return float(np.mean((p-y)**2)),float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def main():
    if not GRADED.exists() or GRADED.stat().st_size==0:
        pd.DataFrame().to_csv(OUT,index=False); print('No graded FanDuel paper data yet.'); return
    try:z=pd.read_csv(GRADED,low_memory=False)
    except pd.errors.EmptyDataError:
        pd.DataFrame().to_csv(OUT,index=False); return
    z=z[z.result.isin(['WIN','LOSS','PUSH'])].copy()
    if z.empty or 'failure_regime_flags' not in z.columns:
        pd.DataFrame().to_csv(OUT,index=False); print('No settled failure-regime data yet.'); return
    expanded=[]
    for _,r in z.iterrows():
        raw=str(r.get('failure_regime_flags','NONE'))
        flags=[f for f in raw.split(',') if f and f!='nan'] or ['NONE']
        if raw=='NONE':flags=['NONE']
        for f in flags:
            q=r.copy(); q['failure_regime']=f; expanded.append(q)
    e=pd.DataFrame(expanded)
    rows=[]
    for flag,x in e.groupby('failure_regime',dropna=False):
        wins=int(x.result.eq('WIN').sum()); losses=int(x.result.eq('LOSS').sum()); pushes=int(x.result.eq('PUSH').sum()); staked=wins+losses
        profit=float(pd.to_numeric(x.get('flat_profit_units'),errors='coerce').fillna(0).sum()); brier,ll=binary_scores(x)
        rows.append({'failure_regime':flag,'independent_bets':staked,'pushes':pushes,'wins':wins,'losses':losses,
                     'win_rate_ex_push':wins/staked if staked else np.nan,'profit_units':profit,'roi':profit/staked if staked else np.nan,
                     'brier_score_ex_push':brier,'log_loss_ex_push':ll,
                     'mean_edge':float(pd.to_numeric(x.get('model_market_edge'),errors='coerce').mean()),
                     'mean_clv_implied_prob':float(pd.to_numeric(x.get('clv_implied_prob'),errors='coerce').mean())})
    out=pd.DataFrame(rows).sort_values(['independent_bets','failure_regime'],ascending=[False,True])
    out.to_csv(OUT,index=False); print(out.round(5).to_string(index=False))

if __name__=='__main__':main()
