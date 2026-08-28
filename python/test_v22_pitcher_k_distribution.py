"""Audit Poisson vs negative-binomial pricing on V2.2 OOF pitcher-K projections."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from test_pitcher_k_distribution import poisson_cdf, nb_cdf, estimate_alpha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
PRED=OUT/'pitcher_k_batter_hand_predictions.csv'
LINES=(3.5,4.5,5.5,6.5,7.5,8.5,9.5)
TARGET_MODEL='v22_lineup_all'


def logloss(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def main():
    if not PRED.exists():
        raise SystemExit(f'Missing {PRED}; run V2.2 challenger first.')
    p=pd.read_csv(PRED,low_memory=False)
    if 'model' not in p.columns:
        raise ValueError('V2.2 prediction file is missing model column.')
    p=p[p.model.eq(TARGET_MODEL)].copy()
    p['date']=pd.to_datetime(p.date,errors='coerce')
    p['season']=pd.to_numeric(p.get('season',p.date.dt.year),errors='coerce')
    p['strikeouts']=pd.to_numeric(p.strikeouts,errors='coerce')
    p['projected_k']=pd.to_numeric(p.projected_k,errors='coerce')
    p=p.dropna(subset=['season','strikeouts','projected_k'])
    if p.empty:
        raise SystemExit(f'No {TARGET_MODEL} rows available.')

    rows=[]
    for year in sorted(p.season.astype(int).unique()):
        prior=p[p.season<year]; test=p[p.season==year]
        if len(prior)<300 or len(test)<100: continue
        alpha=estimate_alpha(prior.strikeouts,prior.projected_k)
        for line in LINES:
            cut=int(np.floor(line)); y=(test.strikeouts>line).astype(float).to_numpy()
            pp=np.array([1-poisson_cdf(cut,float(mu)) for mu in test.projected_k])
            pn=np.array([1-nb_cdf(cut,float(mu),alpha) for mu in test.projected_k])
            for dist,pred in [('poisson',pp),('negative_binomial',pn)]:
                rows.append({
                    'season':year,'model':TARGET_MODEL,'line':line,'alpha':alpha,'distribution':dist,
                    'starts':len(test),'brier':float(np.mean((pred-y)**2)),'log_loss':logloss(y,pred),
                    'actual_over_rate':float(y.mean()),'mean_pred_over':float(pred.mean()),
                    'calibration_bias':float(pred.mean()-y.mean()),
                })
    out=pd.DataFrame(rows)
    if out.empty:
        raise SystemExit('Insufficient V2.2 OOF seasons for distribution audit.')
    out.to_csv(OUT/'v22_pitcher_k_distribution_metrics.csv',index=False)
    summary=(out.groupby(['model','distribution'],as_index=False)
             .agg(mean_brier=('brier','mean'),mean_log_loss=('log_loss','mean'),mean_abs_calibration_bias=('calibration_bias',lambda x:float(np.mean(np.abs(x)))),
                  seasons=('season','nunique'),mean_alpha=('alpha','mean'))
             .sort_values(['mean_brier','mean_log_loss']))
    summary.to_csv(OUT/'v22_pitcher_k_distribution_summary.csv',index=False)
    best=summary.iloc[0]
    print(summary.round(6).to_string(index=False))
    print(f"Best V2.2 pricing distribution by mean Brier: {best.distribution}; Brier={best.mean_brier:.6f}; log loss={best.mean_log_loss:.6f}")

if __name__=='__main__': main()
