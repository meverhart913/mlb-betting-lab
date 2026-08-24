"""Compare Poisson vs negative-binomial K probability calibration on OOF starts."""
from __future__ import annotations

from math import exp
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
LINES=(3.5,4.5,5.5,6.5,7.5,8.5,9.5)


def poisson_cdf(k:int,mu:float)->float:
    if k<0:return 0.0
    term=exp(-mu); total=term
    for i in range(1,k+1):
        term*=mu/i; total+=term
    return min(max(total,0.0),1.0)


def nb_cdf(k:int,mu:float,alpha:float)->float:
    if alpha<=1e-9:return poisson_cdf(k,mu)
    if k<0:return 0.0
    r=1.0/alpha; p=r/(r+mu)
    term=p**r; total=term
    for i in range(1,k+1):
        term*=((i-1+r)/i)*(1-p); total+=term
    return min(max(total,0.0),1.0)


def estimate_alpha(y,mu)->float:
    y=np.asarray(y,float); mu=np.asarray(mu,float)
    mse=np.mean((y-mu)**2)
    denom=np.mean(mu**2)
    if denom<=0:return 0.0
    return float(np.clip((mse-np.mean(mu))/denom,0.0,2.0))


def main():
    p=pd.read_csv(OUT/"pitcher_k_walkforward_predictions.csv",low_memory=False)
    p["date"]=pd.to_datetime(p["date"],errors="coerce"); p["season"]=p["date"].dt.year
    rows=[]
    for model,g in p.groupby("model"):
        years=sorted(g.season.dropna().astype(int).unique())
        for year in years:
            prior=g[g.season<year]; test=g[g.season==year]
            if len(prior)<300 or len(test)<100:continue
            alpha=estimate_alpha(prior.strikeouts,prior.projected_k)
            for line in LINES:
                cutoff=int(np.floor(line))
                actual=(test.strikeouts>line).astype(float).to_numpy()
                pp=np.array([1-poisson_cdf(cutoff,float(mu)) for mu in test.projected_k])
                pn=np.array([1-nb_cdf(cutoff,float(mu),alpha) for mu in test.projected_k])
                rows.append({"season":year,"model":model,"line":line,"alpha":alpha,"distribution":"poisson","brier":float(np.mean((pp-actual)**2)),"actual_over_rate":float(actual.mean()),"mean_pred_over":float(pp.mean())})
                rows.append({"season":year,"model":model,"line":line,"alpha":alpha,"distribution":"negative_binomial","brier":float(np.mean((pn-actual)**2)),"actual_over_rate":float(actual.mean()),"mean_pred_over":float(pn.mean())})
    out=pd.DataFrame(rows); out.to_csv(OUT/"pitcher_k_distribution_metrics.csv",index=False)
    summary=out.groupby(["model","distribution"],as_index=False).agg(mean_brier=("brier","mean"),seasons=("season","nunique"),mean_alpha=("alpha","mean")).sort_values("mean_brier")
    summary.to_csv(OUT/"pitcher_k_distribution_summary.csv",index=False)
    print(summary.round(6).to_string(index=False))

if __name__=="__main__":main()
