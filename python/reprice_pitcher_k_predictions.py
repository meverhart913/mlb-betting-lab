"""Replace market probabilities/edges with validated no-vig consensus values.

This protects downstream grading from legacy arithmetic-median American odds.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/"data"/"current"
OUT=ROOT/"outputs"


def main():
    pred_path=OUT/"pitcher_k_prop_predictions.csv"
    props_path=CURRENT/"pitcher_k_props.csv"
    if not pred_path.exists() or not props_path.exists():
        raise SystemExit("Missing pitcher K predictions or consensus props.")
    pred=pd.read_csv(pred_path,low_memory=False)
    props=pd.read_csv(props_path,low_memory=False)
    required={"event_id","pitcher_name","line","market_over_prob_no_vig","market_under_prob_no_vig"}
    missing=required-set(props.columns)
    if missing:
        raise ValueError("Consensus prop file missing: "+", ".join(sorted(missing)))
    key=["event_id","pitcher_name","line"]
    keep=key+["market_over_prob_no_vig","market_under_prob_no_vig","best_over_price","best_over_sportsbook","best_under_price","best_under_sportsbook"]
    p=props[keep].drop_duplicates(key,keep="last")
    drop=[c for c in keep if c not in key and c in pred.columns]
    pred=pred.drop(columns=drop).merge(p,on=key,how="left")
    mo=pd.to_numeric(pred["market_over_prob_no_vig"],errors="coerce")
    mu=pd.to_numeric(pred["market_under_prob_no_vig"],errors="coerce")
    bad=(mo.notna() & ~mo.between(0.02,0.98)) | (mu.notna() & ~mu.between(0.02,0.98))
    if bad.any():
        sample=pred.loc[bad,key+["market_over_prob_no_vig","market_under_prob_no_vig"]].head().to_dict("records")
        raise ValueError(f"Invalid consensus probabilities: {sample}")
    pred["over_edge"]=pd.to_numeric(pred["fair_over_prob"],errors="coerce")-mo
    pred["under_edge"]=pd.to_numeric(pred["fair_under_prob"],errors="coerce")-mu
    pred["research_side"]=np.where(pred["over_edge"]>=pred["under_edge"],"OVER","UNDER")
    pred["model_market_edge"]=pred[["over_edge","under_edge"]].max(axis=1)
    pred["decision"]="NO BET - prospective prop edge not validated"
    pred=pred.sort_values("model_market_edge",ascending=False)
    pred.to_csv(pred_path,index=False)
    print(pred[["pitcher_name","line","projected_k","market_over_prob_no_vig","research_side","model_market_edge"]].head(20).round(4).to_string(index=False))

if __name__=="__main__":main()
