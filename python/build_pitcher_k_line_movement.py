"""Summarize first-to-latest movement in pitcher strikeout prop consensus."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/"data"/"current"
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)


def implied(v):
    x=pd.to_numeric(v,errors="coerce")
    if pd.isna(x) or x==0:return np.nan
    return 100/(x+100) if x>0 else -x/(-x+100)


def legacy_novig(row):
    """Compatibility only for snapshots created before consensus-v2."""
    po=implied(row.get("over_price_median")); pu=implied(row.get("under_price_median"))
    return po/(po+pu) if pd.notna(po) and pd.notna(pu) and po+pu>0 else np.nan


def consensus_over(row):
    p=pd.to_numeric(row.get("market_over_prob_no_vig"),errors="coerce")
    if pd.notna(p) and 0.02 <= p <= 0.98:
        return float(p)
    # Do not trust legacy arithmetic-median American odds when they cross zero.
    op=pd.to_numeric(row.get("over_price_median"),errors="coerce")
    up=pd.to_numeric(row.get("under_price_median"),errors="coerce")
    if pd.notna(op) and pd.notna(up) and op*up < 0 and abs(op) < 100:
        return np.nan
    return legacy_novig(row)


def main():
    p=CURRENT/"pitcher_k_props_history.csv"
    if not p.exists():
        pd.DataFrame().to_csv(OUT/"pitcher_k_line_movement.csv",index=False); return
    h=pd.read_csv(p,low_memory=False)
    if h.empty:
        h.to_csv(OUT/"pitcher_k_line_movement.csv",index=False); return
    h["snapshot_time_et"]=pd.to_datetime(h["snapshot_time_et"],errors="coerce")
    h=h[h.snapshot_time_et.notna()].sort_values("snapshot_time_et")
    rows=[]
    keys=["date","event_id","pitcher_name","line"]
    for key,g in h.groupby(keys,dropna=False):
        first=g.iloc[0]; latest=g.iloc[-1]
        p0=consensus_over(first); p1=consensus_over(latest)
        rows.append({
            "date":key[0],"event_id":key[1],"pitcher_name":key[2],"line":key[3],
            "snapshots":len(g),"first_snapshot_et":first.snapshot_time_et,"latest_snapshot_et":latest.snapshot_time_et,
            "first_market_over_prob_no_vig":p0,"latest_market_over_prob_no_vig":p1,
            "market_over_prob_move":p1-p0 if pd.notna(p0) and pd.notna(p1) else np.nan,
            "first_best_over_price":first.get("best_over_price"),"latest_best_over_price":latest.get("best_over_price"),
            "first_best_under_price":first.get("best_under_price"),"latest_best_under_price":latest.get("best_under_price"),
        })
    out=pd.DataFrame(rows).sort_values(["date","event_id","pitcher_name","line"])
    out.to_csv(OUT/"pitcher_k_line_movement.csv",index=False)

if __name__=="__main__":main()
