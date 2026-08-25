"""Prospective V2.2 lineup-handedness feature builder.

This file is intentionally NOT wired into the scheduled workflow yet. It is a gated
integration target for the historical V2.2 research branch. Promotion requires the
historical walk-forward challenger to beat Statcast-all with >=90% lineup-row coverage
and >=80% mean batter-match coverage.

Inputs expected after V2.2 research promotion:
  data/current/pitcher_k_v2_lineup_context.csv
  data/features/batter_k_by_pitcher_hand_pregame.csv

Output:
  data/current/pitcher_k_v22_lineup_hand_context.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'data/current/pitcher_k_v2_lineup_context.csv'
HIST=ROOT/'data/features/batter_k_by_pitcher_hand_pregame.csv'
OUT=ROOT/'data/current/pitcher_k_v22_lineup_hand_context.csv'


def latest_prior(hist: pd.DataFrame,batter_id:int,hand:str,day:pd.Timestamp):
    z=hist[(hist.batter_id==batter_id)&(hist.pitcher_hand==hand)&(hist.game_date<day)]
    if z.empty: return None
    return z.sort_values('game_date').iloc[-1]


def main():
    if not CTX.exists(): raise SystemExit(f'Missing {CTX}')
    if not HIST.exists(): raise SystemExit(f'Missing {HIST}; V2.2 historical feature pipeline has not been promoted')
    c=pd.read_csv(CTX,low_memory=False); h=pd.read_csv(HIST,low_memory=False)
    if c.empty:
        pd.DataFrame().to_csv(OUT,index=False); print('No posted pregame lineups to score'); return
    c['batter_id']=pd.to_numeric(c.batter_id,errors='coerce'); c['pitcher_id']=pd.to_numeric(c.pitcher_id,errors='coerce'); c['date']=pd.to_datetime(c.date,errors='coerce')
    h['batter_id']=pd.to_numeric(h.batter_id,errors='coerce'); h['game_date']=pd.to_datetime(h.game_date,errors='coerce'); h['pitcher_hand']=h.pitcher_hand.astype(str).str.upper().str[0]
    for col in ['batter_k_pa_30d','batter_k_pa_90d','batter_k_pa_365d','batter_pa_90d']:
        h[col]=pd.to_numeric(h[col],errors='coerce')
    rows=[]
    for (game,pitcher),g in c.groupby(['game_id','pitcher_id'],dropna=False):
        hand=str(g.pitcher_hand.iloc[0]).upper()[:1]; day=pd.Timestamp(g.date.iloc[0]); vals=[]
        for r in g.sort_values('batting_order').head(9).itertuples(index=False):
            q=latest_prior(h,int(r.batter_id),hand,day) if pd.notna(r.batter_id) and hand in {'L','R'} else None
            if q is None: continue
            pa90=float(q.batter_pa_90d) if pd.notna(q.batter_pa_90d) else 0.0
            k90=float(q.batter_k_pa_90d) if pd.notna(q.batter_k_pa_90d) else np.nan
            k365=float(q.batter_k_pa_365d) if pd.notna(q.batter_k_pa_365d) else np.nan
            rel=np.clip(pa90/(pa90+40.0),0,1)
            blend=(rel*k90+(1-rel)*k365) if pd.notna(k90) and pd.notna(k365) else (k90 if pd.notna(k90) else k365)
            vals.append({'batting_order':getattr(r,'batting_order',np.nan),'batter_id':int(r.batter_id),'k_blend':blend,'pa90':pa90})
        v=pd.DataFrame(vals); matched=len(v); total=min(9,g.batter_id.nunique())
        rows.append({'date':day.date().isoformat(),'game_id':game,'pitcher_id':pitcher,'pitcher_hand':hand,'lineup_batters':total,'matched_batters':matched,'lineup_match_coverage':matched/max(total,1),'lineup_k_hand_blend_mean':float(v.k_blend.mean()) if matched else np.nan,'lineup_k_hand_top4':float(v.sort_values('batting_order').head(4).k_blend.mean()) if matched else np.nan,'lineup_pa90_mean':float(v.pa90.mean()) if matched else np.nan,'v22_feature_status':'eligible' if total>=9 and matched>=8 else 'insufficient_coverage'})
    out=pd.DataFrame(rows); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False); print(f'Built V2.2 live handedness features for {len(out)} pitcher-lineup pairs; eligible={(out.v22_feature_status=="eligible").sum() if len(out) else 0}')

if __name__=='__main__': main()
