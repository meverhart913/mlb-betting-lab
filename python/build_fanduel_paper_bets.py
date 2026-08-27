"""Build frozen FanDuel pitcher-K paper selections from V2.2 projections.

Consumes the current/free normalized market snapshot plus pitcher projections.
Every FanDuel alt line and side is priced at its actual American odds. Exactly
one highest-EV candidate per pitcher/snapshot is selected. Research only.
"""
from __future__ import annotations
from datetime import datetime
from math import exp, floor
from pathlib import Path
from zoneinfo import ZoneInfo
import re, unicodedata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MARKET=ROOT/'data/market/pitcher_k_historical_raw.csv'
PRED=ROOT/'outputs/pitcher_k_prop_predictions.csv'
OUT=ROOT/'outputs'
CURRENT=ROOT/'data/current'
HISTORY=CURRENT/'fanduel_paper_bets_history.csv'


def norm(s):
    x=unicodedata.normalize('NFKD',str(s or '')).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]','',x)

def implied(odds):
    x=float(odds)
    return 100/(x+100) if x>0 else -x/(-x+100)

def decimal(odds):
    x=float(odds)
    return 1+x/100 if x>0 else 1+100/(-x)

def cdf(k,mu):
    if k<0:return 0.0
    term=exp(-mu); total=term
    for i in range(1,k+1):
        term*=mu/i; total+=term
    return min(max(total,0),1)

def side_prob(mu,line,side):
    if abs(line-round(line))<1e-9:
        k=int(round(line))
        pu=cdf(k-1,mu); po=1-cdf(k,mu)
    else:
        pu=cdf(floor(line),mu); po=1-pu
    return po if side=='over' else pu

def main():
    if not MARKET.exists() or not PRED.exists():
        raise SystemExit('Missing market snapshot or projection output')
    m=pd.read_csv(MARKET,low_memory=False)
    p=pd.read_csv(PRED,low_memory=False)
    if m.empty or p.empty:
        print('No market/projection rows; no paper selections.')
        return
    m=m[m.sportsbook.astype(str).str.casefold().eq('fanduel')].copy()
    m['name_key']=m.pitcher_name.map(norm); p['name_key']=p.pitcher_name.map(norm)
    # One projection per pitcher. The live projection is independent of alt line.
    p=p.sort_values('model_market_edge',ascending=False).drop_duplicates('name_key')
    keep=['name_key','pitcher_id','pitcher_name','projected_k']
    z=m.merge(p[keep].rename(columns={'pitcher_name':'model_pitcher_name'}),on='name_key',how='inner')
    rows=[]
    frozen=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
    for r in z.itertuples(index=False):
        try:
            mu=float(r.projected_k); line=float(r.line); odds=float(r.price); side=str(r.side).lower()
            if side not in {'over','under'} or odds==0: continue
            prob=side_prob(mu,line,side); imp=implied(odds); dec=decimal(odds)
            ev=prob*(dec-1)-(1-prob)
            rows.append({'date':r.date,'event_id':r.event_id,'pitcher_id':r.pitcher_id,
                'pitcher_name':r.model_pitcher_name,'away_team':getattr(r,'away_team',None),
                'home_team':getattr(r,'home_team',None),'snapshot_time_et':getattr(r,'snapshot_time_et',None),
                'frozen_at_et':frozen,'sportsbook':'FanDuel','line':line,'side':side.upper(),
                'american_odds':int(odds),'projected_k':mu,'model_probability':prob,
                'fanduel_implied_probability':imp,'probability_edge':prob-imp,'expected_value_per_unit':ev})
        except (TypeError,ValueError): continue
    allq=pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True); CURRENT.mkdir(parents=True,exist_ok=True)
    allq.to_csv(OUT/'fanduel_alt_k_candidates.csv',index=False)
    if allq.empty:
        pd.DataFrame().to_csv(OUT/'fanduel_paper_selections.csv',index=False); return
    # One independent paper wager per pitcher/start/snapshot: highest actual EV.
    sel=(allq.sort_values(['expected_value_per_unit','probability_edge'],ascending=False)
         .drop_duplicates(['date','event_id','pitcher_id']))
    sel['protocol_version']='2026-08-26-v1'
    sel['threshold_0']=sel.probability_edge.ge(0)
    for x in (0.025,0.05,0.075,0.10): sel[f'threshold_{str(x).replace(".","_")}']=sel.probability_edge.ge(x)
    sel['decision']='PAPER BET' # never implies real-money authorization
    sel.to_csv(OUT/'fanduel_paper_selections.csv',index=False)
    old=pd.read_csv(HISTORY,low_memory=False) if HISTORY.exists() else pd.DataFrame()
    hist=pd.concat([old,sel],ignore_index=True,sort=False)
    keys=['date','event_id','pitcher_id','snapshot_time_et','protocol_version']
    hist=hist.drop_duplicates(keys,keep='first')
    hist.to_csv(HISTORY,index=False)
    print(f'FanDuel candidates={len(allq)}; frozen independent paper selections={len(sel)}; history={len(hist)}')
    print(sel.sort_values('expected_value_per_unit',ascending=False).head(30).round(4).to_string(index=False))

if __name__=='__main__': main()
