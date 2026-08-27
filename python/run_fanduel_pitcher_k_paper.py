"""Generate FanDuel-only pitcher-K paper selections from the free PropLine snapshot.

Research/paper mode only. This intentionally does not place bets.
Rules are frozen in docs/FANDUEL_PROSPECTIVE_PROTOCOL.md.
"""
from __future__ import annotations

from datetime import date
from math import floor
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

from build_pitcher_k_model import build_table
from run_pitcher_k_props import live_features, poisson_cdf, schedule, selected_component_weight
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; MARKET=DATA/'market/pitcher_k_historical_raw.csv'; CURRENT=DATA/'current'; OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True); CURRENT.mkdir(parents=True,exist_ok=True)
CANDIDATES=OUT/'fanduel_pitcher_k_candidates.csv'; SELECTIONS=OUT/'fanduel_pitcher_k_paper_selections.csv'; HISTORY=CURRENT/'fanduel_pitcher_k_paper_history.csv'


def norm_name(s):
    x=unicodedata.normalize('NFKD',str(s or '')).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]','',x)


def implied_prob(price):
    x=float(price); return 100/(x+100) if x>0 else -x/(-x+100)


def profit_for_win(price):
    x=float(price); return x/100 if x>0 else 100/(-x)


def fair_probs(line,mu):
    if abs(line-round(line))<1e-9:
        k=int(round(line)); pu=poisson_cdf(k-1,mu); pp=max(poisson_cdf(k,mu)-poisson_cdf(k-1,mu),0.0); po=1-poisson_cdf(k,mu)
    else:
        cut=floor(line); pu=poisson_cdf(cut,mu); pp=0.0; po=1-pu
    return float(po),float(pu),float(pp)


def fit_live_projection(day):
    hist=build_table(); hist=hist[pd.to_numeric(hist['batters_faced'],errors='coerce').gt(0)].copy(); hist['k_rate_target']=(hist.strikeouts/hist.batters_faced).clip(0,0.7)
    bf_feats,kr_feats,all_feats=specialized_features(hist); target=pd.Timestamp(day); train=hist[hist.date<target]
    if len(train)<1500: raise SystemExit('Not enough historical starts to fit FanDuel live pitcher-K model.')
    bf=hgb('poisson',leaves=12,l2=4.0); kr=hgb('squared_error',leaves=12,l2=4.0); direct=hgb('poisson',leaves=15,l2=2.0)
    bf.fit(train[bf_feats],train.batters_faced); kr.fit(train[kr_feats],train.k_rate_target,m__sample_weight=train.batters_faced); direct.fit(train[all_feats],train.strikeouts)
    slate=schedule(day)
    if slate.empty: return pd.DataFrame()
    live=live_features(slate,target); missing=[c for c in all_feats if c not in live.columns]
    if missing: raise ValueError('FanDuel live feature builder is missing: '+', '.join(missing))
    bf_hat=np.clip(bf.predict(live[bf_feats]),5,40); kr_hat=np.clip(kr.predict(live[kr_feats]),0.02,0.55); component=np.clip(bf_hat*kr_hat,0.05,None); direct_hat=np.clip(direct.predict(live[all_feats]),0.05,None); w=selected_component_weight()
    live['projected_bf']=bf_hat; live['projected_k_rate']=kr_hat; live['projected_k']=np.clip(w*component+(1-w)*direct_hat,0.05,None); live['name_key']=live.pitcher_name.map(norm_name)
    return live


def append_history(fresh):
    if fresh.empty: return
    if HISTORY.exists(): old=pd.read_csv(HISTORY,low_memory=False); h=pd.concat([old,fresh],ignore_index=True,sort=False)
    else: h=fresh.copy()
    # A recommendation is immutable once frozen. Re-runs cannot replace it.
    keys=['date','game_id','pitcher_id','line','side','collected_at_utc']
    h=h.drop_duplicates([c for c in keys if c in h.columns],keep='first')
    h.to_csv(HISTORY,index=False)


def main():
    day=date.today().isoformat()
    if not MARKET.exists(): raise SystemExit('Missing free PropLine market snapshot; run fetch_propline_sample.py first.')
    raw=pd.read_csv(MARKET,low_memory=False)
    if raw.empty: print('No PropLine pitcher-K rows available.'); return
    raw['date']=pd.to_datetime(raw.date,errors='coerce').dt.date.astype('string'); raw['sportsbook']=raw.sportsbook.astype(str).str.strip().str.lower(); raw['side']=raw.side.astype(str).str.strip().str.lower(); raw['line']=pd.to_numeric(raw.line,errors='coerce'); raw['price']=pd.to_numeric(raw.price,errors='coerce')
    fd=raw[raw.date.eq(day)&raw.sportsbook.eq('fanduel')&raw.side.isin(['over','under'])&raw.line.notna()&raw.price.notna()].copy()
    if fd.empty: print(f'No FanDuel pitcher-K quotes found for {day}.'); pd.DataFrame().to_csv(CANDIDATES,index=False); pd.DataFrame().to_csv(SELECTIONS,index=False); return
    live=fit_live_projection(day)
    if live.empty: print(f'No probable MLB starters found for {day}.'); return
    fd['name_key']=fd.pitcher_name.map(norm_name)
    keep=['game_id','pitcher_id','pitcher_name','name_key','projected_bf','projected_k_rate','projected_k']
    z=fd.merge(live[keep].rename(columns={'pitcher_name':'mlb_pitcher_name'}),on='name_key',how='inner')
    rows=[]
    for r in z.itertuples(index=False):
        po,pu,pp=fair_probs(float(r.line),float(r.projected_k)); side=str(r.side).upper(); mp=po if side=='OVER' else pu; ip=implied_prob(r.price); ev=mp*profit_for_win(r.price)-(1-mp-pp)
        rows.append({'date':day,'event_id':getattr(r,'event_id',None),'game_id':r.game_id,'pitcher_id':r.pitcher_id,'pitcher_name':r.mlb_pitcher_name,'commence_time_utc':getattr(r,'commence_time_utc',None),'collected_at_utc':getattr(r,'collected_at_utc',None),'collected_at_et':getattr(r,'collected_at_et',None),'sportsbook':'FanDuel','line':float(r.line),'side':side,'fanduel_price':float(r.price),'projected_k':float(r.projected_k),'projected_bf':float(r.projected_bf),'projected_k_rate':float(r.projected_k_rate),'fair_over_prob':po,'fair_under_prob':pu,'push_prob':pp,'model_win_prob':mp,'fanduel_implied_prob':ip,'model_market_edge':mp-ip,'expected_profit_per_unit':ev,'decision':'PAPER BET - NOT LIVE','protocol_version':'2026-08-26-v1'})
    cand=pd.DataFrame(rows).sort_values(['expected_profit_per_unit','model_market_edge'],ascending=False); cand.to_csv(CANDIDATES,index=False)
    sel=(cand[cand.expected_profit_per_unit.gt(0)].sort_values(['expected_profit_per_unit','model_market_edge'],ascending=False).drop_duplicates(['game_id','pitcher_id'],keep='first').sort_values('expected_profit_per_unit',ascending=False))
    sel.to_csv(SELECTIONS,index=False); append_history(sel)
    print(f'FanDuel candidate rows: {len(cand):,}; frozen independent paper selections: {len(sel):,}')
    if len(sel): print(sel[['pitcher_name','side','line','fanduel_price','projected_k','model_win_prob','model_market_edge','expected_profit_per_unit']].round(4).to_string(index=False))
    else: print('NO PAPER BETS: no positive-EV FanDuel candidate survived.')

if __name__=='__main__': main()
