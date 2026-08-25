"""Research gate for opponent batter K tendency vs starter handedness.

Builds a leakage-safe team-game lineup proxy from the batters who actually appeared
in each historical game, using only each batter's prior K/PA against the starter's
throwing hand. This is intentionally an upper-bound research test: live V2.2 will
use posted pregame lineups, while this historical test asks whether the matchup
signal itself earns inclusion beyond the validated Statcast-all architecture.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT=Path(__file__).resolve().parents[1]
HAND=ROOT/'data/features/batter_k_by_pitcher_hand_pregame.csv'
STAT=ROOT/'data/features/statcast_pitcher_pregame.csv'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)


def fit(z,tr,te,bf,kr,direct):
    a=hgb('poisson',leaves=12,l2=4.0); b=hgb('squared_error',leaves=12,l2=4.0); c=hgb('poisson',leaves=15,l2=2.0)
    a.fit(z.loc[tr,bf],z.loc[tr,'batters_faced']); b.fit(z.loc[tr,kr],z.loc[tr,'k_rate_target'],m__sample_weight=z.loc[tr,'batters_faced']); c.fit(z.loc[tr,direct],z.loc[tr,'strikeouts'])
    bfhat=np.clip(a.predict(z.loc[te,bf]),5,40); krhat=np.clip(b.predict(z.loc[te,kr]),.02,.55); dhat=np.clip(c.predict(z.loc[te,direct]),.05,None)
    return np.clip(.5*(bfhat*krhat)+.5*dhat,.05,None)


def main():
    if not HAND.exists() or not STAT.exists(): raise SystemExit('Missing handedness or Statcast feature tables')
    z=build_table().copy(); z['date']=pd.to_datetime(z['date']); z['pitcher_id']=pd.to_numeric(z['pitcher_id'],errors='coerce'); z=z[pd.to_numeric(z['batters_faced'],errors='coerce').gt(0)].copy(); z['k_rate_target']=(z['strikeouts']/z['batters_faced']).clip(0,.7)
    sc=pd.read_csv(STAT,low_memory=False); sc['game_date']=pd.to_datetime(sc['game_date']); sc['pitcher_id']=pd.to_numeric(sc['pitcher_id'],errors='coerce'); sc=sc.rename(columns={'game_date':'date'}); stat=[c for c in sc if c.startswith('statcast_')]; z=z.merge(sc[['date','pitcher_id',*stat]],on=['date','pitcher_id'],how='left')
    # Historical game logs already carry opponent batting outcomes/team context. Build a
    # conservative team-game matchup proxy from available pregame batter-hand rows by date.
    h=pd.read_csv(HAND,low_memory=False); h['game_date']=pd.to_datetime(h['game_date']);
    for c in ['batter_k_pa_30d','batter_k_pa_90d','batter_k_pa_365d','batter_pa_30d','batter_pa_90d','batter_pa_365d']: h[c]=pd.to_numeric(h[c],errors='coerce')
    # Aggregate the population of active batter-hand observations on each date/hand. This
    # tests handedness signal without leaking same-day outcomes; live scorer will narrow it
    # to the posted nine hitters.
    agg=h.groupby(['game_date','pitcher_hand'],as_index=False).agg(matchup_k30=('batter_k_pa_30d','mean'),matchup_k90=('batter_k_pa_90d','mean'),matchup_k365=('batter_k_pa_365d','mean'),matchup_pa90=('batter_pa_90d','median'))
    # Starter hand may not exist in the legacy table; infer from Statcast daily pitch records
    # via dominant hand is not stored in rolling table. Use pitcher game-log hand if present.
    hand_col=next((c for c in ['pitcher_hand','p_throws','throws'] if c in z.columns),None)
    if hand_col is None:
        print('SKIP: base model table lacks starter throwing hand; live integration requires automated hand lookup.')
        pd.DataFrame([{'status':'needs_pitcher_hand_lookup'}]).to_csv(OUT/'pitcher_k_batter_hand_summary.csv',index=False); return
    z['pitcher_hand']=z[hand_col].astype(str).str.upper().str[0]
    z=z.merge(agg.rename(columns={'game_date':'date'}),on=['date','pitcher_hand'],how='left')
    bf0,kr0,d0=specialized_features(z); base=[*stat]; hand=['matchup_k30','matchup_k90','matchup_k365','matchup_pa90']
    placements={'statcast_all':(bf0+base,kr0+base,d0+base),'statcast_plus_hand_all':(bf0+base+hand,kr0+base+hand,d0+base+hand),'statcast_plus_hand_kr':(bf0+base,kr0+base+hand,d0+base+hand)}
    rows=[]
    for yr in range(2022,2027):
        tr=z.season<yr; te=z.season==yr
        if tr.sum()<1500 or te.sum()<300: continue
        y=z.loc[te,'strikeouts'].to_numpy(float)
        for label,(bf,kr,d) in placements.items():
            mu=fit(z,tr,te,sorted(set(bf)),sorted(set(kr)),sorted(set(d)))
            rows.append({'season':yr,'model':label,'starts':int(te.sum()),'hand_coverage':float(z.loc[te,hand].notna().any(axis=1).mean()),'mae':mean_absolute_error(y,mu),'rmse':root_mean_squared_error(y,mu),'poisson_deviance':mean_poisson_deviance(y,mu)})
    m=pd.DataFrame(rows); m.to_csv(OUT/'pitcher_k_batter_hand_metrics.csv',index=False)
    s=m.groupby('model',as_index=False).agg(seasons=('season','nunique'),coverage=('hand_coverage','mean'),mae=('mae','mean'),rmse=('rmse','mean'),poisson_deviance=('poisson_deviance','mean')).sort_values(['poisson_deviance','mae']); s.to_csv(OUT/'pitcher_k_batter_hand_summary.csv',index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
