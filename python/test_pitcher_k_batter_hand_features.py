"""Research gate for batter K tendency versus starter handedness.

This first V2.2 ablation tests whether leakage-safe handedness features add signal on
top of Statcast-all. It deliberately does not promote itself to the live scorer.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb, specialized_features
ROOT=Path(__file__).resolve().parents[1]; HAND=ROOT/'data/features/batter_k_by_pitcher_hand_pregame.csv'; STAT=ROOT/'data/features/statcast_pitcher_pregame.csv'; PITCHER_HAND=ROOT/'data/features/pitcher_handedness.csv'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
def fit(z,tr,te,bf,kr,direct):
 a=hgb('poisson',leaves=12,l2=4.0); b=hgb('squared_error',leaves=12,l2=4.0); c=hgb('poisson',leaves=15,l2=2.0); a.fit(z.loc[tr,bf],z.loc[tr,'batters_faced']); b.fit(z.loc[tr,kr],z.loc[tr,'k_rate_target'],m__sample_weight=z.loc[tr,'batters_faced']); c.fit(z.loc[tr,direct],z.loc[tr,'strikeouts']); return np.clip(.5*(np.clip(a.predict(z.loc[te,bf]),5,40)*np.clip(b.predict(z.loc[te,kr]),.02,.55))+.5*np.clip(c.predict(z.loc[te,direct]),.05,None),.05,None)
def main():
 for p in [HAND,STAT,PITCHER_HAND]:
  if not p.exists(): raise SystemExit(f'Missing {p}')
 z=build_table().copy(); z['date']=pd.to_datetime(z.date); z['pitcher_id']=pd.to_numeric(z.pitcher_id,errors='coerce'); z=z[pd.to_numeric(z.batters_faced,errors='coerce').gt(0)].copy(); z['k_rate_target']=(z.strikeouts/z.batters_faced).clip(0,.7)
 sc=pd.read_csv(STAT,low_memory=False); sc['game_date']=pd.to_datetime(sc.game_date); sc['pitcher_id']=pd.to_numeric(sc.pitcher_id,errors='coerce'); sc=sc.rename(columns={'game_date':'date'}); stat=[c for c in sc if c.startswith('statcast_')]; z=z.merge(sc[['date','pitcher_id',*stat]],on=['date','pitcher_id'],how='left')
 ph=pd.read_csv(PITCHER_HAND); ph['pitcher_id']=pd.to_numeric(ph.pitcher_id,errors='coerce'); ph=ph[(ph.pitcher_hand.isin(['L','R'])) & (pd.to_numeric(ph.hand_share,errors='coerce')>=.99)]; z=z.merge(ph[['pitcher_id','pitcher_hand']],on='pitcher_id',how='left')
 h=pd.read_csv(HAND,low_memory=False); h['game_date']=pd.to_datetime(h.game_date)
 cols=['batter_k_pa_30d','batter_k_pa_90d','batter_k_pa_365d','batter_pa_90d']
 for c in cols: h[c]=pd.to_numeric(h[c],errors='coerce')
 # Date/hand population aggregate is only a plumbing ablation. It cannot establish a
 # lineup edge because it is not opponent-specific. A later historical-lineup table must
 # beat this benchmark before live promotion.
 agg=h.groupby(['game_date','pitcher_hand'],as_index=False).agg(matchup_k30=('batter_k_pa_30d','mean'),matchup_k90=('batter_k_pa_90d','mean'),matchup_k365=('batter_k_pa_365d','mean'),matchup_pa90=('batter_pa_90d','median')); z=z.merge(agg.rename(columns={'game_date':'date'}),on=['date','pitcher_hand'],how='left')
 bf0,kr0,d0=specialized_features(z); hand=['matchup_k30','matchup_k90','matchup_k365','matchup_pa90']; placements={'statcast_all':(bf0+stat,kr0+stat,d0+stat),'statcast_plus_hand_all':(bf0+stat+hand,kr0+stat+hand,d0+stat+hand),'statcast_plus_hand_kr_direct':(bf0+stat,kr0+stat+hand,d0+stat+hand)}; rows=[]
 for yr in range(2022,2027):
  tr=z.season<yr; te=z.season==yr
  if tr.sum()<1500 or te.sum()<300: continue
  y=z.loc[te,'strikeouts'].to_numpy(float)
  for label,(bf,kr,d) in placements.items():
   mu=fit(z,tr,te,sorted(set(bf)),sorted(set(kr)),sorted(set(d))); rows.append({'season':yr,'model':label,'starts':int(te.sum()),'pitcher_hand_coverage':float(z.loc[te,'pitcher_hand'].notna().mean()),'matchup_coverage':float(z.loc[te,hand].notna().any(axis=1).mean()),'mae':mean_absolute_error(y,mu),'rmse':root_mean_squared_error(y,mu),'poisson_deviance':mean_poisson_deviance(y,mu)})
 m=pd.DataFrame(rows); m.to_csv(OUT/'pitcher_k_batter_hand_metrics.csv',index=False); s=m.groupby('model',as_index=False).agg(seasons=('season','nunique'),pitcher_hand_coverage=('pitcher_hand_coverage','mean'),matchup_coverage=('matchup_coverage','mean'),mae=('mae','mean'),rmse=('rmse','mean'),poisson_deviance=('poisson_deviance','mean')).sort_values(['poisson_deviance','mae']); s.to_csv(OUT/'pitcher_k_batter_hand_summary.csv',index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
