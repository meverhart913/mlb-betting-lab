"""Walk-forward V2.2 test: validated Statcast-all vs actual starting-lineup K/hand features."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_poisson_deviance,root_mean_squared_error
from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb,specialized_features
R=Path(__file__).resolve().parents[1]; D=R/'data/features'; OUT=R/'outputs'; OUT.mkdir(exist_ok=True)
def fit(z,tr,te,bf,kr,d):
 a=hgb('poisson',leaves=12,l2=4); b=hgb('squared_error',leaves=12,l2=4); c=hgb('poisson',leaves=15,l2=2); a.fit(z.loc[tr,bf],z.loc[tr,'batters_faced']); b.fit(z.loc[tr,kr],z.loc[tr,'k_rate_target'],m__sample_weight=z.loc[tr,'batters_faced']); c.fit(z.loc[tr,d],z.loc[tr,'strikeouts']); return np.clip(.5*np.clip(a.predict(z.loc[te,bf]),5,40)*np.clip(b.predict(z.loc[te,kr]),.02,.55)+.5*np.clip(c.predict(z.loc[te,d]),.05,None),.05,None)
def main():
 z=build_table().copy(); z['date']=pd.to_datetime(z.date); z['pitcher_id']=pd.to_numeric(z.pitcher_id,errors='coerce'); z['game_id']=pd.to_numeric(z.game_id,errors='coerce'); z=z[pd.to_numeric(z.batters_faced,errors='coerce').gt(0)].copy(); z['k_rate_target']=(z.strikeouts/z.batters_faced).clip(0,.7)
 sc=pd.read_csv(D/'statcast_pitcher_pregame.csv',low_memory=False); sc['game_date']=pd.to_datetime(sc.game_date); sc['pitcher_id']=pd.to_numeric(sc.pitcher_id,errors='coerce'); sc=sc.rename(columns={'game_date':'date'}); stat=[c for c in sc if c.startswith('statcast_')]; z=z.merge(sc[['date','pitcher_id',*stat]],on=['date','pitcher_id'],how='left')
 lu=pd.read_csv(D/'historical_lineup_hand_features.csv',low_memory=False); lu['game_id']=pd.to_numeric(lu.game_id,errors='coerce'); lu['pitcher_id']=pd.to_numeric(lu.pitcher_id,errors='coerce'); lu['date']=pd.to_datetime(lu.date); lf=[c for c in lu if c.startswith('lineup_') and c not in ['lineup_batters']]; z=z.merge(lu[['game_id','pitcher_id',*lf]],on=['game_id','pitcher_id'],how='left')
 bf0,kr0,d0=specialized_features(z); models={'statcast_all':(bf0+stat,kr0+stat,d0+stat),'v22_lineup_all':(bf0+stat+lf,kr0+stat+lf,d0+stat+lf),'v22_lineup_kr_direct':(bf0+stat,kr0+stat+lf,d0+stat+lf),'v22_lineup_kr_only':(bf0+stat,kr0+stat+lf,d0+stat)}; rows=[]
 for yr in range(2022,2027):
  tr=z.season<yr; te=z.season==yr
  if tr.sum()<1500 or te.sum()<300: continue
  y=z.loc[te,'strikeouts'].to_numpy(float)
  for name,(bf,kr,d) in models.items():
   mu=fit(z,tr,te,sorted(set(bf)),sorted(set(kr)),sorted(set(d))); rows.append({'season':yr,'model':name,'starts':int(te.sum()),'lineup_row_coverage':float(z.loc[te,'lineup_match_coverage'].notna().mean()),'mean_batter_match_coverage':float(z.loc[te,'lineup_match_coverage'].mean()),'mae':mean_absolute_error(y,mu),'rmse':root_mean_squared_error(y,mu),'poisson_deviance':mean_poisson_deviance(y,mu)})
 m=pd.DataFrame(rows); m.to_csv(OUT/'pitcher_k_batter_hand_metrics.csv',index=False); s=m.groupby('model',as_index=False).agg(seasons=('season','nunique'),lineup_row_coverage=('lineup_row_coverage','mean'),batter_match_coverage=('mean_batter_match_coverage','mean'),mae=('mae','mean'),rmse=('rmse','mean'),poisson_deviance=('poisson_deviance','mean')).sort_values(['poisson_deviance','mae']); base=float(s.loc[s.model.eq('statcast_all'),'poisson_deviance'].iloc[0]); s['deviance_improvement_vs_statcast']=base-s.poisson_deviance; s['promotion_candidate']=(s.model.ne('statcast_all') & s.lineup_row_coverage.ge(.90) & s.batter_match_coverage.ge(.80) & s.poisson_deviance.lt(base)); s.to_csv(OUT/'pitcher_k_batter_hand_summary.csv',index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
