"""Build leakage-safe MLB starter/team features and walk-forward model tests.

Inputs expected under data/:
  mlb_games_2018_present.csv
  mlb_game_enrichment.csv
  mlb_pitcher_game_logs.csv
  mlb_team_game_logs.csv
  mlb_odds_part_*.csv (optional for market comparison)

Outputs under outputs/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

def ip_to_outs(x):
    try:
        a,b=str(x).split('.'); return int(a)*3+int(b)
    except: return np.nan

def american_prob(x):
    x=pd.to_numeric(x,errors='coerce'); return np.where(x>0,100/(x+100),-x/(-x+100))

g=pd.read_csv(DATA/'mlb_games_2018_present.csv',low_memory=False); e=pd.read_csv(DATA/'mlb_game_enrichment.csv',low_memory=False)
p=pd.read_csv(DATA/'mlb_pitcher_game_logs.csv',low_memory=False); t=pd.read_csv(DATA/'mlb_team_game_logs.csv',low_memory=False)
for d in (g,e,p,t): d['date']=pd.to_datetime(d['date'],errors='coerce')
g=g.merge(e[['game_id','home_starter_id','away_starter_id']].drop_duplicates('game_id'),on='game_id',how='left')
g['home_win']=(pd.to_numeric(g.home_score,errors='coerce')>pd.to_numeric(g.away_score,errors='coerce')).astype(float)

# Starter rolling stats, shifted one appearance so today's game never enters today's features.
p['outs']=p.innings_pitched.map(ip_to_outs); p['ip']=p.outs/3
for c in ['earned_runs','walks','strikeouts','home_runs','hits','batters_faced','pitches']: p[c]=pd.to_numeric(p[c],errors='coerce')
p=p.sort_values(['pitcher_id','date','game_id']); starters=p[p.is_starter==1].copy()
for n in (3,5,10):
    grp=starters.groupby('pitcher_id',group_keys=False)
    for c in ['ip','earned_runs','walks','strikeouts','home_runs','hits','pitches']:
        starters[f'sp_{c}_{n}']=grp[c].transform(lambda s:s.shift().rolling(n,min_periods=1).sum())
    starters[f'sp_era_{n}']=9*starters[f'sp_earned_runs_{n}']/starters[f'sp_ip_{n}'].replace(0,np.nan)
    starters[f'sp_whip_{n}']=(starters[f'sp_walks_{n}']+starters[f'sp_hits_{n}'])/starters[f'sp_ip_{n}'].replace(0,np.nan)
    starters[f'sp_k9_{n}']=9*starters[f'sp_strikeouts_{n}']/starters[f'sp_ip_{n}'].replace(0,np.nan)
    starters[f'sp_bb9_{n}']=9*starters[f'sp_walks_{n}']/starters[f'sp_ip_{n}'].replace(0,np.nan)
    starters[f'sp_hr9_{n}']=9*starters[f'sp_home_runs_{n}']/starters[f'sp_ip_{n}'].replace(0,np.nan)
spcols=[c for c in starters if c.startswith('sp_') and any(c.endswith('_'+str(n)) for n in (3,5,10))]
sm=starters[['game_id','side']+spcols].pivot(index='game_id',columns='side'); sm.columns=[f'{side}_{c}' for c,side in sm.columns]; sm=sm.reset_index()
g=g.merge(sm,on='game_id',how='left')

# Team form from box-score logs, also shifted.
for c in ['runs','hits','home_runs','walks','strikeouts','pitching_earned_runs','pitching_walks','pitching_strikeouts','pitching_home_runs','errors']: t[c]=pd.to_numeric(t[c],errors='coerce')
t=t.sort_values(['team_id','date','game_id']); tg=t.groupby('team_id',group_keys=False)
for n in (10,30):
    for c in ['runs','hits','home_runs','walks','strikeouts','pitching_earned_runs','pitching_walks','pitching_strikeouts','pitching_home_runs','errors']:
        t[f'team_{c}_{n}']=tg[c].transform(lambda s:s.shift().rolling(n,min_periods=3).mean())
tcols=[c for c in t if c.startswith('team_') and c not in ['team_id','team_name']]
tm=t[['game_id','side']+tcols].pivot(index='game_id',columns='side'); tm.columns=[f'{side}_{c}' for c,side in tm.columns]; tm=tm.reset_index(); g=g.merge(tm,on='game_id',how='left')

# Convert home/away features to matchup differences.
features=[]
for c in spcols+tcols:
    h='home_'+c; a='away_'+c
    if h in g and a in g: g['diff_'+c]=g[h]-g[a]; features.append('diff_'+c)

# Optional sportsbook consensus closing ML.
parts=sorted(DATA.glob('mlb_odds_part_*.csv'))
if parts:
    o=pd.concat([pd.read_csv(x,low_memory=False) for x in parts],ignore_index=True); o=o[o.market=='moneyline'].copy()
    # Match on date/team names; doubleheaders are deliberately excluded when ambiguous.
    o['date']=pd.to_datetime(o.date,errors='coerce'); o['ph']=american_prob(o.close_home_odds); o['pa']=american_prob(o.close_away_odds); o['market_prob']=o.ph/(o.ph+o.pa)
    oc=o.groupby(['date','home_team','away_team']).agg(market_home_prob=('market_prob','median')).reset_index()
    counts=g.groupby(['date','home_team','away_team']).size().rename('n').reset_index(); oc=oc.merge(counts,on=['date','home_team','away_team'],how='left'); oc=oc[oc.n==1].drop(columns='n')
    g=g.merge(oc,on=['date','home_team','away_team'],how='left')

usable=g[g.home_win.notna()].copy(); X=usable[features]; y=usable.home_win.astype(int); years=usable.date.dt.year
models={'logistic':Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',LogisticRegression(max_iter=2000))]),'hist_gb':Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingClassifier(max_iter=250,l2_regularization=1.0))])}
rows=[]; preds=[]
for year in range(2021,2026):
    tr=years<year; te=years==year
    if tr.sum()<500 or te.sum()==0: continue
    for name,model in models.items():
        model.fit(X[tr],y[tr]); pr=model.predict_proba(X[te])[:,1]; yy=y[te]
        r={'season':year,'model':name,'games':int(te.sum()),'accuracy':accuracy_score(yy,pr>=.5),'log_loss':log_loss(yy,pr),'brier':brier_score_loss(yy,pr)}
        if 'market_home_prob' in usable:
            mp=usable.loc[te,'market_home_prob']; ok=mp.notna(); r['market_games']=int(ok.sum())
            if ok.any(): r['market_log_loss']=log_loss(yy[ok],mp[ok]); r['market_brier']=brier_score_loss(yy[ok],mp[ok])
        rows.append(r); z=usable.loc[te,['game_id','date','home_team','away_team','home_win']].copy(); z['model']=name; z['model_home_prob']=pr; preds.append(z)
pd.DataFrame(rows).to_csv(OUT/'pitcher_model_walkforward_metrics.csv',index=False); pd.concat(preds).to_csv(OUT/'pitcher_model_walkforward_predictions.csv',index=False)
g[['game_id','date','home_team','away_team','home_win']+features+(['market_home_prob'] if 'market_home_prob' in g else [])].to_csv(OUT/'pitcher_modeling_table.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(f'Features: {len(features)}')