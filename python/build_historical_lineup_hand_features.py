"""Build opponent starting-lineup K features against each starter's throwing hand.

Joins historical starting orders to batter/hand rolling features using an as-of lookup
strictly before game date. Output is one row per game/opponent team and contains only
pregame batter history; same-day outcomes are never used.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data/features'; LINE=D/'historical_starting_lineups.csv'; B=D/'batter_k_by_pitcher_hand_pregame.csv'; PH=D/'pitcher_handedness.csv'; LOG=ROOT/'data/mlb_pitcher_game_logs.csv'; OUT=D/'historical_lineup_hand_features.csv'
def main():
 l=pd.read_csv(LINE); l['game_date']=pd.to_datetime(l.game_date); l['batter_id']=pd.to_numeric(l.batter_id,errors='coerce'); l['team_id']=pd.to_numeric(l.team_id,errors='coerce')
 b=pd.read_csv(B,low_memory=False); b['game_date']=pd.to_datetime(b.game_date); b['batter_id']=pd.to_numeric(b.batter_id,errors='coerce')
 p=pd.read_csv(LOG,low_memory=False); p=p[pd.to_numeric(p.is_starter,errors='coerce').eq(1)].copy(); p['date']=pd.to_datetime(p.date); p['pitcher_id']=pd.to_numeric(p.pitcher_id,errors='coerce')
 ph=pd.read_csv(PH); ph['pitcher_id']=pd.to_numeric(ph.pitcher_id,errors='coerce'); p=p.merge(ph[['pitcher_id','pitcher_hand']],on='pitcher_id',how='left')
 # Map starter game to opponent lineup using game id/team IDs reconstructed from lineup game_pk.
 # MLB game_id in logs is expected to be gamePk; normalize numeric strings defensively.
 p['game_pk']=pd.to_numeric(p.game_id,errors='coerce'); starters=p[['game_pk','date','pitcher_id','pitcher_hand','side']].dropna(subset=['game_pk','pitcher_hand'])
 # For each game, home starter faces away lineup and vice versa.
 sides=l[['game_pk','team_id','side']].drop_duplicates(); home=sides[sides.side.eq('home')][['game_pk','team_id']].rename(columns={'team_id':'home_team'}); away=sides[sides.side.eq('away')][['game_pk','team_id']].rename(columns={'team_id':'away_team'}); starters=starters.merge(home,on='game_pk',how='left').merge(away,on='game_pk',how='left'); starters['opp_team']=np.where(starters.side.eq('home'),starters.away_team,starters.home_team)
 rows=[]
 for r in starters.itertuples(index=False):
  nine=l[(l.game_pk==r.game_pk)&(l.team_id==r.opp_team)].sort_values('batting_order').head(9)
  vals=[]
  for x in nine.itertuples(index=False):
   hist=b[(b.batter_id==x.batter_id)&(b.pitcher_hand==r.pitcher_hand)&(b.game_date<r.date)].sort_values('game_date')
   if hist.empty: continue
   q=hist.iloc[-1]; vals.append(q)
  if not vals: continue
  def arr(c): return pd.to_numeric(pd.Series([q.get(c,np.nan) for q in vals]),errors='coerce')
  k30,k90,k365,pa90=arr('batter_k_pa_30d'),arr('batter_k_pa_90d'),arr('batter_k_pa_365d'),arr('batter_pa_90d')
  # Reliability-weighted lineup rate shrinks small 90d samples toward 365d history.
  rel=(pa90/(pa90+40)).clip(0,1); blended=(rel*k90+(1-rel)*k365)
  rows.append({'game_id':r.game_pk,'date':r.date,'pitcher_id':r.pitcher_id,'opponent_team_id':r.opp_team,'pitcher_hand':r.pitcher_hand,'lineup_batters':len(nine),'matched_batters':len(vals),'lineup_match_coverage':len(vals)/max(len(nine),1),'lineup_k30_mean':k30.mean(),'lineup_k90_mean':k90.mean(),'lineup_k365_mean':k365.mean(),'lineup_k_blend_mean':blended.mean(),'lineup_k_blend_top4':blended.head(4).mean(),'lineup_pa90_mean':pa90.mean()})
 out=pd.DataFrame(rows); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False); print(f'Wrote {len(out):,} historical lineup matchup rows; mean matched coverage={out.lineup_match_coverage.mean():.3f}' if len(out) else 'No matchup rows')
if __name__=='__main__': main()
