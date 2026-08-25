"""Build opponent starting-lineup K features against each starter's throwing hand.

This implementation is leakage-safe and vectorized. Historical lineup batters are
matched to the most recent batter/hand feature row strictly BEFORE the game date via
merge_asof, replacing the previous O(starts * lineup * history) nested filtering loop.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/features'
LINE=D/'historical_starting_lineups.csv'
B=D/'batter_k_by_pitcher_hand_pregame.csv'
PH=D/'pitcher_handedness.csv'
LOG=ROOT/'data/mlb_pitcher_game_logs.csv'
OUT=D/'historical_lineup_hand_features.csv'


def main():
    l=pd.read_csv(LINE,low_memory=False)
    l['game_date']=pd.to_datetime(l.game_date)
    l['game_pk']=pd.to_numeric(l.game_pk,errors='coerce')
    l['batter_id']=pd.to_numeric(l.batter_id,errors='coerce')
    l['team_id']=pd.to_numeric(l.team_id,errors='coerce')
    l=l.dropna(subset=['game_pk','batter_id','team_id']).copy()

    b=pd.read_csv(B,low_memory=False)
    b['game_date']=pd.to_datetime(b.game_date)
    b['batter_id']=pd.to_numeric(b.batter_id,errors='coerce')
    b['pitcher_hand']=b.pitcher_hand.astype(str).str.upper()
    keep=['game_date','batter_id','pitcher_hand','batter_k_pa_30d','batter_k_pa_90d','batter_k_pa_365d','batter_pa_90d']
    b=b[keep].dropna(subset=['game_date','batter_id','pitcher_hand']).copy()
    for c in keep[3:]: b[c]=pd.to_numeric(b[c],errors='coerce')

    p=pd.read_csv(LOG,low_memory=False)
    p=p[pd.to_numeric(p.is_starter,errors='coerce').eq(1)].copy()
    p['date']=pd.to_datetime(p.date)
    p['pitcher_id']=pd.to_numeric(p.pitcher_id,errors='coerce')
    p['game_pk']=pd.to_numeric(p.game_id,errors='coerce')
    ph=pd.read_csv(PH)
    ph['pitcher_id']=pd.to_numeric(ph.pitcher_id,errors='coerce')
    ph['pitcher_hand']=ph.pitcher_hand.astype(str).str.upper()
    p=p.merge(ph[['pitcher_id','pitcher_hand']],on='pitcher_id',how='left')
    starters=p[['game_pk','date','pitcher_id','pitcher_hand','side']].dropna(subset=['game_pk','date','pitcher_id','pitcher_hand'])

    sides=l[['game_pk','team_id','side']].drop_duplicates()
    home=sides[sides.side.eq('home')][['game_pk','team_id']].rename(columns={'team_id':'home_team'})
    away=sides[sides.side.eq('away')][['game_pk','team_id']].rename(columns={'team_id':'away_team'})
    starters=starters.merge(home,on='game_pk',how='left').merge(away,on='game_pk',how='left')
    starters['opp_team']=np.where(starters.side.eq('home'),starters.away_team,starters.home_team)
    starters=starters.dropna(subset=['opp_team'])

    # Expand each starter to the opposing starting nine in one join.
    opp_lineups=l[['game_pk','team_id','batting_order','batter_id']].rename(columns={'team_id':'opp_team'})
    x=starters.merge(opp_lineups,on=['game_pk','opp_team'],how='inner')
    x=x[pd.to_numeric(x.batting_order,errors='coerce').between(1,9)].copy()
    x['batter_id']=pd.to_numeric(x.batter_id,errors='coerce')
    x=x.dropna(subset=['batter_id'])

    # Leakage guard: allow_exact_matches=False guarantees same-day batter results can
    # never feed that game's starter projection.
    left=x.sort_values(['date','batter_id','pitcher_hand']).reset_index(drop=True)
    right=b.sort_values(['game_date','batter_id','pitcher_hand']).reset_index(drop=True)
    matched=pd.merge_asof(
        left,right,
        left_on='date',right_on='game_date',
        by=['batter_id','pitcher_hand'],
        direction='backward',allow_exact_matches=False
    )

    matched['has_match']=matched['game_date'].notna()
    pa90=pd.to_numeric(matched['batter_pa_90d'],errors='coerce')
    k90=pd.to_numeric(matched['batter_k_pa_90d'],errors='coerce')
    k365=pd.to_numeric(matched['batter_k_pa_365d'],errors='coerce')
    rel=(pa90/(pa90+40)).clip(0,1)
    matched['lineup_k_blend']=(rel*k90+(1-rel)*k365)
    matched['top4_blend']=matched['lineup_k_blend'].where(pd.to_numeric(matched.batting_order,errors='coerce').le(4))

    keys=['game_pk','date','pitcher_id','opp_team','pitcher_hand']
    g=matched.groupby(keys,dropna=False)
    out=g.agg(
        lineup_batters=('batter_id','nunique'),
        matched_batters=('has_match','sum'),
        lineup_k30_mean=('batter_k_pa_30d','mean'),
        lineup_k90_mean=('batter_k_pa_90d','mean'),
        lineup_k365_mean=('batter_k_pa_365d','mean'),
        lineup_k_blend_mean=('lineup_k_blend','mean'),
        lineup_k_blend_top4=('top4_blend','mean'),
        lineup_pa90_mean=('batter_pa_90d','mean'),
    ).reset_index()
    out['lineup_match_coverage']=out.matched_batters/out.lineup_batters.clip(lower=1)
    out=out.rename(columns={'game_pk':'game_id','opp_team':'opponent_team_id'})
    out=out[out.matched_batters.gt(0)].sort_values(['date','game_id','pitcher_id'])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT,index=False)
    cov=out.lineup_match_coverage.mean() if len(out) else float('nan')
    print(f'Wrote {len(out):,} historical lineup matchup rows; mean matched coverage={cov:.3f}')

if __name__=='__main__': main()
