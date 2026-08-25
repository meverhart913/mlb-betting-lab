"""Backfill historical MLB starting batting orders from the public MLB Stats API.

Stores only pregame-identifiable lineup structure: game/date/team/opponent, batting
order and batter id. The source endpoint is fetched after the fact, but the starting
lineup itself is not an outcome statistic. V2.2 combines it only with batter features
shifted strictly before game date.
"""
from __future__ import annotations
import argparse,time
from datetime import date,timedelta
from pathlib import Path
import pandas as pd, requests
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/features/historical_starting_lineups.csv'; BASE='https://statsapi.mlb.com/api/v1'
def schedule(a,b):
 r=requests.get(f'{BASE}/schedule',params={'sportId':1,'startDate':a,'endDate':b,'hydrate':'team'},timeout=60); r.raise_for_status();
 for d in r.json().get('dates',[]):
  for g in d.get('games',[]): yield d['date'],g
def lineup(game_pk,side):
 r=requests.get(f'{BASE}/game/{game_pk}/boxscore',timeout=60); r.raise_for_status(); t=r.json().get('teams',{}).get(side,{})
 order=t.get('battingOrder') or []
 # battingOrder is ordered MLBAM player IDs for starters/participants. First nine are
 # the starting order for standard MLB games; dedupe defensively while preserving order.
 seen=[]
 for x in order:
  try: x=int(x)
  except: continue
  if x not in seen: seen.append(x)
 return seen[:9]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--start',required=True); ap.add_argument('--end',required=True); ap.add_argument('--sleep-seconds',type=float,default=.08); x=ap.parse_args(); rows=[]
 for ds,g in schedule(x.start,x.end):
  pk=g.get('gamePk'); teams=g.get('teams',{}); home=teams.get('home',{}).get('team',{}).get('id'); away=teams.get('away',{}).get('team',{}).get('id')
  if not pk or not home or not away: continue
  for side,team,opp in [('home',home,away),('away',away,home)]:
   try: bats=lineup(pk,side)
   except Exception as e: print('WARN',pk,side,e); continue
   if len(bats)<9: print('WARN incomplete lineup',pk,side,len(bats)); continue
   for slot,bid in enumerate(bats,1): rows.append({'game_date':ds,'game_pk':pk,'team_id':team,'opponent_team_id':opp,'side':side,'batting_order':slot,'batter_id':bid})
   time.sleep(max(0,x.sleep_seconds))
 out=pd.DataFrame(rows); OUT.parent.mkdir(parents=True,exist_ok=True)
 if OUT.exists(): out=pd.concat([pd.read_csv(OUT),out],ignore_index=True)
 if not out.empty: out=out.drop_duplicates(['game_pk','team_id','batting_order'],keep='last').sort_values(['game_date','game_pk','team_id','batting_order'])
 out.to_csv(OUT,index=False); print(f'Wrote {len(out):,} historical lineup rows to {OUT}')
if __name__=='__main__': main()
