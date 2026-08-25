"""Build a stable MLB pitcher-id -> throwing-hand lookup from Statcast.

Uses pitch-level Statcast where p_throws is directly observed. The lookup is safe for
historical modeling because throwing hand is a stable player attribute, not a game
outcome. Conflicting observations are resolved by the modal observed hand and flagged.
"""
from __future__ import annotations
import argparse
from datetime import date,timedelta
from io import StringIO
from pathlib import Path
import time
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/features/pitcher_handedness.csv'; BASE='https://baseballsavant.mlb.com/statcast_search/csv'
def chunks(a,b,n):
 c=a
 while c<=b:
  h=min(b,c+timedelta(days=n-1)); yield c,h; c=h+timedelta(days=1)
def fetch(a,b):
 r=requests.get(BASE,params={'all':'true','type':'pitcher','player_type':'pitcher','game_date_gt':a.isoformat(),'game_date_lt':b.isoformat(),'hfGT':'R|'},timeout=120); r.raise_for_status(); t=r.text.strip(); return pd.read_csv(StringIO(t),low_memory=False) if t else pd.DataFrame()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--start',default='2021-03-15'); ap.add_argument('--end',default=date.today().isoformat()); ap.add_argument('--chunk-days',type=int,default=3); ap.add_argument('--sleep-seconds',type=float,default=.2); x=ap.parse_args(); parts=[]
 for a,b in chunks(date.fromisoformat(x.start),date.fromisoformat(x.end),x.chunk_days):
  print('Fetching',a,b); q=fetch(a,b)
  if not q.empty:
   miss={'pitcher','p_throws'}-set(q.columns)
   if miss: raise ValueError('Missing Statcast columns: '+','.join(sorted(miss)))
   parts.append(q[['pitcher','p_throws']])
  time.sleep(max(0,x.sleep_seconds))
 if not parts: raise SystemExit('No Statcast pitcher-hand observations')
 z=pd.concat(parts,ignore_index=True); z['pitcher_id']=pd.to_numeric(z.pitcher,errors='coerce'); z['pitcher_hand']=z.p_throws.astype(str).str.upper(); z=z[z.pitcher_id.notna() & z.pitcher_hand.isin(['L','R'])]
 counts=z.groupby(['pitcher_id','pitcher_hand']).size().rename('n').reset_index(); totals=counts.groupby('pitcher_id').n.sum(); best=counts.sort_values(['pitcher_id','n'],ascending=[True,False]).drop_duplicates('pitcher_id'); best['observations']=best.pitcher_id.map(totals); best['hand_share']=best.n/best.observations; best['hand_conflict']=best.hand_share.lt(.99); out=best[['pitcher_id','pitcher_hand','observations','hand_share','hand_conflict']].sort_values('pitcher_id'); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False); print(f'Wrote {len(out):,} pitcher handedness rows; conflicts={int(out.hand_conflict.sum())}')
if __name__=='__main__': main()
