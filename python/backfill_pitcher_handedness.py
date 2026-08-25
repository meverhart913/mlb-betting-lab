"""Build MLB pitcher-id -> throwing-hand lookup from MLB Stats API.

Pitching hand is a stable player attribute, so querying the player endpoint avoids an
expensive six-year pitch-level Statcast download. IDs come from the existing pitcher
logs and are fetched in batches.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]; LOG=ROOT/'data/mlb_pitcher_game_logs.csv'; OUT=ROOT/'data/features/pitcher_handedness.csv'; BASE='https://statsapi.mlb.com/api/v1/people'
def batches(xs,n=50):
 for i in range(0,len(xs),n): yield xs[i:i+n]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--batch-size',type=int,default=50); x=ap.parse_args()
 p=pd.read_csv(LOG,low_memory=False); ids=sorted(pd.to_numeric(p.pitcher_id,errors='coerce').dropna().astype(int).unique().tolist()); rows=[]
 for chunk in batches(ids,max(1,x.batch_size)):
  r=requests.get(BASE,params={'personIds':','.join(map(str,chunk)),'hydrate':'currentTeam'},timeout=60); r.raise_for_status()
  for person in r.json().get('people',[]):
   hand=(person.get('pitchHand') or {}).get('code'); pid=person.get('id')
   if pid is not None and hand in ('L','R'): rows.append({'pitcher_id':int(pid),'pitcher_hand':hand,'observations':1,'hand_share':1.0,'hand_conflict':False})
 out=pd.DataFrame(rows).drop_duplicates('pitcher_id').sort_values('pitcher_id'); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False); print(f'Wrote {len(out):,}/{len(ids):,} pitcher handedness rows from MLB Stats API')
if __name__=='__main__': main()
