import argparse,time
from pathlib import Path
import pandas as pd, requests, urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL="https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"

def session(verify):
    s=requests.Session()
    r=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504],allowed_methods=["GET"])
    s.mount("https://",HTTPAdapter(max_retries=r)); s.verify=verify
    if not verify: urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return s

def n(x):
    try:return float(x)
    except:return None

def extract(gid,j):
    gd=j.get("gameData",{}); box=j.get("liveData",{}).get("boxscore",{})
    game=gd.get("game",{}); dt=gd.get("datetime",{}); weather=gd.get("weather",{})
    venue=gd.get("venue",{}); prob=gd.get("probablePitchers",{})
    date=dt.get("officialDate") or str(dt.get("dateTime",""))[:10]; season=game.get("season")
    e={"game_id":gid,"date":date,"season":season,"game_type":game.get("type"),
       "day_night":game.get("dayNight"),"venue_id":venue.get("id"),"venue_name":venue.get("name"),
       "temperature_f":n(weather.get("temp")),"weather_condition":weather.get("condition"),"wind":weather.get("wind"),
       "home_probable_pitcher_id":(prob.get("home") or {}).get("id"),
       "away_probable_pitcher_id":(prob.get("away") or {}).get("id")}
    prs=[]; trs=[]
    for side in ["away","home"]:
        t=box.get("teams",{}).get(side,{}) or {}; team=t.get("team",{}) or {}
        ts=t.get("teamStats",{}) or {}; b=ts.get("batting",{}) or {}; p=ts.get("pitching",{}) or {}; f=ts.get("fielding",{}) or {}
        ids=t.get("pitchers",[]) or []; starter=ids[0] if ids else None; players=t.get("players",{}) or {}
        e[f"{side}_team_id"]=team.get("id"); e[f"{side}_starter_id"]=starter
        e[f"{side}_starter_name"]=(players.get(f"ID{starter}",{}).get("person",{}) or {}).get("fullName") if starter else None
        trs.append({"game_id":gid,"date":date,"season":season,"side":side,"team_id":team.get("id"),"team_name":team.get("name"),
          "runs":n(b.get("runs")),"hits":n(b.get("hits")),"doubles":n(b.get("doubles")),"triples":n(b.get("triples")),
          "home_runs":n(b.get("homeRuns")),"walks":n(b.get("baseOnBalls")),"strikeouts":n(b.get("strikeOuts")),"at_bats":n(b.get("atBats")),
          "pitching_earned_runs":n(p.get("earnedRuns")),"pitching_walks":n(p.get("baseOnBalls")),
          "pitching_strikeouts":n(p.get("strikeOuts")),"pitching_home_runs":n(p.get("homeRuns")),"errors":n(f.get("errors"))})
        for pid in ids:
            q=players.get(f"ID{pid}",{}) or {}; st=q.get("stats",{}).get("pitching",{}) or {}; person=q.get("person",{}) or {}
            prs.append({"game_id":gid,"date":date,"season":season,"side":side,"pitcher_id":pid,"pitcher_name":person.get("fullName"),
              "is_starter":int(pid==starter),"innings_pitched":st.get("inningsPitched"),"hits":n(st.get("hits")),
              "earned_runs":n(st.get("earnedRuns")),"walks":n(st.get("baseOnBalls")),"strikeouts":n(st.get("strikeOuts")),
              "home_runs":n(st.get("homeRuns")),"batters_faced":n(st.get("battersFaced")),"pitches":n(st.get("numberOfPitches"))})
    return e,prs,trs

def append(path,rows):
    if rows:
        pd.DataFrame(rows).to_csv(path,mode="a",header=not Path(path).exists(),index=False)

def main():
    a=argparse.ArgumentParser(); a.add_argument("--input",default="mlb_games_2018_present.csv"); a.add_argument("--outdir",default="mlb_enrichment")
    a.add_argument("--start-year",type=int,default=2018); a.add_argument("--end-year",type=int,default=2026); a.add_argument("--limit",type=int)
    a.add_argument("--no-verify-ssl",action="store_true"); a.add_argument("--sleep",type=float,default=.12); x=a.parse_args()
    out=Path(x.outdir); out.mkdir(exist_ok=True); ep=out/"mlb_game_enrichment.csv"; pp=out/"mlb_pitcher_game_logs.csv"; tp=out/"mlb_team_game_logs.csv"; fp=out/"failures.csv"
    g=pd.read_csv(x.input,low_memory=False); g["season"]=pd.to_numeric(g["season"],errors="coerce")
    g=g[(g.season>=x.start_year)&(g.season<=x.end_year)].drop_duplicates("game_id").sort_values(["season","date","game_id"])
    done=set(pd.read_csv(ep,usecols=["game_id"]).game_id.astype(int)) if ep.exists() else set(); g=g[~g.game_id.astype(int).isin(done)]
    if x.limit:g=g.head(x.limit)
    s=session(not x.no_verify_ssl); eb=[];pb=[];tb=[];fb=[]; total=len(g); print(f"{len(done):,} complete; {total:,} remaining")
    for i,r in enumerate(g.itertuples(index=False),1):
        try:
            z=s.get(URL.format(int(r.game_id)),timeout=45); z.raise_for_status(); e,p,t=extract(int(r.game_id),z.json()); eb.append(e);pb+=p;tb+=t
        except Exception as ex:fb.append({"game_id":r.game_id,"date":r.date,"error":repr(ex)})
        if i%100==0 or i==total:
            append(ep,eb);append(pp,pb);append(tp,tb);append(fp,fb);eb=[];pb=[];tb=[];fb=[];print(f"{i:,}/{total:,} ({i/total:.1%})")
        time.sleep(x.sleep)
    print("Done. Zip the mlb_enrichment folder and upload it back to ChatGPT.")
if __name__=="__main__":main()
