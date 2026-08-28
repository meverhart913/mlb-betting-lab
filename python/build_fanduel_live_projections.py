"""Build today's pitcher-K projections before collecting the decision quote.

This ordering protects timing integrity: confirmed-lineup/Statcast inputs and the
model projection are created first, then a later workflow step collects the
FanDuel quote used for the paper decision. Failure-regime fields are diagnostic
only and do not alter the frozen selection rule.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from run_fanduel_hybrid_paper import hybrid_predictions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = DATA / "features"
OUT = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"


def add_diagnostics(p: pd.DataFrame, day: str) -> pd.DataFrame:
    z=p.copy(); target=pd.Timestamp(day)
    z['pitcher_id']=pd.to_numeric(z.pitcher_id,errors='coerce')

    logs=pd.read_csv(DATA/'mlb_pitcher_game_logs.csv',low_memory=False)
    logs['pitcher_id']=pd.to_numeric(logs.pitcher_id,errors='coerce')
    logs['date']=pd.to_datetime(logs.date,errors='coerce')
    logs['is_starter']=pd.to_numeric(logs.is_starter,errors='coerce').fillna(0).astype(int)
    prior=(logs[(logs.is_starter.eq(1)) & logs.date.lt(target)]
           .groupby('pitcher_id').agg(prior_start_count=('game_id','count'),last_start_date=('date','max')).reset_index())
    prior['days_rest_diagnostic']=(target-prior.last_start_date).dt.days
    z=z.merge(prior,on='pitcher_id',how='left')

    parts=[]
    for f in FEATURES.glob('statcast_pitcher_daily*.csv'):
        try:
            q=pd.read_csv(f,usecols=['game_date','pitcher_id'])
        except Exception:
            continue
        q['game_date']=pd.to_datetime(q.game_date,errors='coerce'); q['pitcher_id']=pd.to_numeric(q.pitcher_id,errors='coerce')
        parts.append(q[q.game_date.lt(target)])
    if parts:
        sc=pd.concat(parts,ignore_index=True).drop_duplicates(['game_date','pitcher_id'])
        cnt=sc.groupby('pitcher_id').size().rename('statcast_appearance_count').reset_index()
        z=z.merge(cnt,on='pitcher_id',how='left')
    else:
        z['statcast_appearance_count']=np.nan

    z['lineup_status']=np.where(z.model_version.astype(str).str.startswith('v22_'),'CONFIRMED_9','NOT_CONFIRMED')
    flags=[]
    for _,r in z.iterrows():
        f=[]
        starts=pd.to_numeric(r.get('prior_start_count'),errors='coerce')
        scn=pd.to_numeric(r.get('statcast_appearance_count'),errors='coerce')
        rest=pd.to_numeric(r.get('days_rest_diagnostic'),errors='coerce')
        cov=pd.to_numeric(r.get('lineup_match_coverage'),errors='coerce')
        bf=pd.to_numeric(r.get('projected_bf'),errors='coerce')
        if pd.isna(starts) or starts < 5: f.append('ROOKIE_LOW_START_HISTORY')
        elif starts < 10: f.append('LIMITED_START_HISTORY')
        if pd.isna(scn) or scn < 3: f.append('LOW_STATCAST_HISTORY')
        if pd.notna(rest) and (rest < 4 or rest > 7): f.append('UNUSUAL_REST')
        if r.get('lineup_status')!='CONFIRMED_9': f.append('LINEUP_NOT_CONFIRMED')
        if pd.notna(cov) and cov < .80: f.append('LOW_LINEUP_COVERAGE')
        if pd.notna(bf) and (bf < 18 or bf > 30): f.append('BF_EXTREME')
        flags.append(','.join(f) if f else 'NONE')
    z['failure_regime_flags']=flags
    return z


def main() -> None:
    day = date.today().isoformat()
    projections = hybrid_predictions(day)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if projections.empty:
        pd.DataFrame().to_csv(OUT, index=False)
        print(f"No live pitcher-K projections available for {day}.")
        return
    projections=add_diagnostics(projections,day)
    cols = [c for c in [
        "date", "game_id", "pitcher_id", "pitcher_name", "away_team", "home_team",
        "name_key", "projected_k", "projected_bf", "projected_k_rate", "days_rest",
        "prior_start_count", "statcast_appearance_count", "days_rest_diagnostic",
        "lineup_match_coverage", "lineup_status", "failure_regime_flags",
        "model_version", "model_generated_at_et"
    ] if c in projections.columns]
    projections[cols].to_csv(OUT, index=False)
    show=[c for c in ['pitcher_name','projected_k','model_version','prior_start_count','statcast_appearance_count','lineup_status','failure_regime_flags'] if c in projections]
    print(projections[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
