"""Project today's starting-pitcher strikeouts and price sportsbook K props.

Research mode only. Fair probabilities come from a Poisson count distribution
around the selected walk-forward model's projected strikeout mean. A market edge
is reported, but every decision remains NO BET until prospective price/CLV
validation establishes a profitable rule.
"""
from __future__ import annotations

from datetime import date
from math import exp, floor, factorial
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from build_pitcher_k_model import build_table, candidate_models, feature_cols

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def norm_name(s) -> str:
    x = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def american_prob(v):
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x == 0: return np.nan
    return 100.0/(x+100.0) if x > 0 else -x/(-x+100.0)


def poisson_cdf(k: int, mu: float) -> float:
    if k < 0: return 0.0
    term = exp(-mu)
    total = term
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return min(max(total, 0.0), 1.0)


def schedule(day: str) -> pd.DataFrame:
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={"sportId":1,"date":day,"gameType":"R","hydrate":"probablePitcher"}, timeout=30)
    r.raise_for_status()
    rows=[]
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            for side, opp in (("away","home"),("home","away")):
                team = ((g.get("teams") or {}).get(side) or {}).get("team") or {}
                other = ((g.get("teams") or {}).get(opp) or {}).get("team") or {}
                pp = ((g.get("teams") or {}).get(side) or {}).get("probablePitcher") or {}
                if pp.get("id"):
                    rows.append({"date":d.get("date"),"game_id":g.get("gamePk"),"side":side,"pitcher_id":pp.get("id"),"pitcher_name":pp.get("fullName"),"team_id":team.get("id"),"opponent_team_id":other.get("id"),"away_team":(((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name"),"home_team":(((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name")})
    return pd.DataFrame(rows)


def live_features(hist: pd.DataFrame, slate: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    # Reuse the already-built historical feature rows. For live games, construct
    # each feature directly from raw pre-target logs using the same definitions.
    p = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    t = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    p["date"] = pd.to_datetime(p["date"], errors="coerce"); t["date"] = pd.to_datetime(t["date"], errors="coerce")
    p["pitcher_id"] = pd.to_numeric(p["pitcher_id"], errors="coerce"); t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce")
    p["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(int)
    for c in ["strikeouts","walks","hits","earned_runs","home_runs","batters_faced","pitches"]: p[c]=pd.to_numeric(p[c], errors="coerce")
    for c in ["strikeouts","at_bats","walks","runs","hits","home_runs"]: t[c]=pd.to_numeric(t[c], errors="coerce")
    rows=[]
    for r in slate.itertuples(index=False):
        row=r._asdict(); q=p[(p.pitcher_id==r.pitcher_id)&(p.is_starter==1)&(p.date<target)].sort_values(["date","game_id"])
        row["days_rest"]=(target-q.date.iloc[-1]).days if len(q) else np.nan
        for w in (3,5,10):
            z=q.tail(w)
            for c in ["strikeouts","walks","hits","earned_runs","home_runs","batters_faced","pitches"]: row[f"sp_{c}_{w}"]=z[c].mean() if len(z)>=max(2,w//2) else np.nan
            outs=[]
            for v in z.get("innings_pitched", pd.Series(dtype=object)):
                try:
                    a,b=(str(v).split(".",1)+["0"])[:2]; outs.append(int(a)*3+int(b))
                except Exception: pass
            row[f"sp_outs_{w}"]=np.mean(outs) if len(outs)>=max(2,w//2) else np.nan
            if len(z)>=max(2,w//2):
                row[f"sp_k_rate_{w}"]=z.strikeouts.sum()/z.batters_faced.sum() if z.batters_faced.sum()>0 else np.nan
                row[f"sp_k_per_100_pitches_{w}"]=100*z.strikeouts.sum()/z.pitches.sum() if z.pitches.sum()>0 else np.nan
        o=t[(t.team_id==r.opponent_team_id)&(t.date<target)].sort_values(["date","game_id"])
        for w in (10,30):
            z=o.tail(w); minp=max(5,w//2)
            if len(z)>=minp:
                row[f"opp_team_k_per_ab_{w}"]=z.strikeouts.sum()/z.at_bats.sum() if z.at_bats.sum()>0 else np.nan
                denom=z.at_bats.sum()+z.walks.sum(); row[f"opp_team_k_per_pa_{w}"]=z.strikeouts.sum()/denom if denom>0 else np.nan
                row[f"opp_team_runs_{w}"]=z.runs.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    day=date.today().isoformat(); props_path=CURRENT/"pitcher_k_props.csv"
    if not props_path.exists(): raise SystemExit("Missing pitcher_k_props.csv; run fetch_pitcher_k_props.py first.")
    props=pd.read_csv(props_path, low_memory=False)
    if props.empty:
        print("No current pitcher strikeout props available."); return
    hist=build_table(); feats=feature_cols(hist)
    summary_path=OUT/"pitcher_k_model_summary.csv"
    model_name="poisson_glm"
    if summary_path.exists():
        s=pd.read_csv(summary_path)
        if not s.empty and "model" in s: model_name=str(s.sort_values("mean_poisson_deviance").iloc[0]["model"])
    model=candidate_models()[model_name]
    target=pd.Timestamp(day); train=hist[hist.date<target]
    model.fit(train[feats], train["strikeouts"])
    slate=schedule(day)
    if slate.empty: raise SystemExit(f"No probable MLB starters found for {day}.")
    live=live_features(hist, slate, target)
    mu=np.clip(model.predict(live[feats]),0.05,None); live["projected_k"]=mu; live["name_key"]=live.pitcher_name.map(norm_name)
    props["name_key"]=props.pitcher_name.map(norm_name)
    z=props.merge(live[["game_id","pitcher_id","pitcher_name","name_key","projected_k"]].rename(columns={"pitcher_name":"mlb_pitcher_name"}), on="name_key", how="left")
    rows=[]
    for r in z.itertuples(index=False):
        if pd.isna(r.projected_k) or pd.isna(r.line): continue
        line=float(r.line); mu=float(r.projected_k); cut=floor(line)
        p_under_eq=poisson_cdf(cut,mu)
        if abs(line-round(line))<1e-9:
            k=int(round(line)); p_under=poisson_cdf(k-1,mu); p_push=max(poisson_cdf(k,mu)-poisson_cdf(k-1,mu),0); p_over=1-poisson_cdf(k,mu)
        else:
            p_under=p_under_eq; p_push=0.0; p_over=1-p_under
        imp_o=american_prob(r.over_price_median); imp_u=american_prob(r.under_price_median)
        den=(imp_o+imp_u) if pd.notna(imp_o) and pd.notna(imp_u) else np.nan
        market_over=imp_o/den if pd.notna(den) and den>0 else np.nan
        market_under=imp_u/den if pd.notna(den) and den>0 else np.nan
        over_edge=p_over-market_over if pd.notna(market_over) else np.nan; under_edge=p_under-market_under if pd.notna(market_under) else np.nan
        side="OVER" if pd.notna(over_edge) and (pd.isna(under_edge) or over_edge>=under_edge) else "UNDER"
        edge=max(over_edge,under_edge) if pd.notna(over_edge) and pd.notna(under_edge) else np.nan
        rows.append({"date":r.date,"event_id":r.event_id,"pitcher_id":r.pitcher_id,"pitcher_name":r.mlb_pitcher_name or r.pitcher_name,"away_team":r.away_team,"home_team":r.home_team,"line":line,"projection_model":model_name,"projected_k":mu,"fair_over_prob":p_over,"fair_under_prob":p_under,"push_prob":p_push,"market_over_prob_no_vig":market_over,"market_under_prob_no_vig":market_under,"best_over_price":r.best_over_price,"best_over_sportsbook":r.best_over_sportsbook,"best_under_price":r.best_under_price,"best_under_sportsbook":r.best_under_sportsbook,"research_side":side,"model_market_edge":edge,"decision":"NO BET - prospective prop edge not validated"})
    out=pd.DataFrame(rows).sort_values(["model_market_edge"],ascending=False)
    out.to_csv(OUT/"pitcher_k_prop_predictions.csv",index=False)
    print(out.round(4).to_string(index=False)); print("\nResearch mode only: all rows remain NO BET pending prospective price and CLV validation.")

if __name__=="__main__": main()
