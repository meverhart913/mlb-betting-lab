"""Project today's starting-pitcher strikeouts and price sportsbook K props.

Research mode only. The live projection architecture is selected from historical
walk-forward results. Current implementation blends a specialized BF x K-rate
model with a direct nonlinear count model, then prices posted K lines with a
Poisson count distribution. All decisions remain NO BET until prospective
market validation establishes a durable edge.
"""
from __future__ import annotations

from datetime import date
from math import exp, floor
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb, specialized_features

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
    if pd.isna(x) or x == 0:
        return np.nan
    return 100.0/(x+100.0) if x > 0 else -x/(-x+100.0)


def poisson_cdf(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    term = exp(-mu)
    total = term
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return min(max(total, 0.0), 1.0)


def schedule(day: str) -> pd.DataFrame:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId":1,"date":day,"gameType":"R","hydrate":"probablePitcher"},
        timeout=30,
    )
    r.raise_for_status()
    rows=[]
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            for side, opp in (("away","home"),("home","away")):
                team = ((g.get("teams") or {}).get(side) or {}).get("team") or {}
                other = ((g.get("teams") or {}).get(opp) or {}).get("team") or {}
                pp = ((g.get("teams") or {}).get(side) or {}).get("probablePitcher") or {}
                if pp.get("id"):
                    rows.append({
                        "date":d.get("date"),"game_id":g.get("gamePk"),"side":side,
                        "pitcher_id":pp.get("id"),"pitcher_name":pp.get("fullName"),
                        "team_id":team.get("id"),"opponent_team_id":other.get("id"),
                        "away_team":(((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name"),
                        "home_team":(((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name"),
                    })
    return pd.DataFrame(rows)


def innings_to_outs(v):
    try:
        parts=str(v).split(".",1); a=int(parts[0]); b=int(parts[1]) if len(parts)>1 else 0
        return a*3+b if b in (0,1,2) else np.nan
    except Exception:
        return np.nan


def live_features(slate: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    p = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    t = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    p["pitcher_id"] = pd.to_numeric(p["pitcher_id"], errors="coerce")
    t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce")
    p["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(int)
    for c in ["strikeouts","walks","hits","earned_runs","home_runs","batters_faced","pitches"]:
        p[c]=pd.to_numeric(p[c], errors="coerce")
    for c in ["strikeouts","at_bats","walks","runs","hits","home_runs"]:
        t[c]=pd.to_numeric(t[c], errors="coerce")
    p["outs"] = p["innings_pitched"].map(innings_to_outs)

    rows=[]
    for r in slate.itertuples(index=False):
        row=r._asdict()
        q=p[(p.pitcher_id==r.pitcher_id)&(p.is_starter==1)&(p.date<target)].sort_values(["date","game_id"])
        row["days_rest"]=(target-q.date.iloc[-1]).days if len(q) else np.nan
        for w in (3,5,10):
            z=q.tail(w); minp=max(2,w//2)
            for c in ["strikeouts","walks","hits","earned_runs","home_runs","batters_faced","pitches","outs"]:
                row[f"sp_{c}_{w}"]=z[c].mean() if len(z)>=minp else np.nan
            if len(z)>=minp:
                bf=z.batters_faced.sum(); pitches=z.pitches.sum(); outs=z.outs.sum(); ks=z.strikeouts.sum()
                row[f"sp_k_rate_{w}"]=ks/bf if bf>0 else np.nan
                row[f"sp_k_per_100_pitches_{w}"]=100*ks/pitches if pitches>0 else np.nan
                row[f"sp_pitches_per_bf_{w}"]=pitches/bf if bf>0 else np.nan
                row[f"sp_outs_per_bf_{w}"]=outs/bf if bf>0 else np.nan

        o=t[(t.team_id==r.opponent_team_id)&(t.date<target)].sort_values(["date","game_id"])
        for w in (10,30):
            z=o.tail(w); minp=max(5,w//2)
            if len(z)>=minp:
                ab=z.at_bats.sum(); bb=z.walks.sum(); ks=z.strikeouts.sum(); hits=z.hits.sum(); pa=ab+bb
                row[f"opp_team_k_per_ab_{w}"]=ks/ab if ab>0 else np.nan
                row[f"opp_team_k_per_pa_{w}"]=ks/pa if pa>0 else np.nan
                row[f"opp_team_runs_{w}"]=z.runs.mean()
                row[f"opp_team_walk_per_pa_{w}"]=bb/pa if pa>0 else np.nan
                row[f"opp_team_hits_per_ab_{w}"]=hits/ab if ab>0 else np.nan
                row[f"opp_team_pa_per_game_{w}"]=(z.at_bats+z.walks).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def selected_component_weight() -> float:
    path = OUT / "pitcher_k_ensemble_summary.csv"
    if not path.exists():
        return 1.0
    s = pd.read_csv(path)
    if s.empty or "component_weight" not in s.columns:
        return 1.0
    metric = "mean_poisson_deviance" if "mean_poisson_deviance" in s.columns else "mean_mae"
    return float(s.sort_values([metric, "mean_mae" if "mean_mae" in s.columns else metric]).iloc[0]["component_weight"])


def main() -> None:
    day=date.today().isoformat()
    props_path=CURRENT/"pitcher_k_props.csv"
    if not props_path.exists():
        raise SystemExit("Missing pitcher_k_props.csv; run fetch_pitcher_k_props.py first.")
    props=pd.read_csv(props_path, low_memory=False)
    if props.empty:
        print("No current pitcher strikeout props available.")
        return

    hist=build_table()
    hist=hist[pd.to_numeric(hist["batters_faced"], errors="coerce").gt(0)].copy()
    hist["k_rate_target"]=(hist["strikeouts"]/hist["batters_faced"]).clip(0,0.7)
    bf_feats, kr_feats, all_feats = specialized_features(hist)
    target=pd.Timestamp(day)
    train=hist[hist.date<target]
    if len(train) < 1500:
        raise SystemExit("Not enough historical starts to fit live pitcher K ensemble.")

    bf=hgb("poisson", leaves=12, l2=4.0)
    kr=hgb("squared_error", leaves=12, l2=4.0)
    direct=hgb("poisson", leaves=15, l2=2.0)
    bf.fit(train[bf_feats], train["batters_faced"])
    kr.fit(train[kr_feats], train["k_rate_target"], m__sample_weight=train["batters_faced"])
    direct.fit(train[all_feats], train["strikeouts"])

    slate=schedule(day)
    if slate.empty:
        raise SystemExit(f"No probable MLB starters found for {day}.")
    live=live_features(slate, target)
    missing=[c for c in all_feats if c not in live.columns]
    if missing:
        raise ValueError("Live pitcher K feature builder is missing: " + ", ".join(missing))
    bf_hat=np.clip(bf.predict(live[bf_feats]),5,40)
    kr_hat=np.clip(kr.predict(live[kr_feats]),0.02,0.55)
    component=np.clip(bf_hat*kr_hat,0.05,None)
    direct_hat=np.clip(direct.predict(live[all_feats]),0.05,None)
    weight=selected_component_weight()
    mu=np.clip(weight*component+(1.0-weight)*direct_hat,0.05,None)
    live["projected_bf"]=bf_hat
    live["projected_k_rate"]=kr_hat
    live["component_k"]=component
    live["direct_k"]=direct_hat
    live["projected_k"]=mu
    live["component_weight"]=weight
    live["name_key"]=live.pitcher_name.map(norm_name)

    props["name_key"]=props.pitcher_name.map(norm_name)
    cols=["game_id","pitcher_id","pitcher_name","name_key","projected_bf","projected_k_rate","component_k","direct_k","projected_k","component_weight"]
    z=props.merge(live[cols].rename(columns={"pitcher_name":"mlb_pitcher_name"}), on="name_key", how="left")
    rows=[]
    for r in z.itertuples(index=False):
        if pd.isna(r.projected_k) or pd.isna(r.line):
            continue
        line=float(r.line); mu=float(r.projected_k); cut=floor(line)
        p_under_eq=poisson_cdf(cut,mu)
        if abs(line-round(line))<1e-9:
            k=int(round(line)); p_under=poisson_cdf(k-1,mu)
            p_push=max(poisson_cdf(k,mu)-poisson_cdf(k-1,mu),0)
            p_over=1-poisson_cdf(k,mu)
        else:
            p_under=p_under_eq; p_push=0.0; p_over=1-p_under
        imp_o=american_prob(r.over_price_median); imp_u=american_prob(r.under_price_median)
        den=(imp_o+imp_u) if pd.notna(imp_o) and pd.notna(imp_u) else np.nan
        market_over=imp_o/den if pd.notna(den) and den>0 else np.nan
        market_under=imp_u/den if pd.notna(den) and den>0 else np.nan
        over_edge=p_over-market_over if pd.notna(market_over) else np.nan
        under_edge=p_under-market_under if pd.notna(market_under) else np.nan
        side="OVER" if pd.notna(over_edge) and (pd.isna(under_edge) or over_edge>=under_edge) else "UNDER"
        edge=max(over_edge,under_edge) if pd.notna(over_edge) and pd.notna(under_edge) else np.nan
        rows.append({
            "date":r.date,"event_id":r.event_id,"snapshot_time_et":getattr(r,"snapshot_time_et",None),
            "pitcher_id":r.pitcher_id,"pitcher_name":r.mlb_pitcher_name or r.pitcher_name,
            "away_team":r.away_team,"home_team":r.home_team,"line":line,
            "projection_model":"walkforward_selected_ensemble","component_weight":r.component_weight,
            "projected_bf":r.projected_bf,"projected_k_rate":r.projected_k_rate,"component_k":r.component_k,
            "direct_k":r.direct_k,"projected_k":mu,"fair_over_prob":p_over,"fair_under_prob":p_under,
            "push_prob":p_push,"market_over_prob_no_vig":market_over,"market_under_prob_no_vig":market_under,
            "best_over_price":r.best_over_price,"best_over_sportsbook":r.best_over_sportsbook,
            "best_under_price":r.best_under_price,"best_under_sportsbook":r.best_under_sportsbook,
            "research_side":side,"model_market_edge":edge,
            "decision":"NO BET - prospective prop edge not validated",
        })
    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.sort_values(["model_market_edge"],ascending=False)
    out.to_csv(OUT/"pitcher_k_prop_predictions.csv",index=False)
    print(f"Selected component weight: {weight:.2f}")
    print(out.round(4).to_string(index=False))
    print("\nResearch mode only: all rows remain NO BET pending prospective price and CLV validation.")

if __name__=="__main__":
    main()
