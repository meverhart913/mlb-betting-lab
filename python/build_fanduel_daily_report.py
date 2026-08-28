"""Build a compact mobile-friendly FanDuel pitcher-K paper report.

This is presentation only. It does not alter selection rules or the immutable
paper ledger.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TODAY = ROOT / "outputs/fanduel_pitcher_k_paper_today.csv"
PROJ = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"
OUT_CSV = ROOT / "outputs/fanduel_pitcher_k_daily_report.csv"
OUT_MD = ROOT / "outputs/fanduel_pitcher_k_daily_report.md"


def fmt_odds(v):
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x): return ""
    return f"+{int(x)}" if x > 0 else str(int(x))


def risk_flags(row) -> str:
    frozen=str(row.get("failure_regime_flags", "") or "").strip()
    flags=[] if frozen in {"", "nan", "NONE"} else [x for x in frozen.split(",") if x]
    model=str(row.get("model_version", ""))
    if "fallback" in model and "LINEUP_NOT_CONFIRMED" not in flags:
        flags.append("LINEUP_NOT_CONFIRMED")
    cov=pd.to_numeric(row.get("lineup_match_coverage"), errors="coerce")
    if pd.notna(cov) and cov < .80 and "LOW_LINEUP_COVERAGE" not in flags:
        flags.append("LOW_LINEUP_COVERAGE")
    bf=pd.to_numeric(row.get("projected_bf"), errors="coerce")
    if pd.notna(bf) and (bf < 18 or bf > 30) and "BF_EXTREME" not in flags:
        flags.append("BF_EXTREME")
    edge=pd.to_numeric(row.get("model_market_edge"), errors="coerce")
    if pd.notna(edge) and abs(edge) >= .15:
        flags.append("LARGE_MODEL_MARKET_GAP")
    return ",".join(dict.fromkeys(flags)) if flags else "NONE"


def main():
    OUT_CSV.parent.mkdir(exist_ok=True)
    try:
        s=pd.read_csv(TODAY, low_memory=False) if TODAY.exists() and TODAY.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        s=pd.DataFrame()
    try:
        p=pd.read_csv(PROJ, low_memory=False) if PROJ.exists() and PROJ.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        p=pd.DataFrame()

    if s.empty:
        if p.empty:
            out=pd.DataFrame(columns=["pitcher","status","model","projected_k","notes"])
        else:
            notes=[]
            for _,r in p.iterrows():
                base="lineup not confirmed; V2.1 Statcast fallback" if "fallback" in str(r.get("model_version","")) else "V2.2 confirmed-lineup model"
                f=str(r.get("failure_regime_flags","NONE"))
                notes.append(base + (f"; flags: {f}" if f not in {"NONE","nan",""} else ""))
            out=pd.DataFrame({
                "pitcher":p.get("pitcher_name", ""),
                "status":"WAITING_FOR_ELIGIBLE_FANDUEL_MARKET",
                "model":p.get("model_version", ""),
                "projected_k":pd.to_numeric(p.get("projected_k"), errors="coerce").round(2),
                "notes":notes,
            })
        out.to_csv(OUT_CSV,index=False)
        lines=["# FanDuel Pitcher K Paper Report", "", "No eligible FanDuel paper selections in this snapshot.", ""]
        if not out.empty:
            lines.append("## Projection status")
            for r in out.head(40).itertuples(index=False):
                lines.append(f"- {r.pitcher}: {r.projected_k} K, {r.model}, {r.status}. {r.notes}")
        OUT_MD.write_text("\n".join(lines)+"\n",encoding="utf-8")
        print(f"No frozen selections; report contains {len(out)} projection-status rows.")
        return

    rows=[]
    for _,r in s.iterrows():
        model=str(r.get("model_version", ""))
        lineup=str(r.get("lineup_status", "CONFIRMED" if model.startswith("v22_") else "NOT_CONFIRMED"))
        ev=pd.to_numeric(r.get("expected_profit_per_unit"), errors="coerce")
        edge=pd.to_numeric(r.get("model_market_edge"), errors="coerce")
        prob=pd.to_numeric(r.get("model_win_prob"), errors="coerce")
        imp=pd.to_numeric(r.get("fanduel_implied_prob"), errors="coerce")
        rows.append({
            "pitcher":r.get("pitcher_name"),
            "side":r.get("side"),
            "line":r.get("line"),
            "fanduel_price":fmt_odds(r.get("fanduel_price")),
            "projected_k":round(float(r.get("projected_k")),2) if pd.notna(r.get("projected_k")) else np.nan,
            "model_probability":round(float(prob)*100,1) if pd.notna(prob) else np.nan,
            "fanduel_implied_probability":round(float(imp)*100,1) if pd.notna(imp) else np.nan,
            "edge_pct_pts":round(float(edge)*100,1) if pd.notna(edge) else np.nan,
            "ev_per_unit":round(float(ev),3) if pd.notna(ev) else np.nan,
            "model":model,
            "lineup":lineup,
            "minutes_to_start":round(float(r.get("minutes_to_start")),0) if pd.notna(r.get("minutes_to_start")) else np.nan,
            "status":r.get("paper_status", "PAPER"),
            "risk_flags":risk_flags(r),
            "model_time":r.get("model_generated_at_et"),
            "quote_time_utc":r.get("collected_at_utc"),
        })
    out=pd.DataFrame(rows).sort_values("ev_per_unit",ascending=False)
    out.to_csv(OUT_CSV,index=False)

    lines=["# FanDuel Pitcher K Paper Report","","Research only. No wagers are placed.",""]
    for r in out.itertuples(index=False):
        lines += [
            f"## {r.pitcher}: {r.side} {r.line} ({r.fanduel_price})",
            f"Projected K: **{r.projected_k}** | Model win: **{r.model_probability}%** | FanDuel implied: **{r.fanduel_implied_probability}%**",
            f"Edge: **{r.edge_pct_pts} pts** | EV: **{r.ev_per_unit} units** | {r.status}",
            f"Model: {r.model} | Lineup: {r.lineup} | Risk flags: {r.risk_flags}",
            f"Minutes to start: {r.minutes_to_start}",
            "",
        ]
    OUT_MD.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out.to_string(index=False))

if __name__=="__main__": main()
