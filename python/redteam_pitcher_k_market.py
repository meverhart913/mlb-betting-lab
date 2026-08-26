"""Red-team V2.2 pitcher-K market results.

Recomputes market edges using only conventional regulated US sportsbooks and
reports both line-shopping and FanDuel-only results. Headline results are reduced
to one actionable wager per pitcher start so correlated alt-lines are not counted
as independent bets.
"""
from __future__ import annotations

from math import exp, floor
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "market" / "pitcher_k_historical_raw.csv"
GRADED = ROOT / "outputs" / "pitcher_k_market_graded.csv"
OUT = ROOT / "outputs"
REGULATED = {"fanduel", "draftkings", "betmgm", "betrivers"}
THRESHOLDS = (0.00, 0.025, 0.05, 0.075, 0.10)


def norm_name(v):
    x = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def implied_prob(price):
    x = pd.to_numeric(price, errors="coerce")
    if pd.isna(x) or x == 0:
        return np.nan
    return 100/(x+100) if x > 0 else -x/(-x+100)


def profit_for_win(price):
    return price/100 if price > 0 else 100/(-price)


def poisson_cdf(k, mu):
    if k < 0: return 0.0
    term = exp(-mu); total = term
    for i in range(1, k+1):
        term *= mu/i; total += term
    return float(min(max(total, 0.0), 1.0))


def fair_probs(line, mu):
    if abs(line-round(line)) < 1e-9:
        k = int(round(line))
        pu = poisson_cdf(k-1, mu)
        pp = max(poisson_cdf(k, mu)-poisson_cdf(k-1, mu), 0.0)
        po = 1-poisson_cdf(k, mu)
    else:
        cut = floor(line); pu = poisson_cdf(cut, mu); pp = 0.0; po = 1-pu
    return po, pu, pp


def build_market(raw, mode):
    z = raw.copy()
    z["date"] = pd.to_datetime(z.date, errors="coerce").dt.normalize()
    z["name_key"] = z.pitcher_name.map(norm_name)
    z["sportsbook"] = z.sportsbook.astype(str).str.lower()
    z["side"] = z.side.astype(str).str.lower()
    z["line"] = pd.to_numeric(z.line, errors="coerce")
    z["price"] = pd.to_numeric(z.price, errors="coerce")
    if mode == "fanduel": z = z[z.sportsbook.eq("fanduel")]
    else: z = z[z.sportsbook.isin(REGULATED)]
    z = z[z.side.isin(["over","under"]) & z.line.notna() & z.price.notna()]
    idx = ["date","name_key","line","sportsbook"]
    w = z.pivot_table(index=idx, columns="side", values="price", aggfunc="last").reset_index()
    w["over_imp"] = w.get("over", pd.Series(index=w.index,dtype=float)).map(implied_prob)
    w["under_imp"] = w.get("under", pd.Series(index=w.index,dtype=float)).map(implied_prob)
    d = w.over_imp + w.under_imp
    w["over_nv"] = np.where(d>0,w.over_imp/d,np.nan)
    w["under_nv"] = np.where(d>0,w.under_imp/d,np.nan)
    rows=[]
    for keys,g in w.groupby(["date","name_key","line"],dropna=False):
        over=pd.to_numeric(g.get("over"),errors="coerce"); under=pd.to_numeric(g.get("under"),errors="coerce")
        oi=over.idxmax() if over.notna().any() else None; ui=under.idxmax() if under.notna().any() else None
        rows.append({"date":keys[0],"name_key":keys[1],"line":keys[2],
          "market_over_prob_no_vig":float(pd.to_numeric(g.over_nv,errors="coerce").median()),
          "market_under_prob_no_vig":float(pd.to_numeric(g.under_nv,errors="coerce").median()),
          "best_over_price":float(over.loc[oi]) if oi is not None else np.nan,
          "best_over_sportsbook":str(g.loc[oi,"sportsbook"]) if oi is not None else None,
          "best_under_price":float(under.loc[ui]) if ui is not None else np.nan,
          "best_under_sportsbook":str(g.loc[ui,"sportsbook"]) if ui is not None else None,
          "sportsbook_count":int(g.sportsbook.nunique())})
    return pd.DataFrame(rows)


def evaluate(raw, base, mode):
    market=build_market(raw,mode)
    g=market.merge(base,on=["date","name_key","line"],how="inner")
    rows=[]
    for r in g.itertuples(index=False):
        po,pu,pp=fair_probs(float(r.line),float(r.projected_k))
        oe=po-float(r.market_over_prob_no_vig); ue=pu-float(r.market_under_prob_no_vig)
        side="OVER" if oe>=ue else "UNDER"; edge=max(oe,ue)
        price=float(r.best_over_price if side=="OVER" else r.best_under_price)
        book=r.best_over_sportsbook if side=="OVER" else r.best_under_sportsbook
        if not np.isfinite(price): continue
        if r.actual_k==r.line: result="PUSH"; profit=0.0
        elif (side=="OVER" and r.actual_k>r.line) or (side=="UNDER" and r.actual_k<r.line): result="WIN"; profit=profit_for_win(price)
        else: result="LOSS"; profit=-1.0
        rows.append({"mode":mode,"date":r.date,"pitcher_name":r.pitcher_name,"game_id":r.game_id,"line":r.line,
          "projected_k":r.projected_k,"actual_k":r.actual_k,"research_side":side,"model_market_edge":edge,
          "price":price,"sportsbook":book,"sportsbook_count":r.sportsbook_count,"result":result,"flat_profit_units":profit})
    return pd.DataFrame(rows)


def summarize(df, mode):
    rows=[]
    for t in THRESHOLDS:
        cand=df[df.model_market_edge>=t].copy()
        # One wager per pitcher start: choose the highest model edge among available conventional lines.
        one=(cand.sort_values(["model_market_edge"],ascending=False)
                 .drop_duplicates(["date","pitcher_name"],keep="first"))
        wins=int(one.result.eq("WIN").sum()); losses=int(one.result.eq("LOSS").sum()); pushes=int(one.result.eq("PUSH").sum())
        staked=wins+losses; profit=float(one.flat_profit_units.sum())
        rows.append({"mode":mode,"min_edge":t,"independent_bets":staked,"pushes":pushes,"wins":wins,"losses":losses,
          "win_rate_ex_push":wins/staked if staked else np.nan,"profit_units":profit,"roi":profit/staked if staked else np.nan,
          "raw_alt_line_candidates":len(cand),"unique_pitcher_starts":one[["date","pitcher_name"]].drop_duplicates().shape[0]})
    return pd.DataFrame(rows)


def main():
    raw=pd.read_csv(RAW,low_memory=False)
    old=pd.read_csv(GRADED,low_memory=False)
    old["date"]=pd.to_datetime(old.date,errors="coerce").dt.normalize(); old["name_key"]=old.pitcher_name.map(norm_name)
    base=(old[["date","name_key","line","game_id","pitcher_name","projected_k","actual_k"]]
          .drop_duplicates(["date","name_key","line"]))
    audits=[]; summaries=[]
    for mode in ("regulated_line_shop","fanduel"):
        d=evaluate(raw,base,mode); audits.append(d); summaries.append(summarize(d,mode))
    audit=pd.concat(audits,ignore_index=True,sort=False); summary=pd.concat(summaries,ignore_index=True,sort=False)
    OUT.mkdir(exist_ok=True)
    audit.to_csv(OUT/"pitcher_k_market_redteam_graded.csv",index=False)
    summary.to_csv(OUT/"pitcher_k_market_redteam_summary.csv",index=False)
    book_counts=raw.assign(sportsbook=raw.sportsbook.astype(str).str.lower()).sportsbook.value_counts().rename_axis("sportsbook").reset_index(name="quote_rows")
    book_counts.to_csv(OUT/"pitcher_k_market_book_counts.csv",index=False)
    print("RED-TEAM MARKET VALIDATION")
    print(summary.round(5).to_string(index=False))
    print("\nRAW BOOK COUNTS")
    print(book_counts.to_string(index=False))

if __name__ == "__main__": main()
