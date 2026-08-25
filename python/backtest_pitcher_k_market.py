"""Backtest V2.2 pitcher-K projections against normalized historical prop quotes.

Expected input: data/market/pitcher_k_historical_raw.csv with one row per
sportsbook/side quote and columns date, pitcher_name, line, side, price,
sportsbook, snapshot_time_et. The script removes vig within each sportsbook,
forms a median no-vig market consensus, chooses the model's higher-edge side,
and grades flat-stake ROI at multiple predeclared edge thresholds.

This is evaluation only. It never places bets.
"""
from __future__ import annotations

from math import exp, floor
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRED = OUT / "pitcher_k_batter_hand_predictions.csv"
MARKET = ROOT / "data" / "market" / "pitcher_k_historical_raw.csv"
GRADED = OUT / "pitcher_k_market_graded.csv"
SUMMARY = OUT / "pitcher_k_market_threshold_summary.csv"
BY_YEAR = OUT / "pitcher_k_market_year_summary.csv"
THRESHOLDS = (0.00, 0.025, 0.05, 0.075, 0.10)


def norm_name(v) -> str:
    x = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def implied_prob(price):
    x = pd.to_numeric(price, errors="coerce")
    if pd.isna(x) or x == 0:
        return np.nan
    return 100.0 / (x + 100.0) if x > 0 else -x / (-x + 100.0)


def profit_for_win(price: float) -> float:
    return price / 100.0 if price > 0 else 100.0 / (-price)


def poisson_cdf(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    term = exp(-mu)
    total = term
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return float(min(max(total, 0.0), 1.0))


def fair_probs(line: float, mu: float):
    if abs(line - round(line)) < 1e-9:
        k = int(round(line))
        p_under = poisson_cdf(k - 1, mu)
        p_push = max(poisson_cdf(k, mu) - poisson_cdf(k - 1, mu), 0.0)
        p_over = 1.0 - poisson_cdf(k, mu)
    else:
        cut = floor(line)
        p_under = poisson_cdf(cut, mu)
        p_push = 0.0
        p_over = 1.0 - p_under
    return p_over, p_under, p_push


def prepare_market(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "pitcher_name", "line", "side", "price", "sportsbook"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("Historical market file missing: " + ", ".join(sorted(missing)))
    z = raw.copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.normalize()
    z["line"] = pd.to_numeric(z["line"], errors="coerce")
    z["price"] = pd.to_numeric(z["price"], errors="coerce")
    z["side"] = z["side"].astype(str).str.lower()
    z["name_key"] = z["pitcher_name"].map(norm_name)
    z = z[z.side.isin(["over", "under"]) & z.line.notna() & z.price.notna()].copy()
    if "snapshot_time_et" in z.columns:
        z["snapshot_time_et"] = pd.to_datetime(z["snapshot_time_et"], errors="coerce")

    idx = ["date", "name_key", "line", "sportsbook"]
    if "snapshot_time_et" in z.columns:
        idx.append("snapshot_time_et")
    wide = z.pivot_table(index=idx, columns="side", values="price", aggfunc="last").reset_index()
    wide["over_imp"] = wide.get("over", pd.Series(index=wide.index, dtype=float)).map(implied_prob)
    wide["under_imp"] = wide.get("under", pd.Series(index=wide.index, dtype=float)).map(implied_prob)
    denom = wide.over_imp + wide.under_imp
    wide["over_nv"] = np.where(denom > 0, wide.over_imp / denom, np.nan)
    wide["under_nv"] = np.where(denom > 0, wide.under_imp / denom, np.nan)

    group = ["date", "name_key", "line"]
    rows = []
    for keys, g in wide.groupby(group, dropna=False):
        over = pd.to_numeric(g.get("over"), errors="coerce")
        under = pd.to_numeric(g.get("under"), errors="coerce")
        oi = over.idxmax() if over.notna().any() else None
        ui = under.idxmax() if under.notna().any() else None
        rows.append({
            "date": keys[0],
            "name_key": keys[1],
            "line": float(keys[2]),
            "market_over_prob_no_vig": float(pd.to_numeric(g.over_nv, errors="coerce").median()),
            "market_under_prob_no_vig": float(pd.to_numeric(g.under_nv, errors="coerce").median()),
            "best_over_price": float(over.loc[oi]) if oi is not None else np.nan,
            "best_over_sportsbook": str(g.loc[oi, "sportsbook"]) if oi is not None else None,
            "best_under_price": float(under.loc[ui]) if ui is not None else np.nan,
            "best_under_sportsbook": str(g.loc[ui, "sportsbook"]) if ui is not None else None,
            "sportsbook_count": int(g.sportsbook.nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not PRED.exists():
        raise SystemExit("Missing V2.2 prediction export; run test_pitcher_k_batter_hand_features.py first.")
    if not MARKET.exists():
        raise SystemExit(
            "Missing data/market/pitcher_k_historical_raw.csv. Historical prop quotes must be collected before ROI can be calculated."
        )

    p = pd.read_csv(PRED, low_memory=False)
    p = p[p.model.eq("v22_lineup_all")].copy()
    p["date"] = pd.to_datetime(p.date, errors="coerce").dt.normalize()
    p["name_key"] = p.pitcher_name.map(norm_name)
    p["projected_k"] = pd.to_numeric(p.projected_k, errors="coerce")
    p["actual_k"] = pd.to_numeric(p.strikeouts, errors="coerce")
    p = p.dropna(subset=["date", "name_key", "projected_k", "actual_k"])

    market = prepare_market(pd.read_csv(MARKET, low_memory=False))
    g = market.merge(
        p[["season", "date", "game_id", "pitcher_id", "pitcher_name", "name_key", "projected_k", "actual_k"]],
        on=["date", "name_key"],
        how="inner",
    )
    rows = []
    for r in g.itertuples(index=False):
        pov, pun, ppush = fair_probs(float(r.line), float(r.projected_k))
        moe = pov - float(r.market_over_prob_no_vig)
        mue = pun - float(r.market_under_prob_no_vig)
        side = "OVER" if moe >= mue else "UNDER"
        edge = max(moe, mue)
        price = float(r.best_over_price if side == "OVER" else r.best_under_price)
        if not np.isfinite(price):
            continue
        if r.actual_k == r.line:
            result, profit = "PUSH", 0.0
        elif (side == "OVER" and r.actual_k > r.line) or (side == "UNDER" and r.actual_k < r.line):
            result, profit = "WIN", profit_for_win(price)
        else:
            result, profit = "LOSS", -1.0
        rows.append({
            "season": int(r.season), "date": r.date, "game_id": r.game_id,
            "pitcher_id": r.pitcher_id, "pitcher_name": r.pitcher_name,
            "line": r.line, "projected_k": r.projected_k, "actual_k": r.actual_k,
            "fair_over_prob": pov, "fair_under_prob": pun, "push_prob": ppush,
            "market_over_prob_no_vig": r.market_over_prob_no_vig,
            "market_under_prob_no_vig": r.market_under_prob_no_vig,
            "research_side": side, "model_market_edge": edge,
            "price": price,
            "sportsbook": r.best_over_sportsbook if side == "OVER" else r.best_under_sportsbook,
            "sportsbook_count": r.sportsbook_count, "result": result, "flat_profit_units": profit,
        })
    graded = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    graded.to_csv(GRADED, index=False)
    if graded.empty:
        raise SystemExit("No V2.2 predictions matched the supplied historical prop quotes.")

    summary = []
    for threshold in THRESHOLDS:
        x = graded[graded.model_market_edge >= threshold].copy()
        wins = int(x.result.eq("WIN").sum())
        losses = int(x.result.eq("LOSS").sum())
        pushes = int(x.result.eq("PUSH").sum())
        staked = wins + losses
        profit = float(x.flat_profit_units.sum())
        summary.append({
            "min_edge": threshold, "bets": staked, "pushes": pushes, "wins": wins, "losses": losses,
            "win_rate_ex_push": wins / staked if staked else np.nan,
            "profit_units": profit, "roi": profit / staked if staked else np.nan,
            "mean_model_edge": float(x.model_market_edge.mean()) if len(x) else np.nan,
        })
    s = pd.DataFrame(summary)
    s.to_csv(SUMMARY, index=False)

    yearly = graded.groupby("season", as_index=False).agg(
        matched_lines=("line", "size"), mean_edge=("model_market_edge", "mean"),
        profit_units=("flat_profit_units", "sum"),
    )
    yearly["roi_all_candidates"] = yearly.profit_units / yearly.matched_lines
    yearly.to_csv(BY_YEAR, index=False)
    print("V2.2 HISTORICAL MARKET THRESHOLD TEST")
    print(s.round(5).to_string(index=False))
    print("\nBY SEASON")
    print(yearly.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
