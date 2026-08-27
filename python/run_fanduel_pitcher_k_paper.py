"""Generate FanDuel-only pitcher-K paper selections from the free PropLine snapshot.

Research/paper mode only. This intentionally does not place bets.

Rules are frozen in docs/FANDUEL_PROSPECTIVE_PROTOCOL.md:
- FanDuel only
- retain every available main/alt strikeout line and both sides
- compute model probability and EV at the actual FanDuel American price
- freeze at most one highest-EV paper selection per pitcher start
- retain all candidates for audit/calibration
"""
from __future__ import annotations

from datetime import date, datetime
from math import floor
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import unicodedata

import numpy as np
import pandas as pd

from build_pitcher_k_model import build_table
from run_pitcher_k_props import live_features, poisson_cdf, schedule, selected_component_weight
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MARKET = DATA / "market" / "pitcher_k_historical_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CANDIDATES = OUT / "fanduel_pitcher_k_candidates.csv"
SELECTIONS = OUT / "fanduel_pitcher_k_paper_selections.csv"


def norm_name(s) -> str:
    x = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def implied_prob(price: float) -> float:
    x = float(price)
    return 100.0 / (x + 100.0) if x > 0 else -x / (-x + 100.0)


def profit_for_win(price: float) -> float:
    return float(price) / 100.0 if price > 0 else 100.0 / (-float(price))


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
    return float(p_over), float(p_under), float(p_push)


def fit_live_projection(day: str) -> pd.DataFrame:
    hist = build_table()
    hist = hist[pd.to_numeric(hist["batters_faced"], errors="coerce").gt(0)].copy()
    hist["k_rate_target"] = (hist["strikeouts"] / hist["batters_faced"]).clip(0, 0.7)
    bf_feats, kr_feats, all_feats = specialized_features(hist)
    target = pd.Timestamp(day)
    train = hist[hist.date < target]
    if len(train) < 1500:
        raise SystemExit("Not enough historical starts to fit FanDuel live pitcher-K model.")

    bf = hgb("poisson", leaves=12, l2=4.0)
    kr = hgb("squared_error", leaves=12, l2=4.0)
    direct = hgb("poisson", leaves=15, l2=2.0)
    bf.fit(train[bf_feats], train["batters_faced"])
    kr.fit(train[kr_feats], train["k_rate_target"], m__sample_weight=train["batters_faced"])
    direct.fit(train[all_feats], train["strikeouts"])

    slate = schedule(day)
    if slate.empty:
        return pd.DataFrame()
    live = live_features(slate, target)
    missing = [c for c in all_feats if c not in live.columns]
    if missing:
        raise ValueError("FanDuel live feature builder is missing: " + ", ".join(missing))

    bf_hat = np.clip(bf.predict(live[bf_feats]), 5, 40)
    kr_hat = np.clip(kr.predict(live[kr_feats]), 0.02, 0.55)
    component = np.clip(bf_hat * kr_hat, 0.05, None)
    direct_hat = np.clip(direct.predict(live[all_feats]), 0.05, None)
    weight = selected_component_weight()
    live["projected_bf"] = bf_hat
    live["projected_k_rate"] = kr_hat
    live["component_k"] = component
    live["direct_k"] = direct_hat
    live["projected_k"] = np.clip(weight * component + (1.0 - weight) * direct_hat, 0.05, None)
    live["component_weight"] = weight
    live["name_key"] = live.pitcher_name.map(norm_name)
    return live


def main() -> None:
    day = date.today().isoformat()
    if not MARKET.exists():
        raise SystemExit("Missing free PropLine market snapshot; run fetch_propline_sample.py first.")

    raw = pd.read_csv(MARKET, low_memory=False)
    if raw.empty:
        print("No PropLine pitcher-K rows available.")
        pd.DataFrame().to_csv(CANDIDATES, index=False)
        pd.DataFrame().to_csv(SELECTIONS, index=False)
        return

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date.astype("string")
    raw["sportsbook"] = raw["sportsbook"].astype(str).str.strip().str.lower()
    raw["side"] = raw["side"].astype(str).str.strip().str.lower()
    raw["line"] = pd.to_numeric(raw["line"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    fd = raw[
        raw["date"].eq(day)
        & raw["sportsbook"].eq("fanduel")
        & raw["side"].isin(["over", "under"])
        & raw["line"].notna()
        & raw["price"].notna()
    ].copy()
    if fd.empty:
        print(f"No FanDuel pitcher-K quotes found in the free sample for {day}.")
        pd.DataFrame().to_csv(CANDIDATES, index=False)
        pd.DataFrame().to_csv(SELECTIONS, index=False)
        return

    live = fit_live_projection(day)
    if live.empty:
        print(f"No probable MLB starters found for {day}.")
        pd.DataFrame().to_csv(CANDIDATES, index=False)
        pd.DataFrame().to_csv(SELECTIONS, index=False)
        return

    fd["name_key"] = fd.pitcher_name.map(norm_name)
    keep = [
        "game_id", "pitcher_id", "pitcher_name", "name_key", "away_team", "home_team",
        "projected_bf", "projected_k_rate", "component_k", "direct_k", "projected_k", "component_weight",
    ]
    z = fd.merge(live[keep].rename(columns={"pitcher_name": "mlb_pitcher_name", "away_team": "mlb_away_team", "home_team": "mlb_home_team"}), on="name_key", how="inner")
    if z.empty:
        print("FanDuel quotes did not match today's MLB probable starters.")
        pd.DataFrame().to_csv(CANDIDATES, index=False)
        pd.DataFrame().to_csv(SELECTIONS, index=False)
        return

    rows = []
    generated = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    for r in z.itertuples(index=False):
        po, pu, pp = fair_probs(float(r.line), float(r.projected_k))
        side = str(r.side).lower()
        model_prob = po if side == "over" else pu
        market_prob = implied_prob(float(r.price))
        win_profit = profit_for_win(float(r.price))
        # Expected flat-stake profit including push probability. A push returns stake.
        ev = model_prob * win_profit - (1.0 - model_prob - pp)
        edge = model_prob - market_prob
        rows.append({
            "date": day,
            "generated_time_et": generated,
            "quote_snapshot_time_et": getattr(r, "snapshot_time_et", None),
            "event_id": getattr(r, "event_id", None),
            "game_id": r.game_id,
            "pitcher_id": r.pitcher_id,
            "pitcher_name": r.mlb_pitcher_name,
            "away_team": getattr(r, "mlb_away_team", None) or getattr(r, "away_team", None),
            "home_team": getattr(r, "mlb_home_team", None) or getattr(r, "home_team", None),
            "sportsbook": "FanDuel",
            "line": float(r.line),
            "side": side.upper(),
            "price": float(r.price),
            "projected_k": float(r.projected_k),
            "projected_bf": float(r.projected_bf),
            "projected_k_rate": float(r.projected_k_rate),
            "fair_over_prob": po,
            "fair_under_prob": pu,
            "push_prob": pp,
            "model_side_prob": model_prob,
            "fanduel_implied_prob_raw": market_prob,
            "model_market_edge_raw": edge,
            "expected_profit_per_unit": ev,
            "paper_only": True,
            "protocol_version": "2026-08-26-v1",
        })

    cand = pd.DataFrame(rows)
    cand = cand.sort_values(["expected_profit_per_unit", "model_market_edge_raw"], ascending=[False, False])
    cand.to_csv(CANDIDATES, index=False)

    # One frozen paper selection per pitcher start. Do not require an edge threshold here;
    # threshold performance is evaluated prospectively after grading. Negative-EV rows are
    # retained in candidates but are not paper selections.
    eligible = cand[cand.expected_profit_per_unit.gt(0)].copy()
    selections = (
        eligible.sort_values(["expected_profit_per_unit", "model_market_edge_raw"], ascending=[False, False])
        .drop_duplicates(["game_id", "pitcher_id"], keep="first")
        .sort_values("expected_profit_per_unit", ascending=False)
    )
    selections["decision"] = "PAPER BET - NOT LIVE"
    selections.to_csv(SELECTIONS, index=False)

    print(f"FanDuel candidate rows: {len(cand):,}")
    print(f"Independent positive-EV paper selections: {len(selections):,}")
    if len(selections):
        cols = ["pitcher_name", "side", "line", "price", "projected_k", "model_side_prob", "model_market_edge_raw", "expected_profit_per_unit"]
        print(selections[cols].round(4).to_string(index=False))
    else:
        print("NO PAPER BETS: no positive-EV FanDuel candidate survived today's selection rule.")


if __name__ == "__main__":
    main()
