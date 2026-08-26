"""Generate prospective FanDuel pitcher-K paper selections with V2.2.

Rules are frozen in docs/FANDUEL_PROSPECTIVE_PROTOCOL.md:
- FanDuel only
- evaluate every available main/alt strikeout line and both sides
- actual FanDuel price drives expected value
- at most one frozen selection per pitcher start
- only information available at collection time may be used

This script never places a wager.
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
from run_pitcher_k_props import live_features, schedule
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
F = D / "features"
MKT = D / "market" / "pitcher_k_historical_raw.csv"
OUT = ROOT / "outputs"
CURRENT = D / "current"
HISTORY = CURRENT / "fanduel_pitcher_k_paper_history.csv"
TODAY_OUT = OUT / "fanduel_pitcher_k_paper_today.csv"
AUDIT_OUT = OUT / "fanduel_pitcher_k_all_candidates.csv"

# A snapshot is eligible for freezing when it is close enough to first pitch to
# make confirmed lineups plausible, while still leaving normal bet-entry time.
MIN_MINUTES_TO_START = 45
MAX_MINUTES_TO_START = 195


def norm_name(v) -> str:
    x = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def implied_prob(price: float) -> float:
    x = float(price)
    if x > 0:
        return 100.0 / (x + 100.0)
    return -x / (-x + 100.0)


def win_profit(price: float) -> float:
    return float(price) / 100.0 if price > 0 else 100.0 / (-float(price))


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


def load_daily(pattern: str) -> pd.DataFrame:
    fs = sorted(F.glob(pattern))
    if not fs:
        raise SystemExit(f"Missing live V2.2 source files matching {F / pattern}")
    z = pd.concat([pd.read_csv(p, low_memory=False) for p in fs], ignore_index=True, sort=False)
    return z


def current_statcast(starters: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    raw = load_daily("statcast_pitcher_daily*.csv")
    raw["game_date"] = pd.to_datetime(raw.game_date, errors="coerce")
    raw["pitcher_id"] = pd.to_numeric(raw.pitcher_id, errors="coerce")
    raw = raw[raw.game_date.notna() & raw.pitcher_id.notna() & raw.game_date.lt(target)].copy()
    numeric = [c for c in raw.columns if c not in {"game_date", "pitcher_id"}]
    for c in numeric:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    rows = []
    for r in starters.itertuples(index=False):
        g = raw[raw.pitcher_id.eq(r.pitcher_id)].sort_values("game_date")
        row = {"pitcher_id": r.pitcher_id}
        for c in numeric:
            for w in (3, 5, 10):
                z = g[c].dropna().tail(w)
                row[f"statcast_{c}_{w}"] = float(z.mean()) if len(z) >= max(2, w // 2) else np.nan
        if "statcast_mean_velocity_3" in row and "statcast_mean_velocity_10" in row:
            row["statcast_velocity_trend_3v10"] = row["statcast_mean_velocity_3"] - row["statcast_mean_velocity_10"]
        if "statcast_whiff_per_swing_3" in row and "statcast_whiff_per_swing_10" in row:
            row["statcast_whiff_trend_3v10"] = row["statcast_whiff_per_swing_3"] - row["statcast_whiff_per_swing_10"]
        rows.append(row)
    return pd.DataFrame(rows)


def boxscore_lineup(game_id: int, opponent_side: str) -> list[int]:
    url = f"https://statsapi.mlb.com/api/v1/game/{int(game_id)}/boxscore"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    team = (((r.json().get("teams") or {}).get(opponent_side)) or {})
    players = team.get("players") or {}

    ordered = []
    for p in players.values():
        bo = p.get("battingOrder")
        pid = ((p.get("person") or {}).get("id"))
        try:
            bo_num = int(bo)
        except (TypeError, ValueError):
            continue
        if pid and bo_num > 0:
            ordered.append((bo_num, int(pid)))
    if ordered:
        ids = []
        for _, pid in sorted(ordered):
            if pid not in ids:
                ids.append(pid)
        if len(ids) >= 9:
            return ids[:9]

    # Some pregame boxscores expose the batting order directly.
    direct = team.get("battingOrder") or []
    if len(direct) >= 9:
        return [int(x) for x in direct[:9]]
    return []


def batter_roll(raw: pd.DataFrame, batter_id: int, hand: str, target: pd.Timestamp):
    g = raw[
        raw.batter_id.eq(batter_id)
        & raw.pitcher_hand.eq(hand)
        & raw.game_date.lt(target)
    ]
    row = {}
    for days in (30, 90, 365):
        z = g[g.game_date.ge(target - pd.Timedelta(days=days))]
        pa = float(z.plate_appearances.sum())
        ks = float(z.strikeouts.sum())
        row[f"batter_k_pa_{days}d"] = ks / pa if pa > 0 else np.nan
        row[f"batter_pa_{days}d"] = pa
    return row


def current_lineup_features(starters: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    b = load_daily("batter_k_by_pitcher_hand_daily*.csv")
    b["game_date"] = pd.to_datetime(b.game_date, errors="coerce")
    b["batter_id"] = pd.to_numeric(b.batter_id, errors="coerce")
    b["pitcher_hand"] = b.pitcher_hand.astype(str).str.upper()
    b["plate_appearances"] = pd.to_numeric(b.plate_appearances, errors="coerce").fillna(0)
    b["strikeouts"] = pd.to_numeric(b.strikeouts, errors="coerce").fillna(0)

    ph = pd.read_csv(F / "pitcher_handedness.csv")
    ph["pitcher_id"] = pd.to_numeric(ph.pitcher_id, errors="coerce")
    ph["pitcher_hand"] = ph.pitcher_hand.astype(str).str.upper()
    hands = dict(zip(ph.pitcher_id, ph.pitcher_hand))

    rows = []
    for r in starters.itertuples(index=False):
        hand = hands.get(float(r.pitcher_id))
        if hand not in {"L", "R"}:
            continue
        opponent_side = "away" if r.side == "home" else "home"
        ids = boxscore_lineup(int(r.game_id), opponent_side)
        if len(ids) < 9:
            continue
        vals = []
        for order, bid in enumerate(ids[:9], start=1):
            q = batter_roll(b, bid, hand, target)
            q["batting_order"] = order
            vals.append(q)
        x = pd.DataFrame(vals)
        pa90 = pd.to_numeric(x.batter_pa_90d, errors="coerce")
        k90 = pd.to_numeric(x.batter_k_pa_90d, errors="coerce")
        k365 = pd.to_numeric(x.batter_k_pa_365d, errors="coerce")
        rel = (pa90 / (pa90 + 40)).clip(0, 1)
        blend = rel * k90 + (1 - rel) * k365
        matched = blend.notna()
        rows.append({
            "game_id": r.game_id,
            "pitcher_id": r.pitcher_id,
            "opponent_team_id": r.opponent_team_id,
            "pitcher_hand": hand,
            "lineup_batters": 9,
            "matched_batters": int(matched.sum()),
            "lineup_k30_mean": float(pd.to_numeric(x.batter_k_pa_30d, errors="coerce").mean()),
            "lineup_k90_mean": float(k90.mean()),
            "lineup_k365_mean": float(k365.mean()),
            "lineup_k_blend_mean": float(blend.mean()),
            "lineup_k_blend_top4": float(blend.iloc[:4].mean()),
            "lineup_pa90_mean": float(pa90.mean()),
            "lineup_match_coverage": float(matched.mean()),
        })
    return pd.DataFrame(rows)


def fit_v22_and_predict(day: str) -> pd.DataFrame:
    target = pd.Timestamp(day)
    z = build_table().copy()
    z["date"] = pd.to_datetime(z.date, errors="coerce")
    z["pitcher_id"] = pd.to_numeric(z.pitcher_id, errors="coerce")
    z["game_id"] = pd.to_numeric(z.game_id, errors="coerce")
    z = z[pd.to_numeric(z.batters_faced, errors="coerce").gt(0) & z.date.lt(target)].copy()
    z["k_rate_target"] = (z.strikeouts / z.batters_faced).clip(0, 0.7)

    sc = pd.read_csv(F / "statcast_pitcher_pregame.csv", low_memory=False)
    sc["game_date"] = pd.to_datetime(sc.game_date, errors="coerce")
    sc["pitcher_id"] = pd.to_numeric(sc.pitcher_id, errors="coerce")
    sc = sc.rename(columns={"game_date": "date"})
    stat = [c for c in sc if c.startswith("statcast_")]
    z = z.merge(sc[["date", "pitcher_id", *stat]], on=["date", "pitcher_id"], how="left")

    lu = pd.read_csv(F / "historical_lineup_hand_features.csv", low_memory=False)
    lu["game_id"] = pd.to_numeric(lu.game_id, errors="coerce")
    lu["pitcher_id"] = pd.to_numeric(lu.pitcher_id, errors="coerce")
    lf = [c for c in lu if c.startswith("lineup_") and c != "lineup_batters"]
    z = z.merge(lu[["game_id", "pitcher_id", *lf]], on=["game_id", "pitcher_id"], how="left")

    bf0, kr0, d0 = specialized_features(z)
    bf = sorted(set(bf0 + stat + lf))
    kr = sorted(set(kr0 + stat + lf))
    direct = sorted(set(d0 + stat + lf))

    a = hgb("poisson", leaves=12, l2=4)
    b = hgb("squared_error", leaves=12, l2=4)
    c = hgb("poisson", leaves=15, l2=2)
    a.fit(z[bf], z.batters_faced)
    b.fit(z[kr], z.k_rate_target, m__sample_weight=z.batters_faced)
    c.fit(z[direct], z.strikeouts)

    slate = schedule(day)
    if slate.empty:
        return pd.DataFrame()
    base = live_features(slate, target)
    sc_live = current_statcast(slate, target)
    lu_live = current_lineup_features(slate, target)
    live = base.merge(sc_live, on="pitcher_id", how="left").merge(
        lu_live[["game_id", "pitcher_id", *lf]], on=["game_id", "pitcher_id"], how="inner"
    ) if not lu_live.empty else pd.DataFrame()
    if live.empty:
        print("No confirmed nine-batter opponent lineups available for V2.2 yet.")
        return live

    for col in set(bf + kr + direct):
        if col not in live.columns:
            live[col] = np.nan
    bh = np.clip(a.predict(live[bf]), 5, 40)
    kh = np.clip(b.predict(live[kr]), 0.02, 0.55)
    comp = np.clip(bh * kh, 0.05, None)
    dh = np.clip(c.predict(live[direct]), 0.05, None)
    live["projected_bf"] = bh
    live["projected_k_rate"] = kh
    live["component_k"] = comp
    live["direct_k"] = dh
    live["projected_k"] = np.clip(0.5 * comp + 0.5 * dh, 0.05, None)
    live["model_version"] = "v22_lineup_all_live"
    live["model_generated_at_et"] = pd.Timestamp.now(tz="America/New_York").isoformat()
    live["name_key"] = live.pitcher_name.map(norm_name)
    return live


def pair_fanduel(raw: pd.DataFrame, day: str) -> pd.DataFrame:
    z = raw.copy()
    z["sportsbook"] = z.sportsbook.astype(str).str.lower()
    z["side"] = z.side.astype(str).str.lower()
    z["date"] = pd.to_datetime(z.date, errors="coerce").dt.date.astype("string")
    z["line"] = pd.to_numeric(z.line, errors="coerce")
    z["price"] = pd.to_numeric(z.price, errors="coerce")
    z["name_key"] = z.pitcher_name.map(norm_name)
    z["commence_time_utc"] = pd.to_datetime(z.commence_time_utc, errors="coerce", utc=True)
    z["collected_at_utc"] = pd.to_datetime(z.collected_at_utc, errors="coerce", utc=True)
    z = z[
        z.sportsbook.eq("fanduel")
        & z.date.eq(day)
        & z.side.isin(["over", "under"])
        & z.line.notna()
        & z.price.notna()
    ].copy()
    if z.empty:
        return z
    idx = ["date", "event_id", "name_key", "pitcher_name", "line", "commence_time_utc", "collected_at_utc"]
    w = z.pivot_table(index=idx, columns="side", values="price", aggfunc="last").reset_index()
    w = w.rename(columns={"over": "over_price", "under": "under_price"})
    return w


def select_candidates(market: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    if market.empty or projections.empty:
        return pd.DataFrame()
    g = market.merge(
        projections[["game_id", "pitcher_id", "pitcher_name", "name_key", "projected_bf", "projected_k_rate", "projected_k", "lineup_match_coverage", "model_version", "model_generated_at_et"]],
        on="name_key", how="inner", suffixes=("_market", "_model")
    )
    rows = []
    for r in g.itertuples(index=False):
        minutes = (r.commence_time_utc - r.collected_at_utc).total_seconds() / 60.0
        po, pu, pp = fair_probs(float(r.line), float(r.projected_k))
        op = float(r.over_price) if pd.notna(r.over_price) else np.nan
        up = float(r.under_price) if pd.notna(r.under_price) else np.nan
        oi = implied_prob(op) if np.isfinite(op) else np.nan
        ui = implied_prob(up) if np.isfinite(up) else np.nan
        if np.isfinite(oi) and np.isfinite(ui) and oi + ui > 0:
            onv = oi / (oi + ui)
            unv = ui / (oi + ui)
        else:
            onv, unv = oi, ui
        options = []
        if np.isfinite(op):
            options.append(("OVER", op, po, onv, po * win_profit(op) - pu))
        if np.isfinite(up):
            options.append(("UNDER", up, pu, unv, pu * win_profit(up) - po))
        for side, price, pwin, market_p, ev in options:
            rows.append({
                "date": r.date,
                "game_id": r.game_id,
                "event_id": r.event_id,
                "pitcher_id": r.pitcher_id,
                "pitcher_name": r.pitcher_name_model,
                "line": float(r.line),
                "side": side,
                "fanduel_price": price,
                "fanduel_implied_prob": implied_prob(price),
                "fanduel_no_vig_prob": market_p,
                "model_win_prob": pwin,
                "push_prob": pp,
                "model_market_edge": pwin - market_p if np.isfinite(market_p) else np.nan,
                "expected_profit_per_unit": ev,
                "projected_k": r.projected_k,
                "projected_bf": r.projected_bf,
                "projected_k_rate": r.projected_k_rate,
                "lineup_match_coverage": r.lineup_match_coverage,
                "commence_time_utc": r.commence_time_utc,
                "collected_at_utc": r.collected_at_utc,
                "minutes_to_start": minutes,
                "timing_eligible": MIN_MINUTES_TO_START <= minutes <= MAX_MINUTES_TO_START,
                "model_version": r.model_version,
                "model_generated_at_et": r.model_generated_at_et,
            })
    return pd.DataFrame(rows)


def freeze(candidates: pd.DataFrame) -> pd.DataFrame:
    CURRENT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        pd.DataFrame().to_csv(TODAY_OUT, index=False)
        return pd.DataFrame()
    candidates.to_csv(AUDIT_OUT, index=False)

    eligible = candidates[candidates.timing_eligible].copy()
    if eligible.empty:
        pd.DataFrame().to_csv(TODAY_OUT, index=False)
        print("No FanDuel candidates are inside the frozen decision window.")
        return pd.DataFrame()

    # Highest actual expected value, not largest projection difference.
    chosen = (
        eligible.sort_values(["expected_profit_per_unit", "model_market_edge"], ascending=False)
        .drop_duplicates(["date", "game_id", "pitcher_id"], keep="first")
        .copy()
    )
    chosen["edge_ge_0"] = chosen.model_market_edge.ge(0)
    chosen["edge_ge_025"] = chosen.model_market_edge.ge(0.025)
    chosen["edge_ge_05"] = chosen.model_market_edge.ge(0.05)
    chosen["edge_ge_075"] = chosen.model_market_edge.ge(0.075)
    chosen["edge_ge_10"] = chosen.model_market_edge.ge(0.10)
    chosen["paper_status"] = "FROZEN_PAPER_SELECTION"

    if HISTORY.exists():
        hist = pd.read_csv(HISTORY, low_memory=False)
    else:
        hist = pd.DataFrame()
    if not hist.empty:
        keys = set(zip(hist.date.astype(str), pd.to_numeric(hist.game_id, errors="coerce"), pd.to_numeric(hist.pitcher_id, errors="coerce")))
        keep = []
        for r in chosen.itertuples(index=False):
            keep.append((str(r.date), float(r.game_id), float(r.pitcher_id)) not in keys)
        new = chosen[pd.Series(keep, index=chosen.index)].copy()
    else:
        new = chosen.copy()

    if not new.empty:
        all_hist = pd.concat([hist, new], ignore_index=True, sort=False) if not hist.empty else new
        all_hist.to_csv(HISTORY, index=False)
    chosen.to_csv(TODAY_OUT, index=False)
    print(f"FanDuel V2.2: {len(candidates):,} side/line candidates; {len(chosen):,} one-per-pitcher selections; {len(new):,} newly frozen.")
    if len(chosen):
        show = ["pitcher_name", "side", "line", "fanduel_price", "projected_k", "model_win_prob", "model_market_edge", "expected_profit_per_unit", "minutes_to_start"]
        print(chosen[show].round(4).to_string(index=False))
    return chosen


def main():
    day = date.today().isoformat()
    if not MKT.exists():
        raise SystemExit("Missing current normalized market; run fetch_propline_sample.py first.")
    raw = pd.read_csv(MKT, low_memory=False)
    market = pair_fanduel(raw, day)
    if market.empty:
        print(f"No FanDuel pitcher-K quotes found for {day}.")
        return
    projections = fit_v22_and_predict(day)
    candidates = select_candidates(market, projections)
    freeze(candidates)


if __name__ == "__main__":
    main()
