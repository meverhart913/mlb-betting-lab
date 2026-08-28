"""Run FanDuel pitcher-K paper selection with V2.2 and a V2.1 fallback.

V2.2 lineup-handedness is used whenever a confirmed nine-batter opponent lineup
is available at model time. Starters without a confirmed lineup are projected
with the Statcast-only V2.1 architecture instead of being silently dropped.
Every row records the model version actually used.

Research only. This script never places a wager.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from build_pitcher_k_model import build_table
from run_pitcher_k_props import live_features, schedule
from run_v22_fanduel_paper import (
    MKT,
    F,
    freeze,
    pair_fanduel,
    select_candidates,
    current_statcast,
    fit_v22_and_predict,
)
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def fit_v21_and_predict(day: str) -> pd.DataFrame:
    """Fit the Statcast-only V2.1 architecture and score today's starters."""
    target = pd.Timestamp(day)
    z = build_table().copy()
    z["date"] = pd.to_datetime(z.date, errors="coerce")
    z["pitcher_id"] = pd.to_numeric(z.pitcher_id, errors="coerce")
    z = z[pd.to_numeric(z.batters_faced, errors="coerce").gt(0) & z.date.lt(target)].copy()
    z["k_rate_target"] = (z.strikeouts / z.batters_faced).clip(0, 0.7)

    sc = pd.read_csv(F / "statcast_pitcher_pregame.csv", low_memory=False)
    sc["game_date"] = pd.to_datetime(sc.game_date, errors="coerce")
    sc["pitcher_id"] = pd.to_numeric(sc.pitcher_id, errors="coerce")
    sc = sc.rename(columns={"game_date": "date"})
    stat = [c for c in sc if c.startswith("statcast_")]
    z = z.merge(sc[["date", "pitcher_id", *stat]], on=["date", "pitcher_id"], how="left")

    bf0, kr0, d0 = specialized_features(z)
    bf = sorted(set(bf0 + stat))
    kr = sorted(set(kr0 + stat))
    direct = sorted(set(d0 + stat))

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
    live = base.merge(sc_live, on="pitcher_id", how="left")
    if live.empty:
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
    live["lineup_match_coverage"] = np.nan
    live["model_version"] = "v21_statcast_fallback_live"
    live["model_generated_at_et"] = pd.Timestamp.now(tz="America/New_York").isoformat()
    # fit_v22_and_predict uses the same normalization helper indirectly; market
    # merge only needs the same canonical alphanumeric key.
    live["name_key"] = (
        live.pitcher_name.astype(str).str.normalize("NFKD")
        .str.encode("ascii", errors="ignore").str.decode("ascii")
        .str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )
    return live


def hybrid_predictions(day: str) -> pd.DataFrame:
    v22 = fit_v22_and_predict(day)
    v21 = fit_v21_and_predict(day)
    if v21.empty:
        return v22
    if v22.empty:
        return v21

    used = set(zip(pd.to_numeric(v22.game_id, errors="coerce"), pd.to_numeric(v22.pitcher_id, errors="coerce")))
    keep = [
        (float(g), float(p)) not in used
        for g, p in zip(pd.to_numeric(v21.game_id, errors="coerce"), pd.to_numeric(v21.pitcher_id, errors="coerce"))
    ]
    fallback = v21[pd.Series(keep, index=v21.index)].copy()
    out = pd.concat([v22, fallback], ignore_index=True, sort=False)
    print(
        f"Live model routing: V2.2 lineup={len(v22)} starters; "
        f"V2.1 Statcast fallback={len(fallback)} starters; total={len(out)}."
    )
    return out


def main() -> None:
    day = date.today().isoformat()
    if not MKT.exists():
        raise SystemExit("Missing current normalized FanDuel market file.")
    raw = pd.read_csv(MKT, low_memory=False)
    market = pair_fanduel(raw, day)
    if market.empty:
        print(f"No FanDuel pitcher-K quotes found for {day}.")
        return

    projections = hybrid_predictions(day)
    OUT.mkdir(exist_ok=True)
    if not projections.empty:
        cols = [c for c in [
            "date", "game_id", "pitcher_id", "pitcher_name", "away_team", "home_team",
            "projected_k", "projected_bf", "projected_k_rate", "lineup_match_coverage",
            "model_version", "model_generated_at_et"
        ] if c in projections.columns]
        projections[cols].to_csv(OUT / "fanduel_pitcher_k_live_projections.csv", index=False)

    candidates = select_candidates(market, projections)
    freeze(candidates)


if __name__ == "__main__":
    main()
