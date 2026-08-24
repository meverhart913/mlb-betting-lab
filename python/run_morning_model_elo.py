"""Run the validated baseball-only morning model with tuned pregame Elo added.

Elo was promoted because it improved expanding-season out-of-sample baseball
probability scoring. It did not improve the sportsbook market baseline, so all
rows remain NO BET and market probability remains the stronger benchmark.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_morning_model as base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"

ELO_K = 12.0
ELO_HOME_ADV = 50.0
ELO_CARRY = 0.90
ELO_BASE = 1500.0


def elo_expected(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((home_rating + ELO_HOME_ADV) - away_rating) / 400.0))


def build_elo_history(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], int | None]:
    g = games.sort_values(["date", "game_id"]).copy()
    ratings: dict[str, float] = {}
    current_year = None
    rows = []
    for r in g.itertuples(index=False):
        year = pd.Timestamp(r.date).year
        if current_year is None:
            current_year = year
        elif year != current_year:
            ratings = {team: ELO_BASE + ELO_CARRY * (rating - ELO_BASE) for team, rating in ratings.items()}
            current_year = year
        home = str(r.home_team)
        away = str(r.away_team)
        rh = ratings.get(home, ELO_BASE)
        ra = ratings.get(away, ELO_BASE)
        prob = elo_expected(rh, ra)
        rows.append({"game_id": r.game_id, "elo_diff": rh - ra, "elo_prob_home": prob})
        hs = pd.to_numeric(r.home_score, errors="coerce")
        aws = pd.to_numeric(r.away_score, errors="coerce")
        if pd.notna(hs) and pd.notna(aws) and hs != aws:
            actual = 1.0 if hs > aws else 0.0
            delta = ELO_K * (actual - prob)
            ratings[home] = rh + delta
            ratings[away] = ra - delta
    return pd.DataFrame(rows).drop_duplicates("game_id", keep="last"), ratings, current_year


def live_elo_for(schedule: pd.DataFrame, target_date: str) -> pd.DataFrame:
    games = pd.read_csv(DATA / "mlb_games_2018_present.csv", low_memory=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    target = pd.Timestamp(target_date)
    prior = games[games["date"].notna() & (games["date"] < target)].copy()
    _, ratings, last_year = build_elo_history(prior)

    target_year = target.year
    if last_year is not None and target_year != last_year:
        ratings = {team: ELO_BASE + ELO_CARRY * (rating - ELO_BASE) for team, rating in ratings.items()}

    rows = []
    for r in schedule.itertuples(index=False):
        rh = ratings.get(str(r.home_team), ELO_BASE)
        ra = ratings.get(str(r.away_team), ELO_BASE)
        rows.append({"game_id": r.game_id, "elo_diff": rh - ra, "elo_prob_home": elo_expected(rh, ra)})
    return pd.DataFrame(rows)


def historical_training_table() -> pd.DataFrame:
    hist = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    hist = hist[hist["home_win"].notna()].copy()
    games = pd.read_csv(DATA / "mlb_games_2018_present.csv", low_memory=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    elo, _, _ = build_elo_history(games[games["date"].notna()].copy())
    return hist.merge(elo[["game_id", "elo_diff"]], on="game_id", how="left")


def fit_model(feature_cols: list[str]):
    hist = historical_training_table()
    missing = [c for c in feature_cols if c not in hist.columns]
    if missing:
        raise ValueError("Historical training table is missing: " + ", ".join(missing))
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(max_iter=2500, C=0.3)),
    ])
    model.fit(hist[feature_cols], hist["home_win"].astype(int))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--odds", default=str(CURRENT / "morning_odds.csv"))
    args = ap.parse_args()

    odds_path = Path(args.odds)
    if not odds_path.exists():
        raise SystemExit(f"Missing {odds_path}.")
    odds = pd.read_csv(odds_path)
    odds = odds[odds["date"].astype(str) == args.date].copy()
    schedule = base.schedule_for(args.date)
    if schedule.empty:
        raise SystemExit(f"No MLB regular-season games found for {args.date}.")

    live = base.build_live_features(schedule)
    live = live.merge(live_elo_for(schedule, args.date), on="game_id", how="left")
    merged = live.merge(odds, on=["date", "away_team", "home_team"], how="left")

    feature_cols = [c for c in base.expected_feature_cols() if c in merged.columns] + ["elo_diff"]
    model = fit_model(feature_cols)
    merged["model_home_prob"] = model.predict_proba(merged[feature_cols])[:, 1]
    ph = base.american_prob(merged["home_moneyline"])
    pa = base.american_prob(merged["away_moneyline"])
    merged["market_home_prob"] = ph / (ph + pa)
    merged["model_edge_home"] = merged["model_home_prob"] - merged["market_home_prob"]
    merged["research_signal"] = np.where(merged["model_edge_home"] >= 0, "HOME", "AWAY")
    merged["decision"] = "NO BET - market remains stronger than validated baseball model"
    merged["starter_status"] = np.where(
        merged["home_starter_id"].notna() & merged["away_starter_id"].notna(),
        "both probable starters available",
        "starter missing/unconfirmed",
    )
    merged["model_version"] = "baseball_plus_tuned_elo_k12_ha50_carry090"

    keep = [
        "date", "game_id", "away_team", "home_team", "away_starter", "home_starter", "starter_status",
        "sportsbook", "snapshot_time_et", "away_moneyline", "home_moneyline", "market_home_prob",
        "elo_diff", "elo_prob_home", "model_home_prob", "model_edge_home", "research_signal", "decision", "model_version",
    ]
    out = merged[[c for c in keep if c in merged.columns]].sort_values(["date", "game_id"])
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "morning_model_predictions.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\nTuned Elo improves the baseball model but not the market baseline; every row remains NO BET.")


if __name__ == "__main__":
    main()
