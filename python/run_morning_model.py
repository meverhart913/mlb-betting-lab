"""Run the MLB research model using only information available morning-of-game.

Inputs
------
data/current/morning_odds.csv
    date,sportsbook,away_team,home_team,away_moneyline,home_moneyline,snapshot_time_et

data/mlb_team_game_logs.csv
data/mlb_pitcher_game_logs.csv
data/mlb_games_2018_present.csv

data/mlb_game_enrichment.csv

Outputs
-------
outputs/morning_model_predictions.csv

This remains a research pipeline. It intentionally emits NO BET for every row
until a profitable, leakage-safe decision rule is validated out of sample.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PITCHER_WINDOWS = (3, 5, 10)
TEAM_WINDOWS = (10, 30)
PITCHER_COLS = ["ip", "earned_runs", "walks", "strikeouts", "home_runs", "hits", "pitches"]
PITCHER_RATE_COLS = ["era", "whip", "k9", "bb9", "hr9"]
TEAM_COLS = [
    "runs", "hits", "home_runs", "walks", "strikeouts", "pitching_earned_runs",
    "pitching_walks", "pitching_strikeouts", "pitching_home_runs", "errors",
]


def expected_feature_cols() -> list[str]:
    """Return exactly the matchup features created by the historical builder.

    Constructing this list explicitly prevents identifiers such as team_id from
    being accidentally promoted into model features just because their column
    names share a prefix with legitimate statistics.
    """
    cols: list[str] = []
    for n in PITCHER_WINDOWS:
        cols.extend(f"diff_sp_{c}_{n}" for c in PITCHER_COLS)
        cols.extend(f"diff_sp_{c}_{n}" for c in PITCHER_RATE_COLS)
    for n in TEAM_WINDOWS:
        cols.extend(f"diff_team_{c}_{n}" for c in TEAM_COLS)
    return sorted(cols)


def american_prob(v: pd.Series) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce").astype(float)
    valid = x.abs().between(100, 5000)
    p = pd.Series(np.nan, index=x.index, dtype=float)
    p.loc[valid & (x > 0)] = 100.0 / (x.loc[valid & (x > 0)] + 100.0)
    p.loc[valid & (x < 0)] = -x.loc[valid & (x < 0)] / (-x.loc[valid & (x < 0)] + 100.0)
    return p


def ip_to_outs(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        text = str(v)
        if "." not in text:
            return float(int(text) * 3)
        inn, partial = text.split(".", 1)
        partial = int(partial)
        return float(int(inn) * 3 + partial) if partial in (0, 1, 2) else np.nan
    except Exception:
        return np.nan


def schedule_for(day: str) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": day, "gameType": "R", "hydrate": "probablePitcher"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            h = g.get("teams", {}).get("home", {})
            a = g.get("teams", {}).get("away", {})
            rows.append({
                "game_id": g.get("gamePk"),
                "date": d.get("date"),
                "away_team": (a.get("team") or {}).get("name"),
                "home_team": (h.get("team") or {}).get("name"),
                "away_team_id": (a.get("team") or {}).get("id"),
                "home_team_id": (h.get("team") or {}).get("id"),
                "away_starter_id": (a.get("probablePitcher") or {}).get("id"),
                "home_starter_id": (h.get("probablePitcher") or {}).get("id"),
                "away_starter": (a.get("probablePitcher") or {}).get("fullName"),
                "home_starter": (h.get("probablePitcher") or {}).get("fullName"),
            })
    return pd.DataFrame(rows)


def latest_team_features(logs: pd.DataFrame, team_id, target: pd.Timestamp, prefix: str) -> dict:
    z = logs[(logs["team_id"] == team_id) & (logs["date"] < target)].sort_values(["date", "game_id"])
    out = {}
    for n in TEAM_WINDOWS:
        q = z.tail(n)
        for c in TEAM_COLS:
            out[f"{prefix}_team_{c}_{n}"] = pd.to_numeric(q[c], errors="coerce").mean() if len(q) >= 3 else np.nan
    return out


def latest_pitcher_features(logs: pd.DataFrame, pitcher_id, target: pd.Timestamp, prefix: str) -> dict:
    out = {}
    if pd.isna(pitcher_id):
        return out
    z = logs[(logs["pitcher_id"] == pitcher_id) & (logs["is_starter"] == 1) & (logs["date"] < target)].sort_values(["date", "game_id"])
    for n in PITCHER_WINDOWS:
        q = z.tail(n)
        sums = {c: pd.to_numeric(q[c], errors="coerce").sum(min_count=1) for c in PITCHER_COLS}
        ip = sums.get("ip", np.nan)
        for c, val in sums.items():
            out[f"{prefix}_sp_{c}_{n}"] = val
        out[f"{prefix}_sp_era_{n}"] = 9 * sums["earned_runs"] / ip if pd.notna(ip) and ip > 0 else np.nan
        out[f"{prefix}_sp_whip_{n}"] = (sums["walks"] + sums["hits"]) / ip if pd.notna(ip) and ip > 0 else np.nan
        out[f"{prefix}_sp_k9_{n}"] = 9 * sums["strikeouts"] / ip if pd.notna(ip) and ip > 0 else np.nan
        out[f"{prefix}_sp_bb9_{n}"] = 9 * sums["walks"] / ip if pd.notna(ip) and ip > 0 else np.nan
        out[f"{prefix}_sp_hr9_{n}"] = 9 * sums["home_runs"] / ip if pd.notna(ip) and ip > 0 else np.nan
    return out


def build_live_features(schedule: pd.DataFrame) -> pd.DataFrame:
    teams = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    pitchers = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    teams["date"] = pd.to_datetime(teams["date"], errors="coerce")
    pitchers["date"] = pd.to_datetime(pitchers["date"], errors="coerce")
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    pitchers["pitcher_id"] = pd.to_numeric(pitchers["pitcher_id"], errors="coerce")
    pitchers["is_starter"] = pd.to_numeric(pitchers["is_starter"], errors="coerce").fillna(0).astype(int)
    pitchers["outs"] = pitchers["innings_pitched"].map(ip_to_outs)
    pitchers["ip"] = pitchers["outs"] / 3.0
    for c in PITCHER_COLS[1:]:
        pitchers[c] = pd.to_numeric(pitchers[c], errors="coerce")
    for c in TEAM_COLS:
        teams[c] = pd.to_numeric(teams[c], errors="coerce")

    rows = []
    for r in schedule.itertuples(index=False):
        target = pd.Timestamp(r.date)
        row = r._asdict()
        row.update(latest_team_features(teams, r.home_team_id, target, "home"))
        row.update(latest_team_features(teams, r.away_team_id, target, "away"))
        row.update(latest_pitcher_features(pitchers, r.home_starter_id, target, "home"))
        row.update(latest_pitcher_features(pitchers, r.away_starter_id, target, "away"))
        rows.append(row)
    x = pd.DataFrame(rows)

    # Only construct differences for the explicitly defined model statistics.
    # This deliberately excludes schedule identifiers such as home_team_id.
    for feature in expected_feature_cols():
        base = feature.removeprefix("diff_")
        home = f"home_{base}"
        away = f"away_{base}"
        if home in x.columns and away in x.columns:
            x[feature] = x[home] - x[away]
    return x


def fit_model(feature_cols: list[str]):
    hist = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    hist = hist[hist["home_win"].notna()].copy()
    missing = [c for c in feature_cols if c not in hist.columns]
    if missing:
        raise ValueError(
            "Historical modeling table is missing live feature columns: " + ", ".join(missing)
        )
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(max_iter=2500, C=0.5)),
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
        raise SystemExit(f"Missing {odds_path}. Copy morning_odds_template.csv to morning_odds.csv and enter the morning moneylines.")
    odds = pd.read_csv(odds_path)
    odds = odds[odds["date"].astype(str) == args.date].copy()
    schedule = schedule_for(args.date)
    if schedule.empty:
        raise SystemExit(f"No MLB regular-season games found for {args.date}.")
    live = build_live_features(schedule)
    merged = live.merge(odds, on=["date", "away_team", "home_team"], how="left")

    feature_cols = [c for c in expected_feature_cols() if c in merged.columns]
    if not feature_cols:
        raise ValueError("No expected live matchup features were created.")
    model = fit_model(feature_cols)
    merged["model_home_prob"] = model.predict_proba(merged[feature_cols])[:, 1]
    ph = american_prob(merged["home_moneyline"])
    pa = american_prob(merged["away_moneyline"])
    merged["market_home_prob"] = ph / (ph + pa)
    merged["model_edge_home"] = merged["model_home_prob"] - merged["market_home_prob"]
    merged["research_signal"] = np.where(merged["model_edge_home"] >= 0, "HOME", "AWAY")
    merged["decision"] = "NO BET - research model not validated profitable"
    merged["starter_status"] = np.where(
        merged["home_starter_id"].notna() & merged["away_starter_id"].notna(),
        "both probable starters available",
        "starter missing/unconfirmed",
    )

    keep = [
        "date", "game_id", "away_team", "home_team", "away_starter", "home_starter", "starter_status",
        "sportsbook", "snapshot_time_et", "away_moneyline", "home_moneyline", "market_home_prob",
        "model_home_prob", "model_edge_home", "research_signal", "decision",
    ]
    out = merged[[c for c in keep if c in merged.columns]].sort_values(["date", "game_id"])
    out.to_csv(OUT / "morning_model_predictions.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\nResearch mode only: every row remains NO BET until a profitable rule is validated out of sample.")


if __name__ == "__main__":
    main()
