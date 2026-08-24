"""Build leakage-safe MLB starter/team features and walk-forward model tests.

The model deliberately uses only information available before each game:
- starting-pitcher rolling statistics are shifted one appearance;
- team rolling statistics are shifted one game;
- closing moneyline data is used only as a benchmark/backtest price, never as a
  predictive feature in the baseball-only models.

Inputs under data/:
  mlb_games_2018_present.csv
  mlb_game_enrichment.csv
  mlb_pitcher_game_logs.csv
  mlb_team_game_logs.csv
  mlb_odds_part_*.csv (optional, for market comparison and ROI simulation)

Outputs under outputs/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PITCHER_WINDOWS = (3, 5, 10)
TEAM_WINDOWS = (10, 30)
PITCHER_COUNT_COLS = ["ip", "earned_runs", "walks", "strikeouts", "home_runs", "hits", "pitches"]
TEAM_COLS = [
    "runs", "hits", "home_runs", "walks", "strikeouts", "pitching_earned_runs",
    "pitching_walks", "pitching_strikeouts", "pitching_home_runs", "errors",
]


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def ip_to_outs(value) -> float:
    """Convert MLB innings notation (e.g. 5.2 = 5 innings, 2 outs) to outs."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        if "." not in text:
            return float(int(text) * 3)
        innings, partial = text.split(".", 1)
        outs = int(partial)
        if outs not in (0, 1, 2):
            return np.nan
        return float(int(innings) * 3 + outs)
    except (TypeError, ValueError):
        return np.nan


def american_prob(value) -> np.ndarray:
    x = pd.to_numeric(value, errors="coerce").astype(float)
    return np.where(x > 0, 100.0 / (x + 100.0), np.where(x < 0, -x / (-x + 100.0), np.nan))


def american_profit(value) -> np.ndarray:
    """Profit on a $1 stake for winning American odds."""
    x = pd.to_numeric(value, errors="coerce").astype(float)
    return np.where(x > 0, x / 100.0, np.where(x < 0, 100.0 / -x, np.nan))


def make_game_labels(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    hs = pd.to_numeric(games["home_score"], errors="coerce")
    aws = pd.to_numeric(games["away_score"], errors="coerce")
    completed = hs.notna() & aws.notna() & (hs != aws)
    games["home_win"] = np.where(completed, (hs > aws).astype(float), np.nan)
    return games


def pitcher_features(pitchers: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    p = pitchers.copy()
    p["outs"] = p["innings_pitched"].map(ip_to_outs)
    p["ip"] = p["outs"] / 3.0
    for c in ["earned_runs", "walks", "strikeouts", "home_runs", "hits", "batters_faced", "pitches"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["is_starter"] = pd.to_numeric(p["is_starter"], errors="coerce").fillna(0).astype(int)
    p = p.sort_values(["pitcher_id", "date", "game_id"])
    starters = p[p["is_starter"] == 1].copy()

    for n in PITCHER_WINDOWS:
        grp = starters.groupby("pitcher_id", group_keys=False)
        for c in PITCHER_COUNT_COLS:
            starters[f"sp_{c}_{n}"] = grp[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
        ip = starters[f"sp_ip_{n}"].replace(0, np.nan)
        starters[f"sp_era_{n}"] = 9 * starters[f"sp_earned_runs_{n}"] / ip
        starters[f"sp_whip_{n}"] = (starters[f"sp_walks_{n}"] + starters[f"sp_hits_{n}"]) / ip
        starters[f"sp_k9_{n}"] = 9 * starters[f"sp_strikeouts_{n}"] / ip
        starters[f"sp_bb9_{n}"] = 9 * starters[f"sp_walks_{n}"] / ip
        starters[f"sp_hr9_{n}"] = 9 * starters[f"sp_home_runs_{n}"] / ip

    cols = [c for c in starters.columns if c.startswith("sp_") and any(c.endswith(f"_{n}") for n in PITCHER_WINDOWS)]
    wide = starters[["game_id", "side"] + cols].drop_duplicates(["game_id", "side"], keep="last").pivot(index="game_id", columns="side")
    wide.columns = [f"{side}_{c}" for c, side in wide.columns]
    return wide.reset_index(), cols


def team_features(teams: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    t = teams.copy()
    for c in TEAM_COLS:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values(["team_id", "date", "game_id"])
    grp = t.groupby("team_id", group_keys=False)
    for n in TEAM_WINDOWS:
        for c in TEAM_COLS:
            t[f"team_{c}_{n}"] = grp[c].transform(lambda s: s.shift(1).rolling(n, min_periods=3).mean())
    cols = [c for c in t.columns if c.startswith("team_") and c not in {"team_id", "team_name"}]
    wide = t[["game_id", "side"] + cols].drop_duplicates(["game_id", "side"], keep="last").pivot(index="game_id", columns="side")
    wide.columns = [f"{side}_{c}" for c, side in wide.columns]
    return wide.reset_index(), cols


def add_market(games: pd.DataFrame) -> pd.DataFrame:
    parts = sorted(DATA.glob("mlb_odds_part_*.csv"))
    if not parts:
        return games
    odds = pd.concat([pd.read_csv(x, low_memory=False) for x in parts], ignore_index=True)
    require_columns(
        odds,
        ["date", "home_team", "away_team", "market", "close_home_odds", "close_away_odds"],
        "odds data",
    )
    odds = odds[odds["market"].astype(str).str.lower() == "moneyline"].copy()
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.normalize()
    odds["ph"] = american_prob(odds["close_home_odds"])
    odds["pa"] = american_prob(odds["close_away_odds"])
    denom = odds["ph"] + odds["pa"]
    odds["market_home_prob"] = odds["ph"] / denom.replace(0, np.nan)
    odds["close_home_odds"] = pd.to_numeric(odds["close_home_odds"], errors="coerce")
    odds["close_away_odds"] = pd.to_numeric(odds["close_away_odds"], errors="coerce")
    consensus = odds.groupby(["date", "home_team", "away_team"], as_index=False).agg(
        market_home_prob=("market_home_prob", "median"),
        close_home_odds=("close_home_odds", "median"),
        close_away_odds=("close_away_odds", "median"),
        sportsbook_quotes=("market_home_prob", "count"),
    )
    counts = games.groupby(["date", "home_team", "away_team"]).size().rename("game_count").reset_index()
    consensus = consensus.merge(counts, on=["date", "home_team", "away_team"], how="left")
    consensus = consensus[consensus["game_count"] == 1].drop(columns="game_count")
    return games.merge(consensus, on=["date", "home_team", "away_team"], how="left")


def backtest_edges(frame: pd.DataFrame, model_name: str, season: int) -> list[dict]:
    if "market_home_prob" not in frame.columns:
        return []
    rows: list[dict] = []
    for threshold in (0.02, 0.03, 0.05, 0.075, 0.10):
        z = frame.dropna(subset=["market_home_prob", "close_home_odds", "close_away_odds", "model_home_prob", "home_win"]).copy()
        if z.empty:
            continue
        edge = z["model_home_prob"] - z["market_home_prob"]
        side = np.where(edge >= threshold, "home", np.where(edge <= -threshold, "away", "pass"))
        z["bet_side"] = side
        z = z[z["bet_side"] != "pass"].copy()
        if z.empty:
            rows.append({"season": season, "model": model_name, "edge_threshold": threshold, "bets": 0, "wins": 0, "hit_rate": np.nan, "roi": np.nan})
            continue
        z["won"] = np.where(z["bet_side"] == "home", z["home_win"] == 1, z["home_win"] == 0)
        chosen_odds = np.where(z["bet_side"] == "home", z["close_home_odds"], z["close_away_odds"])
        profit_if_win = american_profit(chosen_odds)
        z["profit"] = np.where(z["won"], profit_if_win, -1.0)
        rows.append({
            "season": season,
            "model": model_name,
            "edge_threshold": threshold,
            "bets": int(len(z)),
            "wins": int(z["won"].sum()),
            "hit_rate": float(z["won"].mean()),
            "roi": float(z["profit"].mean()),
        })
    return rows


def main() -> None:
    games = pd.read_csv(DATA / "mlb_games_2018_present.csv", low_memory=False)
    enrichment = pd.read_csv(DATA / "mlb_game_enrichment.csv", low_memory=False)
    pitchers = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    teams = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)

    require_columns(games, ["game_id", "date", "home_team", "away_team", "home_score", "away_score"], "games")
    require_columns(enrichment, ["game_id", "date", "home_starter_id", "away_starter_id"], "game enrichment")
    require_columns(pitchers, ["game_id", "date", "side", "pitcher_id", "is_starter", "innings_pitched"], "pitcher logs")
    require_columns(teams, ["game_id", "date", "side", "team_id"] + TEAM_COLS, "team logs")

    for df in (games, enrichment, pitchers, teams):
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    games = make_game_labels(games)
    games = games.merge(
        enrichment[["game_id", "home_starter_id", "away_starter_id"]].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    sp_wide, sp_cols = pitcher_features(pitchers)
    tm_wide, tm_cols = team_features(teams)
    games = games.merge(sp_wide, on="game_id", how="left").merge(tm_wide, on="game_id", how="left")

    features: list[str] = []
    for c in sp_cols + tm_cols:
        h, a = f"home_{c}", f"away_{c}"
        if h in games.columns and a in games.columns:
            name = f"diff_{c}"
            games[name] = games[h] - games[a]
            features.append(name)
    if not features:
        raise ValueError("No matchup features were created; verify side values are exactly 'home'/'away'.")

    games = add_market(games)
    usable = games[games["home_win"].notna() & games["date"].notna()].copy()
    if usable.empty:
        raise ValueError("No completed games with valid dates were available for modeling.")

    # Coverage diagnostics make silent join failures visible.
    pitcher_feature_names = [f"diff_{c}" for c in sp_cols if f"diff_{c}" in usable]
    team_feature_names = [f"diff_{c}" for c in tm_cols if f"diff_{c}" in usable]
    diagnostics = pd.DataFrame([
        {"metric": "completed_games", "value": len(usable)},
        {"metric": "pitcher_features", "value": len(pitcher_feature_names)},
        {"metric": "team_features", "value": len(team_feature_names)},
        {"metric": "games_with_any_pitcher_feature", "value": int(usable[pitcher_feature_names].notna().any(axis=1).sum()) if pitcher_feature_names else 0},
        {"metric": "games_with_any_team_feature", "value": int(usable[team_feature_names].notna().any(axis=1).sum()) if team_feature_names else 0},
        {"metric": "games_with_market", "value": int(usable["market_home_prob"].notna().sum()) if "market_home_prob" in usable else 0},
    ])
    diagnostics.to_csv(OUT / "pitcher_model_data_diagnostics.csv", index=False)

    x = usable[features]
    y = usable["home_win"].astype(int)
    years = usable["date"].dt.year
    models = {
        "logistic": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("m", LogisticRegression(max_iter=2500, C=0.5)),
        ]),
        "hist_gb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=2.0, random_state=42)),
        ]),
    }

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    bet_rows: list[dict] = []
    min_year = max(2021, int(years.min()) + 2)
    max_year = int(years.max())

    for year in range(min_year, max_year + 1):
        train = years < year
        test = years == year
        if train.sum() < 500 or test.sum() == 0:
            continue
        for name, model in models.items():
            model.fit(x.loc[train], y.loc[train])
            prob = model.predict_proba(x.loc[test])[:, 1]
            actual = y.loc[test]
            auc = roc_auc_score(actual, prob) if actual.nunique() > 1 else np.nan
            row = {
                "season": year,
                "model": name,
                "games": int(test.sum()),
                "accuracy": accuracy_score(actual, prob >= 0.5),
                "log_loss": log_loss(actual, prob, labels=[0, 1]),
                "brier": brier_score_loss(actual, prob),
                "roc_auc": auc,
                "mean_prediction": float(np.mean(prob)),
                "home_win_rate": float(actual.mean()),
            }
            if "market_home_prob" in usable.columns:
                market = usable.loc[test, "market_home_prob"]
                ok = market.notna()
                row["market_games"] = int(ok.sum())
                if ok.any():
                    row["market_log_loss"] = log_loss(actual.loc[ok], market.loc[ok], labels=[0, 1])
                    row["market_brier"] = brier_score_loss(actual.loc[ok], market.loc[ok])
            metric_rows.append(row)

            cols = ["game_id", "date", "home_team", "away_team", "home_win"]
            for optional in ["market_home_prob", "close_home_odds", "close_away_odds", "sportsbook_quotes"]:
                if optional in usable.columns:
                    cols.append(optional)
            pred = usable.loc[test, cols].copy()
            pred["model"] = name
            pred["model_home_prob"] = prob
            prediction_rows.append(pred)
            bet_rows.extend(backtest_edges(pred, name, year))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    betting = pd.DataFrame(bet_rows)
    metrics.to_csv(OUT / "pitcher_model_walkforward_metrics.csv", index=False)
    predictions.to_csv(OUT / "pitcher_model_walkforward_predictions.csv", index=False)
    betting.to_csv(OUT / "pitcher_model_edge_backtest.csv", index=False)

    table_cols = ["game_id", "date", "home_team", "away_team", "home_win"] + features
    for optional in ["market_home_prob", "close_home_odds", "close_away_odds", "sportsbook_quotes"]:
        if optional in games.columns:
            table_cols.append(optional)
    games[table_cols].to_csv(OUT / "pitcher_modeling_table.csv", index=False)

    print("\nDATA DIAGNOSTICS")
    print(diagnostics.to_string(index=False))
    print("\nWALK-FORWARD METRICS")
    print(metrics.round(4).to_string(index=False))
    if not betting.empty:
        print("\nEDGE BACKTEST (flat $1 closing-line stakes)")
        print(betting.round(4).to_string(index=False))
    print(f"\nFeatures: {len(features)} ({len(pitcher_feature_names)} pitcher, {len(team_feature_names)} team)")


if __name__ == "__main__":
    main()
