"""Re-evaluate model predictions against validated American moneyline prices.

Historical odds exports contain a small number of malformed rows where spread
values (for example -1.5) appear in moneyline fields. This evaluator treats the
consensus closing prices as usable only when both sides look like plausible
American odds, recomputes the no-vig market probability from those prices, and
then calculates benchmark scoring and flat-$1 edge backtests.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def implied(x: pd.Series) -> np.ndarray:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    return np.where(v > 0, 100.0 / (v + 100.0), np.where(v < 0, -v / (-v + 100.0), np.nan))


def win_profit(x: np.ndarray) -> np.ndarray:
    v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    return np.where(v > 0, v / 100.0, np.where(v < 0, 100.0 / -v, np.nan))


def main() -> None:
    path = OUT / "pitcher_model_walkforward_predictions.csv"
    p = pd.read_csv(path, low_memory=False)
    required = {"date", "home_win", "model", "model_home_prob", "close_home_odds", "close_away_odds"}
    missing = sorted(required - set(p.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {', '.join(missing)}")

    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["season"] = p["date"].dt.year
    p["close_home_odds"] = pd.to_numeric(p["close_home_odds"], errors="coerce")
    p["close_away_odds"] = pd.to_numeric(p["close_away_odds"], errors="coerce")

    # American moneyline prices are conventionally <= -100 or >= +100.
    # A generous 5000 cap removes obvious corrupt values without trimming any
    # realistic MLB price we should rely on for this research dataset.
    valid_home = p["close_home_odds"].abs().between(100, 5000, inclusive="both")
    valid_away = p["close_away_odds"].abs().between(100, 5000, inclusive="both")
    p["valid_market"] = valid_home & valid_away
    p.loc[~p["valid_market"], ["close_home_odds", "close_away_odds"]] = np.nan

    ph = implied(p["close_home_odds"])
    pa = implied(p["close_away_odds"])
    p["clean_market_home_prob"] = ph / (ph + pa)

    metric_rows = []
    bet_rows = []
    for (season, model), z in p.groupby(["season", "model"], dropna=True):
        z = z.dropna(subset=["home_win", "model_home_prob"])
        m = z.dropna(subset=["clean_market_home_prob"])
        row = {
            "season": int(season),
            "model": model,
            "games": int(len(z)),
            "valid_market_games": int(len(m)),
            "invalid_or_missing_market_games": int(len(z) - len(m)),
            "model_log_loss": log_loss(z["home_win"].astype(int), z["model_home_prob"], labels=[0, 1]),
            "model_brier": brier_score_loss(z["home_win"].astype(int), z["model_home_prob"]),
        }
        if len(m):
            row["market_log_loss"] = log_loss(m["home_win"].astype(int), m["clean_market_home_prob"], labels=[0, 1])
            row["market_brier"] = brier_score_loss(m["home_win"].astype(int), m["clean_market_home_prob"])
            mm = z.loc[m.index]
            row["model_log_loss_market_subset"] = log_loss(mm["home_win"].astype(int), mm["model_home_prob"], labels=[0, 1])
            row["model_brier_market_subset"] = brier_score_loss(mm["home_win"].astype(int), mm["model_home_prob"])
        metric_rows.append(row)

        for threshold in (0.02, 0.03, 0.05, 0.075, 0.10):
            b = m.copy()
            if b.empty:
                continue
            edge = b["model_home_prob"] - b["clean_market_home_prob"]
            b["side"] = np.where(edge >= threshold, "home", np.where(edge <= -threshold, "away", "pass"))
            b = b[b["side"] != "pass"].copy()
            if b.empty:
                bet_rows.append({"season": int(season), "model": model, "edge_threshold": threshold, "bets": 0, "wins": 0, "hit_rate": np.nan, "roi": np.nan})
                continue
            b["won"] = np.where(b["side"] == "home", b["home_win"] == 1, b["home_win"] == 0)
            price = np.where(b["side"] == "home", b["close_home_odds"], b["close_away_odds"])
            b["profit"] = np.where(b["won"], win_profit(price), -1.0)
            bet_rows.append({
                "season": int(season),
                "model": model,
                "edge_threshold": threshold,
                "bets": int(len(b)),
                "wins": int(b["won"].sum()),
                "hit_rate": float(b["won"].mean()),
                "roi": float(b["profit"].mean()),
            })

    metrics = pd.DataFrame(metric_rows)
    bets = pd.DataFrame(bet_rows)
    metrics.to_csv(OUT / "clean_market_comparison.csv", index=False)
    bets.to_csv(OUT / "clean_market_edge_backtest.csv", index=False)

    print("\nVALIDATED MARKET COMPARISON")
    print(metrics.round(4).to_string(index=False))
    print("\nVALIDATED EDGE BACKTEST")
    print(bets.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
