"""Grade frozen FanDuel pitcher-K paper selections and calculate CLV diagnostics."""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data/current/fanduel_pitcher_k_paper_history.csv"
LOG = ROOT / "data/mlb_pitcher_game_logs.csv"
ARCHIVE = ROOT / "data/market/free_archive"
OUT = ROOT / "outputs"
GRADED = OUT / "fanduel_pitcher_k_paper_graded.csv"
SUMMARY = OUT / "fanduel_pitcher_k_paper_summary.csv"
CALIB = OUT / "fanduel_pitcher_k_calibration.csv"
MODEL_SUMMARY = OUT / "fanduel_pitcher_k_model_version_summary.csv"
THRESHOLDS = (0.00, 0.025, 0.05, 0.075, 0.10)


def norm_name(v):
    x = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def implied_prob(price):
    x = pd.to_numeric(price, errors="coerce")
    if pd.isna(x) or x == 0:
        return np.nan
    return 100 / (x + 100) if x > 0 else -x / (-x + 100)


def win_profit(price):
    return float(price) / 100 if float(price) > 0 else 100 / (-float(price))


def binary_scores(x: pd.DataFrame):
    q = x[x.result.isin(["WIN", "LOSS"]) & x.model_win_prob.notna()].copy()
    if q.empty:
        return np.nan, np.nan
    y = q.result.eq("WIN").astype(float).to_numpy()
    p = np.clip(pd.to_numeric(q.model_win_prob, errors="coerce").to_numpy(float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, log_loss


def load_archive():
    # Prospective CLV must use only immutable verified-current FanDuel decision
    # snapshots. Public PropLine sample files in the same archive are research-only
    # and must never become sportsbook CLV evidence.
    fs = list(ARCHIVE.rglob("decision-*.csv")) if ARCHIVE.exists() else []
    parts = []
    for f in fs:
        try:
            q = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if q.empty or "collected_at_utc" not in q.columns:
            continue
        parts.append(q)
    if not parts:
        return pd.DataFrame()
    z = pd.concat(parts, ignore_index=True, sort=False)
    z["sportsbook"] = z.sportsbook.astype(str).str.lower()
    z["side"] = z.side.astype(str).str.upper()
    z["name_key"] = z.pitcher_name.map(norm_name)
    z["line"] = pd.to_numeric(z.line, errors="coerce")
    z["price"] = pd.to_numeric(z.price, errors="coerce")
    z["collected_at_utc"] = pd.to_datetime(z.collected_at_utc, errors="coerce", utc=True)
    z["commence_time_utc"] = pd.to_datetime(z.commence_time_utc, errors="coerce", utc=True)
    return z[z.sportsbook.eq("fanduel")].dropna(subset=["collected_at_utc", "commence_time_utc"])


def grade():
    if not HISTORY.exists():
        print("No FanDuel paper history yet.")
        return
    h = pd.read_csv(HISTORY, low_memory=False)
    if h.empty:
        print("FanDuel paper history is empty.")
        return
    h["game_id"] = pd.to_numeric(h.game_id, errors="coerce")
    h["pitcher_id"] = pd.to_numeric(h.pitcher_id, errors="coerce")
    h["line"] = pd.to_numeric(h.line, errors="coerce")
    h["fanduel_price"] = pd.to_numeric(h.fanduel_price, errors="coerce")
    h["model_win_prob"] = pd.to_numeric(h.model_win_prob, errors="coerce")
    h["model_market_edge"] = pd.to_numeric(h.model_market_edge, errors="coerce")
    h["collected_at_utc"] = pd.to_datetime(h.collected_at_utc, errors="coerce", utc=True)
    h["commence_time_utc"] = pd.to_datetime(h.commence_time_utc, errors="coerce", utc=True)
    h["name_key"] = h.pitcher_name.map(norm_name)

    logs = pd.read_csv(LOG, low_memory=False)
    logs["game_id"] = pd.to_numeric(logs.game_id, errors="coerce")
    logs["pitcher_id"] = pd.to_numeric(logs.pitcher_id, errors="coerce")
    logs["strikeouts"] = pd.to_numeric(logs.strikeouts, errors="coerce")
    logs["is_starter"] = pd.to_numeric(logs.get("is_starter"), errors="coerce").fillna(0).astype(int)
    completed_game_ids = set(logs.dropna(subset=["game_id"]).game_id.astype(float))
    actual = (logs.dropna(subset=["game_id", "pitcher_id"])
              .sort_values(["date", "game_id"])
              .drop_duplicates(["game_id", "pitcher_id"], keep="last")
              [["game_id", "pitcher_id", "strikeouts", "is_starter"]]
              .rename(columns={"strikeouts": "actual_k", "is_starter": "actual_is_starter"}))
    h = h.drop(columns=[c for c in ["actual_k", "actual_is_starter"] if c in h.columns]).merge(actual, on=["game_id", "pitcher_id"], how="left")

    results = []
    profits = []
    for r in h.itertuples(index=False):
        game_completed = pd.notna(r.game_id) and float(r.game_id) in completed_game_ids
        if pd.isna(r.actual_k):
            if game_completed:
                results.append("VOID_STARTER_CHANGE"); profits.append(0.0)
            else:
                results.append("PENDING"); profits.append(np.nan)
            continue
        if int(r.actual_is_starter or 0) != 1:
            results.append("VOID_STARTER_CHANGE"); profits.append(0.0); continue
        if float(r.actual_k) == float(r.line):
            results.append("PUSH"); profits.append(0.0)
        elif (r.side == "OVER" and float(r.actual_k) > float(r.line)) or (r.side == "UNDER" and float(r.actual_k) < float(r.line)):
            results.append("WIN"); profits.append(win_profit(r.fanduel_price))
        else:
            results.append("LOSS"); profits.append(-1.0)
    h["result"] = results
    h["flat_profit_units"] = profits

    arc = load_archive()
    h["closing_same_line_price"] = np.nan
    h["closing_same_line_implied_prob"] = np.nan
    h["clv_implied_prob"] = np.nan
    h["latest_same_side_line"] = np.nan
    h["latest_same_side_price"] = np.nan
    if not arc.empty:
        for i, r in h.iterrows():
            if pd.isna(r.collected_at_utc) or pd.isna(r.commence_time_utc):
                continue
            q = arc[
                arc.name_key.eq(r.name_key)
                & arc.side.eq(str(r.side).upper())
                & arc.collected_at_utc.gt(r.collected_at_utc)
                & arc.collected_at_utc.lt(r.commence_time_utc)
            ].copy()
            if "event_id" in arc.columns and pd.notna(r.get("event_id", np.nan)):
                q = q[q.event_id.astype(str).eq(str(r.event_id))]
            if q.empty:
                continue
            latest = q.sort_values("collected_at_utc").iloc[-1]
            h.at[i, "latest_same_side_line"] = latest.line
            h.at[i, "latest_same_side_price"] = latest.price
            same = q[np.isclose(pd.to_numeric(q.line, errors="coerce"), float(r.line), equal_nan=False)]
            if not same.empty:
                c = same.sort_values("collected_at_utc").iloc[-1]
                cp = float(c.price)
                h.at[i, "closing_same_line_price"] = cp
                cip = implied_prob(cp)
                h.at[i, "closing_same_line_implied_prob"] = cip
                h.at[i, "clv_implied_prob"] = cip - implied_prob(r.fanduel_price)

    OUT.mkdir(exist_ok=True)
    h.to_csv(GRADED, index=False)
    h.drop(columns=["name_key"], errors="ignore").to_csv(HISTORY, index=False)

    decided = h[h.result.isin(["WIN", "LOSS", "PUSH", "VOID_STARTER_CHANGE"])].copy()
    rows = []
    for t in THRESHOLDS:
        x = decided[decided.model_market_edge.ge(t)]
        wins = int(x.result.eq("WIN").sum()); losses = int(x.result.eq("LOSS").sum()); pushes = int(x.result.eq("PUSH").sum()); voids=int(x.result.eq("VOID_STARTER_CHANGE").sum())
        staked = wins + losses
        profit = float(x.flat_profit_units.fillna(0).sum())
        brier, logloss = binary_scores(x)
        rows.append({
            "min_edge": t, "independent_bets": staked, "pushes": pushes, "voids": voids, "wins": wins, "losses": losses,
            "win_rate_ex_push": wins / staked if staked else np.nan,
            "profit_units": profit, "roi": profit / staked if staked else np.nan,
            "brier_score_ex_push": brier, "log_loss_ex_push": logloss,
            "mean_clv_implied_prob": float(pd.to_numeric(x.clv_implied_prob, errors="coerce").mean()) if len(x) else np.nan,
        })
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)

    settled = decided[decided.result.isin(["WIN", "LOSS"]) & decided.model_win_prob.notna()].copy()
    if not settled.empty:
        settled["won"] = settled.result.eq("WIN").astype(int)
        settled["prob_bin"] = pd.cut(settled.model_win_prob, bins=[0,.4,.5,.6,.7,.8,1.0], include_lowest=True)
        cal = settled.groupby("prob_bin", observed=True).agg(
            bets=("won", "size"), mean_model_prob=("model_win_prob", "mean"), observed_win_rate=("won", "mean")
        ).reset_index()
    else:
        cal = pd.DataFrame(columns=["prob_bin", "bets", "mean_model_prob", "observed_win_rate"])
    cal.to_csv(CALIB, index=False)

    if "model_version" in decided.columns and not decided.empty:
        mr=[]
        for name,x in decided.groupby("model_version",dropna=False):
            wins=int(x.result.eq("WIN").sum()); losses=int(x.result.eq("LOSS").sum()); pushes=int(x.result.eq("PUSH").sum()); voids=int(x.result.eq("VOID_STARTER_CHANGE").sum())
            staked=wins+losses; profit=float(x.flat_profit_units.fillna(0).sum()); brier,logloss=binary_scores(x)
            mr.append({"model_version":name,"independent_bets":staked,"pushes":pushes,"voids":voids,"wins":wins,"losses":losses,
                       "win_rate_ex_push":wins/staked if staked else np.nan,"profit_units":profit,"roi":profit/staked if staked else np.nan,
                       "brier_score_ex_push":brier,"log_loss_ex_push":logloss,
                       "mean_clv_implied_prob":float(pd.to_numeric(x.clv_implied_prob,errors="coerce").mean())})
        pd.DataFrame(mr).to_csv(MODEL_SUMMARY,index=False)

    print(f"FanDuel paper ledger: {len(h)} selections; settled bets={h.result.isin(['WIN','LOSS','PUSH']).sum()}; voids={h.result.eq('VOID_STARTER_CHANGE').sum()}; pending={h.result.eq('PENDING').sum()}")
    if SUMMARY.exists(): print(pd.read_csv(SUMMARY).round(5).to_string(index=False))


if __name__ == "__main__":
    grade()
