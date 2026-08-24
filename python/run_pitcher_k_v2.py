"""Create a parallel Pitcher-K V2 research projection from V1 + posted lineup context.

This does not alter V1. V2 applies a conservative lineup K-rate adjustment to the
V1 projected strikeout mean, capped at +/-15%. The posted lineup is compared with
that specific opponent team's recent K/PA; league recent K/PA is only a fallback.
"""
from __future__ import annotations

from math import exp, floor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
V1 = OUT / "pitcher_k_prop_predictions.csv"
CTX = CURRENT / "pitcher_k_v2_lineup_context.csv"
V2 = OUT / "pitcher_k_v2_predictions.csv"
HIST = CURRENT / "pitcher_k_v2_projection_history.csv"


def poisson_cdf(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    term = exp(-mu)
    total = term
    for i in range(1, k + 1):
        term *= mu / i
        total += term
    return min(max(total, 0.0), 1.0)


def load_team_logs() -> pd.DataFrame:
    logs = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
    logs["team_id"] = pd.to_numeric(logs["team_id"], errors="coerce")
    for c in ["strikeouts", "at_bats", "walks"]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    return logs


def k_rate(z: pd.DataFrame) -> float:
    pa = z["at_bats"].sum() + z["walks"].sum()
    return float(z["strikeouts"].sum() / pa) if pa > 0 else np.nan


def baseline_team_k_rate(opponent_team_id: int, day: str, logs: pd.DataFrame) -> tuple[float, str]:
    target = pd.Timestamp(day)
    team = logs[(logs.team_id == opponent_team_id) & (logs.date < target)].sort_values(["date", "game_id"]).tail(30)
    rate = k_rate(team) if not team.empty else np.nan
    if pd.notna(rate) and 0.05 <= rate <= 0.50:
        return float(rate), "opponent_last_30_games"

    league = logs[logs.date < target].sort_values("date").tail(30 * 15)
    fallback = k_rate(league)
    return (float(fallback) if pd.notna(fallback) else 0.23), "league_recent_fallback"


def append_history(fresh: pd.DataFrame) -> None:
    if fresh.empty:
        return
    if HIST.exists():
        old = pd.read_csv(HIST, low_memory=False)
        out = pd.concat([old, fresh], ignore_index=True, sort=False)
    else:
        out = fresh.copy()
    keys = [c for c in ["date", "event_id", "pitcher_id", "line", "snapshot_time_et", "v2_model"] if c in out.columns]
    out = out.drop_duplicates(keys, keep="last")
    out.to_csv(HIST, index=False)


def main() -> None:
    if not V1.exists():
        raise SystemExit("Missing V1 pitcher K predictions.")
    if not CTX.exists():
        raise SystemExit("Missing V2 lineup context.")
    v1 = pd.read_csv(V1, low_memory=False)
    ctx = pd.read_csv(CTX, low_memory=False)
    if v1.empty or ctx.empty:
        pd.DataFrame().to_csv(V2, index=False)
        print("No posted pregame lineup context available for V2 scoring.")
        return

    ctx["pitcher_id"] = pd.to_numeric(ctx["pitcher_id"], errors="coerce")
    ctx["opponent_team_id"] = pd.to_numeric(ctx["opponent_team_id"], errors="coerce")
    summaries = ctx.groupby(["game_id", "pitcher_id"], dropna=False).agg(
        lineup_batters=("batter_id", "nunique"),
        lineup_weighted_k_per_pa=("lineup_weighted_k_per_pa", "first"),
        pitcher_hand=("pitcher_hand", "first"),
        opponent_team_id=("opponent_team_id", "first"),
        lineup_snapshot_time_et=("snapshot_time_et", "first"),
        game_start_et=("game_start_et", "first"),
    ).reset_index()

    # V1 game_id is MLB's gamePk. Match both game and pitcher so doubleheaders cannot cross-join.
    v1["game_id"] = pd.to_numeric(v1["game_id"], errors="coerce")
    v1["pitcher_id"] = pd.to_numeric(v1["pitcher_id"], errors="coerce")
    z = v1.merge(summaries, on=["game_id", "pitcher_id"], how="inner")
    logs = load_team_logs()
    rows = []
    for r in z.itertuples(index=False):
        if r.lineup_batters < 9 or pd.isna(r.lineup_weighted_k_per_pa) or pd.isna(r.opponent_team_id):
            continue
        neutral, baseline_source = baseline_team_k_rate(int(r.opponent_team_id), str(r.date), logs)
        ratio = float(r.lineup_weighted_k_per_pa) / neutral if neutral > 0 else 1.0
        factor = float(np.clip(ratio, 0.85, 1.15))
        mu = float(np.clip(r.projected_k * factor, 0.05, None))
        line = float(r.line)
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
        mo = pd.to_numeric(getattr(r, "market_over_prob_no_vig", np.nan), errors="coerce")
        market_under = pd.to_numeric(getattr(r, "market_under_prob_no_vig", np.nan), errors="coerce")
        over_edge = p_over - mo if pd.notna(mo) else np.nan
        under_edge = p_under - market_under if pd.notna(market_under) else np.nan
        side = "OVER" if pd.notna(over_edge) and (pd.isna(under_edge) or over_edge >= under_edge) else "UNDER"
        edge = max(over_edge, under_edge) if pd.notna(over_edge) and pd.notna(under_edge) else np.nan
        d = r._asdict()
        d.update({
            "v2_model": "v2_lineup_k_rate_0_2",
            "v1_projected_k": float(r.projected_k),
            "lineup_weighted_k_per_pa": float(r.lineup_weighted_k_per_pa),
            "opponent_recent_k_per_pa": neutral,
            "opponent_baseline_source": baseline_source,
            "lineup_adjustment_factor": factor,
            "v2_projected_k": mu,
            "v2_fair_over_prob": p_over,
            "v2_fair_under_prob": p_under,
            "v2_push_prob": p_push,
            "v2_research_side": side,
            "v2_model_market_edge": edge,
            "v2_decision": "NO BET - V2 prospective validation only",
        })
        rows.append(d)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("v2_model_market_edge", ascending=False)
    out.to_csv(V2, index=False)
    append_history(out)
    print(f"V2 scored {len(out)} pregame posted-lineup pitcher/line rows; V1 remains unchanged.")


if __name__ == "__main__":
    main()
