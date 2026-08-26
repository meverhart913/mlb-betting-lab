"""Grade accumulated free pitcher-K market snapshots against completed MLB appearances.

This does not evaluate model profitability. It attaches actual strikeouts to every
archived quote once the game is final so the market history is permanently ready
for later V2.2/V2.x backtests. Starter eligibility is preserved separately for
model matching instead of blocking outcome grading.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "market" / "free_archive"
PITCHERS = ROOT / "data" / "mlb_pitcher_game_logs.csv"
OUT = ROOT / "data" / "market" / "free_pitcher_k_archive_graded.csv"
STATUS = ROOT / "outputs" / "free_pitcher_k_archive_status.csv"


def norm_name(v) -> str:
    x = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", x)


def main() -> None:
    files = sorted(ARCHIVE.rglob("propline_pitcher_k_*.csv"))
    if not files:
        raise SystemExit("No archived free PropLine pitcher-K files found.")
    if not PITCHERS.exists():
        raise SystemExit("Missing data/mlb_pitcher_game_logs.csv")

    parts = []
    for f in files:
        z = pd.read_csv(f, low_memory=False)
        z["archive_file"] = str(f.relative_to(ARCHIVE))
        parts.append(z)
    market = pd.concat(parts, ignore_index=True, sort=False)
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["name_key"] = market["pitcher_name"].map(norm_name)
    if "snapshot_time_et" in market.columns:
        market["snapshot_time_et"] = pd.to_datetime(market["snapshot_time_et"], errors="coerce")

    p = pd.read_csv(PITCHERS, low_memory=False)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["is_starter"] = pd.to_numeric(p.get("is_starter"), errors="coerce").fillna(0).astype(int)
    p["strikeouts"] = pd.to_numeric(p["strikeouts"], errors="coerce")
    p["name_key"] = p["pitcher_name"].map(norm_name)
    # Grade any pitcher who actually appeared. V2.2 eligibility remains a later
    # filter because the model itself only generates starting-pitcher forecasts.
    p = p[p.strikeouts.notna()].copy()
    result_cols = ["date", "name_key", "game_id", "pitcher_id", "pitcher_name", "strikeouts", "is_starter"]
    results = p[[c for c in result_cols if c in p.columns]].copy()
    results = results.rename(columns={
        "pitcher_name": "mlb_pitcher_name",
        "strikeouts": "actual_k",
        "is_starter": "mlb_is_starter",
    })
    results = results.drop_duplicates(["date", "name_key"], keep="last")

    g = market.merge(results, on=["date", "name_key"], how="left")
    g["line"] = pd.to_numeric(g["line"], errors="coerce")
    g["actual_k"] = pd.to_numeric(g["actual_k"], errors="coerce")
    side = g["side"].astype(str).str.lower()
    graded = g.actual_k.notna() & g.line.notna() & side.isin(["over", "under"])
    g["market_result"] = "PENDING"
    push = graded & g.actual_k.eq(g.line)
    win = graded & ~push & (((side == "over") & (g.actual_k > g.line)) | ((side == "under") & (g.actual_k < g.line)))
    loss = graded & ~push & ~win
    g.loc[push, "market_result"] = "PUSH"
    g.loc[win, "market_result"] = "WIN"
    g.loc[loss, "market_result"] = "LOSS"

    keys = [c for c in ["date", "pitcher_name", "line", "side", "price", "sportsbook", "event_id", "snapshot_time_et"] if c in g.columns]
    if keys:
        g = g.drop_duplicates(keys, keep="last")
    g = g.sort_values([c for c in ["date", "pitcher_name", "snapshot_time_et", "sportsbook", "side"] if c in g.columns])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT, index=False)

    STATUS.parent.mkdir(exist_ok=True)
    status = pd.DataFrame([{
        "archive_files": len(files),
        "quote_rows": len(g),
        "unique_dates": int(g.date.nunique()),
        "unique_pitchers": int(g.name_key.nunique()),
        "graded_rows": int(g.actual_k.notna().sum()),
        "starter_graded_rows": int((g.actual_k.notna() & pd.to_numeric(g.get("mlb_is_starter"), errors="coerce").eq(1)).sum()),
        "pending_rows": int(g.actual_k.isna().sum()),
        "earliest_date": g.date.min().date() if len(g) else None,
        "latest_date": g.date.max().date() if len(g) else None,
    }])
    status.to_csv(STATUS, index=False)
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
