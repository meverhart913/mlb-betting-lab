"""As-of lookup for leakage-safe batter K features by opposing pitcher handedness.

Given a lineup date, batter IDs, and opposing pitcher hand, return each batter's
most recent feature row strictly before that date. This avoids exact-date joins,
because the rolling table only contains dates on which the batter previously
faced that hand.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "features" / "batter_k_by_pitcher_hand_pregame.csv"


def load_table(path: Path = SRC) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce")
    df["pitcher_hand"] = df["pitcher_hand"].astype(str).str.upper()
    return df[df.game_date.notna() & df.batter_id.notna()].sort_values(
        ["batter_id", "pitcher_hand", "game_date"]
    )


def lookup(lineup: pd.DataFrame, target_date, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return latest row strictly before target_date for each batter/hand pair."""
    if table is None:
        table = load_table()
    target = pd.Timestamp(target_date)
    q = lineup[["batter_id", "pitcher_hand"]].copy()
    q["batter_id"] = pd.to_numeric(q["batter_id"], errors="coerce")
    q["pitcher_hand"] = q["pitcher_hand"].astype(str).str.upper()
    q["target_date"] = target

    prior = table[table["game_date"] < target].copy()
    latest = (
        prior.sort_values("game_date")
        .groupby(["batter_id", "pitcher_hand"], as_index=False, sort=False)
        .tail(1)
    )
    out = q.merge(latest, on=["batter_id", "pitcher_hand"], how="left")
    out["feature_age_days"] = (out["target_date"] - out["game_date"]).dt.days
    return out


if __name__ == "__main__":
    raise SystemExit("Import lookup() from a V2.1 scorer; this module has no standalone job.")
