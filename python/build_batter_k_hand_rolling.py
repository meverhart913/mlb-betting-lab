"""Build leakage-safe batter K/PA features by opposing pitcher handedness."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features"
SRC = DATA / "batter_k_by_pitcher_hand_daily.csv"
OUT = DATA / "batter_k_by_pitcher_hand_pregame.csv"
WINDOWS = (30, 90, 365)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}; run backfill_batter_k_hand_splits.py first.")
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        df.to_csv(OUT, index=False)
        return

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce")
    df["plate_appearances"] = pd.to_numeric(df["plate_appearances"], errors="coerce").fillna(0)
    df["strikeouts"] = pd.to_numeric(df["strikeouts"], errors="coerce").fillna(0)
    df = df[df["game_date"].notna() & df["batter_id"].notna()].sort_values(["batter_id", "pitcher_hand", "game_date"])

    rows = []
    for (batter_id, hand), g in df.groupby(["batter_id", "pitcher_hand"], sort=False):
        g = g.sort_values("game_date").copy()
        for r in g.itertuples(index=False):
            prior = g[g["game_date"] < r.game_date]
            row = {"game_date": r.game_date, "batter_id": batter_id, "pitcher_hand": hand}
            for days in WINDOWS:
                cutoff = r.game_date - pd.Timedelta(days=days)
                z = prior[prior["game_date"] >= cutoff]
                pa = float(z["plate_appearances"].sum())
                k = float(z["strikeouts"].sum())
                row[f"batter_k_pa_{days}d"] = k / pa if pa > 0 else pd.NA
                row[f"batter_pa_{days}d"] = pa
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} leakage-safe batter/hand pregame rows to {OUT}.")


if __name__ == "__main__":
    main()
