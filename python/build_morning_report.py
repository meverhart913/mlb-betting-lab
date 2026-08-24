"""Assemble model output plus automated pregame context into one game-level report.

This is a reporting layer, not a feature-promotion shortcut. Weather, bullpen,
roster/IL, and lineup fields remain informational until historically validated.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
OUT = ROOT / "outputs" / "morning_report.csv"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def side_pivot(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    if df.empty or "game_id" not in df or "side" not in df:
        return pd.DataFrame(columns=["game_id"])
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["game_id"])
    wide = df[["game_id", "side"] + cols].drop_duplicates(["game_id", "side"], keep="last").pivot(index="game_id", columns="side")
    wide.columns = [f"{side}_{prefix}{name}" for name, side in wide.columns]
    return wide.reset_index()


def main() -> None:
    pred = read(ROOT / "outputs" / "morning_model_predictions.csv")
    if pred.empty:
        raise SystemExit("Missing morning_model_predictions.csv; run the morning model first.")

    odds = read(CURRENT / "morning_odds.csv")
    weather = read(CURRENT / "morning_context.csv")
    bullpen = read(CURRENT / "bullpen_context.csv")
    roster = read(CURRENT / "roster_context.csv")
    lineup = read(CURRENT / "lineup_context.csv")

    report = pred.copy()
    if not odds.empty:
        extra = [c for c in [
            "date", "away_team", "home_team", "quote_count",
            "best_away_moneyline", "best_away_sportsbook", "best_home_moneyline", "best_home_sportsbook",
        ] if c in odds.columns]
        report = report.merge(odds[extra].drop_duplicates(["date", "away_team", "home_team"]),
                              on=["date", "away_team", "home_team"], how="left")

    if not weather.empty and "game_id" in weather:
        wcols = [c for c in [
            "game_id", "game_datetime_utc", "venue_name", "roof_type", "turf_type",
            "temperature_f", "precip_probability_pct", "precipitation_in", "wind_mph", "wind_gust_mph",
        ] if c in weather.columns]
        report = report.merge(weather[wcols].drop_duplicates("game_id"), on="game_id", how="left")

    bp_cols = [
        "bullpen_pitches_1d", "bullpen_pitches_2d", "bullpen_pitches_3d",
        "relief_appearances_1d", "unique_relievers_1d", "relievers_20plus_1d",
        "relievers_30plus_2d", "back_to_back_relievers", "three_straight_relievers",
        "max_reliever_pitches_1d",
    ]
    bpw = side_pivot(bullpen, bp_cols, "")
    if len(bpw.columns) > 1:
        report = report.merge(bpw, on="game_id", how="left")

    luw = side_pivot(lineup, ["lineup_player_count", "lineup_status"], "")
    if len(luw.columns) > 1:
        report = report.merge(luw, on="game_id", how="left")

    if not roster.empty and "team_name" in roster:
        rcols = [c for c in ["team_name", "active_roster_count", "active_pitchers", "active_position_players", "inferred_il_count"] if c in roster.columns]
        away = roster[rcols].drop_duplicates("team_name").rename(columns={c: f"away_{c}" for c in rcols if c != "team_name"})
        home = roster[rcols].drop_duplicates("team_name").rename(columns={c: f"home_{c}" for c in rcols if c != "team_name"})
        report = report.merge(away, left_on="away_team", right_on="team_name", how="left").drop(columns=["team_name"], errors="ignore")
        report = report.merge(home, left_on="home_team", right_on="team_name", how="left").drop(columns=["team_name"], errors="ignore")

    # Human-readable context flags. These do not alter the model decision.
    def fatigue(row, side):
        p = pd.to_numeric(row.get(f"{side}_bullpen_pitches_2d"), errors="coerce")
        b2b = pd.to_numeric(row.get(f"{side}_back_to_back_relievers"), errors="coerce")
        if (pd.notna(p) and p >= 120) or (pd.notna(b2b) and b2b >= 3):
            return "high"
        if (pd.notna(p) and p >= 70) or (pd.notna(b2b) and b2b >= 1):
            return "moderate"
        return "low"
    report["away_bullpen_fatigue_flag"] = report.apply(lambda r: fatigue(r, "away"), axis=1)
    report["home_bullpen_fatigue_flag"] = report.apply(lambda r: fatigue(r, "home"), axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.sort_values(["date", "game_id"]).to_csv(OUT, index=False)
    print(f"Wrote {len(report):,} game rows to {OUT} with model, price, weather, bullpen, roster/IL, and lineup context.")


if __name__ == "__main__":
    main()
