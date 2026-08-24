"""Test leakage-safe bullpen workload features against the current combined model.

Bullpen features use only relief appearances on calendar days before each game.
The experiment is intentionally separate from production feature selection until
it shows stable expanding-season walk-forward improvement.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def bullpen_features() -> pd.DataFrame:
    pitchers = pd.read_csv(DATA / "mlb_pitcher_game_logs.csv", low_memory=False)
    teams = pd.read_csv(DATA / "mlb_team_game_logs.csv", low_memory=False)
    pitchers["date"] = pd.to_datetime(pitchers["date"], errors="coerce").dt.normalize()
    teams["date"] = pd.to_datetime(teams["date"], errors="coerce").dt.normalize()
    pitchers["is_starter"] = pd.to_numeric(pitchers["is_starter"], errors="coerce").fillna(0).astype(int)
    pitchers["pitches"] = pd.to_numeric(pitchers["pitches"], errors="coerce").fillna(0)

    rel = pitchers[pitchers["is_starter"] == 0].merge(
        teams[["game_id", "side", "team_id"]].drop_duplicates(["game_id", "side"]),
        on=["game_id", "side"], how="left",
    )
    rel = rel[rel["team_id"].notna() & rel["date"].notna()].copy()

    daily = {}
    for (team_id, day), g in rel.groupby(["team_id", "date"]):
        by_pitcher = g.groupby("pitcher_id")["pitches"].sum()
        daily[(team_id, day)] = {
            "pitches": float(by_pitcher.sum()),
            "apps": int(len(g)),
            "unique": int(by_pitcher.size),
            "twenty": int((by_pitcher >= 20).sum()),
            "max": float(by_pitcher.max()) if len(by_pitcher) else 0.0,
            "pitchers": set(by_pitcher.index.tolist()),
        }

    rows = []
    sched = teams[["game_id", "date", "side", "team_id"]].drop_duplicates(["game_id", "side"])
    for r in sched.itertuples(index=False):
        if pd.isna(r.date) or pd.isna(r.team_id):
            continue
        d1, d2, d3 = r.date - pd.Timedelta(days=1), r.date - pd.Timedelta(days=2), r.date - pd.Timedelta(days=3)
        q1 = daily.get((r.team_id, d1), {})
        q2 = daily.get((r.team_id, d2), {})
        q3 = daily.get((r.team_id, d3), {})
        def val(q, k): return q.get(k, 0.0)
        s1, s2, s3 = q1.get("pitchers", set()), q2.get("pitchers", set()), q3.get("pitchers", set())
        rows.append({
            "game_id": r.game_id,
            "side": r.side,
            "bp_pitches_1d": val(q1, "pitches"),
            "bp_pitches_2d": val(q1, "pitches") + val(q2, "pitches"),
            "bp_pitches_3d": val(q1, "pitches") + val(q2, "pitches") + val(q3, "pitches"),
            "bp_apps_1d": val(q1, "apps"),
            "bp_apps_2d": val(q1, "apps") + val(q2, "apps"),
            "bp_unique_1d": val(q1, "unique"),
            "bp_unique_2d": len(s1 | s2),
            "bp_20plus_1d": val(q1, "twenty"),
            "bp_max_pitches_1d": val(q1, "max"),
            "bp_back_to_back": len(s1 & s2),
            "bp_three_straight": len(s1 & s2 & s3),
        })
    long = pd.DataFrame(rows)
    feature_names = [c for c in long.columns if c.startswith("bp_")]
    wide = long.pivot(index="game_id", columns="side", values=feature_names)
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    wide = wide.reset_index()
    for name in feature_names:
        h, a = f"home_{name}", f"away_{name}"
        if h in wide.columns and a in wide.columns:
            wide[f"diff_{name}"] = wide[h] - wide[a]
    return wide[["game_id"] + [c for c in wide.columns if c.startswith("diff_bp_")]]


def model() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(max_iter=2500, C=0.5)),
    ])


def evaluate(frame: pd.DataFrame, features: list[str], label: str) -> list[dict]:
    rows = []
    years = frame["date"].dt.year
    for year in range(max(2021, int(years.min()) + 2), int(years.max()) + 1):
        train = years < year
        test = years == year
        if train.sum() < 500 or test.sum() == 0:
            continue
        m = model()
        m.fit(frame.loc[train, features], frame.loc[train, "home_win"].astype(int))
        p = m.predict_proba(frame.loc[test, features])[:, 1]
        y = frame.loc[test, "home_win"].astype(int)
        rows.append({
            "season": year, "feature_set": label, "games": int(test.sum()),
            "accuracy": accuracy_score(y, p >= 0.5),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
        })
    return rows


def main() -> None:
    base = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base[base["home_win"].notna() & base["date"].notna()].copy()
    bp = bullpen_features()
    z = base.merge(bp, on="game_id", how="left")

    baseline = sorted([c for c in z.columns if c.startswith("diff_sp_") or c.startswith("diff_team_")])
    bp_cols = sorted([c for c in z.columns if c.startswith("diff_bp_")])
    if not baseline or not bp_cols:
        raise ValueError(f"Expected baseline and bullpen features; got baseline={len(baseline)}, bullpen={len(bp_cols)}")

    rows = evaluate(z, baseline, "team_plus_pitcher")
    rows += evaluate(z, baseline + bp_cols, "team_plus_pitcher_plus_bullpen")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bullpen_feature_comparison.csv", index=False)
    summary = out.groupby("feature_set")[["log_loss", "brier", "auc", "accuracy"]].mean().reset_index()
    summary.to_csv(OUT / "bullpen_feature_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Bullpen features tested: {len(bp_cols)}")


if __name__ == "__main__":
    main()
