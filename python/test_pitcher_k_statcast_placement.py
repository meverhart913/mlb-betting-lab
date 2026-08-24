"""Walk-forward ablation for where Statcast belongs in the Pitcher-K ensemble.

Requires the full-coverage leakage-safe Statcast pregame table. Compares the
current BF x K-rate + direct architecture with Statcast added selectively to:
BF/opportunity, K-rate, direct K, or combinations. Research-only V2.1 gate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from build_pitcher_k_model import build_table
from test_pitcher_k_ensemble import hgb, specialized_features

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features" / "statcast_pitcher_pregame.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def fit_predict(z, train, test, bf_feats, kr_feats, direct_feats, component_weight=0.5):
    bf = hgb("poisson", leaves=12, l2=4.0)
    kr = hgb("squared_error", leaves=12, l2=4.0)
    direct = hgb("poisson", leaves=15, l2=2.0)
    bf.fit(z.loc[train, bf_feats], z.loc[train, "batters_faced"])
    kr.fit(
        z.loc[train, kr_feats], z.loc[train, "k_rate_target"],
        m__sample_weight=z.loc[train, "batters_faced"],
    )
    direct.fit(z.loc[train, direct_feats], z.loc[train, "strikeouts"])
    bf_hat = np.clip(bf.predict(z.loc[test, bf_feats]), 5, 40)
    kr_hat = np.clip(kr.predict(z.loc[test, kr_feats]), 0.02, 0.55)
    component = np.clip(bf_hat * kr_hat, 0.05, None)
    direct_hat = np.clip(direct.predict(z.loc[test, direct_feats]), 0.05, None)
    mu = np.clip(component_weight * component + (1-component_weight) * direct_hat, 0.05, None)
    return mu


def main():
    if not FEATURES.exists():
        raise SystemExit("Missing full-coverage Statcast pregame features")
    z = build_table().copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    z["pitcher_id"] = pd.to_numeric(z["pitcher_id"], errors="coerce")
    z = z[pd.to_numeric(z["batters_faced"], errors="coerce").gt(0)].copy()
    z["k_rate_target"] = (z["strikeouts"] / z["batters_faced"]).clip(0, .7)

    sc = pd.read_csv(FEATURES, low_memory=False)
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")
    sc["pitcher_id"] = pd.to_numeric(sc["pitcher_id"], errors="coerce")
    sc = sc.rename(columns={"game_date":"date"})
    stat = [c for c in sc.columns if c.startswith("statcast_")]
    z = z.merge(sc[["date","pitcher_id",*stat]], on=["date","pitcher_id"], how="left")
    bf0, kr0, direct0 = specialized_features(z)

    placements = {
        "baseline": (bf0, kr0, direct0),
        "stat_bf": (bf0+stat, kr0, direct0),
        "stat_kr": (bf0, kr0+stat, direct0),
        "stat_direct": (bf0, kr0, direct0+stat),
        "stat_bf_kr": (bf0+stat, kr0+stat, direct0),
        "stat_kr_direct": (bf0, kr0+stat, direct0+stat),
        "stat_all": (bf0+stat, kr0+stat, direct0+stat),
    }
    rows=[]
    for year in range(2022, 2027):
        train=z["season"]<year; test=z["season"]==year
        if train.sum()<1500 or test.sum()<300: continue
        coverage=float(z.loc[test,stat].notna().any(axis=1).mean()) if stat else 0
        y=z.loc[test,"strikeouts"].to_numpy(float)
        for label,(bf,kr,direct) in placements.items():
            mu=fit_predict(z,train,test,sorted(set(bf)),sorted(set(kr)),sorted(set(direct)))
            rows.append({"season":year,"placement":label,"starts":int(test.sum()),
                         "statcast_coverage":coverage,"mae":mean_absolute_error(y,mu),
                         "rmse":root_mean_squared_error(y,mu),
                         "poisson_deviance":mean_poisson_deviance(y,mu)})
    m=pd.DataFrame(rows)
    s=m.groupby("placement",as_index=False).agg(seasons=("season","nunique"),
        mean_coverage=("statcast_coverage","mean"),mean_mae=("mae","mean"),
        mean_rmse=("rmse","mean"),mean_poisson_deviance=("poisson_deviance","mean"))
    s=s.sort_values(["mean_poisson_deviance","mean_mae"])
    m.to_csv(OUT/"pitcher_k_statcast_placement_metrics.csv",index=False)
    s.to_csv(OUT/"pitcher_k_statcast_placement_summary.csv",index=False)
    print(s.round(6).to_string(index=False))

if __name__=="__main__": main()
