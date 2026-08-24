"""Test a mechanics-based pitcher K projection: expected BF x expected K rate.

This challenger separates opportunity (batters faced) from strikeout skill and
matchup. It uses the same leakage-safe feature table and expanding-season
walk-forwards as the direct count model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline

from build_pitcher_k_model import build_table, feature_cols

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def model(loss="squared_error"):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=3.0, random_state=913)),
    ])


def main() -> None:
    z = build_table()
    z = z[pd.to_numeric(z["batters_faced"], errors="coerce").gt(0)].copy()
    z["k_rate_target"] = (z["strikeouts"] / z["batters_faced"]).clip(0, 0.7)
    feats = feature_cols(z)
    rows=[]; preds=[]
    for year in sorted(z["season"].dropna().astype(int).unique()):
        if year < 2022: continue
        train=z["season"]<year; test=z["season"]==year
        if train.sum()<1500 or test.sum()<300: continue
        bf=model("poisson"); kr=model("squared_error")
        bf.fit(z.loc[train,feats], z.loc[train,"batters_faced"])
        kr.fit(z.loc[train,feats], z.loc[train,"k_rate_target"], m__sample_weight=z.loc[train,"batters_faced"])
        bf_hat=np.clip(bf.predict(z.loc[test,feats]), 5, 40)
        kr_hat=np.clip(kr.predict(z.loc[test,feats]), 0.02, 0.55)
        mu=np.clip(bf_hat*kr_hat,0.05,None)
        y=z.loc[test,"strikeouts"].to_numpy(float)
        rows.append({"season":year,"model":"bf_x_k_rate","starts":int(test.sum()),"mae":mean_absolute_error(y,mu),"rmse":root_mean_squared_error(y,mu),"poisson_deviance":mean_poisson_deviance(y,mu),"mean_actual_k":float(y.mean()),"mean_projected_k":float(mu.mean())})
        q=z.loc[test,["game_id","date","pitcher_id","pitcher_name","strikeouts","batters_faced"]].copy()
        q["projected_bf"]=bf_hat; q["projected_k_rate"]=kr_hat; q["projected_k"]=mu; preds.append(q)
    metrics=pd.DataFrame(rows); predictions=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()
    metrics.to_csv(OUT/"pitcher_k_component_metrics.csv",index=False)
    predictions.to_csv(OUT/"pitcher_k_component_predictions.csv",index=False)
    print(metrics.round(5).to_string(index=False))

if __name__=="__main__": main()
