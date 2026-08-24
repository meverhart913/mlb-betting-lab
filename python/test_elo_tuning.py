"""Tune leakage-safe MLB Elo and test incremental value vs baseball and market baselines."""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def implied(v):
    x = pd.to_numeric(v, errors="coerce").astype(float)
    return np.where(x > 0, 100/(x+100), np.where(x < 0, -x/(-x+100), np.nan))


def build_elo(games: pd.DataFrame, k: float, home_adv: float, carryover: float) -> pd.DataFrame:
    g = games.sort_values(["date", "game_id"]).copy()
    ratings: dict[str, float] = {}
    current_year = None
    rows = []
    for r in g.itertuples(index=False):
        year = pd.Timestamp(r.date).year
        if current_year is None:
            current_year = year
        elif year != current_year:
            ratings = {t: 1500 + carryover * (rt - 1500) for t, rt in ratings.items()}
            current_year = year
        h = str(r.home_team); a = str(r.away_team)
        rh = ratings.get(h, 1500.0); ra = ratings.get(a, 1500.0)
        exp_h = 1.0 / (1.0 + 10 ** (-(rh + home_adv - ra) / 400.0))
        rows.append({"game_id": r.game_id, "elo_home": rh, "elo_away": ra, "elo_diff": rh - ra, "elo_prob_home": exp_h})
        if pd.notna(r.home_win):
            y = float(r.home_win)
            delta = k * (y - exp_h)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
    return pd.DataFrame(rows)


def fit_prob(train_x, train_y, test_x):
    m = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(max_iter=2500, C=0.3)),
    ])
    m.fit(train_x, train_y)
    return m.predict_proba(test_x)[:, 1]


def main():
    t = pd.read_csv(OUT / "pitcher_modeling_table.csv", low_memory=False)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t[t["date"].notna() & t["home_win"].notna()].copy()
    t["home_win"] = t["home_win"].astype(int)
    base_features = [c for c in t.columns if c.startswith("diff_sp_") or c.startswith("diff_team_")]

    # Coarse grid first. Refinement is only justified around an out-of-sample winner.
    grid_rows = []
    best = None
    for k, home_adv, carryover in itertools.product((12, 20, 28, 36), (20, 35, 50), (0.60, 0.75, 0.90)):
        elo = build_elo(t[["game_id","date","home_team","away_team","home_win"]], k, home_adv, carryover)
        z = t.merge(elo, on="game_id", how="left")
        years = z["date"].dt.year
        losses=[]; aucs=[]; accs=[]
        for year in sorted(years.unique()):
            if year < 2022: continue
            train = years < year; test = years == year
            if train.sum() < 1000 or test.sum() < 100: continue
            features = base_features + ["elo_diff"]
            p = fit_prob(z.loc[train, features], z.loc[train,"home_win"], z.loc[test, features])
            y = z.loc[test,"home_win"]
            losses.append(log_loss(y,p,labels=[0,1])); aucs.append(roc_auc_score(y,p)); accs.append(accuracy_score(y,p>=.5))
        if not losses: continue
        row={"k":k,"home_adv":home_adv,"carryover":carryover,"mean_log_loss":np.mean(losses),"mean_auc":np.mean(aucs),"mean_accuracy":np.mean(accs),"seasons":len(losses)}
        grid_rows.append(row)
        if best is None or row["mean_log_loss"] < best["mean_log_loss"]: best=row

    grid = pd.DataFrame(grid_rows).sort_values("mean_log_loss")
    grid.to_csv(OUT / "elo_tuning_grid.csv", index=False)
    if best is None: raise SystemExit("No Elo tuning results")

    elo = build_elo(t[["game_id","date","home_team","away_team","home_win"]], best["k"], best["home_adv"], best["carryover"])
    z = t.merge(elo, on="game_id", how="left")
    z["season"] = z["date"].dt.year
    ch = pd.to_numeric(z["close_home_odds"], errors="coerce")
    ca = pd.to_numeric(z["close_away_odds"], errors="coerce")
    valid_market = ch.abs().between(100,5000) & ca.abs().between(100,5000)
    ph = implied(ch); pa = implied(ca)
    z["market_prob"] = ph/(ph+pa)
    z["market_logit"] = np.log(z["market_prob"].clip(.01,.99)/(1-z["market_prob"].clip(.01,.99)))

    details=[]
    for year in sorted(z["season"].dropna().astype(int).unique()):
        if year < 2022: continue
        train = z["season"] < year; test = z["season"] == year
        if train.sum()<1000 or test.sum()<100: continue
        ytr=z.loc[train,"home_win"]; y=z.loc[test,"home_win"]
        for name,features in {
            "baseball_base": base_features,
            "baseball_plus_elo": base_features+["elo_diff"],
            "elo_only":["elo_diff"],
        }.items():
            p=fit_prob(z.loc[train,features],ytr,z.loc[test,features])
            details.append({"season":year,"model":name,"games":int(test.sum()),"log_loss":log_loss(y,p,labels=[0,1]),"auc":roc_auc_score(y,p),"accuracy":accuracy_score(y,p>=.5)})

        mt = test & valid_market
        mtrain = train & valid_market
        if mtrain.sum()>=1000 and mt.sum()>=100:
            ym=z.loc[mt,"home_win"]
            details.append({"season":year,"model":"market_only","games":int(mt.sum()),"log_loss":log_loss(ym,z.loc[mt,"market_prob"],labels=[0,1]),"auc":roc_auc_score(ym,z.loc[mt,"market_prob"]),"accuracy":accuracy_score(ym,z.loc[mt,"market_prob"]>=.5)})
            p=fit_prob(z.loc[mtrain,["market_logit","elo_diff"]],z.loc[mtrain,"home_win"],z.loc[mt,["market_logit","elo_diff"]])
            details.append({"season":year,"model":"market_plus_elo","games":int(mt.sum()),"log_loss":log_loss(ym,p,labels=[0,1]),"auc":roc_auc_score(ym,p),"accuracy":accuracy_score(ym,p>=.5)})

    d=pd.DataFrame(details)
    d.to_csv(OUT / "elo_best_comparison.csv",index=False)
    summary=d.groupby("model",as_index=False).agg(seasons=("season","count"),mean_log_loss=("log_loss","mean"),mean_auc=("auc","mean"),mean_accuracy=("accuracy","mean")).sort_values("mean_log_loss")
    summary.to_csv(OUT / "elo_best_summary.csv",index=False)
    pd.DataFrame([best]).to_csv(OUT / "elo_best_params.csv",index=False)
    print("BEST ELO PARAMETERS")
    print(pd.DataFrame([best]).to_string(index=False))
    print("\nELO COMPARISON")
    print(summary.round(6).to_string(index=False))

if __name__ == "__main__":
    main()
