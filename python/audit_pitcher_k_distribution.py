"""Audit whether Poisson is appropriate for V2.2 strikeout count probabilities.

Dispersion is estimated only from pre-2026 out-of-sample V2.2 predictions and the
Poisson vs negative-binomial comparison is evaluated on 2026 holdout starts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs/pitcher_k_batter_hand_predictions.csv"
OUT = ROOT / "outputs"


def alpha_mom(y, mu):
    mu = np.clip(np.asarray(mu, float), 0.05, None)
    y = np.asarray(y, float)
    vals = ((y - mu) ** 2 - mu) / (mu ** 2)
    vals = vals[np.isfinite(vals)]
    # Robust moment estimate: trim extreme 1% tails before averaging.
    if len(vals) > 200:
        lo, hi = np.quantile(vals, [0.01, 0.99])
        vals = vals[(vals >= lo) & (vals <= hi)]
    return float(max(np.mean(vals), 0.0)) if len(vals) else 0.0


def nb_logpmf(y, mu, alpha):
    if alpha <= 1e-8:
        return poisson.logpmf(y, mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return nbinom.logpmf(y, n, p)


def over_prob_nb(line, mu, alpha):
    cut = int(np.floor(line))
    if alpha <= 1e-8:
        return 1 - poisson.cdf(cut, mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return 1 - nbinom.cdf(cut, n, p)


def main():
    if not PRED.exists():
        raise SystemExit(f"Missing {PRED}")
    z = pd.read_csv(PRED, low_memory=False)
    z = z[z.model.eq("v22_lineup_all")].copy()
    z["season"] = pd.to_numeric(z.season, errors="coerce")
    z["projected_k"] = pd.to_numeric(z.projected_k, errors="coerce")
    z["strikeouts"] = pd.to_numeric(z.strikeouts, errors="coerce")
    z = z.dropna(subset=["season", "projected_k", "strikeouts"])

    train = z[z.season.lt(2026)]
    test = z[z.season.eq(2026)]
    if len(train) < 1000 or len(test) < 300:
        raise SystemExit("Insufficient pre-2026/2026 V2.2 predictions for distribution audit")

    alpha = alpha_mom(train.strikeouts, train.projected_k)
    y = test.strikeouts.to_numpy(float)
    mu = np.clip(test.projected_k.to_numpy(float), 0.05, None)
    p_ll = poisson.logpmf(y, mu)
    nb_ll = nb_logpmf(y, mu, alpha)

    # Probability calibration test at a synthetic half-strikeout threshold near
    # each model mean. This compares distribution shape without using sportsbook
    # outcomes to choose the winner.
    lines = np.floor(mu) + 0.5
    actual_over = (y > lines).astype(float)
    pois_over = 1 - poisson.cdf(np.floor(lines).astype(int), mu)
    nb_over = np.array([over_prob_nb(l, m, alpha) for l, m in zip(lines, mu)])
    pois_brier = float(np.mean((actual_over - pois_over) ** 2))
    nb_brier = float(np.mean((actual_over - nb_over) ** 2))

    summary = pd.DataFrame([{
        "train_starts_pre2026": len(train),
        "holdout_starts_2026": len(test),
        "nb_alpha_pre2026": alpha,
        "poisson_mean_nll_2026": float(-np.mean(p_ll)),
        "nb_mean_nll_2026": float(-np.mean(nb_ll)),
        "poisson_brier_halfline_2026": pois_brier,
        "nb_brier_halfline_2026": nb_brier,
        "recommended_distribution": "negative_binomial" if (-np.mean(nb_ll) < -np.mean(p_ll) and nb_brier <= pois_brier) else "poisson",
    }])
    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "pitcher_k_distribution_audit.csv", index=False)

    bins = pd.cut(test.projected_k, bins=[0,3,4,5,6,7,8,10,20], include_lowest=True)
    disp = test.assign(mu=test.projected_k, bin=bins).groupby("bin", observed=True).agg(
        starts=("strikeouts", "size"),
        mean_actual=("strikeouts", "mean"),
        mean_projected=("mu", "mean"),
        actual_variance=("strikeouts", "var"),
    ).reset_index()
    disp["variance_to_mean_actual"] = disp.actual_variance / disp.mean_actual.replace(0, np.nan)
    disp.to_csv(OUT / "pitcher_k_distribution_by_projection.csv", index=False)
    print(summary.to_string(index=False))
    print("\n2026 dispersion by projected-K bin")
    print(disp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
