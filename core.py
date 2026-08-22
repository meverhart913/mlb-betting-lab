from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    date: str
    market: str
    outcome: int
    odds: float
    opposite_odds: float | None
    features: tuple[float, ...]


@dataclass(frozen=True)
class MarketResult:
    market: str
    observations: int
    bets: int
    wins: int
    hit_rate: float
    wilson_low: float
    wilson_high: float
    profit: float
    roi: float
    average_ev: float
    max_drawdown: float
    qualified: bool
    reason: str


def load_csv(path: str | Path, feature_names: list[str]) -> list[Observation]:
    rows: list[Observation] = []
    required = {"date", "market", "outcome", "decimal_odds", *feature_names}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            try:
                outcome = int(row["outcome"])
                odds = float(row["decimal_odds"])
                opposite = row.get("opposite_decimal_odds", "").strip()
                features = tuple(float(row[name]) for name in feature_names)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value on line {line}") from exc
            if outcome not in (0, 1) or odds <= 1 or not row["date"] or not row["market"]:
                raise ValueError(f"invalid row on line {line}")
            opposite_odds = float(opposite) if opposite else None
            if opposite_odds is not None and opposite_odds <= 1:
                raise ValueError(f"invalid opposite odds on line {line}")
            if not all(math.isfinite(value) for value in (*features, odds)):
                raise ValueError(f"non-finite value on line {line}")
            rows.append(Observation(row["date"], row["market"], outcome, odds, opposite_odds, features))
    return rows


def fair_implied_probability(odds: float, opposite_odds: float | None) -> float:
    offered = 1 / odds
    if opposite_odds is None:
        return offered
    return offered / (offered + 1 / opposite_odds)


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1 / (1 + math.exp(-value))


def _fit(rows: list[Observation], l2: float = 0.2, steps: int = 350) -> tuple[list[float], list[float], list[float]]:
    width = len(rows[0].features)
    means = [sum(row.features[j] for row in rows) / len(rows) for j in range(width)]
    scales = []
    for j in range(width):
        variance = sum((row.features[j] - means[j]) ** 2 for row in rows) / len(rows)
        scales.append(max(math.sqrt(variance), 1e-9))
    weights = [0.0] * (width + 1)
    rate = 0.15
    for step in range(steps):
        gradient = [0.0] * len(weights)
        for row in rows:
            x = [1.0] + [(value - means[j]) / scales[j] for j, value in enumerate(row.features)]
            error = _sigmoid(sum(w * v for w, v in zip(weights, x))) - row.outcome
            for j, value in enumerate(x):
                gradient[j] += error * value
        for j in range(len(weights)):
            penalty = 0 if j == 0 else l2 * weights[j]
            weights[j] -= rate * (gradient[j] / len(rows) + penalty / len(rows))
        rate = 0.15 / math.sqrt(1 + step / 50)
    return weights, means, scales


def _predict(model: tuple[list[float], list[float], list[float]], features: tuple[float, ...]) -> float:
    weights, means, scales = model
    x = [1.0] + [(value - means[j]) / scales[j] for j, value in enumerate(features)]
    return _sigmoid(sum(w * v for w, v in zip(weights, x)))


def wilson_interval(wins: int, bets: int, z: float = 1.96) -> tuple[float, float]:
    if bets == 0:
        return 0.0, 0.0
    p = wins / bets
    denominator = 1 + z * z / bets
    center = (p + z * z / (2 * bets)) / denominator
    margin = z * math.sqrt(p * (1 - p) / bets + z * z / (4 * bets * bets)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def backtest_market(
    market: str,
    rows: list[Observation],
    *,
    warmup: int = 100,
    refit_every: int = 20,
    min_edge: float = 0.03,
    min_bets: int = 100,
    target_hit_rate: float = 0.70,
    min_wilson_low: float = 0.50,
) -> MarketResult:
    rows = sorted(rows, key=lambda row: row.date)
    if len(rows) <= warmup:
        return MarketResult(market, len(rows), 0, 0, 0, 0, 0, 0, 0, 0, 0, False, f"needs more than {warmup} observations")
    profits: list[float] = []
    evs: list[float] = []
    model = None
    for index in range(warmup, len(rows)):
        if model is None or (index - warmup) % refit_every == 0:
            model = _fit(rows[:index])
        row = rows[index]
        probability = _predict(model, row.features)
        implied = fair_implied_probability(row.odds, row.opposite_odds)
        ev = probability * row.odds - 1
        if probability - implied >= min_edge and ev > 0:
            profits.append(row.odds - 1 if row.outcome else -1.0)
            evs.append(ev)
    bets = len(profits)
    wins = sum(value > 0 for value in profits)
    hit_rate = wins / bets if bets else 0
    low, high = wilson_interval(wins, bets)
    cumulative = peak = drawdown = 0.0
    for profit in profits:
        cumulative += profit
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    total_profit = sum(profits)
    roi = total_profit / bets if bets else 0
    failures = []
    if bets < min_bets: failures.append(f"only {bets}/{min_bets} bets")
    if hit_rate < target_hit_rate: failures.append(f"hit rate {hit_rate:.1%} below {target_hit_rate:.1%}")
    if low < min_wilson_low: failures.append(f"Wilson low {low:.1%} below {min_wilson_low:.1%}")
    if roi <= 0: failures.append("ROI is not positive")
    return MarketResult(market, len(rows), bets, wins, hit_rate, low, high, total_profit, roi,
                        sum(evs) / bets if bets else 0, drawdown, not failures, "; ".join(failures) or "all gates passed")


def analyze(rows: Iterable[Observation], **kwargs: object) -> list[MarketResult]:
    grouped: dict[str, list[Observation]] = {}
    for row in rows:
        grouped.setdefault(row.market, []).append(row)
    results = [backtest_market(name, market_rows, **kwargs) for name, market_rows in grouped.items()]
    return sorted(results, key=lambda result: (result.qualified, result.roi, result.bets), reverse=True)


def markdown_report(results: list[MarketResult]) -> str:
    lines = ["# MLB Betting Lab report", "", "> Historical results are not a guarantee. `QUALIFIED` means configured gates passed, not that future bets will win.", "",
             "| Market | Status | Bets | Hit rate (95% CI) | Profit | ROI | Avg model EV | Max drawdown | Note |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r.market} | {'QUALIFIED' if r.qualified else 'REJECT'} | {r.bets} | {r.hit_rate:.1%} ({r.wilson_low:.1%}–{r.wilson_high:.1%}) | {r.profit:.2f}u | {r.roi:.1%} | {r.average_ev:.1%} | {r.max_drawdown:.2f}u | {r.reason} |")
    return "\n".join(lines) + "\n"
