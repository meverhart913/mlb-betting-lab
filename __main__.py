from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from .core import analyze, load_csv, markdown_report


def generate_demo(path: Path, games: int, seed: int) -> None:
    random.seed(seed)
    markets = {"moneyline": (1.91, 1.9), "runline": (2.25, 1.35), "total_over": (1.95, 1.15), "batter_hit": (1.72, 0.9)}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "market", "outcome", "decimal_odds", "opposite_decimal_odds", "starter_edge", "offense_edge", "bullpen_edge"])
        for game in range(games):
            for market, (odds, strength) in markets.items():
                features = [random.gauss(0, 1) for _ in range(3)]
                probability = 1 / (1 + math.exp(-(strength * features[0] + 0.55 * features[1] + 0.25 * features[2])))
                writer.writerow([f"{2020 + game // 180:04d}-{1 + (game // 30) % 6:02d}-{1 + game % 28:02d}", market,
                                 int(random.random() < probability), odds, round(odds, 2), *(round(x, 5) for x in features)])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Leakage-resistant MLB betting market backtester")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("generate-demo", help="create explicitly synthetic validation data")
    demo.add_argument("--output", type=Path, default=Path("demo.csv"))
    demo.add_argument("--games", type=int, default=800)
    demo.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("analyze", help="rank markets from a historical CSV")
    run.add_argument("csv", type=Path)
    run.add_argument("--features", required=True, help="comma-separated pregame numeric columns")
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--refit-every", type=int, default=20)
    run.add_argument("--min-edge", type=float, default=0.03)
    run.add_argument("--min-bets", type=int, default=100)
    run.add_argument("--target-hit-rate", type=float, default=0.70)
    run.add_argument("--min-wilson-low", type=float, default=0.50)
    run.add_argument("--report", type=Path)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "generate-demo":
        if args.games < 1: raise SystemExit("--games must be positive")
        generate_demo(args.output, args.games, args.seed)
        print(f"Wrote synthetic demo data to {args.output}")
        return 0
    features = [name.strip() for name in args.features.split(",") if name.strip()]
    if not features: raise SystemExit("at least one feature is required")
    results = analyze(load_csv(args.csv, features), warmup=args.warmup, refit_every=args.refit_every,
                      min_edge=args.min_edge, min_bets=args.min_bets, target_hit_rate=args.target_hit_rate,
                      min_wilson_low=args.min_wilson_low)
    report = markdown_report(results)
    print(report, end="")
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
