#!/usr/bin/env python3
"""Audit FanDuel decision-cycle efficiency by nominal daily pull window.

Uses immutable cycle-audit JSON files already committed by the prospective workflow.
No market/API calls are made.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "market" / "free_archive"
OUT = ROOT / "outputs" / "fanduel_schedule_efficiency.csv"
ET = ZoneInfo("America/New_York")
WINDOWS = [(10, 45), (13, 45), (16, 45), (19, 45)]


def parse_dt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return dt.astimezone(ET)
    except ValueError:
        return None


def nominal_window(dt):
    mins = dt.hour * 60 + dt.minute
    h, m = min(WINDOWS, key=lambda x: abs(mins - (x[0] * 60 + x[1])))
    return f"{h:02d}:{m:02d} ET"


def first(obj, *keys, default=0):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def main():
    rows = []
    for path in sorted(ARCHIVE.glob("*/cycle-audit-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ts = parse_dt(first(data, "quote_timestamp", "quote_ts", "cycle_timestamp", "timestamp", default=None))
        if ts is None:
            # Filenames are local HHMMSS in the current archive convention.
            try:
                day = path.parent.name
                hhmmss = path.stem.split("cycle-audit-", 1)[1].split("-", 1)[0]
                ts = datetime.strptime(day + hhmmss, "%Y-%m-%d%H%M%S").replace(tzinfo=ET)
            except Exception:
                continue
        rows.append({
            "date": ts.date().isoformat(),
            "window": nominal_window(ts),
            "actual_time": ts.strftime("%H:%M:%S"),
            "status": first(data, "status", "cycle_status", default="UNKNOWN"),
            "market_rows": int(first(data, "market_rows", "quote_rows", "market_row_count", default=0) or 0),
            "eligible": int(first(data, "eligible_candidates", "eligible_count", "eligible_rows", default=0) or 0),
            "new": int(first(data, "new_selections", "new_selection_count", "new_frozen", default=0) or 0),
            "path": str(path.relative_to(ROOT)),
        })

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["window"]].append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = "window,cycles,days_with_cycle,market_rows,eligible_candidates,new_frozen,zero_new_cycles,new_per_cycle\n"
    lines = [header]
    for h, m in WINDOWS:
        key = f"{h:02d}:{m:02d} ET"
        rs = grouped.get(key, [])
        cycles = len(rs)
        days = len({r["date"] for r in rs})
        market = sum(r["market_rows"] for r in rs)
        eligible = sum(r["eligible"] for r in rs)
        new = sum(r["new"] for r in rs)
        zero = sum(r["new"] == 0 for r in rs)
        rate = new / cycles if cycles else 0.0
        lines.append(f"{key},{cycles},{days},{market},{eligible},{new},{zero},{rate:.3f}\n")
    OUT.write_text("".join(lines), encoding="utf-8")

    print(OUT.relative_to(ROOT))
    print("".join(lines), end="")


if __name__ == "__main__":
    main()
