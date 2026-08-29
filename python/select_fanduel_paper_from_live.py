"""Price the current FanDuel board using already-built live projections.

The projections must predate the quote collection. This script enforces that
relationship and the frozen paper-eligibility rules before any selection can be
written to the prospective ledger.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from run_v22_fanduel_paper import AUDIT_OUT, MKT, freeze, pair_fanduel, select_candidates

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"


def assert_projection_before_quote(projection_times, quote_times) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Batch-level sanity check: every projection must predate every quote in the cycle."""
    p = pd.to_datetime(projection_times, errors="coerce", utc=True)
    q = pd.to_datetime(quote_times, errors="coerce", utc=True)
    latest_projection = p.max()
    earliest_quote = q.min()
    if pd.isna(latest_projection) or pd.isna(earliest_quote):
        raise ValueError("Missing model or quote timestamps; cannot guarantee timing integrity.")
    if latest_projection > earliest_quote:
        raise ValueError(
            f"Timing integrity failure: latest model timestamp {latest_projection} is after "
            f"earliest decision quote {earliest_quote}. Rebuild projections before recollecting market."
        )
    return latest_projection, earliest_quote


def assert_candidate_timing(candidates: pd.DataFrame) -> None:
    """Enforce timing lineage on every priced candidate before it can be frozen."""
    if candidates.empty:
        return
    required = {"model_generated_at_et", "collected_at_utc"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate timing integrity failure: missing columns {missing}.")

    p = pd.to_datetime(candidates["model_generated_at_et"], errors="coerce", utc=True)
    q = pd.to_datetime(candidates["collected_at_utc"], errors="coerce", utc=True)
    invalid = p.isna() | q.isna() | p.gt(q)
    if invalid.any():
        sample_cols = [c for c in ["date", "game_id", "pitcher_id", "pitcher_name", "line", "side"] if c in candidates.columns]
        sample = candidates.loc[invalid, sample_cols].head(3).to_dict("records")
        raise ValueError(
            f"Candidate timing integrity failure: {int(invalid.sum())} candidate row(s) have "
            f"missing timestamps or model time after quote time. Sample={sample}"
        )


def mark_paper_eligibility(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen prospective gate and preserve rejection reasons for audit.

    A line/side can enter the prospective ledger only when it is in the decision
    window, has non-negative model-vs-no-vig edge, and has strictly positive EV at
    the actual FanDuel price. This prevents a negative-value option from becoming
    the one-per-pitcher selection merely because it is the best bad option.
    """
    if candidates.empty:
        return candidates
    x = candidates.copy()
    timing = x["timing_eligible"].fillna(False).astype(bool)
    edge = pd.to_numeric(x["model_market_edge"], errors="coerce")
    ev = pd.to_numeric(x["expected_profit_per_unit"], errors="coerce")
    x["paper_eligible"] = timing & edge.ge(0.0) & ev.gt(0.0)

    reasons = []
    for ok_timing, e, value in zip(timing, edge, ev):
        r = []
        if not ok_timing:
            r.append("OUTSIDE_DECISION_WINDOW")
        if pd.isna(e):
            r.append("MISSING_MARKET_EDGE")
        elif e < 0:
            r.append("NEGATIVE_MARKET_EDGE")
        if pd.isna(value):
            r.append("MISSING_EV")
        elif value <= 0:
            r.append("NONPOSITIVE_EV")
        reasons.append("ELIGIBLE" if not r else ",".join(r))
    x["paper_rejection_reason"] = reasons
    return x


def attach_diagnostics(candidates: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    extras = [c for c in [
        "prior_start_count", "statcast_appearance_count", "days_rest_diagnostic",
        "lineup_status", "failure_regime_flags"
    ] if c in projections.columns]
    if not extras:
        return candidates
    d = projections[["game_id", "pitcher_id", *extras]].drop_duplicates(["game_id", "pitcher_id"])
    return candidates.merge(d, on=["game_id", "pitcher_id"], how="left")


def main() -> None:
    day = date.today().isoformat()
    if not PROJ.exists() or PROJ.stat().st_size == 0:
        print("No prebuilt live projections; nothing to price.")
        return
    try:
        projections = pd.read_csv(PROJ, low_memory=False)
    except pd.errors.EmptyDataError:
        print("Live projection file is empty; nothing to price.")
        return
    if projections.empty:
        print("No live projections; nothing to price.")
        return

    if not MKT.exists() or MKT.stat().st_size == 0:
        print("No current FanDuel market file; nothing to price.")
        return
    try:
        raw = pd.read_csv(MKT, low_memory=False)
    except pd.errors.EmptyDataError:
        print("Current FanDuel market file is empty; nothing to price.")
        return
    market = pair_fanduel(raw, day)
    if market.empty:
        print(f"No FanDuel pitcher-K quotes found for {day}.")
        return

    projections["model_generated_at_et"] = pd.to_datetime(
        projections["model_generated_at_et"], errors="coerce", utc=True
    )
    market["collected_at_utc"] = pd.to_datetime(market["collected_at_utc"], errors="coerce", utc=True)
    assert_projection_before_quote(projections["model_generated_at_et"], market["collected_at_utc"])

    projections["model_generated_at_et"] = projections["model_generated_at_et"].dt.tz_convert("America/New_York").astype(str)
    candidates = attach_diagnostics(select_candidates(market, projections), projections)
    assert_candidate_timing(candidates)
    audited = mark_paper_eligibility(candidates)
    eligible = audited[audited["paper_eligible"]].copy() if not audited.empty else audited
    freeze(eligible)

    # freeze() owns the one-per-pitcher ledger logic and writes an audit of what it
    # receives. Restore the complete decision-cycle audit afterward so rejected
    # lines remain inspectable instead of disappearing from the evidence trail.
    if not audited.empty:
        AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
        audited.to_csv(AUDIT_OUT, index=False)
        print(
            f"Prospective eligibility: {int(audited.paper_eligible.sum())}/{len(audited)} "
            "line/side candidates passed timing + non-negative edge + positive EV."
        )


if __name__ == "__main__":
    main()
