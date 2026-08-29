"""Price the current FanDuel board using already-built live projections.

The projections must predate the quote collection. This script enforces that
relationship and the frozen paper-eligibility rules before any selection can be
written to the prospective ledger. Every decision cycle also persists an
immutable audit record so NO_PAPER outcomes can be reconstructed later without
relying on short-lived workflow artifacts.
"""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

from run_v22_fanduel_paper import AUDIT_OUT, HISTORY, MKT, freeze, pair_fanduel, select_candidates

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "outputs/fanduel_pitcher_k_live_projections.csv"
CYCLE_ARCHIVE = ROOT / "data/market/free_archive"


def write_cycle_audit(day: str, status: str, **fields) -> Path:
    """Persist one immutable machine-readable record for this decision cycle."""
    now = datetime.now(ZoneInfo("America/New_York"))
    folder = CYCLE_ARCHIVE / day
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H%M%S-%f")
    out = folder / f"cycle-audit-{stamp}.json"
    payload = {
        "schema_version": 1,
        "cycle_recorded_at_et": now.isoformat(),
        "date": day,
        "status": status,
        **fields,
    }
    # Exclusive creation prevents a later cycle from overwriting prior evidence.
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"Persisted FanDuel cycle audit -> {out.relative_to(ROOT)}")
    return out


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
    """Apply the frozen prospective gate and preserve rejection reasons for audit."""
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


def history_rows_for_day(day: str) -> int:
    if not HISTORY.exists() or HISTORY.stat().st_size == 0:
        return 0
    try:
        h = pd.read_csv(HISTORY, low_memory=False)
    except (pd.errors.EmptyDataError, OSError):
        return 0
    if h.empty or "date" not in h.columns:
        return 0
    return int(h["date"].astype(str).eq(day).sum())


def main() -> None:
    day = date.today().isoformat()
    if not PROJ.exists() or PROJ.stat().st_size == 0:
        write_cycle_audit(day, "NO_PAPER", no_paper_reason="NO_PREBUILT_PROJECTIONS")
        print("No prebuilt live projections; nothing to price.")
        return
    try:
        projections = pd.read_csv(PROJ, low_memory=False)
    except pd.errors.EmptyDataError:
        write_cycle_audit(day, "NO_PAPER", no_paper_reason="EMPTY_PROJECTION_FILE")
        print("Live projection file is empty; nothing to price.")
        return
    if projections.empty:
        write_cycle_audit(day, "NO_PAPER", no_paper_reason="NO_LIVE_PROJECTIONS")
        print("No live projections; nothing to price.")
        return

    if not MKT.exists() or MKT.stat().st_size == 0:
        write_cycle_audit(day, "NO_PAPER", no_paper_reason="NO_CURRENT_FANDUEL_MARKET", projection_rows=len(projections))
        print("No current FanDuel market file; nothing to price.")
        return
    try:
        raw = pd.read_csv(MKT, low_memory=False)
    except pd.errors.EmptyDataError:
        write_cycle_audit(day, "NO_PAPER", no_paper_reason="EMPTY_FANDUEL_MARKET_FILE", projection_rows=len(projections))
        print("Current FanDuel market file is empty; nothing to price.")
        return
    market = pair_fanduel(raw, day)
    if market.empty:
        write_cycle_audit(
            day,
            "NO_PAPER",
            no_paper_reason="NO_SAME_DAY_FANDUEL_PITCHER_K_QUOTES",
            projection_rows=len(projections),
            raw_market_rows=len(raw),
            raw_sources=sorted(raw["source"].dropna().astype(str).unique().tolist()) if "source" in raw.columns else [],
        )
        print(f"No FanDuel pitcher-K quotes found for {day}.")
        return

    projections["model_generated_at_et"] = pd.to_datetime(
        projections["model_generated_at_et"], errors="coerce", utc=True
    )
    market["collected_at_utc"] = pd.to_datetime(market["collected_at_utc"], errors="coerce", utc=True)
    try:
        latest_projection, earliest_quote = assert_projection_before_quote(
            projections["model_generated_at_et"], market["collected_at_utc"]
        )
    except ValueError as exc:
        write_cycle_audit(
            day,
            "TIMING_INTEGRITY_FAILURE",
            no_paper_reason="BATCH_MODEL_AFTER_QUOTE_OR_MISSING_TIMESTAMP",
            projection_rows=len(projections),
            paired_market_rows=len(market),
            error=str(exc),
        )
        raise

    projections["model_generated_at_et"] = projections["model_generated_at_et"].dt.tz_convert("America/New_York").astype(str)
    candidates = attach_diagnostics(select_candidates(market, projections), projections)
    try:
        assert_candidate_timing(candidates)
    except ValueError as exc:
        write_cycle_audit(
            day,
            "TIMING_INTEGRITY_FAILURE",
            no_paper_reason="PER_CANDIDATE_MODEL_AFTER_QUOTE_OR_MISSING_TIMESTAMP",
            projection_rows=len(projections),
            paired_market_rows=len(market),
            candidate_rows=len(candidates),
            latest_model_timestamp_utc=latest_projection.isoformat(),
            earliest_quote_timestamp_utc=earliest_quote.isoformat(),
            error=str(exc),
        )
        raise

    audited = mark_paper_eligibility(candidates)
    eligible = audited[audited["paper_eligible"]].copy() if not audited.empty else audited
    chosen = freeze(eligible)

    if not audited.empty:
        AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
        audited.to_csv(AUDIT_OUT, index=False)
        print(
            f"Prospective eligibility: {int(audited.paper_eligible.sum())}/{len(audited)} "
            "line/side candidates passed timing + non-negative edge + positive EV."
        )

    rejection_counts = (
        audited.loc[~audited["paper_eligible"], "paper_rejection_reason"].value_counts().sort_index().to_dict()
        if not audited.empty else {}
    )
    status = "PAPER_SELECTION" if len(chosen) else "NO_PAPER"
    if len(chosen):
        no_paper_reason = None
    elif candidates.empty:
        no_paper_reason = "NO_MODEL_MARKET_MATCHED_CANDIDATES"
    elif eligible.empty:
        no_paper_reason = "NO_ELIGIBLE_CANDIDATES"
    else:
        no_paper_reason = "NO_NEW_FROZEN_SELECTION"

    write_cycle_audit(
        day,
        status,
        no_paper_reason=no_paper_reason,
        projection_rows=len(projections),
        paired_market_rows=len(market),
        candidate_rows=len(candidates),
        eligible_candidate_rows=len(eligible),
        one_per_pitcher_selected_rows=len(chosen),
        frozen_history_rows_for_date=history_rows_for_day(day),
        rejection_reason_counts={str(k): int(v) for k, v in rejection_counts.items()},
        latest_model_timestamp_utc=latest_projection.isoformat(),
        earliest_quote_timestamp_utc=earliest_quote.isoformat(),
        latest_quote_timestamp_utc=market["collected_at_utc"].max().isoformat(),
        timing_integrity="PASS",
        model_versions=sorted(audited["model_version"].dropna().astype(str).unique().tolist()) if "model_version" in audited.columns else [],
    )


if __name__ == "__main__":
    main()
