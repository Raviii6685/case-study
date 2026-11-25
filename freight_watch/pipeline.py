"""End-to-end batch run: shipments -> weekly metrics -> flags -> retrieval -> verdicts -> CSV.

Shape (Section 3): a deterministic prompt chain, not an autonomous agent. The
steps never change with the input, so a fixed pipeline is the most reliable and
testable choice; the LLM is used only at two narrow, validated points.
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .data import load_shipments
from .flagging import anomalous_weeks
from .llm import LLMClient, UsageLedger
from .llm_tasks import llm_explain, template_explanation
from .metrics import weekly_costs
from .notes import load_notes
from .retrieval import NoteIndex
from .verdict import FlagRow, ImpactResolver, Verdict, decide, unexplained_reason

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    output: pd.DataFrame
    weekly: pd.DataFrame
    verdicts: list[Verdict]
    ledger: UsageLedger
    summary: dict


def run(
    llm: config.LLMSettings | None = None,
    output_dir: Path = config.OUTPUT_DIR,
    use_dense: bool = True,
    write: bool = True,
    shipments_path: Path = config.SHIPMENTS_CSV,
    notes_path: Path = config.NOTES_CSV,
    own_threshold: float = config.OWN_HISTORY_THRESHOLD_PCT,
    peer_threshold: float = config.PEER_THRESHOLD_PCT,
    all_weeks: bool = False,
) -> RunResult:
    started = time.monotonic()
    llm = llm or config.LLMSettings()
    ledger = UsageLedger(llm.input_per_m, llm.output_per_m)
    client = LLMClient(llm, ledger) if llm.enabled else None

    shipments = load_shipments(shipments_path)
    weekly = weekly_costs(shipments)
    flagged = anomalous_weeks(weekly, own_threshold, peer_threshold)
    notes = load_notes(notes_path)
    index = NoteIndex(notes, use_dense=use_dense)
    resolver = ImpactResolver(client)

    rows = [_flag_row(r) for r in flagged.itertuples()]
    candidate_notes = [n for row in rows for n in index.candidates_for(row.route, row.week_of)]
    resolver.prefetch(candidate_notes)  # one impact call per distinct note, in parallel
    verdicts = [decide(row, index, resolver) for row in rows]
    reasons = _reasons(verdicts, client)

    rows_out = [_output_row(v, reasons[i]) for i, v in enumerate(verdicts)]
    if all_weeks:
        rows_out += _normal_week_rows(weekly, flagged)
    output = pd.DataFrame(rows_out, columns=config.OUTPUT_COLUMNS)
    if all_weeks:
        output = output.sort_values(["week_of", "route"], kind="stable").reset_index(drop=True)
    summary = {
        "llm_provider": llm.provider,
        "llm_model": llm.model or None,
        "temperature": llm.temperature if llm.enabled else None,
        "seed": llm.seed if llm.enabled else None,
        "retrieval": index.mode,
        "thresholds": {
            "own_history_pct": own_threshold,
            "peer_pct": peer_threshold,
            "baseline_weeks": config.BASELINE_WEEKS,
        },
        "output_scope": "all route-weeks" if all_weeks else "flagged route-weeks only",
        "shipments": len(shipments),
        "route_weeks": len(weekly),
        "route_weeks_with_fewer_than_8_prior_weeks": int((weekly["prior_weeks_used"] < config.BASELINE_WEEKS).sum()),
        "flagged_rows": int(len(verdicts)),
        "unexplained": int((output["flagged"] == "Yes").sum()),
        "justified": int((output["flagged"] == "No (justified)").sum()),
        "impact_check_audit": resolver.audit,
        "token_usage": ledger.summary(),
        "wall_time_s": round(time.monotonic() - started, 2),
    }
    if write:
        _write_outputs(output_dir, output, weekly, verdicts, ledger, summary)
    return RunResult(output, weekly, verdicts, ledger, summary)


def _reasons(verdicts: list[Verdict], client: LLMClient | None) -> list[str]:
    def reason(v: Verdict) -> str:
        if not v.justified:
            return unexplained_reason(v)  # no LLM call needed: nothing to explain
        fallback = template_explanation(v.matched, v.row.week_of)
        if client is None:
            return fallback
        facts = {
            "route": v.row.route,
            "week_of": v.row.week_of.isoformat(),
            "cost_rise_vs_history": f"{v.row.vs_own_pct:+.1f}%",
            "cost_gap_vs_similar_routes": f"{v.row.vs_peer_pct:+.1f}%",
            "note_id": v.matched.note_id,
            "note_date": v.matched.date.isoformat(),
            "note_applies_to": v.matched.applies_to,
            "note_window": f"{v.matched.window_start} to {v.matched.window_end}",
        }
        return llm_explain(client, v.matched, facts) or fallback

    with ThreadPoolExecutor(max_workers=4) as pool:  # independent rows -> parallel, order preserved
        return list(pool.map(reason, verdicts))


def _normal_week_rows(weekly: pd.DataFrame, flagged: pd.DataFrame) -> list[dict]:
    """--all-weeks: unremarkable weeks, flagged = "No". Not in the sample, so opt-in only."""
    flagged_keys = set(zip(flagged["route"], flagged["week_of"]))
    rows = []
    for w in weekly.itertuples():
        if (w.route, w.week_of) in flagged_keys:
            continue
        rows.append({
            "route": w.route,
            "week_of": w.week_of.date().isoformat(),
            "cost_per_tonne_km": round(float(w.cost_per_tonne_km), 2),
            "vs_own_history": _pct(w.vs_own_pct, "vs this route's past average", "n/a (no prior weeks)"),
            "vs_similar_routes": _pct(w.vs_peer_pct, "vs similar-length routes this week", "n/a (no peer routes this week)"),
            "flagged": "No",
            "matched_note_id": "",
            "reason": "Not flagged: cost is not rising out of the ordinary vs its own history or similar routes.",
        })
    return rows


def _flag_row(r) -> FlagRow:
    return FlagRow(
        route=r.route,
        route_type=r.route_type,
        week_of=r.week_of.date(),
        cost_per_tonne_km=float(r.cost_per_tonne_km),
        vs_own_pct=float(r.vs_own_pct),
        vs_peer_pct=float(r.vs_peer_pct),
        triggers=tuple(r.triggers),
    )


def _pct(value: float, suffix: str, missing: str) -> str:
    return missing if value is None or math.isnan(value) else f"{value:+.1f}% {suffix}"


def _output_row(v: Verdict, reason: str) -> dict:
    row = v.row
    return {
        "route": row.route,
        "week_of": row.week_of.isoformat(),
        "cost_per_tonne_km": round(row.cost_per_tonne_km, 2),
        "vs_own_history": _pct(row.vs_own_pct, "vs this route's past average", "n/a (no prior weeks)"),
        "vs_similar_routes": _pct(row.vs_peer_pct, "vs similar-length routes this week", "n/a (no peer routes this week)"),
        "flagged": "No (justified)" if v.justified else "Yes",
        "matched_note_id": v.matched.note_id if v.justified else "",
        "reason": reason,
    }


def trace_record(v: Verdict) -> dict:
    """Audit trail for one flagged row: what was retrieved and why each note passed or failed."""
    return {
        "route": v.row.route,
        "week_of": v.row.week_of.isoformat(),
        "triggers": list(v.row.triggers),
        "vs_own_pct": round(v.row.vs_own_pct, 2),
        "vs_peer_pct": round(v.row.vs_peer_pct, 2),
        "verdict": "justified" if v.justified else "unexplained",
        "matched_note_id": v.matched.note_id if v.justified else None,
        "candidates": [
            {
                "note_id": a.hit.note.note_id,
                "score": round(a.hit.score, 4),
                "dense": round(a.hit.dense, 4),
                "keyword": round(a.hit.keyword, 4),
                "passes": a.passes,
                "rejections": [{"code": r.code, "message": r.message} for r in a.rejections],
            }
            for a in v.assessments
        ],
    }


def _write_outputs(out: Path, output, weekly, verdicts, ledger, summary) -> None:
    out.mkdir(parents=True, exist_ok=True)
    output.to_csv(out / "flagged_routes.csv", index=False)
    weekly_export = weekly.drop(columns=["freight_cost_inr", "tonne_km"]).copy()
    weekly_export["week_of"] = weekly_export["week_of"].dt.date
    weekly_export.round(4).to_csv(out / "weekly_metrics.csv", index=False)
    with (out / "run_trace.jsonl").open("w") as fh:
        for v in verdicts:
            fh.write(json.dumps(trace_record(v)) + "\n")
    (out / "token_log.json").write_text(json.dumps(ledger.to_json(), indent=2))
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote %d rows to %s", len(output), out / "flagged_routes.csv")
