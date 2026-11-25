"""Eval harness: "how do we know it's right?" (Section 10: golden set + automated rubrics).

Five independent checks, all against hand-made labels in evals/ rather than the
pipeline's own logic:

  1. Format contract   -- columns, order and types match sample_output_format_v2.csv
  2. Sample rows       -- the three graded sample rows reproduce exactly
  3. Golden set        -- 21 hand-labelled route-weeks incl. trap cases
  4. Citation audit    -- every "justified" row re-checked against hand-labelled
                          note windows / scope / cost impact (the "fabricated" test)
  5. Impact classifier -- rule (and LLM, when it ran) vs hand labels on all 10 notes
  6. Unseen-note probes -- 12 new notes (new date phrasings, negations, decreases,
                          an injected instruction) the rules were not written for
  7. Label-free checks  -- metamorphic tests that need no hand labels at all: change
                          the input in a way whose correct effect is known by logic,
                          and check the output changes exactly that way
"""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .notes import build_note, load_notes

EVALS_DIR = config.ROOT / "evals"


@dataclass
class Check:
    name: str
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, ok: bool, failure: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(failure)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def evaluate(output_dir: Path = config.OUTPUT_DIR) -> list[Check]:
    output = pd.read_csv(output_dir / "flagged_routes.csv", dtype=str).fillna("")
    summary = json.loads((output_dir / "run_summary.json").read_text())
    note_labels = pd.read_csv(EVALS_DIR / "note_labels.csv", dtype=str).fillna("")
    return [
        check_format(output),
        check_sample_rows(output),
        check_golden(output, pd.read_csv(EVALS_DIR / "golden_labels.csv", dtype=str).fillna("")),
        check_citations(output, note_labels),
        check_impact_classifier(note_labels, summary.get("impact_check_audit", {})),
        check_probe_notes(pd.read_csv(EVALS_DIR / "probe_notes.csv", dtype=str)),
        check_metamorphic(),
    ]


def read_sample() -> pd.DataFrame:
    """The provided sample has an unquoted comma inside one `reason` value (row 3:
    "(N006, 2025-09-22)"), so a strict CSV parser rejects it. `reason` is the last
    column, so any overflow fields belong to it."""
    lines = config.SAMPLE_OUTPUT_CSV.read_text().strip().splitlines()
    header = next(csv.reader([lines[0]]))
    rows = []
    for line in lines[1:]:
        fields = next(csv.reader([line]))
        rows.append(fields[: len(header) - 1] + [",".join(fields[len(header) - 1:])])
    return pd.DataFrame(rows, columns=header).fillna("")


def check_format(output: pd.DataFrame) -> Check:
    check = Check("Format contract")
    sample = read_sample()
    check.record(list(output.columns) == list(sample.columns),
                 f"columns {list(output.columns)} != {list(sample.columns)}")
    for row in output.itertuples():
        problems = []
        try:
            date.fromisoformat(row.week_of)
            float(row.cost_per_tonne_km)
        except ValueError as exc:
            problems.append(str(exc))
        if row.flagged not in {"Yes", "No (justified)", "No"}:
            problems.append(f"flagged={row.flagged!r}")
        if (row.flagged == "No (justified)") != bool(row.matched_note_id):
            problems.append("matched_note_id must be set exactly when flagged == No (justified)")
        if row.matched_note_id and row.matched_note_id not in row.reason:
            problems.append("reason must name the matched note")
        check.record(not problems, f"{row.route} {row.week_of}: {'; '.join(problems)}")
    return check


def check_sample_rows(output: pd.DataFrame) -> Check:
    check = Check("Sample rows reproduce exactly")
    sample = read_sample()
    graded = ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    for want in sample.itertuples():
        got = output[(output["route"] == want.route) & (output["week_of"] == want.week_of)]
        if got.empty:
            check.record(False, f"{want.route} {want.week_of}: missing from output")
            continue
        diffs = [f"{c}: {got.iloc[0][c]!r} != {getattr(want, c)!r}" for c in graded if got.iloc[0][c] != getattr(want, c)]
        check.record(not diffs, f"{want.route} {want.week_of}: {'; '.join(diffs)}")
    return check


def check_golden(output: pd.DataFrame, golden: pd.DataFrame) -> Check:
    check = Check("Golden set (incl. traps)")
    for case in golden.itertuples():
        got = output[(output["route"] == case.route) & (output["week_of"] == case.week_of)]
        actual = got.iloc[0]["flagged"] if len(got) else "not_flagged"
        actual = "not_flagged" if actual == "No" else actual
        actual_note = got.iloc[0]["matched_note_id"] if len(got) else ""
        ok = actual == case.expected and actual_note == case.expected_note_id
        check.record(ok, f"[{case.category}] {case.route} {case.week_of}: expected {case.expected} "
                         f"{case.expected_note_id or '-'}, got {actual} {actual_note or '-'} ({case.rationale})")
    return check


def check_citations(output: pd.DataFrame, labels: pd.DataFrame) -> Check:
    """The brief's spot-check: a justified row must cite a note that fits route, window and impact."""
    check = Check("Citation audit of justified rows")
    by_id = {r.note_id: r for r in labels.itertuples()}
    for row in output[output["flagged"] == "No (justified)"].itertuples():
        label = by_id.get(row.matched_note_id)
        if label is None:
            check.record(False, f"{row.route} {row.week_of}: cites unknown note {row.matched_note_id}")
            continue
        week_start = date.fromisoformat(row.week_of)
        week_end = week_start + timedelta(days=6)
        start = date.fromisoformat(label.window_start)
        end = date.fromisoformat(label.window_end) if label.window_end else date.max
        problems = []
        if label.scope not in (row.route, "All Routes"):
            problems.append(f"note scope is {label.scope}")
        if not (start <= week_end and week_start <= end):
            problems.append(f"note window {label.window_start}..{label.window_end or 'open'} misses the week")
        if label.raises_cost != "yes":
            problems.append("note does not describe a cost increase")
        if label.scope == "All Routes" and _pct(row.vs_similar_routes) >= config.PEER_THRESHOLD_PCT:
            problems.append("all-routes note used to explain a gap vs peer routes")
        if label.stated_pct_max and _pct(row.vs_own_history) > float(label.stated_pct_max) + config.MAGNITUDE_TOLERANCE_PCT:
            problems.append("rise is far larger than the note's stated impact")
        check.record(not problems, f"{row.route} {row.week_of} -> {row.matched_note_id}: {'; '.join(problems)}")
    return check


def check_impact_classifier(labels: pd.DataFrame, audit: dict) -> Check:
    check = Check("Cost-impact classifier vs hand labels")
    notes = {n.note_id: n for n in load_notes(config.NOTES_CSV)}
    scored_llm = 0
    for label in labels.itertuples():
        expected = label.raises_cost == "yes"
        rule = notes[label.note_id].raises_cost
        check.record(rule == expected, f"{label.note_id}: rule says {rule}, label says {expected}")
        llm = audit.get(label.note_id, {}).get("llm")
        if isinstance(llm, bool):
            check.record(llm == expected, f"{label.note_id}: LLM says {llm}, label says {expected}")
            scored_llm += 1
    check.notes.append(f"LLM judgements scored: {scored_llm} of {len(audit)} audited notes"
                       + ("" if scored_llm else " (LLM did not run or was unavailable; rule only)"))
    return check


def check_probe_notes(probes: pd.DataFrame) -> Check:
    check = Check("Unseen-note probes (rule only)")
    for p in probes.itertuples():
        note = build_note(p.note_id, p.date, p.applies_to, p.note)
        got = (note.raises_cost, note.window_start.isoformat(), note.window_end.isoformat())
        want = (p.expected_raises_cost == "yes", p.expected_window_start, p.expected_window_end)
        check.record(got == want, f"{p.note_id} ({p.probe}): got {got}, expected {want}")
    return check


def check_metamorphic() -> Check:
    """Label-free: each transformation has a logically known effect on the output."""
    from .pipeline import run

    check = Check("Label-free metamorphic checks")
    shipments = pd.read_csv(config.SHIPMENTS_CSV)
    notes = pd.read_csv(config.NOTES_CSV, dtype=str)
    graded = [c for c in config.OUTPUT_COLUMNS if c != "reason"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run_with(ships: pd.DataFrame = shipments, note_rows: pd.DataFrame = notes) -> pd.DataFrame:
            ships.to_csv(tmp / "s.csv", index=False)
            note_rows.to_csv(tmp / "n.csv", index=False)
            return run(output_dir=tmp, use_dense=False, write=False,
                       shipments_path=tmp / "s.csv", notes_path=tmp / "n.csv").output

        base = run_with()
        flags = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes"]
        justified = base[base.flagged == "No (justified)"]

        out = run_with(note_rows=notes.iloc[::-1])
        check.record(out[graded].equals(base[graded]), "reordering the notes changed the output")

        out = run_with(note_rows=notes.iloc[0:0])
        check.record(out[flags].equals(base[flags]) and (out.flagged == "Yes").all(),
                     "with no notes, flags must be unchanged and every row unexplained")

        negated = notes.assign(note=notes.note + " Freight costs were not affected.")
        out = run_with(note_rows=negated)
        check.record((out.flagged == "Yes").all(), "notes that say costs were not affected still justified a row")

        for note_id in sorted(justified.matched_note_id.unique()):
            out = run_with(note_rows=notes[notes.note_id != note_id])
            hit = base.matched_note_id == note_id
            ok = (out[flags].equals(base[flags]) and (out[hit].flagged == "Yes").all()
                  and out[~hit][graded].equals(base[~hit][graded]))
            check.record(ok, f"removing {note_id} must unexplain exactly the rows it justified")

            moved = notes.copy()
            moved.loc[moved.note_id == note_id, "applies_to"] = "Nowhere-Else"
            out = run_with(note_rows=moved)
            check.record((out[hit].flagged == "Yes").all(), f"{note_id} still justified rows after moving it to another route")

        scaled = shipments.assign(freight_cost_inr=shipments.freight_cost_inr * 3)
        out = run_with(ships=scaled)
        same_pct = out[["route", "week_of", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]]
        check.record(same_pct.equals(base[same_pct.columns]), "tripling every cost changed the % comparisons or verdicts")

        doubled = pd.concat([shipments, shipments.assign(shipment_id=shipments.shipment_id + "-dup")])
        out = run_with(ships=doubled)
        check.record(out[graded].equals(base[graded]), "duplicating every shipment changed the output")
    return check


def render(checks: list[Check]) -> str:
    lines = ["# Evaluation report", "", "| Check | Passed | Failed |", "|---|---|---|"]
    lines += [f"| {c.name} | {c.passed} | {c.failed} |" for c in checks]
    lines += ["", *[f"- {c.name}: {n}" for c in checks for n in c.notes]]
    for c in checks:
        if c.failures:
            lines += ["", f"## Failures: {c.name}", *[f"- {f}" for f in c.failures]]
    verdict = "ALL CHECKS PASSED" if all(c.ok for c in checks) else "SOME CHECKS FAILED"
    return "\n".join(lines + ["", f"**{verdict}**", ""])


def _pct(text: str) -> float:
    try:
        return float(text.split("%")[0])
    except ValueError:
        return float("nan")
