"""Deterministic verdicts: may this note justify this flagged week?

"The LLM decides WHAT, code decides WHETHER" (Section 9). A note justifies a
flag only if every hard check passes:

  ROUTE      note applies to this route (or All Routes)
  WINDOW     note's effective window overlaps this Monday-Sunday week
  IMPACT     note describes a cost increase (rule AND, when enabled, the LLM agree)
  MARKET     an All-Routes note cannot explain a gap vs *peer* routes, because the
             peers were hit by the same change
  MAGNITUDE  a note that states its size ("5-7%") cannot explain a much larger rise

No note passes -> unexplained, matched_note_id blank. Never a guess.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import MAGNITUDE_TOLERANCE_PCT
from .flagging import TRIGGER_PEER
from .llm import LLMClient
from .llm_tasks import llm_impact_check
from .notes import Note
from .retrieval import Hit, NoteIndex, query_for

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlagRow:
    route: str
    route_type: str
    week_of: date
    cost_per_tonne_km: float
    vs_own_pct: float
    vs_peer_pct: float
    triggers: tuple[str, ...]

    @property
    def week_end(self) -> date:
        return self.week_of + timedelta(days=6)


@dataclass(frozen=True)
class Rejection:
    code: str
    message: str


@dataclass
class Assessment:
    hit: Hit
    rejections: list[Rejection]

    @property
    def passes(self) -> bool:
        return not self.rejections


@dataclass
class Verdict:
    row: FlagRow
    matched: Note | None
    assessments: list[Assessment] = field(default_factory=list)

    @property
    def justified(self) -> bool:
        return self.matched is not None

    @property
    def closest(self) -> Assessment | None:
        """Near-miss to cite for an unexplained row: fewest failed checks, then retrieval rank."""
        if not self.assessments:
            return None
        return min(enumerate(self.assessments), key=lambda pair: (len(pair[1].rejections), pair[0]))[1]


class ImpactResolver:
    """Per-note cost-impact decision, computed once per note and memoised.

    Rule-only when offline. With an LLM: raises_cost = rule AND llm, so the LLM
    can only make a verdict more conservative. If the LLM is unavailable or its
    output fails validation, we fall back to the rule and record that fact.
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client
        self._decisions: dict[str, tuple[bool, str]] = {}
        self.audit: dict[str, dict] = {}
        self._lock = threading.Lock()

    def prefetch(self, notes: list[Note], workers: int = 4) -> None:
        todo = sorted({n.note_id: n for n in notes}.values(), key=lambda n: n.note_id)
        with ThreadPoolExecutor(max_workers=workers) as pool:  # independent calls -> parallel
            list(pool.map(self.raises_cost, todo))

    def raises_cost(self, note: Note) -> tuple[bool, str]:
        with self._lock:
            if note.note_id in self._decisions:
                return self._decisions[note.note_id]
        entry = {"rule": note.raises_cost, "rule_evidence": note.impact_evidence, "llm": None, "llm_evidence": None}
        decision, evidence = note.raises_cost, note.impact_evidence
        if self.client is not None:
            judgement = llm_impact_check(self.client, note)
            if judgement is None:
                entry["llm"] = "unavailable -> rule used"
            else:
                entry["llm"], entry["llm_evidence"] = judgement.raises_cost, judgement.evidence
                if not judgement.raises_cost:
                    decision, evidence = False, judgement.evidence
            entry["agree"] = entry["llm"] == note.raises_cost if isinstance(entry["llm"], bool) else None
        entry["final"] = decision
        with self._lock:
            self._decisions[note.note_id] = (decision, evidence)
            self.audit[note.note_id] = entry
        return decision, evidence


def assess(note: Note, row: FlagRow, resolver: ImpactResolver) -> list[Rejection]:
    out: list[Rejection] = []
    if not note.applies_to_route(row.route):
        out.append(Rejection("ROUTE", f"applies to {note.applies_to}, not {row.route}"))
    if not note.overlaps(row.week_of, row.week_end):
        out.append(Rejection(
            "WINDOW",
            f"covers {note.window_start:%Y-%m-%d} to {note.window_end:%Y-%m-%d}, "
            f"outside this week ({row.week_of:%Y-%m-%d} to {row.week_end:%Y-%m-%d})",
        ))
    raises, evidence = resolver.raises_cost(note)
    if not raises:
        out.append(Rejection("IMPACT", f'does not describe a cost increase ("{evidence}")'))
    if note.is_all_routes and TRIGGER_PEER in row.triggers:
        out.append(Rejection(
            "MARKET",
            f"applies to all routes, so it cannot explain why this route is "
            f"{row.vs_peer_pct:+.1f}% above similar routes",
        ))
    if note.stated_pct_range and row.vs_own_pct > note.stated_pct_range[1] + MAGNITUDE_TOLERANCE_PCT:
        lo, hi = note.stated_pct_range
        out.append(Rejection(
            "MAGNITUDE",
            f"accounts for roughly {lo:g}-{hi:g}%, but cost rose {row.vs_own_pct:+.1f}% vs history",
        ))
    return out


def decide(row: FlagRow, index: NoteIndex, resolver: ImpactResolver) -> Verdict:
    candidates = index.candidates_for(row.route, row.week_of)
    hits = index.search(query_for(row.route, row.week_of), candidates=candidates)
    assessments = [Assessment(h, assess(h.note, row, resolver)) for h in hits]
    matched = next((a.hit.note for a in assessments if a.passes), None)
    return Verdict(row=row, matched=matched, assessments=assessments)


def unexplained_reason(verdict: Verdict) -> str:
    closest = verdict.closest
    if closest is None:
        return ("No matching note found for this route or date range. "
                "Cost rise looks unexplained and worth a human review.")
    note = closest.hit.note
    why = "; ".join(r.message for r in closest.rejections)
    return (f"The closest note ({note.note_id}, {note.date.isoformat()}) {why} -- it does not justify "
            "this rise. No genuine justification found; flagged for review.")
