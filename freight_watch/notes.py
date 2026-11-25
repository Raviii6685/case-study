"""Context notes: parsing, effective date windows, and a rule-based cost-impact check.

Each note becomes structured facts the verdict layer can check deterministically:
  * scope        -- which routes it applies to (from the applies_to column)
  * window       -- the dates it actually describes, read from its text
  * cost impact  -- does it describe a cost *increase*? (rule here; an LLM
                    cross-check lives in llm_tasks.py)
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from .config import OPEN_ENDED_NOTE_WEEKS

ALL_ROUTES = "All Routes"

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_ORD = r"(?:st|nd|rd|th)?"
# A date mention: "Mar 8", "March 8th, 2025", "8 Mar", "8th of March", "2025-03-08".
_DATE = re.compile(
    rf"(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})"
    rf"|(?P<m1>{_MON})\s+(?P<d1>\d{{1,2}}){_ORD}(?:,?\s*(?P<y1>\d{{4}}))?\b"
    rf"|\b(?P<d2>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<m2>{_MON})(?:,?\s*(?P<y2>\d{{4}}))?",
    re.IGNORECASE,
)
_RANGE_JOINER = re.compile(r"^\s*(?:to|and|until|till|through|thru|-|–|—)\s*$", re.IGNORECASE)
_END_CUE = re.compile(r"(?:until|till|through|thru|ending(?:\s+on)?|lasting\s+until)\s*$", re.IGNORECASE)
_START_CUE = re.compile(r"(?:starting|from|beginning|effective|commencing|since)(?:\s+on)?\s*$", re.IGNORECASE)
_DURATION = re.compile(
    r"\bfor\s+(?:about|around|roughly|nearly|approximately|over|almost)?\s*"
    r"(?P<n>a|an|one|two|three|four|five|six|seven|eight|\d+)\s+(?P<unit>day|week)s?\b",
    re.IGNORECASE,
)
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_OPEN_ENDED = re.compile(r"\bstarting\s+this\s+week\b|\bfrom\s+this\s+week\b|\bonwards\b|\bgoing\s+forward\b", re.IGNORECASE)
_PERCENT_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*%")

_COST = r"(?:costs?|rates?|prices?|fares?|tariffs?|tolls?|charges?|fees?)"
# Whole-note vetoes: the note says costs did NOT go up. Any hit -> no cost impact.
NO_IMPACT_PATTERNS = [
    r"not significantly affected",
    rf"{_COST} (?:were|was|are|is|remained|stayed) (?:not|unchanged|flat|stable)",
    r"not part of this dataset",
    r"no (?:major|significant|notable) disruptions?",
    r"remained (?:normal|stable|unchanged)",
    r"without (?:a|any) (?:rate|price|cost) (?:change|increase|rise)",
    r"returned to normal",
    r"improved road conditions",
    r"did not (?:affect|impact|change|increase|raise|push up)",
    rf"no (?:impact|effect|change|increase|rise) (?:on|in|to) (?:freight |transport |trip )?{_COST}",
    r"\b(?:waived|withdrawn|rolled back|revoked|scrapped|cancell?ed|postponed|called off)\b",
    rf"(?:fell|dropped|declined|decreased|eased|reduced|lowered|cut)\s+(?:freight |transport |trip )?{_COST}",
    rf"{_COST} (?:fell|dropped|declined|decreased|eased|were reduced|were lowered|were cut)",
]
# Phrases that describe a cost increase (only counted when not negated in the same sentence).
IMPACT_PATTERNS = [
    rf"higher (?:trip |freight |transport |fuel )?{_COST}",
    r"surcharges?",
    r"pushing up",
    rf"push(?:ed|es|ing)? (?:up (?:\w+ )?{_COST}|(?:\w+ )?{_COST} up)",
    rf"(?:{_COST}|diesel|fuel) (?:rose|increased|went up|climbed|jumped|spiked|surged|were raised|were hiked|soared)",
    rf"(?:{_COST}|fuel|diesel|toll) (?:hike|increase|rise)",
    rf"(?:increase|rise|hike|jump|spike) in (?:freight |transport |toll |fuel |diesel )?{_COST}",
    r"detours?",
    r"more expensive",
    r"(?:new|additional|extra) (?:toll|levy|charge|fee)s?",
]
_NEGATOR = re.compile(r"\b(?:not|no|never|without|none|neither|nor|unchanged|absorbed|exempt(?:ed)?|avoided)\b", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.;!?])\s+")


class Note(BaseModel):
    note_id: str
    date: date
    applies_to: str
    text: str
    window_start: date
    window_end: date
    window_basis: str
    raises_cost: bool
    impact_evidence: str
    stated_pct_range: tuple[float, float] | None = None

    @property
    def is_all_routes(self) -> bool:
        return self.applies_to.strip().lower() == ALL_ROUTES.lower()

    def applies_to_route(self, route: str) -> bool:
        return self.is_all_routes or self.applies_to.strip().lower() == route.strip().lower()

    def overlaps(self, start: date, end: date) -> bool:
        return self.window_start <= end and start <= self.window_end


def load_notes(path: Path) -> list[Note]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return [build_note(r.note_id, r.date, r.applies_to, r.note) for r in frame.itertuples()]


def build_note(note_id: str, note_date: str, applies_to: str, text: str) -> Note:
    issued = date.fromisoformat(note_date.strip())
    start, end, basis = effective_window(issued, text)
    raises, evidence = rule_cost_impact(text)
    pct = _PERCENT_RANGE.search(text)
    return Note(
        note_id=note_id.strip(),
        date=issued,
        applies_to=applies_to.strip(),
        text=text.strip(),
        window_start=start,
        window_end=end,
        window_basis=basis,
        raises_cost=raises,
        impact_evidence=evidence,
        stated_pct_range=(float(pct.group(1)), float(pct.group(2))) if pct else None,
    )


def effective_window(issued: date, text: str) -> tuple[date, date, str]:
    """Which dates does the note describe? First matching rule wins:

    1. explicit range       "from Feb 24 to Mar 8", "between Mar 3 and Mar 9", "Mar 3 - Mar 9"
    2. end date only        "until Mar 15", "through March 20th"  -> issued .. that date
    3. start date only      "starting Apr 7", "effective from 7 April" -> open-ended from that date
    4. stated duration      "for about a week", "for two weeks"  -> issued .. issued + duration
    5. open-ended wording   "starting this week", "onwards"      -> issued .. +8 weeks
    6. otherwise            the Monday-Sunday week the note was issued in

    Open-ended changes last 8 weeks: by then the rolling baseline has absorbed them.
    """
    mentions = [(m, _parse_date(m, issued)) for m in _DATE.finditer(text)]
    mentions = [(m, d) for m, d in mentions if d is not None]

    # if there are mutliple date entries it will only send the frist one but there can be posbbilve chance of the second date as anser ???
    for (m1, d1), (m2, d2) in zip(mentions, mentions[1:]):
        if _RANGE_JOINER.match(text[m1.end():m2.start()]):
            if d2 < d1:  # e.g. "Dec 28 to Jan 4"
                d2 = d2.replace(year=d2.year + 1)
            return d1, d2, f"explicit range in text ({text[m1.start():m2.end()]})"
# why did we do m.start() - 25
    for m, d in mentions:
        before = text[max(0, m.start() - 25):m.start()]
        if _END_CUE.search(before):
            return min(issued, d), max(issued, d), f"end date in text ({before.strip().split()[-1]} {m.group(0)})"
        if _START_CUE.search(before):
            return d, _open_end(d), f"start date in text ({m.group(0)}), open-ended for {OPEN_ENDED_NOTE_WEEKS} weeks"

    duration = _DURATION.search(text)
    if duration:
        n = _NUMBERS.get(duration.group("n").lower()) or int(duration.group("n"))
        days = n * (7 if duration.group("unit").lower() == "week" else 1)
        return issued, issued + timedelta(days=days - 1), f"stated duration ({duration.group(0).strip()})"

    if _OPEN_ENDED.search(text):
        return issued, _open_end(issued), f"open-ended change, valid until absorbed by the {OPEN_ENDED_NOTE_WEEKS}-week baseline"
    monday = issued - timedelta(days=issued.weekday())
    return monday, monday + timedelta(days=6), "the week the note was issued"


def rule_cost_impact(text: str) -> tuple[bool, str]:
    """(raises_cost, evidence). Conservative: a whole-note veto wins, and an increase
    phrase only counts if its own sentence is not negated ("no surcharge was applied")."""
    for pattern in NO_IMPACT_PATTERNS:
        hit = re.search(pattern, text, re.IGNORECASE)
        if hit:
            return False, hit.group(0)
    negated = None
    for sentence in _SENTENCES.split(text):
        for pattern in IMPACT_PATTERNS:
            hit = re.search(pattern, sentence, re.IGNORECASE)
            if not hit:
                continue
            if _NEGATOR.search(sentence):
                negated = negated or sentence.strip()
                continue
            return True, hit.group(0)
    if negated:
        return False, f"increase mentioned but negated: {negated[:80]}"
    return False, "no cost increase described"


def _open_end(start: date) -> date:
    monday = start - timedelta(days=start.weekday())
    return monday + timedelta(weeks=OPEN_ENDED_NOTE_WEEKS) - timedelta(days=1)


def _parse_date(m: re.Match, issued: date) -> date | None:
    try:
        if m.group("iso"):
            return date.fromisoformat(m.group("iso"))
        month, day, year = (m.group("m1"), m.group("d1"), m.group("y1")) if m.group("m1") else (
            m.group("m2"), m.group("d2"), m.group("y2"))
        parsed = date(int(year) if year else issued.year, _MONTHS[month[:3].lower()], int(day))
    except (ValueError, KeyError):
        return None
    if not year and parsed < issued - timedelta(days=180):  # "Jan 5" in a note issued in late December
        parsed = parsed.replace(year=parsed.year + 1)
    return parsed
