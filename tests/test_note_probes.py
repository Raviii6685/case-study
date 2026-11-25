"""Notes the system was never tuned on: new date phrasings, negations, decreases, injection."""

from datetime import date

import pandas as pd
import pytest

from freight_watch.notes import build_note

PROBES = pd.read_csv("evals/probe_notes.csv", dtype=str)


@pytest.mark.parametrize("probe", list(PROBES.itertuples()), ids=lambda p: f"{p.note_id}-{p.probe}")
def test_probe_note(probe):
    note = build_note(probe.note_id, probe.date, probe.applies_to, probe.note)
    assert note.raises_cost == (probe.expected_raises_cost == "yes"), note.impact_evidence
    assert note.window_start == date.fromisoformat(probe.expected_window_start), note.window_basis
    assert note.window_end == date.fromisoformat(probe.expected_window_end), note.window_basis


def test_iso_and_day_first_dates():
    note = build_note("X", "2025-03-03", "A-B", "Detours from 2025-03-03 to 10th of March raised costs.")
    assert (note.window_start, note.window_end) == (date(2025, 3, 3), date(2025, 3, 10))
