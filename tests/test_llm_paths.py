"""LLM-dependent paths, exercised with a scripted fake model (no network)."""

from datetime import date

from conftest import FakeLLM, LLMError

from freight_watch.config import NOTES_CSV
from freight_watch.llm import ChatResult
from freight_watch.llm_tasks import llm_explain, llm_impact_check, template_explanation, validate_explanation
from freight_watch.notes import load_notes
from freight_watch.retrieval import NoteIndex
from freight_watch.verdict import FlagRow, ImpactResolver, decide

NOTES = {n.note_id: n for n in load_notes(NOTES_CSV)}


def test_impact_check_retries_after_invalid_json_then_accepts():
    fake = FakeLLM([ChatResult("not json"), ChatResult('{"raises_cost": true, "evidence": "surcharge"}')])
    judgement = llm_impact_check(fake, NOTES["N002"])
    assert judgement.raises_cost is True
    assert len(fake.calls) == 2 and "Invalid" in fake.calls[1]["messages"][-1]["content"]


def test_impact_check_rejects_evidence_that_is_not_in_the_note():
    fake = FakeLLM([ChatResult('{"raises_cost": true, "evidence": "fuel doubled"}')] * 2)
    assert llm_impact_check(fake, NOTES["N002"]) is None


def test_impact_check_returns_none_when_llm_is_down():
    assert llm_impact_check(FakeLLM([LLMError("503")]), NOTES["N002"]) is None


def test_llm_can_only_make_verdicts_more_conservative():
    row = FlagRow("Ahmedabad-Mumbai", "Medium", date(2025, 1, 20), 3.3, 29.5, 22.5, ("own_history", "peer"))
    index = NoteIndex(list(NOTES.values()), use_dense=False)

    says_no = FakeLLM(lambda m, p: ChatResult('{"raises_cost": false, "evidence": "temporary surcharge"}'))
    assert decide(row, index, ImpactResolver(says_no)).matched is None  # LLM veto respected

    # LLM claims the "costs were not significantly affected" note raises cost: rule still blocks it.
    says_yes = FakeLLM(lambda m, p: ChatResult('{"raises_cost": true, "evidence": "Scheduled highway maintenance"}'))
    md_row = FlagRow("Mumbai-Delhi", "Long", date(2024, 7, 29), 3.3, 29.5, 22.5, ("own_history", "peer"))
    resolver = ImpactResolver(says_yes)
    assert decide(md_row, index, resolver).matched is None
    assert resolver.audit["N005"]["agree"] is False


def test_llm_outage_falls_back_to_rule_and_is_recorded():
    resolver = ImpactResolver(FakeLLM(lambda m, p: (_ for _ in ()).throw(LLMError("down"))))
    assert resolver.raises_cost(NOTES["N002"])[0] is True
    assert resolver.audit["N002"]["llm"] == "unavailable -> rule used"


def test_explanation_validator():
    note = NOTES["N002"]
    good = "Matches note N002 dated 2025-01-20: a festival-week surcharge on this corridor explains the rise."
    assert validate_explanation(good, note) == []
    assert any("N002" in p for p in validate_explanation("A festival happened.", note))
    assert any("other notes" in p for p in validate_explanation(good + " See also N003.", note))
    assert validate_explanation(template_explanation(note, date(2025, 1, 20)), note) == []


def test_bad_explanation_is_rewritten_then_replaced_by_template():
    fake = FakeLLM([ChatResult("Costs went up due to fuel."), ChatResult("Still no citation.")])
    assert llm_explain(fake, NOTES["N002"], {"route": "Ahmedabad-Mumbai"}) is None  # caller uses template
    assert len(fake.calls) == 2
