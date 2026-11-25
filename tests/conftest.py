from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest

from freight_watch.llm import CallRecord, ChatResult, LLMError, UsageLedger
from freight_watch.pipeline import run

warnings.filterwarnings("ignore")


class FakeLLM:
    """Scripted stand-in for LLMClient: each call pops the next response (or calls a function)."""

    model = "fake-model"

    def __init__(self, script: list[ChatResult | Exception] | Callable[[list[dict], str], ChatResult]):
        self.script = script
        self.calls: list[dict] = []
        self.ledger = UsageLedger()

    def chat(self, messages, *, purpose, tools=None, max_tokens=300, json_mode=False) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], "purpose": purpose, "tools": tools})
        item = self.script(messages, purpose) if callable(self.script) else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.ledger.add(CallRecord(purpose, self.model, item.input_tokens, item.output_tokens, 0, False, 1))
        return item


def tool_call(name: str, arguments: str, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture(scope="session")
def offline_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("output")
    result = run(output_dir=out, use_dense=False)
    return out, result


__all__ = ["FakeLLM", "tool_call", "LLMError"]
