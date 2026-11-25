"""Pre-demo health check: `python -m freight_watch doctor [--llm groq]`.

Verifies everything a live walkthrough depends on, and warms the caches so the demo
itself never waits on a download: dependencies, input files, the local embedding
model, the LLM key / model / a one-token round trip, and the committed outputs.
"""

from __future__ import annotations

import importlib
import time

from . import config


def run_checks(provider: str | None, call_llm: bool = True) -> list[tuple[bool, str, str]]:
    results: list[tuple[bool, str, str]] = []

    def record(ok: bool, name: str, detail: str) -> None:
        results.append((ok, name, detail))

    for module, required in [("pandas", True), ("numpy", True), ("pydantic", True), ("fastembed", False), ("mcp", False)]:
        try:
            importlib.import_module(module)
            record(True, f"package {module}", "installed")
        except ImportError:
            record(not required, f"package {module}", "missing" + ("" if required else " (optional)"))

    from .data import load_shipments
    from .notes import load_notes

    for path in (config.SHIPMENTS_CSV, config.NOTES_CSV, config.SAMPLE_OUTPUT_CSV):
        record(path.exists(), f"input {path.name}", "found" if path.exists() else "MISSING")
    try:
        record(True, "shipments load", f"{len(load_shipments(config.SHIPMENTS_CSV))} valid rows")
        record(True, "notes load", f"{len(load_notes(config.NOTES_CSV))} notes")
    except Exception as exc:
        record(False, "inputs load", str(exc))

    try:
        from .retrieval import DenseEmbedder

        started = time.monotonic()
        DenseEmbedder()
        record(True, "embedding model", f"ready in {time.monotonic() - started:.1f}s (cached in .cache/fastembed)")
    except Exception as exc:
        record(False, "embedding model", f"unavailable ({exc}); runs will fall back to BM25-only")

    try:
        settings = config.llm_settings_from_env(provider)
    except ValueError as exc:
        record(False, "LLM settings", str(exc))
        settings = None
    if settings is None or not settings.enabled:
        record(True, "LLM", "offline mode (no key set) -- pipeline still runs; Q&A needs an LLM")
    else:
        from .llm import LLMClient, LLMError, list_models

        try:
            models = list_models(settings)
            record(settings.model in models, "LLM model access",
                   f"{settings.model} {'available' if settings.model in models else 'NOT available'} "
                   f"({len(models)} models on this key)")
        except LLMError as exc:
            record(False, "LLM model access", str(exc))
        if call_llm:
            try:
                client = LLMClient(settings)
                reply = client.chat([{"role": "user", "content": "Reply with the single word OK."}],
                                    purpose="doctor", max_tokens=200)
                usage = client.ledger.summary()
                record(bool(reply.content), "LLM round trip",
                       f"replied {reply.content!r} ({usage['input_tokens']}+{usage['output_tokens']} tokens)")
            except LLMError as exc:
                record(False, "LLM round trip", str(exc))

    for name in ("flagged_routes.csv", "token_log.json", "eval_report.md", "run_trace.jsonl"):
        path = config.OUTPUT_DIR / name
        record(path.exists(), f"output {name}", "present" if path.exists() else "missing -- run `make all`")
    return results


def render(results: list[tuple[bool, str, str]]) -> str:
    lines = [f"{'OK ' if ok else 'XX '} {name:<26} {detail}" for ok, name, detail in results]
    ready = all(ok for ok, _, _ in results)
    return "\n".join(lines + ["", "READY FOR DEMO" if ready else "NOT READY -- fix the XX lines above"])
