"""Minimal OpenAI-compatible chat client with retries, a token ledger and a disk cache.

Works with Groq, Ollama, OpenAI or anything speaking /chat/completions -- no SDK
dependency. Section 8 patterns: exponential backoff with jitter on transient
errors, a hard retry cap, and a typed error the caller must handle (we never
swallow a failure and carry on as if the call succeeded).
"""
'''
json -> python list ; dict -> python objecvta
random for jitter
threading for parallel execution (to stop corruption )
time pausng like for retry and how long a call took
haslib creates hashes so full prompt for hash so that we can use chache arther new llm call
urrllib -> handle http calls so that we can have as less as external libraries depenedenices as low as possible


'''

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, LLMSettings

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
USER_AGENT = "freight-watch/1.0"


# Models often emit typographic dashes / spaces ("Chennai\u2011Bangalore", "2025\u201103\u201103").
# Normalise to ASCII so route names and ISO dates match plain-text checks downstream.
_TYPOGRAPHY = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"}
                            | {c: " " for c in "\u00a0\u2009\u202f\u2007"})


def normalize_text(text: str | None) -> str | None:
    return text.translate(_TYPOGRAPHY) if text else text


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    def as_assistant_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


@dataclass
class CallRecord:
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cached: bool
    attempts: int
    error: str | None = None


class UsageLedger:
    """Thread-safe log of every LLM call -- feeds the token/cost report."""

    def __init__(self, input_per_m: float = 0.0, output_per_m: float = 0.0):
        self.input_per_m, self.output_per_m = input_per_m, output_per_m
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

# here locking should be done so that if there is more complex data structure rather than list we ensure that as we perform read we dont get inconsisitent as there can be half write from add method as gil ensure this donest happens for list but as ds changes this whole falls apart

    def summary(self) -> dict:
        failed = [r for r in self.records if r.error]
        live = [r for r in self.records if not r.cached and not r.error]
        tokens_in = sum(r.input_tokens for r in live)
        tokens_out = sum(r.output_tokens for r in live)
        by_purpose: dict[str, dict] = {}
        for r in self.records:
            p = by_purpose.setdefault(r.purpose, {"calls": 0, "cached": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0})
            p["calls"] += 1
            p["cached"] += int(r.cached)
            p["failed"] += int(bool(r.error))
            p["input_tokens"] += r.input_tokens
            p["output_tokens"] += r.output_tokens
        return {
            "llm_calls_total": len(self.records),
            "llm_calls_live": len(live),
            "llm_calls_served_from_cache": sum(r.cached for r in self.records),
            "llm_calls_failed": len(failed),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "rate_usd_per_1m": {"input": self.input_per_m, "output": self.output_per_m},
            "estimated_cost_usd": round(tokens_in / 1e6 * self.input_per_m + tokens_out / 1e6 * self.output_per_m, 6),
            "by_purpose": by_purpose,
        }

    def to_json(self) -> dict:
        return {"summary": self.summary(), "calls": [asdict(r) for r in self.records]}


class ResponseCache:
    """Content-addressed cache: same model + same request -> same stored answer."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1, sort_keys=True))


def model_specific_params(model: str) -> dict:
    """gpt-oss models reason before answering; keep that short and out of the response
    so the token budget goes to the actual answer (Groq: reasoning_effort, include_reasoning)."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}


def list_models(settings: LLMSettings) -> list[str]:
    """Model ids the configured key can access (GET /models)."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = urllib.request.Request(settings.base_url.rstrip("/") + "/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_s) as resp:
            return sorted(m["id"] for m in json.loads(resp.read()).get("data", []))
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc


class LLMClient:
    def __init__(self, settings: LLMSettings, ledger: UsageLedger | None = None):
        if not settings.enabled:
            raise LLMError("LLM provider is 'offline'")
        self.settings = settings
        self.ledger = ledger or UsageLedger(settings.input_per_m, settings.output_per_m)
        self.cache = ResponseCache(CACHE_DIR / "llm_cache.json") if settings.use_cache else None

    @property
    def model(self) -> str:
        return self.settings.model

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str,
        tools: list[dict] | None = None,
        max_tokens: int = 300,
        json_mode: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "seed": self.settings.seed,
            "max_tokens": max_tokens,
        }
        payload.update(model_specific_params(self.settings.model))
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)):
            result = ChatResult(**hit, cached=True)
            result.content = normalize_text(result.content)
            self.ledger.add(CallRecord(purpose, self.model, result.input_tokens, result.output_tokens, 0, True, 0))
            return result

        started = time.monotonic()
        try:
            body, attempts = self._post_with_retry(payload)
        except LLMError as exc:
            latency = int((time.monotonic() - started) * 1000)
            self.ledger.add(CallRecord(purpose, self.model, 0, 0, latency, False, self.settings.max_retries, str(exc)[:200]))
            raise
        message = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        result = ChatResult(
            content=normalize_text(message.get("content")),
            tool_calls=message.get("tool_calls") or [],
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
        latency = int((time.monotonic() - started) * 1000)
        self.ledger.add(CallRecord(purpose, self.model, result.input_tokens, result.output_tokens, latency, False, attempts))
        if self.cache:
            self.cache.put(key, {k: v for k, v in asdict(result).items() if k != "cached"})
        return result

    def _post_with_retry(self, payload: dict) -> tuple[dict, int]:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        # Cloudflare in front of some providers (Groq) rejects urllib's default
        # "Python-urllib/x.y" User-Agent with HTTP 403 / error code 1010.
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        data = json.dumps(payload).encode()

        last_error = "unknown error"
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as resp:
                    return json.loads(resp.read()), attempt
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS:
                    break  # 4xx like bad request / auth: retrying will not help
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.settings.max_retries:
                time.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))  # 1s, 2s, 4s + jitter
        raise LLMError(f"LLM call failed after retries: {last_error}")
