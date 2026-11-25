"""HTTP client behaviour against a local mock OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from freight_watch import llm as llm_module
from freight_watch.config import LLMSettings
from freight_watch.llm import LLMClient, LLMError


class MockServer:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                status = outer.statuses.pop(0) if outer.statuses else 200
                payload = ({"choices": [{"message": {"content": "ok"}}],
                            "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
                           if status == 200 else {"error": "nope"})
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_module, "CACHE_DIR", tmp_path)


def client(url, use_cache=False):
    return LLMClient(LLMSettings(provider="mock", base_url=url, model="m", input_per_m=1.0, output_per_m=2.0,
                                 use_cache=use_cache))


def test_retries_transient_errors_then_records_usage():
    server = MockServer([503, 429, 200])
    try:
        c = client(server.url)
        result = c.chat([{"role": "user", "content": "hi"}], purpose="t")
        assert result.content == "ok" and len(server.requests) == 3
        assert server.requests[0]["temperature"] == 0.0 and server.requests[0]["seed"] == 7
        summary = c.ledger.summary()
        assert (summary["input_tokens"], summary["output_tokens"], summary["llm_calls_live"]) == (11, 3, 1)
        assert summary["estimated_cost_usd"] == pytest.approx(11 / 1e6 + 6 / 1e6)
        assert c.ledger.records[0].attempts == 3
    finally:
        server.close()


def test_sends_explicit_user_agent():
    # Groq's Cloudflare front door answers urllib's default UA with 403 / error 1010.
    server = MockServer([200])
    try:
        client(server.url).chat([{"role": "user", "content": "hi"}], purpose="t")
        assert server.headers[0]["User-Agent"] == "freight-watch/1.0"
    finally:
        server.close()


def test_does_not_retry_auth_errors_and_logs_the_failure():
    server = MockServer([403, 200])
    try:
        c = client(server.url)
        with pytest.raises(LLMError, match="403"):
            c.chat([{"role": "user", "content": "hi"}], purpose="t")
        assert len(server.requests) == 1
        summary = c.ledger.summary()
        assert summary["llm_calls_failed"] == 1 and summary["llm_calls_live"] == 0
    finally:
        server.close()


def test_gives_up_after_max_retries():
    server = MockServer([500, 500, 500, 200])
    try:
        with pytest.raises(LLMError):
            client(server.url).chat([{"role": "user", "content": "hi"}], purpose="t")
        assert len(server.requests) == 3
    finally:
        server.close()


def test_cache_serves_identical_requests_without_a_network_call():
    server = MockServer([])
    try:
        c = client(server.url, use_cache=True)
        c.chat([{"role": "user", "content": "hi"}], purpose="t")
        again = c.chat([{"role": "user", "content": "hi"}], purpose="t")
        assert again.cached and len(server.requests) == 1
        assert c.ledger.summary()["llm_calls_served_from_cache"] == 1
    finally:
        server.close()


def test_reasoning_params_only_for_gpt_oss_models():
    server = MockServer([200, 200])
    try:
        for model in ("openai/gpt-oss-120b", "qwen2.5:7b-instruct"):
            LLMClient(LLMSettings(provider="mock", base_url=server.url, model=model, use_cache=False)).chat(
                [{"role": "user", "content": "hi"}], purpose="t")
        assert server.requests[0]["reasoning_effort"] == "low" and server.requests[0]["include_reasoning"] is False
        assert "reasoning_effort" not in server.requests[1]
    finally:
        server.close()
