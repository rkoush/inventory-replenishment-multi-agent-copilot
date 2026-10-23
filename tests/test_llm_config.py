"""LLM wiring tests - no network: a fake Groq endpoint records the request payloads."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents.llm import PolicyExtraction, RequestPlan, _strict_schema, get_llm, reasoning_kwargs, smoke_test, structured, strip_reasoning
from agents.report import escape_dollars


def test_default_models_and_env_override():
    code = "import config; print(config.GROQ_MODEL, config.GROQ_ROUTER_MODEL, config.GROQ_MAX_TOKENS, config.GROQ_REASONING_EFFORT)"
    env = {"PATH": "", "GROQ_MODEL": "", "GROQ_ROUTER_MODEL": ""}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "1024", "low"]
    env.update(GROQ_MODEL="qwen/qwen3.8-27b", GROQ_ROUTER_MODEL="openai/gpt-oss-120b")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out[:2] == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]


def test_reasoning_kwargs_only_for_reasoning_models():
    assert reasoning_kwargs("openai/gpt-oss-120b") == {"reasoning_effort": "low", "model_kwargs": {"include_reasoning": False}}
    assert reasoning_kwargs("qwen/qwen3.8-27b") == {"reasoning_effort": "low", "reasoning_format": "hidden"}
    assert reasoning_kwargs("meta-llama/llama-4-scout-17b-16e-instruct") == {}


def test_strict_schema_is_groq_strict_compatible():
    for schema in (RequestPlan, PolicyExtraction):
        s = _strict_schema(schema)
        text = json.dumps(s)
        assert '"default"' not in text
        assert set(s["required"]) == set(s["properties"]) and s["additionalProperties"] is False


def test_strip_reasoning_and_escape_dollars():
    assert strip_reasoning("<think>hmm</think>\nPlan is fine.") == "Plan is fine."
    assert escape_dollars("$4,000 of $5,000") == r"\$4,000 of \$5,000"
    assert escape_dollars(r"already \$1") == r"already \$1"


class _FakeGroq(BaseHTTPRequestHandler):
    requests: list[dict] = []
    content = "OK"

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": self.reply(body)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def reply(self, body: dict) -> str:
        return type(self).content

    def log_message(self, *_):
        pass


@pytest.fixture
def fake_groq(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _FakeGroq)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_API_BASE", f"http://127.0.0.1:{srv.server_port}")
    _FakeGroq.requests = []
    yield _FakeGroq
    srv.shutdown()


def test_gpt_oss_payload_and_structured_parse(fake_groq):
    fake_groq.content = json.dumps({"store": "S001", "budget": 5000, "horizon_days": 7,
                                    "plan": ["inventory_agent"], "reasoning": "ok"})
    llm = get_llm(model="openai/gpt-oss-120b")
    out = structured(llm, RequestPlan).invoke("Plan S001 with $5,000")
    assert isinstance(out, RequestPlan) and out.budget == 5000
    req = fake_groq.requests[-1]
    assert req["model"] == "openai/gpt-oss-120b"
    assert req["reasoning_effort"] == "low" and req["include_reasoning"] is False
    assert "reasoning_format" not in req  # not supported by gpt-oss; exclusive with include_reasoning
    assert req["max_tokens"] == 1024
    rf = req["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True


def test_non_reasoning_model_gets_no_reasoning_params(fake_groq):
    fake_groq.content = "fine"
    get_llm(model="meta-llama/llama-4-scout-17b-16e-instruct").invoke("hi")
    req = fake_groq.requests[-1]
    assert req.get("reasoning_effort") is None and req.get("reasoning_format") is None
    assert "include_reasoning" not in req


def test_smoke_test_reports_ok_and_failure(fake_groq, monkeypatch):
    fake_groq.content = "OK"
    ok, msg = smoke_test("openai/gpt-oss-20b")
    assert ok and "openai/gpt-oss-20b" in msg and fake_groq.requests[-1]["max_tokens"] == 256
    monkeypatch.setenv("GROQ_API_BASE", "http://127.0.0.1:9")  # nothing listening
    monkeypatch.setattr("config.GROQ_MAX_RETRIES", 0)
    ok, msg = smoke_test("openai/gpt-oss-20b")
    assert not ok and "FAILED" in msg
    monkeypatch.delenv("GROQ_API_KEY")
    assert smoke_test()[0] is False


class _FakeGroqGraph(_FakeGroq):
    """Answers by requested schema, like gpt-oss would with strict structured outputs."""

    def reply(self, body: dict) -> str:
        name = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        return {
            "RequestPlan": json.dumps({"store": "S001", "budget": 5000, "horizon_days": 7, "reasoning": "standard",
                                       "plan": ["inventory_agent", "demand_agent", "policy_agent", "optimizer_agent"]}),
            "PolicyExtraction": json.dumps({
                "safety_stock": [{"category": c, "days": d} for c, d in
                                 [("Grocery", 3), ("Household", 7), ("Personal Care", 7), ("Electronics", 10)]],
                "max_sku_budget_share_pct": 30, "weekly_budget_cap": 5000,
                "promos": [{"sku": "HH-002", "uplift_pct": 40}, {"sku": "ELE-001", "uplift_pct": 25}],
                "summary": "Weekly cap $5,000; no SKU over 30%."}),
        }.get(name, "The plan spends most of the $5,000 budget on low-cover SKUs.")


def test_full_graph_on_gpt_oss_path_has_no_fallbacks(mcp_url, monkeypatch):
    from tests.test_graph_offline import run_graph

    srv = HTTPServer(("127.0.0.1", 0), _FakeGroqGraph)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_API_BASE", f"http://127.0.0.1:{srv.server_port}")
    _FakeGroq.requests = []
    try:
        state, _ = run_graph(get_llm(model="openai/gpt-oss-120b"), mcp_url, ["approve"])
    finally:
        srv.shutdown()
    assert not [w for w in state["warnings"] if "failed" in w.lower()], state["warnings"]
    assert len(_FakeGroq.requests) == 3  # plan + policy + report
    assert state["purchase_order"]["po_id"].startswith("PO-")
    assert "low-cover SKUs" in state["report"]
