"""Mock tests for the llm fallback chain (no network).

Run from the project root:  python -m pytest tests/ -q
"""
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llm  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "You are Lee Kuan Yew. Answer only from the passages."},
    {"role": "user", "content": "PASSAGES\n[S1] Test Speech — 1 Jan 2000 (NAS)\nSome text.\n\nQUESTION\nWhy?"},
]

LONG_ANSWER = "This is a grounded answer that cites the speech and is long enough."


def _ok_response(content=LONG_ANSWER, usage=None, finish_reason="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 2000, "completion_tokens": 300}}


def _client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("VIKEY_API_KEY", "test-vikey-key")
    monkeypatch.setenv("VIKEY_BASE_URL", "https://api.vikey.test/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.delenv("VIKEY_MODEL", raising=False)
    monkeypatch.delenv("VIKEY_MODEL_FALLBACK", raising=False)
    llm.reset_state()
    yield
    llm.reset_state()


def test_tier1_success_no_fallback():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES)
    assert out["ok"] is True
    assert out["fallback_used"] is False
    assert out["provider"] == "openrouter"
    assert out["answer"] == LONG_ANSWER
    assert len(out["attempts"]) == 1
    assert out["attempts"][0]["ok"] is True
    assert out["est_cost_idr"] == 0.0
    # Gemma merges system + user into a single user message.
    assert len(seen) == 1
    assert [m["role"] for m in seen[0]["messages"]] == ["user"]


def test_tier1_429_falls_back_and_cools_down():
    def handler(request):
        if "openrouter.ai" in str(request.url):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    first = llm.generate(MESSAGES)
    assert first["ok"] is True
    assert first["fallback_used"] is True
    assert first["provider"] == "vikey"
    assert first["model"] == "openai/gpt-6-luna"
    assert [a["error"] for a in first["attempts"] if not a["ok"]] == ["rate_limited"]
    # (2000*720 + 300*3600) / 1e6 = 2.52
    assert first["est_cost_idr"] == 2.52

    # Second call skips tier 1 (still in cooldown) and goes straight to Vikey.
    second = llm.generate(MESSAGES)
    assert second["ok"] is True
    assert second["attempts"][0]["error"] == "skipped_cooldown"
    assert second["attempts"][1]["ok"] is True


def test_tier1_404_tier2_timeout_tier3_succeeds():
    def handler(request):
        body = json.loads(request.content)
        if "openrouter.ai" in str(request.url):
            return httpx.Response(404, json={"error": "model not found"})
        if body["model"] == "openai/gpt-6-luna":
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES)
    assert out["ok"] is True
    assert out["provider"] == "vikey"
    assert out["model"] == "deepseek/deepseek-v4-flash"
    assert out["fallback_used"] is True
    assert len(out["attempts"]) == 3
    assert [a["error"] for a in out["attempts"][:2]] == ["model_unavailable", "timeout"]
    assert out["attempts"][2]["ok"] is True


def test_all_tiers_fail_returns_no_answer():
    def handler(request):
        if "openrouter.ai" in str(request.url):
            return httpx.Response(500, json={"error": "server error"})
        raise httpx.ConnectError("connection refused")

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES)
    assert out["ok"] is False
    assert out["answer"] is None
    assert out["provider"] is None
    assert len(out["attempts"]) == 3
    assert out["attempts"][0]["error"] == "server_error"
    assert out["attempts"][1]["error"] == "error"
    assert out["attempts"][2]["error"] == "error"


def test_empty_response_moves_to_next_tier():
    def handler(request):
        if "openrouter.ai" in str(request.url):
            return httpx.Response(200, json=_ok_response(content=""))
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES)
    assert out["ok"] is True
    assert out["fallback_used"] is True
    assert out["provider"] == "vikey"
    assert out["attempts"][0]["error"] == "empty_response"


def test_think_block_stripped():
    def handler(request):
        return httpx.Response(200, json=_ok_response(
            content="<think>Let me reason about this at length.</think> " + LONG_ANSWER))

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES)
    assert out["ok"] is True
    assert "<think>" not in out["answer"]
    assert "reason about this" not in out["answer"]
    assert out["answer"] == LONG_ANSWER


def test_prefer_luna_tried_first():
    """prefer='luna' reorders the chain: Luna answers first, no fallback."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES, prefer="luna")
    assert out["ok"] is True
    assert out["model"] == "openai/gpt-6-luna"
    assert out["fallback_used"] is False
    assert len(out["attempts"]) == 1
    assert seen == ["openai/gpt-6-luna"]


def test_prefer_pro_falls_back_to_free():
    """prefer='pro' tries Pro first; on failure it falls back to the free tier."""
    seen = []

    def handler(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "deepseek/deepseek-v4-pro":
            return httpx.Response(500, json={"error": "server error"})
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES, prefer="pro")
    assert out["ok"] is True
    assert out["model"] == "google/gemma-4-26b-a4b-it:free"
    assert out["fallback_used"] is True
    assert seen[0] == "deepseek/deepseek-v4-pro"
    assert out["attempts"][0]["error"] == "server_error"
    assert out["attempts"][1]["ok"] is True
    assert out["attempts"][1]["model"] == "google/gemma-4-26b-a4b-it:free"


def test_prefer_auto_keeps_default_order():
    """prefer='auto' is the same as no preference: free tier first."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES, prefer="auto")
    assert out["ok"] is True
    assert out["model"] == "google/gemma-4-26b-a4b-it:free"
    assert out["fallback_used"] is False
    assert seen == ["google/gemma-4-26b-a4b-it:free"]


def test_prefer_flash_skips_free_when_it_fails():
    """prefer='flash' tries Flash first; on failure it falls back to free."""
    seen = []

    def handler(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "deepseek/deepseek-v4-flash":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_ok_response())

    llm.set_client(_client_for(handler))
    out = llm.generate(MESSAGES, prefer="flash")
    assert out["ok"] is True
    assert out["model"] == "google/gemma-4-26b-a4b-it:free"
    assert out["fallback_used"] is True
    assert seen[0] == "deepseek/deepseek-v4-flash"
    assert out["attempts"][0]["error"] == "rate_limited"
