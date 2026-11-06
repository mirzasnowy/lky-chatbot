"""Model fallback chain for the LKY chatbot (Section 3, 99 Group AI Challenge).

Tier 1: a free OpenRouter model (default google/gemma-4-26b-a4b-it:free).
Tier 2: Vikey openai/gpt-6-luna (paid, ~Rp 4/answer; cheapest tested-good
model: Rp 720 in / Rp 3,600 out per 1M tokens).
Tier 3: Vikey deepseek/deepseek-v4-flash (paid, ~Rp 5/answer).

One attempt per tier; the next tier is the retry. A tier that fails gets a
cooldown so later requests skip it instead of waiting on a call that will
fail. Paid calls are capped per UTC day (PAID_DAILY_CAP, default 300).

All configuration comes from env vars; nothing secret is in this file.
VIKEY_BASE_URL is used exactly as given + "/chat/completions" (never guessed).
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger("lky.llm")

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL_DEFAULT = "google/gemma-4-26b-a4b-it:free"
VIKEY_MODEL_DEFAULT = "openai/gpt-6-luna"
VIKEY_MODEL_FALLBACK_DEFAULT = "deepseek/deepseek-v4-flash"
VIKEY_MODEL_PRO_DEFAULT = "deepseek/deepseek-v4-pro"

# Model choices the UI offers (POST /chat "model" field). "auto" is the
# default chain order below; a named choice is tried first, then the rest
# of the auto chain in order. "pro" is NOT in the auto chain (cost/quality,
# as decided earlier) — it only runs when explicitly chosen.
MODEL_CHOICES = ("auto", "free", "luna", "flash", "pro")

# Rp per 1 million tokens (in, out), from Vikey's price list.
PRICE_IDR = {
    "openai/gpt-6-luna": (720, 3600),
    "deepseek/deepseek-v4-pro": (1800, 3600),
    "deepseek/deepseek-v4-flash": (1080, 1800),
}

_client = httpx.Client()
_cooldown: dict[str, float] = {}
_paid = {"day": None, "n": 0}


class _Fail(Exception):
    """Internal: one tier failed with a classified error code."""


def set_client(client):
    """Replace the HTTP client (used by tests with a MockTransport)."""
    global _client
    _client = client


def reset_state():
    """Clear cooldowns and the paid-call counter (used by tests)."""
    _cooldown.clear()
    _paid.update(day=None, n=0)


def _tier(name, provider, tier, url, key, model, timeout, headers=None):
    return {"name": name, "provider": provider, "tier": tier,
            "url": url, "key": key, "model": model, "timeout": timeout,
            "headers": headers or {}}


def _vikey_tier(name, env, default):
    base, vkey = os.getenv("VIKEY_BASE_URL"), os.getenv("VIKEY_API_KEY")
    if not (base and vkey):
        return None
    return _tier(name, "vikey", "paid",
                 base.rstrip("/") + "/chat/completions", vkey,
                 os.getenv(env, default), 40.0)


def _openrouter_tier():
    if not os.getenv("OPENROUTER_API_KEY"):
        return None
    return _tier("openrouter", "openrouter", "free", OR_URL,
                 os.environ["OPENROUTER_API_KEY"],
                 os.getenv("OPENROUTER_MODEL", OR_MODEL_DEFAULT), 25.0,
                 {"HTTP-Referer": "https://github.com/mirzasnowy/lky-chatbot",
                  "X-Title": "LKY chatbot (99 Group AI Challenge, Section 3)"})


def tier_for_choice(choice):
    """The tier dict for one UI model choice, or None if its key is missing."""
    if choice == "free":
        return _openrouter_tier()
    if choice == "luna":
        return _vikey_tier("vikey-luna", "VIKEY_MODEL", VIKEY_MODEL_DEFAULT)
    if choice == "flash":
        return _vikey_tier("vikey-flash", "VIKEY_MODEL_FALLBACK",
                           VIKEY_MODEL_FALLBACK_DEFAULT)
    if choice == "pro":
        return _vikey_tier("vikey-pro", "VIKEY_MODEL_PRO",
                           VIKEY_MODEL_PRO_DEFAULT)
    return None


def build_chain():
    """The default (auto) tier list from env. A tier whose key is missing
    is skipped. Pro is NOT in the auto chain; it only runs when chosen."""
    chain = []
    t = _openrouter_tier()
    if t:
        chain.append(t)
    for tier in (_vikey_tier("vikey-luna", "VIKEY_MODEL", VIKEY_MODEL_DEFAULT),
                 _vikey_tier("vikey-flash", "VIKEY_MODEL_FALLBACK",
                             VIKEY_MODEL_FALLBACK_DEFAULT)):
        if tier:
            chain.append(tier)
    return chain


def _paid_allowed():
    today = datetime.now(timezone.utc).date()
    if _paid["day"] != today:
        _paid.update(day=today, n=0)
    if _paid["n"] >= int(os.getenv("PAID_DAILY_CAP", "300")):
        return False
    _paid["n"] += 1
    return True


def _classify(status, retry_after):
    if status == 429:
        return "rate_limited", min(retry_after or 300, 900)
    if status in (401, 403):
        return "auth_error", 3600
    if status == 402:
        return "no_credit", 1800
    if status == 404:
        return "model_unavailable", 1800
    if status >= 500:
        return "server_error", 60
    return "bad_request", 60


def _retry_after(response):
    try:
        return int(response.headers.get("retry-after", ""))
    except (ValueError, TypeError):
        return None


def _prep(tier, messages):
    # Some Gemma endpoints reject role=system: merge system into the first
    # user message instead.
    if "gemma" in tier["model"].lower():
        if len(messages) >= 2:
            return [{"role": "user",
                     "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        return messages
    return messages


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def generate(messages, only_tier=None, prefer=None):
    """Run the fallback chain. only_tier='paid' is for the example-chat
    script (best-quality tier); it is never exposed by the public API.
    prefer is a UI model choice (see MODEL_CHOICES): that tier is tried
    first, then the rest of the auto chain in order. 'auto'/None keeps
    the default order (free -> luna -> flash; pro is never in it)."""
    attempts = []
    chain = build_chain()
    if prefer and prefer != "auto":
        pref = tier_for_choice(prefer)
        if pref is not None:
            chain = [pref] + [t for t in chain if t["name"] != pref["name"]]
    chain = [t for t in chain if only_tier in (None, t["tier"])]
    for i, tier in enumerate(chain):
        rec = {"provider": tier["provider"], "model": tier["model"],
               "tier": tier["tier"]}
        t0 = time.monotonic()
        if _cooldown.get(tier["name"], 0) > t0:
            attempts.append({**rec, "ok": False, "error": "skipped_cooldown"})
            continue
        if tier["tier"] == "paid" and not _paid_allowed():
            attempts.append({**rec, "ok": False, "error": "budget_cap"})
            continue
        err = "error"
        try:
            r = _client.post(
                tier["url"],
                headers={"Authorization": "Bearer " + tier["key"],
                         **tier["headers"]},
                json={"model": tier["model"], "messages": _prep(tier, messages),
                      "temperature": 0.3, "max_tokens": 1000},
                timeout=httpx.Timeout(tier["timeout"], connect=5.0))
            if r.status_code != 200:
                err, cd = _classify(r.status_code, _retry_after(r))
                _cooldown[tier["name"]] = time.monotonic() + cd
                raise _Fail(err)
            data = r.json()
            choice = data["choices"][0]
            text = _clean(choice["message"].get("content"))
            if len(text) < 20:
                raise _Fail("empty_response")
            usage = data.get("usage") or {}
            pin, pout = PRICE_IDR.get(tier["model"], (0, 0))
            cost = (round((usage.get("prompt_tokens", 0) * pin
                           + usage.get("completion_tokens", 0) * pout) / 1e6, 2)
                    if tier["tier"] == "paid" else 0.0)
            attempts.append({**rec, "ok": True,
                             "ms": int((time.monotonic() - t0) * 1000)})
            return {"ok": True, "answer": text, "provider": tier["provider"],
                    "model": tier["model"], "tier": tier["tier"],
                    "fallback_used": i > 0,
                    "truncated": choice.get("finish_reason") == "length",
                    "usage": usage, "est_cost_idr": cost, "attempts": attempts}
        except _Fail as e:
            err = str(e)
        except httpx.TimeoutException:
            err = "timeout"
            _cooldown[tier["name"]] = time.monotonic() + 60
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            err = "error"
        attempts.append({**rec, "ok": False, "error": err,
                         "ms": int((time.monotonic() - t0) * 1000)})
    return {"ok": False, "answer": None, "provider": None, "model": None,
            "tier": None, "fallback_used": False, "truncated": False,
            "usage": {}, "est_cost_idr": 0.0, "attempts": attempts}


def log_answer(result, top_score, source_chunk_ids, latency_ms):
    """One JSON line per answer. Never logs question text, IPs, or keys."""
    log.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": result.get("provider"), "model": result.get("model"),
        "tier": result.get("tier"), "fallback_used": result.get("fallback_used"),
        "attempts": [{"model": a["model"], "ok": a.get("ok", False),
                      "error": a.get("error")}
                     for a in result.get("attempts", [])],
        "top_score": top_score, "source_chunk_ids": source_chunk_ids,
        "tokens": result.get("usage", {}),
        "est_cost_idr": result.get("est_cost_idr", 0.0),
        "latency_ms": latency_ms,
    }))
