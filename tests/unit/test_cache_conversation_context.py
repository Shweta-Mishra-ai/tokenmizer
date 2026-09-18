"""
The semantic cache must key on the conversation a prompt was asked in,
not on the prompt alone.

Inside one coding session the same short prompt recurs constantly and
means something different every time: "continue", "yes", "try again",
"run the tests". The cache keyed on the last user message only, so the
second "continue" of a session was served the first one's answer for the
whole TTL — a wrong answer, silently, on the most common turns there are.

Rule: an entry is only a hit for the same prompt asked in the same
conversation state (SemanticCache.conversation_fingerprint of every
message before the final user turn). A single-turn prompt has no prior
state, so repeated identical one-shot questions still hit, which is
where the cache earns its keep.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse
from tokenmizer.semantic_cache.cache import SemanticCache


class TestFingerprint:

    def test_single_turn_has_empty_fingerprint(self):
        assert SemanticCache.conversation_fingerprint(
            [{"role": "user", "content": "what is a JWT"}]) == ""
        assert SemanticCache.conversation_fingerprint([]) == ""

    def test_prior_turns_change_the_fingerprint(self):
        a = SemanticCache.conversation_fingerprint([
            {"role": "user", "content": "add a login endpoint"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ])
        b = SemanticCache.conversation_fingerprint([
            {"role": "user", "content": "add a logout endpoint"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ])
        assert a and b and a != b

    def test_fingerprint_is_deterministic(self):
        msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
                {"role": "user", "content": "z"}]
        assert (SemanticCache.conversation_fingerprint(msgs)
                == SemanticCache.conversation_fingerprint([dict(m) for m in msgs]))


class TestCacheHonoursContext:

    def test_same_prompt_different_context_is_a_miss(self):
        c = SemanticCache(ttl_seconds=3600, max_size=100)
        c.set("continue", "first answer", session_id="s", context="ctx-A")
        assert c.get("continue", session_id="s", context="ctx-B") is None
        hit = c.get("continue", session_id="s", context="ctx-A")
        assert hit is not None and hit.response == "first answer"

    def test_single_turn_repeat_still_hits(self):
        c = SemanticCache(ttl_seconds=3600, max_size=100)
        c.set("what is a JWT", "explainer", session_id="s")
        assert c.get("what is a JWT", session_id="s").response == "explainer"

    def test_semantic_match_never_crosses_context(self, monkeypatch):
        from tokenmizer.semantic_cache.cache import EmbeddingEngine

        class _Fake:
            available = True

            def embed(self, text):
                return [1.0]

        c = SemanticCache(threshold=0.9, ttl_seconds=3600, max_size=100)
        monkeypatch.setattr(c, "_embedder", _Fake())
        monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(lambda a, b: 1.0))
        c.set("continue please", "first answer", session_id="s", context="ctx-A")
        assert c.get("continue", session_id="s", context="ctx-B") is None
        assert c.get("continue", session_id="s", context="ctx-A") is not None

    def test_invalidate_removes_every_context(self):
        c = SemanticCache(ttl_seconds=3600, max_size=100)
        c.set("continue", "a", session_id="s", context="ctx-A")
        c.set("continue", "b", session_id="s", context="ctx-B")
        assert c.invalidate("continue", session_id="s") == 2
        assert c.get("continue", session_id="s", context="ctx-A") is None


@pytest.fixture
def client(monkeypatch):
    calls = []

    async def chat(messages, **kw):
        calls.append(messages)
        return LLMResponse(text=f"answer {len(calls)}", input_tokens=10,
                           output_tokens=5, model="fake", provider="fake",
                           latency_ms=1.0)

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", True)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    app_module._cache.clear()
    with TestClient(app) as c:
        yield c, calls


def _post(c, session, messages):
    r = c.post("/v1/chat/completions",
               json={"model": "gpt-4o", "session_id": session, "messages": messages})
    assert r.status_code == 200, r.text
    return r.json()


def test_repeated_continue_in_one_session_is_not_served_the_old_answer(client):
    c, calls = client
    first = _post(c, "ctx-proxy", [
        {"role": "user", "content": "add the login endpoint"},
        {"role": "assistant", "content": "added api/auth.py"},
        {"role": "user", "content": "continue"},
    ])
    second = _post(c, "ctx-proxy", [
        {"role": "user", "content": "add the login endpoint"},
        {"role": "assistant", "content": "added api/auth.py"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "added the refresh route"},
        {"role": "user", "content": "continue"},
    ])
    assert first["tokenmizer"]["cache_hit"] is False
    assert second["tokenmizer"]["cache_hit"] is False, (
        "the second 'continue' is a different question — it was answered "
        "from the cache with the first one's reply"
    )
    assert len(calls) == 2


def test_identical_conversation_state_still_hits(client):
    c, calls = client
    msgs = [{"role": "user", "content": "what is a JWT, in one line"}]
    _post(c, "ctx-proxy-hit", msgs)
    again = _post(c, "ctx-proxy-hit", msgs)
    assert again["tokenmizer"]["cache_hit"] is True
    assert len(calls) == 1
