"""
A semantic cache hit serves ANOTHER prompt's answer, so a false positive
here is worse than a miss: a miss costs one upstream call, a false
positive returns a confident, fluent answer to a question nobody asked.

Cosine similarity cannot carry that decision by itself, and not because
the threshold is mistuned — because of what the method measures:

  - Polarity barely moves the vector. "How do I enable the cache?" and
    "How do I disable the cache?" differ in one token and share every
    other one. The answers are opposites.
  - A decisive literal is a small part of a long sentence. "roll back
    migration 003" and "roll back migration 004" differ in one digit.
  - `embed()` truncates to `text[:1000]`, so two long prompts differing
    only after the thousandth character embed IDENTICALLY — cosine 1.0,
    whatever the threshold says.

These tests force cosine to 1.0 with a stub embedder, which is the point
rather than a shortcut: the guard, not the score, is what must refuse
these. That also makes them run anywhere — the real model is a 90 MB
download this suite must not depend on.

The guard is the same shape as the output trimmer's content check: a
match on FORM is confirmed against SUBSTANCE before anything is reused.
"""
from __future__ import annotations

import pytest

from tokenmizer.semantic_cache.cache import (
    EmbeddingEngine,
    SemanticCache,
    _decisive_profile,
    _same_question,
)


@pytest.fixture
def cache(monkeypatch):
    """A cache whose embedder calls every pair a perfect match."""

    class _AlwaysIdentical:
        available = True

        def embed(self, text):
            return [1.0]

    c = SemanticCache(threshold=0.9, ttl_seconds=3600, max_size=100)
    monkeypatch.setattr(c, "_embedder", _AlwaysIdentical())
    monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(lambda a, b: 1.0))
    return c


class TestOppositeQuestionsDoNotShareAnAnswer:
    @pytest.mark.parametrize("stored,asked", [
        ("how do I enable the semantic cache",
         "how do I disable the semantic cache"),
        ("show the archived sessions", "hide the archived sessions"),
        ("add the user to the admins group",
         "remove the user from the admins group"),
        ("when should I start the worker", "when should I stop the worker"),
    ])
    def test_a_polarity_flip_is_not_a_cache_hit(self, cache, stored, asked):
        cache.set(stored, "THE STORED ANSWER", session_id="s")
        assert cache.get(asked, session_id="s") is None, (
            "cosine cleared the threshold, so only the decisive-token "
            "check stands between the caller and the opposite answer"
        )

    def test_the_rejection_is_counted_for_the_operator(self, cache):
        cache.set("enable the cache", "answer", session_id="s")
        cache.get("disable the cache", session_id="s")
        assert cache.stats()["rejected_unsafe"] >= 1


class TestDifferentLiteralsDoNotShareAnAnswer:
    @pytest.mark.parametrize("stored,asked", [
        ("roll back migration 003", "roll back migration 004"),
        ("why does auth.py fail on import", "why does cache.py fail on import"),
        ("what does the --force flag do", "what does the --dry-run flag do"),
        ("explain the max_entry_bytes setting", "explain the max_semantic_scan setting"),
    ])
    def test_a_changed_literal_is_not_a_cache_hit(self, cache, stored, asked):
        cache.set(stored, "THE STORED ANSWER", session_id="s")
        assert cache.get(asked, session_id="s") is None


class TestTheThousandCharacterEmbeddingWindow:
    def test_two_long_prompts_differing_past_the_window_do_not_collide(self, cache):
        """embed() truncates to text[:1000]. Two prompts identical up to
        there embed to the same vector — cosine 1.0 — so the threshold
        is irrelevant and only the decisive-token check can refuse."""
        shared = "Here is the config we discussed, please review it. " * 25
        assert len(shared) > 1000

        cache.set(shared + "Which value should max_size take?", "ANSWER A",
                  session_id="s")
        got = cache.get(shared + "Which value should max_bytes take?",
                        session_id="s")

        assert got is None


class TestAGenuineParaphraseStillHits:
    """The guard must not cost the feature it protects. Polarity is
    compared by CLASS, not by token, so a rephrasing that keeps its
    meaning keeps its cache hit."""

    def test_the_same_polarity_said_differently_still_hits(self, cache):
        cache.set("how do I enable the semantic cache", "ANSWER", session_id="s")
        got = cache.get("how do I turn on the semantic cache", session_id="s")
        assert got is not None and got.response == "ANSWER"

    def test_a_neutral_rewording_still_hits(self, cache):
        cache.set("what is a JWT made of", "ANSWER", session_id="s")
        got = cache.get("what does a JWT consist of", session_id="s")
        assert got is not None

    def test_an_exact_repeat_is_unaffected(self, cache):
        cache.set("what is a JWT", "ANSWER", session_id="s")
        assert cache.get("what is a JWT", session_id="s").response == "ANSWER"


class TestTheProfileItself:
    def test_polarity_is_a_class_not_a_token(self):
        assert _same_question("enable the cache", "turn on the cache")
        assert not _same_question("enable the cache", "disable the cache")

    def test_literals_must_match_exactly(self):
        assert not _same_question("migration 003", "migration 004")
        assert _same_question("migration 003 please", "please, migration 003")

    def test_a_prompt_with_no_decisive_tokens_profiles_as_empty(self):
        polarity, literals = _decisive_profile("what is a JWT")
        assert polarity == 0 and literals == frozenset()

    def test_double_negation_is_not_mistaken_for_a_single_one(self):
        assert not _same_question("do not disable the cache",
                                  "disable the cache")
