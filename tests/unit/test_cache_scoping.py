"""
Regression tests for TM-03: the semantic cache must not leak one
session's cached answers to another by default.

Two distinct bugs, both fixed here:

1. `set()`'s scoping decision: any prompt that failed a five-regex
   "looks sensitive" heuristic was stored under a `__shared__` scope
   readable by every session. Ordinary confidential business content
   (e.g. "what's our enterprise churn rate this quarter") sails straight
   through that heuristic. Fix: default to session-scoped for everything;
   sharing is now an explicit opt-in (`cache.share_scope: "shared"`).

2. Even with `set()` correctly scoped, `get()`'s semantic-similarity
   fallback (layer 2) scanned ALL entries in `self._exact` regardless of
   scope — so a private, session-scoped entry from a DIFFERENT session
   could still be returned to a near-miss query, entirely bypassing
   whatever scope `set()` assigned. This is arguably the more dangerous
   half: it means correct exact-match scoping alone does not close the
   leak. Fix: `CacheEntry` now records its own `scope`, and the semantic
   layer only considers entries whose scope is `__shared__` or exactly
   matches the caller's own session_id.
"""
from __future__ import annotations

import pytest

from tokenmizer.semantic_cache.cache import EmbeddingEngine, SemanticCache


class _FakeEmbedder:
    """Deterministic stand-in for the real sentence-transformer model —
    avoids depending on a downloaded model for a scope-filtering test.
    Each prompt maps to a fixed vector; near-identical prompts get
    near-identical vectors so cosine similarity behaves predictably."""

    available = True

    def __init__(self, vectors: dict[str, float]):
        self._vectors = vectors

    def embed(self, text: str):
        # A 1-D "vector" is enough to drive EmbeddingEngine.cosine's dot
        # product predictably for this test.
        return [self._vectors.get(text, 0.0)]


@pytest.fixture
def cache_with_fake_embedder(monkeypatch):
    c = SemanticCache(threshold=0.90, ttl_seconds=3600, max_size=100)
    fake = _FakeEmbedder({
        "what is our enterprise churn rate this quarter": 1.0,
        "what's the churn rate for enterprise this quarter": 1.0,  # near-duplicate
    })
    monkeypatch.setattr(c, "_embedder", fake)
    monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(lambda a, b: 1.0 if a == b else 0.0))
    return c


class TestDefaultScopingIsSessionOnly:

    def test_non_sensitive_prompt_is_not_shared_by_default(self):
        """The core reproduction: an ordinary business question, stored
        by one session, must not be readable by a different session's
        exact-match lookup under the DEFAULT configuration."""
        cache = SemanticCache()
        cache.set(
            "What is the churn rate for our enterprise tier this quarter?",
            "Your enterprise churn was 4.2% — driven by the Acme account.",
            session_id="tenant-A",
        )
        leaked = cache.get(
            "What is the churn rate for our enterprise tier this quarter?",
            session_id="tenant-B",
        )
        assert leaked is None, "tenant-B must not see tenant-A's cached answer by default"

    def test_same_session_can_still_read_its_own_entry(self):
        cache = SemanticCache()
        cache.set("what is a JWT", "JSON Web Token explainer", session_id="tenant-A")
        hit = cache.get("what is a JWT", session_id="tenant-A")
        assert hit is not None
        assert hit.response == "JSON Web Token explainer"

    def test_no_session_id_is_scoped_private_not_shared(self):
        """Even with no session_id at all (caller didn't pass one), the
        default must not be the old global __shared__ bucket."""
        cache = SemanticCache()
        cache.set("what is a JWT", "JSON Web Token explainer")
        # A DIFFERENT caller with an explicit session_id must not see it.
        assert cache.get("what is a JWT", session_id="some-other-session") is None


class TestOptInSharedScope:

    def test_shared_opt_in_restores_cross_session_sharing_for_non_sensitive(self):
        cache = SemanticCache(share_scope="shared")
        cache.set("what is a JWT", "JSON Web Token explainer", session_id="tenant-A")
        hit = cache.get("what is a JWT", session_id="tenant-B")
        assert hit is not None, (
            "with share_scope='shared', a non-sensitive prompt should be "
            "readable across sessions (explicit opt-in restores old behavior)"
        )

    def test_shared_opt_in_still_scopes_sensitive_content(self):
        """Opting into sharing must never override the sensitivity gate —
        a prompt that looks like it contains secrets/PII/business-specific
        content stays session-scoped no matter what."""
        cache = SemanticCache(share_scope="shared")
        cache.set(
            "my database url is postgres://user:pass@host/db",
            "noted",
            session_id="tenant-A",
        )
        leaked = cache.get(
            "my database url is postgres://user:pass@host/db",
            session_id="tenant-B",
        )
        assert leaked is None


class TestSemanticLayerRespectsSameScoping:
    """The deeper half of TM-03: the similarity-based fallback (layer 2)
    must not bypass scoping just because it's a near-miss instead of an
    exact match."""

    def test_semantic_match_never_crosses_session_scope(self, cache_with_fake_embedder):
        cache = cache_with_fake_embedder
        cache.set(
            "what is our enterprise churn rate this quarter",
            "4.2%, driven by the Acme account",
            session_id="tenant-A",
        )
        # A near-duplicate phrasing from a DIFFERENT session — would hit
        # via semantic similarity (cosine=1.0 under the fake embedder) if
        # scope weren't checked.
        leaked = cache.get(
            "what's the churn rate for enterprise this quarter",
            session_id="tenant-B",
        )
        assert leaked is None, (
            "semantic-similarity fallback returned another session's "
            "private cache entry — scope must be checked in the "
            "similarity loop too, not only for exact-key lookups"
        )

    def test_semantic_match_still_works_within_the_same_session(self, cache_with_fake_embedder):
        cache = cache_with_fake_embedder
        cache.set(
            "what is our enterprise churn rate this quarter",
            "4.2%, driven by the Acme account",
            session_id="tenant-A",
        )
        hit = cache.get(
            "what's the churn rate for enterprise this quarter",
            session_id="tenant-A",
        )
        assert hit is not None
        assert hit.response == "4.2%, driven by the Acme account"

    def test_semantic_match_works_for_shared_scope_entries(self, monkeypatch):
        cache = SemanticCache(threshold=0.90, share_scope="shared")
        fake = _FakeEmbedder({
            "what is a jwt": 1.0,
            "explain jwt to me": 1.0,
        })
        monkeypatch.setattr(cache, "_embedder", fake)
        monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(lambda a, b: 1.0 if a == b else 0.0))

        cache.set("what is a jwt", "JSON Web Token explainer", session_id="tenant-A")
        hit = cache.get("explain jwt to me", session_id="tenant-B")
        assert hit is not None, "shared-scope entries must still be reachable via semantic match"
