"""
Regression tests for the second-pass audit fixes.

Each class documents the behaviour that was actually observed before the
fix, so a future change that reintroduces it fails here with an
explanation rather than just a red assertion.
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.decision_tracker import _is_same_decision
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


def _graph(tmp_path, session_id="s"):
    return GraphMemory(session_id, storage_dir=str(tmp_path))


class TestDecisionSwapsAreNotSwallowed:
    """
    _is_same_decision merged on >=0.82 word overlap. Two decisions in one
    slot differ by exactly one word — the technology name, which is the
    whole meaning — so the check got MORE wrong as labels got more
    descriptive:

        "Use MySQL"                       vs "Use MongoDB"                       -> 0.50 (ok)
        "Use MySQL for the user database" vs "Use MongoDB for the user database" -> 0.83 (merged!)

    A merged swap meant: no new node, no SUPERSEDED status, no
    DecisionTransition — so /api/graph/{id}/why and the why_decision MCP
    tool returned nothing for the exact case they exist to answer.
    """

    @pytest.mark.parametrize("a,b", [
        ("Use MySQL for the user database", "Use MongoDB for the user database"),
        ("We will use Redis as the session cache backend",
         "We will use Memcached as the session cache backend"),
        ("Use React for the frontend", "Use Next.js for the frontend"),
        ("Use MySQL", "Use MongoDB"),
    ])
    def test_competing_alternatives_are_distinct(self, a, b):
        assert not _is_same_decision(a, b), (
            f"{a!r} and {b!r} are competing choices for one slot; merging "
            f"them discards the change and records no transition"
        )

    @pytest.mark.parametrize("a,b", [
        ("Use React", "use React for the frontend."),
        ("Use PostgreSQL", "Use PostgreSQL 16 with pgvector"),
        ("Disable caching", "disable caching for the API"),
    ])
    def test_restatements_still_merge(self, a, b):
        """Dedup must keep working: a reworded or refined statement of the
        SAME choice is one node, not a spurious self-supersession."""
        assert _is_same_decision(a, b)

    @pytest.mark.parametrize("a,b", [
        ("Enable JWT auth for the API", "Disable JWT auth for the API"),
        ("Use Docker for deployment", "Do not use Docker for deployment"),
    ])
    def test_semantic_opposites_still_distinct(self, a, b):
        assert not _is_same_decision(a, b)

    def test_chain_of_three_records_every_transition(self, tmp_path):
        g = _graph(tmp_path)
        for label in ("Use MySQL for the user database",
                      "Use MongoDB for the user database",
                      "Use PostgreSQL for the user database"):
            g.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED,
                       summary="benchmarked it | p99 latency halved")

        active = [n.label for n in g.query("database")]
        assert active == ["Use PostgreSQL for the user database"]
        assert len(g.get_transitions()) == 2, (
            "each change must leave a causal record; previously all three "
            "decisions collapsed into one node with zero transitions"
        )
        # History is preserved, not deleted.
        assert len(g._nodes) == 3

    def test_replacement_after_invalidate_is_recorded(self, tmp_path):
        """An INVALIDATED decision used to poison its topic slot forever:
        the replacement merged into the dead node and the status-upgrade
        rule refused to revive it, so the session ended up with NO active
        decision on that topic at all."""
        g = _graph(tmp_path)
        dead = g.add_node(NodeType.DECISION, "Use MongoDB for the user database",
                          NodeStatus.COMPLETED)
        g._nodes[dead].status = NodeStatus.INVALIDATED

        new = g.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                         NodeStatus.COMPLETED)

        assert new and new != dead
        assert [n.label for n in g.query("database")] == [
            "Use PostgreSQL for the user database"
        ]


class TestResumeExcludesDeadDecisions:
    """
    _build_critical sorted ALL decision nodes by importance with no
    status filter, so the ~100-token "must-know facts" block presented
    superseded and explicitly-invalidated decisions as current.
    """

    def test_resume_shows_current_and_warns_about_rejected(self, tmp_path):
        from tokenmizer.checkpoints.manager import CheckpointManager

        g = _graph(tmp_path)
        old = g.add_node(NodeType.DECISION, "Use MongoDB for the user database",
                         NodeStatus.COMPLETED, importance=0.95)
        g._nodes[old].status = NodeStatus.SUPERSEDED
        bad = g.add_node(NodeType.DECISION, "Store passwords in plaintext for speed",
                         NodeStatus.COMPLETED, importance=0.99)
        g._nodes[bad].status = NodeStatus.INVALIDATED
        g.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                   NodeStatus.COMPLETED, importance=0.9)

        mgr = CheckpointManager(storage_dir=str(tmp_path))
        critical = mgr.create(session_id="s", messages=[], graph=g,
                              context_pct=0.9, trigger="manual").resume_critical

        key_line = next(ln for ln in critical.splitlines() if ln.startswith("KEY DECISIONS"))
        assert "PostgreSQL" in key_line
        assert "MongoDB" not in key_line, "a superseded choice was presented as current"
        assert "plaintext" not in key_line, "an invalidated choice was presented as current"
        # Rejected decisions are still surfaced — as a warning, so the
        # model does not cheerfully re-propose them.
        assert "DO NOT REVISIT" in critical
        assert "plaintext" in critical


class TestTokenizerFailsSoft:
    """
    _get_encoding caught only ImportError, but tiktoken downloads its
    vocabulary from a CDN on first use. A network failure therefore
    propagated out of count_messages_tokens — which is on the hot path of
    every proxied request — turning a third-party outage into a 500 on
    every request, with the documented char/4 fallback unreachable.
    """

    def test_network_failure_falls_back_instead_of_raising(self, monkeypatch):
        import tiktoken

        from tokenmizer.core import tokenizer as tk

        def boom(*a, **kw):
            raise ConnectionError("could not reach openaipublic.blob.core.windows.net")

        tk._get_encoding.cache_clear()
        monkeypatch.setattr(tiktoken, "encoding_for_model", boom)
        monkeypatch.setattr(tiktoken, "get_encoding", boom)
        try:
            assert tk.count_tokens("hello world", "gpt-4o") > 0
            assert tk.count_messages_tokens(
                [{"role": "user", "content": "hello"}], "gpt-4o"
            ) > 0
        finally:
            tk._get_encoding.cache_clear()

    def test_failure_is_cached_not_retried_per_call(self, monkeypatch):
        """A failed lookup must not re-attempt the network on every
        request — that turns one outage into a full timeout per call."""
        import tiktoken

        from tokenmizer.core import tokenizer as tk

        calls = []

        def boom(*a, **kw):
            calls.append(1)
            raise ConnectionError("offline")

        tk._get_encoding.cache_clear()
        monkeypatch.setattr(tiktoken, "encoding_for_model", boom)
        monkeypatch.setattr(tiktoken, "get_encoding", boom)
        try:
            for _ in range(5):
                tk.count_tokens("hello", "gpt-4o")
            assert len(calls) == 1
        finally:
            tk._get_encoding.cache_clear()


class TestEnvOverridesYaml:
    """
    Settings.from_yaml passed the file's contents as __init__ kwargs, the
    HIGHEST-priority source in pydantic-settings — above environment
    variables. Every key present in the shipped (and Docker-COPY'd)
    tokenmizer.yaml therefore silently beat its TOKENMIZER_* variable,
    contradicting that file's own header.
    """

    def test_env_wins_for_keys_present_in_yaml(self, tmp_path, monkeypatch):
        from tokenmizer.config.settings import Settings

        cfg = tmp_path / "tokenmizer.yaml"
        cfg.write_text(
            "provider: anthropic\n"
            "state_backend: memory\n"
            "graph_checkpoint:\n"
            "  trigger_at_percent: 0.85\n"
            "  storage_dir: ./from-yaml\n"
        )
        monkeypatch.setenv("TOKENMIZER_PROVIDER", "openai")
        monkeypatch.setenv("TOKENMIZER_STATE_BACKEND", "redis")
        monkeypatch.setenv("TOKENMIZER_GRAPH_CHECKPOINT__TRIGGER_AT_PERCENT", "0.5")

        s = Settings.from_yaml(str(cfg))

        assert s.provider == "openai"
        assert s.state_backend == "redis"
        assert s.graph_checkpoint.trigger_at_percent == 0.5
        # A nested sibling the env did NOT set must still come from YAML.
        assert s.graph_checkpoint.storage_dir == "./from-yaml"

    def test_yaml_still_applies_without_env(self, tmp_path, monkeypatch):
        from tokenmizer.config.settings import Settings

        for var in ("TOKENMIZER_PROVIDER", "TOKENMIZER_STATE_BACKEND"):
            monkeypatch.delenv(var, raising=False)
        cfg = tmp_path / "tokenmizer.yaml"
        cfg.write_text("provider: gemini\nstate_backend: redis\n")

        s = Settings.from_yaml(str(cfg))
        assert s.provider == "gemini"
        assert s.state_backend == "redis"


class TestSessionOwnership:
    """
    Session-scoped routes took session_id straight from the URL with only
    a single shared deployment key for auth, so any authenticated caller
    could read or mutate any other caller's session — and session_id is
    client-chosen, so it did not even need guessing.
    """

    @pytest.fixture
    def store(self, tmp_path):
        from tokenmizer.security.ownership import OwnershipStore
        return OwnershipStore(storage_dir=str(tmp_path))

    def test_first_caller_claims_the_session(self, store):
        from tokenmizer.security.ownership import SessionAccessDenied

        store.check_access("proj", "k_alice", claim=True)
        store.check_access("proj", "k_alice", claim=True)  # idempotent

        with pytest.raises(SessionAccessDenied):
            store.check_access("proj", "k_bob", claim=True)

    def test_reads_do_not_claim_unowned_sessions(self, store):
        """A GET must not stake a claim as a side effect."""
        store.check_access("never-seen", "k_bob", claim=False)
        assert store.owner_of("never-seen") is None

    def test_distinct_keys_are_distinct_principals(self):
        from tokenmizer.security.ownership import DEV_PRINCIPAL, principal_for_key

        assert principal_for_key("alice") != principal_for_key("bob")
        assert principal_for_key("alice") == principal_for_key("alice")
        assert principal_for_key("") == DEV_PRINCIPAL
        # The raw credential must never be recoverable from the principal.
        assert "alice" not in principal_for_key("alice")

    def test_cross_principal_access_is_denied_over_http(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import tokenmizer.api.app as app_module
        from tokenmizer.security.ownership import OwnershipStore

        monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
        monkeypatch.setattr(app_module.settings, "api_key", "alice-key")
        monkeypatch.setattr(app_module.settings, "api_keys", ["bob-key"])
        app_module._graph_cache.clear()

        alice = {"Authorization": "Bearer alice-key"}
        bob = {"Authorization": "Bearer bob-key"}

        with TestClient(app_module.app) as c:
            assert c.post("/api/checkpoint?session_id=alice-proj", headers=alice).status_code == 200

            for path in ("/api/graph/alice-proj",
                         "/api/graph/alice-proj/viz",
                         "/api/resume/alice-proj",
                         "/api/graph/alice-proj/transitions"):
                # 404, not 403: a distinguishable 403 would turn these
                # endpoints into an oracle for which session names exist.
                assert c.get(path, headers=bob).status_code == 404, path

            assert c.post(
                "/api/decision/invalidate?session_id=alice-proj&decision_label=database",
                headers=bob,
            ).status_code == 404

            assert c.get("/api/graph/alice-proj", headers=alice).status_code == 200
            assert c.post("/api/checkpoint?session_id=bob-proj", headers=bob).status_code == 200

        app_module._graph_cache.clear()


class TestCacheInvalidation:
    """
    invalidate() built its key with the DEFAULT "__shared__" scope, but
    under the default share_scope="session" every entry is filed under
    the session's scope — so it matched nothing, removed nothing, and
    reported no error.
    """

    def test_invalidate_removes_session_scoped_entries(self):
        from tokenmizer.semantic_cache.cache import SemanticCache

        cache = SemanticCache(share_scope="session")
        cache.set("what is the deploy command?", "run make deploy", session_id="s1")
        assert cache.get("what is the deploy command?", session_id="s1") is not None

        assert cache.invalidate("what is the deploy command?", session_id="s1") == 1
        assert cache.get("what is the deploy command?", session_id="s1") is None

    def test_invalidate_without_session_clears_every_scope(self):
        from tokenmizer.semantic_cache.cache import SemanticCache

        cache = SemanticCache(share_scope="session")
        cache.set("shared question", "answer", session_id="s1")
        cache.set("shared question", "answer", session_id="s2")

        assert cache.invalidate("shared question") == 2
        assert cache.get("shared question", session_id="s1") is None
        assert cache.get("shared question", session_id="s2") is None


class TestAnalyticsBounds:
    """_records was unbounded and appended per request, with a second
    reference in _by_provider — the only uncapped structure in the
    codebase."""

    def test_records_are_capped(self):
        from tokenmizer.analytics.engine import AnalyticsEngine

        eng = AnalyticsEngine(max_records=50)
        for _ in range(500):
            eng.record(session_id="s", provider="anthropic", model="claude-sonnet-4-6",
                       input_tokens_original=100, input_tokens_sent=80,
                       output_tokens=20, tokens_saved=20, latency_ms=1.0,
                       cache_hit=False, layer_savings={"compression": 20})

        assert len(eng._records) <= 50
        assert sum(len(v) for v in eng._by_provider.values()) <= 50, (
            "the per-provider index must be trimmed too, or it is the leak"
        )
        summary = eng.summary()
        assert summary["total_requests"] == 500, "lifetime count must not shrink"
        assert summary["by_provider"]["anthropic"] == 500

    def test_output_tokens_cost_more_than_input(self):
        """A single blended rate understated real spend several-fold."""
        from tokenmizer.analytics.engine import _cost

        assert _cost(0, 1000, "anthropic") > _cost(1000, 0, "anthropic")


class TestSettingsAreActuallyRead:
    """Several documented settings were never read by any code, so
    changing them did nothing and said nothing."""

    def test_min_tokens_to_compress_is_honoured(self):
        from tokenmizer.compression.engine import CompressionPipeline

        assert CompressionPipeline(min_tokens_to_compress=999).min_tokens_to_compress == 999

    def test_max_resume_tokens_is_honoured(self, tmp_path):
        from tokenmizer.checkpoints.manager import CheckpointManager

        assert CheckpointManager(storage_dir=str(tmp_path),
                                 max_resume_tokens=42)._max_resume_tokens == 42

    def test_routing_enabled_warns_that_it_does_nothing(self, tmp_path, monkeypatch, caplog):
        import logging

        import tokenmizer.config.settings as settings_module

        cfg = tmp_path / "tokenmizer.yaml"
        cfg.write_text("routing:\n  enabled: true\n")
        monkeypatch.setenv("TOKENMIZER_CONFIG", str(cfg))
        monkeypatch.setattr(settings_module, "_settings", None)
        monkeypatch.delenv("TOKENMIZER_ENV", raising=False)

        with caplog.at_level(logging.WARNING):
            settings_module.get_settings()
        monkeypatch.setattr(settings_module, "_settings", None)

        assert any("NOT IMPLEMENTED" in r.message for r in caplog.records), (
            "a setting that does nothing must say so"
        )


class TestStreamDoesNotCacheTruncatedResponses:
    """On a mid-stream provider error the generator fell through to the
    post-stream bookkeeping and cached whatever partial text had arrived,
    serving that truncated answer to every future matching prompt."""

    def test_partial_stream_is_not_cached(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        import tokenmizer.api.app as app_module
        from tokenmizer.providers.providers import ProviderError
        from tokenmizer.security.ownership import OwnershipStore

        monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        monkeypatch.setattr(app_module.settings.cache, "enabled", True)
        app_module._cache.clear()

        class HalfBrokenProvider:
            async def chat_stream(self, messages, model="", max_tokens=0, **kw):
                yield "The answer is "
                raise ProviderError("anthropic", "overloaded", "upstream died mid-stream")

        monkeypatch.setattr(app_module, "_get_provider", lambda: HalfBrokenProvider())

        with TestClient(app_module.app) as c:
            r = c.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "what is the answer?"}],
                "stream": True, "session_id": "s-stream",
            })
            assert r.status_code == 200
            assert "provider_error" in r.text

        assert app_module._cache.get("what is the answer?", session_id="s-stream") is None, (
            "a truncated response was cached and will be served to future callers"
        )
        app_module._cache.clear()
