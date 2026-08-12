"""Unit tests — ontology and graph reasoning (storage → reasoning)."""
import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
from tokenmizer.graph_memory.ontology import (
    EDGE_TYPES,
    NODE_TYPES,
    STATUS_TRANSITIONS,
    is_valid_transition,
    ontology_dict,
)
from tokenmizer.graph_memory.reasoning import (
    consistency_check,
    decision_history,
    impact,
    summarize_reasoning,
    why,
)
from tokenmizer.graph_memory.types import EdgeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("t-reasoning", storage_dir=str(tmp_path))


@pytest.fixture
def story_graph(graph):
    """A session with a real decision story: Postgres → SQLite, React → Next."""
    graph.add_node(NodeType.GOAL, "Build FastAPI auth service",
                   NodeStatus.IN_PROGRESS)
    graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                   NodeStatus.COMPLETED, summary="need concurrent writes")
    graph.add_node(NodeType.DECISION, "Switch from PostgreSQL to SQLite",
                   NodeStatus.COMPLETED,
                   summary="no budget for managed DB | user said keep it free")
    graph.add_node(NodeType.DECISION, "Use React for the frontend",
                   NodeStatus.COMPLETED)
    graph.add_node(NodeType.DECISION, "Switch to Next.js for the frontend",
                   NodeStatus.COMPLETED, summary="better SEO")
    graph.add_node(NodeType.FILE, "api/auth.py", NodeStatus.COMPLETED)
    return graph


# ── Ontology ─────────────────────────────────────────────────────────────────

class TestOntology:

    def test_every_node_type_documented(self):
        from tokenmizer.graph_memory.types import NodeType as NT
        # CONCEPT..SCHEMA — every enum member must carry semantics
        for t in NT:
            assert t in NODE_TYPES, f"NodeType.{t.name} missing from ontology"

    def test_every_edge_type_documented(self):
        for e in EdgeType:
            assert e in EDGE_TYPES, f"EdgeType.{e.name} missing from ontology"
            assert EDGE_TYPES[e]["semantics"]

    def test_lifecycle_paths(self):
        assert is_valid_transition(NodeStatus.COMPLETED, NodeStatus.SUPERSEDED)
        assert is_valid_transition(NodeStatus.SUPERSEDED, NodeStatus.ARCHIVED)
        assert is_valid_transition(NodeStatus.PENDING, NodeStatus.IN_PROGRESS)
        # Terminal states stay terminal
        assert not is_valid_transition(NodeStatus.ARCHIVED, NodeStatus.COMPLETED)
        assert not is_valid_transition(NodeStatus.INVALIDATED, NodeStatus.COMPLETED)
        # A superseded decision cannot silently become active again
        assert not is_valid_transition(NodeStatus.SUPERSEDED, NodeStatus.COMPLETED)
        # Self-loop is always fine
        assert is_valid_transition(NodeStatus.COMPLETED, NodeStatus.COMPLETED)

    def test_ontology_dict_is_json_ready(self):
        import json
        d = ontology_dict()
        json.dumps(d)  # must not raise
        assert d["version"]
        assert "decision" in d["node_types"]
        assert "supersedes" in d["edge_types"]
        assert set(d["status_transitions"]["completed"]) >= {"superseded"}

    def test_transitions_reference_known_statuses(self):
        for frm, tos in STATUS_TRANSITIONS.items():
            assert isinstance(frm, NodeStatus)
            for to in tos:
                assert isinstance(to, NodeStatus)


# ── why(): causal chains ─────────────────────────────────────────────────────

class TestWhy:

    def test_why_traces_supersession_chain(self, story_graph):
        result = why(story_graph, "postgres")
        assert result["matches"], "postgres decisions must match"
        assert len(result["chain"]) == 1
        hop = result["chain"][0]
        assert "PostgreSQL" in hop["from_label"]
        assert "SQLite" in hop["to_label"]
        assert result["current"] is not None
        assert "SQLite" in result["current"]["label"]

    def test_why_finds_current_from_old_label(self, story_graph):
        """Asking about the OLD choice must surface the NEW one."""
        result = why(story_graph, "react")
        assert result["current"] is not None
        assert "Next.js" in result["current"]["label"]

    def test_why_unchanged_decision_has_empty_chain(self, story_graph):
        story_graph.add_node(NodeType.DECISION, "Use bcrypt for password hashing",
                             NodeStatus.COMPLETED)
        result = why(story_graph, "bcrypt")
        assert result["chain"] == []
        assert result["current"] is not None
        assert "bcrypt" in result["current"]["label"]

    def test_why_no_match(self, story_graph):
        result = why(story_graph, "kubernetes")
        assert result["matches"] == []
        assert result["current"] is None

    def test_why_empty_query(self, story_graph):
        assert why(story_graph, "")["matches"] == []


# ── Semantic recall ────────────────────────────────────────────────────────
#
# why()/impact() widen their substring match with embedding similarity via
# _semantic_matches (tokenmizer/graph_memory/reasoning.py), reusing the same
# EmbeddingEngine the semantic cache uses. Every test in this suite runs
# with EmbeddingEngine._load stubbed to a no-op (see conftest.py), so
# .available is False and these two paths behave IDENTICALLY without a
# model — that's covered here explicitly. The "does it actually widen
# recall" behavior is tested with a fake model (deterministic, unit
# vectors, no network) rather than the real one, matching how
# test_cache.py already tests this engine.

def _install_fake_embedder(monkeypatch, vectors: dict):
    """Point the shared EmbeddingEngine singleton at fixed vectors for one
    test, and swap `.cosine` for a pure-Python dot product so this needs
    no numpy (the real `.cosine`/`.encode` require it, same as the real
    sentence-transformers model would, but the fake model's plain-list
    vectors don't need the array machinery). monkeypatch restores the
    real (untrained, unavailable) state afterward — this can't leak into
    other tests."""
    from tokenmizer.semantic_cache.cache import EmbeddingEngine

    class _FakeModel:
        def encode(self, text_or_texts, normalize_embeddings=True):
            if isinstance(text_or_texts, str):
                return vectors[text_or_texts]
            return [vectors[t] for t in text_or_texts]

    def _dot(a, b) -> float:
        if a is None or b is None:
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    engine = EmbeddingEngine.get()
    monkeypatch.setattr(engine, "_model", _FakeModel())
    monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(_dot))
    return engine


class TestSemanticRecall:

    def test_why_finds_a_paraphrase_with_no_shared_words(self, graph, monkeypatch):
        graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                       NodeStatus.COMPLETED, summary="")
        graph.add_node(NodeType.DECISION, "Switch UI framework to Svelte",
                       NodeStatus.COMPLETED, summary="")
        _install_fake_embedder(monkeypatch, {
            "database choice": [1.0, 0.0],
            "Use PostgreSQL for storage": [0.8, 0.6],   # cos = 0.8 — close
            "Switch UI framework to Svelte": [0.0, 1.0],  # cos = 0.0 — unrelated
        })

        result = why(graph, "database choice")

        labels = {m["label"] for m in result["matches"]}
        assert "Use PostgreSQL for storage" in labels, (
            "shares zero words with the query — only semantic recall finds it"
        )
        assert "Switch UI framework to Svelte" not in labels

    def test_semantic_recall_is_additive_not_a_replacement(self, graph, monkeypatch):
        """A substring match must survive even if its embedding similarity
        would fall below the semantic threshold — semantic recall only
        ever ADDS candidates, never removes the lexical floor."""
        graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                       NodeStatus.COMPLETED, summary="")
        _install_fake_embedder(monkeypatch, {
            "postgresql": [1.0, 0.0],
            "Use PostgreSQL for storage": [0.0, 1.0],  # cos = 0.0
        })

        result = why(graph, "postgresql")

        assert any(m["label"] == "Use PostgreSQL for storage" for m in result["matches"])

    def test_no_semantic_widening_without_an_embedding_model(self, story_graph):
        """Every other test in this suite runs exactly this way (see the
        module comment above) — asserted explicitly here so a future
        change to the default stub is caught."""
        from tokenmizer.semantic_cache.cache import EmbeddingEngine
        assert EmbeddingEngine.get().available is False
        result = why(story_graph, "kubernetes")
        assert result["matches"] == []

    def test_impact_is_also_widened(self, graph, monkeypatch):
        dec_id = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                                NodeStatus.COMPLETED, summary="")
        file_id = graph.add_node(NodeType.FILE, "db/schema.sql", NodeStatus.COMPLETED)
        graph.add_edge(dec_id, file_id, EdgeType.IMPLEMENTS)
        _install_fake_embedder(monkeypatch, {
            "database choice": [1.0, 0.0],
            "Use PostgreSQL for storage": [0.9, 0.436],
            "db/schema.sql": [0.0, 1.0],  # unrelated by embedding — reached via the edge instead
        })

        result = impact(graph, "database choice")

        assert any(m["label"] == "Use PostgreSQL for storage" for m in result["matches"])
        assert any(c["target"]["label"] == "db/schema.sql" for c in result["connections"])


# ── impact / history / consistency ───────────────────────────────────────────

class TestReasoningViews:

    def test_impact_returns_typed_connections(self, story_graph):
        result = impact(story_graph, "auth")
        assert result["matches"]
        for c in result["connections"]:
            assert c["relation"]
            assert c["source"]["id"] and c["target"]["id"]

    def test_decision_history_groups_by_topic(self, story_graph):
        hist = decision_history(story_graph)
        assert "database" in hist
        db = hist["database"]
        assert len(db) == 2
        # Oldest first; superseded entry carries its supersession time
        assert "PostgreSQL" in db[0]["label"]
        assert db[0]["superseded_at"] is not None
        assert db[1]["superseded_at"] is None

    def test_consistency_clean_graph(self, story_graph):
        assert consistency_check(story_graph) == []

    def test_consistency_flags_active_contradiction(self, graph):
        """Two active same-topic decisions (forced past the tracker) must
        be reported — this is the reasoning net under the classifier."""
        id1 = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                             NodeStatus.COMPLETED)
        id2 = graph.add_node(NodeType.DECISION, "Switch to MySQL for storage",
                             NodeStatus.COMPLETED)
        # Force the superseded one back to COMPLETED, simulating a tracker miss
        graph._nodes[id1].status = NodeStatus.COMPLETED
        issues = consistency_check(graph)
        kinds = [i["kind"] for i in issues]
        assert "active_contradiction" in kinds
        flagged = next(i for i in issues if i["kind"] == "active_contradiction")
        assert set(flagged["ids"]) == {id1, id2}

    def test_consistency_flags_missing_transition(self, graph):
        nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                             NodeStatus.COMPLETED)
        graph._nodes[nid].status = NodeStatus.SUPERSEDED  # no transition recorded
        issues = consistency_check(graph)
        assert any(i["kind"] == "missing_transition" for i in issues)

    def test_summarize_reasoning_shape(self, story_graph):
        s = summarize_reasoning(story_graph)
        assert s["session_id"] == "t-reasoning"
        active_labels = [d["label"] for d in s["active_decisions"]]
        assert any("SQLite" in label for label in active_labels)
        assert not any("PostgreSQL for storage" in label for label in active_labels)
        assert len(s["recent_changes"]) == 2  # both supersessions are recent
        assert s["consistency"] == []


# ── API endpoints ────────────────────────────────────────────────────────────

class TestReasoningEndpoints:

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from tokenmizer.api import app as app_module
        from tokenmizer.api.app import app
        monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
        with TestClient(app) as c:
            yield c

    def _seed(self, session_id, tmp_path):
        from tokenmizer.api import app as app_module
        g = GraphMemory(session_id, storage_dir=str(tmp_path))
        g.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                   NodeStatus.COMPLETED)
        g.add_node(NodeType.DECISION, "Switch from PostgreSQL to SQLite",
                   NodeStatus.COMPLETED, summary="cost")
        app_module._graph_cache[session_id] = g
        return g

    def test_ontology_endpoint(self, client):
        r = client.get("/api/ontology")
        assert r.status_code == 200
        body = r.json()
        assert "decision" in body["node_types"]
        assert "supersedes" in body["edge_types"]

    def test_why_endpoint(self, client, tmp_path):
        self._seed("reason-api-test", tmp_path)
        r = client.get("/api/graph/reason-api-test/why", params={"q": "postgres"})
        assert r.status_code == 200
        body = r.json()
        assert body["chain"], "supersession chain must be present"
        assert "SQLite" in body["current"]["label"]

    def test_reasoning_endpoint(self, client, tmp_path):
        self._seed("reason-view-test", tmp_path)
        r = client.get("/api/graph/reason-view-test/reasoning")
        assert r.status_code == 200
        body = r.json()
        assert body["consistency"] == []
        assert body["history_by_topic"]["database"]


# ── MCP tool ─────────────────────────────────────────────────────────────────

class TestWhyDecisionTool:

    def test_missing_args_is_error(self):
        from tokenmizer.mcp import server as mcp
        text, is_error = mcp.handle_tool_call("why_decision", {"session_id": "x"})
        assert is_error is True
        assert "query" in text

    def test_tool_listed(self):
        from tokenmizer.mcp import server as mcp
        names = {t["name"] for t in mcp.TOOLS}
        assert "why_decision" in names
        tool = next(t for t in mcp.TOOLS if t["name"] == "why_decision")
        assert set(tool["inputSchema"]["required"]) == {"session_id", "query"}
