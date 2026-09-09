"""
GraphMemory.query() — summary search and the optional semantic blend.

query() is the ranker on the request path: api/app.py calls it every turn to
choose which nodes get injected into the prompt, and it is what
SmartMessageWindow leans on once older turns have been dropped. Two gaps:

1. It scored `node.label` only. A node's `summary` — where a decision's
   rationale and an error's detail live — was unreachable by any query, even
   with the exact phrase sitting in that field.

2. It was the only retrieval path in the package still doing pure token
   overlap. EmbeddingEngine was already here and already used by
   semantic_cache, reasoning.py and decision_tracker.py; the one ranker on
   the hot path was not wired to it.

These tests drive the blend with a FAKE engine returning vectors chosen by
the test. That is deliberate: conftest stubs EmbeddingEngine._load for the
whole suite precisely so no test downloads a model from huggingface.co, and
what is under test here is the blending logic — the rank cutoff, additivity,
cache invalidation, graceful degradation — not the model's judgement. How
well the real model ranks real questions is measured by
`python -m benchmarks.graph_retrieval.query_eval`, which is the right place
for it (recall@6 85% -> 92% with the blend on).
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import NodeStatus, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("semantic-test", storage_dir=str(tmp_path))


def _labels(nodes):
    return [n.label for n in nodes]


# ── Summary is searchable ────────────────────────────────────────────────────

def test_a_node_is_findable_by_words_only_in_its_summary(graph):
    nid = graph.add_node(NodeType.DECISION, "Use PgBouncer",
                         NodeStatus.IN_PROGRESS, importance=0.9)
    graph._nodes[nid].summary = "Reason: connection pooling collapsed under load"

    found = _labels(graph.query("connection pooling", top_k=5))
    assert "Use PgBouncer" in found, (
        "summary text is unreachable by the ranker, so the rationale behind "
        f"every decision is invisible to retrieval. got {found}"
    )


def test_label_matching_still_works_without_a_summary(graph):
    graph.add_node(NodeType.DECISION, "Use PostgreSQL for orders",
                   NodeStatus.IN_PROGRESS, importance=0.9)
    assert _labels(graph.query("postgresql orders", top_k=5))


# ── A controllable stand-in for the embedding model ──────────────────────────

class _FakeEngine:
    """Returns a vector chosen per text, so the test decides similarity.

    Vectors are 2-D unit-ish vectors: `near` points the same way as the
    query, `far` points away from it. Cosine is computed by the real
    EmbeddingEngine.cosine, so only the model is faked, not the maths.
    """

    available = True

    def __init__(self, vectors: dict[str, list[float]], default=(0.0, 1.0)):
        self._vectors = vectors
        self._default = list(default)
        self.batch_calls = 0

    def _vec(self, text: str):
        for key, vector in self._vectors.items():
            if key.lower() in text.lower():
                return list(vector)
        return list(self._default)

    def embed(self, text: str):
        return self._vec(text)

    def embed_batch(self, texts: list[str]):
        self.batch_calls += 1
        return [self._vec(t) for t in texts]


@pytest.fixture
def fake_engine(monkeypatch):
    """Install a fake engine and hand it back so tests can assert on it."""
    from tokenmizer.semantic_cache.cache import EmbeddingEngine

    holder = {}

    def _cosine(a, b) -> float:
        # The real cosine uses numpy, which ships in the `cache` extra and
        # is not installed in CI. In production this path is unreachable
        # without it (engine.available is False), but the fake bypasses
        # `available`, so it has to bypass numpy too.
        if a is None or b is None:
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    def install(vectors, default=(0.0, 1.0)):
        engine = _FakeEngine(vectors, default)
        monkeypatch.setattr(EmbeddingEngine, "get", staticmethod(lambda: engine))
        monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(_cosine))
        holder["engine"] = engine
        return engine

    return install


# ── The semantic blend ───────────────────────────────────────────────────────

def test_semantic_surfaces_a_node_sharing_no_words_with_the_question(
        tmp_path, fake_engine):
    """The case token overlap cannot serve, and the reason Mem0, Zep and
    Graphiti retrieve by embedding."""
    fake_engine({
        "storing orders in": (1.0, 0.0),          # the question
        "PostgreSQL":        (1.0, 0.0),          # the answer, no shared words
    })
    graph = GraphMemory("sem-on", storage_dir=str(tmp_path),
                        semantic_retrieval=True)
    for label in ("Use PostgreSQL for order storage",
                  "Rewrite the CSS grid in the header",
                  "Bump the Node version in CI"):
        graph.add_node(NodeType.DECISION, label, NodeStatus.IN_PROGRESS,
                       importance=0.8)

    found = _labels(graph.query("what are we storing orders in", top_k=3))
    assert "Use PostgreSQL for order storage" in found, found


def test_semantic_is_additive_and_never_drops_a_keyword_result(
        tmp_path, fake_engine):
    """Turning the blend on must not remove anything the keyword ranker would
    have returned — it only ever adds score to what is already there."""
    labels = ["Use PostgreSQL for order storage", "Redis for refresh tokens",
              "bcrypt for password hashing", "Rewrite the CSS grid",
              "Bump the Node version in CI", "Adopt Ruff for linting"]
    question = "which database stores orders"

    def build(semantic):
        g = GraphMemory(f"add-{semantic}", storage_dir=str(tmp_path / str(semantic)),
                        semantic_retrieval=semantic)
        for i, label in enumerate(labels):
            g.add_node(NodeType.DECISION, label, NodeStatus.IN_PROGRESS,
                       importance=0.5 + i * 0.05)
        return g

    keyword = set(_labels(build(False).query(question, top_k=6)))
    fake_engine({"Ruff": (1.0, 0.0), "database stores orders": (1.0, 0.0)})
    blended = set(_labels(build(True).query(question, top_k=6)))

    assert keyword <= blended, f"semantic dropped {keyword - blended}"


def test_semantic_rescues_a_bounded_number_of_nodes(tmp_path, fake_engine):
    """A rank cutoff rather than an absolute cosine threshold — but still a
    cutoff. Every node scoring slightly would be as useless as none.

    Asserted as the DIFFERENCE against the keyword-only ranking, not as an
    absolute count, because of a separate pre-existing behaviour this test
    ran into: query()'s `score > 0.05` floor is cleared by node.importance
    alone (a node with importance 0.4 scores ~0.12 before any keyword
    overlap), so the keyword pass already admits every node for an unrelated
    query and top_k is what really bounds the result. The comment there
    calling 0.05 a "minimum threshold — don't return completely unrelated
    nodes" does not describe what it does. Left alone deliberately: changing
    that floor re-tunes every existing query and belongs behind its own
    measurement, not smuggled in with this change.
    """
    graph_off = GraphMemory("bound-off", storage_dir=str(tmp_path / "off"))
    graph_on = GraphMemory("bound-on", storage_dir=str(tmp_path / "on"),
                           semantic_retrieval=True)
    for g in (graph_off, graph_on):
        for i in range(30):
            g.add_node(NodeType.TASK, f"unrelated task number {i}",
                       NodeStatus.IN_PROGRESS, importance=0.4)

    baseline = len(graph_off.query("quantum chromodynamics", top_k=50))
    fake_engine({}, default=(1.0, 0.0))  # everything maximally "similar"
    with_blend = len(graph_on.query("quantum chromodynamics", top_k=50))

    assert with_blend - baseline <= GraphMemory._SEMANTIC_RESCUE_LIMIT, (
        f"semantic added {with_blend - baseline} nodes, "
        f"limit is {GraphMemory._SEMANTIC_RESCUE_LIMIT}"
    )


def test_blend_is_a_noop_without_an_embedding_engine(tmp_path, monkeypatch):
    """Same graceful-degradation contract semantic_cache and reasoning.py
    already have: a host without sentence-transformers, or one that cannot
    reach huggingface.co, behaves exactly as before rather than failing on
    the request path."""
    from tokenmizer.semantic_cache.cache import EmbeddingEngine

    class _Unavailable:
        available = False

    monkeypatch.setattr(EmbeddingEngine, "get", staticmethod(lambda: _Unavailable()))

    # One graph, queried with the flag off and then on. Two separately built
    # graphs are not comparable by order: their nodes get microsecond-apart
    # created_at values, and the recency term in the score breaks ties
    # differently depending on clock granularity — it did, on Windows CI.
    graph = GraphMemory("degrade", storage_dir=str(tmp_path))
    for label in ("Use PostgreSQL", "Redis for tokens", "bcrypt hashing"):
        graph.add_node(NodeType.DECISION, label, NodeStatus.IN_PROGRESS,
                       importance=0.8)

    graph.semantic_retrieval = False
    without = _labels(graph.query("database choice", top_k=5))
    graph.semantic_retrieval = True
    with_flag = _labels(graph.query("database choice", top_k=5))

    assert with_flag == without


def test_node_embeddings_are_cached_across_queries(tmp_path, fake_engine):
    """Embedding every node on every turn would put a model forward pass on
    the request path for text that has not changed."""
    engine = fake_engine({})
    graph = GraphMemory("cache", storage_dir=str(tmp_path),
                        semantic_retrieval=True)
    graph.add_node(NodeType.DECISION, "Use PostgreSQL", NodeStatus.IN_PROGRESS,
                   importance=0.9)

    graph.query("database", top_k=3)
    after_first = engine.batch_calls
    graph.query("something else entirely", top_k=3)

    assert after_first == 1
    assert engine.batch_calls == 1, "re-embedded unchanged nodes"


def test_edited_node_is_re_embedded(tmp_path, fake_engine):
    """A stale cache would make an edited node unfindable by its new
    wording — worse than no cache."""
    engine = fake_engine({})
    graph = GraphMemory("cache-edit", storage_dir=str(tmp_path),
                        semantic_retrieval=True)
    nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL",
                         NodeStatus.IN_PROGRESS, importance=0.9)

    graph.query("database", top_k=3)
    cached_text = graph._embedding_cache[nid][0]

    graph._nodes[nid].summary = "Reason: needed row-level security"
    graph.query("row level security", top_k=3)

    assert graph._embedding_cache[nid][0] != cached_text, "cache went stale"
    assert engine.batch_calls == 2
