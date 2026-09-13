"""
Cross-session recall: rank a principal's other sessions' nodes alongside
the current session's, each result tagged with the session_id it came
from. Off by default (Settings.graph_checkpoint.cross_session_recall) —
see graph_memory/cross_session.py and config/settings.py.
"""
from __future__ import annotations

from tokenmizer.graph_memory.cross_session import (
    MAX_OTHER_SESSIONS,
    query_across_sessions,
)
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import NodeStatus, NodeType


def _graph(tmp_path, session_id: str) -> GraphMemory:
    return GraphMemory(session_id, storage_dir=str(tmp_path))


def test_setting_defaults_to_off():
    from tokenmizer.config.settings import Settings

    assert Settings().graph_checkpoint.cross_session_recall is False


def test_merges_and_tags_nodes_from_other_sessions(tmp_path):
    current = _graph(tmp_path, "checkout-service")
    current.add_node(NodeType.DECISION, "Use Redis for the session cache",
                     status=NodeStatus.COMPLETED, importance=0.9)

    other = _graph(tmp_path, "billing-service")
    other.add_node(NodeType.DECISION, "Use Redis for idempotency keys",
                   status=NodeStatus.COMPLETED, importance=0.9)
    other._persist(force=True)

    results = query_across_sessions(
        current, ["billing-service"], "redis", top_k=8,
        load_graph=lambda sid: _graph(tmp_path, sid),
    )

    sessions_seen = {sid for _, sid in results}
    assert sessions_seen == {"checkout-service", "billing-service"}
    labels_by_session = {sid: node.label for node, sid in results}
    assert "Redis for the session cache" in labels_by_session["checkout-service"]
    assert "Redis for idempotency keys" in labels_by_session["billing-service"]


def test_results_are_ranked_by_score_across_sessions(tmp_path):
    current = _graph(tmp_path, "s-current")
    current.add_node(NodeType.TASK, "unrelated task about deployment",
                     importance=0.1)

    other = _graph(tmp_path, "s-other")
    other.add_node(NodeType.DECISION, "Use PostgreSQL for orders",
                   status=NodeStatus.COMPLETED, importance=0.9)
    other._persist(force=True)

    results = query_across_sessions(
        current, ["s-other"], "postgresql orders", top_k=8,
        load_graph=lambda sid: _graph(tmp_path, sid),
    )

    assert results, "expected the other session's matching node to surface"
    top_node, top_session = results[0]
    assert top_session == "s-other"
    assert "PostgreSQL" in top_node.label


def test_caps_at_max_other_sessions(tmp_path):
    current = _graph(tmp_path, "s-current")
    current.add_node(NodeType.TASK, "some task", importance=0.5)

    other_ids = [f"s-{i}" for i in range(MAX_OTHER_SESSIONS + 3)]
    for sid in other_ids:
        g = _graph(tmp_path, sid)
        g.add_node(NodeType.DECISION, "Use PostgreSQL for orders",
                  status=NodeStatus.COMPLETED, importance=0.9)
        g._persist(force=True)

    loaded = []

    def load_graph(sid):
        loaded.append(sid)
        return _graph(tmp_path, sid)

    query_across_sessions(current, other_ids, "postgresql", top_k=50,
                          load_graph=load_graph)

    assert len(loaded) == MAX_OTHER_SESSIONS


def test_a_failed_load_is_skipped_not_fatal(tmp_path):
    current = _graph(tmp_path, "s-current")
    current.add_node(NodeType.TASK, "some task", importance=0.5)

    good = _graph(tmp_path, "s-good")
    good.add_node(NodeType.DECISION, "Use PostgreSQL for orders",
                 status=NodeStatus.COMPLETED, importance=0.9)
    good._persist(force=True)

    def load_graph(sid):
        if sid == "s-broken":
            raise RuntimeError("simulated load failure")
        return _graph(tmp_path, sid)

    results = query_across_sessions(
        current, ["s-broken", "s-good"], "postgresql orders", top_k=8,
        load_graph=load_graph,
    )

    assert any(sid == "s-good" for _, sid in results)
