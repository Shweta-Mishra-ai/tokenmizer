"""
Regression tests for TM-13: /api/decision/invalidate must not let a
short/empty label match every (or most) active decisions in a session.

Bug: matching was `label_lower in node.label.lower()` — a raw substring
check — with no minimum length on decision_label. The empty string is a
substring of everything, so `decision_label=""` invalidated every active
decision in one call. Short labels caused a milder version of the same
problem (a single common letter/word matching unrelated decisions).

Fix: reject decision_label below a minimum length (matching the noise-
pattern convention already used in graph_memory/validator.py, which
treats <=3 chars as noise), and match on a WORD-BOUNDARY substring
instead of a raw substring — so "sql" no longer spuriously matches
"SQLAlchemy" or "MySQL" as a side effect of literal containment. An
explicit node_id parameter is also added for precise, unambiguous
targeting when the caller has one (e.g. from a prior /api/graph/{id}/viz
or /transitions response).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    with TestClient(app) as c:
        yield c


def _seed(session_id, tmp_path):
    g = GraphMemory(session_id, storage_dir=str(tmp_path))
    g.add_node(NodeType.DECISION, "Use PostgreSQL for the primary datastore",
              NodeStatus.COMPLETED, summary="rationale here", importance=0.9)
    g.add_node(NodeType.DECISION, "Use Stripe for payment processing",
              NodeStatus.COMPLETED, summary="rationale here", importance=0.9)
    g.add_node(NodeType.DECISION, "Use Playwright for end to end testing",
              NodeStatus.COMPLETED, summary="rationale here", importance=0.9)
    app_module._graph_cache[session_id] = g
    return g


class TestEmptyOrShortLabelRejected:

    def test_empty_label_does_not_invalidate_everything(self, client, tmp_path):
        g = _seed("invalidate-empty-test", tmp_path)
        r = client.post("/api/decision/invalidate",
                        params={"session_id": "invalidate-empty-test", "decision_label": ""})
        assert r.status_code in (400, 422), (
            f"empty decision_label should be rejected outright, got {r.status_code}: {r.text}"
        )
        # And no decision should have been touched.
        statuses = {n.status for n in g._nodes.values() if n.type == NodeType.DECISION}
        assert statuses == {NodeStatus.COMPLETED}

    def test_very_short_label_is_rejected(self, client, tmp_path):
        _seed("invalidate-short-test", tmp_path)
        r = client.post("/api/decision/invalidate",
                        params={"session_id": "invalidate-short-test", "decision_label": "e"})
        assert r.status_code in (400, 422), (
            f"a 1-character label should be rejected as too ambiguous, got "
            f"{r.status_code}: {r.text}"
        )


class TestWordBoundaryMatchingAvoidsFalsePositives:

    def test_short_substring_does_not_match_unrelated_decision(self, client, tmp_path):
        """'sql' as a raw substring would match both 'PostgreSQL' AND any
        future 'MySQL'/'SQLAlchemy' decision — word-boundary matching
        must not treat a fragment inside a longer word as a hit."""
        g = _seed("invalidate-boundary-test", tmp_path)
        g.add_node(NodeType.DECISION, "Use SQLAlchemy as the ORM layer",
                  NodeStatus.COMPLETED, summary="rationale", importance=0.9)

        r = client.post("/api/decision/invalidate",
                        params={"session_id": "invalidate-boundary-test",
                                "decision_label": "postgresql"})
        assert r.status_code == 200, r.text
        affected = {n["node_id"] for n in r.json()["affected_nodes"]}
        # Only the PostgreSQL decision should be affected — NOT the
        # SQLAlchemy one, even though "sql" is literally a substring of
        # "SQLAlchemy" too.
        sqlalchemy_id = next(
            nid for nid, n in g._nodes.items() if "SQLAlchemy" in n.label
        )
        assert sqlalchemy_id not in affected

    def test_legitimate_partial_match_still_works(self, client, tmp_path):
        _seed("invalidate-legit-test", tmp_path)
        r = client.post("/api/decision/invalidate",
                        params={"session_id": "invalidate-legit-test",
                                "decision_label": "stripe"})
        assert r.status_code == 200, r.text
        assert len(r.json()["affected_nodes"]) == 1


class TestExplicitNodeIdTargeting:

    def test_node_id_param_targets_precisely(self, client, tmp_path):
        g = _seed("invalidate-nodeid-test", tmp_path)
        target_id = next(
            nid for nid, n in g._nodes.items() if "Stripe" in n.label
        )
        r = client.post("/api/decision/invalidate",
                        params={"session_id": "invalidate-nodeid-test",
                                "node_id": target_id})
        assert r.status_code == 200, r.text
        assert g._nodes[target_id].status == NodeStatus.INVALIDATED
        # Nothing else should be touched.
        others_invalidated = [
            n for nid, n in g._nodes.items()
            if nid != target_id and n.status == NodeStatus.INVALIDATED
        ]
        assert others_invalidated == []
