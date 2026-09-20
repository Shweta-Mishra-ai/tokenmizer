"""
The graph must say what happened to an error, and link what belongs
together.

On the eval corpus before this: a "Fixed: 422 error — ..." turn produced a
completed task AND an ERROR node still marked FAILED, so every resume
carried a fixed bug as open; a route list was one task label cut at 80
chars; dependencies and decisions floated as unconnected dots (11 of 25
nodes unclustered on the fastapi_auth session). FIXES, BLOCKS and
DEPENDS_ON existed in the ontology and no code ever created them.
"""
from __future__ import annotations

from tokenmizer.graph_memory.graph import EdgeType, GraphMemory, NodeStatus, NodeType
from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
from tokenmizer.graph_memory.patterns import _clip


def _graph(tmp_path, messages):
    g = GraphMemory("rel", storage_dir=str(tmp_path))
    g.extract_from_messages(messages, incremental=False)
    return g


def _edges(g, kind):
    return [(g._nodes[e.source_id].label, g._nodes[e.target_id].label)
            for e in g._edges if e.type == kind]


class TestResolvedErrors:

    def test_fixed_prefix_marks_the_error_resolved(self):
        data = get_hybrid_extractor().heuristic_extract([
            {"role": "user", "content": "Login keeps returning 422"},
            {"role": "assistant", "content":
             "Fixed: 422 error — missing email validation in LoginRequest model."},
        ])
        assert any("422" in e for e in data.errors)
        assert any("422" in e for e in data.resolved_errors)

    def test_resolved_error_is_completed_with_a_fixes_edge(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "user", "content": "Login keeps returning 422"},
            {"role": "assistant", "content":
             "Fixed: 422 error — missing email validation in LoginRequest model."},
        ])
        errors = [n for n in g._nodes.values() if n.type == NodeType.ERROR]
        assert errors and all(n.status == NodeStatus.COMPLETED for n in errors), (
            [(n.label, n.status) for n in errors])
        assert _edges(g, EdgeType.FIXES), "the fixing task must FIXES the error"

    def test_unfixed_error_stays_open(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "The build fails with exit code 1 on the Windows runner."},
        ])
        errors = [n for n in g._nodes.values() if n.type == NodeType.ERROR]
        assert errors and errors[0].status == NodeStatus.FAILED

    def test_open_error_blocks_the_task_about_it(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Working on: refresh token rotation in api/auth.py. "
             "Getting a 500 from refresh token rotation when the Redis key expires."},
        ])
        blocks = _edges(g, EdgeType.BLOCKS)
        assert blocks, [(n.type.value, n.label) for n in g._nodes.values()]
        assert "refresh token rotation" in blocks[0][1].lower()


class TestRelations:

    def test_dependency_is_linked_from_the_decision_that_names_it(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Decided: Redis for refresh token storage. Adding: redis to requirements.txt"},
        ])
        deps = _edges(g, EdgeType.DEPENDS_ON)
        assert any(t.lower() == "redis" for _, t in deps), deps

    def test_decision_naming_a_file_is_linked_to_it(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Decided: keep the rate limiter in config.py. Updated config.py."},
        ])
        rel = _edges(g, EdgeType.RELATED_TO)
        assert any(s.lower().startswith("keep the rate") and t == "config.py" for s, t in rel), rel

    def test_route_list_becomes_one_task_per_route_and_implements_the_endpoint(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Implemented: POST /api/auth/register, POST /api/auth/login (returns JWT), "
             "POST /api/auth/logout. Files updated: api/auth.py"},
        ])
        tasks = sorted(n.label for n in g._nodes.values() if n.type == NodeType.TASK)
        assert tasks == ["Implemented POST /api/auth/login", "Implemented POST /api/auth/logout",
                         "Implemented POST /api/auth/register"], tasks
        endpoints = sorted(n.label for n in g._nodes.values() if n.type == NodeType.ENDPOINT)
        assert endpoints == ["POST /api/auth/login", "POST /api/auth/logout",
                             "POST /api/auth/register"], endpoints
        impl = _edges(g, EdgeType.IMPLEMENTS)
        assert ("Implemented POST /api/auth/logout", "POST /api/auth/logout") in impl

    def test_goal_loses_its_leading_article(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "user", "content": "We're building a real-time analytics dashboard in React"},
        ])
        goals = [n.label for n in g._nodes.values() if n.type == NodeType.GOAL]
        assert goals and not goals[0].lower().startswith(("a ", "an ", "the ")), goals


class TestClipParentheses:

    def test_dangling_parenthetical_is_dropped(self):
        assert _clip("Redis for refresh token storage (not DB — faster revocation and", 40) \
            == "Redis for refresh token storage"

    def test_balanced_parenthetical_is_kept(self):
        assert _clip("bcrypt for password hashing (industry standard)") \
            == "bcrypt for password hashing (industry standard)"
