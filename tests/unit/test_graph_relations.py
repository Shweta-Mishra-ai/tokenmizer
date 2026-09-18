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


class TestSupersession:
    """A transition needs evidence the state CHANGED over time.

    The old pattern listed "instead of" beside "switched from" and then
    required a trailing "to/with/for", so "Next.js instead of React for
    better SEO" parsed as old="React", new="better SEO" — a decision to
    use "better SEO", superseding the real one. It also could not match
    "date-fns instead of moment.js" at all. And even when a supersession
    WAS parsed, `_apply_extracted` never read it: the pair only became a
    transition if the topic classifier happened to bucket both sides
    together, which on the corpus session that exists to demonstrate the
    feature it did not.
    """

    def test_forward_form_records_the_change(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Switching from cenkalti/backoff to a hand-rolled retry loop. Removed the dep."},
        ])
        assert len(g._transitions) == 1, [(t.from_label, t.to_label) for t in g._transitions]
        t = g._transitions[0]
        assert "backoff" in t.from_label and "retry loop" in t.to_label
        old = next(n for n in g._nodes.values() if n.id == t.from_decision_id)
        assert old.status == NodeStatus.SUPERSEDED
        assert _edges(g, EdgeType.SUPERSEDES)

    def test_replace_form_records_the_change(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content": "Replaced moment.js with date-fns."},
        ])
        assert len(g._transitions) == 1
        assert "moment" in g._transitions[0].from_label

    def test_instead_of_is_one_decision_not_a_change(self, tmp_path):
        """One sentence naming a choice and its rejected alternative is a
        single decision; the corpus labels it with the whole phrase."""
        g = _graph(tmp_path, [
            {"role": "assistant", "content": "Decided: Next.js instead of React for better SEO"},
        ])
        assert g._transitions == []
        labels = [n.label for n in g._nodes.values() if n.type == NodeType.DECISION]
        # One decision, stated as the phrase. The defect was a SECOND node
        # labelled "Use better SEO" — the rationale recorded as the thing
        # chosen — which then superseded the real decision.
        assert len(labels) == 1, labels
        assert not any(x.lower().startswith("use better") for x in labels), labels

    def test_common_noun_operand_is_not_a_supersession(self, tmp_path):
        """'Replaced the pattern with X' names no technology."""
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Replaced the pattern with typed-exception matching in hybrid_extractor.py."},
        ])
        assert g._transitions == []

    def test_a_supersession_is_recorded_once(self, tmp_path):
        """Re-extracting the same transcript must not stack duplicates."""
        msgs = [{"role": "assistant", "content": "Switched from moment.js to date-fns."}]
        g = _graph(tmp_path, msgs)
        g.extract_from_messages(msgs, incremental=False)
        assert len(g._transitions) == 1

    def test_why_walks_the_chain(self, tmp_path):
        from tokenmizer.graph_memory.reasoning import why

        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Switched from moment.js to date-fns because it is tree-shakeable."},
        ])
        answer = why(g, "date-fns")
        assert answer["chain"], answer
        assert answer["current"] and "date-fns" in answer["current"]["label"]


class TestNearDuplicateDecisions:
    """Two phrasings of one decision must merge, and two real alternatives
    must not. Before this, "PostgreSQL for order storage" and "Postgres for
    orders" were read as competing products in the database slot, so every
    resume block for that session carried a "Conflicting (unresolved)"
    line about a conflict that did not exist.
    """

    from tokenmizer.graph_memory.decision_tracker import _is_same_decision as _same

    SAME = [
        ("PostgreSQL for order storage", "Postgres for orders"),
        ("cenkalti/backoff for it", "Use cenkalti/backoff"),
        ("Use PostgreSQL", "Use PostgreSQL 16 with pgvector"),
        ("Use Next.js", "Use NextJS"),
    ]
    DIFFERENT = [
        ("Use MySQL for the user database", "Use MongoDB for the user database"),
        ("Use PostgreSQL", "Use MongoDB"),
        ("Use Redis for the cache", "Use Redis for sessions"),
        ("Use bcrypt", "Avoid bcrypt"),
        ("Use Next.js for the frontend", "Use React for the frontend"),
    ]

    def test_same_decision_phrased_two_ways(self):
        from tokenmizer.graph_memory.decision_tracker import _is_same_decision
        for a, b in self.SAME:
            assert _is_same_decision(a, b), f"{a!r} vs {b!r} should merge"

    def test_competing_alternatives_stay_separate(self):
        from tokenmizer.graph_memory.decision_tracker import _is_same_decision
        for a, b in self.DIFFERENT:
            assert not _is_same_decision(a, b), f"{a!r} vs {b!r} must not merge"

    def test_no_false_conflict_in_the_resume_block(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content": "Decided: PostgreSQL for order storage."},
            {"role": "user", "content": "remind me what we picked"},
            {"role": "assistant", "content": "Decided: Postgres for orders."},
        ])
        block = g.to_context_block(token_budget=400)
        assert "Conflicting" not in block, block


class TestTaskLabelsReadAsStatements:

    def test_the_consumed_verb_is_restored(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content": "Removed the dependency from go.mod."},
        ])
        labels = [n.label for n in g._nodes.values() if n.type == NodeType.TASK]
        assert any(x.lower().startswith("removed the dependency") for x in labels), labels

    def test_a_complete_phrase_is_left_alone(self, tmp_path):
        g = _graph(tmp_path, [
            {"role": "assistant", "content": "Completed: virtual environment setup."},
        ])
        labels = [n.label for n in g._nodes.values() if n.type == NodeType.TASK]
        assert "virtual environment setup" in labels, labels

    def test_the_changes_line_does_not_collide_with_its_own_separator(self, tmp_path):
        """context_block joins transitions with " | " and the transition
        line used " | Reason: ", so the Changes line read as a list whose
        items did not line up with the changes they described."""
        g = _graph(tmp_path, [
            {"role": "assistant", "content":
             "Switched from moment.js to date-fns because it is tree-shakeable."},
        ])
        line = g._transitions[0].to_context_line()
        assert "|" not in line, line
