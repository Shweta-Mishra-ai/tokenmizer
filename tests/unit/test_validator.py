"""Unit tests — graph validator and confidence scoring."""
import pytest

from tokenmizer.graph_memory.validator import GraphValidator, get_validator


@pytest.fixture
def v():
    return GraphValidator(min_confidence=0.50)


class TestHardRejects:

    def test_empty_label_rejected(self, v):
        r = v.validate("", "task")
        assert r.accepted is False

    def test_noise_word_rejected(self, v):
        for word in ["this", "that", "it", "ok", "done", "yes"]:
            r = v.validate(word, "task")
            assert r.accepted is False, f"'{word}' should be rejected"

    def test_pure_number_rejected(self, v):
        r = v.validate("42", "task")
        assert r.accepted is False

    def test_too_short_rejected(self, v):
        r = v.validate("ab", "task")
        assert r.accepted is False

    def test_generic_single_verb_rejected(self, v):
        r = v.validate("implement", "task")
        assert r.accepted is False

    def test_url_rejected(self, v):
        r = v.validate("https://example.com/api", "task")
        assert r.accepted is False


class TestAccepted:

    def test_specific_task_accepted(self, v):
        r = v.validate("Implement JWT refresh token rotation", "task")
        assert r.accepted is True
        assert r.confidence >= 0.55

    def test_file_with_extension_accepted(self, v):
        r = v.validate("api/auth.py", "file")
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_decision_with_rationale_accepted(self, v):
        r = v.validate(
            "Use Redis for session storage",
            "decision",
            summary="Faster than PostgreSQL for ephemeral data",
        )
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_decision_without_rationale_lower_confidence(self, v):
        with_rationale = v.validate("Use Redis", "decision", summary="faster than postgres")
        without_rationale = v.validate("Use Redis", "decision", summary="")
        assert with_rationale.confidence > without_rationale.confidence

    def test_environment_with_version_accepted(self, v):
        r = v.validate("Python 3.12", "environment")
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_dependency_with_version_accepted(self, v):
        r = v.validate("fastapi>=0.111.0", "dependency")
        assert r.accepted is True
        assert r.confidence >= 0.65

    def test_goal_accepted(self, v):
        r = v.validate("Build FastAPI authentication service with JWT", "goal")
        assert r.accepted is True

    def test_error_accepted(self, v):
        r = v.validate("422 Unprocessable Entity on login endpoint", "error")
        assert r.accepted is True


class TestConfidenceScoring:

    def test_longer_label_higher_confidence(self, v):
        short = v.validate("Fix bug", "task")
        long = v.validate("Fix authentication 422 error in login endpoint", "task")
        if short.accepted and long.accepted:
            assert long.confidence >= short.confidence

    def test_summary_boosts_confidence(self, v):
        no_summary = v.validate("Use PostgreSQL", "decision", summary="")
        with_summary = v.validate("Use PostgreSQL", "decision",
                                  summary="needed for concurrent writes in production")
        if no_summary.accepted and with_summary.accepted:
            assert with_summary.confidence > no_summary.confidence

    def test_confidence_bounded_0_to_1(self, v):
        for label, node_type in [
            ("Implement auth", "task"),
            ("api/main.py", "file"),
            ("Use Redis", "decision"),
        ]:
            r = v.validate(label, node_type)
            assert 0.0 <= r.confidence <= 1.0


class TestTypeMismatch:

    def test_file_path_as_task_gets_corrected(self, v):
        r = v.validate("api/auth.py", "task")
        # Should either reject or correct type to "file"
        if r.accepted:
            assert r.corrected_type == "file"

    def test_endpoint_as_task_gets_corrected(self, v):
        r = v.validate("POST /api/auth/login", "task")
        if r.accepted:
            assert r.corrected_type == "endpoint"

    def test_dep_pattern_gets_corrected(self, v):
        r = v.validate("fastapi==0.111.0", "task")
        if r.accepted:
            assert r.corrected_type == "dependency"


class TestGraphIntegration:
    """Test validator wired into GraphMemory.add_node()."""

    def test_noise_node_not_added(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.TASK, "this")
        assert nid == ""  # rejected
        assert len(g._nodes) == 0

    def test_good_node_added_with_confidence(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test2", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.TASK, "Implement JWT authentication middleware")
        assert nid != ""
        assert len(g._nodes) == 1
        node = g._nodes[nid]
        assert node.confidence > 0.0
        assert node.confidence <= 1.0

    def test_rejected_nodes_not_in_stats(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test3", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "this")   # rejected
        g.add_node(NodeType.TASK, "that")   # rejected
        g.add_node(NodeType.TASK, "Implement refresh token rotation")  # accepted
        stats = g.stats()
        assert stats["node_count"] == 1

    def test_avg_confidence_in_stats(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test4", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Implement auth endpoint")
        g.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                   summary="concurrent writes needed")
        stats = g.stats()
        assert "avg_confidence" in stats
        assert 0.0 < stats["avg_confidence"] <= 1.0

    def test_semantic_edges_not_accidental(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
        g = GraphMemory("edge-test", storage_dir=str(tmp_path))

        # Add task about auth
        t1 = g.add_node(NodeType.TASK, "Implement auth middleware", NodeStatus.IN_PROGRESS)
        # Add unrelated task about database
        t2 = g.add_node(NodeType.TASK, "Set up PostgreSQL connection pooling", NodeStatus.IN_PROGRESS)
        # Add file for auth
        f1 = g.add_node(NodeType.FILE, "api/auth.py")

        if t1 and t2 and f1:
            # auth.py should be linked to auth task but NOT to postgres task
            auth_task_node = g._nodes.get(t1)
            edges = [(e.source_id, e.target_id) for e in g._edges]
            if auth_task_node:
                # The postgres task should not have an edge to auth.py
                assert (t2, f1) not in edges, \
                    "PostgreSQL task should NOT be linked to api/auth.py"


class TestExtractorConfidenceBlending:
    """
    The validator blends the extractor's corroboration confidence into its
    own score:  final = max(heuristic, (heuristic + extractor) / 2).
    The blend is monotone (evidence can only raise a score), is not an
    override (weak heuristics still fail the threshold), and never applies
    to hard-rejected labels.
    """

    def test_corroborated_short_decision_now_accepted(self):
        v = GraphValidator(min_confidence=0.65)
        # Heuristics alone: "Vitest for tests" scores ~0.55 -> rejected
        alone = v.validate("Vitest for tests", "decision")
        assert not alone.accepted, (
            f"precondition changed: heuristics alone now score "
            f"{alone.confidence} — update this test's premise"
        )
        # Corroborated by both LLM and heuristic pass -> accepted
        corroborated = v.validate("Vitest for tests", "decision",
                                  extractor_confidence=0.95)
        assert corroborated.accepted
        assert corroborated.confidence > alone.confidence

    def test_heuristic_only_tier_does_not_auto_pass(self):
        """0.65 extractor tier must NOT automatically clear a 0.65
        threshold — blending is evidence, not an override."""
        v = GraphValidator(min_confidence=0.65)
        r = v.validate("Vitest for tests", "decision", extractor_confidence=0.65)
        assert not r.accepted

    def test_blend_never_lowers_heuristic_score(self):
        v = GraphValidator(min_confidence=0.50)
        strong = "Decided to use PostgreSQL because we need concurrent writes"
        base = v.validate(strong, "decision", summary="benchmarked")
        blended = v.validate(strong, "decision", summary="benchmarked",
                             extractor_confidence=0.65)
        assert blended.confidence >= base.confidence

    def test_hard_rejects_survive_high_extractor_confidence(self):
        """0.95 corroboration cannot resurrect junk."""
        v = GraphValidator(min_confidence=0.50)
        r = v.validate("ok", "decision", extractor_confidence=0.95)
        assert not r.accepted


class TestCharLenDeadBranchRemoved:
    """
    Regression test for TM-29a. `elif char_len > 40: confidence += 0.05`
    was dead code — the preceding `elif char_len > 20: confidence +=
    0.10` already caught every label longer than 20 characters,
    including all of them above 40, so the documented "diminishing
    returns on very long labels" behavior never actually ran; every long
    label has always gotten the flat +0.10 in practice.

    Making that documented behavior actually fire (checking the longer
    threshold first) was tried and reverted: it measurably REDUCED task-
    extraction recall against this repo's own memory-accuracy fixture —
    several legitimately long, specific task labels lost enough
    confidence to drop below the acceptance threshold. Since that
    regression is concrete and immediate while "diminishing returns" was
    never validated behavior to begin with, the fix removes the dead
    branch rather than activating it: labels over 20 chars keep getting
    the flat +0.10 that has actually been shipping. This test locks in
    that specific choice — the flat bonus for ANY length past 20 chars,
    including well past 40 — so a future "fix" doesn't reintroduce the
    same recall regression without a real evaluation harness to justify it.
    """

    def test_length_bonus_stays_flat_past_40_chars(self):
        v = GraphValidator(min_confidence=0.0)
        short_long = "Use PostgreSQL for storage"                              # 27 chars, 4 words
        very_long = "Use PostgreSQLReplicationClusterConfiguration for storage"  # 58 chars, 4 words
        assert len(short_long.split()) == len(very_long.split()) == 4
        assert 20 < len(short_long) <= 40
        assert len(very_long) > 40

        r_short = v.validate(label=short_long, node_type="concept")
        r_long = v.validate(label=very_long, node_type="concept")
        assert r_long.confidence == r_short.confidence, (
            f"length bonus should stay flat for any label over 20 chars "
            f"(including well past 40) — got long={r_long.confidence} "
            f"short={r_short.confidence}"
        )


class TestGetValidatorDoesNotLeakGlobalState:
    """Regression test for TM-29b: get_validator(min_confidence=X) used
    to permanently overwrite the module-level singleton — one caller
    passing an explicit override changed behavior for every OTHER caller
    that just calls get_validator() with no arguments, for the rest of
    the process lifetime."""

    def test_explicit_override_does_not_leak_to_default_calls(self, monkeypatch):
        import tokenmizer.graph_memory.validator as validator_module
        monkeypatch.setattr(validator_module, "_validator", None)

        default_validator = get_validator()
        default_threshold = default_validator.min_confidence

        get_validator(min_confidence=0.99)  # explicit override, elsewhere

        again = get_validator()  # a caller with no override
        assert again.min_confidence == default_threshold, (
            f"an explicit min_confidence override leaked into the default "
            f"get_validator() call — expected {default_threshold}, got "
            f"{again.min_confidence}"
        )


class TestSourceRoleIsActuallyPassed:
    """Regression test for TM-29c: add_node() never passed source_role
    to validator.validate(), so every node got the "assistant" default
    bonus regardless of which role's message it was actually extracted
    from. This test operates at the GraphValidator level directly (the
    add_node() wiring is a separate, larger change tracked as a
    follow-up — see PR description)."""

    def test_user_role_scores_lower_than_assistant_for_identical_label(self):
        v = GraphValidator(min_confidence=0.0)
        label = "Refactor the auth module for clarity"
        r_assistant = v.validate(label=label, node_type="concept", source_role="assistant")
        r_user = v.validate(label=label, node_type="concept", source_role="user")
        assert r_assistant.confidence > r_user.confidence
