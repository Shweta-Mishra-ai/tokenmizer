"""
Regression tests for the fixes in CHANGELOG's [0.5.4] entry ("decision
and error extraction, targeted at an external benchmark").

That release shipped four specific fixes — a question-context guard, two
vocabulary additions for decisions, and one for errors — verified only
by confirming the repo's own 14-session eval corpus did not regress
(90%/95% decisions, 93%/96% errors unchanged). That corpus does not
contain the gaps the fixes target, so nothing before this file actually
exercised the new code paths: zero tests referenced
`_is_question_context`, "leaning toward"/"recommend", the new tech
whitelist entries, or the new error vocabulary. A regex change with no
test asserting its own behavior is exactly the kind of unverified fix
this project's own audit history flags elsewhere (see
test_negation_scope.py, whose template this file follows).
"""
from __future__ import annotations

from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor
from tokenmizer.graph_memory.patterns import _is_question_context

extractor = HybridExtractor(min_confidence=0.50)


def _decision_labels(messages):
    return [d.get("label", "") for d in extractor.heuristic_extract(messages).decisions]


def _errors(messages):
    return extractor.heuristic_extract(messages).errors


class TestQuestionContextGuard:
    """The bug this shipped to fix: a question weighing options was
    recorded as a decision made, because the negation guard only looks
    backward for words like "not" and a question has none."""

    def test_is_question_context_true_when_clause_ends_in_question_mark(self):
        content = "Should we go with Postgres for this?"
        # match_start at "go with" (index of "go")
        idx = content.index("go with")
        assert _is_question_context(content, idx) is True

    def test_is_question_context_false_for_a_statement(self):
        content = "We are going with Postgres for this."
        idx = content.index("going with")
        assert _is_question_context(content, idx) is False

    def test_is_question_context_scoped_to_current_clause_not_next(self):
        # The match's own clause is a statement; a LATER clause happens to
        # end in "?" — that must not retroactively suppress this match.
        content = "We are going with Postgres. Does that work for you?"
        idx = content.index("going with")
        assert _is_question_context(content, idx) is False

    def test_question_weighing_postgres_or_redis_yields_no_decision(self):
        messages = [{
            "role": "user",
            "content": "Should we go with Postgres or is Redis better for this?",
        }]
        labels = _decision_labels(messages)
        assert not any("postgres" in label.lower() or "redis" in label.lower()
                        for label in labels), (
            f"a question weighing options was extracted as a decision: {labels}"
        )

    def test_decision_header_format_also_respects_question_guard(self):
        messages = [{"role": "user", "content": "Decided: should we use Kafka?"}]
        labels = _decision_labels(messages)
        assert not any("kafka" in label.lower() for label in labels), (
            f"a question-form 'Decided:' header was extracted as a decision: {labels}"
        )

    def test_real_decision_after_a_question_is_still_extracted(self):
        """The guard must not become a blanket suppressor — a genuine
        decision in its own (non-question) clause still needs to fire."""
        messages = [{
            "role": "user",
            "content": "Should we use Postgres? Yes — decided: going with Postgres.",
        }]
        labels = _decision_labels(messages)
        assert any("postgres" in label.lower() for label in labels), (
            f"the question guard suppressed a real decision in a later clause: {labels}"
        )


class TestNewDecisionVocabulary:
    """New trigger verbs ('leaning toward', 'recommend(ed/s)') and new
    tech-name whitelist entries the benchmark's missed-items list named."""

    def test_leaning_toward_is_a_decision_trigger(self):
        messages = [{"role": "user", "content": "Leaning toward using dbt for transforms."}]
        labels = _decision_labels(messages)
        assert any("dbt" in label.lower() for label in labels), (
            f"'leaning toward' + new tech name 'dbt' not extracted: {labels}"
        )

    def test_recommends_is_a_decision_trigger(self):
        messages = [{"role": "user", "content": "The team recommends sqlc for the query layer."}]
        labels = _decision_labels(messages)
        assert any("sqlc" in label.lower() for label in labels), (
            f"'recommends' + new tech name 'sqlc' not extracted: {labels}"
        )

    def test_recommended_past_tense_is_a_decision_trigger(self):
        messages = [{"role": "user", "content": "Airflow was recommended for orchestration."}]
        labels = _decision_labels(messages)
        assert any("airflow" in label.lower() for label in labels), (
            f"past-tense 'recommended' + new tech name 'airflow' not extracted: {labels}"
        )

    def test_new_tech_whitelist_entries_are_recognized(self):
        for tech in ["sqlc", "dbt", "nats", "pnpm", "uv", "ruff", "kong", "airflow"]:
            messages = [{"role": "user", "content": f"Going with {tech} for this."}]
            labels = _decision_labels(messages)
            assert any(tech in label.lower() for label in labels), (
                f"new whitelist tech name {tech!r} was not extracted as a decision: {labels}"
            )


class TestNewErrorVocabulary:
    """Expanded error-symptom vocabulary and wider trailing-context
    capture from the same release."""

    def test_nil_pointer_dereference_is_extracted(self):
        messages = [{"role": "user", "content": "Hit a nil pointer dereference in the handler."}]
        errors = _errors(messages)
        assert any("nil pointer" in e.lower() for e in errors), f"got: {errors}"

    def test_thundering_herd_is_extracted(self):
        messages = [{"role": "user", "content": "We're seeing a thundering herd on cache expiry."}]
        errors = _errors(messages)
        assert any("thundering herd" in e.lower() for e in errors), f"got: {errors}"

    def test_symptom_with_between_clause_keeps_the_qualifying_context(self):
        messages = [{
            "role": "user",
            "content": "There's a deadlock between the two mutexes during shutdown.",
        }]
        errors = _errors(messages)
        assert any("deadlock" in e.lower() and "mutex" in e.lower() for e in errors), (
            f"the wider trailing-context capture (between-clause) dropped the "
            f"qualifying detail: {errors}"
        )

    def test_symptom_with_gerund_continuation_keeps_the_qualifying_context(self):
        messages = [{
            "role": "user",
            "content": "Consumer lag is growing unbounded on the ingest topic.",
        }]
        errors = _errors(messages)
        assert any("consumer lag" in e.lower() for e in errors), f"got: {errors}"
