"""
Extraction on phrasings the benchmark corpus does not use.

The corpus reported 96% error F1 while the plainest ways a developer reports
a defect were missed entirely — "the build fails with exit code 1", "CI is
failing on the Windows runner", "Docker build broke after the base image
bump", a qualified exception path like `psycopg2.errors.UniqueViolation`, and
any status report that opens a sentence ("Getting a 500 from /api/orders").
None of those shapes appear in the corpus, so the corpus could not see the
gap and closing it does not move the corpus score.

That makes these cases invisible to `python -m benchmarks.eval`, so they are
pinned here instead. The negative cases matter as much as the positive ones:
the failure-verb pattern is the loosest in the set, and an earlier draft that
did not require an object clause cost 13 points of corpus error precision by
matching bare nouns ("cache the fail") and fragments ("Three fail").
"""
from __future__ import annotations

import time

import pytest

from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor


def _errors(text: str) -> list[str]:
    return HybridExtractor().heuristic_extract(
        [{"role": "assistant", "content": text}]
    ).errors


def _decisions(text: str, role: str = "assistant") -> list[str]:
    data = HybridExtractor().heuristic_extract([{"role": role, "content": text}])
    return [d["label"] if isinstance(d, dict) else d for d in data.decisions]


# ── Errors: shapes that were silently missed ─────────────────────────────────

@pytest.mark.parametrize("text", [
    "The build fails with exit code 1 on the linter step.",
    "CI is failing on the Windows runner.",
    "The migration failed halfway and left the table locked.",
    "Docker build broke after the base image bump.",
])
def test_plain_failure_verbs_are_errors(text):
    assert _errors(text), f"no error extracted from {text!r}"


@pytest.mark.parametrize("text", [
    "psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint",
    "sqlalchemy.exc.IntegrityError: FOREIGN KEY constraint failed",
    "requests.exceptions.ConnectionError: connection refused",
])
def test_qualified_exception_paths_are_errors(text):
    """A dotted module path defeated the bare-class branch, and
    UniqueViolation does not end in Error or Exception at all."""
    assert _errors(text), f"no error extracted from {text!r}"


@pytest.mark.parametrize("text", [
    "Getting a 500 Internal Server Error from /api/orders.",
    "Returns 502 under load.",
    "Got an HTTP 429 from the rate limiter.",
])
def test_sentence_initial_status_reports_are_errors(text):
    """_ERROR_TYPED was case-sensitive for the sake of its exception-class
    branch, which also made every capitalised production verb unmatchable.
    Mid-sentence reports matched and sentence-initial ones did not."""
    assert _errors(text), f"no error extracted from {text!r}"


# ── Errors: things that must NOT be errors ───────────────────────────────────

@pytest.mark.parametrize("text", [
    "Loaded config.Settings from the environment.",
    "All tests pass and the build is green.",
    "It catches only ImportError deliberately.",
    "We cache the fail count in Redis.",
])
def test_non_failures_are_not_errors(text):
    assert _errors(text) == [], f"false positive on {text!r}"


# ── Decisions: shapes that were silently missed ──────────────────────────────

def test_supersession_survives_a_slash_in_the_dependency_name():
    """The old side's character class excluded `/`, so every Go module,
    scoped npm package and org/repo name broke the whole match — losing the
    transition and the new decision with it."""
    labels = _decisions(
        "Switching from cenkalti/backoff to a hand-rolled retry loop."
    )
    assert any("retry loop" in i.lower() for i in labels), labels


def test_adoption_phrasing_without_a_choosing_verb():
    labels = _decisions(
        "Should talk gRPC to the inventory service.", role="user"
    )
    assert any("grpc" in i.lower() for i in labels), labels


def test_a_technology_named_in_the_problem_is_not_a_decision():
    """Pass 4 inferred a decision from a bare tech name with no context gate,
    turning a statement of the problem into a choice."""
    labels = _decisions(
        "We need to migrate 40M rows from MySQL to Postgres with no downtime.",
        role="user",
    )
    assert not any(i.strip().lower() == "use postgres" for i in labels), labels


# ── The constraint every pattern here has to respect ─────────────────────────

def test_error_scan_stays_linear_on_adversarial_input():
    """patterns.py documents a 6.3s ReDoS these patterns were rewritten to
    avoid; anything added to the battery has to keep that property, since
    this runs on whatever a caller posts to the proxy."""
    payload = "word." * 3000  # ~15 KB
    started = time.monotonic()
    _errors(payload)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"error scan took {elapsed:.1f}s on a 15KB message"
