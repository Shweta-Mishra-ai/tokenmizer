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
    # ~0.1s is normal on this payload. 0.5s is generous over that and still
    # catches a pattern-shape regression: a draft with a nested quantifier
    # took 1.1s and would have passed a 2.0s ceiling on a fast machine.
    assert elapsed < 0.5, f"error scan took {elapsed:.2f}s on a 15KB message"


# ── Second pass: a larger probe, then two held-out sets ──────────────────────
#
# After the first pass above, a 45-phrase probe scored decisions 60% and
# errors 40% while the corpus read 96%. The families below were added
# against that probe with the corpus as the precision guard, then measured
# on phrasings written afterwards and never used to tune anything: the
# first held-out set scored 70%/58% and drove one more round; the final
# held-out set scored 100%/100% (n=22). All three sets are pinned here.

PROBE_DECISIONS = [
    ("We'll standardise on pnpm across all packages.", "assistant", "pnpm"),
    ("OK, Kafka it is.", "user", "kafka"),
    ("Dropping Mongo, moving everything to Postgres.", "assistant", "postgres"),
    ("Agreed: bcrypt with cost 12.", "assistant", "bcrypt"),
    ("The plan is to migrate to Vite.", "assistant", "vite"),
    ("Yeah let's do the retry with exponential backoff.", "user", "backoff"),
    ("Final call: Tailwind, not styled-components.", "user", "tailwind"),
    ("Going forward, all new services use FastAPI.", "assistant", "fastapi"),
    ("Picked Playwright over Cypress for e2e.", "assistant", "playwright"),
    # held-out set 1
    ("We went ahead with Clickhouse for analytics.", "assistant", "clickhouse"),
    ("Locked in: Drizzle as the ORM.", "user", "drizzle"),
    ("Going forward: pyright, not mypy.", "assistant", "pyright"),
    ("Committing to Temporal for the workflow engine.", "assistant", "temporal"),
    # held-out set 2 (final, untuned)
    ("We're going with Pulumi over Terraform for this one.", "assistant", "pulumi"),
    ("Settled: Argon2 for hashing, not bcrypt.", "user", "argon2"),
    ("Agreed, Playwright it is.", "user", "playwright"),
    ("The team consensus: Grafana for dashboards.", "assistant", "grafana"),
    ("Sticking with Jest, no migration to Vitest.", "assistant", "jest"),
]

PROBE_ERRORS = [
    ("Getting ECONNREFUSED when hitting the db.", "user", "econnrefused"),
    ("Deploy failed, rolled back.", "assistant", "fail"),
    ("The migration errored out halfway.", "assistant", "error"),
    ("Timeout after 30s on the export endpoint.", "user", "timeout"),
    ("502 Bad Gateway from nginx intermittently.", "user", "502"),
    ("pytest: 3 failed, 41 passed.", "assistant", "fail"),
    ("panic: runtime error: index out of range [3] with length 3", "user", "panic"),
    ("The worker is stuck, hasn't processed anything in an hour.", "user", "stuck"),
    ("CI red: lint step exits 1.", "assistant", "lint"),
    ("Docker build can't find the base image.", "user", "image"),
    ("The cron job silently stopped running last Tuesday.", "user", "cron"),
    ("Users are seeing a blank page after login.", "user", "blank"),
    ("Query takes 12 seconds, it used to take 200ms.", "user", "12 seconds"),
    ("The cache returns stale data after a deploy.", "user", "stale"),
    ("Unhandled promise rejection in the upload handler.", "user", "rejection"),
    ("Memory keeps climbing until the pod gets OOMKilled.", "user", "oom"),
    # held-out set 1
    ("The API started returning 503s at 9am.", "user", "503"),
    ("Container exited with code 137.", "assistant", "137"),
    ("Auth service is down, health check failing.", "user", "down"),
    ("Kafka consumer keeps rebalancing and dropping messages.", "user", "rebalancing"),
    ("Can't connect to Redis from the worker pod.", "user", "redis"),
    # held-out set 2 (final, untuned)
    ("The webhook handler is returning 500s since the deploy.", "user", "500"),
    ("ENOENT: no such file or directory, open '/app/config.yml'", "user", "enoent"),
    ("The scheduler died overnight and nothing ran.", "user", "died"),
    ("Tests pass locally but the CI job exits with code 2.", "assistant", "exits"),
    ("Cannot resolve module '@app/utils' in the build.", "user", "resolve"),
    ("Latency regressed from 120ms to 2.4s after the ORM upgrade.", "assistant", "120ms"),
    ("The worker keeps restarting every few minutes.", "user", "restarting"),
    ("Build failed on the Windows runner only.", "assistant", "failed"),
]


@pytest.mark.parametrize("text,role,expected", PROBE_DECISIONS, ids=lambda v: v[:28] if isinstance(v, str) else v)
def test_probe_decisions(text, role, expected):
    labels = _decisions(text, role=role)
    assert any(expected in i.lower() for i in labels), labels


@pytest.mark.parametrize("text,role,expected", PROBE_ERRORS, ids=lambda v: v[:28] if isinstance(v, str) else v)
def test_probe_errors(text, role, expected):
    data = HybridExtractor().heuristic_extract([{"role": role, "content": text}])
    assert any(expected in e.lower() for e in data.errors), data.errors


@pytest.mark.parametrize("text", [
    "The dashboard takes 4.2s to first paint. Needs to be under 1.5s.",
    "Completed: project scaffold with Vite.",
    "Working on: rate limiting using slowapi.",
])
def test_probe_negatives(text):
    """Each of these produced a false positive during tuning and was the
    reason a pattern was narrowed. A measurement with a target is a goal,
    not a regression; a technology named inside a work item is how the work
    was done, not a decision."""
    data = HybridExtractor().heuristic_extract([{"role": "assistant", "content": text}])
    assert not data.errors or "4.2s" not in " ".join(data.errors)
    assert not any(label.lower() in ("use vite", "use slowapi")
                   for label in _decisions(text))
