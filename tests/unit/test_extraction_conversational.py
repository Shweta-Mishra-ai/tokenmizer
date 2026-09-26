"""
Conversational forms, and the precision bugs found while adding them.

The external 100-session benchmark (tokenmizer-research, memorybench) showed
the heuristic pass finding about half of what a session states in every
category but files — completed-task recall 53%, pending 37%, decisions 54%,
errors 39% — because nearly every pattern keyed on a marker ("Done:",
"TODO:", "Decided:") and most of a working session has none. The shapes
added for that are pinned here with phrasings written for these tests, not
copied from either benchmark corpus, so a pass here is not a pass on the
corpus restated.

The negative cases are the point as much as the positive ones. Every loose
shape here has a false-positive twin — a condition ("once X is merged"), a
question, a negation in another clause, an adjective ("retry for failed
deliveries"), a pronoun subject ("everything is in place") — and each has a
test that it stays out.
"""
from __future__ import annotations

import time

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor


def _x(text: str, role: str = "assistant"):
    return HybridExtractor().heuristic_extract([{"role": role, "content": text}])


def _done(text, role="assistant"):
    return _x(text, role).tasks_done


def _todo(text, role="assistant"):
    x = _x(text, role)
    return x.tasks_todo + x.tasks_wip


def _decisions(text, role="assistant"):
    return [d["label"] for d in _x(text, role).decisions]


def _errors(text, role="assistant"):
    return _x(text, role).errors


def _hit(labels, needle):
    return any(needle.lower() in label.lower() for label in labels)


# ── Completed ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,needle", [
    ("Wrapped up the CSV export for the admin panel.", "csv export"),
    ("Knocked out the flaky retry logic this morning.", "retry logic"),
    ("Got the websocket reconnect working after lunch.", "websocket reconnect"),
    ("I got the nightly backup job merged.", "nightly backup job"),
    ("The invoice PDF generator is in and working.", "invoice pdf generator"),
    ("Structured logging is now in place across the workers.", "structured logging"),
    ("The PR's merged.", "the pr"),
    ("- [x] webhook signature verification", "webhook signature verification"),
])
def test_conversational_completion(text, needle):
    assert _hit(_done(text), needle), _done(text)


@pytest.mark.parametrize("text", [
    "Once the migration is merged we can deploy.",
    "When the backfill is done, re-enable the cron.",
    "If the cache layer is in place by Friday we ship.",
    "Everything is in place now.",
    "No integration tests are in place yet.",
    "The unfinished export job still needs a retry.",
])
def test_conditions_pronouns_and_negated_verbs_are_not_completions(text):
    assert _done(text) == [], _done(text)


def test_phrasal_particle_stays_with_the_verb():
    """`finished UP dark mode` was labelled "up dark mode"."""
    labels = _done("Finished up dark mode using CSS custom properties.")
    assert labels and not labels[0].lower().startswith("up "), labels


# ── Pending ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,needle", [
    ("Haven't touched the billing webhooks yet.", "billing webhooks"),
    ("We'll circle back to the SSO flow after the release.", "sso flow"),
    ("Next steps: load testing the search endpoint.", "load testing the search endpoint"),
    ("Remaining: a runbook for the failover drill.", "runbook for the failover drill"),
    ("- [ ] rotate the staging credentials", "rotate the staging credentials"),
    ("The GDPR export is next on the list.", "gdpr export"),
    ("The retention policy hasn't been implemented yet.", "retention policy"),
    ("Tomorrow I'll tackle the flaky e2e suite.", "flaky e2e suite"),
])
def test_conversational_pending(text, needle):
    assert _hit(_todo(text), needle), _todo(text)


@pytest.mark.parametrize("text", [
    "I'll explain the reasoning below.",
    "Is the audit log next on the list?",
    "The port is open.",
    "That is next.",
])
def test_non_tasks_are_not_pending(text):
    assert _todo(text) == [], _todo(text)


def test_todo_label_drops_the_deferral_tail():
    labels = _todo("We'll circle back to the audit log once this lands.")
    assert labels == ["the audit log"], labels


def test_missing_after_an_article_is_a_defect_not_a_todo():
    """ "Bug: a missing statistical test" was filed as planned work."""
    assert _todo("Bug: a missing statistical test on the headline comparison.") == []


def test_early_todo_in_a_long_session_survives(tmp_path):
    """To-dos were read from the last 20 messages only, so one stated early
    in a session over 30 messages never reached the graph."""
    msgs = [{"role": "assistant", "content": "Still need: a load test against the order path."}]
    for i in range(40):
        msgs.append({"role": "user" if i % 2 else "assistant", "content": f"Continuing step {i}."})
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.extract_from_messages(msgs, incremental=False)
    pending = [n.label for n in g._nodes.values()
               if n.type == NodeType.TASK and n.status in (NodeStatus.PENDING, NodeStatus.IN_PROGRESS)]
    assert _hit(pending, "load test"), pending


# ── Plan and completion are one task ────────────────────────────────────────

def test_completion_closes_the_matching_todo_across_calls(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    first = [{"role": "assistant", "content": "Still need: rate limiting on the auth routes."}]
    g.extract_from_messages(first)
    second = first + [
        {"role": "user", "content": "Status?"},
        {"role": "assistant", "content": "Added rate limiting on the auth routes."},
    ]
    g.extract_from_messages(second)
    tasks = [(n.label, n.status) for n in g._nodes.values() if n.type == NodeType.TASK]
    assert len(tasks) == 1 and tasks[0][1] == NodeStatus.COMPLETED, tasks


def test_distinct_tasks_with_most_words_in_common_stay_apart(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.add_node(NodeType.TASK, "Add tests for the login flow", NodeStatus.PENDING)
    g.add_node(NodeType.TASK, "Added tests for the logout flow", NodeStatus.COMPLETED)
    assert sum(1 for n in g._nodes.values() if n.type == NodeType.TASK) == 2


def test_two_completed_tasks_are_never_fuzzy_merged(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.add_node(NodeType.TASK, "Worker 0 implemented feature 0 in the API", NodeStatus.COMPLETED)
    g.add_node(NodeType.TASK, "Worker 0 implemented feature 1 in the API", NodeStatus.COMPLETED)
    assert sum(1 for n in g._nodes.values() if n.type == NodeType.TASK) == 2


# ── Decisions ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,role,needle", [
    ("Use Litestream for SQLite replication.", "user", "litestream"),
    ("Build it with Temporal for the retries.", "user", "temporal"),
    ("I think we should use Valkey for the rate limiter.", "user", "valkey for the rate limiter"),
    ("After some back and forth, Caddy felt like the right call for TLS.", "assistant", "caddy"),
    ("It wasn't close, but Buildkite won out over Jenkins.", "assistant", "buildkite"),
    ("For the queue, SQS made the most sense.", "assistant", "sqs"),
])
def test_conversational_decisions(text, role, needle):
    assert _hit(_decisions(text, role), needle), _decisions(text, role)


@pytest.mark.parametrize("text,role", [
    ("Use `npm run dev` to start the server.", "assistant"),   # an instruction, not a choice
    ("Pick up where we left off.", "user"),
    ("Kafka wasn't the right call for this.", "assistant"),
    ("If Redis is the better option we can switch later.", "assistant"),
    ("Is Postgres the right choice here?", "assistant"),
])
def test_non_decisions(text, role):
    assert _decisions(text, role) == [], _decisions(text, role)


def test_user_proposal_is_a_decision_only_once_accepted():
    he = HybridExtractor()
    accepted = he.heuristic_extract([
        {"role": "user", "content": "Can we do this with Celery beat?"},
        {"role": "assistant", "content": "Sounds good, setting that up."},
    ])
    assert _hit([d["label"] for d in accepted.decisions], "celery beat")
    rejected = he.heuristic_extract([
        {"role": "user", "content": "Can we do this with Celery beat?"},
        {"role": "assistant", "content": "Sure, but I'd use a plain cron job instead."},
    ])
    assert not _hit([d["label"] for d in rejected.decisions], "celery beat")


def test_proposal_accepted_in_a_later_extraction_call(tmp_path):
    """The proxy sees the user's turn and the assistant's reply in
    different calls; the reply must still be able to accept."""
    g = GraphMemory("s", storage_dir=str(tmp_path))
    turn = [{"role": "user", "content": "How about Meilisearch for the product search?"}]
    g.extract_from_messages(turn)
    g.extract_from_messages(turn + [{"role": "assistant", "content": "Agreed, wiring it in now."}])
    decisions = [n.label for n in g._nodes.values() if n.type == NodeType.DECISION]
    assert _hit(decisions, "meilisearch"), decisions


@pytest.mark.parametrize("text,wrong", [
    ("Hit a snag: excessive allocations causing GC pressure.", "gc pressure"),   # c-USING
    ("TODO: watchOS companion app.", "companion app"),                          # watCHOS
    ("Somewhere in the refactor we picked up duplicate rows.", "duplicate rows"),
])
def test_choosing_verbs_do_not_match_inside_other_words(text, wrong):
    assert not _hit(_decisions(text), wrong), _decisions(text)


def test_bare_use_x_collapses_into_the_decision_that_names_it():
    labels = _decisions("Decided: Pydantic v2 for request validation.")
    assert labels == ["Pydantic v2 for request validation"], labels


def test_a_tool_named_in_reported_work_is_not_a_decision():
    assert _decisions("Finished up dark mode using CSS custom properties.") == []


# ── Errors ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,needle", [
    ("Bug: the eval split overlapping the training window.", "eval split overlapping"),
    ("**Issue:** tokens refreshed twice on cold start.", "tokens refreshed twice"),
    ("Ran into connection pool exhaustion under load.", "connection pool exhaustion"),
    ("Hit a snag with the S3 multipart upload.", "s3 multipart upload"),
    ("We're seeing duplicate charges from the retry path.", "duplicate charges"),
    ("There's a race in the fixture teardown.", "race in the fixture teardown"),
])
def test_conversational_errors(text, needle):
    assert _hit(_errors(text), needle), _errors(text)


@pytest.mark.parametrize("text", [
    "We still need a background retry for failed webhook deliveries.",
    "Haven't started a rollback job for failed deploys yet.",
    "The wrapper retries failed requests with backoff.",
    "The baseline is a logistic regression at 0.71 AUC.",
    "Added soft deletes on the tenant table.",
    "We're seeing a 20% speedup on the hot path.",
    "There's no error in the logs.",
    "Errors: none.",
])
def test_non_errors(text):
    assert _errors(text) == [], _errors(text)


def test_error_label_starts_on_a_whole_word():
    """A window that opened after an apostrophe began on the contraction's
    tail: "t started a rollback job", "s work on a memory leak"."""
    labels = _errors("Let's work on a memory leak in the image cache.")
    assert labels and not labels[0].startswith(("s ", "t ", "re ")), labels


def test_error_label_ends_on_a_whole_word():
    labels = _errors("Issue: a crash from a missing NSMotionUsageDescription key.")
    assert _hit(labels, "NSMotionUsageDescription key"), labels


def test_a_real_failed_deploy_is_still_an_error():
    assert _errors("The failed deploy from last night broke checkout.")


# ── Files ────────────────────────────────────────────────────────────────────

def test_multi_dot_filenames_are_kept_whole():
    files = _x("Made the changes in vite.config.ts and docker-compose.override.yml.").files
    assert "vite.config.ts" in files and "config.ts" not in files, files
    assert "docker-compose.override.yml" in files, files


def test_extensionless_build_file_keeps_its_directory():
    assert "fastlane/Fastfile" in _x("Updated: fastlane/Fastfile.").files


def test_a_javascript_library_name_is_not_a_file():
    files = _x("Switched from moment.js to date-fns in src/utils/dates.ts.").files
    assert "moment.js" not in files and "src/utils/dates.ts" in files, files


# ── Cost ─────────────────────────────────────────────────────────────────────

def test_multi_dot_file_pattern_stays_linear():
    """An unbounded repeat of `.segment` took 3.5s on this payload."""
    started = time.monotonic()
    _x("word." * 3000)
    assert time.monotonic() - started < 2.0


# ── Second round: found on held-out v1, fixed before held-out v2 was scored ──

@pytest.mark.parametrize("text,kind,needle", [
    ("**Decision:** Postgres for the orders store.", "decision", "postgres for the orders store"),
    ("**TODO:** backfill the legacy rows.", "todo", "backfill the legacy rows"),
    ("**Bug:** duplicate emails on retry.", "error", "duplicate emails on retry"),
])
def test_markdown_bold_headers(text, kind, needle):
    """`**` in front of the keyword defeated every header pattern."""
    got = {"decision": _decisions, "todo": _todo, "error": _errors}[kind](text)
    assert _hit(got, needle), got


def test_python_dunder_names_survive_emphasis_stripping():
    assert "src/pkg/__init__.py" in _x("Edited src/pkg/__init__.py.").files


def test_intransitive_completion_takes_the_subject():
    """"landed in the last commit" was labelled "Landed in in the last commit"."""
    assert _done("The retry fix landed in the last commit.") == ["The retry fix"]


def test_a_defect_report_is_not_finished_work():
    assert _done("Bug: a breaking change shipped in a minor version.") == []


def test_wip_verb_does_not_match_inside_another_word():
    """`rewriting` matched `writing`; the aspect verb says it is finished."""
    assert _todo("Ended up rewriting half of conftest.py along the way.") == []
    assert _hit(_todo("Rewriting the parser for streaming input."), "parser")


@pytest.mark.parametrize("text,role,needle", [
    ("We'll go with Litestream for replication.", "assistant", "litestream"),
    ("Opting for a monorepo.", "assistant", "monorepo"),
    ("I'd prefer argon2 over bcrypt.", "user", "argon2"),
    ("Final choice: Tailwind for styling.", "assistant", "tailwind"),
])
def test_more_choosing_verbs(text, role, needle):
    assert _hit(_decisions(text, role), needle), _decisions(text, role)


def test_go_with_needs_a_subject_or_modal():
    assert _decisions("Just go with the flow on naming for now.") == []


def test_fronted_up_next():
    assert _hit(_todo("Up next is the GDPR export."), "gdpr export")


# ── Marker-free forms ────────────────────────────────────────────────────────
#
# Work reported without a state word: what no longer needs attention, what
# it made possible, what to keep in mind. Every positive case has false
# positives next to it that share its words.

@pytest.mark.parametrize("text,needle", [
    ("No more worrying about the rate limiter on the login route.", "rate limiter"),
    ("Crossed the SSO integration off the list this morning.", "sso integration"),
    ("That leaves the Stripe webhook retries behind us.", "stripe webhook retries"),
    ("With the schema migration sorted, the rest should go quicker.", "schema migration"),
    ("Once the connection pooling was in, the timeouts stopped.", "connection pooling"),
    ("The audit log export is no longer on my plate.", "audit log export"),
    ("The incident was resolved within an hour.", "the incident"),
])
def test_marker_free_completion(text, needle):
    assert _hit(_done(text), needle), _done(text)


@pytest.mark.parametrize("text", [
    "Once the migration is merged, we can deploy.",   # future condition
    "I checked the logs off and on all morning.",
    "That left the team behind us in the queue.",
    "No more than three retries are allowed.",
    "Since the cache was in, why is it slow?",
])
def test_marker_free_completion_false_positives(text):
    assert _done(text) == [], _done(text)


@pytest.mark.parametrize("text,needle", [
    ("Let's not forget about the runbook for the payment service.", "runbook"),
    ("Parking the dark-mode toggle for now, it's not urgent.", "dark-mode toggle"),
    ("The GDPR export is on hold until legal signs off.", "gdpr export"),
])
def test_marker_free_pending(text, needle):
    assert _hit(_todo(text), needle), _todo(text)


@pytest.mark.parametrize("text", [
    "Don't forget that the API is rate limited.",
    "Let's not forget why we did this in the first place.",
    "The parking service handles reservations.",
])
def test_marker_free_pending_false_positives(text):
    assert _todo(text) == [], _todo(text)


def test_noun_phrase_it_is_is_a_decision():
    assert _hit(_decisions("Rather than overthinking it, the managed Postgres it is."),
                "managed postgres")


@pytest.mark.parametrize("text", [
    "The code is fine as it is.", "That's just how it is.",
    "I don't know what it is.", "Whatever it is, we'll find it.",
])
def test_it_is_that_is_not_a_choice(text):
    assert _decisions(text) == [], _decisions(text)


@pytest.mark.parametrize("text,needle", [
    ("Tracked the flakiness down to a connection pool exhaustion under load.",
     "connection pool exhaustion"),
    ("CI is unhappy — a connection pool exhaustion under load.", "connection pool exhaustion"),
    ("Everything looked green, then a connection pool exhaustion again.",
     "connection pool exhaustion"),
])
def test_marker_free_errors(text, needle):
    assert _hit(_errors(text), needle), _errors(text)


@pytest.mark.parametrize("text", [
    "CI is red — rerunning the job now.",
    "Then run it again.",
    "We tracked the ticket down to the billing team.",
])
def test_marker_free_error_false_positives(text):
    assert _errors(text) == [], _errors(text)
