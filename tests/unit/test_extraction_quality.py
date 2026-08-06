"""
Extraction quality regression guards.

The eval harness (`python -m benchmarks.eval`) is the diagnostic tool;
these are the floors that keep its findings from silently eroding. They
assert measured properties of the extractor, not implementation details,
so a rewrite that keeps quality is free to change how it gets there.

Floors are set BELOW the measured values on purpose. A test pinned to the
exact current number turns every honest improvement into a failing test,
which trains people to edit the assertion instead of reading it.
"""
from __future__ import annotations

import json
import time

import pytest

from benchmarks.eval import corpus as corpus_mod
from benchmarks.eval.__main__ import evaluate
from benchmarks.eval.metrics import covers, label_quality, score
from tokenmizer.graph_memory.hybrid_extractor import _clip, get_hybrid_extractor


class TestClip:
    """`_clip` bounds a captured span to one readable clause. It replaced
    a fixed 80-character cut that produced labels ending mid-word and
    labels spanning three sentences."""

    def test_stops_at_a_sentence_boundary(self):
        out = _clip("fixed the login bug in api/auth.py. Login works now. "
                    "Also updated the docs.")
        assert out == "fixed the login bug in api/auth.py"

    @pytest.mark.parametrize("text,keep", [
        ("evaluation script in evaluate.py with metrics", "evaluate.py"),
        ("added redis to requirements.txt for the cache", "requirements.txt"),
        ("environment setup with Python 3.12 and bcrypt", "3.12"),
    ])
    def test_does_not_break_on_dots_inside_tokens(self, text, keep):
        """A bare `[.!?]` also matches the dot in `evaluate.py` and
        `Python 3.12`, silently truncating those labels. The clause
        pattern requires whitespace or end-of-string after the dot."""
        assert keep in _clip(text)

    def test_drops_a_dangling_connective(self):
        assert not _clip("wired up the payment webhook and").endswith(" and")

    def test_never_exceeds_the_budget(self):
        assert len(_clip("x" * 500, 90)) <= 90

    def test_handles_empty_and_whitespace(self):
        assert _clip("") == ""
        assert _clip("   \n  ") == ""


class TestDeduplication:
    def test_bare_basename_does_not_shadow_a_full_path(self):
        """The file patterns match at two granularities, so one mention
        of `scripts/backfill.py` yields both it and `backfill.py`."""
        ex = get_hybrid_extractor()
        out = ex._drop_shadowed_paths(
            ["scripts/backfill.py", "backfill.py", "standalone.py"]
        )
        assert "scripts/backfill.py" in out
        assert "backfill.py" not in out
        assert "standalone.py" in out, "a file with no full path must survive"

    def test_vaguer_decision_is_collapsed_into_the_specific_one(self):
        ex = get_hybrid_extractor()
        out = ex._drop_vaguer_decisions([
            {"label": "bcrypt for password hashing"},
            {"label": "Use bcrypt"},
            {"label": "Redis for refresh token storage"},
        ])
        labels = [d["label"] for d in out]
        assert "bcrypt for password hashing" in labels
        assert "Use bcrypt" not in labels
        assert "Redis for refresh token storage" in labels

    def test_unrelated_decisions_are_both_kept(self):
        ex = get_hybrid_extractor()
        out = ex._drop_vaguer_decisions([
            {"label": "PostgreSQL for the user database"},
            {"label": "Use Kubernetes for deployment"},
        ])
        assert len(out) == 2


class TestErrorsAreExtractedFromTheWholeSession:
    """Errors used to be scanned only in the recent window, so a session
    that diagnosed failures early and spent its remaining turns fixing
    them carried none of them forward. Measured recall was 1 of 12."""

    def test_error_stated_early_is_still_extracted(self):
        messages = [
            {"role": "user", "content": "CI is flaky, roughly one run in four fails."},
            {"role": "assistant", "content":
                "Three failure modes: a port collision in the integration tests, "
                "a race in the fixture teardown, and an OOM on the Windows runner."},
        ] + [
            {"role": "assistant", "content": f"Completed: unrelated cleanup step {i}."}
            for i in range(12)
        ]
        errors = " ".join(get_hybrid_extractor().heuristic_extract(messages).errors).lower()
        assert "port collision" in errors
        assert "teardown" in errors or "race" in errors
        assert "oom" in errors or "memory" in errors

    def test_bare_symptom_words_are_not_emitted_alone(self):
        """"race" or "timeout" with no subject identifies nothing."""
        res = get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant", "content": "There was an error. It failed."}]
        )
        assert all(e.lower() not in {"error", "failed", "race", "timeout"}
                   for e in res.errors)


class TestMetrics:
    def test_coverage_is_anchored_on_the_expectation(self):
        """Otherwise one enormous label containing every keyword scores
        perfect recall."""
        assert covers("fixed the 422 error in the login endpoint", "422 error")
        assert not covers("422", "422 error in the login endpoint")

    def test_precision_counts_spurious_output(self):
        sc = score("decisions", ["Use PostgreSQL", "unrelated noise here"],
                   ["Use PostgreSQL"])
        assert sc.recall == 1.0
        assert sc.precision == 0.5

    def test_label_quality_flags_truncation_and_sprawl(self):
        q = label_quality([
            "a" * 70,                                   # truncated mid-word
            "First sentence. Second sentence here.",    # multi-sentence
            "clean label",
        ])
        assert q.truncated == 1
        assert q.multi_sentence == 1


class TestCorpus:
    def test_corpus_loads_and_is_labelled(self):
        sessions = corpus_mod.load()
        assert len(sessions) >= 8, "corpus shrank — n is the whole point"
        assert len({s.domain for s in sessions}) >= 6, "corpus lost diversity"
        for s in sessions:
            assert s.origin in corpus_mod.VALID_ORIGINS
            assert any(s.ground_truth.values()), f"{s.id} has no labels"

    def test_malformed_sessions_are_rejected_loudly(self, tmp_path):
        """A silently-skipped session would quietly change every score."""
        (tmp_path / "bad.json").write_text('{"id": "x", "origin": "made-up", '
                                           '"messages": [], "ground_truth": {}}')
        with pytest.raises(corpus_mod.CorpusError):
            corpus_mod.load(tmp_path)


class TestQualityFloors:
    """Whole-pipeline floors over the committed corpus. Set below the
    measured values so ordinary variation does not red the build, but
    high enough that a real regression does."""

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        # Class-scoped so the whole corpus is extracted once, not per test.
        return evaluate(corpus_mod.load())

    @pytest.mark.parametrize("category,min_f1", [
        ("completed_tasks", 0.82),   # measured 0.91
        ("pending_tasks",   0.82),   # measured 0.95
        ("decisions",       0.84),   # measured 0.92
        ("files",           0.92),   # measured 0.99
        ("errors",          0.82),   # measured 0.90
    ])
    def test_category_f1_floor(self, result, category, min_f1):
        assert result["micro"][category]["f1"] >= min_f1

    def test_macro_f1_floor(self, result):
        f1s = [m["f1"] for m in result["micro"].values()]
        assert sum(f1s) / len(f1s) >= 0.85   # measured 0.92

    def test_real_transcripts_floor(self, result):
        """Scored separately, because the synthetic half is easier and a
        headline macro hides that. Do not delete this to make a number
        look better — the gap between the two IS the finding."""
        real = [s for s in corpus_mod.load() if s.origin == "real"]
        assert len(real) >= 6, "real-transcript sample shrank"
        sub = evaluate(real)
        f1s = [m["f1"] for m in sub["micro"].values()]
        assert sum(f1s) / len(f1s) >= 0.80   # measured 0.87

    def test_label_quality_floor(self, result):
        """Before clipping: 23% truncated, 27% multi-sentence."""
        q = result["labels"]
        assert q.truncated_pct <= 15
        assert q.multi_sentence_pct <= 5


class TestCorpusGrounding:
    """Ground truth has to come from the transcript.

    A label written from hindsight rather than from a message is
    unreachable for every extractor — heuristic, LLM or human — so it
    caps recall at a number no code change can move, and it does so
    silently. Two such labels were in the committed corpus before this
    check existed."""

    def test_committed_corpus_is_grounded(self):
        corpus_mod.validate_grounding(corpus_mod.load())

    def test_an_ungrounded_label_is_rejected(self, tmp_path):
        (tmp_path / "s.json").write_text(json.dumps({
            "id": "s", "origin": "real", "domain": "test",
            "messages": [{"role": "assistant", "content": "Fixed the login bug."}],
            "ground_truth": {"errors": ["CUDA out of memory"]},
        }))
        sessions = corpus_mod.load(tmp_path)
        assert corpus_mod.ungrounded(sessions[0]) == [("errors", "CUDA out of memory")]
        with pytest.raises(corpus_mod.CorpusError, match="not supported"):
            corpus_mod.validate_grounding(sessions)

    def test_grounding_is_per_message_not_per_transcript(self, tmp_path):
        """Tokens scattered across separate turns are not evidence that any
        turn stated the fact. Pooling the whole transcript would let almost
        any label pass."""
        (tmp_path / "s.json").write_text(json.dumps({
            "id": "s", "origin": "real", "domain": "test",
            "messages": [{"role": "user", "content": "the backfill is slow"},
                         {"role": "assistant", "content": "raised the batch size"}],
            "ground_truth": {"errors": ["backfill batch size limit"]},
        }))
        with pytest.raises(corpus_mod.CorpusError):
            corpus_mod.validate_grounding(corpus_mod.load(tmp_path))


class TestMultiStatementTurns:
    """`(.{5,80})` ran past the full stop, so one match spanned two
    statements and the second was unreachable — finditer does not re-scan
    a span it already consumed."""

    def _extract(self, text):
        return get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant", "content": text}])

    def test_both_decisions_in_one_turn_survive(self):
        labels = [d["label"] for d in self._extract(
            "Decided: TypeScript for type safety. "
            "Decided: Recharts for data visualization (good React integration)."
        ).decisions]
        assert "TypeScript for type safety" in labels
        assert any("Recharts" in x for x in labels)

    def test_a_short_first_decision_does_not_swallow_the_second(self):
        """The clause cut has a minimum length, so a decision shorter than
        it used to run on into the next sentence."""
        labels = [d["label"] for d in self._extract(
            "Decided: Python 3.12 runtime. Decided: bcrypt for password hashing."
        ).decisions]
        assert "Python 3.12 runtime" in labels
        assert any("bcrypt" in x for x in labels)

    def test_dots_inside_tokens_still_do_not_split(self):
        labels = [d["label"] for d in self._extract(
            "Decided: date-fns instead of moment.js — tree-shakeable."
        ).decisions]
        assert any("moment.js" in x for x in labels)


class TestNotYetDone:
    def test_present_tense_intent_is_not_completed_work(self):
        """`migrated?` also matched the present tense, so an opening turn
        describing the goal was recorded as finished."""
        out = get_hybrid_extractor().heuristic_extract(
            [{"role": "user",
              "content": "We need to migrate 40M rows from MySQL to Postgres."}])
        assert not any("40M rows" in t for t in out.tasks_done)

    def test_a_fix_is_not_outstanding_work(self):
        out = get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant",
              "content": "Fixed by adding a 5 second context timeout in "
                         "internal/order/client.go. Tests pass now."}])
        assert not any("timeout" in t for t in out.tasks_wip + out.tasks_todo)

    def test_past_tense_missing_is_not_a_todo(self):
        out = get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant",
              "content": "Fixed: 422 error — was missing email validation "
                         "in the LoginRequest model."}])
        assert not any("email validation" in t for t in out.tasks_todo)


class TestErrorVocabulary:
    def _errors(self, text, role="assistant"):
        return get_hybrid_extractor().heuristic_extract(
            [{"role": role, "content": text}]).errors

    def test_a_bare_exception_name_is_an_error(self):
        """`ProxyError` is the most precise form an error label can take
        and was rejected for being short."""
        assert "ProxyError" in self._errors(
            "tiktoken has no egress here. The failure is a ProxyError.")

    def test_a_vulnerability_class_is_an_error(self):
        assert any("IDOR" in e for e in self._errors(
            "Confirmed the IDOR: every route takes session_id from the URL."))

    def test_a_caught_exception_is_not_an_error(self):
        assert not any("ImportError" in e for e in self._errors(
            "_get_encoding catches only ImportError, so the network error "
            "propagates out."))

    def test_a_status_code_named_in_a_decision_is_not_an_error(self):
        assert self._errors(
            "Decided: 404 rather than 403 for denied requests.") == []

    def test_a_received_status_code_is_an_error(self):
        assert any("500" in e for e in self._errors(
            "Every request returns 500 on an air-gapped host."))

    def test_data_loss_prose_is_an_error(self):
        assert any("discards" in e for e in self._errors(
            "Two processes each write the complete blob, so the later save "
            "discards everything the earlier one added."))

    def test_a_fixed_data_loss_is_not_reported_as_current(self):
        assert not any("resurrect" in e for e in self._errors(
            "The stale writer no longer resurrects a prune."))

    def test_a_second_error_in_one_sentence_is_reachable(self):
        errs = self._errors(
            "sqlite3.OperationalError subclasses DatabaseError, and "
            "OperationalError covers database is locked.")
        assert any("database is locked" in e for e in errs)

    def test_a_bare_determiner_plus_stopword_is_not_an_error(self):
        assert self._errors("any regressions", role="user") == []


class TestExtensionlessFiles:
    """Every file pattern keyed on a dot, so `Dockerfile` and `Makefile`
    could not be extracted at all."""

    @pytest.mark.parametrize("name", ["Dockerfile", "Makefile", "go.mod"])
    def test_short_and_extensionless_filenames_survive(self, name):
        out = get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant", "content": f"Updated the {name}."}])
        assert name in out.files


class TestScanCost:
    """The extractor runs on the hot path of a proxy, over whatever a
    caller sends. A pattern that backtracks superlinearly is a denial of
    service, not a slow test."""

    @pytest.mark.parametrize("content,label", [
        ("word." * 3000, "repeated single-token sentences"),
        ("a.b.c.d.e.f.g.h " * 2000, "dotted tokens"),
        ("a" * 40000, "one enormous token"),
        (("Fixed the " + "path/to/file.py " * 40 + ". ") * 30, "path spam"),
    ])
    def test_pathological_input_stays_bounded(self, content, label):
        """`(?:[\\w./-]+\\s+){0,3}` — a token repeat nested in a window
        repeat — took 6.3 seconds on the first case here. Flattening the
        windows brought it under 0.2s. The bound is loose on purpose; it
        is there to catch a return to superlinear, not to police ms."""
        started = time.perf_counter()
        get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant", "content": content}])
        elapsed = time.perf_counter() - started
        assert elapsed < 2.0, f"{label}: {elapsed:.1f}s to scan {len(content)} chars"


class TestVocabularyBoundaries:
    """Symptom vocabulary must match whole words. The flat subject window
    that fixed the backtracking briefly let it match inside one."""

    @pytest.mark.parametrize("content,forbidden", [
        ("All of them trace to one cause: no egress.", "race"),
        ("With no key configured, single-user use is unchanged.", "hang"),
        ("The panic button is in the corner of the settings page.", "panic button"),
    ])
    def test_a_symptom_inside_a_word_is_not_an_error(self, content, forbidden):
        errors = get_hybrid_extractor().heuristic_extract(
            [{"role": "user", "content": content}]).errors
        assert not any(forbidden in e.lower() for e in errors), errors

    def test_a_label_does_not_start_in_the_previous_sentence(self):
        """The subject window can step over a full stop; the label must
        not come out as two half-thoughts."""
        errors = get_hybrid_extractor().heuristic_extract(
            [{"role": "assistant",
              "content": "Limits apply per worker. Also flock is unreliable on NFS."}]
        ).errors
        assert any("flock is unreliable" in e for e in errors), errors
        assert not any("per worker" in e for e in errors), errors


class TestRestatedErrors:
    def test_one_failure_named_twice_yields_one_label(self):
        errors = get_hybrid_extractor().heuristic_extract([
            {"role": "user", "content": "Login keeps returning 422"},
            {"role": "assistant", "content": "Fixed: 422 error — missing email "
                                             "validation in the LoginRequest model."},
        ]).errors
        with_422 = [e for e in errors if "422" in e]
        assert len(with_422) == 1, with_422
        assert "email validation" in with_422[0], with_422
