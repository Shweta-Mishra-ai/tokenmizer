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
        ("completed_tasks", 0.62),   # measured 0.75
        ("decisions",       0.48),   # measured 0.60
        ("files",           0.80),   # measured 0.91
        ("errors",          0.70),   # measured 0.87
    ])
    def test_category_f1_floor(self, result, category, min_f1):
        assert result["micro"][category]["f1"] >= min_f1

    def test_macro_f1_floor(self, result):
        f1s = [m["f1"] for m in result["micro"].values()]
        assert sum(f1s) / len(f1s) >= 0.65   # measured 0.75

    def test_label_quality_floor(self, result):
        """Before clipping: 23% truncated, 27% multi-sentence."""
        q = result["labels"]
        assert q.truncated_pct <= 15
        assert q.multi_sentence_pct <= 12
