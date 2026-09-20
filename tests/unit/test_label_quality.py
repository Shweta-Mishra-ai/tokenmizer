"""
Label quality: the two guards, and the metric that measures them.

Three defects, found by reading what the eval's label-quality block was
actually counting rather than trusting the number:

1. "truncated mid-word" was guessed from the label's shape — any label of
   60+ characters ending on a letter. 22 of the 26 it counted were clean
   labels that simply ended in a word. It is now checked against the
   transcript the label came from, which is exact.

2. Once measured properly, 5 labels really were cut mid-token — by the
   patterns' fixed 80-character capture, which had no reason to stop on a
   word boundary. `_CLAUSE_SPAN` now ends on one.

3. "near-duplicate pairs" pooled every session's labels into one bucket
   and compared across node types, so `persistence.py` in three sessions
   counted as pairs, and a task naming the file it touched counted as a
   duplicate of that file. Both are artefacts of the measurement.

And two extraction false positives the first fix exposed:

- "vanished from completed tasks and appeared as a spurious file" was
  recorded as finished work, because `completed` was read as the verb.
- "Working on a CI step that runs the image with the network removed to
  prove the bake worked" produced the completed task "to prove the bake
  worked" from `removed`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.eval.metrics import label_quality  # noqa: E402
from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor  # noqa: E402
from tokenmizer.graph_memory.patterns import _CLAUSE_SPAN  # noqa: E402


def _done(text: str) -> list[str]:
    return HybridExtractor().heuristic_extract(
        [{"role": "assistant", "content": text}]).tasks_done


def _decisions(text: str) -> list[str]:
    return [d["label"] for d in HybridExtractor().heuristic_extract(
        [{"role": "assistant", "content": text}]).decisions]


class TestCaptureEndsOnAWordBoundary:
    """A fixed-width capture that stops mid-token ships labels like
    '...confusion matri'. The span is a budget, not a place to stop
    reading a word."""

    SENTENCE = (
        "Completed: the evaluation script in evaluate.py with precision, "
        "recall, F1 and a confusion matrix over the whole corpus"
    )

    def test_no_label_ends_mid_word(self):
        for label in _done(self.SENTENCE):
            assert self.SENTENCE.split(label)[-1][:1] not in (
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
                "y", "z",
            ), f"{label!r} stops inside a word"

    def test_a_single_very_long_token_is_still_captured(self):
        """The boundary rule must not drop the fact when there is no
        boundary inside the budget — one long URL, for instance."""
        url = "https://example.com/" + "a" * 120
        assert _done(f"Implemented {url}"), (
            "a span with no word boundary inside 80 chars must fall back "
            "to a hard cut, not vanish"
        )

    def test_pattern_carries_the_fallback_branch(self):
        assert "|" in _CLAUSE_SPAN, (
            "without the second branch, a long unbroken token fails the "
            "whole pattern instead of being cut"
        )


class TestCompletionVerbThatIsNotACompletion:

    def test_category_noun_after_the_verb_is_not_finished_work(self):
        text = ("Found that any task whose label ends in a file path was "
                "retyped as a FILE node, so it vanished from completed "
                "tasks and appeared as a spurious file.")
        assert not any("appeared as a spurious file" in t for t in _done(text)), (
            "'completed tasks' is a noun phrase; the session is talking "
            "about the category, not reporting work"
        )

    def test_measurement_sentence_is_not_a_task(self):
        text = ("Yes — completed tasks F1 dropped from 77 to 75 and "
                "decisions from 63 to 60.")
        assert not _done(text), _done(text)

    def test_wip_clause_does_not_yield_a_completed_task(self):
        text = ("Working on a CI step that runs the image with the network "
                "removed to prove the bake worked.")
        assert not any("prove the bake" in t for t in _done(text)), (
            "the clause already said this is not done"
        )

    def test_a_real_completion_still_lands(self):
        """The guards must not cost recall on ordinary completions."""
        assert _done("Completed: rate limiting with slowapi on every route")
        assert _done("Removed the dependency from go.mod")
        assert any("network" in t.lower()
                   for t in _done("Removed the network from the test container"))


class TestTechNameInsideALongerIdentifier:

    def test_react_lazy_is_not_a_decision_to_use_react(self):
        labels = _decisions(
            "Decided: code splitting with React.lazy for the chart routes.")
        assert "Use React" not in labels, labels
        assert any("code splitting" in x for x in labels)

    def test_a_versioned_name_is_still_a_decision(self):
        """Only a dot is guarded: `postgres-15` is how a version is
        written, and that IS the choice."""
        assert any("postgres" in x.lower()
                   for x in _decisions("Decided: postgres-15 for the primary store"))


class TestLabelQualityMetric:

    def test_truncation_is_checked_against_the_source(self):
        source = "Added a data_loss_detected flag so destroyed memory is queryable"
        cut = "Added a data_loss_detected flag so destroyed memory is queryab"
        clean = "Added a data_loss_detected flag so destroyed memory is queryable"

        assert label_quality([cut], source).truncated == 1
        assert label_quality([clean], source).truncated == 0

    def test_a_long_label_that_simply_ends_in_a_word_is_not_truncated(self):
        source = "Fixed the backfill timeout by batching at 10k rows in scripts/backfill.py."
        label = "Fixed the backfill timeout by batching at 10k rows in scripts/backfill.py"

        assert label_quality([label], source).truncated == 0, (
            "the old shape-based guess counted this; it is a complete label"
        )

    def test_labels_from_different_sessions_are_not_duplicates(self):
        a, b = ["persistence.py"], ["persistence.py"]

        assert label_quality(a + b, groups=[a, b]).near_duplicates == 0
        # ...and pooling them, which is what it used to do, does count it,
        # so this test fails loudly if the grouping is ever dropped.
        assert label_quality(a + b).near_duplicates == 1

    def test_a_real_duplicate_inside_one_bucket_is_still_counted(self):
        """The fix must not turn the metric into one that never fires."""
        bucket = ["Use PostgreSQL", "Use PostgreSQL for refresh tokens"]

        assert label_quality(bucket, groups=[bucket]).near_duplicates == 1

    def test_an_error_and_its_fix_are_exempt(self):
        bucket = ["Fixed the fixture teardown race", "race in the fixture teardown"]
        exempt = {frozenset(bucket)}

        assert label_quality(bucket, groups=[bucket]).near_duplicates == 1
        assert label_quality(bucket, groups=[bucket],
                             exempt=exempt).near_duplicates == 0


@pytest.mark.parametrize("category", ["completed_tasks", "decisions", "files", "errors"])
def test_corpus_targets_hold(category):
    """The roadmap's numeric targets, pinned so a regression is a failing
    test rather than a number someone has to re-read."""
    from benchmarks.eval import corpus as corpus_mod
    from benchmarks.eval.__main__ import evaluate

    result = evaluate(corpus_mod.load(None))
    q = result["labels"]

    assert q.truncated_pct < 5, f"{q.truncated} labels cut mid-word"
    assert q.near_duplicates < 15, f"{q.near_duplicates} near-duplicate pairs"
    assert result["micro"][category]["precision"] >= 0.90, (
        f"{category} precision fell to {result['micro'][category]['precision']:.0%}"
    )
