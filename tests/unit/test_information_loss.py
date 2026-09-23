"""
TokenMizer may drop tokens. It may not drop information.

Every layer here is lossy on purpose — that is the product — so the line
that matters is *what* it is allowed to lose. These tests hold that line
at the two places an audit found it crossed, both silently and both on
the default configuration.

The shape of both bugs is the same, and it is the shape this file exists
to catch: a rule that matches on a PREFIX or a SHAPE, applied as if it
had matched on meaning.

  - `OutputTrimmer` matched filler by shape and deleted it wherever the
    shape occurred — inside a code fence, mid-sentence, and over a real
    question about real state.
  - `GraphMemory._msg_hash` identified a message by its first 500
    characters, so a reply sharing a preamble with an earlier one was
    "already processed" and never extracted.

Both had already been found and fixed ELSEWHERE in this codebase — the
prompt-side `_FILLER` list is anchored to a sentence start for exactly
this reason, and `RepetitiveHistoryPruner` stopped keying on the first 60
characters after it was caught deleting whole replies. Neither fix was
carried to these two. That is what this file is really pinning: the same
mistake, in the places it had not yet been made twice.
"""
from __future__ import annotations

import tempfile

import pytest

from tokenmizer.compression.output_trimmer import OutputTrimmer
from tokenmizer.graph_memory.graph import GraphMemory


@pytest.fixture
def trimmer():
    return OutputTrimmer()


# ── The response path ────────────────────────────────────────────────────────

class TestTheTrimmerNeverEditsCode:
    """The trimmer is the only stage that rewrites text a person reads
    after the model wrote it. A deletion here is invisible: there is
    nothing to compare the answer against."""

    FIXTURE = (
        "Here is the transcript parser test fixture:\n\n"
        "```python\n"
        'TRANSCRIPT = """\n'
        "Sure! I can help with that.\n"
        "The balance is 42.\n"
        '"""\n'
        "```\n\n"
        "Use it in test_parse()."
    )

    @pytest.mark.parametrize("level", ["lite", "full", "ultra"])
    def test_a_filler_phrase_inside_a_fenced_block_is_not_touched(
            self, trimmer, level):
        """`^` with re.MULTILINE matched the start of any line, so this
        deleted `Sure! ` from inside a Python string literal — corrupting
        code the reader then pastes."""
        out, _ = trimmer.trim(self.FIXTURE, level)
        assert "Sure! I can help with that." in out
        assert out == self.FIXTURE

    def test_an_inline_code_span_is_not_touched(self, trimmer):
        text = "Run `Sure.sh --check` before the deploy, then re-run the suite."
        out, _ = trimmer.trim(text, "ultra")
        assert out == text


class TestTheTrimmerNeverEditsProse:
    @pytest.mark.parametrize("level", ["lite", "full", "ultra"])
    def test_a_real_sentence_beginning_with_a_filler_word_survives(
            self, trimmer, level):
        """"Sure," opening a mid-answer sentence is a concession, not a
        pleasantry. This used to lose its first word."""
        text = ("Two options exist.\n\n"
                "Sure, the second is slower, but it is correct under "
                "concurrency.\n\nPick the second.")
        assert trimmer.trim(text, level)[0] == text

    def test_an_opening_pleasantry_is_still_removed(self, trimmer):
        """The fix must not cost the feature: at the very start of the
        response, it is still filler."""
        text = ("Certainly! The answer is to use a connection pool.\n\n"
                "It bounds concurrency.")
        out, saved = trimmer.trim(text, "full")
        assert not out.startswith("Certainly")
        assert "connection pool" in out
        assert saved > 0


class TestTheTrimmerNeverDeletesAQuestionThatWasAsked:
    def test_a_closing_question_about_real_state_survives(self, trimmer):
        """"Is there anything else in the 003 batch you want rolled
        back?" matched the generic sign-off pattern and was deleted, so
        the reader never saw that they had been asked anything."""
        text = ("I applied the migration.\n\n"
                "Is there anything else in the 003 batch you want rolled back?")
        assert trimmer.trim(text, "full")[0] == text

    def test_a_sign_off_naming_a_file_survives(self, trimmer):
        text = "Done.\n\nLet me know if you need the same change in settings.py too."
        assert trimmer.trim(text, "full")[0] == text

    @pytest.mark.parametrize("text", [
        "The migration is applied.\n\nIs there anything else I can help with?",
        "The answer is to use a connection pool.\n\nLet me know if you need anything else!",
        "The answer is to use a connection pool for this workload.\n\nHope this helps!",
    ])
    def test_a_genuinely_generic_sign_off_is_still_removed(self, trimmer, text):
        out, saved = trimmer.trim(text, "full")
        assert len(out) < len(text) and saved > 0


class TestUltraDoesNotDeleteTheNumbers:
    def test_a_summary_paragraph_carrying_the_measurements_survives(self, trimmer):
        """On `ultra` the "In summary, ..." rule deleted the only
        paragraph in the answer that held any numbers."""
        text = ("We measured three runs.\n\n"
                "In summary, p99 went from 120ms to 38ms and error rate "
                "from 2.1% to 0.03%.\n\nDetails above.")
        assert trimmer.trim(text, "ultra")[0] == text

    def test_a_contentless_restatement_is_still_removed_on_ultra(self, trimmer):
        text = ("Use a connection pool, sized to the worker count.\n\n"
                "In summary, use a connection pool.\n\n"
                "That is the whole change.")
        out, _ = trimmer.trim(text, "ultra")
        assert "In summary" not in out
        assert "sized to the worker count" in out


# ── The extraction path ──────────────────────────────────────────────────────

# Long enough that the part which differs falls beyond the old 500-char
# hash window — which is the normal shape of an assistant reply, not a
# contrived one.
_PREAMBLE = (
    "Here is the updated module. I kept the existing structure and only "
    "changed the parts we discussed, so the diff stays small and "
    "reviewable. The imports are unchanged, the public API is unchanged, "
    "and the tests should still pass without edits. Full file below for "
    "convenience, copy it over the old one. Note the config block near "
    "the top is the only thing that moved. "
) * 2


def _graph(name: str) -> GraphMemory:
    return GraphMemory(name, storage_dir=tempfile.mkdtemp())


class TestASharedPreambleDoesNotSwallowATurn:
    def test_two_replies_sharing_an_opening_are_two_messages(self):
        assert len(_PREAMBLE) > 500, "fixture must exceed the old hash window"
        graph = _graph("prefix")

        a = graph._msg_hash({"content": _PREAMBLE + "Decided: use PostgreSQL."})
        b = graph._msg_hash({"content": _PREAMBLE + "Decided: use Redis."})

        assert a != b, (
            "hashing only the first 500 characters made two different "
            "replies the same message"
        )

    def test_the_second_decision_is_still_extracted_a_turn_later(self):
        """The live shape: the proxy calls extract_from_messages() once
        per request, so the second reply arrives in a LATER call, after
        the first one's hash is already in _processed_hashes. The whole
        turn was filtered out as already-processed and its decision was
        lost permanently, with nothing logged."""
        graph = _graph("prefix-turns")
        graph.extract_from_messages([{
            "role": "assistant",
            "content": _PREAMBLE + "Decided: use PostgreSQL for the order storage layer.",
        }])
        graph.extract_from_messages([{
            "role": "assistant",
            "content": _PREAMBLE + "Decided: use Redis for the session token store.",
        }])

        labels = {n.label for n in graph._nodes.values()}
        assert any("PostgreSQL" in x for x in labels)
        assert any("Redis" in x for x in labels), (
            f"the second decision was dropped as a duplicate: {labels}"
        )

    def test_a_genuinely_repeated_message_is_still_deduped(self):
        """The fix must not cost the feature it paid for: an identical
        message really is the same message."""
        graph = _graph("prefix-dedup")
        message = {"role": "assistant",
                   "content": _PREAMBLE + "Decided: use PostgreSQL."}

        graph.extract_from_messages([message])
        before = len(graph._processed_hashes)
        graph.extract_from_messages([message])

        assert len(graph._processed_hashes) == before == 1
