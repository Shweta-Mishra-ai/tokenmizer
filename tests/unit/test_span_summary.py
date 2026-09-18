"""
The note for turns windowing drops.

Once a session crosses `memory.max_tokens_before_summary`, every turn
older than the protected tail is replaced by the graph's context block.
Everything the ontology captured survives that — it is a node. Everything
else leaves the session permanently: a budget, a deadline, a licence
restriction, a latency target. Each is one sentence, said once, and none
of them is a task, a decision, a file or an error.

Measured by `benchmarks/resume_quality/runner.py`: out-of-ontology
retention 17% -> 100% on its fixtures, +23 tokens of resume block per
session, and no section lost on the captured transcripts in the eval
corpus. The fixtures were written by the same person as the selector, so
they show the mechanism works end to end and not that it generalises —
the runner says so too.

What is pinned here is the behaviour that makes it safe:
- the selector keeps facts, not chatter
- it never restates something the graph already holds
- there is one node per session, rewritten rather than accumulated
- nothing here can fail a chat request
"""
from __future__ import annotations

import pytest

from tokenmizer.compression.window import SmartMessageWindow
from tokenmizer.graph_memory.graph import GraphMemory, NodeType
from tokenmizer.graph_memory.summary import select_sentences


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("span-summary", storage_dir=str(tmp_path))


def _msgs(*contents: str) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c}
            for i, c in enumerate(contents)]


class TestSelector:

    @pytest.mark.parametrize("sentence,expect", [
        ("Keep the JS bundle under 500KB gzipped for the whole app", True),
        ("The p99 must stay under 120ms on every pricing route", True),
        ("We cannot ship anything AGPL licensed, legal signed off on MIT", True),
        ("Everything has to run in eu-west-1 for data residency reasons", True),
        # No figure, no rule — the graph's own sections cover this shape.
        ("I had a look at the resolver and it seems reasonable enough", False),
        ("Let me know when you want me to start on the next part", False),
    ])
    def test_a_sentence_is_kept_only_if_it_carries_a_fact(self, sentence, expect):
        kept = select_sentences(_msgs(sentence), known=[])
        assert bool(kept) is expect, kept

    def test_a_sentence_the_graph_already_holds_is_dropped(self):
        sentence = "We must use bcrypt with cost factor 12 for password hashing"

        assert select_sentences(_msgs(sentence), known=[])
        assert not select_sentences(
            _msgs(sentence),
            known=["Use bcrypt with cost factor 12 for password hashing"],
        ), "restating a node spends the resume budget on nothing"

    def test_two_rules_in_one_sentence_are_both_kept(self):
        """A compound sentence states two rules and clipping to one clause
        silently keeps the first. Both are the user's."""
        kept = select_sentences(_msgs(
            "This must be done by the 14th — the demo is that morning — and "
            "nothing ships without the on-call engineer approving it."
        ), known=[])
        joined = " ".join(kept).lower()

        assert "14th" in joined
        assert "on-call" in joined

    def test_the_limit_is_respected(self):
        kept = select_sentences(_msgs(
            "The bundle must stay under 500KB for the main entry point.",
            "The p99 has to be under 120ms across every pricing route.",
            "We cannot use anything AGPL licensed anywhere in the tree.",
            "Everything needs to run in eu-west-1 for data residency.",
            "The demo is due by the 14th and cannot move for any reason.",
        ), known=[], limit=3)

        assert len(kept) == 3


class TestOneNodePerSession:

    def test_a_second_call_rewrites_rather_than_accumulates(self, graph):
        span = _msgs(
            "Keep the bundle under 500KB and we cannot ship AGPL code.",
            "Understood, starting now.",
            "The p99 must stay under 120ms as well.",
            "Noted.",
        )
        first = graph.record_span_summary(span)
        assert first

        graph._summarised_turns = 0        # force a re-selection
        second = graph.record_span_summary(span + _msgs(
            "Everything has to run in eu-west-1 for data residency.",
            "Understood.",
        ))

        assert second == first, "the span is a prefix that only extends"
        summaries = [n for n in graph._nodes.values() if n.type == NodeType.SUMMARY]
        assert len(summaries) == 1
        assert "eu-west-1" in summaries[0].label

    def test_nothing_worth_keeping_creates_no_node(self, graph):
        graph.record_span_summary(_msgs(
            "how is it going", "Fine, still working through the list.",
            "ok thanks", "No problem at all.",
        ))

        assert not [n for n in graph._nodes.values() if n.type == NodeType.SUMMARY]

    def test_an_empty_span_is_a_no_op(self, graph):
        assert graph.record_span_summary([]) is None

    def test_the_node_survives_the_validator(self, graph):
        """A SUMMARY is a verbatim record, not a claim the extractor
        inferred, so it must not be scored as one — the validator rejects
        prose for being prose, which is exactly what this node is."""
        node_id = graph.record_span_summary(_msgs(
            "The p99 must stay under 120ms on the pricing path.",
            "Understood.",
            "We cannot use anything AGPL licensed in this repo.",
            "Noted.",
        ))

        assert node_id, "the note was rejected by the extraction validator"


class TestItReachesTheResumeBlock:

    def test_the_note_appears_in_the_context_block(self, graph):
        graph.extract_from_messages(_msgs(
            "Building a pricing service.",
            "Decided: Postgres. Completed: the schema in migrations/001.sql.",
        ), incremental=False)
        graph.record_span_summary(_msgs(
            "The p99 must stay under 120ms and it has to run in eu-west-1.",
            "Understood.",
            "We cannot ship anything AGPL licensed.",
            "Noted.",
        ))

        block = graph.to_context_block(token_budget=400)

        assert "Noted:" in block
        assert "120ms" in block
        # ...and it has not displaced what was already there.
        assert "Decided:" in block
        assert "Files:" in block


class TestWindowingIsNeverFailedByIt:

    def test_a_broken_summary_does_not_break_windowing(self, graph, monkeypatch):
        """The caller came for an answer. A failure here is logged and the
        turn continues; the windowing below is correct without it."""
        def boom(*a, **k):
            raise RuntimeError("selector exploded")

        monkeypatch.setattr(GraphMemory, "record_span_summary", boom)

        long_conversation = _msgs(*[f"turn {i} with enough words to matter here"
                                    for i in range(40)])
        windowed, saved = SmartMessageWindow(
            token_budget=50, protect_recent=6).apply(long_conversation, graph)

        assert len(windowed) < len(long_conversation)
        assert saved > 0
