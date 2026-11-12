"""A keyword gate may skip a regex only where the regex cannot match.

_PREFILTER lists, per pattern, words at least one of which every match
contains. A wrong list does not fail loudly: it silently stops extracting.
"""
from __future__ import annotations

import pytest

from benchmarks.eval.corpus import load
from tokenmizer.graph_memory import hybrid_extractor as H
from tokenmizer.graph_memory.patterns import mask_mentions

GATED = [p for p in vars(H).values()
         if hasattr(p, "finditer") and id(p) in H._PREFILTER]


def _texts():
    for corpus in (None, "benchmarks/eval/corpus_domains"):
        for s in load(corpus):
            for m in s.messages:
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    yield c
                    yield mask_mentions(c)


HEADER_LINES = [
    f"{lead}{word}: {tail}"
    for word in ("TODO", "To-Do", "to do", "Next steps", "Next up", "Up next", "Remaining",
                 "Open items", "Follow-ups", "Backlog", "Outstanding", "Not yet started",
                 "Action items", "Still needed", "Deferred", "Next tasks", "Queued", "Planned",
                 "Tomorrow", "Later", "Still open", "Upcoming", "Scheduled", "Parked",
                 "On hold", "Wishlist", "Pendientes", "Por hacer", "A faire", "À faire",
                 "Offen", "Ausstehend", "Baaki", "Pending hai",
                 "Decision", "Tech choice", "Approach", "Stack", "Agreed", "Final call",
                 "Verdict", "Conclusion", "Chosen", "Our pick", "Selection", "Decisión",
                 "Décision", "Entscheidung", "Faisla", "Going with", "Settled",
                 "Locked in", "Going forward", "Consensus")
    for lead in ("", "- ", "**")
    for tail in ("write the migration guide for orders",)
] + [f"{box} write the migration guide" for box in
     ("- [x]", "- [X]", "[done]", "[completed]", "[fixed]", "[shipped]", "☒", "✅",
      "✔️", "✓", "- [ ]", "[todo]", "[to do]", "[pending]", "[wip]", "[open]",
      "☐", "⬜")]


# Sentences for gated patterns the corpora never trigger, so that every
# gate is checked against at least one real match.
PROSE = [
    "the importer keeps failing on large files",
    "The checkout service broke after the deploy.",
    "The full import took 40 minutes instead of 5.",
    "The login page is working now.",
    "The export endpoint is ready.",
    "Rate limiting is done.",
    "The refund flow is implemented and the tests are passing.",
]


@pytest.mark.parametrize("pattern", GATED, ids=lambda p: p.pattern[:40])
def test_the_gate_never_skips_a_match(pattern):
    matched = 0
    for text in [*_texts(), *HEADER_LINES, *PROSE]:
        if pattern.search(text):
            matched += 1
            assert H._may_match(pattern, text.lower()), (pattern.pattern[:80], text[:200])
    assert matched, "no test text matches this pattern, so its gate is unchecked"
