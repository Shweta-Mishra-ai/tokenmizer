"""
Regression guard for the hybrid_extractor.py / patterns.py split.

patterns.py holds the regex vocabulary and small pure text-analysis
helpers hybrid_extractor.py's heuristic pass applies — extracted purely
for file size (hybrid_extractor.py was 1651 lines), zero behavior
change. hybrid_extractor.py re-imports everything it actually uses, so
external code that did `from tokenmizer.graph_memory.hybrid_extractor
import _clip` before the split must keep working unchanged.

The real regression guard for "did the split change extraction
behavior" is tests/unit/test_extraction_quality.py's F1 numbers, not
this file — this file only guards the two things a file-move can get
wrong that F1 numbers wouldn't directly point at: a broken re-export,
or the split producing two disagreeing copies of the same name.
"""
from __future__ import annotations

from tokenmizer.graph_memory import hybrid_extractor, patterns


def test_hybrid_extractor_still_exposes_moved_names_by_bare_import():
    """The one name external code actually imports directly (see
    test_extraction_quality.py) must still resolve through
    hybrid_extractor, not just through patterns."""
    from tokenmizer.graph_memory.hybrid_extractor import _clip
    assert _clip is patterns._clip


def test_no_split_brained_duplicate_definitions():
    """Every pattern name hybrid_extractor imports from patterns must be
    the SAME object, not a second, independently-recompiled copy — that
    would silently double memory and let the two drift if one were ever
    edited without the other."""
    shared_names = [
        "_DECISION", "_DECISION_FOR", "_FILE_PATH", "_ERROR_TYPED",
        "_TASK_DONE", "_SUPERSEDED", "EXTRACTION_SYSTEM", "_clip",
    ]
    for name in shared_names:
        assert getattr(hybrid_extractor, name) is getattr(patterns, name), name
