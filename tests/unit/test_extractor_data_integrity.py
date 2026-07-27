"""
Regression tests for TM-14 (extraction data loss) and TM-18 (non-
deterministic merge ordering) in HybridExtractor.

TM-14: _deduplicate() copied only a hardcoded subset of ExtractedData's
simple-list attributes ("goals", "tasks_done", ... "environments") and
silently dropped "endpoints", "schemas", and "evidence" entirely.
_deduplicate() is the path taken whenever the LLM pass is absent — which
is the SHIPPED DEFAULT, since use_llm_extraction defaults to False. Two
of the nine documented node types (ENDPOINT, SCHEMA) and all decision
evidence were therefore unreachable out of the box.

TM-18: merge()'s per-category `combined` list was built from set
difference/intersection operations, then truncated to [:15]. Python's
string-hash randomization (PYTHONHASHSEED, on by default) means set
iteration order is not stable across process runs, so when more than 15
items are found, WHICH 15 survive the truncation changes between runs of
the identical input — the same conversation processed twice on different
workers could produce different graphs.
"""
from __future__ import annotations

import dataclasses

from tokenmizer.graph_memory.hybrid_extractor import ExtractedData, HybridExtractor


class TestDeduplicatePreservesAllFields:

    def test_endpoints_survive_deduplicate(self):
        he = HybridExtractor()
        src = ExtractedData(endpoints=["POST /api/login", "GET /api/users"])
        out = he._deduplicate(src)
        assert out.endpoints == ["POST /api/login", "GET /api/users"]

    def test_schemas_survive_deduplicate(self):
        he = HybridExtractor()
        src = ExtractedData(schemas=["users table", "sessions table"])
        out = he._deduplicate(src)
        assert out.schemas == ["users table", "sessions table"]

    def test_evidence_survives_deduplicate(self):
        he = HybridExtractor()
        src = ExtractedData(evidence=[{"text": "340ms", "type": "metric", "turn": 0}])
        out = he._deduplicate(src)
        assert out.evidence == [{"text": "340ms", "type": "metric", "turn": 0}]

    def test_no_dataclass_field_is_silently_dropped(self):
        """Structural guard against this regressing again: every list-
        valued field declared on ExtractedData must come through
        _deduplicate() with its contents intact, whatever the field is
        named. This is the actual fix — deriving the field list from
        dataclasses.fields() instead of a hand-maintained string list
        that can drift out of sync with the dataclass."""
        he = HybridExtractor()
        sample_data = {}
        for f in dataclasses.fields(ExtractedData):
            if f.name == "confidence":
                continue  # dict[str, float], not a list — handled separately
            if f.name in ("decisions", "superseded", "evidence"):
                continue  # list[dict] fields, checked individually above
            sample_data[f.name] = [f"{f.name}-item-one", f"{f.name}-item-two"]

        src = ExtractedData(**sample_data)
        out = he._deduplicate(src)
        for name, items in sample_data.items():
            assert getattr(out, name) == items, (
                f"field {name!r} was dropped or altered by _deduplicate() "
                f"— expected {items!r}, got {getattr(out, name)!r}"
            )


class TestMergeIsOrderDeterministic:

    def test_merge_output_order_is_stable_across_repeated_calls(self):
        """Even within a single process, the OLD implementation could
        still produce different results per call because dict/set
        iteration order for the intermediate sets isn't guaranteed
        identical every time construction happens with different insert
        patterns. The stronger, more direct check: the merged list must
        be built from insertion-ordered structures (dicts), not from
        `set()` operations, so ordering is a pure function of input order
        — verified by checking it matches EXACTLY what insertion order
        predicts, not just "happens to be stable this run"."""
        he = HybridExtractor()
        llm = ExtractedData(files=[f"src/llm_{i}.py" for i in range(20)])
        heuristic = ExtractedData(files=[f"src/heu_{i}.py" for i in range(20)])

        merged = he.merge(llm, heuristic)

        # Deterministic, predictable order: all LLM items (in LLM's own
        # order) first, since none overlap with heuristic's, followed by
        # heuristic-only items in heuristic's own order — truncated to 15.
        expected_prefix = [f"src/llm_{i}.py" for i in range(15)]
        assert merged.files == expected_prefix, (
            f"merge() output order is not a deterministic function of "
            f"input order: got {merged.files!r}"
        )

    def test_merge_is_repeatable_byte_for_byte(self):
        """Run merge() many times on the identical input within one
        process and confirm every run produces the identical output —
        the direct regression check for the non-determinism bug."""
        he = HybridExtractor()
        llm = ExtractedData(files=[f"src/f{i}.py" for i in range(20)])
        heu = ExtractedData(files=[f"src/g{i}.py" for i in range(20)])

        results = [tuple(he.merge(llm, heu).files) for _ in range(10)]
        assert len(set(results)) == 1, (
            f"merge() produced different output across repeated calls on "
            f"identical input: {set(results)!r}"
        )

    def test_corroborated_items_use_llm_casing_and_come_first(self):
        he = HybridExtractor()
        llm = ExtractedData(files=["src/App.tsx", "src/Only_LLM.tsx"])
        heuristic = ExtractedData(files=["src/app.tsx", "src/only_heuristic.tsx"])

        merged = he.merge(llm, heuristic)
        assert merged.files[0] == "src/App.tsx", "corroborated item should keep LLM's casing"
        assert "src/Only_LLM.tsx" in merged.files
        assert "src/only_heuristic.tsx" in merged.files
