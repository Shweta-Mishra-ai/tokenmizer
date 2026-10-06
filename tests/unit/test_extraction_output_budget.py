"""The LLM extraction call's output budget, and a reply cut off by it (#89).

The call used to hard-code max_tokens=800. A model that reasons before it
answers spends part of that on reasoning, so the JSON that followed was cut
mid-structure, json.loads rejected all of it, and the whole LLM pass failed
silently on every call. The budget is now a setting, and a cut-off reply
keeps the fields that arrived intact.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.config.settings import GraphCheckpointSettings
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.hybrid_extractor import (
    DEFAULT_EXTRACTION_MAX_TOKENS,
    HybridExtractor,
    _parse_json_object,
)

FULL = json.dumps({
    "goals": ["ship v1"],
    "tasks_done": ["set up the database"],
    "decisions": [{"label": "Use PostgreSQL", "reason": "needs transactions"}],
    "files": ["src/db.py"],
})
MESSAGES = [{"role": "user", "content": "we decided on postgres"}]


class TestTruncatedReplyKeepsCompleteFields:

    def test_cut_inside_an_array_keeps_the_finished_fields(self):
        assert _parse_json_object(FULL[:60]) == {
            "goals": ["ship v1"],
            "tasks_done": ["set up the database"],
        }

    def test_cut_inside_a_string_value_never_completes_the_value(self):
        cut = FULL.index("PostgreSQL") + 4          # '... "Use Post'
        data = _parse_json_object(FULL[:cut])
        assert data is not None
        assert "Post" not in json.dumps(data)
        assert data["goals"] == ["ship v1"]

    def test_cut_after_a_finished_decision_label_keeps_the_label(self):
        cut = FULL.index('"reason"') + 3            # inside the reason key
        data = _parse_json_object(FULL[:cut])
        assert data["decisions"] == [{"label": "Use PostgreSQL"}]

    def test_escaped_quotes_and_braces_inside_strings_do_not_confuse_it(self):
        assert _parse_json_object('{"goals": ["say \\"hi\\"", "a } b", "tail') == {
            "goals": ["say \"hi\"", "a } b"]
        }

    def test_fenced_reply_cut_off_is_recovered(self):
        raw = '```json\n{"goals": ["x"], "tasks_done": ["y", "z'
        assert _parse_json_object(raw) == {"goals": ["x"], "tasks_done": ["y"]}

    def test_recovery_is_logged_so_the_cause_is_visible(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tokenmizer.graph_memory.hybrid_extractor"):
            _parse_json_object(FULL[:60])
        assert any("cut off" in r.message for r in caplog.records)

    @pytest.mark.parametrize("raw", [
        "let me think about the facts...",     # reasoning only, no object at all
        "{",
        '{"goals":',
        '{"a": 1,}',                           # complete but malformed: not truncation
        "[1, 2]",
        "",
    ])
    def test_nothing_recoverable_is_none(self, raw):
        assert _parse_json_object(raw) is None

    def test_a_complete_reply_is_untouched_and_not_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tokenmizer.graph_memory.hybrid_extractor"):
            assert _parse_json_object(FULL) == json.loads(FULL)
        assert not caplog.records


class TestOutputBudget:

    async def test_default_budget_is_no_longer_800(self):
        seen = {}

        async def provider(messages, system="", max_tokens=0):
            seen["max_tokens"] = max_tokens
            return {"text": FULL}

        await HybridExtractor().llm_extract(MESSAGES, provider)
        assert seen["max_tokens"] == DEFAULT_EXTRACTION_MAX_TOKENS
        assert seen["max_tokens"] > 800

    async def test_configured_budget_reaches_the_provider_call(self):
        seen = {}

        async def provider(messages, system="", max_tokens=0):
            seen["max_tokens"] = max_tokens
            return {"text": FULL}

        await HybridExtractor(max_output_tokens=4096).llm_extract(MESSAGES, provider)
        assert seen["max_tokens"] == 4096

    async def test_truncated_reply_still_yields_extracted_data(self):
        async def provider(messages, system="", max_tokens=0):
            return {"text": FULL[:FULL.index("PostgreSQL") + 4]}

        data = await HybridExtractor().llm_extract(MESSAGES, provider)
        assert data is not None, "a cut-off reply discarded the whole LLM pass"
        assert data.goals == ["ship v1"]
        assert data.tasks_done == ["set up the database"]

    def test_setting_defaults_to_the_extractor_default_and_is_bounded(self):
        assert GraphCheckpointSettings().extraction_max_tokens == DEFAULT_EXTRACTION_MAX_TOKENS
        with pytest.raises(ValueError):
            GraphCheckpointSettings(extraction_max_tokens=100)
        with pytest.raises(ValueError):
            GraphCheckpointSettings(extraction_max_tokens=100_000)

    async def test_the_proxy_sends_the_configured_budget_to_the_provider(
        self, tmp_path, monkeypatch
    ):
        """End to end through _update_graph: the setting reaches the provider
        call, and a reply cut off by it still lands in the graph."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "use_llm_extraction", True)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "extraction_max_tokens", 3000)
        seen = {}

        class _CutOffProvider:
            async def chat(self, **kwargs):
                seen["max_tokens"] = kwargs["max_tokens"]

                class R:
                    text = FULL[:FULL.index("PostgreSQL") + 4]
                return R()

        monkeypatch.setattr(app_module, "_get_extraction_provider", lambda: _CutOffProvider())

        graph = GraphMemory("budget-e2e", storage_dir=str(tmp_path))
        raw = [{"role": "user", "content": "Use PostgreSQL for the primary datastore please"}]
        await app_module._update_graph(
            "budget-e2e", graph, raw, [dict(m) for m in raw],
            "claude-sonnet-4-6", {}, "irrelevant query",
        )
        await asyncio.gather(*list(app_module._background_tasks), return_exceptions=True)

        assert seen["max_tokens"] == 3000
        labels = {n.label for n in graph._nodes.values()}
        # "set up the database" appears only in the cut-off LLM reply, never
        # in the user's message: its presence proves the recovered fields
        # reached the graph.
        assert "set up the database" in labels, labels
