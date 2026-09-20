"""
`semantic_retrieval: auto` — on when the model is actually there.

It used to default to `false`, so a deployment with the embedding model
sitting in its image ranked context by token overlap anyway unless someone
found the flag. Measured on the retrieval eval, keyword ranking gets
recall@6 **82% over 40 paraphrased questions**, and the misses are exactly
the paraphrases embeddings exist for ("what is slow about the dashboard"
against a node that says "re-render").

The one thing that must not happen is assuming the capability is present.
sentence-transformers ships no weights: the package being installed says
nothing about whether the model loads on an air-gapped host, behind an
egress proxy, or during a Hub outage. "auto" therefore asks by trying, and
it asks once at startup rather than on every request.

(The 92%-with-embeddings figure quoted in the docs predates this branch
and was NOT reproduced here — the sandbox this was written in has no
egress, so the model never loaded. The keyword number is the one measured
today, and the docs say so.)
"""
from __future__ import annotations

import pytest

from tokenmizer.config.settings import Settings, resolve_semantic_retrieval


class _Engine:
    def __init__(self, available: bool):
        self.available = available


class TestAutoAsksRatherThanAssumes:

    def test_auto_is_the_default(self):
        assert Settings().graph_checkpoint.semantic_retrieval == "auto"

    def test_auto_is_on_when_the_model_loads(self, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get",
                            classmethod(lambda cls: _Engine(True)))

        assert resolve_semantic_retrieval("auto") is True

    def test_auto_is_off_when_the_model_does_not_load(self, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get",
                            classmethod(lambda cls: _Engine(False)))

        assert resolve_semantic_retrieval("auto") is False, (
            "an installed package is not a loaded model"
        )

    def test_auto_is_off_when_asking_itself_fails(self, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod

        def boom(cls):
            raise RuntimeError("no such module")

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get", classmethod(boom))

        assert resolve_semantic_retrieval("auto") is False, (
            "a broken capability check must never turn the feature on"
        )

    @pytest.mark.parametrize("pinned", [True, False])
    def test_an_explicit_value_is_obeyed_without_asking(self, pinned, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod

        def must_not_be_called(cls):
            raise AssertionError("a pinned setting must not consult the model")

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get",
                            classmethod(must_not_be_called))

        assert resolve_semantic_retrieval(pinned) is pinned


class TestTheInProcessMemoryAgrees:
    """An agent using `Memory` directly must not get quietly worse ranking
    than the same deployment's proxy."""

    def test_memory_follows_the_configured_default(self, tmp_path, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod
        from tokenmizer.agents import Memory

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get",
                            classmethod(lambda cls: _Engine(True)))

        memory = Memory("auto-default", storage_dir=str(tmp_path))

        assert memory._graph.semantic_retrieval is True

    def test_an_explicit_argument_still_wins(self, tmp_path, monkeypatch):
        import tokenmizer.semantic_cache.cache as cache_mod
        from tokenmizer.agents import Memory

        monkeypatch.setattr(cache_mod.EmbeddingEngine, "get",
                            classmethod(lambda cls: _Engine(True)))

        memory = Memory("pinned-off", storage_dir=str(tmp_path),
                        semantic_retrieval=False)

        assert memory._graph.semantic_retrieval is False


class TestTheEvalIsBigEnoughToMeanSomething:

    def test_forty_cases_not_thirteen(self):
        from benchmarks.graph_retrieval.query_eval import CASES

        assert len(CASES) >= 40, (
            "n=13 is too small to justify changing a default; the number "
            "moves by 8 points if one case flips"
        )

    def test_every_case_is_answerable_from_its_transcript(self):
        """An ungrounded case measures extraction, not retrieval, and reads
        as a retrieval failure forever."""
        from benchmarks.graph_retrieval.query_eval import validate_grounding

        assert validate_grounding() == []

    def test_the_questions_are_paraphrases(self):
        """If the question uses the node's own words, the person would not
        have needed to ask. Those are the cases token overlap cannot
        serve, so they are the ones worth measuring."""
        from benchmarks.graph_retrieval.query_eval import CASES

        literal = [(q, expected) for _s, q, expected in CASES
                   if expected.lower() in q.lower()]
        assert not literal, f"these questions quote the answer: {literal}"
