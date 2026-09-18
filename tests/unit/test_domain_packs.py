"""
Domain packs: the same ontology, a different vocabulary.

Goal / task / decision / file / error is a *coding* ontology and every
regex family in `patterns.py` is a coding phrasing, so a research log, an
incident review or a product discussion extracted almost nothing — not
because the shapes are wrong but because nobody in those rooms says
"Decided:" or "Fixed:".

Measured on `benchmarks/eval/corpus_domains`, three labelled sessions:
**macro F1 11% with the coding patterns alone, 96% with the packs.**
Reproduce either with `--ignore-packs`.

The two rules that keep this safe are what these tests pin:

1. A pack only ever ADDS. Its families run after the coding ones, so a
   coding session is bit-for-bit what it was and a research session that
   also names a file still gets the file.
2. An unknown or missing domain degrades to coding, not to nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokenmizer.graph_memory.domains import PACKS, get_pack  # noqa: E402
from tokenmizer.graph_memory.graph import GraphMemory  # noqa: E402
from tokenmizer.graph_memory.hybrid_extractor import (  # noqa: E402
    HybridExtractor,
    get_hybrid_extractor,
)

RESEARCH = [
    {"role": "user", "content":
        "The research question is whether retrieval augmentation beats "
        "fine-tuning on our support corpus."},
    {"role": "assistant", "content":
        "The experiment showed RAG at 71 percent. This contradicts the vendor "
        "benchmark. Concluded that retrieval wins on our corpus."},
]

OPS = [
    {"role": "user", "content":
        "We are seeing an incident on the pricing service, checkout is failing."},
    {"role": "assistant", "content":
        "Error rate is 34 percent. Root cause is connection pool exhaustion. "
        "Rolled back to build 4471."},
]

PRODUCT = [
    {"role": "user", "content":
        "The goal this quarter is to get new teams to their first shared "
        "board within one day."},
    {"role": "assistant", "content":
        "Customers said the invite flow is the drop-off point. We agreed "
        "that onboarding owns the first-run path."},
    {"role": "user", "content": "where are we"},
    {"role": "assistant", "content":
        "Shipped the one-click invite link. In discovery on the template "
        "gallery. Blocked on legal for the shared-link policy."},
]

CODING = [
    # "We're building" and not "Build a": the coding goal openers are a
    # deliberately conservative list and the bare imperative is not on it,
    # because "build the docker image" is a task, not a session goal.
    {"role": "user", "content": "We're building a FastAPI auth service with JWT."},
    {"role": "assistant", "content":
        "Decided: PostgreSQL for storage. Completed: the login endpoint in "
        "api/auth.py. Fixed: the 422 on missing email."},
]


def _count(data) -> int:
    return (len(data.goals) + len(data.decisions) + len(data.tasks_done)
            + len(data.tasks_wip) + len(data.errors))


class TestAPackAddsRecallWhereCodingFindsNothing:

    @pytest.mark.parametrize("pack,messages", [
        ("research", RESEARCH), ("ops", OPS), ("product", PRODUCT),
    ])
    def test_the_pack_finds_much_more_than_the_coding_patterns(self, pack, messages):
        coding = HybridExtractor(domain="coding").heuristic_extract(messages)
        with_pack = HybridExtractor(domain=pack).heuristic_extract(messages)

        assert _count(with_pack) > _count(coding) + 2, (
            f"{pack}: coding found {_count(coding)}, pack found "
            f"{_count(with_pack)} — the pack is not earning its place"
        )

    def test_research(self):
        d = HybridExtractor(domain="research").heuristic_extract(RESEARCH)
        assert any("retrieval augmentation" in g for g in d.goals)
        assert any("retrieval wins" in x["label"] for x in d.decisions)
        assert any("contradicts" in e for e in d.errors), (
            "a result that breaks the story is exactly what a resume must "
            "not drop"
        )

    def test_ops(self):
        d = HybridExtractor(domain="ops").heuristic_extract(OPS)
        assert any("incident" in g.lower() for g in d.goals)
        assert any("Root cause" in x["label"] for x in d.decisions)
        assert any("Error rate" in e for e in d.errors)

    def test_product(self):
        d = HybridExtractor(domain="product").heuristic_extract(PRODUCT)
        assert d.goals
        assert any("Shipped" in t for t in d.tasks_done)
        assert any("Blocked" in e for e in d.errors)


class TestAPackNeverSubtracts:

    def test_a_coding_session_is_unchanged_by_every_pack(self):
        """The families run in addition to the coding ones, so the coding
        result must be a subset of every pack's result."""
        base = HybridExtractor(domain="coding").heuristic_extract(CODING)

        for name in PACKS:
            other = HybridExtractor(domain=name).heuristic_extract(CODING)
            assert set(base.tasks_done) <= set(other.tasks_done), name
            assert set(base.files) <= set(other.files), name
            assert ({x["label"] for x in base.decisions}
                    <= {x["label"] for x in other.decisions}), name

    def test_a_research_session_still_gets_its_files(self):
        d = HybridExtractor(domain="research").heuristic_extract([
            {"role": "assistant", "content":
                "Concluded that the split is wrong. Updated analysis/eval.py."},
        ])
        assert "analysis/eval.py" in d.files


class TestDegradesToCoding:

    @pytest.mark.parametrize("name", [None, "", "  ", "nonsense", "CODING"])
    def test_an_unknown_domain_is_the_coding_pack(self, name):
        pack = get_pack(name)
        assert pack.name == "coding"
        assert pack.decisions == () and pack.errors == ()

    def test_the_extractor_cache_is_per_domain(self):
        assert get_hybrid_extractor("ops") is get_hybrid_extractor("ops")
        assert get_hybrid_extractor("ops") is not get_hybrid_extractor("research")


class TestResumeVocabulary:

    def _block(self, pack, messages, tmp_path):
        g = GraphMemory("vocab", storage_dir=str(tmp_path), domain=pack)
        g.extract_from_messages(messages, incremental=False)
        return g.to_context_block(token_budget=400)

    def test_research_reads_as_research(self, tmp_path):
        block = self._block("research", RESEARCH, tmp_path)
        assert "Question:" in block
        assert "Concluded:" in block
        assert "Contradictions:" in block
        assert "Goal:" not in block and "Decided:" not in block

    def test_ops_reads_as_an_incident(self, tmp_path):
        block = self._block("ops", OPS, tmp_path)
        assert "Incident:" in block
        assert "Symptoms:" in block

    def test_coding_is_untouched(self, tmp_path):
        block = self._block("coding", CODING, tmp_path)
        assert "Goal:" in block
        assert "Decided:" in block


class TestTheCorpusBacksTheClaim:
    """A pack ships with its own labelled corpus or it is a claim rather
    than a measurement — the same rule the coding numbers live under."""

    def _evaluate(self, ignore_packs: bool) -> float:
        from benchmarks.eval import __main__ as ev
        from benchmarks.eval import corpus as corpus_mod

        previous = ev.IGNORE_PACKS
        ev.IGNORE_PACKS = ignore_packs
        try:
            sessions = corpus_mod.load(
                Path(__file__).resolve().parents[2]
                / "benchmarks" / "eval" / "corpus_domains")
            result = ev.evaluate(sessions)
        finally:
            ev.IGNORE_PACKS = previous
        f1s = [m["f1"] for m in result["micro"].values()]
        return sum(f1s) / len(f1s)

    def test_every_pack_has_at_least_one_labelled_session(self):
        from benchmarks.eval import corpus as corpus_mod

        sessions = corpus_mod.load(
            Path(__file__).resolve().parents[2]
            / "benchmarks" / "eval" / "corpus_domains")
        covered = {s.pack for s in sessions}

        missing = set(PACKS) - covered - {"coding"}
        assert not missing, f"packs with no corpus: {sorted(missing)}"

    def test_the_packs_are_what_make_the_difference(self):
        without = self._evaluate(ignore_packs=True)
        with_packs = self._evaluate(ignore_packs=False)

        assert without < 0.35, (
            f"the coding patterns already score {without:.0%} on these "
            f"sessions — the corpus is not testing what it claims to"
        )
        assert with_packs > 0.85, f"packs score only {with_packs:.0%}"
