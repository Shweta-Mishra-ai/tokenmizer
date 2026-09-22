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

from tokenmizer.graph_memory import domains as domains_mod  # noqa: E402
from tokenmizer.graph_memory.domains import (  # noqa: E402
    PACKS,
    get_pack,
    normalize_domain,
)
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

    @pytest.mark.parametrize("name,expected", [
        ("research", "research"), ("  research  ", "research"),
        ("RESEARCH", "research"), ("ReSeArCh", "research"),
        ("ops", "ops"), ("product", "product"),
    ])
    def test_a_valid_name_is_normalised_then_resolved(self, name, expected):
        """The strip/lower was only ever tested on the degrade path, so
        nothing pinned that it resolves a real pack through the same
        normalisation."""
        assert get_pack(name).name == expected

    def test_the_extractor_cache_is_per_domain(self):
        assert get_hybrid_extractor("ops") is get_hybrid_extractor("ops")
        assert get_hybrid_extractor("ops") is not get_hybrid_extractor("research")

    def test_the_cache_key_is_normalised_too(self):
        """Otherwise "ops", " ops" and "OPS" each build and cache their
        own identical extractor."""
        assert get_hybrid_extractor("ops") is get_hybrid_extractor("  OPS  ")


class TestANonStringDomainDoesNotCrash:
    """`GraphMemory` is a public export and `domain=` is one of its
    parameters, so the value is whatever a caller passed. A non-string
    used to reach `.strip()` and raise AttributeError three frames below
    the call that caused it — the same deferred-failure shape the corpus
    loader was fixed for.

    It was in TWO places: `get_pack` and `get_hybrid_extractor` held the
    same `(name or "coding").strip().lower()` expression, and guarding
    only the first left the public call still crashing in the second.
    Both now go through `normalize_domain`.
    """

    @pytest.mark.parametrize("bad", [123, 3.5, True, ["research"], {"a": 1},
                                     object()])
    def test_get_pack_degrades_instead_of_raising(self, bad):
        assert get_pack(bad).name == "coding"

    @pytest.mark.parametrize("bad", [123, ["research"], {"a": 1}])
    def test_normalize_domain_returns_the_coding_key(self, bad):
        assert normalize_domain(bad) == "coding"

    @pytest.mark.parametrize("bad", [123, ["research"]])
    def test_the_extractor_factory_degrades_too(self, bad):
        """The second copy of the expression — this is the one that was
        still raising after get_pack alone was fixed."""
        assert get_hybrid_extractor(bad) is get_hybrid_extractor("coding")

    def test_the_public_graph_api_survives_it(self, tmp_path):
        """End to end through the exported class, which is how a library
        user would actually hit this."""
        g = GraphMemory("bad-domain", storage_dir=str(tmp_path), domain=123)
        g.extract_from_messages([
            {"role": "user", "content": "Building an auth service"},
            {"role": "assistant",
             "content": "Completed: rate limiting with slowapi on every route"},
        ], incremental=False)

        assert any("rate limiting" in n.label.lower()
                   for n in g._nodes.values()), (
            "it must still extract with the coding pack, not just avoid "
            "the crash"
        )

    def test_it_says_so_rather_than_degrading_silently(self, caplog):
        domains_mod._warned_domains.clear()
        with caplog.at_level("WARNING"):
            get_pack(12345)

        assert "domain must be a string" in caplog.text
        assert "12345" in caplog.text, "the offending value names itself"

    def test_it_warns_once_per_value_not_once_per_turn(self, caplog):
        """extract_from_messages() calls get_hybrid_extractor() every
        turn, so warning per call would log the same mistake all session
        for one bad argument passed once at construction."""
        domains_mod._warned_domains.clear()
        with caplog.at_level("WARNING"):
            for _ in range(5):
                get_pack(999)

        assert caplog.text.count("domain must be a string") == 1

    def test_a_value_whose_repr_raises_is_still_handled(self):
        """The warn-once key is built from repr(), so a __repr__ that
        throws must not turn a degraded fallback into a crash."""
        class _Hostile:
            def __repr__(self):
                raise RuntimeError("no repr for you")

        domains_mod._warned_domains.clear()
        assert get_pack(_Hostile()).name == "coding"


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
