"""
Coverage gaps flagged on PR #64 by an automated review bot, checked one
by one rather than accepted or dismissed wholesale.

Two of the four ("IGNORE_PACKS", "Session.pack") turned out to already
be exercised — `test_domain_packs.py::TestTheCorpusBacksTheClaim`
toggles `IGNORE_PACKS` and measures the F1 difference, and asserts every
pack has a session carrying it. This file does not duplicate those; it
adds what neither that file nor `test_label_quality.py` actually
covered:

- `extract()` itself was only ever exercised indirectly, through
  `evaluate()` over the whole corpus. Its SECOND return value — the
  FIXES-edge label pairs `label_quality`'s near-duplicate check exempts
  — had no assertion anywhere pinning that it is populated correctly,
  or empty when there is nothing to exempt.
- `IGNORE_PACKS` was measured by its effect on aggregate F1 across many
  sessions; nothing asserted that `extract()`, called directly, actually
  passes a different `domain` to `GraphMemory` when the flag flips.
- `Session.pack` reaching `GraphMemory(domain=...)` was implied by the
  F1 test above but never asserted directly — this checks the plumbing,
  not the downstream effect of the plumbing being right.

And one real bug, found while reading `extract()` to write the tests
above: `Session.pack` was the only field `corpus._validate()` did not
type-check. Every other field in that function raises `CorpusError`
loudly at load time — the module's own docstring says a corpus is
"raised loudly rather than skipped" — but a non-string `pack` (a
malformed corpus JSON file, `"pack": 123`) loaded silently and only
failed three calls later, inside `domains.py`, as
`AttributeError: 'int' object has no attribute 'strip'`. Reproduced
before fixing:

    >>> from benchmarks.eval import corpus as c
    >>> c.load(dir_with({"pack": 123, ...}))   # used to succeed
    >>> extract(that_session)                   # then crashed here instead
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.eval import __main__ as ev  # noqa: E402
from benchmarks.eval import corpus as corpus_mod  # noqa: E402
from benchmarks.eval.__main__ import extract  # noqa: E402


def _session(id_="s1", pack=None, messages=None):
    return corpus_mod.Session(
        id=id_, origin="synthetic", domain="unspecified",
        # The exact phrasing test_label_quality.py already relies on to
        # produce a completed TASK node — a short, generic label like
        # "login endpoint" alone is below the extractor's validator
        # threshold and yields no node at all, which is a fact about the
        # extractor, not something this file is testing.
        messages=messages or [
            {"role": "user", "content": "Building an auth service"},
            {"role": "assistant",
             "content": "Completed: rate limiting with slowapi on every route"},
        ],
        ground_truth={}, pack=pack,
    )


class TestExtractReturnsLabelsPerCategory:
    """extract()'s first return value: labels grouped by the same
    categories evaluate() scores against."""

    def test_the_categories_match_the_selectors(self):
        got, _fixes = extract(_session())
        assert set(got) == set(ev._SELECTORS)

    def test_a_stated_completion_lands_in_completed_tasks(self):
        got, _fixes = extract(_session())
        assert any("rate limiting" in t.lower() for t in got["completed_tasks"])

    def test_a_category_with_nothing_said_is_an_empty_list_not_missing(self):
        got, _fixes = extract(_session(messages=[
            {"role": "user", "content": "Just chatting, nothing to extract."},
        ]))
        assert got["errors"] == []
        assert set(got) == set(ev._SELECTORS), (
            "an empty category must still be a key — evaluate() indexes "
            "got[cat] unconditionally"
        )


class TestExtractReturnsFixesPairs:
    """The label_quality near-duplicate exemption ships or does not ship
    depending on this. Untested, it can go empty forever and nothing
    would notice: label_quality still runs, it just stops exempting
    anything, and near_duplicates creeps up for a reason nobody sees."""

    def test_an_ordinary_session_with_no_fix_has_no_exempt_pairs(self):
        _got, fixes = extract(_session())
        assert fixes == set()

    def test_a_fix_and_the_error_it_resolves_are_returned_as_a_pair(self):
        """flaky_ci is a real committed corpus session, not a synthetic
        one built for this test — the same session
        test_label_quality.py's fixture-teardown-race exemption example
        was drawn from."""
        sessions = corpus_mod.load(None)
        flaky = next(s for s in sessions if s.id == "flaky_ci")

        _got, fixes = extract(flaky)

        assert fixes, "flaky_ci states a fix; extract() must report it"
        assert any(
            "teardown" in " ".join(pair).lower() for pair in fixes
        ), f"expected the fixture-teardown-race pair among {fixes}"


class TestIgnorePacksReachesExtractDirectly:
    """test_domain_packs.py measures IGNORE_PACKS by its effect on
    aggregate F1. This checks the mechanism one level down: that
    extract() itself passes a different `domain` to GraphMemory when
    the flag flips, independent of whether that difference happens to
    move any particular session's score."""

    def test_domain_is_none_when_ignore_packs_is_set(self):
        seen = {}

        class _Spy:
            def __init__(self, *a, domain=None, **kw):
                seen["domain"] = domain
                self._nodes, self._edges = {}, []

            def extract_from_messages(self, *a, **kw):
                pass

        previous = ev.IGNORE_PACKS
        try:
            with patch.object(ev, "GraphMemory", _Spy):
                ev.IGNORE_PACKS = True
                extract(_session(pack="research"))
                assert seen["domain"] is None, (
                    "IGNORE_PACKS=True must reach GraphMemory as "
                    "domain=None regardless of what the session declares"
                )

                ev.IGNORE_PACKS = False
                extract(_session(pack="research"))
                assert seen["domain"] == "research"
        finally:
            ev.IGNORE_PACKS = previous


class TestSessionPackReachesGraphMemory:
    """The plumbing extract() does with session.pack, isolated from
    what GraphMemory then does with the value — that part is covered by
    graph_memory's own domain tests."""

    def test_a_declared_pack_is_passed_through(self):
        seen = {}

        class _Spy:
            def __init__(self, *a, domain=None, **kw):
                seen["domain"] = domain
                self._nodes, self._edges = {}, []

            def extract_from_messages(self, *a, **kw):
                pass

        with patch.object(ev, "GraphMemory", _Spy):
            extract(_session(pack="ops"))
        assert seen["domain"] == "ops"

    def test_no_pack_passes_none_not_the_string_none(self):
        seen = {}

        class _Spy:
            def __init__(self, *a, domain=None, **kw):
                seen["domain"] = domain
                self._nodes, self._edges = {}, []

            def extract_from_messages(self, *a, **kw):
                pass

        with patch.object(ev, "GraphMemory", _Spy):
            extract(_session(pack=None))
        assert seen["domain"] is None


class TestCorpusValidatesThePackField:
    """The one field _validate() did not type-check. Every other field
    in that function raises CorpusError loudly; this one crashed three
    calls later instead, inside domains.py, as an AttributeError with
    no mention of the corpus file that caused it."""

    def _write(self, tmp_path: Path, **overrides) -> Path:
        import json

        raw = {
            "id": "x", "origin": "synthetic",
            "messages": [{"role": "user", "content": "hi"}],
            "ground_truth": {},
        }
        raw.update(overrides)
        d = tmp_path / "corpus"
        d.mkdir(exist_ok=True)
        f = d / "bad.json"
        f.write_text(json.dumps(raw))
        return d

    def test_a_non_string_pack_is_a_corpus_error_at_load_time(self, tmp_path):
        d = self._write(tmp_path, pack=123)

        with pytest.raises(corpus_mod.CorpusError, match="pack"):
            corpus_mod.load(str(d))

    def test_a_string_pack_still_loads(self, tmp_path):
        d = self._write(tmp_path, pack="research")

        sessions = corpus_mod.load(str(d))
        assert sessions[0].pack == "research"

    def test_no_pack_key_still_loads_as_none(self, tmp_path):
        d = self._write(tmp_path)

        sessions = corpus_mod.load(str(d))
        assert sessions[0].pack is None

    def test_an_explicit_null_pack_still_loads_as_none(self, tmp_path):
        d = self._write(tmp_path, pack=None)

        sessions = corpus_mod.load(str(d))
        assert sessions[0].pack is None

    def test_the_committed_corpus_still_loads_clean(self):
        """The fix must not be so strict it rejects real data."""
        sessions = corpus_mod.load(None)
        assert len(sessions) >= 10
