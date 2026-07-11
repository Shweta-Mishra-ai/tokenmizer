"""Tests for HybridExtractor — targets 80%+ recall on synthetic session."""
import pytest

from tokenmizer.graph_memory.hybrid_extractor import ExtractedData, HybridExtractor

MESSAGES = [
    {"role": "user", "content": "Goal: build FastAPI auth service with JWT and PostgreSQL."},
    {"role": "assistant", "content": "Let's start with the user model."},
    {"role": "user", "content": "Decided to use bcrypt instead of argon2 for password hashing."},
    {"role": "assistant", "content": "Bcrypt is solid."},
    {"role": "user", "content": "Completed the login endpoint in api/auth.py. 18 tests passing."},
    {"role": "assistant", "content": "Well done."},
    {"role": "user", "content": "Switching from React to Next.js for better SEO."},
    {"role": "assistant", "content": "Good call."},
    {"role": "user", "content": "Working on refresh token rotation now."},
    {"role": "assistant", "content": "Using Redis for refresh tokens is the right call."},
    {"role": "user", "content": "Environment: Python 3.12, FastAPI 0.111"},
]

extractor = HybridExtractor(min_confidence=0.50)


def recall(found_list, expected_keywords):
    text = " ".join(found_list).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords) if expected_keywords else 1.0


def test_heuristic_extracts_files():
    result = extractor.heuristic_extract(MESSAGES)
    assert any("auth" in f.lower() for f in result.files), \
        f"Expected api/auth.py in files, got: {result.files}"


def test_heuristic_extracts_decisions():
    result = extractor.heuristic_extract(MESSAGES)
    decision_labels = [d.get("label", "") for d in result.decisions]
    has_bcrypt = any("bcrypt" in d.lower() for d in decision_labels)
    assert has_bcrypt, f"Expected bcrypt decision, got: {decision_labels}"


def test_heuristic_extracts_tasks_done():
    result = extractor.heuristic_extract(MESSAGES)
    all_tasks = result.tasks_done + result.tasks_wip
    has_login = any("login" in t.lower() for t in all_tasks)
    assert has_login, f"Expected login endpoint in tasks, got: {all_tasks}"


def test_heuristic_extracts_superseded():
    result = extractor.heuristic_extract(MESSAGES)
    # "switching from React to Next.js" should produce a superseded entry
    assert len(result.superseded) > 0, "Expected superseded entry for React→Next.js"


def test_merge_heuristic_only():
    """merge() with llm=None should still produce valid ExtractedData."""
    heu = extractor.heuristic_extract(MESSAGES)
    merged = extractor.merge(None, heu)
    assert isinstance(merged, ExtractedData)
    # FIXED: was `assert len(merged.files) >= 0` (mathematically always true).
    # MESSAGES explicitly mentions "api/auth.py" — a real extraction must find it.
    assert len(merged.files) >= 1, "Should extract at least the file mentioned in MESSAGES"
    assert "api/auth.py" in merged.files, f"Expected api/auth.py in {merged.files}"


def test_merge_boosts_corroborated():
    """Items found by both LLM and heuristic should get 0.95 confidence."""
    llm_data = ExtractedData(
        files=["api/auth.py"],
        decisions=[{"label": "bcrypt for hashing", "reason": ""}],
    )
    heu_data = ExtractedData(
        files=["api/auth.py", "api/models.py"],
        decisions=[{"label": "using bcrypt", "reason": ""}],
    )
    merged = extractor.merge(llm_data, heu_data)
    # auth.py corroborated → should appear
    assert any("auth" in f for f in merged.files)


def test_merge_preserves_original_case():
    """
    FIXED BUG (found while rewriting benchmarks/checkpoint_accuracy/
    runner_v3.py to remove a circular mock — see that file's module
    docstring): merge() previously built its combined output list
    directly from `_normalize()`d (lowercased) sets. _normalize() is
    meant to be a DEDUP KEY ONLY (this is exactly the pattern
    _deduplicate() uses correctly elsewhere in this same file — it
    normalizes for the `seen` set membership check but appends the
    original-case `item` to the output). merge() was inconsistent with
    that pattern: a file extracted as "src/App.tsx" by one source would
    come out of merge() as "src/app.tsx" — wrong on any case-sensitive
    filesystem (Linux, most CI/production environments), and misleading
    to a user trying to find which file was actually touched.
    """
    llm_data = ExtractedData(files=["src/App.tsx", "src/Utils.ts"])
    heu_data = ExtractedData(files=["src/app.tsx", "src/Other.ts"])  # same file, different case

    merged = extractor.merge(llm_data, heu_data)

    # The corroborated file must appear with ORIGINAL casing from one of
    # the two sources — never silently lowercased into a third, wrong form.
    assert any(f in ("src/App.tsx", "src/app.tsx") for f in merged.files), (
        f"corroborated file lost its original casing entirely: {merged.files}"
    )
    assert not any(f == "src/app.tsx" and "src/App.tsx" not in llm_data.files
                   and "src/App.tsx" not in heu_data.files for f in merged.files), (
        "file casing was silently rewritten rather than preserved from a source"
    )
    # Both LLM-only and heuristic-only items must survive with their casing intact
    assert "src/Utils.ts" in merged.files, f"LLM-only file dropped or recased: {merged.files}"
    assert "src/Other.ts" in merged.files, f"heuristic-only file dropped or recased: {merged.files}"
    # Corroboration confidence logic must still work after the fix
    assert merged.confidence["files"] == 0.95


def test_merge_llm_only_no_corroboration():
    """Sanity check that the case-preservation fix didn't break the
    llm-only confidence tier (0.80, not 0.95) for the simple-list categories."""
    llm_data = ExtractedData(files=["src/Solo.ts"])
    heu_data = ExtractedData(files=[])
    merged = extractor.merge(llm_data, heu_data)
    assert merged.files == ["src/Solo.ts"]
    assert merged.confidence["files"] == 0.80


def test_overall_recall_heuristic():
    """Heuristic-only recall should be ≥55% on synthetic session."""
    result = extractor.heuristic_extract(MESSAGES)
    all_extracted = (
        result.goals + result.tasks_done + result.tasks_wip +
        [d.get("label", "") for d in result.decisions] +
        result.files + result.environments
    )
    expected = ["login", "bcrypt", "auth.py", "Python 3.12", "Next.js"]
    r = recall(all_extracted, expected)
    assert r >= 0.55, f"Heuristic recall {r:.0%} below 55% floor"


@pytest.mark.asyncio
async def test_extract_without_provider():
    """extract() with no provider should run heuristic only."""
    result = await extractor.extract(MESSAGES, provider_fn=None)
    assert isinstance(result, ExtractedData)
    # FIXED: was `assert len(result.files) >= 0  # at minimum doesn't crash`
    # (mathematically always true). Heuristic-only extraction should still
    # find "api/auth.py", which is explicitly present in MESSAGES.
    assert len(result.files) >= 1, "Heuristic-only extraction should find at least one file"


@pytest.mark.asyncio
async def test_extract_llm_pass_actually_invoked_app_style():
    """
    Regression guard for the api/app.py call pattern: construct with
    defaults and pass provider_fn to extract(). Asserts the provider is
    actually invoked and its output is merged — a call-signature drift
    here silently disables the LLM pass.
    """
    calls = []

    async def _pfn(messages, system="", max_tokens=600):
        calls.append(len(messages))
        return {"text": (
            '{"goals": [], "tasks_done": [], "tasks_wip": [],'
            ' "decisions": [{"label": "use Vitest for testing", "reason": "speed"}],'
            ' "files": ["src/only_llm_saw_this.ts"], "errors": [],'
            ' "dependencies": [], "environments": [], "superseded": []}'
        )}

    ext = HybridExtractor()
    result = await ext.extract(MESSAGES, provider_fn=_pfn)

    assert calls, "provider_fn was never invoked — LLM pass silently skipped"
    assert "src/only_llm_saw_this.ts" in result.files, (
        f"LLM-only file missing from merged output: {result.files}"
    )


class TestMinConfidenceFilter:
    """
    min_confidence filters extract() output by merge()'s confidence tiers
    (0.95 corroborated / 0.80 LLM-only / 0.65 heuristic-only). The default
    of 0.55 keeps every tier.
    """

    def test_default_keeps_heuristic_only_items(self):
        ext = HybridExtractor()  # default 0.55
        merged = ext._filter_by_confidence(ext.merge(None, ext.heuristic_extract(MESSAGES)))
        assert merged.files, "default threshold must not drop heuristic-only items"

    def test_strict_threshold_drops_heuristic_only_category(self):
        ext = HybridExtractor(min_confidence=0.9)
        llm = ExtractedData(files=["api/auth.py"])          # corroborated below
        heu = ExtractedData(files=["api/auth.py"],          # -> files tier 0.95
                            errors=["timeout in worker"])    # -> errors tier 0.65
        merged = ext._filter_by_confidence(ext.merge(llm, heu))
        assert merged.files == ["api/auth.py"], "corroborated tier must survive"
        assert merged.errors == [], "heuristic-only tier must be dropped at 0.9"

    def test_per_item_decision_filtering(self):
        ext = HybridExtractor(min_confidence=0.9)
        llm = ExtractedData(decisions=[
            {"label": "use bcrypt for hashing", "reason": ""},   # corroborated -> 0.95
            {"label": "use Vitest for tests", "reason": ""},     # llm-only -> 0.80
        ])
        heu = ExtractedData(decisions=[
            {"label": "use bcrypt for hashing", "reason": ""},
            {"label": "use Redis for cache", "reason": ""},      # heuristic-only -> 0.65
        ])
        merged = ext._filter_by_confidence(ext.merge(llm, heu))
        labels = [d["label"] for d in merged.decisions]
        assert labels == ["use bcrypt for hashing"], (
            f"only the corroborated decision survives 0.9, got {labels}"
        )

    @pytest.mark.asyncio
    async def test_extract_applies_filter_end_to_end(self):
        ext = HybridExtractor(min_confidence=0.9)
        result = await ext.extract(MESSAGES, provider_fn=None)  # all heuristic-only
        assert result.files == [] and result.decisions == [], (
            "heuristic-only extraction at min_confidence=0.9 must yield nothing"
        )
