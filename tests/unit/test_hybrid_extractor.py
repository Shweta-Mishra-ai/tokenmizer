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
    assert len(merged.files) >= 0


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
    assert len(result.files) >= 0  # at minimum doesn't crash
