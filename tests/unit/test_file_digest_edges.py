"""
Files whose shape the digest did not survive: one line longer than the
budget emptied it, and nesting deeper than the stack failed the request.
"""
from __future__ import annotations

from tokenmizer.core.tokenizer import count_tokens
from tokenmizer.filters.file_intelligence import FileIntelligence, _truncate_to_budget


def test_a_line_longer_than_the_budget_is_cut_not_dropped():
    text = "header\n" + ", ".join(f"col{i}" for i in range(5000))
    out, cut = _truncate_to_budget(text, 100)
    assert cut and out.startswith("header\ncol0, col1")
    assert out.endswith("…") and count_tokens(out) <= 100


def test_short_text_is_untouched():
    assert _truncate_to_budget("a\nb", 100) == ("a\nb", False)


def test_a_wide_csv_lists_its_columns():
    """The digest of a 50,000-column CSV was just its shape line, 11 tokens
    of a 400-token budget: the column list was one line, so it was dropped."""
    body = ",".join(f"c{i}" for i in range(50000)) + "\n" + ",".join("1" for _ in range(50000))
    r = FileIntelligence().process(body, "wide.csv", token_budget=400)
    assert "Columns: c0 (number), c1 (number)" in r.content
    assert r.extracted_tokens > 300 and count_tokens(r.content) <= 400


def test_json_nested_past_the_stack_is_truncated_not_an_error():
    r = FileIntelligence().process("[" * 200_000 + "]" * 200_000, "deep.json", token_budget=200)
    assert r.strategy_used == "fallback_truncation"
    assert 0 < count_tokens(r.content) <= 200
