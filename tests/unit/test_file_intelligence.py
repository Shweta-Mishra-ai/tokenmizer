"""Unit tests — file intelligence layer."""
from tokenmizer.filters.file_intelligence import (
    CSVExtractor,
    FileIntelligence,
    JSONExtractor,
    TextExtractor,
    detect_file_type,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SMALL_CSV = """id,name,age,salary,department
1,Alice,30,75000,Engineering
2,Bob,25,55000,Marketing
3,Carol,35,90000,Engineering
4,Dave,28,62000,Sales
5,Eve,32,80000,Engineering"""

LARGE_CSV_HEADER = "id,product,price,quantity,category,region,date\n"
LARGE_CSV = LARGE_CSV_HEADER + "\n".join(
    f"{i},Product_{i},{10+i*0.5},{i%100},Cat_{i%5},Region_{i%3},2024-01-{(i%28)+1:02d}"
    for i in range(1, 5001)  # 5000 rows
)

SAMPLE_JSON = """{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25}
  ],
  "total": 2,
  "page": 1
}"""

LARGE_JSON_LIST = "[" + ",".join(
    f'{{"id":{i},"value":"item_{i}","score":{i*1.5},"active":true}}'
    for i in range(1000)
) + "]"


# ── Type detection ─────────────────────────────────────────────────────────────

class TestDetection:
    def test_csv_by_extension(self):
        assert detect_file_type("data.csv", b"a,b,c") == "csv"

    def test_xlsx_by_extension(self):
        assert detect_file_type("report.xlsx", b"\x50\x4b") == "excel"

    def test_pdf_by_extension(self):
        assert detect_file_type("doc.pdf", b"%PDF-1.4") == "pdf"

    def test_json_by_content(self):
        assert detect_file_type("unknown", b'{"key": "value"}') == "json"

    def test_csv_by_content_sniff(self):
        result = detect_file_type("data.dat", b"col1,col2,col3\n1,2,3\n4,5,6\n")
        assert result == "csv"

    def test_python_code(self):
        assert detect_file_type("script.py", b"def hello(): pass") == "code"


# ── CSV extractor ─────────────────────────────────────────────────────────────

class TestCSVExtractor:
    def test_small_csv_extracted(self):
        extractor = CSVExtractor()
        result = extractor.extract(SMALL_CSV, "data.csv", token_budget=400)
        assert result.extracted_tokens <= 420
        assert "5" in result.content  # row count
        assert result.file_type == "csv"

    def test_large_csv_massive_savings(self):
        extractor = CSVExtractor()
        result = extractor.extract(LARGE_CSV, "big_data.csv", token_budget=500)

        # Key assertion: tokens saved must be huge
        assert result.tokens_saved > 10_000
        assert result.savings_pct > 90.0
        assert result.extracted_tokens <= 550

    def test_schema_in_output(self):
        extractor = CSVExtractor()
        result = extractor.extract(SMALL_CSV, "data.csv", token_budget=400)
        assert "Columns:" in result.content
        assert "name" in result.content

    def test_stats_for_numeric_columns(self):
        extractor = CSVExtractor()
        result = extractor.extract(SMALL_CSV, "data.csv", token_budget=400)
        # salary column is numeric — should have stats
        assert "salary" in result.content.lower() or "Stats" in result.content

    def test_sample_rows_present(self):
        extractor = CSVExtractor()
        result = extractor.extract(LARGE_CSV, "big.csv", token_budget=500)
        assert "Sample rows" in result.content

    def test_shape_info_present(self):
        extractor = CSVExtractor()
        result = extractor.extract(LARGE_CSV, "big.csv", token_budget=500)
        assert "5,000" in result.content or "5000" in result.content  # row count

    def test_categorical_summary(self):
        extractor = CSVExtractor()
        result = extractor.extract(SMALL_CSV, "data.csv", token_budget=400)
        # department has 3 unique values — should appear in categories
        assert "Engineering" in result.content or "Categories" in result.content

    def test_tsv_delimiter(self):
        tsv = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n"
        extractor = CSVExtractor()
        result = extractor.extract(tsv, "data.tsv", token_budget=300, delimiter="\t")
        assert result.extracted_tokens > 0
        assert "name" in result.content


# ── JSON extractor ────────────────────────────────────────────────────────────

class TestJSONExtractor:
    def test_small_json_schema(self):
        extractor = JSONExtractor()
        result = extractor.extract(SAMPLE_JSON, "users.json", token_budget=400)
        assert "Schema" in result.content
        assert "users" in result.content

    def test_large_json_savings(self):
        extractor = JSONExtractor()
        result = extractor.extract(LARGE_JSON_LIST, "items.json", token_budget=500)
        assert result.tokens_saved > 1000
        assert result.extracted_tokens <= 550

    def test_array_length_shown(self):
        extractor = JSONExtractor()
        result = extractor.extract(LARGE_JSON_LIST, "items.json", token_budget=500)
        assert "1,000" in result.content or "1000" in result.content

    def test_invalid_json_fallback(self):
        extractor = JSONExtractor()
        result = extractor.extract("not valid json at all!!!", "bad.json", token_budget=100)
        assert result is not None  # doesn't crash
        assert result.content  # has some content


# ── Text extractor ────────────────────────────────────────────────────────────

class TestTextExtractor:
    def test_small_text_passthrough(self):
        extractor = TextExtractor()
        small = "This is a small text file. It fits in budget."
        result = extractor.extract(small, "note.txt", token_budget=1000)
        assert result.strategy_used == "passthrough"
        assert result.tokens_saved == 0

    def test_large_text_truncated(self):
        extractor = TextExtractor()
        large = "This is a line of text.\n" * 500
        result = extractor.extract(large, "log.txt", token_budget=200)
        assert result.extracted_tokens <= 220
        assert result.tokens_saved > 0

    def test_code_preserves_structure(self):
        extractor = TextExtractor()
        code = """import os
import sys
from pathlib import Path

class MyClass:
    def __init__(self, name: str):
        self.name = name

    def process(self):
        # lots of implementation
        result = []
        for i in range(1000):
            result.append(i * 2)
        return result

def main():
    obj = MyClass("test")
    return obj.process()
""" * 20  # repeat to make it large

        result = extractor.extract(code, "script.py", token_budget=200, file_type="code")
        # imports and class/function defs should be preserved
        assert "import" in result.content or "class" in result.content or "def" in result.content


# ── FileIntelligence dispatcher ───────────────────────────────────────────────

class TestFileIntelligence:
    def test_csv_dispatched_correctly(self):
        fi = FileIntelligence()
        result = fi.process(LARGE_CSV.encode(), "data.csv", token_budget=500)
        assert result.file_type == "csv"
        assert result.tokens_saved > 5000

    def test_json_dispatched_correctly(self):
        fi = FileIntelligence()
        result = fi.process(LARGE_JSON_LIST.encode(), "data.json", token_budget=400)
        assert result.file_type == "json"
        assert result.tokens_saved > 0

    def test_process_message_files_detects_inline_csv(self):
        fi = FileIntelligence()
        # Message with embedded CSV data (>50 lines)
        csv_block = "id,name,value\n" + "\n".join(f"{i},item_{i},{i*10}" for i in range(60))
        messages = [
            {"role": "user", "content": f"Analyze this data:\n{csv_block}"}
        ]
        processed, saved = fi.process_message_files(messages, token_budget_per_file=300)
        # FIXED: was `assert saved >= 0` (always true, and the comment
        # "saved some tokens" wasn't actually checked). This masked a real
        # bug: the "Analyze this data:" preamble line diluted the tabular-
        # detection heuristic below its threshold, so CSV compression
        # silently never triggered and saved was always exactly 0 here.
        # See tokenmizer/filters/file_intelligence.py _extract_file_block fix.
        assert saved > 0, (
            f"Expected real token savings from compressing a 60-row CSV, got {saved}. "
            f"CSV detection may be failing due to the preamble text."
        )
        assert len(processed) == 1

    def test_process_message_files_skips_short_messages(self):
        fi = FileIntelligence()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        processed, saved = fi.process_message_files(messages)
        assert processed == messages  # unchanged
        assert saved == 0

    def test_process_message_files_code_fence(self):
        fi = FileIntelligence()
        large_csv = "a,b,c\n" + "\n".join(f"{i},{i*2},{i*3}" for i in range(200))
        messages = [
            {"role": "user", "content": f"Here's my data:\n```csv\n{large_csv}\n```\nAnalyze it."}
        ]
        processed, saved = fi.process_message_files(messages, token_budget_per_file=300)
        assert saved > 0

    def test_budget_respected(self):
        fi = FileIntelligence()
        budget = 300
        result = fi.process(LARGE_CSV.encode(), "big.csv", token_budget=budget)
        assert result.extracted_tokens <= budget + 30  # small slack


# ── Output trimmer ────────────────────────────────────────────────────────────

class TestOutputTrimmer:
    def test_removes_certainly(self):
        from tokenmizer.compression.output_trimmer import OutputTrimmer
        t = OutputTrimmer()
        text = "Certainly! Here is the answer to your question.\n\nThe result is 42."
        trimmed, saved = t.trim(text)
        assert saved > 0
        assert "42" in trimmed
        assert "Certainly" not in trimmed

    def test_removes_closing_filler(self):
        from tokenmizer.compression.output_trimmer import OutputTrimmer
        t = OutputTrimmer()
        text = "The answer is 42.\n\nLet me know if you need anything else!"
        trimmed, saved = t.trim(text)
        assert "42" in trimmed
        # FIXED: was `assert saved >= 0` — always true, didn't check filler
        # was actually removed. Verified trim() genuinely saves tokens here
        # (10, for this input) — require > 0 so a no-op trim() would fail.
        assert saved > 0, f"Expected filler removal to save tokens, got {saved}"
        assert "Let me know" not in trimmed, "Filler line should have been removed"

    def test_preserves_real_content(self):
        from tokenmizer.compression.output_trimmer import OutputTrimmer
        t = OutputTrimmer()
        code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
        trimmed, saved = t.trim(code)
        assert "fibonacci" in trimmed  # code preserved

    def test_empty_string_safe(self):
        from tokenmizer.compression.output_trimmer import OutputTrimmer
        t = OutputTrimmer()
        trimmed, saved = t.trim("")
        assert trimmed == ""
        assert saved == 0

    def test_full_level_removes_closing_fillers(self):
        from tokenmizer.compression.output_trimmer import OutputTrimmer

        t = OutputTrimmer()

        text = (
            "Certainly!\n\n"
            "Here is your answer.\n\n"
            "Hope this helps!"
        )

        trimmed, saved = t.trim(text, "full")

        assert "Hope this helps!" not in trimmed
        assert saved > 0


# ── Smart window ──────────────────────────────────────────────────────────────

class TestSmartWindow:
    def test_short_conversation_unchanged(self, tmp_path):
        from tokenmizer.compression.window import SmartMessageWindow
        from tokenmizer.graph_memory.graph import GraphMemory

        w = SmartMessageWindow(token_budget=4000, protect_recent=8)
        g = GraphMemory("sw-test", storage_dir=str(tmp_path))

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        windowed, saved = w.apply(messages, g)
        assert windowed == messages
        assert saved == 0

    def test_long_conversation_windowed(self, tmp_path):
        from tokenmizer.compression.window import SmartMessageWindow
        from tokenmizer.graph_memory.graph import GraphMemory

        w = SmartMessageWindow(token_budget=500, protect_recent=4)
        g = GraphMemory("sw-long", storage_dir=str(tmp_path))
        g.add_node(g._nodes.__class__.__mro__[0] if False else
                   __import__('tokenmizer.graph_memory.graph', fromlist=['NodeType']).NodeType.TASK,
                   "Build auth system")

        # 20 messages, each ~50 tokens
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"This is message number {i} with some content to make it a bit longer. " * 3}
            for i in range(20)
        ]

        from tokenmizer.core.tokenizer import count_messages_tokens
        original_tokens = count_messages_tokens(messages)

        if original_tokens > 500:
            windowed, saved = w.apply(messages, g)
            assert saved > 0
            assert len(windowed) < len(messages)


class TestSilentFailuresAreLogged:
    """
    Regression tests: two parse-failure fallback paths in this module
    caught broad exceptions and returned a degraded result with NO log
    line at all — inconsistent with every other fallback path in this
    same file (CSV parse failure, file-type sniff failure), which do log
    a warning. An operator watching server logs had zero visibility into
    either failure mode; only the API response consumer (who may not be
    the one troubleshooting a production deployment) could tell.
    """

    def test_jsonl_parse_failure_is_logged(self, caplog):
        import logging

        from tokenmizer.filters.file_intelligence import JSONExtractor

        # Not valid JSON, not valid JSONL either (each line individually
        # invalid) -> falls all the way through to the truncation fallback.
        content = "{not json\nnor this one either{{{"
        with caplog.at_level(logging.WARNING, logger="tokenmizer.filters.file_intelligence"):
            result = JSONExtractor().extract(content, "broken.json")
        assert result.strategy_used == "fallback_truncation"
        assert any("json" in r.message.lower() for r in caplog.records), (
            "JSONL parse failure fell back to truncation with no log line — "
            "inconsistent with this file's other fallback paths"
        )

    def test_excel_parse_failure_is_logged(self, caplog, monkeypatch):
        import logging

        from tokenmizer.filters.file_intelligence import ExcelExtractor

        def _boom(*a, **k):
            raise ValueError("corrupted workbook")

        import openpyxl
        monkeypatch.setattr(openpyxl, "load_workbook", _boom)

        with caplog.at_level(logging.WARNING, logger="tokenmizer.filters.file_intelligence"):
            result = ExcelExtractor().extract(b"not a real xlsx", "broken.xlsx")
        assert result.strategy_used == "error"
        assert any("excel" in r.message.lower() for r in caplog.records), (
            "Excel parse failure returned an error result with no log line"
        )
