"""
The Excel reader held every row of every sheet in memory before applying
its row cap, and expanded any .xlsx however far its XML inflated.
"""
from __future__ import annotations

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from tokenmizer.filters.file_intelligence import ExcelExtractor  # noqa: E402


def _xlsx(rows: int) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(["id", "city"])
    for i in range(rows):
        ws.append([i, "Pune"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_rows_past_the_cap_are_counted_not_kept(monkeypatch):
    monkeypatch.setattr(ExcelExtractor, "MAX_ROWS_PER_SHEET", 50)
    out = ExcelExtractor().extract(_xlsx(200), "t.xlsx", token_budget=400)
    assert "200 rows" in out.content          # the sheet's true size survives


def test_a_workbook_that_inflates_past_the_limit_is_refused(monkeypatch):
    data = _xlsx(2000)
    monkeypatch.setattr(ExcelExtractor, "MAX_UNCOMPRESSED_BYTES", len(data))
    out = ExcelExtractor().extract(data, "bomb.xlsx")
    assert out.strategy_used == "refused_oversize"
    assert "GiB limit" in out.content


def test_a_file_that_is_not_a_zip_still_reports_a_parse_error():
    out = ExcelExtractor().extract(b"not a workbook", "x.xlsx")
    assert out.strategy_used == "error"
