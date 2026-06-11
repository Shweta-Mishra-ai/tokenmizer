---
name: analyze
description: Analyze a large file (CSV, Excel, PDF, JSON, code) and return a token-efficient summary. Instead of reading thousands of rows or pages, get schema + statistics + sample in under 500 tokens. Use when user mentions a file path, asks to analyze data, pastes many rows, or references a CSV/Excel/PDF/JSON file.
---

Analyze a file using TokenMizer's file intelligence layer.

## IMPORTANT rule

**Never ask the user to paste the file content.** Always call TokenMizer to analyze it from the path. Pasting a 50,000-row CSV = 400,000 tokens. TokenMizer reduces it to ~450 tokens.

## What to do

Parse $ARGUMENTS:
- First word = file path
- Remaining words = query (what user wants to know)

```bash
FILE_PATH=$(echo "$ARGUMENTS" | awk '{print $1}')
QUERY=$(echo "$ARGUMENTS" | cut -d' ' -f2-)

python3 -c "
from tokenmizer.filters.file_intelligence import FileIntelligence
fi = FileIntelligence()
result = fi.process(
    open('${FILE_PATH}', 'rb').read(),
    '${FILE_PATH}'.split('/')[-1],
    token_budget=600,
    query='${QUERY}'
)
print(f'File: {result.file_type} | {result.original_tokens:,} → {result.extracted_tokens} tokens ({result.savings_pct:.0f}% saved)')
print()
print(result.content)
"
```

## Token savings by file type

| Type | Typical savings |
|---|---|
| CSV (50k rows) | 99.9% |
| PDF (200 pages) | 98.8% |
| Excel (10 sheets) | 99.7% |
| JSON (1k items) | 95% |
| Code (large file) | 60-80% |

## If TokenMizer not installed

```bash
pip install "tokenmizer[anthropic]"
```

## Examples of $ARGUMENTS

- `/data/sales.csv` → analyze with no specific query
- `/data/sales.csv which regions are underperforming` → targeted analysis
- `/reports/Q1.pdf key findings and risks` → relevant page extraction
- `/data/users.xlsx find inactive accounts` → per-sheet analysis
