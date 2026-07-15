# Contributing to TokenMizer

Thanks for considering a contribution. This document covers setup, review
expectations, and the areas where a PR is most likely to land.

## Project Philosophy

Quality over quantity. The core value chain is Graph → Checkpoint → Resume;
a contribution that improves extraction accuracy, resume quality, or context
continuity is worth more than a new surface feature.

---

## Development Setup

```bash
git clone https://github.com/Shweta-Mishra-ai/tokenmizer
cd tokenmizer

pip install -e ".[dev]"

pytest tests/ -v
ruff check tokenmizer/
```

---

## Before Opening a PR

```bash
pytest tests/ -v                                # full suite must pass
ruff check tokenmizer/                          # no lint errors
pytest tests/memory_accuracy/ -v                # extraction-accuracy regression check
python scripts/mcp_e2e_check.py                 # MCP stdio transport, end to end

# Optional, appreciated for extraction-affecting changes:
python -m benchmarks.checkpoint_accuracy.runner
```

CI runs the same suite across Python 3.10–3.12 on Linux and 3.12 on Windows,
plus a lint pass and a Docker build/health check. A PR that fails any of
these will not be merged as-is.

---

## What We're Looking For

### High priority
- **Graph extraction accuracy** — real transcripts where a task, decision,
  or file is missed; new regex/heuristic patterns; LLM-pass prompt tuning
- **Decision tracker** — new topic buckets in `decision_tracker.py`,
  sharper `_is_same_decision` matching (the negation and semantic-opposite
  fixes merged so far are representative of the contribution this module
  rewards)
- **Reasoning and ontology** — `graph_memory/reasoning.py` and
  `graph_memory/ontology.py` are recent additions; consistency-check
  rules, additional `why()` / `impact()` query shapes, and ontology
  coverage of edge cases are all in scope
- **Bug fixes** — see [open issues](https://github.com/Shweta-Mishra-ai/tokenmizer/issues),
  particularly the `bug` and `help wanted` labels
- **Provider support** — new providers, or fixes to existing ones

### Medium priority
- **File intelligence** — extraction for additional file types (docx,
  parquet, etc.)
- **Compression heuristics** — patterns that reduce tokens without losing
  meaning
- **Performance** — graph operations on large sessions (see the open
  issue on `_persist()`'s full-rewrite cost)

### Out of scope for now
- New dashboard UI features
- New MCP tools beyond the current six, without a prior issue discussing the use case
- Provider-specific wrapper features
- Changes to the public API response format without prior discussion

If you're unsure whether something fits, open an issue first — it's a
much cheaper way to find out than a full PR.

---

## Code Standards

### Style
- Python 3.10+ type hints on public functions
- Docstrings on public classes and methods
- `ruff check tokenmizer/ --fix` before committing
- Comments should be short and factual, describing a constraint or
  invariant the code doesn't already make obvious — not a narration of
  what the code does, and not a log of how a bug was found

### Architecture rules
- No raw dicts across layer boundaries — use the DTOs in `tokenmizer/core/dto.py`
- No `os.getenv()` outside `config/settings.py` — inject settings instead
- External imports are lazy — `try: import X except ImportError` inside the function that needs it
- Every provider returns `LLMResponseDTO` — no provider-specific type leaks upward

### Testing
- New extraction pattern → a test in `tests/memory_accuracy/` or `tests/unit/test_decision_tracker.py`
- New provider → `tests/unit/` with mocked responses
- Chaos/failure-mode scenario → `tests/chaos/`
- Match the existing test style: a docstring stating the invariant being
  guarded, not a changelog entry

---

## Contributing to Graph Extraction

This is the area with the most leverage. The heuristic extractor
(`tokenmizer/graph_memory/hybrid_extractor.py`, with fallback logic in
`graph.py`) misses real-world phrasing regularly.

1. Find a phrase that should produce a node but doesn't.
2. Write a failing test:

```python
def test_new_pattern():
    msgs = [{"role": "assistant", "content": "The database migration is complete."}]
    result = extractor.heuristic_extract(msgs)
    labels = " ".join(t.lower() for t in result.tasks_done)
    assert "migration" in labels
```

3. Add the pattern.
4. Confirm the test passes and run the full memory-accuracy suite to check
   for regressions elsewhere.

---

## Contributing to the Decision Tracker

To add a topic bucket in `tokenmizer/graph_memory/decision_tracker.py`:

```python
# In _TOPIC_KEYWORDS:
"monitoring": ["datadog", "grafana", "prometheus", "sentry", "newrelic",
               "logging", "observability", "monitoring platform"],
```

```python
from tokenmizer.graph_memory.decision_tracker import classify_topics
assert "monitoring" in classify_topics("Use Datadog for monitoring")
assert "monitoring" in classify_topics("Grafana for metrics")
```

`classify_topics()` returns a set — a decision can legitimately belong to
more than one bucket (e.g. "Use FastAPI with PostgreSQL" touches both
`web_framework` and `database`). Don't special-case single-topic matching.

---

## Reporting an Issue

```
Python version:
OS:
Provider:

What happened:

Expected:

Minimal reproduction:
```

For an extraction miss, include the exact message text where a node
should have been created — that transcript is the reproduction.

---

## Questions

Open a GitHub Discussion rather than an issue for anything that isn't a
concrete bug or feature request.

---

## License

By contributing, you agree your contribution is licensed under the
project's MIT license.
