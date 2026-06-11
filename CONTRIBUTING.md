# Contributing to TokenMizer

Thank you for helping make AI context management better.

## Project Philosophy

**Quality over quantity.** The graph extraction pipeline (Graph → Checkpoint → Resume) is the core value. Every contribution that improves extraction accuracy, resume quality, or context continuity is more valuable than new features.

---

## Development Setup

```bash
git clone https://github.com/Shweta-Mishra-ai/tokenmizer
cd tokenmizer

# Install with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest tests/ -v
ruff check tokenmizer/
```

---

## Before You Open a PR

```bash
# All tests must pass
pytest tests/ -v

# No linting errors
ruff check tokenmizer/

# Memory accuracy regression check
pytest tests/memory_accuracy/ -v

# Optional but appreciated
python -m benchmarks.checkpoint_accuracy.runner
```

---

## What We Welcome

### High Priority
- **Graph extraction improvements** — better patterns, LLM prompt tuning, new node type detection
- **Decision tracker** — new topic categories, better `_is_same_decision` normalization
- **Memory accuracy tests** — real-world session data (anonymized) for benchmarking
- **Bug fixes** — check open issues, especially extraction misses
- **Provider fixes** — new providers or fixes to existing ones

### Medium Priority
- **File intelligence** — better extraction for new file types (docx, parquet, etc.)
- **Compression heuristics** — patterns that reduce tokens without losing meaning
- **Performance improvements** — faster graph operations for large sessions

### Currently Not Accepting
- New dashboard UI features
- New MCP tools beyond the existing 5
- Provider-specific wrapper features
- Changes to the public API response format without discussion

---

## Code Standards

### Style
- Python 3.10+ type hints on all public functions
- Docstrings on all public classes and methods
- `ruff` for linting — run `ruff check tokenmizer/ --fix` before committing

### Architecture Rules
- **No raw dicts across layer boundaries** — use DTOs from `tokenmizer/core/dto.py`
- **No `os.getenv()` outside `config/settings.py`** — inject settings
- **All external imports are lazy** — use `try: import X except ImportError` inside methods
- **Every provider must return `LLMResponseDTO`** — no provider-specific types leaking up

### Testing Rules
- New extraction patterns → add to `tests/memory_accuracy/`
- New provider → add to `tests/unit/` with mocked responses
- Chaos scenarios → add to `tests/chaos/`
- Coverage target: 70% minimum (enforced in CI)

---

## Graph Extraction Contributions

This is the most impactful area. The heuristic extractor in `tokenmizer/graph_memory/graph.py` misses real-world phrases. To improve it:

1. Find a phrase pattern that gets missed
2. Write a test showing it's missed:

```python
def test_new_pattern():
    msgs = [{"role": "assistant", "content": "The database migration is complete."}]
    r = _heuristic_extract(msgs)
    task_labels = " ".join(t["label"].lower() for t in r["tasks"])
    assert "database migration" in task_labels or "migration" in task_labels
```

3. Add the pattern to `graph.py`
4. Verify the test passes
5. Run the full memory accuracy suite to check for regressions

---

## Decision Tracker Contributions

To add a new topic category to `tokenmizer/graph_memory/decision_tracker.py`:

```python
# Add to _TOPIC_KEYWORDS dict:
"monitoring": ["datadog", "grafana", "prometheus", "sentry", "newrelic",
               "logging", "observability", "monitoring platform"],
```

Then test it:
```python
from tokenmizer.graph_memory.decision_tracker import classify_topic
assert classify_topic("Use Datadog for monitoring") == "monitoring"
assert classify_topic("Grafana for metrics") == "monitoring"
```

---

## Issue Reports

When reporting a bug, include:

```
Python version: 3.x.x
OS: macOS / Linux / Windows
Provider: anthropic / openai / etc.

What happened:
[describe]

Expected:
[describe]

Minimal reproduction:
[code or session transcript]
```

For extraction misses, include the actual message text where the node should have been extracted.

---

## Questions

Open a GitHub Discussion — not an issue.

---

## License

By contributing, you agree your contributions will be licensed under MIT.
