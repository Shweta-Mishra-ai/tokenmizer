# Contributing to TokenMizer

Thanks for your interest in contributing. This guide covers everything
needed to get a change from idea to merged PR.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Convention](#commit-convention)
- [Pull Request Process](#pull-request-process)
- [Running Tests](#running-tests)
- [Improving extraction](#improving-extraction)
- [Good First Issues](#good-first-issues)

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork
   ```bash
   git clone https://github.com/YOUR_USERNAME/tokenmizer.git
   cd tokenmizer
   ```
3. **Create a branch**
   ```bash
   git checkout -b fix/your-fix-name
   ```

---

## Project Structure

This reflects the actual current layout — verify against `find tokenmizer -name "*.py"`
before trusting a stale copy of this section. Two things worth knowing up
front. Three things are placeholders rather than working code, and are
labelled as such in their own modules so nobody wires against them by
mistake:

| Path | Reality |
|---|---|
| `tokenmizer/agents/` | Empty package. No code. |
| `tokenmizer/storage/__init__.py` | Documents a `StorageBackend` protocol nothing imports or formally implements. Do not assume a shared interface exists at the type level. |
| `tokenmizer/state/backend.py` | Has **no callers**. `state_backend: redis` is accepted by config and changes nothing. |

```
tokenmizer/
│
├── pyproject.toml               # Package metadata, optional extras (anthropic/openai/gemini/cohere/cache/files/compression/redis/observability/dev)
├── tokenmizer.yaml               # Example config — copy and edit, or use TOKENMIZER_* env vars (env always wins)
├── server.json                   # MCP registry manifest
├── glama.json                    # Glama directory maintainer verification
│
├── tokenmizer/
│   ├── cli.py                    # `tokenmizer` command: serve, stats, checkpoint, resume
│   │
│   ├── api/
│   │   ├── app.py                # FastAPI app — /v1/chat/completions, pipeline, durability
│   │   ├── routes_graph.py       # Session/graph/checkpoint/reasoning endpoints
│   │   └── rate_limiter.py       # Per-client token-bucket rate limiting
│   │
│   ├── config/
│   │   └── settings.py           # pydantic-settings — TOKENMIZER_* env prefix, tokenmizer.yaml backing
│   │
│   ├── core/
│   │   ├── dto.py                # Cross-layer data transfer objects
│   │   ├── errors.py             # Shared exception types (CheckpointPersistError, StorageError, ...)
│   │   └── tokenizer.py          # tiktoken-backed counting, provider-aware
│   │
│   ├── providers/
│   │   └── providers.py          # BaseProvider + Anthropic/OpenAI/DeepSeek/Mistral/OpenRouter/Grok/Cohere/Gemini/Ollama
│   │
│   ├── compression/
│   │   ├── engine.py             # CodeBlockGuard, CommentStripper, JSON flattening
│   │   ├── output_trimmer.py     # Terse-output levels (lite/full/ultra)
│   │   └── window.py             # Smart message windowing before summarization
│   │
│   ├── graph_memory/             # The knowledge graph — the core of the product
│   │   ├── types.py              # NodeType, NodeStatus, EdgeType, MemoryNode, MemoryEdge, DecisionTransition
│   │   ├── graph.py               # GraphMemory: add_node, persistence, pruning, decay
│   │   ├── hybrid_extractor.py    # LLM pass + heuristic pass + confidence-scored merge
│   │   ├── patterns.py            # Regex vocabulary + pure helpers hybrid_extractor.py applies
│   │   ├── decision_tracker.py    # Topic classification + same-decision / contradiction detection
│   │   ├── validator.py           # Confidence scoring and hard-reject rules for candidate nodes
│   │   ├── ontology.py            # Machine-readable node/edge semantics + status state machine
│   │   ├── reasoning.py           # why() / impact() / decision_history() / consistency_check()
│   │   ├── persistence.py         # Per-row SQLite storage (schema v2) + v1 migration
│   │   ├── filelock.py            # Cross-process advisory lock for session writes
│   │   ├── pruning.py             # Importance decay + node-count cap
│   │   ├── context_block.py       # Tiered, token-budgeted resume context
│   │   ├── helpers.py             # Content flattening, trigger/evidence inference
│   │   └── visualization.py       # Self-contained interactive HTML graph export
│   │
│   ├── checkpoints/
│   │   └── manager.py             # Checkpoint create/save/load, tiered resume blocks
│   │
│   ├── state/
│   │   └── backend.py             # UNUSED — no callers; see the module docstring
│   │
│   ├── semantic_cache/
│   │   └── cache.py               # Embedding-similarity response cache
│   │
│   ├── filters/
│   │   └── file_intelligence.py   # CSV/Excel/PDF/JSON → token-efficient summaries
│   │
│   ├── security/
│   │   ├── auth.py                 # API key verification (fail-closed on config error)
│   │   ├── ownership.py            # Session ownership — who may read/modify a session
│   │   ├── middleware.py           # Prompt-injection keyword filter (best-effort — see SECURITY.md)
│   │   └── redaction.py             # Secret/PII pattern redaction
│   │
│   ├── mcp/
│   │   └── server.py                # MCP stdio transport + the 6 tool handlers
│   │
│   ├── dashboard/
│   │   └── page.py                  # Self-contained HTML for GET /
│   │
│   ├── analytics/
│   │   └── engine.py                # Token-savings and silent-failure analytics
│   │
│   └── storage/
│       └── __init__.py              # StorageBackend protocol (documentation-level, not enforced — see note above)
│
├── scripts/                      # mcp_e2e_check.py, gen_demo_gif.py, setup/install helpers
├── benchmarks/
│   ├── eval/                     # extraction P/R/F1 harness + labelled corpus
│   ├── checkpoint_accuracy/      # graph vs summary baseline
│   ├── graph_retrieval/          # category recall
│   ├── persistence/              # write amplification + concurrency
│   └── latency/                  # end-to-end proxy latency (needs a running server)
└── tests/                        # 44 files, 606 tests — see TESTING.md for how to run them
```

---

## Development Setup

### Prerequisites

- Python 3.10+ (CI matrix: 3.10, 3.11, 3.12 and 3.13 on Linux, plus 3.12
  on Windows — every version claimed in the pyproject classifiers)
- A provider API key (Anthropic, OpenAI, etc.) for anything that exercises
  the LLM extraction pass — heuristic-only extraction and most of the
  graph/checkpoint/MCP tests need no key at all

### Install dependencies

```bash
pip install -e ".[dev]"
```

### Configuration

There's no `.env.example` — `tokenmizer.yaml` at the repo root is the
config reference; copy it, edit it, or override any field with a
`TOKENMIZER_*` environment variable (env vars always win).

```bash
cp tokenmizer.yaml my-config.yaml
TOKENMIZER_CONFIG=my-config.yaml tokenmizer serve
```

### Run locally

```bash
tokenmizer serve                # proxy + dashboard on :8000
python -m tokenmizer.mcp.server  # MCP stdio server, standalone
```

---

## Making Changes

### Layer rules

| Layer | Rule |
|-------|------|
| `graph_memory/` | No GitHub/provider API calls; pure graph logic and SQLite persistence |
| `providers/` | Only LLM provider calls; every provider returns the same response shape — no provider-specific type leaks upward |
| `api/` | Orchestrates only — delegates to `graph_memory/`, `checkpoints/`, `security/` |
| `mcp/` | Talks to the running proxy over HTTP (via `TOKENMIZER_URL`); no direct graph/DB access |
| `security/` | Pure functions where possible; `redaction.py` and `auth.py` have no I/O beyond what they're explicitly checking |
| `config/` | The only place the FastAPI app reads `os.environ` directly for settings — everything under `api/` and `graph_memory/` takes settings as a parameter. `mcp/server.py` is a separate standalone process (not part of the FastAPI settings graph) and reads its own `TOKENMIZER_*` env vars directly — that's intentional, not a violation |

### Adding a new decision topic bucket

`tokenmizer/graph_memory/decision_tracker.py`:

```python
# In _TOPIC_KEYWORDS:
"monitoring": ["datadog", "grafana", "prometheus", "sentry", "newrelic",
               "logging", "observability", "monitoring platform"],
```

```python
from tokenmizer.graph_memory.decision_tracker import classify_topics
assert "monitoring" in classify_topics("Use Datadog for monitoring")
```

`classify_topics()` returns a set — a decision can belong to more than one
bucket (e.g. "Use FastAPI with PostgreSQL" touches both `web_framework`
and `database`). Don't special-case single-topic matching.

### Improving graph extraction

The extractor (`tokenmizer/graph_memory/hybrid_extractor.py`, heuristic
fallback in `graph.py`) misses real-world phrasing regularly.

1. Find a phrase that should produce a node but doesn't.
2. Write a failing test:
   ```python
   def test_new_pattern():
       msgs = [{"role": "assistant", "content": "The database migration is complete."}]
       result = extractor.heuristic_extract(msgs)
       labels = " ".join(t.lower() for t in result.tasks_done)
       assert "migration" in labels
   ```
3. Add the pattern, confirm the test passes, and run the full
   memory-accuracy suite to check for regressions elsewhere.

### Adding a new provider

`tokenmizer/providers/providers.py`:

1. Subclass `BaseProvider` (or an existing OpenAI-compatible provider if
   the API is OpenAI-shaped — see `DeepSeekProvider`, `MistralProvider`,
   `OpenRouterProvider`, `GrokProvider` for that pattern).
2. Implement `chat()` returning the same response shape every other
   provider returns.
3. Wire it into `build_provider()`'s provider-name dispatch.
4. Add a unit test with a mocked HTTP response — no real API calls in tests.

### Adding a new MCP tool

`tokenmizer/mcp/server.py`:

1. Add the tool's schema to `TOOLS`.
2. Implement `handle_yourtool(args) -> tuple[str, bool]` — the second
   value is `is_error`, and it must be structural (never inferred from
   the message text — see the other handlers for the pattern).
3. Register it in `handle_tool_call()`'s dispatch dict.
4. Add a test in `tests/unit/test_mcp_server.py` and update the tool-count
   assertion in `scripts/mcp_e2e_check.py`.

### Extending reasoning or the ontology

`tokenmizer/graph_memory/reasoning.py` and `ontology.py` are new — the
ontology *describes and audits* the graph but does not gate writes
(ingestion stays permissive by design; see the module docstring). A new
consistency-check rule belongs in `consistency_check()`; a new query
shape is a new function alongside `why()` / `impact()`, not a parameter
bolted onto an existing one.

---

## Commit Convention

Recent history mostly follows `type(scope): description`, loosely based
on [Conventional Commits](https://www.conventionalcommits.org/) — it's a
convention to follow, not a CI-enforced rule.

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure, no behavior change |
| `test` | Adding or updating tests |
| `chore` | Dependencies, config, tooling |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |
| `security` | Security fix or hardening |

```bash
feat(reasoning): add impact() typed-neighborhood query
fix(decision-tracker): treat negated decisions as distinct
docs(readme): credit external contributors
test(mcp): cover why_decision tool error paths
security(redaction): add URL-embedded credential pattern
```

---

## Pull Request Process

1. Branch from `main` with a descriptive name.
2. Write tests for new functionality.
3. Run the full suite locally before pushing (see [Running Tests](#running-tests)).
4. Fill out the PR description: what changed and why, not just what.
5. Reference the issue it fixes (`Fixes #NN`) if one exists.

### PR checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] `ruff check tokenmizer/` is clean
- [ ] New behavior has a test that fails without the fix
- [ ] Commit messages follow the convention above
- [ ] No secrets, API keys, or `.env`-style files committed
- [ ] Layer boundaries respected (see [Making Changes](#making-changes))

---

## Running Tests

```bash
pytest tests/ -v                                # full suite
pytest tests/unit/test_decision_tracker.py -v   # a single module
pytest tests/memory_accuracy/ -v                # extraction-accuracy regression
python scripts/mcp_e2e_check.py                 # MCP stdio transport, end to end
pytest tests/ --cov=tokenmizer --cov-report=term-missing   # coverage report (no enforced floor)

### Changing extraction

Extraction changes must be measured, not eyeballed:

```bash
python -m benchmarks.eval              # precision / recall / F1 per category
python -m benchmarks.eval --errors     # every miss and false positive, by session
python -m benchmarks.eval --sweep      # how strict is "found"?
```

`tests/unit/test_extraction_quality.py` holds F1 floors that fail the
build on a regression. If your change moves a number the wrong way, say
so in the PR with the before/after table — a trade is fine, an unnoticed
regression is not.

Tuning a constant? Sweep it and put the table in the code comment, as
`_MIN_CLAUSE_CHARS` does. The point is that the next person can disagree
with the trade-off rather than with a magic number.

**Adding corpus sessions is one of the most useful contributions
available.** The committed corpus is 8 hand-written synthetic sessions;
it is the weakest part of every extraction claim this project makes.
Format and rules are documented in `benchmarks/eval/corpus.py`. Label
spans that appear in the transcript, not hindsight summaries — scoring
against "auth endpoints" when nobody said it measures paraphrasing, not
extraction.
```

Tests run against real SQLite in temp directories and mocked provider
responses — no network access or live API keys required for the unit and
integration suites. See [TESTING.md](TESTING.md) for exactly what each
suite verifies.

---

## Improving extraction

**This is the contribution the project needs most, and it does not require
writing any code.**

Extraction is measured against a labelled corpus in
`benchmarks/eval/corpus/`. There are 14 sessions in it and the same person
wrote every label. That is an honest ceiling on what the reported numbers
can tell anyone about *their* workload, and the only way past it is
transcripts nobody here wrote.

### Run the harness

```bash
python -m benchmarks.eval                 # score the committed corpus
python -m benchmarks.eval --errors        # every miss and every false positive
python -m benchmarks.eval --corpus DIR    # score YOUR labelled sessions
python -m benchmarks.eval --sweep         # what a threshold change costs
```

`--errors` is the one to start with. It prints, per session and per
category, exactly what was missed and what was invented — which is a
to-do list, not a score.

### Add a session

One JSON file per session. The format and the labelling rule are
documented at the top of [`benchmarks/eval/corpus.py`](benchmarks/eval/corpus.py);
read that before labelling, because the rule is what makes the numbers
mean anything.

Two constraints the harness enforces, so you will hit them anyway:

1. **Every label must be quoted from a single message.** A label written
   from hindsight is unreachable for *any* extractor, so it caps recall at
   a number no code change can move. `validate_grounding()` refuses to
   score a corpus containing one — it found two in ours.
2. **Labelling is exhaustive, not selective.** If the rule matches, it gets
   labelled, including decisions that were later superseded. Picking which
   of several stated decisions "counts" turns precision into a measure of
   your taste, and penalises the extractor for being right.

Redact freely — rename files, change company names, replace secrets. What
matters is the *shape* of the prose, not its content.

### Change a pattern

Every extraction change needs a number attached. The workflow:

```bash
python -m benchmarks.eval > before.txt
# ... make the change ...
python -m benchmarks.eval > after.txt
diff before.txt after.txt
```

Put the before/after in the PR description. A change that improves recall
while quietly costing precision is not an improvement, and the report
shows both so the trade is visible rather than argued about.

`tests/unit/test_extraction_quality.py` holds per-category F1 floors, a
separate floor for the real-transcript half so the synthetic sessions
cannot carry the headline, and a scan-cost bound. **Do not edit an
assertion to make it pass.** They are set below the measured values on
purpose, so a real regression reds the build and ordinary variation does
not; if one fails, something got worse.

---

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/Shweta-Mishra-ai/tokenmizer/labels/good%20first%20issue).
Representative examples of what lands here:

- A new topic bucket in the decision tracker
- A missed extraction pattern with a failing test attached
- A new secret-redaction pattern in `security/redaction.py`
- A new consistency-check rule in `reasoning.py`
- Test coverage for an existing but under-tested function

If nothing's currently labeled, [open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose)
describing what you'd like to work on — that's a normal way to start.

---

## Questions?

Open a GitHub Discussion for anything that isn't a concrete bug or
feature request. For bug reports, see the issue template — it asks for
Python version, OS, provider, and a minimal reproduction; for an
extraction miss specifically, the exact message text that should have
produced a node is the reproduction.

By contributing, you agree your contribution is licensed under the
project's MIT license.

Built by [Shweta Mishra](https://github.com/Shweta-Mishra-ai)
