# Changelog

## [0.2.6] — 2026-07-02 — MCP registry readiness

- NEW console script `tokenmizer-mcp` — runs the MCP stdio server directly
  (no `python3 -m tokenmizer.mcp.server` incantation needed in client configs).
- README carries the `mcp-name` ownership marker required by the official
  MCP registry's PyPI package validation.

## [0.2.5] — 2026-07-02 — restart no longer breaks checkpoints

### Critical
- **`graph_memory/graph.py`:** `_load()` rebuilt nodes/edges from SQLite
  without converting `type`/`status` strings back to their str-Enum types.
  Str-Enum equality kept passing (which hid the bug from every unit test),
  but any `.value` access crashed — concretely, **after a server restart,
  checkpointing any reloaded session returned HTTP 500**, and resume paths
  touching `.value` were equally broken. Enums are now restored on load;
  nodes/edges with unrecognized types (forward-compat) are skipped with a
  warning instead of killing the whole load. Found by the new MCP e2e
  check, not unit tests — reloaded graphs were never checkpoint-ed in tests
  before. Four regression tests added.

### MCP server
- `serverInfo.version` no longer hardcoded (was stale "0.2.3").
- `resume_session` no longer claims "No checkpoint found" when a checkpoint
  exists but its graph was empty — the two cases now produce distinct,
  accurate messages.
- NEW `scripts/mcp_e2e_check.py`: boots the real proxy in-process, spawns
  the real MCP stdio server, and exercises handshake + all 5 tools
  end-to-end. This is the check that caught the restart bug.

## [0.2.4] — 2026-07-02 — launch-readiness fixes

### Critical — the endpoint did not work
- **`api/app.py`:** the `@app.post("/v1/chat/completions")` decorator was
  attached to the `_check_rate_limit` helper (comments between decorator and
  the intended handler don't matter to Python — a decorator binds to the next
  `def`). The real `chat_completions` handler was never registered; every
  request to the flagship endpoint returned `null`. Decorator moved onto the
  actual handler.
- **`security/middleware.py`:** `injection_guard(request)` had no `Request`
  type annotation, so FastAPI treated `request` as a required string QUERY
  parameter — every POST /v1/chat/completions got a 422 even with the route
  fixed. Annotation added (with import-safe fallback).
- **`tests/integration/test_api_endpoint.py`:** NEW — e2e tests that POST to
  the real app with a mocked provider. Both bugs above are now regression-
  tested, including an explicit route-binding assertion.

### OpenAI compatibility
- `ChatRequest` now accepts unknown OpenAI fields (`extra="allow"`) instead
  of silently depending on pydantic's ignore behavior; content blocks
  (multimodal list format) no longer 422 — normalized to text.
- `temperature`, `top_p`, `stop` are now actually forwarded to ALL providers
  (previously only Anthropic received kwargs; OpenAI/Cohere/Gemini/Ollama
  dropped them silently). Provider-specific naming handled (`stop` →
  `stop_sequences` for Anthropic/Cohere/Gemini, `p` for Cohere top_p).

### Models
- Newest Claude models verified working via passthrough: `claude-fable-5`,
  `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5`. Context-window
  lookup now matches longest key first (broad `"claude"` no longer shadows
  specific entries).

### Honest quality gates
- Coverage: removed omits for product modules (api/app.py, auth.py,
  providers, state, analytics, compression engine were all excluded —
  coverage theater). Honest measured coverage: 61% over ALL product code;
  fail_under set to 50 in pyproject (single source of truth; CI override
  removed).
- CI: ruff is now blocking (`--exit-zero` removed); lint violations fixed.

### Docker
- `pip install -e .` ran before the package source was copied — build broke.
  Deps now installed from pyproject first (cached layer), package installed
  after COPY.
- `--workers 2` → `--workers 1`: session locks, graph cache, rate limiter,
  analytics are in-process; multiple workers held divergent state for the
  same session (last-writer-wins graph loss). Documented the Redis path for
  horizontal scaling.

### Portability
- Benchmark runners crashed with `UnicodeEncodeError` on Windows (cp1252
  console vs emoji output). stdout reconfigured to UTF-8 where supported.

### Docs
- README: install instructions no longer claim a PyPI package that isn't
  published (source install until first PyPI release); dead PyPI badges
  removed; benchmark table updated to freshly measured v0.2.4 numbers with
  date and sample-size caveat.

## [Unreleased] — tokenizer, cache, and version consistency fixes

### Correctness — core value proposition
- **`core/tokenizer.py`:** Claude/Anthropic models were counted with tiktoken
  (an OpenAI tokenizer) — wrong vocabulary, typically 5-20% error, worse on
  code. Now routes Claude models through the Anthropic SDK's local tokenizer
  when available, with an honestly-documented fallback to a cl100k_base
  approximation when it isn't. Added `is_claude_model()` so callers can flag
  the approximation to users.
- **`compression/engine.py`:** the LLMLingua quality gate logged a warning
  when ML compression had no real effect, but never actually reverted `text`
  to the heuristic-only result — `text` had already been overwritten before
  the check ran. The gate is now a real gate: heuristic result is saved
  before ML runs, and reverted to on gate failure.
- **`semantic_cache/cache.py`:** `SemanticCache.set()` scoped non-sensitive
  prompts under `session_id` whenever one was provided, but `get()` only
  ever checks the caller's own session key or `"__shared__"` — a non-sensitive
  prompt stored by session A was permanently unreachable from session B.
  Cross-session cache sharing for generic queries was silently, completely
  broken. Fixed: non-sensitive prompts always scope to `"__shared__"`.
- **`semantic_cache/cache.py`:** `SemanticCache` had no `_preference_store`
  attribute; `api/app.py`'s `/api/cache/stats` endpoint referenced
  `_cache._preferences` (wrong name, and the attribute didn't exist at all),
  causing an `AttributeError` on every single call to that endpoint. Fixed:
  attribute added, call site corrected.
- **`graph_memory/graph.py` — found via writing a real (non-vacuous) test:**
  `GraphMemory._load()` parsed `processed_hashes` JSON inline with
  `nodes_json`/`edges_json`, all inside one try block. If `processed_hashes`
  was corrupted (e.g. an interrupted write), the exception fired *before*
  the node-population loop ran — so a session with perfectly valid
  `nodes_json` lost every node on reload, purely because of an unrelated
  corrupted field. This directly contradicted the graceful-recovery
  behavior `test_partial_write_recovery` was supposed to verify (its
  assertion had been `len(g2._nodes) >= 0`, which is always true and
  therefore never caught this). Fixed: `processed_hashes` parsing is now
  isolated in its own try/except; corruption there costs incremental-
  extraction dedup only (safe — `add_node()` already dedupes), not the
  whole graph.

### Test quality
- `tests/unit/test_decision_cache_async.py`: replaced
  `assert result is not None or True` (always passes, tested nothing) with
  a real assertion — now meaningful since the cache scope bug above is fixed.
- `tests/chaos/test_recovery.py`: replaced `assert len(g2._nodes) >= 0`
  (always passes) with an assertion that actually verifies the pre-corruption
  node survives — this is what caught the `_load()` bug above.

### Consistency
- Version bumped to `0.2.3` in all 8 locations that had drifted
  (`pyproject.toml`, `tokenmizer/__init__.py`, `mcp/server.py`,
  `api/app.py`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`,
  both SVGs in `docs/assets/`) to match CHANGELOG's own newest entry, plus
  `SECURITY.md`'s supported-versions table (was still `0.1.x (alpha)`).
- Hardcoded `"python"` binary replaced with `"python3"` (or a detected
  `$PYTHON` variable) in 8 locations across `.mcp.json`, README.md, USAGE.md,
  `.claude-plugin/plugin.json`, `mcp/server.py`'s docstring,
  `scripts/setup.sh`, `scripts/install.sh` (which had already correctly
  detected a `$PYTHON` variable earlier in the script but then ignored it
  when writing `.mcp.json`), and a Claude Code skill file. Breaks on any
  Debian/Ubuntu system where only `python3` exists.
- README: added a visible streaming-unsupported warning directly after the
  Quick Start code example (previously only mentioned deep in the CLI
  section) — Cursor/Continue.dev users were hitting an unexplained HTTP 501.

## [Unreleased] — security/correctness audit pass

Full senior-level audit covering security, silent failures, dead code,
and benchmark honesty. See `TESTING.md` for what's verified vs. not, and
git log for the full per-commit breakdown (9 commits, each independently
reviewable).

### Security (see `tokenmizer/security/`)
- **Critical:** `auth.py` failed OPEN (auth silently disabled) on any
  settings-read exception. Now fails CLOSED (503).
- Background LLM extraction call was not redacted — a real secret pasted
  into a session could reach a third-party extraction model. Fixed by
  redacting once at ingestion, before any downstream consumer.
- `redact_messages()` crashed/silently skipped on multimodal content
  (None or list `content`). Added AWS/Slack/Stripe/JWT patterns.
- "Prompt injection detection" relabeled honestly as a basic keyword
  filter (it's a 10-pattern denylist, not a security boundary); fixed
  wrong `429` status code to `400`; fixed multimodal-content bypass.
- `config/settings.py`: a YAML parse error silently fell back to
  insecure defaults with zero indication. Now logs loudly.

### Correctness
- `compression/engine.py`: LLMLingua (lossy ML compression) could be
  applied to code blocks, risking semantic corruption. Added
  `CodeBlockGuard` to exclude code from the compressor entirely.
- `CommentStripper`'s `//` regex matched inside string literals (e.g.
  URLs), truncating them. Also found and fixed a *pre-existing* bug:
  trailing `# comment` style was never stripped at all (only
  leading-`#` comments were).
- `hybrid_extractor.py`: `merge()` permanently lowercased file paths and
  labels into its output instead of using normalization only as a dedup
  key — wrong on case-sensitive filesystems. Found via testing the
  benchmark rewrite below.

### Reliability — silent failures made visible
- Checkpoint save/list, graph eviction persist, and state-backend writes
  used to fail silently (caught, logged at low severity, swallowed).
  Now raise typed errors or are tracked via a new
  `AnalyticsEngine.record_silent_failure()` counter, surfaced through
  `/api/stats`.
- `graph_memory/graph.py`: decision-contradiction-check failures and DB
  reinit failures were debug-only. Now logged at warning and tracked via
  `decision_tracking_failures`/`persistence_broken` on `GraphStatsDTO`.

### Performance
- `graph_memory/graph.py`: `_persist()` rewrote the entire node/edge set
  on every call. Added dirty-flag tracking to skip redundant rewrites
  (the underlying O(n)-per-write cost when a write IS needed is
  documented as a tracked follow-up requiring a real schema migration,
  not silently left as solved).

### Benchmark honesty
- `benchmarks/checkpoint_accuracy/runner_v3.py`'s `MockLLMProvider`
  sampled its fake output from the same ground-truth dict used to score
  recall — circular, not a real measurement, despite being presented as
  one (90-100% hybrid recall) in the README. Rewritten to verify
  `merge()`'s logic contract against known-overlap fixtures instead, plus
  a real `--live` path. The unsubstantiated README number was removed,
  not replaced with another guess.

### Cleanup
- Removed dead imports across `benchmarks/`, `tests/`, and `tokenmizer/`
  (each verified individually before deletion).
- Added `scripts/static_audit.py` (unused-import / silent-failure-pattern
  scanner) and `scripts/run_stdlib_tests.py` (zero-dependency regression
  suite) as permanent repo tooling.

---

## [0.2.3] — 2026-06-22

### Core reliability fixes

**to_context_block() — quality rewrite**
- Superseded decisions no longer shown as `~~old label~~` (wastes tokens, risks LLM confusion)
- Now shows: `Note: N decision(s) changed — see graph history` or compact `Changed: X → Y` transition
- Completed tasks: importance+recency weighted, capped at 6 most recent (not all 50)
- Node deduplication: skip nodes with same 20-char normalized prefix
- Active decisions show rationale inline; invalidated decisions show `[DO NOT USE]` warning

**query_at_time() — temporal correctness**
- Was broken: called `query()` which excludes SUPERSEDED nodes → temporal queries returned same as current
- Fixed: scans ALL nodes, filters by `valid_from <= at_time AND (valid_until==0 OR valid_until > at_time)`
- Verified: PostgreSQL visible at t_middle, correctly hidden after superseded by SQLite

**merge() confidence — no longer silently overwritten**
- Corroboration confidence (0.95/0.80/0.65) now stored per-decision dict

- `_apply_extracted()` passes `confidence=` to `add_node()` directly
- `add_node()` uses caller confidence when explicitly provided, validator default otherwise

**chat_completions() split: 272L → 86L orchestrator**
- `_check_rate_limit()` 10L
- `_apply_compression_layers()` 36L
- `_update_graph()` 102L
- `_call_provider()` 62L

### Refactoring

**heuristic_extract() 194L → two methods**
- `_extract_one_message()` 140L — all 5 passes on a single message
- `heuristic_extract()` 36L — window setup + calls helper per message

**visualization.py extracted (new, 231L)**
- `to_vis_json()` and `to_obsidian_canvas()` moved from graph.py
- graph.py now has 10-line thin wrappers; backward compat preserved

**graph.py: 1835L → 1172L (-36%)**

### Long session performance
- 200-turn session: <5ms/turn (verified)
- Auto-prune at 200 nodes: works on fresh graphs via importance fallback
- `processed_hashes` capped at 500 entries

## [0.2.0] — 2026-06-07

### What's in this release

#### HybridExtractor — 3-pass extraction pipeline
- Pass 1: LLM-powered structured extraction (JSON schema, ~85-90% recall)
- Pass 2: Enhanced heuristic sweep (file paths, decisions, errors, dependencies)
- Pass 3: Confidence-weighted merge — corroborated items reach 0.95 confidence
- Result: ~88-92% recall vs ~45-55% heuristic-only

#### Rate limiting
- Token-bucket rate limiter: 60 req/min per client, burst of 10
- Stale bucket eviction + hard size cap (50k clients) prevents memory growth
- Returns `Retry-After` header on 429

#### Bug fixes
- Compression quality gate: fixed inverted ratio check (was rejecting good compression)
- Session locks: replaced unbounded dict with LRU-bounded (max 1000 sessions)
- CORS: restricted from wildcard to specific methods and headers
- Semantic cache: safe-by-default scoping (session-scoped unless explicitly generic)
- `to_context_block`: replaced O(n²) tiktoken loop with single-count + char estimate
- Rate limiter: added hard size cap to prevent unbounded memory growth

#### Known limitations (planned for v0.3)
- **Streaming not yet supported**: `stream=true` returns HTTP 501. True SSE streaming
  requires rearchitecting the pipeline (compression + cache + graph all need the full
  response). Planned for v0.3.
- Context Router: reserved in config, not yet implemented.

---

## [0.1.0] — 2025

### Initial alpha release

**Known limitations in 0.1.0 (fixed in 1.0.0)**
- Heuristic extraction: ~45–55% task recall, ~70–80% file recall
- No streaming support
- Session lock dict unbounded (memory leak on long-running servers)
- No rate limiting on proxy endpoint
- Version mismatch across files
