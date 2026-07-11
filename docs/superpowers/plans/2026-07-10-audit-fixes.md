# Audit Fixes Implementation Plan (2026-07-10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all confirmed bugs from the 2026-07-10 three-track audit (graph memory, MCP server, silent failures), redesign the graph visualization, then truth-pass the README and write a demo storyboard.

**Architecture:** Localized fixes to existing modules; no restructuring. Classifier stays deterministic (heuristics only, per owner decision). Visualization becomes a self-contained no-CDN HTML artifact rendering supersession history as a first-class element.

**Tech Stack:** Python 3.10+, pytest, D3 (inlined), stdlib only for new code.

## Global Constraints

- All 218 existing tests must stay green (`pytest tests/ -v` in `.venv`).
- `ruff check tokenmizer/` must pass (CI-blocking).
- No new required dependencies.
- Version stays 0.3.1 until release; `server.json` synced to it.
- Every bug fix gets a regression test that fails before the fix.

---

### Task 1: Fix LLM extraction call site (app.py)

**Files:** Modify `tokenmizer/api/app.py:423-424`; Test `tests/unit/test_hybrid_extractor.py`

- [ ] Test: construct the exact call pattern app.py uses (`HybridExtractor()` + `extract(msgs, provider_fn=fn)` with a stub async fn) and assert LLM results appear in output.
- [ ] Fix: `ext = HybridExtractor()` then `extracted = await ext.extract(_msgs, provider_fn=_pfn)`.
- [ ] Add a signature-compat assertion test importing the app module's background path (guards against future drift).

### Task 2: MCP server hardening (server.py)

**Files:** Modify `tokenmizer/mcp/server.py`; Test new `tests/unit/test_mcp_server.py`

- [ ] Handlers return `(text, is_error)` tuples (or raise a typed `ToolInputError`); `tools/call` sets `isError` structurally, never via `startswith("❌")`.
- [ ] Validate required args (`session_id`, `file_path`) with clear messages; validate `arguments` is a dict; validate `token_budget` is int.
- [ ] Wrap per-request dispatch in try/except → JSON-RPC error `-32603`, keep loop alive. Guard non-dict `req`. Log dropped malformed JSON at warning.
- [ ] `logging.basicConfig(stream=sys.stderr)` in `run_stdio_server()`; log every tool error at warning.
- [ ] Unknown tool → isError true.
- [ ] Sync `server.json` version to 0.3.1.
- [ ] Tests: missing arg → isError true; `[1,2]` request line → server survives; unknown tool → isError; malformed JSON → loop continues; initialize failure → error response not crash.

### Task 3: Topic classifier fixes (decision_tracker.py)

**Files:** Modify `tokenmizer/graph_memory/decision_tracker.py`; Test `tests/unit/test_validator.py` or new `test_decision_tracker.py`

- [ ] `classify_topics()` (new, plural) returns a **set** of all matched topics; `classify_topic()` kept as thin wrapper (first of set, stable order) for compat.
- [ ] Fix "Go" collision: "go" only counts as language keyword when followed/preceded by language context (e.g. not when followed by "with"/"for" as imperative); simplest robust rule: drop bare "go" from single-word keywords, keep bigrams ("golang", "go lang", "in go", "go backend").
- [ ] Vocabulary: merge tech names from hybrid_extractor.py `_DECISION_FOR` (supabase, clerk, firebase, etc.); add auth providers (clerk, auth0, firebase auth), BaaS (supabase, firebase, appwrite), API (trpc already present — ensure reachable).
- [ ] `find_contradicting_decisions`: compare topic **sets** — contradiction if intersection non-empty.
- [ ] Tests: each audit example ("Go with tRPC" → api_style not language; "Use Supabase..." → backend topic; FastAPI+SQLAlchemy+Postgres → 3 topics; Postgres→SQLite supersedes the multi-topic node).

### Task 4: Silent-failure logging fixes

**Files:** `hybrid_extractor.py:318` (debug→warning), `compression/engine.py:334` (add warning log), `graph.py:215` (warn unless "no such table"), `filters/file_intelligence.py:99,540` (debug→warning for total failure; keep per-page at debug but warn if ALL pages empty)

- [ ] Each with a test asserting the log record is emitted (caplog).

### Task 5: ARCHIVED reachability + invalidate scope

**Files:** `tokenmizer/graph_memory/graph.py` (prune/decay path), `tokenmizer/api/app.py` (invalidate endpoint)

- [ ] In `apply_importance_decay` or `prune()`: SUPERSEDED decisions older than N days (default 7, config-able) → ARCHIVED. Matches README's "superseded shown 7 days" claim.
- [ ] `invalidate_decision`: only match COMPLETED (active) decisions by default; return list of affected node ids; document multi-match behavior.
- [ ] Tests: supersede → age past threshold → decay run → ARCHIVED; invalidate no longer flips SUPERSEDED nodes.

### Task 6: Redaction + checkpoint robustness

**Files:** `tokenmizer/security/redaction.py`, `tokenmizer/checkpoints/manager.py`

- [ ] Add patterns: URL-embedded credentials (`scheme://user:pass@host`), Cohere/Mistral/Together/Azure key formats (documented best-effort), generic high-entropy `Bearer` tokens.
- [ ] `CheckpointManager.create()`: redact `next_action` snippet before persist (defense-in-depth, cheap).
- [ ] `_safe_init_db`: set `self.persistence_broken = True` on total failure; surface via stats.
- [ ] `get_latest()`: distinguish empty vs error (return sentinel/raise on DB error).
- [ ] Tests for each: connection-string redacted; broken dir → flag set; corrupt DB → error distinct from empty.

### Task 7: Visualization redesign

**Files:** Rewrite `tokenmizer/graph_memory/visualization.py` (`to_share_html`), keep `to_vis_json`/`to_obsidian_canvas` API stable

- [ ] Inline d3 (vendored minified or hand-rolled force layout — decide by size; no CDN).
- [ ] Render `transitions`: supersession arcs with reason tooltips; superseded nodes visually chained to successors; "decision timeline" side panel ordered by timestamp.
- [ ] Filter chips by node type + status; text search; active-only toggle.
- [ ] "Export PNG" button (canvas serialization of the SVG).
- [ ] Test: generated HTML contains transitions data, no external URLs, valid JSON payloads.

### Task 8: README truth-pass + demo storyboard

**Files:** `README.md`, new `docs/DEMO_SCRIPT.md`

- [ ] Fix 4-state table (ARCHIVED now reachable — describe actual trigger), note LLM extraction fixed, sync claims to code, dry-run quick-start commands in the venv.
- [ ] Demo storyboard: checkpoint → resume → visualization → live MCP tool calls from Claude Code, with exact commands and expected on-screen results.
