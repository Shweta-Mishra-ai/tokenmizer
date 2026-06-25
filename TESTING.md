# Testing status — what's actually verified vs. what isn't

This document exists because of an honesty constraint during the audit/fix
pass that produced this codebase's current state: the environment doing the
fixing had **no network access**, so `fastapi`, `pydantic`, `tiktoken`,
`httpx`, and `llmlingua` could not be installed. Pure-Python-stdlib code
could be tested directly and was. Anything depending on those packages could
be written and read carefully, but not executed — and this file says exactly
which is which, rather than letting that distinction quietly disappear.

## Verified with real, executed tests (stdlib only, no mocking of the code under test)

Run `python3 scripts/run_stdlib_tests.py` — 23 tests, last run: all passing.

- `security/redaction.py` — secret pattern matching (Anthropic/OpenAI/AWS/
  Slack/Stripe/JWT/Bearer keys), multimodal content handling (None content,
  list content with text+image blocks), image data passes through
  byte-for-byte unmodified.
- `compression/engine.py` — `CodeBlockGuard` round-trip losslessness,
  fenced/inline code detection, `CommentStripper`'s URL-in-string and
  hex-color-in-string preservation, real-comment stripping for both Python
  and JS-style markers (leading and trailing).
- `graph_memory/graph.py` — dirty-flag persistence mechanics against real
  SQLite I/O in a temp directory: first-persist-always-writes, dirty-clears-
  on-success, redundant-persist-is-skipped (verified via file mtime),
  `force=True` bypass, dedup-touch marks dirty, duplicate-edge does NOT
  mark dirty, data survives a full reload from disk.
- `graph_memory/hybrid_extractor.py` — `merge()` case preservation
  (corroborated/LLM-only items keep original casing, not lowercased),
  confidence tier assignment (0.95/0.80/0.65), regression check against the
  pre-existing corroboration test.
- `benchmarks/checkpoint_accuracy/runner_v3.py` — actually run end-to-end
  (default mode) against the real `SESSIONS` fixtures and the real
  `HybridExtractor.merge()` — this is what caught the case-folding bug in
  the first place, mid-audit, before it was fixed.
- Full-repo syntax check: every `.py` file in the repository compiles
  (`python3 -m py_compile`) with zero errors.

## Written and reviewed, but NOT executed in this environment

These require `pip install -e ".[dev]"` on a machine with network access.
The logic was reasoned through carefully and cross-checked against the
existing codebase's own patterns (e.g. `_deduplicate()`'s correct
normalize-as-key approach was used as the reference for fixing `merge()`),
but "carefully reasoned" is not the same claim as "tested," and this file
exists specifically so that distinction isn't lost.

- `tests/unit/test_security.py` — the `TestAuthFailClosed` and
  `TestInjectionDetection` classes use `pytest.mark.asyncio` and FastAPI's
  `HTTPException`, both of which need `fastapi`/`pytest-asyncio` installed.
  **This is the highest-priority thing to run** — it covers the fail-open
  auth fix, which is the single most severe finding in the whole audit.
- `tokenmizer/api/app.py` changes (auto-checkpoint retry logic, redaction-
  at-ingestion, `checkpoint_status` in the response payload) — needs the
  full FastAPI app running to exercise the actual HTTP request path. The
  underlying pieces it calls (`redact_messages`, `CheckpointManager.create`,
  `StateBackend.set`) are independently tested above; what's NOT verified
  is their wiring together inside the real endpoint.
- `tokenmizer/state/backend.py`'s `RedisBackend` — needs a real Redis
  instance. `InMemoryBackend`'s logic is simple enough to have been read
  carefully, but "read carefully" is explicitly not "tested."

## To run the full suite for real

```bash
pip install -e ".[dev]" --break-system-packages   # or in a venv
pytest tests/ -v
python3 scripts/run_stdlib_tests.py                # redundant with pytest but
                                                     # zero-dependency, useful
                                                     # for quick CI smoke checks
python3 scripts/static_audit.py                     # unused-import / silent-
                                                     # failure-pattern scanner
python3 benchmarks/checkpoint_accuracy/runner_v3.py          # merge-logic fixtures
python3 benchmarks/checkpoint_accuracy/runner_v3.py --live   # real LLM, real cost
```

If `pytest` finds a real failure in the "written but not executed" category
above, that's the system working as intended — paste the output and it'll
get fixed, the same way the stdlib-testable fixes in this pass were
iterated on until they actually passed (see git log for two cases where
writing the test caught a real bug that static reading had missed: the
`merge()` case-folding bug, and a pre-existing trailing-comment regex bug
in `CommentStripper` that predated this audit entirely).
