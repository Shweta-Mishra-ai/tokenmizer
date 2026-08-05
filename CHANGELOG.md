# Changelog

## [0.5.0] — 2026-08-05 — cross-process safety, verified benchmarks, launch prep

### New — graph writes are safe across processes
Per-row storage (0.4.2) stopped concurrent writers from destroying whole
sessions, but not from undoing each other. A worker holding a pre-prune
view of a session would faithfully write its stale node set back,
reinstating every row another worker had just deleted.

`persist()` is now a read-modify-write under an OS-level advisory lock
(`fcntl` on POSIX, `msvcrt` on Windows), one lock file per session. Under
the lock, an instance reconciles against what is actually on disk and
adopts another process's deletions instead of reverting them. Rows it
created but never persisted are untouched — new work is never mistaken
for stale state.

Measured (`benchmarks/persistence/runner.py`): **4 OS processes writing
one session, 100/100 nodes persisted, zero lost.** A stale writer no
longer resurrects a prune.

Still per-process, and now stated as such in the startup warning, the
README and SECURITY.md: rate-limit buckets (so limits apply *per worker*),
analytics counters, and the semantic cache.

> `flock` is unreliable on NFS. Keep `storage_dir` on a local filesystem
> for multi-process use.

### New — benchmarks that produce the numbers in the README
- `benchmarks/persistence/runner.py` measures write amplification,
  no-op persist cost, persist latency, 4-process concurrency, and the
  stale-writer case.
- `benchmarks/graph_retrieval/runner.py` **was broken** — it called
  `GraphMemory(storage_dir=...)` without the required `session_id` and
  crashed on every run, while being referenced as a benchmark. Fixed.

### Changed — README benchmark figures now match the runners
The quoted table was measured on **v0.2.4** and no longer matched what
the committed runner produced. Re-measured on 0.5.0:

| | Task | Decision | File | Info preserved |
|---|---|---|---|---|
| was (v0.2.4) | 76% | 85% | 100% | 87% |
| now (v0.5.0) | 76% | **92%** | 100% | **89%** |

Per-session spread and the n=3 sample size are now stated inline rather
than in a footnote, because a three-session synthetic benchmark is
directional and should read that way.

### Fixed — documentation that described features which do not exist
- **SECURITY.md advertised encryption at rest**, complete with an
  `encrypt_storage` / `encryption_key` config block. No such setting has
  ever existed. Replaced with what to actually do (filesystem/volume
  encryption) and a note about file permissions.
- **SECURITY.md "Known Limitations" listed the IDOR as unfixed** ("any
  caller who knows a session_id can access its graph") after 0.4.1 closed
  it. Replaced with a Session Isolation section describing the real
  model, and a limitations list that is current.
- **USAGE.md documented Redis-backed state and multi-provider routing**
  as production features. Neither is implemented.
- The supported-versions table still said `0.2.x`.
- `tokenmizer/storage/__init__.py` described a unified storage protocol
  that nothing imports or conforms to.

### Changed — packaging and release readiness
- `py.typed` added and shipped in the wheel (PEP 561), so the type
  annotations are actually visible to consumers instead of being
  discarded as `Any`.
- Classifiers: `Development Status :: 4 - Beta` (from `3 - Alpha`),
  Python 3.13, `Typing :: Typed`, AI topic. Added `Changelog` and
  `Security` project URLs and an explicit sdist manifest.
- Wheel and sdist build clean and install into a fresh venv with a
  working `tokenmizer` entry point.

### Changed — CI actually verifies the deployment story
- The Docker job now runs on **pull requests**, not only `main`.
- New steps assert the image works **with the network removed**
  (`--network none`): token counting must use the baked tiktoken
  vocabulary rather than the char/4 fallback, and graph persist + reload
  must succeed. Previously the vocabulary bake was claimed but never
  verified, so it could have silently stopped working and only surfaced
  in an air-gapped deployment.
- New `migration` job seeds a database using the **previous release tag**
  and asserts this version reads it without loss. Nothing else in the
  suite exercises a real cross-version upgrade.

### Added — tests
`tests/unit/test_multiprocess.py` (7 tests) covers the stale-writer
case, reconciliation not eating unpersisted work, another process's
additions surviving, lock exclusivity across real subprocesses, distinct
sessions not contending, hostile `session_id` values not escaping the
lock directory, and 4-process lossless concurrency.

**496 tests, 77% coverage, ruff clean.**

## [0.4.2] — 2026-08-05 — per-row storage (#27), provider fixes, comment cleanup

Closes the three items left open by 0.4.1.

### Changed — graph storage is now per-row (schema v2), closes #27
`graphs.nodes_json` / `edges_json` (one JSON blob per session) is replaced
by `graph_nodes`, `graph_edges` and `graph_meta`, one row per node and
per edge.

Two problems the blob layout caused:

- **Write amplification.** Every persist rewrote the whole graph, once
  per chat turn, even for a single new node. Measured on a 151-node
  graph: adding one node wrote **1 row instead of 151**, and a turn that
  changes nothing now writes **0 rows** instead of the entire session.
  The 200-node auto-prune cap existed largely to bound this cost.
- **Concurrent writers destroyed sessions.** Two processes holding one
  session each wrote the complete blob, so the later save discarded
  everything the earlier one had added. Disjoint changes from two writers
  now merge; only a genuine same-node conflict is last-writer-wins.

Corruption is also contained: one unreadable row costs that row, where
one bad byte in a blob cost the entire session.

**Change detection is derived, not tracked.** Rather than a dirty-set
that every mutation site must register with — node state is mutated from
at least six places, and a site that forgot would silently never reach
disk — persist() serializes the graph and compares each row against the
exact string last written. Serialization is O(nodes) CPU with no I/O;
the part that costs (SQLite writes, WAL, fsync) is O(changed). Nothing
can be missed because nothing has to be remembered.

**Migration.** Automatic and per-session: the first time a session with
no v2 rows is opened, its v1 blob is read, hydrated, and written out as
rows. A database holding a mix converges one session at a time. The v1
row is **deliberately not deleted**, so downgrading to a pre-migration
build still finds the data it expects as of the moment of migration —
changes made after the upgrade are lost on downgrade, which is the normal
meaning of a rollback.

### Fixed — provider adapters (the modules skipped by both audit passes)
- **Retryable-error detection matched substrings.** `"rate" in err`
  fires on `"gene**rate**"` and `"mode**rate**"`, so `Failed to generate
  completion` — a permanent failure — was retried three times, costing
  the caller 1+2+4s before returning the same error. Now uses the OpenAI
  SDK's typed exceptions, with word-boundary matching only as a fallback.
- **Anthropic prompt caching never engaged.** `cache_control` was
  attached when the system prompt exceeded 800 *characters* (~200
  tokens), but Anthropic's minimum cacheable prefix is 1024 tokens (2048
  for Haiku) and it silently ignores `cache_control` below that — so the
  advertised "L5 Prompt Cache — 90% on repeated system prompts" almost
  never applied. Thresholds are now in tokens, per model family. The
  streaming path never applied caching at all; it now uses the same rule.
- **Provider/model mismatch was silent.** `default_model` defaults to a
  Claude model, so switching only `provider` to `openai` sent
  `claude-sonnet-4-6` to the OpenAI API and produced an opaque
  model-not-found error. Now warns at startup naming the actual cause.

### Fixed — file intelligence overshot its token budget
`token_budget` is the module's entire contract (the `analyze_file` MCP
tool documents it as "Max tokens for the summary"), but the log and code
strategies assembled a fixed set of sections and only then measured,
overshooting by ~5%. A single budget clamp now applies to every strategy
at the exit point, reserving room for its own trim marker.

### Changed — comments no longer narrate past bugs
Roughly 90 comment blocks across 20 files described bugs that had already
been fixed (`FIXED (TM-xx): previously this…`), in one case running 25
lines inside a docstring and 17 lines inside `pyproject.toml`. Two
concrete harms, both observed during the 0.4.1 audit: the archaeology
buried the logic (finding the decision-merge conflict meant reading past
three unrelated fix narratives), and some of it asserted protections that
no longer held — a comment explained at length why cache eviction was
safe because of a lock the request path never actually took.

Comments now state why the current code is the way it is, in the present
tense, keeping the constraint and dropping the history. Load-bearing
rationale is preserved (e.g. why `use`/`using` must stay out of the stop
words, why set iteration must not decide which extracted items survive).
No logic changed; the full suite passes unchanged either side.

## [0.4.1] — 2026-08-05 — second audit pass: correctness, durability, isolation

Findings from a full-codebase audit. Every item below was reproduced
before being fixed and has a regression test in
`tests/unit/test_audit_fixes.py` or `tests/unit/test_durability.py`.

### Fixed — decision supersession silently discarded changes (critical)
`_is_same_decision` merged any two decision labels sharing ≥82% of their
words. Two decisions in the same slot differ by exactly one word — the
technology name, which is the entire meaning — so the check got *more*
wrong as labels got more descriptive:

| Labels | Overlap | Old result |
|---|---|---|
| `Use MySQL` / `Use MongoDB` | 0.50 | correctly distinct |
| `Use MySQL for the user database` / `Use MongoDB for the user database` | 0.83 | **merged** |

A merged swap produced no new node, no `SUPERSEDED` status, and no
`DecisionTransition` — so `/api/graph/{id}/why` and the `why_decision`
MCP tool returned nothing for precisely the case they exist to answer,
and three successive decisions collapsed into one node. Worse, once a
decision was `SUPERSEDED` or `INVALIDATED` it permanently poisoned its
topic slot: the replacement merged into the dead node, the
status-upgrade-only rule refused to revive it, and the session ended up
with **no** active decision on that topic.

Decisions naming different technologies from the same topic bucket are
now never treated as duplicates. Restatements and refinements (`Use
PostgreSQL` → `Use PostgreSQL 16 with pgvector`) still merge, and
re-adding a stale decision still cannot resurrect it.

### Fixed — resume surfaced superseded and invalidated decisions as current
`_build_critical` sorted *all* decision nodes by importance with no
status filter, so the ~100-token "must-know facts" block presented
choices the team had already moved off — and ones explicitly rejected via
`/api/decision/invalidate` — as `KEY DECISIONS`, while the current
decision could be crowded out. `query()` and `to_context_block()` had
always filtered these; the checkpoint resume path, the product's core
output, did not. All resume builders now go through one `_live_nodes()`
filter. Invalidated decisions are surfaced separately as `DO NOT REVISIT`
so the model doesn't re-propose them.

### Fixed — token counting took the whole proxy down (critical)
`_get_encoding` caught only `ImportError`, but tiktoken downloads its BPE
vocabulary from a CDN on first use. Any egress restriction, proxy, or CDN
outage raised a network error that propagated out of
`count_messages_tokens` — on the hot path of every request — so **every
request 500'd**, and the documented char/4 fallback was unreachable
(it only ran when tiktoken was absent entirely). Now fails soft, caches
the failure so it isn't retried per request, and the Dockerfile
pre-downloads the vocabulary at build time (`TIKTOKEN_CACHE_DIR`).

### Fixed — corrupt-DB recovery destroyed every session (critical)
`graph_memory.db` and `checkpoints.db` are each shared by *every* session
in a `storage_dir`. Recovery called `unlink()`, so a single bad read by
one session permanently deleted everyone's memory — and
`persistence_broken` stayed `False`, so `stats()` reported healthy over an
emptied database. Additionally `sqlite3.OperationalError` (which includes
routine `database is locked` contention) subclasses `DatabaseError`, so
ordinary write contention triggered that deletion.

Now: lock errors are treated as contention and destroy nothing; genuine
corruption is scoped to the affected session's row where possible;
otherwise the file is **quarantined by rename** so `.recover` can salvage
it. A graph that failed to *read* refuses to *write* over the stored row,
so a transient failure can't become permanent loss. Actual loss is
reported as `data_loss_detected` in `GET /api/graph/{id}`.

### New — mid-session durability guarantees
See README "Durability". Shutdown (SIGTERM) now drains in-flight
background extraction and force-persists every cached graph instead of
logging one line and exiting; a periodic flush bounds hard-kill exposure
to 30s; a graph whose persist fails is **kept in memory** rather than
evicted; and sessions with an in-flight request are never evicted (the
previous guard checked a lock only the background task ever took, so it
was inert for request traffic).

### Fixed — any caller could read or modify any session (security)
Session-scoped routes took `session_id` straight from the URL, with a
single shared deployment key as the only auth and no ownership model at
all. Since clients choose their own `session_id`, reading someone else's
session needed no guesswork, and `/api/decision/invalidate` made it a
write primitive. Sessions are now claimed by the first principal that
uses them; `api_keys` adds further credentials, each its own principal.
Denied requests return 404, not 403, so the endpoints can't be used to
probe which sessions exist. Dev mode and single-key deployments are
behaviour-compatible.

### Fixed — environment variables did not override `tokenmizer.yaml`
`from_yaml` passed the file's contents as `__init__` kwargs — the
highest-priority source in pydantic-settings, above env vars — so every
key present in the shipped (and Docker-`COPY`'d) config silently beat its
`TOKENMIZER_*` variable, contradicting that file's own header.
`TOKENMIZER_PROVIDER=openai` resolved to `anthropic`. API keys appeared to
work only because those lines happen to be commented out.

### Fixed — streaming cached truncated responses
On a mid-stream provider error the generator fell through to post-stream
bookkeeping and cached whatever partial text had arrived, serving that
truncated answer to every future matching prompt. Failed and cache-hit
streams no longer write to the cache, and non-`ProviderError` exceptions
are handled instead of killing the generator mid-stream.

### Fixed — smaller correctness and resource bugs
- `SemanticCache.invalidate()` was a **silent no-op** under the default
  `share_scope="session"`: it built its key with the `"__shared__"` scope
  that `set()` never uses. Now removes the entry and reports how many.
- `AnalyticsEngine._records` was unbounded and appended per request (with
  a second reference in `_by_provider`) — the only uncapped structure in
  the codebase. Now age- and count-bounded; lifetime counters are kept
  separately so totals don't shrink as records age out.
- Cost figures applied one blended rate to input *and* output tokens.
  Output costs up to 5× more, so `cost_saved_usd` was materially wrong.
  Rates are now per-direction.
- `CheckpointManager._prev_snapshots` held a full graph snapshot per
  session forever; now LRU-bounded.
- `stats()` excluded evicted nodes from `node_count` but not from
  `edge_count` — the same inconsistency TM-35 set out to fix, left
  half-done.
- `CheckpointManager._db_connect` leaked the handle on a failed PRAGMA
  (fixed in its graph twin, never propagated).
- `SUPERSEDES` edges were inferred by word overlap against *every*
  superseded decision, inventing causal history that `/why` then reported
  as fact. Only the edge created alongside a real `DecisionTransition`
  remains.
- Rate limiting keyed on the connecting address, so every client behind a
  load balancer shared one bucket. Opt-in `trust_proxy_headers` reads the
  forwarded address, indexed from the right by `trusted_proxy_hops` (the
  leftmost `X-Forwarded-For` entry is caller-controlled).

### Changed — settings that did nothing now do something, or say so
`compression.min_tokens_to_compress`, `graph_checkpoint.max_resume_tokens`,
`graph_checkpoint.extraction_model` and `memory.enabled` were all
documented but read by nothing; they are now honoured. `routing.*` has no
implementation at all — it is kept so existing configs load, but logs a
warning at startup and is labelled NOT IMPLEMENTED in the README and
config file.

### Removed — Redis from `docker-compose.yml`
The stack ran a Redis container, gated startup on its health check, and
persisted a volume for it. `tokenmizer/state/backend.py` has no callers —
not a byte was ever written to Redis. Removed rather than left as
advertised-but-fake production readiness. `stop_grace_period: 30s` added
so the shutdown flush can complete.

## [0.4.0] — 2026-07-11 — from storage to reasoning: ontology + graph reasoning

### New — TokenMizer Ontology
- `tokenmizer/graph_memory/ontology.py`: the formal, machine-readable
  vocabulary of the graph — every node type with semantics, every edge
  type with domain/range/semantics, and the status **state machine**
  (which lifecycle transitions are legal, e.g. COMPLETED→SUPERSEDED→
  ARCHIVED; SUPERSEDED can never silently become COMPLETED again).
- Served at `GET /api/ontology` for MCP clients, docs, and tooling.
- Design principle: the ontology describes and audits, it does not gate
  writes — graph ingestion stays permissive; violations are surfaced by
  the consistency audit instead of causing silent data loss.

### New — Graph Reasoning (`tokenmizer/graph_memory/reasoning.py`)
- **`why()`** — "Why is X the current choice?" Walks the supersession
  chain in both directions and returns the old→new trail with trigger,
  reason, and evidence per hop, plus the currently active decision.
  `GET /api/graph/{id}/why?q=react`
- **`impact()`** — typed 1-hop neighborhood: which files/tasks/errors
  connect to a node and via which relation.
- **`decision_history()`** — decision timeline grouped by topic bucket.
- **`consistency_check()`** — ontology-based audit: two active decisions
  sharing a topic (contradictions the tracker missed), SUPERSEDED
  decisions with no transition record (lost history), transitions
  referencing pruned nodes.
- **`GET /api/graph/{id}/reasoning`** — the combined reasoning view.
- All reasoning is deterministic and local — no LLM calls.

### New — MCP tool `why_decision` (6 tools now)
- Ask your agent "why did we pick X?" — it traces the decision trail:
  struck-through old choices, replaced-by hops with reasons/evidence,
  and the current active choice. Covered in unit tests and the e2e check.

### Changed
- `glama.json` added (Glama MCP directory maintainer verification).
- README: "From Storage to Reasoning" section; internal demo-script and
  planning docs removed from the repository.
- Version 0.4.0 everywhere (enforced by test_version_consistency).

## [0.3.2] — 2026-07-10 — full-repo audit: graph memory, MCP server, visualization

### Critical — the LLM extraction path never worked
- **`api/app.py`:** the background extraction called
  `HybridExtractor(provider_fn=_pfn)` — a kwarg `__init__` never accepted —
  so it raised `TypeError` on EVERY call, and even without that,
  `ext.extract(_msgs)` omitted `provider_fn`, which defaults to `None` and
  skips the LLM pass. Net effect: **`use_llm_extraction: true` has never
  once produced an LLM extraction** — every call raised, was caught by the
  broad except, and logged as if it were a transient provider failure.
  Fixed (`HybridExtractor()` + `extract(_msgs, provider_fn=_pfn)`) with a
  regression test that replicates the exact call pattern.

### Graph memory — topic classifier
- **"Go" the verb collided with Go the language:** "Go with tRPC for the
  API layer" classified as `language` (first-single-word-hit-wins never
  reached "trpc"), so a later "use gRPC instead" never superseded it.
  Bare "go" removed from language keywords; unambiguous forms kept
  ("golang", "in go", "use go", ...).
- **Vocabulary drift:** hybrid_extractor.py knew supabase/clerk/etc., the
  classifier didn't — those decisions classified as None and supersession
  silently never fired. New buckets: backend_platform, auth_provider,
  payments, observability, state_management, package_manager, styling.
- **Multi-topic decisions collapsed to one topic:** "Use FastAPI with
  SQLAlchemy and PostgreSQL" returned only `web_framework`; a later
  Postgres→SQLite switch was never detected as contradicting it. New
  `classify_topics()` returns ALL matched topics; contradiction detection
  is now set-intersection. Bigrams match first and consume their words
  ("session store" no longer leaks a spurious `auth_mechanism`).
  `classify_topic()` kept as a backward-compatible wrapper.
- **`ARCHIVED` was unreachable:** documented, fully wired into decay/
  prune/query logic — and nothing ever set it. The README advertised a
  4-state model whose 4th state could not occur. SUPERSEDED decisions now
  age into ARCHIVED after 7 days (from supersession time, not creation).
- `/api/decision/invalidate` substring-matched across ALL decision nodes
  regardless of status — a short label could flip already-SUPERSEDED
  history nodes to INVALIDATED, destroying their supersession record.
  Now only ACTIVE decisions are eligible, and the response lists exactly
  which nodes were affected.

### MCP server — hardened like a client integration
- **`isError` was string-sniffing** (`result_text.startswith("❌")`): a
  missing required argument produced `"Tool error: 'session_id'"` with
  `isError: false` — MCP clients saw a *successful* result. Handlers now
  return `(text, is_error)` structurally; missing/invalid args produce
  typed validation errors with `isError: true`.
- **Three server-killer inputs fixed:** an exception inside `initialize`
  crashed the whole stdio loop with no JSON-RPC error (client hung, then
  watched the subprocess die); a valid-JSON-but-not-an-object line
  (`[1,2]`) raised AttributeError and killed the loop; malformed JSON was
  silently dropped (`continue` — the request hung forever, no log, no
  error). Per-request try/except now returns `-32603`, non-objects get
  `-32600`, parse failures get `-32700` + a warning log.
- **`logger` was dead code** — defined, never called once. Every caught
  exception was invisible to the operator. `run_stdio_server()` now
  configures stderr logging (stdout stays protocol-only), controlled by
  `TOKENMIZER_MCP_LOG_LEVEL`.
- Input validation: `session_id`/`file_path` presence+type, `level` enum,
  `token_budget` positive-int (bool explicitly rejected — it's an int
  subclass), non-dict `arguments`.
- 18 new unit tests (`tests/unit/test_mcp_server.py`) covering all of the
  above; `scripts/mcp_e2e_check.py` still ALL PASS.
- `server.json` version synced (was 0.2.6, three releases stale).

### Silent failures made visible (continuing the v0.2.x hardening)
- `hybrid_extractor.llm_extract`: debug→warning (a provider outage
  silently degraded every turn to heuristic-only).
- `compression/engine.filter_json`: swallowed ALL exceptions with zero
  logging (returned unfiltered content). Real failures now log at
  warning; the benign not-JSON case stays quiet.
- `graph._load_transitions`: DB corruption looked identical to the benign
  first-run case (both debug). Corruption now warns; first-run stays debug.
- `file_intelligence.detect_file_type`: sniff failure silently reclassified
  files as "text" (worse extraction, no signal) — now warns. PDFs where
  NO page yields text now warn once (scanned/image-only detection).
- `CheckpointManager`: new `persistence_broken` flag — previously the
  constructor swallowed a triple init failure and reported healthy while
  every subsequent save was doomed. `get_latest()` now raises
  `StorageError` on DB read failure instead of returning None ("no
  checkpoint found, 404" vs "your checkpoints exist but are unreadable"
  are different problems; callers can finally tell them apart).
- **`_db_connect` leaked the connection when the WAL PRAGMA failed on a
  corrupt DB file** (both graph.py and manager.py) — on Windows the open
  handle blocked the documented delete-and-recreate recovery path
  (WinError 32). Found because the chaos test failed for the RIGHT reason
  once get_latest stopped swallowing errors.

### Security — redaction gaps
- URL-embedded credentials (`postgres://admin:pass@host/db`) matched NO
  pattern — no `password=` literal, no recognized prefix. Now redacted
  (credential part only; host survives for readability).
- Added OpenRouter / Hugging Face / xAI / Together key patterns.
- Checkpoint `next_action` (a raw 200-char message slice persisted to
  SQLite) is now independently redacted — defense-in-depth so the
  single-point-of-application assumption in chat_completions() is not the
  only thing between a pasted key and the checkpoint DB.

### Visualization — redesigned (was a generic node soup)
- The old shareable HTML exported `transitions` (the supersession
  history — the one thing this product tracks that a generic graph view
  doesn't) and then never rendered them. It also loaded D3 from a CDN, so
  the "self-contained" artifact broke offline.
- New artifact (zero external deps, hand-rolled force layout):
  supersession arcs (dashed red, old→new, arrowheads), a clickable
  "Decision history" timeline panel (struck-through old label → new label
  with trigger/reason/timestamp; click = spotlight both nodes + center),
  glow rings on active decisions vs dashed rings + strikethrough on
  superseded/archived, red rings on invalidated, per-type filter chips,
  "Active only" toggle, text search, wheel-zoom/pan, one-click PNG export.
- Verified in a real browser (not just unit-tested): filters, search,
  timeline spotlight, and layout all exercised via DOM inspection.

### CLI
- `tokenmizer --help` crashed on Windows (cp1252) with UnicodeEncodeError
  from the 🧠 emoji — found by dry-running the README quick start. Same
  UTF-8 reconfigure fix the benchmark runners got in v0.2.4; the CLI had
  been missed.

### Round 2 — extraction-quality residuals (same audit, follow-up pass)
- **Near-duplicate decision nodes merged instead of self-superseding:**
  one message ("Decided: use React for the frontend.") could emit two
  decision variants ("Use React" + "use React for the frontend.") via
  different regex passes; they became two nodes and one superseded the
  other — a bogus "Changed:" line in every resume. `_is_same_decision`
  now recognizes containment (smaller label's words ⊆ larger's, min 2
  words so "Use PostgreSQL" is NOT collapsed into "Switch from PostgreSQL
  to SQLite"), and `add_node` fuzzy-merges same-decision variants: keeps
  the existing node, upgrades to the longer label, backfills the summary,
  never resurrects SUPERSEDED status. Verified end-to-end: the demo
  scenario now produces exactly one transition (React→Next.js) and a
  clean resume block.
- **Validator now honors extractor corroboration confidence:** it used to
  recompute confidence purely from label length/wording, so a doubly-
  corroborated short decision (0.95) could be rejected while a verbose
  weakly-sourced one passed. `validate()` gains `extractor_confidence`;
  final = max(heuristic, (heuristic+extractor)/2) — monotone (evidence
  only raises), not an override (heuristic-only 0.65 still fails a 0.65
  threshold), and hard rejects remain absolute (0.95 cannot resurrect
  junk). Wired from `add_node` via the existing confidence≠0.7 sentinel.
- **`HybridExtractor.min_confidence` is no longer dead code:** extract()
  now filters merged output by merge()'s confidence tiers — default 0.55
  keeps every tier (behavior unchanged); 0.7 drops heuristic-only items;
  0.9 keeps only corroborated ones. Decisions filter per-item, simple
  lists per-category.
- Test fixture fix: `test_prune_preserves_decisions` used ten decisions
  differing only by a trailing digit — 83% word-overlap, which
  `_is_same_decision` always considered the same decision; the merge fix
  made that judgment consequential, collapsing the fixture. Replaced with
  ten genuinely distinct decisions across different topic buckets.

### Tests
- 220 → 275 (55 new: MCP server 18, classifier+dedup 18, archival 3,
  invalidate-scope 2, redaction 5, LLM-pass regression 1, validator
  blending 4, min_confidence filter 4).

## [0.3.1] — 2026-07-03 — shareable graph visualization

- NEW `GET /api/graph/{session_id}/html` — self-contained dark interactive
  force-graph page (D3): drag, zoom, glow-styled typed nodes, legend, live
  stats header. Open in any browser, screenshot, share. Your session's
  memory, visible.

## [0.3.0] — 2026-07-02 — true SSE streaming

- **`stream: true` now works** — real passthrough streaming in OpenAI
  `chat.completion.chunk` SSE format for Anthropic, OpenAI, DeepSeek,
  Mistral, OpenRouter, Grok (OpenAI-compatible base) and Ollama. Cursor,
  Continue.dev and every streaming client can now point at TokenMizer
  without config changes.
- All input-side layers (file intelligence, compression, graph memory,
  context injection) apply to streamed requests; output trimming is
  skipped in stream mode by design. Cache hits stream as a single chunk;
  post-stream analytics + cache writes preserved.
- Providers without passthrough support return an explicit 501 (never a
  fake buffered stream). Mid-stream provider failures emit an SSE `error`
  event instead of silently truncating.

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

## Shipped in [0.2.4] — tokenizer, cache, and version consistency fixes

(Header fixed 2026-07-10: this section was left titled "[Unreleased]" after it shipped.)

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

## Shipped in [0.2.4] — security/correctness audit pass

(Header fixed 2026-07-10: this section was left titled "[Unreleased]" after it shipped.)

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
