# Changelog

## [0.5.0] — 2026-08-06 — audit, durability, and measured extraction quality

Everything between 0.4.0 — the last version published to PyPI — and here,
shipped as one release.

The six sections below were development milestones, not releases. They
previously carried version headers of their own (0.4.1 through 0.7.0), which
implied six installable versions that never existed. The work is unchanged;
only the numbering is, so that every heading in this file corresponds to
something you can actually `pip install`.

---

### Extraction quality: macro F1 74% → 94%

The eval harness built in the previous milestone was pointed at itself.
Everything below is a defect it found, with the before/after it produced.
Corpus grown from 11 sessions to 14 (6 real transcripts, up from 3).

| Category | before | after |
|---|---|---|
| Files | 93% | **99%** |
| Pending tasks | 64% | **95%** |
| Decisions | 66% | **92%** |
| Completed tasks | 71% | **91%** |
| Errors | 75% | **94%** |
| **macro** | **74%** | **94%** |
| real transcripts only | 65% | **90%** |

Part of that jump is better extraction and part is a corpus that was
wrong. Both are itemised below rather than blended into one number.

#### Fixed — a turn stating two decisions lost the second one
`(.{5,80})` ran straight past the full stop, so "Decided: X. Decided: Y."
produced ONE match spanning both. `_clip()` kept the first clause and Y
was gone for good: `finditer` does not re-scan a span it already
consumed, so the second keyword was never looked at. Every capture now
stops at the end of the sentence it started in, with a lookahead so
`moment.js`, `React.lazy` and `go.mod` are not split on their own dots.
Decisions 66% → 88% F1 from this change alone; completed tasks +3.

#### Fixed — present-tense intent recorded as finished work
`migrated?`, `fixed?` and `removed?` also matched the present tense, so
"We need to migrate 40M rows to Postgres" in an opening turn became a
completed task. Present tense is the one reliable signal that something
has *not* happened yet. Completion verbs are now past tense only.

#### Fixed — completed work resurfacing as outstanding work
"Fixed by adding a 5 second context timeout" was recorded as in-progress;
"Completed: four OS processes writing one session" as WIP; "was missing
email validation" as a to-do. A resume built from those tells the next
session to redo work that is already merged. A completion verb earlier in
the same clause now disqualifies the rest of it, and the opening turn's
goal no longer doubles as a WIP item. Pending tasks 50% → 100% precision.

#### Fixed — the validator silently dropped the most precise error labels
A named exception is the most specific form an error label can take, and
the generic length/word-count penalties punished exactly that: `ProxyError`
scored 0.55 against a 0.65 threshold and never reached the graph, while
the vaguer "500 on an air-gapped host" scored 0.90. `IDOR` was rejected
twice over — once by the score, once by a flat 5-character minimum.
Identifier-shaped errors and filenames now clear the bar on their own
evidence. Same defect on files: `go.mod` scored 0.50 and `Dockerfile`
0.40, so neither could ever be extracted.

#### Added — three classes of failure the patterns could not see
* **Data-loss prose.** "The later save discards everything the earlier
  one added" has no exception name, status code or symptom noun. These
  are the failures most worth carrying into a resume.
* **Vulnerability classes.** `IDOR`, `XSS`, `CSRF`, `SSRF`, `RCE`,
  `TOCTOU`, `CVE-…` are named, not described.
* **Extensionless files.** Every file pattern keyed on a dot, so
  `Dockerfile` and `Makefile` were unextractable.

#### Fixed — errors that were not errors
* A status code named in a *decision* ("Decided: 404 rather than 403")
  was recorded as two failures. A code now needs evidence it was
  received.
* An exception named as one that is *caught* ("catches only ImportError")
  is part of the handling code, not a failure that happened.
* "no longer resurrects a prune" describes the fix, not the bug.
* A user turn reading only "any regressions" is a question.
* One greedy match swallowed the errors named after it: in
  "OperationalError subclasses DatabaseError, and OperationalError covers
  database is locked", `database is locked` was unreachable.

#### Fixed — the corpus itself, and a guard so it stays fixed
Two ground-truth labels were not in their transcripts at all
(`CUDA out of memory` in a session that never mentions CUDA; `Chart.js`
in a session that never mentions Chart.js). Such a label is unreachable
for *any* extractor, heuristic or LLM, so it caps recall at a number no
code change can move — silently. `benchmarks.eval` now refuses to score a
corpus containing one.

The rest of the ground truth was inconsistent in the other direction:
`react_dashboard` labelled four decisions, three of which the transcript
never states as decisions, while three that it does state explicitly were
unlabelled — so the extractor was penalised on recall for not inventing
labels and on precision for being right. All 14 sessions were relabelled
against one written rule (`benchmarks/eval/corpus.py`), applied
exhaustively including where it costs us: superseded decisions are still
decisions, and every file named in a turn is labelled.

Roughly half the macro gain is extraction and half is the corpus. The
split is stated here rather than hidden, because a benchmark you also
own the labels for is worth exactly as much as its labelling rule.

#### Fixed — a 6.3-second scan on a 15 KB message
The subject windows in the error patterns were written as a token repeat
nested in a window repeat, `(?:[\w./-]+\s+){0,3}`. The inner `+` can give
back inside every token and the outer `{0,3}` multiplies the alternatives,
so a message built from `"word."` took 6.3 seconds to scan — on the hot
path of a proxy that scans whatever a caller sends it. Python 3.10 has no
atomic groups to fence it with, so the windows are now flat bounded runs
and the sentence-boundary problem they were solving is handled after the
match instead. Same message: 130 ms.

Flattening them introduced a second defect, caught by its own test: a
window that can end mid-word let the symptom vocabulary match *inside* a
word — `race` in "All of them trace", `hang` in "single-user use is
unchanged". The windows now end on whitespace by construction.

#### Fixed — the negation guard suppressed real failures
Added to stop "no longer resurrects a prune" being reported as a current
bug, it fired on any "no"/"not" in a 60-character window — and error prose
is full of them. "WebSocket message **not** triggering re-render — was
missing dependency in useEffect" lost its second failure entirely. Only
the phrases that actually mean *fixed* are excluded now.

#### Fixed — one failure named twice became two nodes
"Login keeps returning 422" and "Fixed: 422 error — missing email
validation" is one bug. Word-overlap dedup could not see it: the two share
only the digits. Errors carrying the same status code or exception class
now collapse to the one that says what actually broke.

#### Added
* `benchmarks.eval.corpus.validate_grounding()` and `ungrounded()`.
* Error patterns for absence defects ("missing dependency in useEffect"),
  inert code ("the fallback is unreachable") and vulnerability classes.
* 34 tests, including a floor on real transcripts scored separately so the
  synthetic half cannot carry the headline, and a scan-cost bound that
  fails if the patterns return to superlinear. 533 → 584 tests.

#### Fixed — the lock sweep never removed anything on Windows
`sweep_stale_locks` unlinked the lock file while its own handle was still
open. POSIX allows that and it is the safer order — the directory entry
goes before anything else can acquire it. Windows refuses with a sharing
violation, so the sweep silently removed nothing and lock files grew
without bound on every Windows deployment. The handle is now released and
closed before the unlink on Windows, which reopens the residual race the
30-day threshold already covers; POSIX keeps the tighter order.

Two Windows-only test defects surfaced with it, both of which had been
hiding real coverage: a lock-contention test passed a closure to
`multiprocessing`, which cannot be pickled where processes are spawned
rather than forked, and a CLI test hardcoded `/tmp` as "a path that is not
a file" — a path that does not exist on Windows, so the CLI correctly
answered "not found" and the assertion failed for a reason unrelated to
the behaviour under test.

#### Fixed — the embedding model could raise on the request path
`EmbeddingEngine._load` caught only `ImportError`. sentence-transformers
ships no weights: `SentenceTransformer(...)` downloads them from
huggingface.co on first use, so with the package installed and the model
not cached — air-gapped host, egress proxy, Hub outage, rate limit — it
raised `OSError` out of `embed()`, which the cache lookup calls while
serving a request. The semantic cache is an optimisation; it must degrade
to exact match, never fail the request that was only trying to use it.

This is the same defect as the tiktoken one fixed earlier in this
release, in a second place: a lazily-downloaded asset behind a
catch-too-narrow, on a hot path.

Two things had kept it invisible. The test suite replaced `_load` with a
no-op for every test, so nothing exercised it — the stub is now a fixture
a test can opt out of. And the Dockerfile baked the model in a step that
hard-failed the build, which meant every image build depended on
huggingface.co being reachable *and* not rate-limiting; it duly broke CI
on a Hub hiccup. That step retries once and then continues, and CI now
asserts the image serves cache hits with `--network none` and no model.

The tiktoken bake stays hard-required, and the difference is the point:
token counting runs on every request, the embedding model does not.

#### Fixed — the first two commands a new reader runs
Found by installing the built wheel into a clean virtualenv rather than
trusting the repo checkout.

`tokenmizer analyze` on a small CSV printed **"-536% smaller"**. A digest
can legitimately be bigger than its source — three rows become a schema,
column types and summary statistics — but reporting that as a negative
percentage reads as a broken program on someone's very first command.

`tokenmizer stats` with nothing running printed **"Cannot reach server:
[Errno 111] Connection refused"**. That is what the stack knows, not what
the reader needs, which is `tokenmizer serve`. Connect and timeout errors
now say what to do; anything else still shows the real error.

#### Changed — the README is 257 lines instead of 1062
It had grown to document everything in one file: architecture, every
configuration key, deployment, the full API surface, three benchmark
suites, comparisons and the roadmap. That is a reference manual wearing a
README's clothes, and the effect is that nobody reads either.

The README now answers the three questions someone has in their first
minute — what is this, does it work, how do I try it — and links out. The
detail moved to `docs/`, unabridged:

| | |
|---|---|
| `docs/architecture.md` | Request pipeline, graph data model, decision lifecycle |
| `docs/configuration.md` | Every setting, environment variables, precedence, providers |
| `docs/api.md` | Endpoints, CLI, MCP tools, Claude Code integration |
| `docs/deployment.md` | Docker, multiple workers, durability, isolation, security |
| `docs/benchmarks.md` | Extraction, memory and storage numbers, and running your own |
| `docs/comparisons.md` | Mem0, Zep, longer context windows, roadmap |

The two guards that read the README — the endpoint table check and the
test-count check — now scan `docs/` as well, so moving content out cannot
quietly disable them. Every internal link and heading anchor across all
eight files is verified.

#### Fixed — the documentation split dropped things it should not have
Two of them, both caught by review rather than by any test, which is why
both now have one.

**Contributor credit.** The README's Contributors section went with the
split — three people and six merged pull requests, removed by a
restructure that nothing was watching. Credit is the only thing an
outside contributor gets, and losing it in a refactor is worse than never
having written it. Restored, and `test_every_merged_contributor_is_
credited` fails if a credited handle ever disappears again.

**Setup.** The Claude Code plugin, the MCP server configuration and the
proxy wiring all moved to `docs/`, leaving a README that explained the
design well and never showed how to actually use the thing. Those are the
first thing anyone needs. They are back above the fold as "Use it from
your tools", with a guard asserting the README still shows how to
install, how to add the plugin, how to configure MCP, and how to point a
client at the proxy.

The support section also stopped explaining itself. A star first, a
sponsorship link second, both optional, no paragraph defending the fact
that the project is free.

#### Fixed — the documentation split produced broken markdown structure
The split concatenated whole README sections onto new pages by demoting
every heading one level uniformly, and by appending a footer that was
sometimes already there. That silently produced three defects, on every
one of the six new pages, that render fine as raw text and only show up
in the rendered view: a second H1 duplicating the page's own title
(`# Architecture` immediately followed by another `# Architecture`), a
jump from H1 straight to H3 with nothing in between, and — because the
footer-append step ran twice on `docs/deployment.md` — a stray
`## Releasing` section stranded after a premature "back to the README"
link, followed by a second one.

Fixed across all six pages: exactly one H1 per page, no level skips, one
footer. `test_docs_pages_have_sane_heading_structure` checks the
structure directly rather than the text, and was mutation-tested by
reproducing the original bug.

Two stale test counts (578, 584) were also found and corrected to 600
while fixing this.

#### Known limits
Errors remain the weakest category at 94%. Across 172 labelled items it now
misses four and invents three; the residue is a defect stated as a
measurement ("error recall is 8 percent") and one failure named twice in
words that share no tokens. That is what `use_llm_extraction` is for. n=14 is a small sample and one author wrote every label.

### Real transcripts in the corpus, and the gaps a sweep found

#### Fixed — the logo advertised v0.2.3 while the package was five releases ahead
On the README, above the fold, the first thing anyone sees. A version
baked into an image goes stale on every single release and nothing
catches it, so the badge now reads "GRAPH MEMORY" instead of a number.
The PyPI badge beside it renders the real published version live from the
registry and cannot drift.

`tests/unit/test_version_consistency.py` now fails the build if any
committed SVG contains a version-shaped string, so this cannot recur.
`scripts/mcp_e2e_check.py` compared the served version against a
hardcoded `"0.2.3"` sentinel — meaningless the moment 0.2.4 shipped; it
now compares against `tokenmizer.__version__`.

#### New — the eval corpus is no longer 100% synthetic
Three sessions condensed from real TokenMizer audit work (the 0.4.2
storage migration, the 0.5.0 concurrency work, the 0.6.0 eval harness)
join the eight hand-written fixtures: **11 sessions, 116 turns, 122
labelled items, 11 domains.**

They are scored separately, and the harness prints the split on every
run:

| Corpus origin | Sessions | Macro F1 |
|---|---|---|
| Synthetic (hand-written) | 8 | **75%** |
| Real (captured transcripts) | 3 | **65%** |

**A 10-point drop from fixtures to real data.** That gap is the honest
measure of how far the heuristics are fitted to text we wrote ourselves.
It is now the headline number in the README rather than a footnote,
because it is the first thing a careful reader would ask for and the
last thing a project wanting to look good would publish.

Overall on the full corpus: files F1 93%, errors 75%, completed tasks
71%, decisions 66%, pending tasks 64%, **macro 74%**.

#### New — `tokenmizer analyze` and `POST /api/analyze`
File analysis was reachable only from inside Claude Code, via the plugin
skill. The README documented the CLI command and HTTP endpoint as a known
missing piece; both now exist and wrap the same `FileIntelligence` the
proxy pipeline uses.

```bash
tokenmizer analyze data.csv --token-budget 300   # local, no server, no key
tokenmizer analyze big.json --raw > digest.txt
```

The endpoint takes `content` inline rather than a path: the server is
usually a container or a remote host, so a client-side path means nothing
to it — and accepting one would be an arbitrary-file-read primitive
against the server. There is a test asserting a `file_path` payload is
rejected.

#### Fixed — six endpoints existed but were undocumented
`/api/cache/stats`, `/api/checkpoints/{id}`, `/api/graph/{id}/viz`,
`/api/graph/{id}/history`, `/api/graph/{id}/transitions` and
`/api/graph/{id}/obsidian` were all live and absent from the API table.
Found by a scripted sweep that compares every route decorator against
every endpoint the README mentions, in both directions. Both directions
are now clean.

#### Added — tests
`tokenmizer analyze` (happy path, `--raw`, missing file, directory
instead of file, non-positive budget) and `POST /api/analyze` (budget is
honoured, three invalid-input cases, path-payload rejection), plus the
asset version-drift guard, plus a test that compares every route
decorator against the README's API table in both directions.
**533 tests, 77% coverage, ruff clean, MCP e2e green.**

### An eval harness, a real corpus, and measured extraction

Addresses the three things flagged as launch risks in 0.5.0: a
three-session benchmark, unmeasured task recall, and hand-tuned constants
with nothing to justify them.

#### New — `python -m benchmarks.eval`
A precision/recall/F1 harness over a labelled corpus, with per-item error
listing so a failure is diagnosable rather than just visible.

**It reports precision, not only recall.** Every extraction number this
project published before now was recall-only, which is the metric an
extractor games by emitting more text — one node containing the whole
transcript scores 100%. Matching is asymmetric and anchored on the
ground-truth item, so a sprawling label cannot buy recall without paying
for it in precision.

Label quality is scored separately from correctness: truncation rate,
multi-sentence rate, near-duplicate count. An extractor can be accurate
and still emit labels nobody wants in a token-budgeted resume block.

`--sweep` moves a constant across a range and prints the effect, so a
threshold can be defended with a table.

#### New — a labelled corpus, and a way to use your own
**8 sessions, 82 turns, 97 labelled items, 8 domains** (Go, Rust, Python,
TypeScript, React, SQL, CI, ML) — up from 3 sessions in one style.

The three original sessions were **relabelled**. Their ground truth used
hindsight summaries ("auth endpoints", "tests") that nobody says out
loud, so scoring against them measured paraphrasing rather than
extraction. Labels are now spans a reader can point at in the transcript.

Every session declares `origin: synthetic | real` and the harness prints
the split, because "89% recall" means something different on hand-written
fixtures than on captured transcripts. **The committed corpus is entirely
synthetic.** `--corpus DIR` runs against your own labelled sessions,
which is the only way to get a number about your workload.

#### Fixed — extraction defects the harness exposed

| | before | after |
|---|---|---|
| Errors F1 | 14% | **87%** |
| Files F1 | 74% | **91%** |
| Completed tasks F1 | 77% | 75% |
| Decisions F1 | 63% | 60% |
| **Macro F1** | **59%** | **75%** |
| Labels truncated mid-word | 23% | **8%** |
| Labels spanning >1 sentence | 27% | **5%** |

Reported in full, including the two categories that moved slightly the
wrong way: clipping labels to one clause costs a little coverage on long
ground-truth items, and that trade bought a 15-point macro gain and
labels a human can read.

- **Errors were scanned only in the recent window**, so a session that
  diagnosed three failures early and spent its remaining turns fixing
  them carried none of them forward. An error is a permanent fact —
  a resolved one explains the code, an unresolved one is the most
  important thing in a resume. Now scanned over full history.
- **The error pattern required a keyword followed by a description**
  ("Error: <text>") from a vocabulary of exception names and HTTP codes.
  Real transcripts say "a port collision in the integration tests", "an
  OOM on the Windows runner", "the backfill is timing out". Recall was
  **1 of 12**. Replaced with typed-exception and symptom-phrase patterns
  that capture the subject, not just the tail.
- **Any task or decision whose label ended in a file path was silently
  retyped as a FILE node** — `_check_type_mismatch` anchors on the end of
  the string, so "User model in api/models.py" became a file. It
  vanished from completed tasks and appeared as a spurious file, costing
  both task recall and file precision.
- **`wrote?` never matched "written"**, the usual way completion is
  narrated ("I've written the connection pool").
- **Duplicate granularity.** One mention of `scripts/backfill.py`
  produced both it and `backfill.py`; "bcrypt for password hashing"
  produced both it and a bare "Use bcrypt". One fact, two nodes, in a
  block with a token budget.
- **Captures ran to a fixed 80 characters** with no notion of where the
  thought ended, producing labels cut mid-word and labels spanning three
  sentences. Now clipped to one clause.

#### Changed — a constant chosen from a table instead of by feel
`_MIN_CLAUSE_CHARS` governs how short a clipped label may be. Swept
against the corpus:

| min_chars | macro F1 | truncated | multi-sentence |
|---|---|---|---|
| 8 | 74% | 6% | 3% |
| **22** | **75%** | **8%** | **5%** |
| 34 | 79% | 17% | 14% |
| 48 | 79% | 24% | 20% |

F1 keeps climbing past 22 only by letting labels sprawl again — the
defect the clipping was added to fix. 22 keeps essentially all the
readability win and takes most of the accuracy gain. The sweep is in the
code comment so the next person can disagree with the trade rather than
with a magic number.

#### Added — tests
`tests/unit/test_extraction_quality.py` (23 tests): clause clipping
including the dots-inside-tokens case, both dedup rules, errors surviving
outside the recent window, bare symptom words not being emitted alone,
the metric's asymmetry, corpus validation, and F1 floors per category.
Floors sit below measured values on purpose — a test pinned to today's
exact number trains people to edit the assertion.

**521 tests, ruff clean, MCP e2e green.**

### Cross-process safety, verified benchmarks, launch prep

#### New — graph writes are safe across processes
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

#### New — benchmarks that produce the numbers in the README
- `benchmarks/persistence/runner.py` measures write amplification,
  no-op persist cost, persist latency, 4-process concurrency, and the
  stale-writer case.
- `benchmarks/graph_retrieval/runner.py` **was broken** — it called
  `GraphMemory(storage_dir=...)` without the required `session_id` and
  crashed on every run, while being referenced as a benchmark. Fixed.

#### Changed — README benchmark figures now match the runners
The quoted table was measured on **v0.2.4** and no longer matched what
the committed runner produced. Re-measured on 0.5.0:

| | Task | Decision | File | Info preserved |
|---|---|---|---|---|
| was (v0.2.4) | 76% | 85% | 100% | 87% |
| now (v0.5.0) | 76% | **92%** | 100% | **89%** |

Per-session spread and the n=3 sample size are now stated inline rather
than in a footnote, because a three-session synthetic benchmark is
directional and should read that way.

#### Fixed — documentation that described features which do not exist
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

#### Changed — packaging and release readiness
- `py.typed` added and shipped in the wheel (PEP 561), so the type
  annotations are actually visible to consumers instead of being
  discarded as `Any`.
- Classifiers: `Development Status :: 4 - Beta` (from `3 - Alpha`),
  Python 3.13, `Typing :: Typed`, AI topic. Added `Changelog` and
  `Security` project URLs and an explicit sdist manifest.
- Wheel and sdist build clean and install into a fresh venv with a
  working `tokenmizer` entry point.

#### Changed — CI actually verifies the deployment story
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

#### Added — tests
`tests/unit/test_multiprocess.py` (9 tests) covers the stale-writer
case, reconciliation not eating unpersisted work, another process's
additions surviving, lock exclusivity across real subprocesses, distinct
sessions not contending, hostile `session_id` values not escaping the
lock directory, 4-process lossless concurrency, and stale-lock
sweeping never touching a held lock.

#### Fixed — repository hygiene
`.gitignore` had `checkpoints/` commented out ("conflicts with
`tokenmizer/checkpoints/`"), which is what an *unanchored* pattern does.
Anchoring it to `/checkpoints/` ignores the runtime storage directory
without touching the source package. 135 lock files and two benchmark
result JSONs had been committed as a result; both are now removed and
ignored, and `sweep_stale_locks()` prunes lock files untouched for 30
days on an hourly cycle so the directory cannot grow without bound.

**498 tests, 77% coverage, ruff clean across `tokenmizer/`, `tests/`,
`benchmarks/` and `scripts/`.**

### Per-row storage (#27), provider fixes, comment cleanup

Closes the three items left open by 0.4.1.

#### Changed — graph storage is now per-row (schema v2), closes #27
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

#### Fixed — provider adapters (the modules skipped by both audit passes)
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

#### Fixed — file intelligence overshot its token budget
`token_budget` is the module's entire contract (the `analyze_file` MCP
tool documents it as "Max tokens for the summary"), but the log and code
strategies assembled a fixed set of sections and only then measured,
overshooting by ~5%. A single budget clamp now applies to every strategy
at the exit point, reserving room for its own trim marker.

#### Changed — comments no longer narrate past bugs
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

### Second audit pass: correctness, durability, isolation

Findings from a full-codebase audit. Every item below was reproduced
before being fixed and has a regression test in
`tests/unit/test_audit_fixes.py` or `tests/unit/test_durability.py`.

#### Fixed — decision supersession silently discarded changes (critical)
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

#### Fixed — resume surfaced superseded and invalidated decisions as current
`_build_critical` sorted *all* decision nodes by importance with no
status filter, so the ~100-token "must-know facts" block presented
choices the team had already moved off — and ones explicitly rejected via
`/api/decision/invalidate` — as `KEY DECISIONS`, while the current
decision could be crowded out. `query()` and `to_context_block()` had
always filtered these; the checkpoint resume path, the product's core
output, did not. All resume builders now go through one `_live_nodes()`
filter. Invalidated decisions are surfaced separately as `DO NOT REVISIT`
so the model doesn't re-propose them.

#### Fixed — token counting took the whole proxy down (critical)
`_get_encoding` caught only `ImportError`, but tiktoken downloads its BPE
vocabulary from a CDN on first use. Any egress restriction, proxy, or CDN
outage raised a network error that propagated out of
`count_messages_tokens` — on the hot path of every request — so **every
request 500'd**, and the documented char/4 fallback was unreachable
(it only ran when tiktoken was absent entirely). Now fails soft, caches
the failure so it isn't retried per request, and the Dockerfile
pre-downloads the vocabulary at build time (`TIKTOKEN_CACHE_DIR`).

#### Fixed — corrupt-DB recovery destroyed every session (critical)
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

#### New — mid-session durability guarantees
See README "Durability". Shutdown (SIGTERM) now drains in-flight
background extraction and force-persists every cached graph instead of
logging one line and exiting; a periodic flush bounds hard-kill exposure
to 30s; a graph whose persist fails is **kept in memory** rather than
evicted; and sessions with an in-flight request are never evicted (the
previous guard checked a lock only the background task ever took, so it
was inert for request traffic).

#### Fixed — any caller could read or modify any session (security)
Session-scoped routes took `session_id` straight from the URL, with a
single shared deployment key as the only auth and no ownership model at
all. Since clients choose their own `session_id`, reading someone else's
session needed no guesswork, and `/api/decision/invalidate` made it a
write primitive. Sessions are now claimed by the first principal that
uses them; `api_keys` adds further credentials, each its own principal.
Denied requests return 404, not 403, so the endpoints can't be used to
probe which sessions exist. Dev mode and single-key deployments are
behaviour-compatible.

#### Fixed — environment variables did not override `tokenmizer.yaml`
`from_yaml` passed the file's contents as `__init__` kwargs — the
highest-priority source in pydantic-settings, above env vars — so every
key present in the shipped (and Docker-`COPY`'d) config silently beat its
`TOKENMIZER_*` variable, contradicting that file's own header.
`TOKENMIZER_PROVIDER=openai` resolved to `anthropic`. API keys appeared to
work only because those lines happen to be commented out.

#### Fixed — streaming cached truncated responses
On a mid-stream provider error the generator fell through to post-stream
bookkeeping and cached whatever partial text had arrived, serving that
truncated answer to every future matching prompt. Failed and cache-hit
streams no longer write to the cache, and non-`ProviderError` exceptions
are handled instead of killing the generator mid-stream.

#### Fixed — smaller correctness and resource bugs
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

#### Changed — settings that did nothing now do something, or say so
`compression.min_tokens_to_compress`, `graph_checkpoint.max_resume_tokens`,
`graph_checkpoint.extraction_model` and `memory.enabled` were all
documented but read by nothing; they are now honoured. `routing.*` has no
implementation at all — it is kept so existing configs load, but logs a
warning at startup and is labelled NOT IMPLEMENTED in the README and
config file.

#### Removed — Redis from `docker-compose.yml`
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
