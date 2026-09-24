# Roadmap

Where TokenMizer stands, measured, and what comes next in the order it
should come. Every item below is tied to a number from the repo's own
benchmarks or to a defect found by reading the code, because a roadmap
that is not anchored to measurement is a wish list.

The audience is anyone deciding whether to adopt, contribute to, or bet
on this project. If a claim here disagrees with the suite or a
benchmark, the suite is right and this file is a bug.

---

## Where it stands (measured, this branch)

| Surface | Number | Source |
|---|---|---|
| Extraction, macro F1 on the labelled corpus | 97% overall; **91% on real transcripts**, 98% synthetic | `python -m benchmarks.eval` |
| Label quality | 0% truncated mid-word; 0 near-duplicate pairs over 170 labels | same run |
| Retrieval, recall@6 on paraphrased questions | **82% keyword, n=40** (was 85% at n=13) | `benchmarks.graph_retrieval.query_eval` |
| Extraction outside coding | **96%** macro F1 with a domain pack, **11%** without | `benchmarks.eval --corpus benchmarks/eval/corpus_domains` |
| Out-of-ontology facts surviving windowing | **17% to 100%**, at +22 tokens of resume per session | `benchmarks.resume_quality.runner` |
| Checkpoint accuracy | 80% task / 100% decision / 100% file recall, 195-token resume | `benchmarks.checkpoint_accuracy.runner` |
| Graph density, fastapi_auth session | 28 nodes, 26 edges, 3 communities + 7 unclustered | `/api/graph/{id}/viz` |
| Independent 100-session benchmark | ties for first at 60% macro F1; decisions 59%, errors 44% (weakest) | tokenmizer-research |
| Suite | 1616 tests, ruff clean | `pytest tests/` |

Read the 91% real-transcript figure as the honest one. It is the reason
several items below exist.

**P0 and P1 are done.** What follows keeps each item's entry so the
reasoning that motivated it stays readable next to what was actually
built — and so the ones that are *not* finished say which part is
missing rather than being quietly dropped. Two are partial and say so:
the semantic-retrieval number was not re-measured (no egress where this
was written), and the domain packs ship with hand-written corpora rather
than captured transcripts. P2 is untouched.

---

## What this branch fixed, and what each fix revealed

These were found by reading the request path end to end rather than by a
bug report, which is itself a finding: the defects that remain live at
the seams between layers, where no layer's own tests look.

| Defect | Why it was silent | What it revealed |
|---|---|---|
| Semantic cache keyed on the last user message only; the second "continue" of a session got the first one's answer | A cache hit looks like success; nothing compares the answer to the question | Cache keys must include conversation state. Done: a fingerprint of every prior message |
| `/api/resume/{id}` 404'd on sessions with hours of memory | Auto-checkpoint measures the request *after* windowing, so with windowing on it rarely fires; nothing else wrote a checkpoint | The checkpoint table is a snapshot, the graph is the truth. Done: resume answers from the graph when it is newer |
| `tools` / `tool_choice` accepted and dropped | `extra="allow"` swallowed them; a warning in a log nobody tails | Every agent framework was broken behind the proxy. Done: forwarded for all 9 providers, streamed or not |
| Injected context prepended to the system prompt | Token counts looked fine; the loss was in the provider's cache-hit rate | Layer 4 was defeating layer 5 on every turn. Done: appended |
| LLM extraction discarded prose-wrapped JSON | Counted as a "silent failure" and nothing more | Local and small models do this constantly; the cheap-extraction story depended on it not happening |
| Keyword gate on context injection (attempted) | Would have *saved* tokens and looked like a win | Measured first: recall 85% to 46%. Reverted. The importance ranking is doing real work; do not gate without the semantic pass |
| A supersession the transcript stated outright was parsed, passed through, and never read | The topic classifier sometimes rediscovered it, so the feature worked in the demo | The corpus session that exists to demonstrate `/why` produced no trail at all. `record_supersession` is now the one place a transition is written, reached from both paths |
| "Next.js instead of React for better SEO" recorded "better SEO" as the technology chosen | A decision node is a decision node; nothing looks at whether it is a noun | Reading two operand orders with one pattern. Split into forward and reverse forms; "instead of" is one decision naming its alternative, not a change over time |
| "PostgreSQL for order storage" and "Postgres for orders" reported as an unresolved conflict | It reads like a real finding, so a user goes looking for the conflict | Two modules knew the tech vocabulary and only one used it. One table in `patterns.py` now, folded by both |
| `FIXES`, `BLOCKS` and `DEPENDS_ON` existed in the ontology and nothing created them | The graph still rendered; it was just sparser than it should be | An ontology is a claim about what the graph contains. 11 of 25 nodes were unclustered because the edges that would group them were never made |
| `/health` returned `ok` whatever had happened | Uptime monitors were green | The counters existed and were only reachable from `/api/stats`. For a tool whose claim is "your context is safe", that is the one check that must not lie |
| Five of six MCP tools answered "connection refused" on a fresh install | The plugin installs the MCP server; nothing starts the proxy | They now read the same SQLite store directly, and only fall back on a transport failure so a running proxy's refusal is never bypassed |

---

## Priorities

Ordered by how much a user loses today, weighed against the cost to do it
well. P0 is the next release.

### P0 — nothing breaks mid-session

**1. Rolling summary for the turns windowing drops.**  *(done)*
A `SUMMARY` node per session holds what the ontology has no node for from
the turns windowing replaced — a budget, a deadline, a licence
restriction, a latency target. Each is one sentence, said once, and none
of them is a task, a decision, a file or an error, so all of them used to
leave the session permanently at the moment windowing engaged.

Selected by `graph_memory/summary.py`: a clause is kept only if it
carries a quantity with a unit or a constraint verb, AND is not already
covered by a node the graph holds — restating a node spends the resume
budget on nothing. One node per session, rewritten as the dropped span
grows rather than accumulated, and never allowed to fail a chat request.

Measured by the new `benchmarks/resume_quality/runner.py`:
out-of-ontology retention **17% to 100%** on its fixtures, **+23 tokens**
of resume block per session, **no section lost** on the captured
transcripts in the eval corpus, and checkpoint accuracy unchanged
(80/100/100). Read the fixture number with the caveat the runner states
itself: those sessions were written by the same person as the selector,
so they show the mechanism works end to end, not that it generalises to
phrasings nobody had in mind. The four notes it produces on the real
transcripts are in the runner's output and are worth reading — they are
facts the graph genuinely had no node for.

With `use_llm_extraction` on, the chat model is a strictly better
sentence selector and can replace `select_sentences` without changing
anything else. That is the next step here, not a rewrite.

**2. Auto-checkpoint on the client's conversation size.**  *(done)*
Occupancy is now the fuller of the two sides — what leaves here after
windowing, and what the client is holding — instead of the sent payload
alone. Windowing keeps the sent side near-constant, so a session forty
turns deep sent the same fraction it sent at turn five and never
checkpointed, in exactly the sessions the trigger exists for. The
retention cap is unchanged. Resume already reads the live graph, so
nothing was being lost by the late trigger; the checkpoint diff and the
"Continue from" hint were, and a resume cannot rebuild those.

**3. Streamed tool-call deltas.**  *(done)*
Anthropic and Ollama built the answer in one piece and emitted it as
chunks; the client waited for the whole thing. Anthropic's raw event
stream is read now (`content_block_start` for the id and name, then
`input_json_delta` for the arguments) instead of `text_stream`, which
carries text only — and its CONTENT-BLOCK index is mapped to a tool-call
ordinal, or a call that follows a text block is announced at index 1 with
nothing at index 0 and every client SDK that assembles by index breaks.
Ollama sends finished calls on its last message, which become one delta
each; its streaming payload also carried no `tools` at all, so a client
that asked for tools *and* a stream got a model that could not see them.

**4. Semantic retrieval on by default when the model is present.**  *(the
default is done; the embedding number is not re-measured)*
`semantic_retrieval: auto` is the default and resolves once at startup by
trying to load the model, because sentence-transformers ships no weights
and "installed" is not "loadable" on an air-gapped host. `true`/`false`
still pin it. `Memory` follows the same setting, so an in-process caller
no longer gets quietly worse ranking than the same deployment's proxy.

The eval is enlarged from 13 cases to **40**, across every corpus session
including the six captured transcripts, with a grounding check that
refuses a question its own transcript cannot answer. Keyword ranking
scores **recall@6 82%** on it — and the misses are exactly the paraphrases
embeddings exist for ("what is slow about the dashboard" against a node
that says "re-render").

**The 92%-with-embeddings figure is not re-measured and should not be
quoted.** It came from the 13-case eval, and this branch was written in a
sandbox with no egress, so the model never loaded here. Running
`--semantic` on a host that can fetch the weights is the remaining work,
and the number it produces replaces the old one.

**5. The CLI's `stats` should show the durability counters too.**  *(done)*
It prints failed writes, sessions with an unreadable graph, sessions with
no durable storage, sessions that lost stored memory, and broken
checkpoint storage. A `/health` call that itself fails is reported as
"could not read", never as healthy: unknown and fine are different
answers, and savings were the only number the CLI used to show.

### P1 — more sessions, more domains, more of the pipeline real

**6. Domain packs for the ontology.**  *(done — four packs, one corpus)*
Goal / task / decision / file / error is a *coding* ontology and every
regex family in `patterns.py` is a coding phrasing, so a research log, an
incident review or a product discussion extracted almost nothing.
Measured, it was worse than "almost": **macro F1 11%** on three labelled
sessions of that kind, with decisions and errors at **0%**.

Packs ship for **research**, **ops** and **product** (coding is the
default and adds nothing). Same ontology, different vocabulary: the
shapes a session has are already the five this ontology holds — something
you are trying to establish, the steps you took, the calls you made, what
went wrong, the artifacts you referenced — so a pack adds the phrasings
that name those shapes in one domain, and the words the resume block uses
for them. The alternative, a node type per domain, would grow the
ontology to thirty types of which a session uses five, each needing a
colour slot, a lane and a section.

**11% to 96%** on `benchmarks/eval/corpus_domains`, and the coding corpus
is bit-for-bit unchanged at 97%, because a pack's families run *after*
the coding ones and can only add. Reproduce the before number with
`python -m benchmarks.eval --corpus benchmarks/eval/corpus_domains
--ignore-packs`.

Fixing this exposed the same bias one layer down: the validator's goal
scorer only rewarded "build / create / develop / implement / design", so
a research question, an incident and a quarterly outcome were extracted
correctly and then rejected for not being about building software.

Still open: a **data** pack (dataset, metric, experiment, result), real
captured transcripts for each pack rather than the synthetic sessions
committed here, and per-pack LLM extraction schemas. The three corpora
are hand-written, which the eval reports as `synthetic` — the caveat that
applies to the coding fixtures applies here with more force, because
there are three sessions rather than fourteen.

**7. Label quality.**  *(done — and the measurement was most of it)*
Both targets are met: 0% truncated mid-word, 0 near-duplicate pairs,
pinned by `tests/unit/test_label_quality.py` so a regression fails a test
instead of moving a number nobody re-reads.

The larger finding was that the block was measuring itself. "Truncated"
was guessed from a label's shape — 60+ characters ending on a letter —
and 22 of the 26 it counted were complete labels that simply ended in a
word; it is now checked against the transcript, which is exact. The five
that really were cut came from the patterns' fixed 80-character capture,
which now ends on a word boundary. "Near-duplicate" pooled every
session's labels into one bucket and compared across node types, so
`persistence.py` in three sessions counted as pairs and a task naming the
file it touched counted as a duplicate of that file; it now compares
within a session, and exempts the pairs the graph itself says are a fix
and the error it fixed.

Fixing the measurement exposed two real extraction defects behind it:
`completed` read as a verb in "vanished from completed tasks", and a
completion verb inside a clause that had already said the work was not
done. Completed-task precision 91% to 98%, decisions 95% to 97%, real
transcripts 89% to 91%.

**8. Model routing: implement it or delete it.**  *(done)*
`model_map` ships: an exact-match substitution from the client's model
name to the model actually called, reported back as
`tokenmizer.model_mapped_from` so a substituted answer is traceable.
`savings.routing` — a hardcoded `0` on every response since the first
release — is gone, and a config carrying a `routing:` block now warns
that it is deprecated whatever it is set to, because the fields were a
belief about behaviour either way. Complexity scoring stays unbuilt: it
is a research problem, and a switch that silently does nothing is worse
than no switch.

**9. Multi-process state.**  *(done for the one that was a bug)*
The rate limiter was the item on that list that was not merely
suboptimal: with `--workers 4` it enforced the configured limit once per
worker, so 60 requests a minute was 240. `state_backend: sqlite` puts the
token buckets in `storage_dir` where every worker on the host shares one
count, with the whole read-modify-write inside `BEGIN IMMEDIATE` — without
that, two workers read the last token, both spend it, and the limit is
lost. Pinned by a test that spends one budget from four real processes and
asserts *exactly* the budget, because a limiter that is merely stricter
than broken is still not the configured limit. `memory` stays the default
and costs nothing for a single process.

`state_backend: redis` was never implemented and is now honest about it:
it behaves as `memory` and warns, naming `sqlite`. `tokenmizer/state/
backend.py` (145 lines, no callers, its own docstring said so) and
`tokenmizer/storage/__init__.py` (a protocol nothing imported and nothing
conformed to) are deleted.

Still per-worker, and deliberately: the semantic cache (a lower hit rate,
not a wrong answer) and the analytics counters (already durable per
worker; a shared view is a reporting change, not a correctness one).
Neither is a limit somebody wrote down. Neither spans hosts either — that
is a load-balancer concern and this says so rather than implying
otherwise.

**10. Gemini and Cohere: tools and streaming.**  *(done)*
Both refused tool requests with a 501 and streamed with a 501, though
both SDKs have had each throughout. Gemini needed the most work: the
adapter used `chats.create` with a history and a plain-text last turn,
which cannot express an assistant turn that asked for a tool or the
client's results coming back — both are content parts — so it moved to
`generate_content` over the whole conversation, with
`function_declarations` and `function_response` parts. Gemini also mints
no call id and matches a result to a call by NAME, so the id is minted
here and the name looked up from the request. Cohere v2 already speaks
the OpenAI tool shape; only its streamed event names and its
document-shaped tool results needed translating, and it has no
`tool_choice` to enforce.

All nine providers now carry tools, streamed or not. Coded against each
SDK's documented objects and tested with fakes shaped like them, not
against live keys — the tests say so.

### P2 — the graph as a product

**11. Graph page.**
The page opens as a radial map — one arc per node type, named on a ring
outside the labels, relations as chords through the middle — with force
and timeline a click away, community hulls, a type legend that filters,
the supersession chain in the node detail, a validated palette in both
themes, and a panel of derived counts, communities, hotspots and
type-to-type flows. Still missing: a checkpoint-to-checkpoint diff view
(the `graph_diff` each checkpoint already stores), a path highlight that
walks a whole `/why` chain on the canvas, and a canvas renderer for
graphs past the 200-node prune cap. Keep the zero-external-dependency
rule; it is why the page can be shared as a file.

**12. Reasoning.**
`impact()` is one hop. Two-hop impact ("which files does the decision
that replaced X touch"), "what changed since checkpoint N", and a
suggestion when the consistency audit finds two active decisions on one
topic (surface both, propose which is newer, never auto-resolve).

**13. Sessions across people and machines.**
Export and import a session as JSON, merge two sessions (union of nodes,
supersession by timestamp), and a read-only share link for the graph
page that does not require the API key.

---

## Principles that decide priority

- **Measure before ranking.** The keyword gate looked like a free token
  win and cost 39 points of recall. Every retrieval, extraction or
  compression change runs its benchmark first and ships with the number.
- **Silent is worse than broken.** A layer that fails must fail into a
  counter the operator can read, and the response must say what it did
  not do (`tokenmizer.checkpoint`, `tokenmizer.fallback`, `source`).
- **The graph is the truth; a checkpoint is a photograph of it.** Anything
  that reads memory reads the graph first.
- **Never widen what a caller sent.** Tool traffic, code blocks and file
  contents pass through byte-for-byte; the proxy compresses prose only.
- **A number in the docs is a test.** Test counts, F1, recall and resume
  size are checked by the suite or re-derived by a runner in CI.

---

[← Back to the README](../README.md)
