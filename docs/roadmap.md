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
| Extraction, macro F1 on the labelled corpus | 96% overall; **90% on real transcripts**, 97% synthetic | `python -m benchmarks.eval` |
| Label quality | 15% of labels truncated mid-word; 40 near-duplicate pairs over 171 labels | same run |
| Retrieval, recall@6 on paraphrased questions | 85% keyword; 92% with `semantic_retrieval` (n=13) | `benchmarks.graph_retrieval.query_eval` |
| Graph density, fastapi_auth session | 28 nodes, 26 edges, 3 communities + 7 unclustered | `/api/graph/{id}/viz` |
| Independent 100-session benchmark | ties for first at 60% macro F1; decisions 59%, errors 44% (weakest) | tokenmizer-research |
| Resume block size | ~160-180 tokens standard tier | `benchmarks/resume_quality` |
| Suite | 1185 tests, ruff clean | `pytest tests/` |

Read the 90% real-transcript figure as the honest one. It is the reason
several items below exist.

---

## What this branch fixed, and what each fix revealed

These were found by reading the request path end to end rather than by a
bug report, which is itself a finding: the defects that remain live at
the seams between layers, where no layer's own tests look.

| Defect | Why it was silent | What it revealed |
|---|---|---|
| Semantic cache keyed on the last user message only; the second "continue" of a session got the first one's answer | A cache hit looks like success; nothing compares the answer to the question | Cache keys must include conversation state. Done: a fingerprint of every prior message |
| `/api/resume/{id}` 404'd on sessions with hours of memory | Auto-checkpoint measures the request *after* windowing, so with windowing on it rarely fires; nothing else wrote a checkpoint | The checkpoint table is a snapshot, the graph is the truth. Done: resume answers from the graph when it is newer |
| `tools` / `tool_choice` accepted and dropped | `extra="allow"` swallowed them; a warning in a log nobody tails | Every agent framework was broken behind the proxy. Done: forwarded for 7 of 9 providers, 501 for the rest |
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

**1. Rolling summary for the turns windowing drops.**
When a session crosses `memory.max_tokens_before_summary`, every turn
older than the last ten is replaced by a 250-token graph block. Anything
the extractor missed on those turns (10% of labels on real transcripts,
and everything outside the ontology: a constraint the user stated once, a
preference, a number) is gone for the rest of the session. Design: a
`SUMMARY` node per windowed span, written by the heuristic extractor's
sentence selector by default and by the chat model when
`use_llm_extraction` is on, budget-capped, superseded when the span is
re-summarised. Measure with `benchmarks/resume_quality` and the
checkpoint-accuracy runner; ship only if retention rises.

**2. Auto-checkpoint on the client's conversation size.**
The trigger compares the *sent* request to the window. The conversation
the client holds is what is actually running out of room. Trigger on
either the raw size or the sent size crossing `trigger_at_percent`, keep
the per-session retention cap. Now that resume reads the live graph this
is about the checkpoint diff and the "Continue from" hint, not about
losing memory.

**3. Streamed tool-call deltas on Anthropic.**  *(partially done: tools
are forwarded and answered in one piece; only the incremental stream is
missing)*
The answer is currently built in one piece and emitted as chunks. The SDK
event stream carries `content_block_start` (tool_use) and
`input_json_delta`; map them to the same dict events the OpenAI adapter
already yields. Then Ollama, which streams `message.tool_calls` at the
end.

**4. Semantic retrieval on by default when the model is present.**
recall@6 85% to 92% (n=13 — small, so enlarge the eval first: 40 cases
across the corpus). Bundle the weight download into the Docker build
(already done for the cache) and turn the flag on when
`EmbeddingEngine.available`. The keyword-gate measurement above is the
reason this is the path and not a cheaper one.

**5. The CLI's `stats` should show the durability counters too.**
`/health` reports `degraded` with the counters behind it, and the
dashboard renders them; `tokenmizer stats` still does not. Same three
lines, same source.

### P1 — more sessions, more domains, more of the pipeline real

**6. Domain packs for the ontology.**
Goal / task / decision / file / error / endpoint / schema is a coding
ontology, and the regex families in `patterns.py` are coding phrasings.
A research, incident-response, product, data-science or writing session
extracts almost nothing today (the graph page already says so). Add a
`domain` setting with packs: *research* (hypothesis, finding, source,
open question), *ops* (alert, runbook step, root cause, mitigation),
*product* (requirement, feedback, decision, owner), *data* (dataset,
metric, experiment, result). Each pack is pattern families plus an LLM
extraction schema plus a labelled eval corpus; a pack ships only with its
corpus, because the coding numbers above are only trustworthy because
that corpus exists.

**7. Label quality.**
Labels spanning more than one sentence are down to 0% and the decision
near-duplicate merge now folds technology spellings and scaffold words,
but 15% of labels are still cut mid-word and the near-duplicate count is
39 across all categories. Run the same merge for tasks and files, and
clip at a word boundary with an ellipsis rather than mid-token. Both are
visible in `benchmarks.eval`'s label-quality block, so the target is
numeric: under 5% truncation, under 15 near-duplicate pairs.

**8. Model routing: implement it or delete it.**
`routing.*` and `savings.routing` have never done anything. Complexity
scoring is a research problem; what users ask for is a `model_map`
(client model alias to provider model, per session or per key). Ship the
map, drop the routing block and the always-zero savings field, with a
one-release deprecation warning for configs that still carry it.

**9. Multi-process state.**
Graph and checkpoint writes are cross-process safe; the semantic cache,
rate limiter and analytics are per worker. Either wire the existing
`state/backend.py` to Redis for those three or remove the setting. A
shared-SQLite counter table is the cheaper option and matches everything
else in the project.

**10. Gemini and Cohere: tools and streaming.**
Both refuse tool requests with a 501 today and stream with a 501. The
google-genai SDK exposes function declarations and a streaming call;
Cohere v2 has both. Same test shape as the three adapters that have them.

### P2 — the graph as a product

**11. Graph page.**
The page now has community hulls, a type legend that filters, a timeline
mode, the supersession chain in the node detail, a light theme and
keyboard shortcuts. Still missing: a checkpoint-to-checkpoint diff view
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
