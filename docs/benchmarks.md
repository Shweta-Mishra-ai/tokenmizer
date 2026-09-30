# Benchmarks

Every number here comes from a committed runner you can execute yourself. Nothing is hand-written; if a figure and a runner disagree, the runner is right and this page is a bug.

---

```bash
python -m benchmarks.eval                            # extraction P/R/F1
python -m benchmarks.eval --errors                   # every miss, every false positive
python -m benchmarks.eval --corpus DIR               # score YOUR sessions
python -m benchmarks.checkpoint_accuracy.runner_v2   # graph vs summary
python -m benchmarks.graph_retrieval.query_eval       # what query() returns
python -m benchmarks.persistence.runner              # storage + concurrency
python -m benchmarks.savings.runner                 # input cost, with and without
pytest tests/ -q                                     # 2004 tests
```

## Extraction quality — precision, recall and F1

`python -m benchmarks.eval` scores extraction against a labelled corpus:
**14 sessions, 144 turns, 172 labelled items, 14 domains** (Go, Rust,
Python, TypeScript, React, SQL, CI, ML, plus six real audit sessions).
Measured on v0.5.4:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Files | 100% | 100% | **100%** |
| Decisions | 97% | 100% | **99%** |
| Completed tasks | 98% | 98% | **98%** |
| Pending tasks | 100% | 90% | **95%** |
| Errors | 93% | 96% | **94%** |
| | | **macro F1** | **97%** |

**Precision is reported, not just recall.** An extractor that emits the
whole transcript as one node scores 100% recall; that is why recall-only
extraction numbers should be distrusted, including earlier ones of ours.

### Fixtures are easier than real sessions

Eight of the sessions are hand-written fixtures; six are condensed from
real TokenMizer audit sessions. Scored separately, and printed on every
run rather than kept in a drawer:

| Corpus origin | Sessions | Macro F1 |
|---|---|---|
| Synthetic (hand-written) | 8 | **98%** |
| Real (captured transcripts) | 6 | **91%** |

**Treat 91% as the number that describes real sessions.** The seven-point
gap is the honest measure of how much the heuristics are fitted to text
we wrote ourselves. Closing it needs more real transcripts, which is the
single most useful contribution anyone could make here.

Two things keep these numbers from being self-congratulatory:

* **Ground truth must be quoted from the transcript.** `--corpus` refuses
  to score a corpus containing a label no single message supports, because
  such a label is unreachable for *any* extractor and caps recall at a
  number no code change can move. Adding the check found two in our own
  corpus.
* **Labelling is exhaustive, not selective.** Every item matching the rule
  in `benchmarks/eval/corpus.py` is labelled, including decisions that were
  later superseded. Choosing which of several stated decisions "counts"
  turns precision into a measure of the annotator's taste.

n=14 is still a small sample and every session was labelled by the same
author. Label quality, scored separately because a correct-but-sprawling
label still wastes resume budget: 15% truncated mid-word, 1% spanning more
than one sentence, mean length 35 characters.

Across 172 labelled items the extractor now misses eight and invents
nine. The residue is where regexes genuinely stop: a defect stated as a
measurement ("error recall is 8 percent"), and one failure named twice in
words that share no tokens ("backfill is timing out" / "the backfill
timeout"). That is what `use_llm_extraction` is for.

To get a number for *your* workload, label a few of your own sessions in
the format documented in `benchmarks/eval/corpus.py` and run
`python -m benchmarks.eval --corpus /path/to/them`.

## Input cost — with the proxy and without

`python -m benchmarks.savings.runner` replays one conversation the way a
chat client sends it — the whole history on every request — through the
real proxy and the real Anthropic adapter, at four lengths. Only the
network is replaced: a stand-in applies the provider's documented prompt
caching rules to the exact request it receives (byte-exact prefix up to a
cache marker, per-model minimum, reads at 0.1x the input price, writes at
1.25x). Cost is in input-token units; `claude-sonnet-5`, cache minimum
1,024 tokens:

| Turns | History | Client alone, no caching | Client caching its own history | Through TokenMizer |
|---:|---:|---:|---:|---:|
| 10 | 462 | 2,333 | 2,333 | 2,833 (**21% more**) |
| 40 | 2,057 | 41,556 | 14,745 | 16,158 (**61% less** / 10% more) |
| 150 | 7,967 | 596,900 | 77,998 | 77,472 (**87% less** / level) |
| 300 | 16,040 | 2,399,067 | 268,767 | 157,818 (**93% less** / **41% less**) |

The two percentages are against the two client columns. Where the saving
comes from: below about 4,000 tokens of history, prompt caching — the
proxy marks the history cacheable and keeps every earlier byte of the
request identical; above it, windowing, with the cut held still between
cuts so the windowed history is cached too.

**Short sessions still cost more on input**, and this is why: a ten-turn
session of 462 tokens is under the model's cache minimum, so nothing can
be cached, and the proxy adds its ~50-token brevity instruction to every
request. That instruction exists to shorten what the model writes back,
which costs several times more per token than input; this benchmark
returns a fixed one-word answer and so charges the instruction while
crediting it nothing. `terse_output.enabled: false` removes it.

Read before quoting:

* **Input only.** Output tokens are not measured, for the reason above.
* **The session is real transcript text cycled to length**, not one real
  session of that length. Turns are short, so fixed per-request overhead
  is a larger share than on a real coding session.
* **Cache entries never expire here.** A client that pauses longer than
  the provider's cache lifetime between turns re-writes instead of reads.
* **Token counts fall back to a character estimate** when no tokenizer can
  load; the run prints which it used. The numbers above were taken with
  the estimate.
* **Answer quality after windowing is not measured.** That needs a real
  model and is the open question for long sessions.

Before this change, the same runner against the previous release: 81%
more at 10 turns, 21% more at 40, 68% less at 150, 87% less at 300 —
and against a client caching its own history, more expensive at every
length (3.4 times as much at 40 turns), because the per-turn context
block and the sliding window changed the start of every request.

## Input cost on a real agent session

The table above replays chat. Agent sessions are different, and the
difference decides whether the proxy helps: in the session measured here,
**over 99% of the tokens sent were tool results** (files and logs), where a
chat session is mostly prose.

`python -m benchmarks.savings.runner --trace session.jsonl` replays a
Claude Code transcript — real text, real tool output, real order — through
the proxy, so you can measure your own sessions. This one is the opening
requests of an agent session on this repository: one session, one author, so
read it as a worked example and not as a typical result. `claude-sonnet-5`, input
only, token counts from a character estimate, 23 requests of it:

| | Tokens sent | Input cost | vs a client caching its own history | Next step's needs still in the request |
|---|---:|---:|---:|---:|
| Client alone, no caching | 711,332 | 711,332 | — | 100% |
| Client caching its own history | 711,332 | 235,094 | — | 100% |
| Proxy before this change | 714,808 (0% fewer) | 260,046 | 10.6% more | 100% (nothing dropped) |
| **Proxy, defaults (`max_tail_tokens: 16000`, path index)** | **215,306 (69.7% fewer)** | **187,289** | **20.3% less** | **97.4%** |
| Proxy, `max_tail_tokens: 8000` | 131,065 (81.6% fewer) | 128,887 | 45.2% less | 90.8% |
| Proxy, defaults without the path index | 200,003 (71.9% fewer) | 176,687 | 24.8% less | 81.6% |

Over 58 requests the defaults are 86.9% fewer tokens and 45.8% less input
cost than a caching client, with 93.9% of the next step's needs in the
request; 8,000 is 93.1%, 63.2% and 90.7%. Without the path index the
defaults retain 71.0% there. (The proxy before this change could not
complete 58 requests: see the injection-filter fix in the changelog.)

**How "needs still in the request" is measured.** Dropping tokens says
nothing about whether the model still had what it needed, so this asks it of
the real agent: what it did next (the arguments of its next tool calls) is
the ground truth for what it needed. The identifiers, paths and strings in
those arguments that came from earlier tool output or the user, and are not
so common as to be noise, are checked against everything the proxy actually
sent. A path written as a directory line with the file name after it counts,
since a readable listing is not a loss. `--retention` prints it, with
examples of what was lost. It is a proxy for information, not for answer
quality: it says the information was in the request, not that the model used
it well.

Why the proxy saved nothing before: the window opened on a user turn, and an
agent loop has almost none. In the 23-request slice the last request holds
50 messages and two of them are from the user; at 58 requests, two in 110.
So the "protected" tail was nearly the whole conversation, and windowing had
nothing to remove. Windowing now cuts at agent steps, keeps a call together
with its results, and caps the verbatim tail in tokens
(`memory.max_tail_tokens`); the newest step is never dropped. A cut lands at
half the ceiling and is held while the tail grows back to it, so the history
stays byte-identical across several requests and the provider's cache can
serve it. Cutting to just under the ceiling instead left no room to grow:
the cut moved on every request, and the cache never hit.

Why information was lost, and what fixed it: replacing old turns with the
graph's 250-token resume block dropped the file paths that a listing or a
search had shown once. They were 60% of what the agent went on to use and no
longer had, and another third were parts of them. No selector predicts which
of a request's ~280 paths will be needed (the needed one ranked about 176th
by recency or by frequency), so the bridge carries all of them, grouped by
directory, within `memory.tool_index_tokens` (about 1,150 used). That took
retention from 71.0% to 93.9% at 58 requests, for about 4 points of the cost
saving. The ceiling itself costs almost nothing in retention: with none,
retention is 97.4% and 94.4%. The loss comes from windowing replacing old
turns, which is the design, not from the ceiling.

**What this does not establish.** The remaining 3 to 6% is content the index
cannot carry, such as the values in a config file read earlier
(`addopts`, `timeout`). And retention is not answer quality: whether the model
copes as well without the dropped output needs a real model.
`memory.max_tail_tokens: 0` keeps every recent message and
`memory.tool_index_tokens: 0` drops the index; 8,000 is the more aggressive
end of the trade, with about 9% of the next step's needs missing.

## Memory quality — graph vs a plain summary

`benchmarks/checkpoint_accuracy/runner_v2.py`, n=3 synthetic sessions:
the graph preserves **89%** of labelled information against **79%** for a
plain-summary baseline (Δ +10%), in an average resume block of **161
tokens** (180 / 168 / 136 across the three sessions) versus ~1,500+
tokens of raw history. The advantage is concentrated in decision recall
(92% vs a baseline that drops as low as 50%); on tasks it ties the
baseline (76% both).

## Retrieval — what `query()` returns for a paraphrase

`python -m benchmarks.graph_retrieval.query_eval`, **40 questions** across
every corpus session, each phrased the way a person asks rather than in
the node's own words ("what is slow about the dashboard", not "WebSocket
re-render"). If the question quoted the answer, the person would not have
needed to ask — and those are exactly the cases token-overlap ranking
cannot serve.

**recall@6 88%** with keyword ranking. Every case is checked to be
answerable from its own transcript before scoring: an ungrounded question
measures extraction, not retrieval, and reads as a retrieval failure
forever.

This was 13 cases until recently, where one case flipping moved the
headline by 8 points. The 92% figure once quoted for
`semantic_retrieval` came from that smaller set and has **not** been
re-measured against these 40 — `--semantic` needs the embedding weights,
and they could not be fetched where this was run.

## Domains other than coding

`python -m benchmarks.eval --corpus benchmarks/eval/corpus_domains`, three
labelled sessions — a research evaluation, a live incident, a product
planning session:

| | coding patterns only | with the pack |
|---|---|---|
| Completed tasks | 15% | **100%** |
| Pending tasks | 29% | **100%** |
| Decisions | 0% | **94%** |
| Errors | 0% | **78%** |
| **macro F1** | **11%** | **93%** |

Add `--ignore-packs` to reproduce the left column. The coding corpus is
unchanged at 97%, because a pack's pattern families run *after* the
coding ones and can only add recall.

These three sessions are hand-written, which the harness reports as
`synthetic`. The caveat that applies to the coding fixtures applies here
with more force: three sessions, one author, and the same person wrote
the patterns. Treat 93% as "the mechanism works on sessions of this
shape", not as a generalisation claim.

## Resume quality — what survives windowing

`python -m benchmarks.resume_quality.runner`. Windowing replaces every
turn older than the protected tail with the resume block, so anything the
ontology has no node for — a budget, a deadline, a licence restriction, a
latency target — used to leave the session at that point permanently.

| | before | after |
|---|---|---|
| Out-of-ontology facts still readable in the resume block | 0% | **100%** |
| Resume block, per session | — | **+25 tokens** |
| Sections lost on the corpus's six real transcripts | — | **0** |

Checkpoint accuracy is unchanged by it (80% / 100% / 100% task /
decision / file recall), which is the regression that mattered: the block
is budgeted, so anything added can push out what was already there.

**The fixtures were written by the same person as the selector**, so the
100% shows the mechanism works end to end and not that it generalises to
phrasings nobody had in mind — the same caveat the synthetic half of the
extraction corpus carries. The runner prints the notes it produces for
the six real transcripts beside it, which is the part worth reading.

## Storage — schema v2 (per-row)

`benchmarks/persistence/runner.py`, measured on v0.5.0:

| Metric | v1 (one blob per session) | v2 (per-row) |
|---|---|---|
| Rows written to add 1 node to a 50-node graph | 51 | **1** (−98.0%) |
| …to a 100-node graph | 101 | **1** (−99.0%) |
| …to a 200-node graph | 201 | **1** (−99.5%) |
| Rows written when a turn changes nothing | 100 | **0** |

Persist latency, one added node on a 200-node graph: **median 6.4 ms,
p95 7.0 ms** (measured on this machine across three runs; expect this to
move with hardware — the write-amplification and correctness numbers
above do not).

Concurrency (4 OS processes writing one session, 25 nodes each):
**100/100 nodes persisted, zero lost.** A stale writer holding a
pre-prune view of the graph no longer reinstates the rows another worker
deleted.

Enable `use_llm_extraction: true` for hybrid extraction (LLM + heuristic merge).

**On LLM/hybrid recall numbers — read this before trusting any percentage
here:** earlier versions of this README quoted "90-100% hybrid recall"
sourced from `runner_v3.py`'s `MockLLMProvider`. That mock sampled its
fake output directly from the same ground-truth dict used to *score*
recall — circular by construction, guaranteed to look good regardless of
what the real extraction logic did. It measured nothing about actual LLM
extraction quality. That number has been removed rather than replaced
with a better-sounding one we can't back up.

What `runner_v3.py` now actually does:
- **Default mode** verifies `HybridExtractor.merge()`'s logic contract
  against fixtures with deliberately known overlap (corroborated /
  LLM-only / heuristic-only items) — confirms merge never drops an item
  either source found, and applies confidence tiers (0.95 corroborated,
  0.80 LLM-only, 0.65 heuristic-only) correctly. This is a real,
  non-circular check, but it's a logic-contract test, not a recall
  measurement.
- **`--live` mode** calls a real configured provider (`ANTHROPIC_API_KEY`
  or `OPENAI_API_KEY`) and scores its actual output against ground truth.
  This is the only path that produces a number meaningful enough to put
  in a table. Run it yourself — we're not publishing a live-mode number
  here because n=3 sessions is too small a sample to generalize, and
  publishing one without a large, ongoing benchmark would just be
  swapping one unsubstantiated number for another.

Heuristic-only numbers above (76-100%) ARE real, deterministic,
reproducible measurements — `runner_v2.py` runs actual heuristic
extraction against actual ground truth with no LLM and no mocking
involved, which is why those numbers are presented with confidence
and the LLM ones currently are not.


---

[← Back to the README](../README.md)
