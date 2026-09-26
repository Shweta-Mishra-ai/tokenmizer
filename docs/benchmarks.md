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
pytest tests/ -q                                     # 1808 tests
```

## Extraction quality — precision, recall and F1

`python -m benchmarks.eval` scores extraction against a labelled corpus:
**14 sessions, 144 turns, 172 labelled items, 14 domains** (Go, Rust,
Python, TypeScript, React, SQL, CI, ML, plus six real audit sessions).
Measured on v0.5.4:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Files | 98% | 100% | **99%** |
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
| Synthetic (hand-written) | 8 | **97%** |
| Real (captured transcripts) | 6 | **90%** |

**Treat 90% as the number that describes real sessions.** The five-point
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

**recall@6 82%** with keyword ranking. Every case is checked to be
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
| Decisions | 0% | **100%** |
| Errors | 0% | **82%** |
| **macro F1** | **11%** | **96%** |

Add `--ignore-packs` to reproduce the left column. The coding corpus is
unchanged at 97%, because a pack's pattern families run *after* the
coding ones and can only add recall.

These three sessions are hand-written, which the harness reports as
`synthetic`. The caveat that applies to the coding fixtures applies here
with more force: three sessions, one author, and the same person wrote
the patterns. Treat 96% as "the mechanism works on sessions of this
shape", not as a generalisation claim.

## Resume quality — what survives windowing

`python -m benchmarks.resume_quality.runner`. Windowing replaces every
turn older than the protected tail with the resume block, so anything the
ontology has no node for — a budget, a deadline, a licence restriction, a
latency target — used to leave the session at that point permanently.

| | before | after |
|---|---|---|
| Out-of-ontology facts still readable in the resume block | 17% | **100%** |
| Resume block, per session | — | **+23 tokens** |
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
