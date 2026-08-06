# How TokenMizer compares

Where TokenMizer sits next to other tools, and where it does not compete with them.

---

## Why TokenMizer and not X?

Engineers ask this every time. Honest answers:

**Why not just use Git history?**
Git stores *what changed*, not *why you decided to change it*. You can't ask Git "what did we decide about auth?" or "why did we switch from MySQL to PostgreSQL?" TokenMizer stores decisions with trigger, reason, and evidence — not diffs.

**Why not RAG (retrieval-augmented generation)?**
RAG retrieves *relevant chunks* — it doesn't model *decision state*. If you switched from bcrypt to Argon2 mid-session, RAG might retrieve both and confuse the model about which is current. TokenMizer tracks decision supersession explicitly: old decision is marked `SUPERSEDED`, new decision is `ACTIVE`. Resume context only includes current state.

**Why not a plain summary at the start of each session?**
Summaries lose structure. You can't query "all superseded decisions" or "what triggered the auth change" from a blob of text. Our benchmark shows graph memory preserves +5% more information than a summary baseline — and unlike summaries, the graph is queryable, editable, and grows incrementally without re-summarizing everything each turn.

**Why not Mem0 or Zep?**
Mem0 and Zep store *facts* ("user prefers Python"). TokenMizer stores *decisions with rationale* — the full causal chain: what was decided, what replaced it, why, what evidence triggered the change, and how confidence shifted. If you need "remember my name across sessions," use Mem0. If you need "remember that we switched from PostgreSQL to SQLite because of cost, and here's the evidence," use TokenMizer.

**Why not just a longer context window?**
Longer context = higher cost + slower inference + model attention dilution on long histories. TokenMizer compresses a session into a resume block averaging **249 tokens** (measured, n=3 — see [Benchmarks](benchmarks.md)) — not by summarizing, but by extracting what actually matters: goals, active decisions, current tasks, recent errors.

---

## Running alongside other token tools

Token tooling divides along one axis: what you send, what you get back,
and what you remember. TokenMizer is the third. It composes with the
other two rather than competing with them.

| Layer | Tool | Where it acts |
|---|---|---|
| Output length | **Caveman** | Shortens what the model writes back |
| Input trimming | **CodeBurn** | Trims the context you send |
| **Memory** | **TokenMizer** | Keeps the decisions, files and errors across the context limit |

> **If you run Caveman too,** set `terse_output.enabled: false` in
> `tokenmizer.yaml`. Both inject a system prompt asking for brevity, and
> two of them fight each other.

---

## Roadmap

| Version | Focus |
|---|---|
| **v0.3** | SSE streaming passthrough (checkpoint on stream close) |
| **v0.4** | Graph ontology · deterministic reasoning API (`why`, `impact`, consistency checks) |
| **v0.5** | Per-row storage schema · cross-process write safety · session ownership · durability guarantees · measured extraction quality *(this release)* |
| v0.6 | Cross-session memory · embedding-based edge linking · LLM-assisted extraction for the defects regexes cannot reach |
| Research | Real-transcript benchmark suite → paper ([tokenmizer-research](https://github.com/Shweta-Mishra-ai/tokenmizer-research)) |

Have a use case that doesn't fit? [Open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose) — extraction misses have their own issue template.


---

[← Back to the README](../README.md)
