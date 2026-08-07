# How TokenMizer compares

Where TokenMizer sits next to other tools, and where it does not compete with them.

The "why not just use X" comparisons live in the
[README](../README.md#why-tokenmizer-and-not-x), not here, so there is one
copy to keep accurate rather than two that can drift apart.

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
