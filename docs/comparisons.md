# How TokenMizer compares

Where TokenMizer sits next to other tools, and where it does not compete with them.

The "why not just use X" comparisons live in the
[README](../README.md#why-tokenmizer-and-not-x), not here, so there is one
copy to keep accurate rather than two that can drift apart.

---

## Running alongside other token tools

Token tooling divides along one axis: what you send, what you get back,
and what you remember. TokenMizer covers all three, and the third is the
one nothing else in the category does.

| Layer | What acts on it | Where TokenMizer does it |
|---|---|---|
| Output length | An output-brevity prompt | `terse_output` (styles: `terse`, `minimal`) |
| Input size | Context trimming | Compression, windowing, file intelligence |
| **Memory** | **Nothing else in this category** | **The session graph: decisions, files and errors kept across the context limit** |

> **If you already inject a brevity prompt** from another tool, set
> `terse_output.enabled: false` in `tokenmizer.yaml`. Two prompts asking for
> brevity fight each other. Alternatively drop the other one and set
> `terse_output.style: minimal`, which asks for smaller changes as well as
> shorter answers — reuse before writing, the standard library before a
> dependency, the shortest diff that works — so there is one brevity prompt
> rather than two. It costs the tokens of that prompt on every turn; the
> saving is in what the model writes back, and shows up in the output-trim
> figures rather than being estimated up front.

---

## Roadmap

| Version | Focus |
|---|---|
| **v0.3** | SSE streaming passthrough (checkpoint on stream close) |
| **v0.4** | Graph ontology · deterministic reasoning API (`why`, `impact`, consistency checks) |
| **v0.5** | Per-row storage schema · cross-process write safety · session ownership · durability guarantees · embedding-based semantic recall and conflict/dedup matching · measured extraction quality *(this release)* |
| v0.6 | Cross-session memory · LLM-assisted extraction for the defects regexes cannot reach |
| Research | 100-session, 8-method benchmark → paper ([tokenmizer-research](https://github.com/Shweta-Mishra-ai/tokenmizer-research)) — TokenMizer 0.5.4 ties for first at 60% macro F1, level with Mem0-style and Graphiti-style |

Have a use case that doesn't fit? [Open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose) — extraction misses have their own issue template.


---

[← Back to the README](../README.md)
