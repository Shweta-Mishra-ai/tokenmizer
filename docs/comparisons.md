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

Moved to [docs/roadmap.md](roadmap.md), which pairs every planned item
with the measurement that motivates it. Shipped so far: v0.3 SSE
passthrough; v0.4 ontology and reasoning API; v0.5 per-row storage,
cross-process safety, ownership, semantic recall; v0.6 (unreleased)
cross-session recall, LLM extraction on the chat provider, tool calling.

Have a use case that doesn't fit? [Open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose) — extraction misses have their own issue template.


---

[← Back to the README](../README.md)
