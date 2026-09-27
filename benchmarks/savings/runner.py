"""
What a session costs with TokenMizer in front of it, and without.

A chat client resends the whole conversation on every request. This
replays a conversation that way, through the real proxy and the real
Anthropic adapter, at several lengths, and prices what reaches the
provider. The one thing replaced is the network: the Anthropic SDK client
is swapped for a stand-in that applies the provider's documented prompt
caching rules to the exact request it is handed —

  * the cache key is the byte-exact prefix up to a `cache_control` marker,
    rendered system first, then messages;
  * a request reads the longest prefix an earlier request wrote, writes
    from there to its last marker, and pays full price for the rest;
  * a marker on a prefix shorter than the model's minimum writes nothing;
  * reads cost 0.1x the input price, 5-minute writes 1.25x.

and returns the usage block the real API would. Nothing about TokenMizer
is stubbed, so a change that breaks the cacheable prefix shows up here as
cost, which is the point.

Three columns per length:

  direct         the client talks to the provider itself and sends no
                 cache markers — what an OpenAI-shaped client pointed
                 straight at Anthropic does
  direct+cache   the same client, caching its own history the way this
                 proxy does (a marker on the turn before the last) — the
                 honest comparison for a client that already caches
  tokenmizer     the same conversation through the proxy

HONEST LIMITATIONS, read before quoting a number:

  * INPUT ONLY. The stand-in returns a fixed one-word answer, so what the
    terse prompt saves on output — its whole reason to exist — is not
    measured. Its input cost (about 50 tokens a request) IS counted, so
    the proxy is charged for it and credited nothing.
  * The conversation is real transcript text (the six captured sessions
    in benchmarks/eval/corpus) cycled to length, not one real session of
    that length. Turns are short, so the fixed per-request overhead is a
    larger share than it would be on a real coding session.
  * Cache entries never expire here; the real 5-minute TTL would turn
    some reads into writes for a client that pauses between turns.
  * Token counts come from tokenmizer.core.tokenizer, which falls back to
    a character estimate when no tokenizer can load. The run says which.
  * Whether answers stay as good once history is windowed is a quality
    question this cannot see. That needs a real model.

Run:  python -m benchmarks.savings.runner [--model claude-sonnet-5] [--turns 10 40 150 300]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

READ, WRITE = 0.1, 1.25


def _render(system, messages) -> list[tuple[str, int, bool]]:
    """(serialized block, tokens, carries a cache marker), in render order."""
    from tokenmizer.core.tokenizer import count_tokens

    blocks: list[tuple[str, int, bool]] = []

    def add(obj, text, marked):
        blocks.append((json.dumps(obj, sort_keys=True, default=str),
                       count_tokens(text or "", "claude"), marked))

    # A string is rendered exactly as one text block, marked or not, so both
    # spellings must hash the same or every marked turn looks new.
    if isinstance(system, str) and system:
        system = [{"type": "text", "text": system}]
    for b in system or []:
        add({"system": {k: v for k, v in b.items() if k != "cache_control"}},
            b.get("text", ""), "cache_control" in b)
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in content:
            text = b.get("text") or json.dumps(b.get("content", ""), default=str)
            add({"role": m["role"],
                 "block": {k: v for k, v in b.items() if k != "cache_control"}},
                text, "cache_control" in b)
    return blocks


class _PromptCache:
    """The provider side of prompt caching, as documented."""

    def __init__(self, minimum: int):
        self.minimum = minimum
        self.entries: set[str] = set()

    def bill(self, system, messages) -> dict:
        blocks = _render(system, messages)
        total = sum(t for _, t, _ in blocks)
        h = hashlib.sha256()
        prefix_hash, prefix_tokens = [], []
        running = 0
        for serialized, tokens, _ in blocks:
            h.update(serialized.encode())
            running += tokens
            prefix_hash.append(h.hexdigest())
            prefix_tokens.append(running)

        marks = [i for i, (_, _, marked) in enumerate(blocks) if marked]
        read = 0
        for i in range(len(blocks)):
            if prefix_hash[i] in self.entries and (not marks or i <= marks[-1]):
                read = prefix_tokens[i]
        write = 0
        for i in marks:
            if prefix_tokens[i] >= self.minimum:
                self.entries.add(prefix_hash[i])
                write = max(write, prefix_tokens[i] - read)
        return {"total": total, "read": read, "write": write,
                "uncached": total - read - write}


def _fake_anthropic(cache: _PromptCache, ledger: list):
    class Messages:
        async def create(self, *, model, messages, max_tokens, system=None, **kw):
            bill = cache.bill(system, messages)
            ledger.append(bill)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=bill["uncached"], output_tokens=1,
                    cache_read_input_tokens=bill["read"],
                    cache_creation_input_tokens=bill["write"]),
            )

    class Client:
        def __init__(self, *a, **k):
            self.messages = Messages()

    return Client


def _cost(bill: dict) -> float:
    return bill["uncached"] + READ * bill["read"] + WRITE * bill["write"]


def _conversation(turns: int) -> list[tuple[str, str]]:
    from benchmarks.eval.corpus import load

    pairs = []
    for s in load():
        if s.origin != "real":
            continue
        ms = s.messages
        pairs += [(ms[i]["content"], ms[i + 1]["content"])
                  for i in range(0, len(ms) - 1, 2)]
    return [(f"[turn {t}] {pairs[t % len(pairs)][0]}", pairs[t % len(pairs)][1])
            for t in range(turns)]


def _direct(turns, model, cache_history: bool) -> list[dict]:
    """The client alone: every request is its own full history."""
    from tokenmizer.providers.providers import _cache_minimum, _mark_history_cacheable

    cache = _PromptCache(_cache_minimum(model))
    history, bills = [], []
    for user, assistant in _conversation(turns):
        history.append({"role": "user", "content": user})
        sent = _mark_history_cacheable(history) if cache_history else history
        bills.append(cache.bill(None, sent))
        history.append({"role": "assistant", "content": assistant})
    return bills


def _through_proxy(turns, model) -> tuple[list[dict], int]:
    os.environ["TOKENMIZER_GRAPH_CHECKPOINT__STORAGE_DIR"] = tempfile.mkdtemp()
    import anthropic
    from fastapi.testclient import TestClient

    from tokenmizer.api import app as A
    from tokenmizer.providers.providers import AnthropicProvider, _cache_minimum

    ledger: list[dict] = []
    anthropic.AsyncAnthropic = _fake_anthropic(_PromptCache(_cache_minimum(model)), ledger)
    provider = AnthropicProvider(api_key="benchmark", model=model)
    A._get_provider = lambda: provider

    async def _unlimited(request):
        return None

    A._check_rate_limit = _unlimited
    A.settings.provider = "anthropic"
    A.settings.cache.enabled = False
    A.settings.api_key = ""
    # A fresh directory per run. A reused one carries the previous run's
    # processed-message hashes, extraction early-returns, and the numbers
    # measure deduplication instead of the proxy.
    A.settings.graph_checkpoint.storage_dir = tempfile.mkdtemp()
    A._graph_cache.clear()

    session = f"savings-{turns}"
    history: list[dict] = []
    with TestClient(A.app) as c:
        for user, assistant in _conversation(turns):
            history.append({"role": "user", "content": user})
            r = c.post("/v1/chat/completions",
                       json={"model": model, "messages": history,
                             "session_id": session})
            if r.status_code != 200:
                raise SystemExit(f"proxy returned {r.status_code}: {r.text[:300]}")
            history.append({"role": "assistant", "content": assistant})
        nodes = len(A._graph_cache[session]._nodes) if session in A._graph_cache else 0
    if len(ledger) != turns:
        raise SystemExit(f"{len(ledger)} provider calls for {turns} turns")
    return ledger, nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--turns", type=int, nargs="+", default=[10, 40, 150, 300])
    args = ap.parse_args()

    import logging
    logging.disable(logging.WARNING)
    from tokenmizer.providers.providers import _cache_minimum
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        tokenizer = "tiktoken"
    except Exception:
        tokenizer = "CHARACTER ESTIMATE (no tokenizer could load)"

    print(f"model {args.model} · cache minimum {_cache_minimum(args.model)} tokens"
          f" · tokenizer: {tokenizer}")
    print("cost is in input-token units: full price 1, cache read "
          f"{READ}, cache write {WRITE}. Input only — see the module docstring.\n")
    print(f"{'turns':>6} {'history':>8} | {'direct':>10} {'direct+cache':>13}"
          f" {'tokenmizer':>11} | {'vs direct':>9} {'vs +cache':>9} | nodes")
    for n in args.turns:
        plain = _direct(n, args.model, cache_history=False)
        cached = _direct(n, args.model, cache_history=True)
        proxied, nodes = _through_proxy(n, args.model)
        if nodes == 0:
            raise SystemExit("the graph is empty: extraction did not run, so "
                             "this would measure an empty proxy")
        a, b, c = (sum(map(_cost, x)) for x in (plain, cached, proxied))
        print(f"{n:>6} {plain[-1]['total']:>8} | {a:>10.0f} {b:>13.0f} {c:>11.0f} |"
              f" {100 * (a - c) / a:>8.1f}% {100 * (b - c) / b:>8.1f}% | {nodes}")


if __name__ == "__main__":
    main()
