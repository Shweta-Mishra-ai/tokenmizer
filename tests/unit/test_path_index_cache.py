"""path_index caches the paths of each tool output; the answer must not change."""
from __future__ import annotations

import random

from tokenmizer.compression import window as W


def _reference(dropped, budget, model="gpt-4o"):
    """path_index as it was before the cache: every word read every time."""
    seen: dict[str, None] = {}
    budget_chars = W._MAX_INDEX_CHARS
    recent = []
    for m in reversed(dropped):
        size = len(m.get("content") or "") if isinstance(m.get("content"), str) else 0
        for call in m.get("tool_calls") or []:
            size += len((call.get("function") or {}).get("arguments") or "")
        if recent and budget_chars - size < 0:
            break
        budget_chars -= size
        recent.append(m)
    for m in reversed(recent):
        texts = []
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            texts.append(m["content"])
        for call in m.get("tool_calls") or []:
            texts.append((call.get("function") or {}).get("arguments") or "")
        for text in texts:
            for word in text[:W._MAX_SCAN_CHARS].split():
                path = W._as_path(word)
                if path:
                    seen.setdefault(path)
    return seen


def _session(seed):
    rng = random.Random(seed)
    msgs = []
    for i in range(60):
        lines = [f"src/pkg_{rng.randint(0, 9)}/mod_{rng.randint(0, 50)}.py  "
                 f"{rng.randint(1, 999)} lines" for _ in range(rng.randint(0, 40))]
        lines += ["see https://example.com/a/b and ./docs/readme.md, 3/4 done",
                  "plain words without any path in them", "-flag/like/this"]
        rng.shuffle(lines)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "bash", "arguments": f'{{"cmd": "cat lib/x_{i}.ts"}}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "\n".join(lines)})
    return msgs


def test_cached_index_equals_the_uncached_one():
    for seed in range(5):
        dropped = _session(seed)
        expected = list(_reference(dropped, 10_000))
        W._PATHS_CACHE.clear()
        first = W.path_index(dropped, 10_000)
        again = W.path_index(dropped, 10_000)          # every output now cached
        assert first == again
        listed = [line.split(": ", 1) for line in first.splitlines()[1:]]
        got = [f"{d}{n}" if d != "(no directory)" else n
               for d, names in listed for n in names.split()]
        assert sorted(got) == sorted(expected)


def test_the_cache_is_bounded():
    W._PATHS_CACHE.clear()
    for i in range(W._PATHS_CACHE_MAX + 50):
        W._paths_in(f"src/a_{i}.py")
    assert len(W._PATHS_CACHE) == W._PATHS_CACHE_MAX
