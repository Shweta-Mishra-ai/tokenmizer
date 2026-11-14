"""
Assembling per-turn context: which recalled nodes the payload already
states, and where injected text goes so a provider's prompt cache still
matches.

Split out of api/app.py, which re-exports every name.
"""
from __future__ import annotations

import re

from tokenmizer.core.bounded_cache import MISSING, BoundedCache
from tokenmizer.security.fencing import fence


def _last_substantive_query(raw_messages: list[dict], min_words: int = 4) -> str:
    """The most recent user turn long enough to retrieve against.

    A follow-up like "why?" carries its subject in the turn before it. Walks
    backwards past the short turns rather than giving up on retrieval, so the
    graph is still searched with the topic actually under discussion.
    """
    for message in reversed(raw_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if len(content.split()) >= min_words:
            return content
    return ""


_WORD = re.compile(r"[a-z0-9][a-z0-9_./-]*[a-z0-9]|[a-z0-9]")


_FILLER = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "are",
    "was", "were", "has", "have", "had", "not", "but", "its", "our",
    "use", "using", "will", "would", "should", "can", "could",
})


_EXTENSION = re.compile(r"^(.+)\.[a-z][a-z0-9]{0,4}$")


# Keyed by digest and bounded by words held: an lru_cache keyed on the text
# kept up to 4,096 whole messages alive — 38 MB over 600 turns of one agent
# session with large tool output, and unbounded in bytes.
_WORDS_CACHE: BoundedCache[frozenset] = BoundedCache(4096, 2_000_000, len)


def _content_words(text: str) -> frozenset:
    """The words of `text` that carry content, and for a file name both
    spellings: with its extension and without. A node label is clipped to a
    clause and often ends at `hybrid_extractor` where the turn said
    `hybrid_extractor.py`; they are the same file. Both sides go through
    here, so a label naming `config.py` still needs `config.py`, not
    `config.yaml`, in the turn.

    Memoised: every request recomputed this for every message it carries,
    and all but the newest are the same text as on the previous request."""
    key = BoundedCache.key(text)
    hit = _WORDS_CACHE.get(key)
    if hit is not MISSING:
        return hit
    words = set()
    for w in _WORD.findall(text.lower()):
        if len(w) <= 2 or w in _FILLER:
            continue
        words.add(w)
        stem = _EXTENSION.match(w)
        if stem and len(stem.group(1)) > 2:
            words.add(stem.group(1))
    result = frozenset(words)
    _WORDS_CACHE.put(key, result)
    return result


def _already_in_payload(node, message_words: list[frozenset]) -> bool:
    """True when one message being sent already states this node.

    Context injection exists to put back what the payload no longer
    carries: turns windowing replaced, turns the client truncated on its
    own, facts from another session. A node whose label and summary are
    both fully present in a single message of this payload tells the
    model nothing it is not already reading, and on a short session that
    is every node. Injecting them anyway made a ten-turn session cost 81%
    more input than it would have without the proxy.

    Conservative by construction: a paraphrased label, a detail that only
    exists in the summary, or words split across two messages all read as
    absent, and the node is injected. The failure mode is a duplicate
    line, never a lost fact.
    """
    words = _content_words(f"{node.label} {node.summary[:50] if node.summary else ''}")
    if not words:
        return False
    return any(words <= mw for mw in message_words)


def _append_volatile(messages: list[dict], fenced: str) -> None:
    """Put per-turn context after everything a provider can cache.

    Providers cache the longest byte-identical prefix of a request, and
    render the system prompt before the conversation. Appending a block
    that changes every turn to the system prompt therefore changed the
    first bytes of the conversation too, and every request re-paid for
    the whole history. Appended to the turn being asked instead, the
    system prompt and every earlier turn go out exactly as they did last
    time; the client never sees this text, so on the next request the
    same user turn goes out without it and the prefix still matches.

    Only a trailing user turn is a safe place. Anything else (a tool
    result closing an agent step) keeps the old placement rather than
    inventing a message some provider would reject.
    """
    last = messages[-1] if messages else None
    if last is not None and last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, list):
            last["content"] = [*content, {"type": "text", "text": fenced}]
            return
        if isinstance(content, str) or content is None:
            last["content"] = f"{content or ''}\n\n{fenced}"
            return
    sys_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"), None
    )
    if sys_idx is not None:
        messages[sys_idx]["content"] = f"{messages[sys_idx]['content']}\n\n{fenced}"
    else:
        messages.insert(0, {"role": "system", "content": fenced})


def _inject_context(messages: list[dict], session_id: str,
                    relevant_pairs: list[tuple]) -> None:
    """Append the ranked nodes to this turn, fenced."""
    ctx_parts = [
        f"  {n.type.value}: {n.label}"
        + (f" ({n.summary[:50]})" if n.summary else "")
        # Tag only nodes recalled from a DIFFERENT session.
        + (f" [from session {sid}]" if sid != session_id else "")
        for n, sid in relevant_pairs[:6]
    ]
    ctx_block = "\n".join(ctx_parts)
    _append_volatile(messages, fence(ctx_block, "relevant session context"))
