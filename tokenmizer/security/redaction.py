"""
Secret & credential redaction.
Applied before ANY node is written to the graph, any checkpoint is saved,
any error message is logged, or any message is sent to ANY LLM provider
(including the cheap background-extraction provider, not just the main
chat provider).

SECURITY NOTE (fixed): previously, only `_call_provider()` in api/app.py
called redact_messages() on its own local copy of `messages`. The
background graph-extraction path (HybridExtractor → cheap LLM provider
such as haiku/gpt-4o-mini/deepseek) received `raw_messages` directly,
UNREDACTED. That meant a real API key, password, or token pasted into a
coding session would be sent verbatim to a third-party extraction model
before this module ever saw it. Redaction is now applied once, at
ingestion, in chat_completions() — see api/app.py — so every downstream
consumer shares the same already-safe copy.
"""
from __future__ import annotations

import re

_PATTERNS = [
    # Anthropic
    re.compile(r'sk-ant-[A-Za-z0-9\-_]{3,}-[A-Za-z0-9\-_]{10,}', re.IGNORECASE),
    # OpenAI
    re.compile(r'sk-proj-[A-Za-z0-9\-_]{20,}', re.IGNORECASE),
    re.compile(r'sk-[A-Za-z0-9]{32,}', re.IGNORECASE),
    # Google / Gemini
    re.compile(r'AIza[A-Za-z0-9\-_]{35}', re.IGNORECASE),
    # GitHub PATs / app tokens
    re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}', re.IGNORECASE),
    # AWS access keys (not the secret — secrets are 40 random b64 chars and
    # collide too easily with normal text; we redact the identifying access
    # key ID, which is the part that's safe to pattern-match confidently)
    re.compile(r'AKIA[0-9A-Z]{16}'),
    re.compile(r'ASIA[0-9A-Z]{16}'),
    # Slack tokens
    re.compile(r'xox[baprs]-[A-Za-z0-9\-]{10,}', re.IGNORECASE),
    # Stripe
    re.compile(r'(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}', re.IGNORECASE),
    # Generic JWTs (three base64url segments separated by dots)
    re.compile(r'eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+'),
    # Bearer tokens in Authorization headers
    re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{20,}', re.IGNORECASE),
    # Generic: password=..., secret=..., token=..., key=...
    re.compile(
        r'(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key'
        r'|private[_\-]?key|auth[_\-]?key|client[_\-]?secret)\s*[=:]\s*'
        r'["\']?[\w\-\.\/+]{8,}',
        re.IGNORECASE,
    ),
    # AUDIT FIX (2026-07-10): URL-embedded credentials — a DATABASE_URL like
    # postgres://admin:S3cr3tPass@db.internal:5432/prod carried a live
    # password that matched NO pattern above (no "password=" literal, no
    # recognized token prefix). Redact the userinfo section of any URL that
    # has one. Only the credential part is replaced; scheme/host survive so
    # the message stays readable.
    re.compile(r'(?<=://)[^\s/:@]+:[^\s/@]+(?=@)'),
    # AUDIT FIX (2026-07-10): provider key formats that were missing —
    # best-effort, documented as such (providers rotate formats):
    # Cohere trial/prod keys are 40 alnum chars bound to a "co-" style
    # context we can't anchor on, so we rely on the generic key=... rule
    # for those; the below are the anchorable ones.
    # xAI / Grok
    re.compile(r'xai-[A-Za-z0-9]{20,}'),
    # OpenRouter
    re.compile(r'sk-or-[A-Za-z0-9\-_]{20,}', re.IGNORECASE),
    # Hugging Face
    re.compile(r'hf_[A-Za-z0-9]{30,}'),
    # Together AI (64 hex chars after explicit assignment handled by generic
    # rule; standalone "together_" prefixed keys:)
    re.compile(r'together_[A-Za-z0-9]{20,}', re.IGNORECASE),
    # Email addresses (PII)
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'),
]


def redact(text: str) -> str:
    """Replace all detected secrets with [REDACTED]. Non-string input is
    passed through as the empty string rather than raising, since callers
    (graph nodes, message content) may legitimately have None/empty values."""
    if not isinstance(text, str):
        return ""
    for pat in _PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def redact_node(label: str, summary: str = "") -> tuple[str, str]:
    return redact(label), redact(summary)


def _redact_content(content):
    """
    Redact secrets from message content of any shape.

    Previously this assumed `content` was always a `str` and called
    `redact()` directly on it. Two real failure modes existed:
      1. content=None (common for tool-call-only messages) → re.sub on
         None raises TypeError, which would have bubbled up as a 500 on
         a perfectly normal multi-turn tool-use conversation.
      2. content=list (multimodal: text + image/document blocks, per the
         Anthropic/OpenAI message schema) → redact() would never see the
         text, OR (depending on caller) the whole list would get cast to
         str(), embedding raw, unredacted secrets inside a stringified
         repr that nothing downstream expected.

    Fix: text blocks are redacted in place; non-text blocks (images,
    documents, tool_use/tool_result) are passed through unchanged — we
    must never attempt to regex-redact binary/base64 image data, both
    because it's not text and because doing so would corrupt the image.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return redact(content)
    if isinstance(content, list):
        cleaned = []
        for block in content:
            if isinstance(block, str):
                cleaned.append(redact(block))
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    cleaned.append({**block, "text": redact(str(block["text"]))})
                elif "content" in block and isinstance(block.get("content"), str):
                    cleaned.append({**block, "content": redact(block["content"])})
                else:
                    # image/document/tool_use/tool_result blocks — leave untouched
                    cleaned.append(block)
            else:
                cleaned.append(block)
        return cleaned
    if isinstance(content, dict):
        if "text" in content:
            return {**content, "text": redact(str(content["text"]))}
        return content
    return content


def redact_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with secrets scrubbed from content.
    Safe for plain-string content, multimodal block lists, and missing/None
    content (tool-call-only messages)."""
    return [
        {**m, "content": _redact_content(m.get("content"))}
        for m in messages
    ]
