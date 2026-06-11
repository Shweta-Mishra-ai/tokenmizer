"""
Secret & credential redaction.
Applied before ANY node is written to the graph, any checkpoint is saved,
or any error message is logged.
"""
from __future__ import annotations
import re

_PATTERNS = [
    # Anthropic
    re.compile(r'sk-ant-[A-Za-z0-9\-_]{3,}-[A-Za-z0-9\-_]{10,}', re.IGNORECASE),
    # OpenAI
    re.compile(r'sk-[A-Za-z0-9]{32,}', re.IGNORECASE),
    re.compile(r'sk-proj-[A-Za-z0-9\-_]{20,}', re.IGNORECASE),
    # Google / Gemini
    re.compile(r'AIza[A-Za-z0-9\-_]{35}', re.IGNORECASE),
    # GitHub PATs
    re.compile(r'ghp_[A-Za-z0-9]{36}', re.IGNORECASE),
    re.compile(r'ghs_[A-Za-z0-9]{36}', re.IGNORECASE),
    # Generic: password=..., secret=..., token=..., key=...
    re.compile(
        r'(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key'
        r'|private[_\-]?key|auth[_\-]?key)\s*[=:]\s*["\']?[\w\-\.]{8,}',
        re.IGNORECASE,
    ),
    # Bearer tokens in Authorization headers
    re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{20,}', re.IGNORECASE),
    # Email addresses (PII)
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'),
]


def redact(text: str) -> str:
    """Replace all detected secrets with [REDACTED]."""
    for pat in _PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def redact_node(label: str, summary: str = "") -> tuple[str, str]:
    return redact(label), redact(summary)


def redact_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with secrets scrubbed from content."""
    return [
        {**m, "content": redact(m.get("content", ""))}
        for m in messages
    ]
