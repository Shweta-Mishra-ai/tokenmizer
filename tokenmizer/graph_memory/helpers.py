"""
Pure helper functions for Graph Memory.

Extracted from graph.py to keep that file focused on GraphMemory behavior.
Re-exported from graph.py for backward compatibility — existing imports like
`from tokenmizer.graph_memory.graph import _content_to_text` continue to work.
"""
from __future__ import annotations

import re


def _extract_evidence_from_text(text: str) -> str:
    """
    Auto-extract the strongest evidence signal from a text snippet.
    Priority: numeric metric > standard/recommendation > quoted reason

    Examples:
        "latency was 340ms with Postgres" → "340ms"
        "OWASP recommends cost factor 10-12" → "OWASP recommends cost factor 10-12"
        "too expensive at $50/month" → "$50/month"
    """
    # Priority 1: numeric metric with unit
    metric = re.search(
        r'(\d+(?:\.\d+)?\s*(?:ms|s|seconds?|minutes?|%|mb|gb|\$/(?:month|mo|year|yr)|'
        r'x\s+(?:faster|slower)|rpm|rps|req/s)[^.!?\n]{0,30})',
        text, re.IGNORECASE
    )
    if metric:
        return metric.group(1).strip()

    # Priority 2: standard/recommendation reference
    standard = re.search(
        r'(?:OWASP|RFC \d+|ISO \d+|W3C|Lighthouse|Google|industry standard|'
        r'best practice|specification)[^.!?\n]{0,60}',
        text, re.IGNORECASE
    )
    if standard:
        return standard.group(0).strip()

    # Priority 3: dollar cost
    cost = re.search(r'\$\d+(?:\.\d+)?(?:/(?:month|mo|year|yr))?[^.!?\n]{0,20}', text)
    if cost:
        return cost.group(0).strip()

    # Priority 4: quoted phrase
    quoted = re.search(r'["\']([^"\']{10,80})["\']', text)
    if quoted:
        return f'"{quoted.group(1)}"'

    return ""


def _content_to_text(content) -> str:
    """
    Normalize message content to a plain string.

    Handles:
    - str: returned as-is
    - None: empty string (some providers send None for tool-call-only messages)
    - list: multimodal content blocks — extract and join "text" fields
            e.g. [{"type":"text","text":"hi"}, {"type":"image","source":{...}}]
    - dict: single content block — extract "text" field if present
    - anything else: str() fallback
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "text" and "content" in block:
                    parts.append(str(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content)


def _infer_trigger(old_label: str, new_label: str, new_summary: str) -> str:
    """
    Heuristic: infer what triggered a decision change from context clues.
    Returns a short trigger string for DecisionTransition.trigger.

    Examples:
      "Use PostgreSQL" → "Use SQLite"  + summary "too expensive"
      → trigger: "cost constraint"

      "Use FastAPI" → "Use Flask"  + summary "team prefers"
      → trigger: "team preference"
    """
    text = (new_summary or "").lower()
    # Cost / budget signals
    if any(w in text for w in ["cost", "expensive", "budget", "cheap", "free", "price"]):
        return "cost constraint"
    # Performance signals
    if any(w in text for w in ["slow", "fast", "performance", "latency", "speed", "memory"]):
        return "performance requirement"
    # Simplicity / MVP signals
    if any(w in text for w in ["simple", "simpler", "mvp", "prototype", "quick", "easy"]):
        return "simplicity preference"
    # Compatibility / integration
    if any(w in text for w in ["compatible", "integrate", "works with", "support", "library"]):
        return "compatibility requirement"
    # Team / preference signals
    if any(w in text for w in ["prefer", "team", "familiar", "experience", "know"]):
        return "team preference"
    # Scale signals — check BEFORE requirement (scale implies a requirement)
    if any(w in text for w in ["scale", "scalab", "concurrent", "traffic", "load", "users"]):
        return "scale requirement"
    # Requirement change
    if any(w in text for w in ["require", "must", "need", "mandatory", "constraint"]):
        return "requirement change"
    # Default: describe the topic change
    return f"decision revised: {old_label[:30]!r} → {new_label[:30]!r}"
