"""
Pure helper functions for Graph Memory.

Extracted from graph.py to keep that file focused on GraphMemory behavior.
Re-exported from graph.py for backward compatibility — existing imports like
`from tokenmizer.graph_memory.graph import _content_to_text` continue to work.
"""
from __future__ import annotations

import json
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


# ── Tool calls and tool results ──────────────────────────────────────────────
#
# An agent states most of what it does through tools, not prose: it edits a
# file with `Edit(file_path=...)`, runs the tests with `Bash(...)`, and hears
# back a traceback in a tool result. _content_to_text keeps only text blocks
# — correctly, since that is what counts toward tokens and what compression
# rewrites — so extraction saw none of it. On a real 612-message agent
# session, most of the files the agent had edited never became FILE nodes.
#
# tool_signals() reads those structured parts directly. It is deliberately
# narrow: file paths come from the ARGUMENTS of tools that change files (a
# read-only tool looking at forty files is browsing, not work), and errors
# come only from results the protocol marks as failed or whose first line
# is unambiguously an error. Nothing here scans tool OUTPUT as prose — a
# file's source code or a test log is full of words the prose patterns would
# misread.

# Argument keys that name a file across the common agent toolkits (Claude
# Code, OpenAI Agents, LangChain file tools, aider, Continue).
_PATH_KEYS = frozenset({
    "file_path", "filepath", "filePath", "path", "file", "filename",
    "file_name", "notebook_path", "target_file", "target", "destination",
    "dest", "new_path", "old_path", "source_path",
})
_PATH_LIST_KEYS = frozenset({"paths", "files", "file_paths", "filePaths"})

# Tools that only look. Matched on the lower-cased name.
_READ_ONLY_TOOL = re.compile(
    r"(?:^|[_\-.])(?:read|view|cat|open|get|list|ls|glob|grep|search|find|"
    r"fetch|lookup|show|describe|stat|head|tail|diff|status|inspect)(?:$|[_\-.])|"
    r"^(?:read|view|glob|grep|ls|list|search|find|fetch|get)",
    re.IGNORECASE,
)

# A path worth a FILE node: no whitespace, a file extension or a known
# extensionless build file, and not a URL.
_TOOL_FILE = re.compile(
    r"^(?!https?://)[^\s\"'<>|*?]{1,260}?"
    r"(?:\.[A-Za-z0-9]{1,8}|/(?:Dockerfile|Makefile|Procfile|Jenkinsfile|Gemfile|"
    r"Rakefile|Podfile|Fastfile|Vagrantfile|Justfile|Caddyfile|Containerfile)|"
    r"^(?:Dockerfile|Makefile|Procfile|Jenkinsfile|Gemfile|Rakefile|Podfile|"
    r"Fastfile|Vagrantfile|Justfile|Caddyfile|Containerfile))$"
)

# The first line of a failed tool result that names the failure.
_TOOL_ERROR_LINE = re.compile(
    r"^\s*(?:[\w.]*(?:Error|Exception|Failure)\b[:(]?.*|"
    r"(?:error|fatal|failed|failure|panic|exception)\b\s*[:!\-].+|"
    r"E\d{3,5}\b.+|npm ERR!.+|FAILED\s+\S+.*|.+: command not found|"
    r".*\bexit(?:ed)? (?:with )?(?:code|status) [1-9]\d*.*|"
    r"Traceback \(most recent call last\):)\s*$",
    re.IGNORECASE,
)


_EXIT_ONLY = re.compile(r"^\W*exit(?:ed)? (?:with )?(?:code|status) \d+\W*$", re.IGNORECASE)


def _walk_paths(args, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(args, dict):
        for k, v in args.items():
            if k in _PATH_KEYS and isinstance(v, str):
                v = v.strip()
                if _TOOL_FILE.match(v):
                    out.append(v)
            elif k in _PATH_LIST_KEYS and isinstance(v, list):
                for x in v[:50]:
                    if isinstance(x, str) and _TOOL_FILE.match(x.strip()):
                        out.append(x.strip())
            elif isinstance(v, (dict, list)):
                _walk_paths(v, out, depth + 1)
    elif isinstance(args, list):
        for x in args[:50]:
            _walk_paths(x, out, depth + 1)


def _tool_error_label(text: str) -> str:
    """The line of a failed tool's output that says what failed.

    A traceback names the exception on its LAST line; everything else
    leads with it. Capped, since a tool result can be megabytes.
    """
    # Lines are capped too: one minified 20 KB line is still one line.
    lines = [ln.strip()[:2000] for ln in (text or "")[:20000].splitlines() if ln.strip()]
    if not lines:
        return ""
    # A traceback may follow a line like "Exit code 1"; wherever it starts,
    # the exception it ends in is the label, never the header line.
    tb = next((i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent")), None)
    if tb is not None:
        for ln in reversed(lines[tb:]):
            if re.match(r"^[\w.]*(?:Error|Exception|Exit|Interrupt)\w*\b", ln):
                return ln[:160]
        lines = lines[:tb] or lines[tb + 1:]
    # The specific line beats the generic one: "FAILED tests/test_a.py::
    # test_x - AssertionError" says what broke, "Exit code 1" only that
    # something did.
    matches = [ln for ln in lines[:8] if _TOOL_ERROR_LINE.match(ln)]
    specific = [ln for ln in matches if not _EXIT_ONLY.match(ln)]
    chosen = (specific or matches or [""])[0]
    return chosen[:160]


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content
        )
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return ""


def tool_signals(message: dict) -> tuple[list[str], list[str]]:
    """(file paths, error labels) from a message's tool calls and results.

    Handles the OpenAI shape (`tool_calls` on an assistant message, a
    `role: "tool"` result), the Anthropic shape (`tool_use` / `tool_result`
    content blocks, `is_error`), and the Gemini shape (`functionCall` parts).
    Never raises: a malformed tool call from any client is skipped, not
    allowed to break extraction for the rest of the conversation.
    """
    files: list[str] = []
    errors: list[str] = []
    if not isinstance(message, dict):
        return files, errors
    try:
        calls = []   # (name, args)
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = None
                calls.append((str(fn.get("name") or ""), args))
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "tool_use":
                calls.append((str(b.get("name") or ""), b.get("input")))
            elif btype == "tool_result" and b.get("is_error"):
                label = _tool_error_label(_block_text(b.get("content")))
                if label:
                    errors.append(label)
            elif "functionCall" in b and isinstance(b["functionCall"], dict):
                fc = b["functionCall"]
                calls.append((str(fc.get("name") or ""), fc.get("args")))
        if message.get("role") == "tool":
            # OpenAI tool results carry no error flag; only a result whose
            # first line is itself an error line counts.
            text = _block_text(content)
            first = next((ln for ln in text[:2000].splitlines() if ln.strip()), "")
            if _TOOL_ERROR_LINE.match(first):
                label = _tool_error_label(text)
                if label:
                    errors.append(label)
        for name, args in calls:
            if name and _READ_ONLY_TOOL.search(name):
                continue
            _walk_paths(args, files)
    except (TypeError, AttributeError, ValueError):
        return files, errors
    return files, errors


# A lone UTF-16 surrogate — the JSON escape "\ud800" with no partner — is a
# legal Python string character and an illegal one in UTF-8. json.loads
# accepts it, so any client can send one; the first .encode() that meets it
# raises, and extraction for the whole conversation failed with it. Replaced
# with U+FFFD, the character Unicode reserves for exactly this.
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


def scrub_surrogates(obj):
    """`obj` with every lone surrogate in every string replaced by U+FFFD.
    Lists and dicts are copied, not mutated; other values pass through."""
    if isinstance(obj, str):
        return _LONE_SURROGATE.sub("\ufffd", obj) if _LONE_SURROGATE.search(obj) else obj
    if isinstance(obj, list):
        return [scrub_surrogates(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_surrogates(v) for k, v in obj.items()}
    return obj
