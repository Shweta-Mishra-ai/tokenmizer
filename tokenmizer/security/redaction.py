"""
Secret & credential redaction.
Applied before ANY node is written to the graph, any checkpoint is saved,
any error message is logged, or any message is sent to ANY LLM provider
(including the cheap background-extraction provider, not just the main
chat provider).

SECURITY NOTE: it is NOT enough for only `_call_provider()` in api/app.py
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

import hashlib
import json
import re
import threading
from collections import OrderedDict

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
    # URL-embedded credentials (postgres://user:pass@host/db): redact the
    # userinfo section only — scheme and host survive so the message stays
    # readable. Connection-string passwords match no keyword or provider
    # pattern, so this rule is their only coverage.
    re.compile(r'(?<=://)[^\s/:@]+:[^\s/@]+(?=@)'),
    # Additional provider key formats with anchorable prefixes. Formats
    # without a stable prefix (e.g. Cohere's plain 40-char keys) rely on
    # the generic key=/token= rule above.
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
]


# A client resends the whole conversation on every request, so without a
# memo every earlier turn is scanned again each time: measured on a real
# 51-request agent session, about 210 ms per request, on the event loop.
# The memo is keyed on a digest, and a text with nothing to redact (almost
# all of them) is stored as None, so an entry costs a few dozen bytes
# whatever the size of the text. A redacted text keeps its redacted copy
# unless it is larger than _MEMO_MAX_CHARS, in which case it is recomputed.
# `key = value` / `key: value` credentials. The value is replaced and the
# key kept: redacting the whole match turned `api_key: Optional[str]` into
# `[REDACTED][str]` and `password = os.environ["X"]` into
# `[REDACTED]["X"]` — ordinary code, rewritten before the model saw it.
# A value is treated as a secret when it is quoted, or when it is a bare
# token that is not code: not a dotted reference, not a call or subscript,
# not a type or identifier made only of letters.
_KEY_VALUE = re.compile(
    r'(?P<key>(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key'
    r'|private[_\-]?key|auth[_\-]?key|client[_\-]?secret)\s*[=:]\s*)'
    r'(?:(?P<q>["\'])(?P<qval>[^"\'\s]{8,})(?P=q)'
    r'|(?P<val>[\w\-./+]{8,})(?P<after>[(\[]?))',
    re.IGNORECASE,
)


def _key_value_sub(m: "re.Match") -> str:
    if m.group("qval") is not None:
        return f"{m.group('key')}{m.group('q')}[REDACTED]{m.group('q')}"
    val, after = m.group("val"), m.group("after")
    looks_like_code = (
        after                                   # a call or a subscript
        or "." in val and not any(c.isdigit() for c in val.split(".")[-1])
        or val.isalpha()                        # a name or a type: Optional
        or re.fullmatch(r"[a-z_]+", val) is not None
    )
    if looks_like_code:
        return m.group(0)
    return f"{m.group('key')}[REDACTED]"


# Email addresses. Kept out of _PATTERNS: an address is personal data to
# keep out of stored memory, but it is also the content of countless
# legitimate requests ("send the report to alice@acme.com"), so the copy
# sent to the provider keeps it unless the operator asks otherwise. The
# lookahead leaves `git@github.com:org/repo` alone.
_EMAIL = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b(?!:)')

_MEMO_MAX_ENTRIES = 32_768
_MEMO_MAX_CHARS = 64_000
_memo: OrderedDict[bytes, str | None] = OrderedDict()
_memo_lock = threading.Lock()


def _memo_clear() -> None:
    with _memo_lock:
        _memo.clear()


def _scrub(text: str, emails: bool = True) -> str:
    for pat in _PATTERNS:
        text = pat.sub("[REDACTED]", text)
    text = _KEY_VALUE.sub(_key_value_sub, text)
    if emails:
        text = _EMAIL.sub("[REDACTED]", text)
    return text


def redact(text: str, emails: bool = True) -> str:
    """Replace all detected secrets with [REDACTED]. Non-string input is
    passed through as the empty string rather than raising, since callers
    (graph nodes, message content) may legitimately have None/empty values.

    `emails=False` leaves email addresses in place (see _EMAIL)."""
    if not isinstance(text, str):
        return ""
    if not text:
        return text
    key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=32,
                          person=b"e1" if emails else b"e0").digest()
    with _memo_lock:
        if key in _memo:
            _memo.move_to_end(key)
            hit = _memo[key]
            return text if hit is None else hit
    out = _scrub(text, emails)
    if out == text:
        value = None
    elif len(out) <= _MEMO_MAX_CHARS:
        value = out
    else:
        return out
    with _memo_lock:
        _memo[key] = value
        while len(_memo) > _MEMO_MAX_ENTRIES:
            _memo.popitem(last=False)
    return out


def redact_node(label: str, summary: str = "") -> tuple[str, str]:
    return redact(label), redact(summary)


def _redact_values(value, emails: bool = True):
    """Redact string values recursively without changing keys or types."""
    if isinstance(value, str):
        return redact(value, emails)
    if isinstance(value, list):
        return [_redact_values(item, emails) for item in value]
    if isinstance(value, dict):
        return {key: _redact_values(item, emails) for key, item in value.items()}
    return value


def _redact_arguments(arguments, emails: bool = True):
    """Scrub OpenAI tool arguments while preserving valid JSON."""
    if not isinstance(arguments, str):
        return _redact_values(arguments, emails)
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return redact(arguments, emails)
    cleaned = _redact_values(parsed, emails)
    if cleaned == parsed:
        return arguments
    return json.dumps(cleaned, separators=(",", ":"))


def _redact_tool_calls(tool_calls, emails: bool = True):
    if not isinstance(tool_calls, list):
        return tool_calls
    cleaned = []
    for call in tool_calls:
        if not isinstance(call, dict):
            cleaned.append(call)
            continue
        function = call.get("function")
        if not isinstance(function, dict) or "arguments" not in function:
            cleaned.append(call)
            continue
        cleaned.append({
            **call,
            "function": {
                **function,
                "arguments": _redact_arguments(function["arguments"], emails),
            },
        })
    return cleaned


# A Gemini part carries exactly one of these keys, and the field inside it
# that holds model- or tool-produced content: the call's arguments, a tool's
# output, code the model wrote, and the output of running it. Everything
# else in the part (name, id, outcome, signatures) is metadata, left as sent.
_GEMINI_PAYLOAD_FIELDS = {
    "functionCall": "args",
    "functionResponse": "response",
    "executableCode": "code",
    "codeExecutionResult": "output",
}


def _gemini_payload_key(block: dict):
    """The Gemini part key whose payload this block carries, or None."""
    for part, field in _GEMINI_PAYLOAD_FIELDS.items():
        if isinstance(block.get(part), dict) and field in block[part]:
            return part
    return None


def _redact_content(content, emails: bool = True):
    """
    Redact secrets from message content of any shape.

    `content` must not be assumed to always be a `str` calling
    `redact()` directly on it. Two real failure modes existed:
      1. content=None (common for tool-call-only messages) → re.sub on
         None raises TypeError, which would have bubbled up as a 500 on
         a perfectly normal multi-turn tool-use conversation.
      2. content=list (multimodal: text + image/document blocks, per the
         Anthropic/OpenAI message schema) → redact() would never see the
         text, OR (depending on caller) the whole list would get cast to
         str(), embedding raw, unredacted secrets inside a stringified
         repr that nothing downstream expected.

    Fix: text and tool-input values are redacted in place; image and
    document blocks pass through unchanged. We must never regex-redact
    binary/base64 image data, both because it is not text and because doing
    so would corrupt the image.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return redact(content, emails)
    if isinstance(content, list):
        cleaned = []
        for block in content:
            if isinstance(block, str):
                cleaned.append(redact(block, emails))
            elif isinstance(block, dict):
                if "text" in block and (
                    block.get("type") == "text"
                    or (block.get("type") is None and isinstance(block["text"], str))
                ):
                    # A Gemini part is {"text": ...} with no "type"; the
                    # extractor reads it as text, so it is redacted as text.
                    cleaned.append({**block, "text": redact(str(block["text"]), emails)})
                elif block.get("type") == "tool_use" and "input" in block:
                    cleaned.append({**block, "input": _redact_values(block["input"], emails)})
                elif _gemini_payload_key(block) is not None:
                    part = _gemini_payload_key(block)
                    field = _GEMINI_PAYLOAD_FIELDS[part]
                    cleaned.append({
                        **block,
                        part: {**block[part], field: _redact_values(block[part][field], emails)},
                    })
                elif "content" in block and isinstance(block.get("content"), (str, list)):
                    # tool_result content is `str | list[block]` per the
                    # Anthropic/OpenAI schema — a tool returning structured
                    # output (e.g. a file-read result) commonly shapes it as
                    # [{"type": "text", "text": ...}], not a bare string. The
                    # str-only check this replaced left that list branch
                    # matching neither this arm nor "leave untouched" for a
                    # good reason, silently passing a secret embedded in a
                    # tool result straight through to the graph, checkpoints,
                    # and the background extraction LLM. Recursing through
                    # _redact_content (which already handles both shapes)
                    # covers it the same way the top-level list does.
                    cleaned.append({**block, "content": _redact_content(block["content"], emails)})
                else:
                    # image/document blocks — leave untouched
                    cleaned.append(block)
            else:
                cleaned.append(block)
        return cleaned
    if isinstance(content, dict):
        if "text" in content:
            return {**content, "text": redact(str(content["text"]), emails)}
        return content
    return content


def redact_messages(messages: list[dict], emails: bool = True) -> list[dict]:
    """Return a copy with secrets scrubbed from content and tool inputs."""
    cleaned = []
    for message in messages:
        item = {**message, "content": _redact_content(message.get("content"), emails)}
        if "tool_calls" in message:
            item["tool_calls"] = _redact_tool_calls(message["tool_calls"], emails)
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and "arguments" in function_call:
            item["function_call"] = {
                **function_call,
                "arguments": _redact_arguments(function_call["arguments"], emails),
            }
        cleaned.append(item)
    return cleaned
