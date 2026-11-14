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

import json
import re

from tokenmizer.core.bounded_cache import MISSING, BoundedCache

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
]


# A client resends the whole conversation on every request, so without a
# memo every earlier turn is scanned again each time: measured on a real
# 51-request agent session, about 210 ms per request, on the event loop.
# The memo is keyed on a digest, and a text with nothing to redact (almost
# all of them) is stored as None, so an entry costs a few dozen bytes
# whatever the size of the text. A redacted text keeps its redacted copy
# unless it is larger than _MEMO_MAX_CHARS, in which case it is recomputed.
_MEMO_MAX_ENTRIES = 32_768
_MEMO_MAX_CHARS = 64_000
# Redacted copies held in all: without this, 32,768 entries of up to 64,000
# characters each could reach 2 GB on a long-running proxy.
_MEMO_MAX_TOTAL_CHARS = 8_000_000
_memo: BoundedCache[str | None] = BoundedCache(
    _MEMO_MAX_ENTRIES, _MEMO_MAX_TOTAL_CHARS, lambda v: len(v or ""))


def _memo_clear() -> None:
    _memo.clear()


_SECRET_WORD = (r'(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key'
                r'|private[_\-]?key|auth[_\-]?key|client[_\-]?secret)')

# Rules that keep the NAME and redact only the VALUE. Group 1 is what
# survives. The single rule these replace matched `<word> [=:] <8+ chars of
# anything identifier-like>` and replaced all of it, which had two failures
# at once:
#
#   - It rewrote ordinary code. `token = generate_token(user)` became
#     `[REDACTED](user)`, `api_key = settings.openai_api_key` became
#     `[REDACTED]`, `password: Optional[SecretStr]` became
#     `[REDACTED][SecretStr]`. The model was then asked to fix code it had
#     been shown a corrupted copy of, and an agent wrote the marker back
#     into the file.
#   - It missed the commonest way a secret is actually written down:
#     `{"password": "..."}`. The quote after the key sits between the word
#     and the colon, so the rule never matched JSON or a Python dict.
_VALUE_RULES = [
    # "password": "value"  /  'api_key': 'value'   (JSON, dict literals)
    re.compile(r'''(["']''' + _SECRET_WORD + r'''["']\s*:\s*)(["'])[^"'\n]{6,}\2''',
               re.IGNORECASE),
    # password = "value"   — a quoted literal on the right-hand side
    re.compile(r'(\b' + _SECRET_WORD + r'''\s*[=:]\s*)(["'])[^"'\n]{8,}\2''',
               re.IGNORECASE),
    # password=hunter2hunter2 — unquoted. It must contain a digit and must
    # not continue into a call, an attribute or a subscript: that is what
    # separates a literal from `get_token()`, `settings.key`, `os.environ[`.
    re.compile(r'(\b' + _SECRET_WORD + r'\s*[=:]\s*)'
               r'(?=[A-Za-z_\-/+]{0,64}\d)[\w\-/+]{8,}(?![\w.(\[])',
               re.IGNORECASE),
    # DB_PASS=..., AWS_SECRET_ACCESS_KEY=... — env-file style. An
    # UPPER_SNAKE name that says it is a secret, assigned a literal.
    re.compile(r'(\b[A-Z][A-Z0-9_]*(?:PASS(?:WORD|WD)?|SECRET|TOKEN|API_?KEY|'
               r'ACCESS_KEY|PRIVATE_KEY|CREDENTIALS?)[A-Z0-9_]*\s*=\s*)'
               r'''["']?[^\s"']{6,}["']?'''),
]

# A PEM private key, whole. No keyword and no prefix rule sees these.
_PEM = re.compile(
    r'-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----.{0,20000}?-----END [A-Z ]{0,40}PRIVATE KEY-----',
    re.DOTALL)

_EMAIL = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
# Local parts that name a transport account, not a person.
_SERVICE_ACCOUNTS = frozenset({"git", "hg", "svn"})
_CARD = re.compile(r'\b(?:\d[ -]?){13,19}\b')
_SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# What redaction does, set once at startup from `security.*` in settings by
# configure(). The defaults are what a bare import gets.
_config = {"secrets": True, "emails": True, "pii": False}


def configure(*, secrets: bool = True, emails: bool = True, pii: bool = False) -> None:
    """Choose what is redacted. Clears the memo, which holds results
    computed under the previous choice."""
    _config.update(secrets=bool(secrets), emails=bool(emails), pii=bool(pii))
    _memo_clear()


def _email_sub(m: re.Match) -> str:
    """Redact an address, but not the things that merely look like one.

    `git clone git@github.com:acme/widgets.git` became `git clone
    [REDACTED]:acme/widgets.git`, an scp target lost its host, and numpy's
    matrix operator (`weights@np.linalg.inv(cov)`) was read as a mailbox.
    An address followed by `:` `/` or `(` is a remote or an expression.
    """
    tail = m.string[m.end():m.end() + 1]
    if tail in (":", "/", "("):
        return m.group(0)
    if m.group(0).split("@", 1)[0].lower() in _SERVICE_ACCOUNTS:
        return m.group(0)
    return "[REDACTED]"


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _card_sub(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[REDACTED]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def _value_sub(m: re.Match) -> str:
    """Keep the name, redact the value, and keep the value's quotes so a
    JSON document or a dict literal is still one afterwards."""
    quote = m.group(2) if m.re.groups >= 2 else ""
    return f"{m.group(1)}{quote}[REDACTED]{quote}"


_SECRET_KEY_NAME = re.compile(r"^" + _SECRET_WORD + r"$", re.IGNORECASE)


def _scrub(text: str) -> str:
    if _config["secrets"]:
        text = _PEM.sub("[REDACTED PRIVATE KEY]", text)
        for pat in _PATTERNS:
            text = pat.sub("[REDACTED]", text)
        for pat in _VALUE_RULES:
            text = pat.sub(_value_sub, text)
    if _config["emails"]:
        text = _EMAIL.sub(_email_sub, text)
    if _config["pii"]:
        text = _CARD.sub(_card_sub, text)
        text = _SSN.sub("[REDACTED]", text)
    return text


def redact(text: str) -> str:
    """Replace all detected secrets with [REDACTED]. Non-string input is
    passed through as the empty string rather than raising, since callers
    (graph nodes, message content) may legitimately have None/empty values."""
    if not isinstance(text, str):
        return ""
    if not text:
        return text
    key = BoundedCache.key(text)
    hit = _memo.get(key)
    if hit is not MISSING:
        return text if hit is None else hit
    out = _scrub(text)
    if out == text:
        _memo.put(key, None)
    elif len(out) <= _MEMO_MAX_CHARS:
        _memo.put(key, out)
    return out


def redact_node(label: str, summary: str = "") -> tuple[str, str]:
    return redact(label), redact(summary)


def _redact_values(value):
    """Redact string values recursively without changing keys or types."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_values(item) for item in value]
    if isinstance(value, dict):
        # In parsed data the key is not in the string being scanned, so a
        # value filed under a secret's name is redacted on the key alone.
        return {
            key: ("[REDACTED]"
                  if (_config["secrets"] and isinstance(key, str)
                      and isinstance(item, str) and len(item) >= 6
                      and _SECRET_KEY_NAME.match(key))
                  else _redact_values(item))
            for key, item in value.items()
        }
    return value


def _redact_arguments(arguments):
    """Scrub OpenAI tool arguments while preserving valid JSON."""
    if not isinstance(arguments, str):
        return _redact_values(arguments)
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return redact(arguments)
    cleaned = _redact_values(parsed)
    if cleaned == parsed:
        return arguments
    return json.dumps(cleaned, separators=(",", ":"))


def _redact_tool_calls(tool_calls):
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
                "arguments": _redact_arguments(function["arguments"]),
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


def _redact_content(content):
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
        return redact(content)
    if isinstance(content, list):
        cleaned = []
        for block in content:
            if isinstance(block, str):
                cleaned.append(redact(block))
            elif isinstance(block, dict):
                if "text" in block and (
                    block.get("type") == "text"
                    or (block.get("type") is None and isinstance(block["text"], str))
                ):
                    # A Gemini part is {"text": ...} with no "type"; the
                    # extractor reads it as text, so it is redacted as text.
                    cleaned.append({**block, "text": redact(str(block["text"]))})
                elif block.get("type") == "tool_use" and "input" in block:
                    cleaned.append({**block, "input": _redact_values(block["input"])})
                elif _gemini_payload_key(block) is not None:
                    part = _gemini_payload_key(block)
                    field = _GEMINI_PAYLOAD_FIELDS[part]
                    cleaned.append({
                        **block,
                        part: {**block[part], field: _redact_values(block[part][field])},
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
                    cleaned.append({**block, "content": _redact_content(block["content"])})
                else:
                    # image/document blocks — leave untouched
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
    """Return a copy with secrets scrubbed from content and tool inputs."""
    cleaned = []
    for message in messages:
        item = {**message, "content": _redact_content(message.get("content"))}
        if "tool_calls" in message:
            item["tool_calls"] = _redact_tool_calls(message["tool_calls"])
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and "arguments" in function_call:
            item["function_call"] = {
                **function_call,
                "arguments": _redact_arguments(function_call["arguments"]),
            }
        cleaned.append(item)
    return cleaned
