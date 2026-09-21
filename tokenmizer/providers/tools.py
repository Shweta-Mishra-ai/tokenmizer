"""
Tool/function-calling translation between the OpenAI wire shape the proxy
speaks and each provider's native one.

The proxy's contract is the OpenAI chat-completions schema: a request
carries `tools` (function declarations) and `tool_choice`; the assistant
answers with `message.tool_calls`; the client runs the tools and sends the
results back as `role: "tool"` messages, each naming the `tool_call_id` it
answers. Agent frameworks (LangChain, CrewAI, AutoGen, the OpenAI Agents
SDK, Aider's function mode) all speak exactly this, so a proxy that drops
it silently breaks every one of them: the model answers as if it had no
tools, and the framework either loops or gives up.

OpenAI-compatible providers (OpenAI, DeepSeek, Mistral, OpenRouter, Grok)
take the shape as-is. Anthropic and Ollama need translation, kept here so
the adapters stay readable and the two directions are tested side by side.

Everything in this module is pure: no I/O, no SDK imports.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Request fields forwarded to a provider that supports tools. Anything
# else in the request body is either sampling (handled separately) or
# ignored, as before.
TOOL_KEYS = ("tools", "tool_choice", "parallel_tool_calls")


def tool_kwargs(kwargs: dict) -> dict:
    """The tool-related entries of a kwargs dict, if any."""
    return {k: kwargs[k] for k in TOOL_KEYS if kwargs.get(k) is not None}


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _get(obj: Any, key: str, default=None):
    """Attribute-or-key access, so SDK objects and plain dicts (test fakes,
    `model_dump()` output) go through the same code."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_tool_call(tc: Any) -> Optional[dict]:
    """One tool call in the OpenAI wire shape, from an SDK object or dict.

        {"id": ..., "type": "function",
         "function": {"name": ..., "arguments": "<json string>"}}

    `arguments` is always a JSON *string* on the wire, even when the
    provider handed us a dict (Ollama does).
    """
    fn = _get(tc, "function")
    if fn is None:
        return None
    name = _get(fn, "name") or ""
    args = _get(fn, "arguments")
    if args is None:
        args = "{}"
    elif not isinstance(args, str):
        args = json.dumps(args)
    return {
        "id": _get(tc, "id") or new_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def normalize_tool_calls(tool_calls: Any) -> list[dict]:
    if not tool_calls:
        return []
    out = []
    for tc in tool_calls:
        n = normalize_tool_call(tc)
        if n is not None:
            out.append(n)
    return out


def parse_arguments(arguments: Any) -> dict:
    """JSON-decode a tool call's arguments. A model that emits malformed
    JSON still gets its call forwarded (the client asked for the raw
    string and gets it unchanged on the OpenAI side); for providers that
    need a dict, the text is kept under `_raw` rather than dropped."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        data = json.loads(arguments)
    except (TypeError, ValueError):
        return {"_raw": str(arguments)}
    return data if isinstance(data, dict) else {"_raw": data}


# ── Anthropic ─────────────────────────────────────────────────────────────────

def anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function declarations -> Anthropic tool definitions."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def anthropic_tool_choice(tool_choice: Any, parallel_tool_calls: Optional[bool] = None
                          ) -> Optional[dict]:
    """OpenAI `tool_choice` -> Anthropic `tool_choice`, or None to omit.

    "none" maps to Anthropic's {"type": "none"} rather than dropping the
    tool definitions: a conversation that already contains tool_use
    blocks is rejected without the definitions that explain them.
    """
    choice: Optional[dict] = None
    if tool_choice is None or tool_choice == "auto":
        choice = {"type": "auto"} if parallel_tool_calls is False else None
    elif tool_choice == "required":
        choice = {"type": "any"}
    elif tool_choice == "none":
        choice = {"type": "none"}
    elif isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if name:
            choice = {"type": "tool", "name": name}
    if choice is not None and parallel_tool_calls is False and choice["type"] != "none":
        choice["disable_parallel_tool_use"] = True
    return choice


def anthropic_messages(conv: list[dict]) -> list[dict]:
    """OpenAI-shaped conversation -> Anthropic messages.

    - an assistant turn with `tool_calls` becomes text + tool_use blocks
    - a `role: "tool"` turn becomes a tool_result block inside a user
      message; consecutive tool turns share one user message, which is
      what Anthropic requires for the results of one tool_use batch

    System messages must already be stripped (see conversation_messages).
    """
    out: list[dict] = []
    for m in conv:
        role = m.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or "",
                "content": m.get("content") if m.get("content") is not None else "",
            }
            if out and out[-1]["role"] == "user" and out[-1].get("_tool_results"):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block], "_tool_results": True})
            continue

        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict] = []
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
            elif isinstance(text, list):
                blocks.extend(b for b in text if isinstance(b, dict))
            for tc in normalize_tool_calls(m["tool_calls"]):
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": parse_arguments(tc["function"]["arguments"]),
                })
            out.append({"role": "assistant", "content": blocks})
            continue

        out.append({"role": role, "content": m.get("content")})

    for m in out:
        m.pop("_tool_results", None)
    return out


def anthropic_response(content_blocks: Any) -> tuple[str, list[dict]]:
    """Anthropic response content -> (text, OpenAI-shaped tool_calls)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content_blocks or []:
        btype = _get(block, "type")
        if btype == "text":
            text_parts.append(_get(block, "text") or "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": _get(block, "id") or new_call_id(),
                "type": "function",
                "function": {
                    "name": _get(block, "name") or "",
                    "arguments": json.dumps(_get(block, "input") or {}),
                },
            })
    return "".join(text_parts), tool_calls


# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_messages(messages: list[dict]) -> list[dict]:
    """Ollama takes the OpenAI shape except that `function.arguments` is a
    dict, not a JSON string."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in normalize_tool_calls(m["tool_calls"]):
                calls.append({"function": {
                    "name": tc["function"]["name"],
                    "arguments": parse_arguments(tc["function"]["arguments"]),
                }})
            out.append({**m, "content": m.get("content") or "", "tool_calls": calls})
        else:
            out.append(m)
    return out


def ollama_tool_calls(tool_calls: Any) -> list[dict]:
    """Ollama's response tool calls (dict arguments, no ids) -> OpenAI shape."""
    return normalize_tool_calls(tool_calls)


# ── Cohere (v2) ───────────────────────────────────────────────────────────────
#
# Cohere v2 already speaks the OpenAI shape for tools: the declarations,
# the assistant's `tool_calls`, and a `role: "tool"` turn carrying a
# `tool_call_id` are all as-is. What differs is the streamed event names
# and that a tool result's content is a list of parts.

def cohere_messages(messages: list[dict]) -> list[dict]:
    """The conversation as Cohere v2 messages."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "",
                "content": [{"type": "document",
                             "document": {"data": m.get("content") or ""}}],
            })
            continue
        if m.get("tool_calls"):
            out.append({
                "role": "assistant",
                # v2 rejects a null content on a tool-call turn; "" is
                # what every other adapter here sends for the same shape.
                "content": m.get("content") or "",
                "tool_calls": normalize_tool_calls(m["tool_calls"]),
            })
            continue
        out.append(m)
    return out


def cohere_stream_event(event: Any) -> Optional[dict]:
    """One Cohere v2 stream event -> a text chunk, a tool-call delta, or
    None for the events that carry no payload (block start/end, message
    lifecycle).

    Returns `{"text": ...}` or `{"tool_calls": [...]}` so the caller does
    not have to know the event vocabulary.
    """
    kind = _get(event, "type") or ""
    delta = _get(event, "delta")
    message = _get(delta, "message") if delta is not None else None

    if kind == "content-delta":
        content = _get(message, "content")
        text = _get(content, "text") if content is not None else None
        return {"text": text} if text else None

    if kind in ("tool-call-start", "tool-call-delta"):
        calls = _get(message, "tool_calls")
        if calls is None:
            return None
        # v2 sends one call per event, indexed by the event's own `index`.
        call = calls[0] if isinstance(calls, list) else calls
        fn = _get(call, "function")
        out: dict = {"index": _get(event, "index") or 0}
        if kind == "tool-call-start":
            out["id"] = _get(call, "id") or new_call_id()
            out["type"] = "function"
        f: dict = {}
        name = _get(fn, "name") if fn is not None else None
        if name:
            f["name"] = name
        args = _get(fn, "arguments") if fn is not None else None
        if args is not None:
            f["arguments"] = args if isinstance(args, str) else json.dumps(args)
        if kind == "tool-call-start":
            f.setdefault("arguments", "")
        out["function"] = f
        return {"tool_calls": [out]}

    return None


# ── Gemini ────────────────────────────────────────────────────────────────────
#
# google-genai calls a declaration a FunctionDeclaration, groups them under
# a Tool, and answers with a function_call part inside the candidate's
# content. The schema is JSON Schema with a different spelling of the keys,
# so the parameters object passes through unchanged.

def gemini_tools(tools: list[dict]) -> list[dict]:
    """OpenAI `tools` -> the `tools` list google-genai's config takes."""
    declarations = []
    for t in tools or []:
        fn = t.get("function") or {}
        if not fn.get("name"):
            continue
        declarations.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return [{"function_declarations": declarations}] if declarations else []


def gemini_tool_config(tool_choice: Any) -> Optional[dict]:
    """OpenAI `tool_choice` -> Gemini's function_calling_config.

    "auto" is Gemini's own default, so it is expressed as None rather than
    as an explicit AUTO — sending a config where none is needed is a
    difference between providers for no reason.
    """
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"function_calling_config": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return {"function_calling_config": {
                "mode": "ANY", "allowed_function_names": [name]}}
    return None


def gemini_response(candidate: Any) -> tuple[str, list[dict]]:
    """A Gemini candidate -> (text, tool_calls in the OpenAI shape).

    Gemini has no id for a call; the client needs one to match the result
    it sends back, so one is minted here — the same thing the Anthropic
    path does not have to do because Anthropic supplies one.
    """
    content = _get(candidate, "content")
    parts = _get(content, "parts") or []
    text_parts: list[str] = []
    calls: list[dict] = []
    for part in parts:
        text = _get(part, "text")
        if text:
            text_parts.append(text)
        fc = _get(part, "function_call")
        if fc is None:
            continue
        name = _get(fc, "name")
        if not name:
            continue
        args = _get(fc, "args")
        calls.append({
            "id": new_call_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args if isinstance(args, dict) else {}),
            },
        })
    return "".join(text_parts), calls


def gemini_contents(conv: list[dict]) -> list[dict]:
    """The conversation as google-genai `contents`, tool turns included.

    An assistant turn that asked for tools becomes function_call parts; the
    client's `role: "tool"` results become function_response parts on a
    user turn, which is where Gemini expects them. Without this a second
    agent-loop turn sent Gemini a conversation with the call and its result
    missing, and the model asked for the same tool again.
    """
    out: list[dict] = []
    for m in conv:
        role = m.get("role")
        if role == "tool":
            # Gemini matches a response to a call by NAME, not by id, and
            # the OpenAI shape only carries the id — so the name is looked
            # up from the assistant turn that requested it, above.
            name = m.get("name") or _tool_name_for(conv, m.get("tool_call_id"))
            out.append({"role": "user", "parts": [{"function_response": {
                "name": name or "tool",
                "response": {"result": m.get("content") or ""},
            }}]})
            continue
        parts: list[dict] = []
        if m.get("content"):
            parts.append({"text": m["content"]})
        for tc in normalize_tool_calls(m.get("tool_calls")):
            parts.append({"function_call": {
                "name": tc["function"]["name"],
                "args": parse_arguments(tc["function"]["arguments"]),
            }})
        if not parts:
            continue
        out.append({"role": "user" if role == "user" else "model", "parts": parts})
    return out


def _tool_name_for(conv: list[dict], call_id: Optional[str]) -> Optional[str]:
    if not call_id:
        return None
    for m in conv:
        for tc in normalize_tool_calls(m.get("tool_calls")):
            if tc["id"] == call_id:
                return tc["function"]["name"]
    return None


# ── Shared response helpers ───────────────────────────────────────────────────

def finish_reason(native: Optional[str], tool_calls: list[dict]) -> str:
    """The OpenAI finish_reason for a response: "tool_calls" whenever the
    model asked for a tool, whatever the provider called it."""
    if tool_calls:
        return "tool_calls"
    if native in ("length", "max_tokens"):
        return "length"
    return "stop"


def strip_tool_fields(messages: list[dict]) -> list[dict]:
    """Messages with tool_calls/tool_call_id removed and tool turns
    rendered as plain text. For consumers that read a conversation as
    prose (graph extraction, the semantic cache's fingerprint): a tool
    result is still content worth remembering, a call id is not."""
    out = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "tool", "content": m.get("content") or ""})
            continue
        if m.get("tool_calls"):
            calls = "; ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in normalize_tool_calls(m["tool_calls"])
            )
            text = m.get("content") or ""
            out.append({"role": m.get("role", "assistant"),
                        "content": (text + "\n" if text else "") + f"[called tools: {calls}]"})
            continue
        out.append({"role": m.get("role"), "content": m.get("content")})
    return out
