"""
Pure helpers of the chat request path: context-window sizes, the retry
decision for a rejected request, savings arithmetic, session-id selection
and content checks.

Split out of api/app.py, which re-exports every name. None of these reads
the app's singletons, so moving them changes nothing a test can patch: the
call sites in app.py still look each name up in app.py's own namespace.
"""
from __future__ import annotations

import uuid

from fastapi import HTTPException, Request

from tokenmizer.api.schemas import ChatRequest

# Context window per model family, in tokens. Only feeds the auto-checkpoint
# trigger (context_pct = tokens sent / window), so an entry that is too
# SMALL checkpoints early (harmless) and one that is too LARGE never
# checkpoints (the failure this table exists to prevent) — when unsure,
# prefer the smaller published figure. Longest matching key wins, so a
# specific entry beats its family's catch-all. Newest Claude models
# (fable-5, opus-4-8, sonnet-5, haiku-4-5) all match the "claude" entry;
# add a specific entry ONLY if a model's window differs.
_CONTEXT_WINDOWS = {
    "claude-fable-5": 200_000, "claude-opus-4-8": 200_000,
    "claude-sonnet": 200_000, "claude-opus": 200_000, "claude-haiku": 200_000,
    "claude": 200_000,
    # OpenAI: the 4.1 family and the 5 series are far larger than 4o; the
    # reasoning models sit at 200k. Longest-key matching keeps "gpt-4o"
    # from being shadowed by "gpt-4", and "gpt-4.1" from matching "gpt-4".
    "gpt-4.1": 1_000_000, "gpt-4o": 128_000, "gpt-4": 128_000, "gpt-3.5": 16_000,
    "gpt-5": 400_000,
    "o1": 200_000, "o3": 200_000, "o4": 200_000,
    "gemini": 1_000_000,
    "deepseek": 128_000,
    "mistral": 128_000, "codestral": 256_000,
    "grok": 128_000,
    "command-r": 128_000,
    # Local models vary by build; 32k is the common default `num_ctx`
    # ceiling, and a too-small figure only checkpoints early.
    "llama": 32_000, "qwen": 32_000, "mixtral": 32_000, "phi": 16_000,
}


def _context_window(model: str) -> int:
    # Longest key first — so "claude-fable-5" wins over the "claude" catch-all
    # if their values ever diverge. (Previously dict order decided; the broad
    # "claude" key shadowed every specific entry.)
    m = model.lower()
    for k in sorted(_CONTEXT_WINDOWS, key=len, reverse=True):
        if k in m:
            return _CONTEXT_WINDOWS[k]
    return 128_000


def _worth_untransformed_retry(exc: Exception) -> bool:
    """Should a failure on the transformed request be retried untransformed?

    Only when the provider REJECTED the request as malformed (400, 413,
    422), which is the case the retry exists for: a pipeline layer built
    something the provider will not accept. Every failure used to be
    retried, so an upstream rate limit, outage or bad key cost a second,
    larger request (each with the adapter's own backoff) before the client
    was told anything.
    """
    seen = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        status = getattr(e, "status_code", None)
        if status is None:
            status = getattr(getattr(e, "response", None), "status_code", None)
        code = getattr(e, "code", None)
        if status is None and isinstance(code, int):
            status = code
        if isinstance(status, int):
            return status in (400, 413, 422)
        e = e.__cause__ or e.__context__
    from tokenmizer.core.errors import ProviderError
    if isinstance(exc, ProviderError):
        # No status to go on. A retryable error is transient by the
        # adapter's own judgement; a rate limit, a dropped connection or a
        # credential problem would fail the same way untransformed.
        if exc.retryable or exc.error_type in ("rate_limit", "connection"):
            return False
        return not exc.error_type.startswith(("http_401", "http_403", "http_404",
                                              "http_429", "http_5"))
    return True


# OpenAI request fields that change the answer and that OpenAI-compatible
# servers accept natively. They were accepted (extra="allow") and dropped,
# so `response_format: json_object` produced prose and nothing said why.
_PASSTHROUGH_FIELDS = ("response_format", "seed", "frequency_penalty",
                       "presence_penalty", "logit_bias", "user",
                       "reasoning_effort")


# Providers whose adapter forwards _PASSTHROUGH_FIELDS untouched.
_PASSTHROUGH_PROVIDERS = frozenset({"openai", "gpt", "deepseek", "mistral",
                                    "grok", "openrouter"})


def _net_layers(savings: dict, original: int, sent: int) -> dict:
    """Per-layer savings plus the proxy's own additions as a negative line,
    so the layers add up to what was really saved: original - sent.

    The old total summed only what each layer removed and never subtracted
    what the proxy added — the terse instruction, injected memory,
    preferences, the windowing bridge. A request that went out four times
    the size it came in was reported as a saving.
    """
    layers = {k: v for k, v in savings.items() if v}
    removed = sum(layers.values())
    overhead = (original - sent) - removed
    if overhead:
        layers["added_by_proxy"] = overhead
    return layers


def _session_id_for(req: ChatRequest, request: Request) -> str:
    """Body `session_id`, else the X-Session-ID header, else a new id.

    The header was in the CORS allow-list and documented as a way to name
    a session, and nothing read it. A standard OpenAI SDK cannot add a body
    field without `extra_body`, but every one of them can set a header —
    so for most clients every request started a brand-new session, memory
    never accumulated, and sessions.db gained a row per request.
    """
    if req.session_id:
        return req.session_id
    header = (request.headers.get("X-Session-ID") or "").strip()
    if header:
        if len(header) > 256:
            raise HTTPException(status_code=422, detail="X-Session-ID is longer than 256 characters.")
        return header
    return str(uuid.uuid4())


def _reject_unrepresentable_content(req: ChatRequest) -> None:
    """422 for content this text proxy would otherwise drop silently.

    Image, audio and file blocks were stripped to their text parts and the
    request answered with a 200: the model replied about a picture it was
    never shown. Refusing names the problem; sending the request straight
    to the provider is the way to use those modalities today.
    """
    for i, m in enumerate(req.messages):
        if not isinstance(m.content, list):
            continue
        for block in m.content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind not in (None, "text"):
                raise HTTPException(
                    status_code=422,
                    detail=(f"messages[{i}] carries a {kind!r} content block. "
                            f"TokenMizer proxies text only and would drop it, so the "
                            f"model would answer without it. Send this request to the "
                            f"provider directly, or remove the block."),
                )
