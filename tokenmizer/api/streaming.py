"""
Server-sent-events passthrough for `stream: true`.

Split out of api/app.py. The function reaches the app's singletons and
patchable helpers as `app_module.<name>`, looked up at call time, the same
way routes_graph.py does — so a test that monkeypatches app_module (the
provider, the cache, analytics, settings) affects this path exactly as it
did when the code lived in app.py. app.py imports this module at the
bottom, after every name it refers to exists.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from tokenmizer.api import app as app_module
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens

logger = logging.getLogger("tokenmizer.api.app")


def _stream_response(req, messages, model, user_content, session_id,
                     savings, orig_input_tokens, raw_messages=None) -> StreamingResponse:
    """True SSE passthrough (OpenAI chat.completion.chunk format).

    Cache hits stream as a single chunk. After the stream closes, analytics
    and the semantic-cache write run on the accumulated text — same
    bookkeeping as the non-stream path, minus output trimming.

    Transform-rejected fallback, matching _call_provider(): if the
    transformed `messages` fail before any content has reached the
    client, retry once with `raw_messages`. Once content has streamed,
    a failure can never retry — the client already has an unrecoverable
    partial answer, and resending would duplicate or interleave output —
    so it is always the existing mid-stream error event.
    """
    import json as _json

    from tokenmizer.providers.providers import BaseProvider, ProviderError

    provider = app_module._get_provider()
    if getattr(type(provider), "chat_stream", None) is BaseProvider.chat_stream:
        raise HTTPException(
            status_code=501,
            detail=(f"Streaming passthrough is not implemented for provider "
                    f"'{app_module.settings.provider}' yet (supported: anthropic, openai, "
                    f"deepseek, mistral, openrouter, grok, ollama). "
                    f"Set stream=false for this provider."),
        )

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish: str | None = None, extra: dict | None = None) -> str:
        payload = {
            "id": resp_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "session_id": session_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            payload.update(extra)
        return "data: " + _json.dumps(payload) + "\n\n"

    cache_ctx = app_module._cache_context(req, model, raw_messages or [])

    tool_kw = app_module._tool_kwargs(req)
    # An adapter that supports tools but cannot stream tool-call deltas
    # answers the request from chat() and the answer is emitted as chunks:
    # the client still gets a valid SSE stream, just not an incremental one.
    buffered_tools = bool(tool_kw) and not getattr(provider, "supports_tool_stream", False)

    async def _buffered_tool_stream(candidate):
        resp = await provider.chat(
            messages=candidate, model=model, max_tokens=app_module._max_tokens(req),
            **app_module._sampling_kwargs(req), **tool_kw,
        )
        if resp.text:
            yield resp.text
        if resp.tool_calls:
            yield {"tool_calls": [{**tc, "index": i} for i, tc in enumerate(resp.tool_calls)]}
        if resp.finish_reason in ("length", "max_tokens"):
            yield {"finish_reason": "length"}

    async def _gen():
        full_text = ""
        saw_tool_calls = False
        truncated = False
        t0 = time.monotonic()
        cache_hit = False
        stream_failed = False
        transform_rejected = False
        yield _chunk({"role": "assistant"})
        try:
            cached = (
                app_module._cache.get(user_content, session_id=session_id, context=cache_ctx)
                if app_module.settings.cache.enabled and user_content and not tool_kw
                else None
            )

            cache_hit = cached is not None

            if cached:
                full_text = cached.response
                yield _chunk({"content": full_text})
            else:
                candidates = [messages]
                if raw_messages is not None and raw_messages != messages:
                    candidates.append(raw_messages)

                last_error: Exception | None = None
                succeeded = False
                for attempt, candidate in enumerate(candidates):
                    last_error = None
                    try:
                        pieces = (
                            _buffered_tool_stream(candidate) if buffered_tools
                            else provider.chat_stream(
                                messages=candidate,
                                model=model,
                                max_tokens=app_module._max_tokens(req),
                                **app_module._sampling_kwargs(req), **tool_kw,
                            )
                        )
                        async for piece in pieces:
                            if isinstance(piece, dict) and "finish_reason" in piece:
                                truncated = piece["finish_reason"] == "length"
                                continue
                            if isinstance(piece, dict):
                                # A tool-call delta (see OpenAIProvider.
                                # chat_stream): re-emitted in the same
                                # OpenAI chunk shape.
                                saw_tool_calls = True
                                yield _chunk({"tool_calls": piece["tool_calls"]})
                                continue
                            full_text += piece
                            yield _chunk({"content": piece})
                        succeeded = True
                        transform_rejected = attempt > 0
                        break
                    except Exception as e:
                        last_error = e
                        # Content already reached the client — an
                        # unrecoverable partial answer is out the door, so
                        # retrying now would duplicate or interleave
                        # output. This failure is final regardless of
                        # attempts left.
                        if full_text or saw_tool_calls:
                            break
                        if attempt == 0 and len(candidates) > 1 and app_module._worth_untransformed_retry(e):
                            correlation_id = uuid.uuid4().hex[:12]
                            logger.warning(
                                f"Provider rejected the transformed stream "
                                f"request [{correlation_id}] for session "
                                f"{session_id!r}; retrying with the "
                                f"untransformed messages: {type(e).__name__}"
                            )
                            continue
                        break

                if not succeeded:
                    raise last_error
                if transform_rejected:
                    logger.error(
                        f"TokenMizer's transformed stream request was "
                        f"rejected but the untransformed one succeeded for "
                        f"session {session_id!r} — a pipeline layer "
                        f"produced a request the provider will not accept. "
                        f"Original error: {last_error}"
                    )
                    app_module._analytics.record_silent_failure("transform_rejected")
                    savings.clear()

        except ProviderError as e:
            # Mid-stream failure: SSE can't change the status code anymore —
            # emit an explicit error event instead of silently truncating.
            # Opaque, like the non-stream path: str(e) is the SDK's own
            # message, with request ids and upstream URLs in it.
            stream_failed = True
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(f"Provider stream error [{correlation_id}]: {e}")
            kind = "rate_limit_error" if e.error_type == "rate_limit" else "provider_error"
            yield "data: " + _json.dumps(
                {"error": {"message": app_module._provider_failure_detail(correlation_id),
                           "type": kind}}
            ) + "\n\n"
        except Exception as e:
            # Anything the provider layer didn't wrap in ProviderError
            # (transport errors, decode errors, bugs) would otherwise
            # escape the generator: the SSE stream dies with no
            # terminator and the bookkeeping below never runs.
            stream_failed = True
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(f"Stream error [{correlation_id}]: {e}")
            yield "data: " + _json.dumps(
                {"error": {"message": f"Stream failed (ref: {correlation_id})",
                           "type": "internal_error"}}
            ) + "\n\n"
        yield _chunk(
            {}, finish=("tool_calls" if saw_tool_calls
                        else "length" if truncated else "stop"),
            extra={"tokenmizer": {"fallback": {"transform_rejected": True}}}
            if transform_rejected else None,
        )
        yield "data: [DONE]\n\n"

        # Post-stream bookkeeping. When the transformed request was
        # rejected, what was actually SENT is raw_messages — mirrors
        # _call_provider()'s `sent_input_tokens = orig_input_tokens`.
        latency_ms = (time.monotonic() - t0) * 1000
        output_tokens = 0 if cache_hit else count_tokens(full_text, model)
        input_tokens = (
            0 if cache_hit
            else orig_input_tokens if transform_rejected
            else count_messages_tokens(messages, model)
        )
        # Never cache a response that didn't finish. `full_text` after a
        # mid-stream failure holds however many tokens arrived before the
        # error — writing that to the cache would serve a silently
        # truncated answer to every future matching prompt, long after the
        # provider recovered. Re-writing a cache HIT is equally pointless.
        if (app_module.settings.cache.enabled and user_content and full_text and not truncated
                and not stream_failed and not cache_hit and not saw_tool_calls):
            app_module._cache.set(user_content, full_text, input_tokens=input_tokens,
                       output_tokens=output_tokens, session_id=session_id,
                       context=cache_ctx)
        app_module._analytics.record(
            session_id=session_id, provider=app_module.settings.provider, model=model,
            input_tokens_original=orig_input_tokens,
            input_tokens_sent=input_tokens, output_tokens=output_tokens,
            tokens_saved=orig_input_tokens - input_tokens, latency_ms=latency_ms,
            cache_hit=cache_hit, layer_savings=app_module._net_layers(savings, orig_input_tokens, input_tokens),
        )

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
