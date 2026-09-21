"""
Provider adapters for all supported LLMs.

Invariants every adapter upholds:
- Gemini receives the full conversation history, not just the last message
- Native async throughout — no run_in_executor wrappers
- SDK imports are lazy, so no provider SDK is required at import time
- Failures are normalized to ProviderError, with `retryable` set honestly
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from tokenmizer.core.errors import ProviderError
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens
from tokenmizer.providers.tools import (
    anthropic_messages,
    anthropic_response,
    anthropic_tool_choice,
    anthropic_tools,
    cohere_messages,
    cohere_stream_event,
    finish_reason,
    gemini_contents,
    gemini_response,
    gemini_tool_config,
    gemini_tools,
    normalize_tool_calls,
    ollama_messages,
    ollama_tool_calls,
    tool_kwargs,
)

logger = logging.getLogger(__name__)

# Sampling params forwarded from the proxy request to every provider.
# Each provider maps these to its own SDK's naming.
_SAMPLING_KEYS = ("temperature", "top_p", "stop")


def _sampling(kwargs: dict) -> dict:
    """Extract only recognized sampling params from arbitrary kwargs."""
    return {k: kwargs[k] for k in _SAMPLING_KEYS if kwargs.get(k) is not None}


def conversation_messages(messages: list[dict]) -> list[dict]:
    """Strip system messages and return a conversation a strict API accepts.

    Anthropic and Gemini take the system prompt as a separate top-level
    parameter, so the conversation handed to them is `messages` minus every
    system entry. Both then require that conversation to BEGIN WITH A USER
    TURN — Anthropic returns 400 "first message must use the 'user' role",
    Gemini rejects a history whose first entry is a model turn.

    Nothing upstream guaranteed that. SmartMessageWindow keeps the last N
    conversation messages, and a chat request always ends on a user turn, so
    an even N (the default is 10) always sliced from an assistant turn. The
    result was a 400 on every turn of any session long enough to be windowed
    — i.e. exactly the sessions TokenMizer exists to serve. Windowing was
    fixed at its source, but the invariant belongs here too: this is the last
    point before the wire and it is shared by all three call sites, so no
    future layer can violate it silently.

    A leading assistant turn is dropped rather than repaired: it is the
    orphaned half of an exchange whose user turn was already windowed away,
    and its content is in the graph context block that replaced it.
    """
    conv = [m for m in messages if m.get("role") != "system"]
    first_user = next((i for i, m in enumerate(conv) if m.get("role") == "user"), None)
    if first_user is None:
        return []
    return conv[first_user:]


def _as_stop_list(stop) -> list[str]:
    return [stop] if isinstance(stop, str) else list(stop)


# Retryable-error detection for OpenAI-compatible providers.
#
# Do NOT reduce this to substring matching on the error text: "rate"
# appears inside "generate" and "moderate", so "Failed to generate
# completion" reads as retryable and costs the caller 1+2+4s of retries
# before the same permanent error comes back. The SDK's typed exceptions
# are authoritative; word-boundary matching is only the fallback for
# when the type is unavailable.
_RETRYABLE_TEXT = re.compile(
    r"\b(rate[ _-]?limit|too many requests|timed?[ _-]?out|timeout|"
    r"overloaded|unavailable|internal server error|bad gateway|"
    r"service unavailable)\b",
    re.IGNORECASE,
)


def _openai_error_is_retryable(exc: Exception) -> bool:
    """True if an OpenAI-compatible SDK error is worth retrying."""
    try:
        import openai
        if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError,
                            openai.APIConnectionError, openai.InternalServerError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in (408, 429, 500, 502, 503, 504)
        if isinstance(exc, openai.APIError):
            return False   # typed, and not one of the retryable kinds
    except Exception:
        pass  # SDK missing or shaped differently — fall through to text
    return bool(_RETRYABLE_TEXT.search(str(exc)))


def _ollama_error_is_retryable(exc: Exception) -> bool:
    """True if an Ollama request error is worth retrying.

    Every OllamaProvider error used to be marked retryable=True
    unconditionally — a malformed request or an unknown model name got
    the same 1+2+4s of pointless retries as a genuinely transient
    failure before the permanent error finally surfaced. Mirrors
    _openai_error_is_retryable's shape: typed check first (an HTTP
    status Ollama actually returned), then _RETRYABLE_TEXT as the text
    fallback — but a bare connection failure or client-side timeout
    reaching a LOCAL server is retried unconditionally, unlike the
    remote-API providers: those most often mean the server is still
    starting up, not that the request itself is bad.
    """
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (408, 429, 500, 502, 503, 504)
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
    except Exception:
        # httpx missing or shaped differently — fall through to matching
        # the message text, which is what this returns anyway for every
        # provider SDK that wraps its own errors.
        pass
    return bool(_RETRYABLE_TEXT.search(str(exc)))


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False
    # OpenAI-shaped tool calls the model asked for, empty for a plain
    # answer. See providers/tools.py for the shape and the translations.
    tool_calls: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseProvider(ABC):

    # Tool/function calling. `supports_tools` — the adapter forwards
    # `tools`/`tool_choice` and returns `tool_calls` (see providers/
    # tools.py). `supports_tool_stream` — chat_stream() can carry tool-call
    # deltas as dict events alongside text chunks; without it the proxy
    # answers a streamed tool request from chat() in one piece. The
    # proxy refuses a tool request up front (501) for an adapter without
    # `supports_tools`, rather than sending the model a conversation it
    # cannot see the tools for.
    supports_tools: bool = False
    supports_tool_stream: bool = False

    # How long an upstream call may hang before it is abandoned.
    #
    # Every vendor SDK here defaults to 600 seconds. On a proxy that is
    # not a timeout, it is an outage: a hung upstream holds the request,
    # the session lock, the extraction slot and the session's place in
    # the graph cache for ten minutes, and a handful of them is the whole
    # worker. 120s matches what the Ollama adapter already used, and is
    # comfortably longer than a slow long-form completion.
    #
    # Set from settings.request_timeout by build_provider() below; a
    # bare BaseProvider (as tests construct) gets this class default. Env
    # parsing/validation lives once in config/settings.py, same as every
    # other TOKENMIZER_* value — not read from os.environ here.
    request_timeout: float = 120.0

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key
        self.default_model = model
        self._retry_delays = [1.0, 2.0, 4.0]

    @property
    def _timeout_kwargs(self) -> dict:
        """`{"timeout": n}` for an SDK client, or `{}` to keep its own
        default. One place, so an adapter cannot be the one that forgot."""
        t = self.request_timeout
        return {"timeout": t} if t and t > 0 else {}

    @abstractmethod
    async def _call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        stream: bool,
        system: str,
        **kwargs,
    ) -> LLMResponse: ...

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 4096,
        stream: bool = False,
        system: str = "",
        **kwargs,
    ) -> LLMResponse:
        model = model or self.default_model
        t0 = time.monotonic()

        for attempt, delay in enumerate([0] + self._retry_delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await self._call(messages, model, max_tokens, stream, system, **kwargs)
                resp.latency_ms = (time.monotonic() - t0) * 1000
                return resp
            except ProviderError as e:
                if not e.retryable or attempt == len(self._retry_delays):
                    raise
                logger.warning(f"[{self.__class__.__name__}] retryable error (attempt {attempt+1}): {e}")
            except Exception as e:
                raise ProviderError(
                    provider=self.__class__.__name__,
                    error_type="unexpected",
                    message=str(e),
                    retryable=False,
                ) from e

        raise ProviderError(self.__class__.__name__, "max_retries", "All retry attempts exhausted", retryable=False)

    def chat_stream(self, messages: list[dict], model: str = "",
                    max_tokens: int = 4096, system: str = "", **kwargs):
        """Async generator yielding text chunks as the provider produces them.

        Providers that support true streaming override this. The base
        implementation raises so the API layer can return a clear 501 for
        providers where passthrough streaming isn't implemented yet, instead
        of silently degrading to a fake (buffered) stream.
        """
        raise ProviderError(
            self.__class__.__name__, "stream_not_supported",
            f"Streaming passthrough not implemented for {self.__class__.__name__}",
            retryable=False,
        )


# Anthropic will not cache a prefix shorter than a per-model minimum and
# silently ignores cache_control below it, so these thresholds must stay
# in TOKENS and at or above the real minimums. Setting them lower (an
# earlier version used 800 CHARACTERS) attaches cache_control to prompts
# that can never be cached, and prompt caching silently never engages.
_CACHE_MIN_TOKENS_DEFAULT = 1024
_CACHE_MIN_TOKENS_HAIKU = 2048


def _anthropic_system_param(system_text: str, model: str):
    """Build the `system` parameter, marking it cacheable only when it is
    actually long enough for Anthropic to cache.

    Returns a plain string when the prefix is too short (no point paying
    the structured-block overhead) and a single cache-controlled text
    block when it is long enough to earn the discount.
    """
    minimum = (_CACHE_MIN_TOKENS_HAIKU if "haiku" in (model or "").lower()
               else _CACHE_MIN_TOKENS_DEFAULT)
    if count_tokens(system_text, model) < minimum:
        return system_text
    return [{
        "type": "text",
        "text": system_text,
        "cache_control": {"type": "ephemeral"},
    }]


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicProvider(BaseProvider):

    supports_tools = True
    # text_stream carries only text, so the raw event stream is read
    # instead: content_block_start(tool_use) then input_json_delta. See
    # chat_stream.
    supports_tool_stream = True

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        super().__init__(api_key, model)

    @staticmethod
    def _tool_params(kwargs: dict) -> dict:
        """`tools` / `tool_choice` in Anthropic's shape, or nothing."""
        tk = tool_kwargs(kwargs)
        if not tk.get("tools"):
            return {}
        params: dict = {"tools": anthropic_tools(tk["tools"])}
        choice = anthropic_tool_choice(tk.get("tool_choice"), tk.get("parallel_tool_calls"))
        if choice is not None:
            params["tool_choice"] = choice
        return params

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self.api_key,
                                          **self._timeout_kwargs)

        # Separate system messages from conversation
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        conv = anthropic_messages(conversation_messages(messages))
        system_text = "\n\n".join(sys_parts) if sys_parts else None

        try:
            kwargs_clean = _sampling(kwargs)
            # Anthropic SDK uses `stop_sequences`, not OpenAI's `stop`
            if "stop" in kwargs_clean:
                kwargs_clean["stop_sequences"] = _as_stop_list(kwargs_clean.pop("stop"))
            if system_text:
                kwargs_clean["system"] = _anthropic_system_param(system_text, model)
            kwargs_clean.update(self._tool_params(kwargs))

            if stream:
                full_text = ""
                async with client.messages.stream(
                    model=model, messages=conv, max_tokens=max_tokens, **kwargs_clean
                ) as s:
                    async for chunk in s.text_stream:
                        full_text += chunk
                    # Real API-reported usage, not a local estimate — the
                    # estimate this replaced counted `conv` only (Anthropic
                    # takes system as a separate top-level param, stripped
                    # out of `conv` above) and never added the system
                    # prompt's tokens back in, silently undercounting
                    # input_tokens for every streamed call with a system
                    # prompt. The non-streaming branch below already uses
                    # resp.usage for the same reason.
                    final = await s.get_final_message()
                # text_stream carries text blocks only; tool_use blocks
                # are read off the final message.
                _, calls = anthropic_response(getattr(final, "content", None))
                return LLMResponse(text=full_text,
                                   input_tokens=final.usage.input_tokens,
                                   output_tokens=final.usage.output_tokens,
                                   model=model, provider="anthropic",
                                   finish_reason=(finish_reason(final.stop_reason, calls)
                                                  if calls else final.stop_reason or "stop"),
                                   tool_calls=calls)

            resp = await client.messages.create(
                model=model, messages=conv, max_tokens=max_tokens, **kwargs_clean
            )
            text, calls = anthropic_response(resp.content)
            return LLMResponse(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=model,
                provider="anthropic",
                finish_reason=(finish_reason(resp.stop_reason, calls)
                               if calls else resp.stop_reason or "stop"),
                tool_calls=calls,
            )
        except anthropic.RateLimitError as e:
            raise ProviderError("anthropic", "rate_limit", str(e), retryable=True, retry_after=60.0)
        except anthropic.APIStatusError as e:
            retryable = e.status_code in (500, 502, 503, 529)
            raise ProviderError("anthropic", f"http_{e.status_code}", str(e), retryable=retryable)

    async def chat_stream(self, messages: list[dict], model: str = "",
                          max_tokens: int = 4096, system: str = "", **kwargs):
        """True SSE passthrough — yields text chunks as Anthropic streams them."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        model = model or self.default_model
        client = anthropic.AsyncAnthropic(api_key=self.api_key,
                                          **self._timeout_kwargs)

        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        conv = anthropic_messages(conversation_messages(messages))
        kwargs_clean = _sampling(kwargs)
        if "stop" in kwargs_clean:
            kwargs_clean["stop_sequences"] = _as_stop_list(kwargs_clean.pop("stop"))
        kwargs_clean.update(self._tool_params(kwargs))
        if sys_parts:
            # Same cacheability rule as the non-streaming path — this used
            # to pass a bare string, so streaming requests never got prompt
            # caching even when the prefix was long enough to qualify.
            kwargs_clean["system"] = _anthropic_system_param(
                "\n\n".join(sys_parts), model
            )

        try:
            async with client.messages.stream(
                model=model, messages=conv, max_tokens=max_tokens, **kwargs_clean
            ) as s:
                # The raw event stream, not `text_stream`: the latter is
                # text only, so a streamed tool request had to be answered
                # from chat() in one piece and re-chunked. The events map
                # one-for-one onto the OpenAI delta shape the proxy
                # re-emits — a tool_use block start carries the id and
                # name, and each input_json_delta a slice of the arguments.
                #
                # Anthropic indexes CONTENT BLOCKS (text and tool_use share
                # one sequence); OpenAI indexes tool calls. Mapping between
                # the two is this dict, not the raw index, or a call that
                # follows a text block is announced at index 1 with nothing
                # at index 0 and every SDK that assembles by index breaks.
                ordinal: dict[int, int] = {}
                async for event in s:
                    kind = getattr(event, "type", "")
                    if kind == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", "") == "tool_use":
                            idx = len(ordinal)
                            ordinal[getattr(event, "index", 0)] = idx
                            yield {"tool_calls": [{
                                "index": idx,
                                "id": getattr(block, "id", "") or "",
                                "type": "function",
                                "function": {"name": getattr(block, "name", "") or "",
                                             "arguments": ""},
                            }]}
                    elif kind == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                yield text
                        elif dtype == "input_json_delta":
                            idx = ordinal.get(getattr(event, "index", 0))
                            if idx is None:
                                continue    # a block we never opened
                            yield {"tool_calls": [{
                                "index": idx,
                                "function": {
                                    "arguments": getattr(delta, "partial_json", "") or ""},
                            }]}
        except anthropic.RateLimitError as e:
            raise ProviderError("anthropic", "rate_limit", str(e), retryable=True)
        except anthropic.APIStatusError as e:
            raise ProviderError("anthropic", f"http_{e.status_code}", str(e), retryable=False)


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _stream_tool_call_delta(tc) -> dict:
    """One streamed tool-call fragment in the OpenAI chunk shape. Unlike a
    whole call, every field is optional here: later fragments carry only
    `index` and a slice of `arguments`."""
    def g(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    fn = g(tc, "function")
    out: dict = {"index": g(tc, "index") or 0}
    if g(tc, "id"):
        out["id"] = g(tc, "id")
        out["type"] = "function"
    if fn is not None:
        f: dict = {}
        if g(fn, "name"):
            f["name"] = g(fn, "name")
        if g(fn, "arguments") is not None:
            f["arguments"] = g(fn, "arguments")
        out["function"] = f
    return out


class OpenAIProvider(BaseProvider):

    # The proxy's wire shape IS this API's shape, so tools pass straight
    # through, streamed or not.
    supports_tools = True
    supports_tool_stream = True

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 base_url: Optional[str] = None):
        super().__init__(api_key, model)
        self._base_url = base_url

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = AsyncOpenAI(
            api_key=self.api_key,
            **({"base_url": self._base_url} if self._base_url else {}),
            **self._timeout_kwargs,
        )

        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        try:
            sampling = _sampling(kwargs)  # OpenAI SDK accepts temperature/top_p/stop natively
            sampling.update(tool_kwargs(kwargs))  # tools/tool_choice pass through as-is
            if stream:
                full_text = ""
                input_tokens = count_messages_tokens(all_messages, model)
                async for chunk in await client.chat.completions.create(
                    model=model, messages=all_messages, max_tokens=max_tokens,
                    stream=True, **sampling
                ):
                    delta = chunk.choices[0].delta.content or ""
                    full_text += delta
                return LLMResponse(text=full_text,
                                   input_tokens=input_tokens,
                                   output_tokens=count_tokens(full_text, model),
                                   model=model, provider="openai")

            resp = await client.chat.completions.create(
                model=model, messages=all_messages, max_tokens=max_tokens, **sampling
            )
            choice = resp.choices[0]
            calls = normalize_tool_calls(getattr(choice.message, "tool_calls", None))
            return LLMResponse(
                text=choice.message.content or "",
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=model,
                provider="openai",
                finish_reason=finish_reason(choice.finish_reason, calls),
                tool_calls=calls,
            )
        except Exception as e:
            raise ProviderError("openai", "api_error", str(e),
                                retryable=_openai_error_is_retryable(e))

    async def chat_stream(self, messages: list[dict], model: str = "",
                          max_tokens: int = 4096, system: str = "", **kwargs):
        """True SSE passthrough for OpenAI and all OpenAI-compatible providers
        (DeepSeek, Mistral, OpenRouter, Grok inherit this)."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")
        model = model or self.default_model
        client = AsyncOpenAI(
            api_key=self.api_key,
            **({"base_url": self._base_url} if self._base_url else {}),
            **self._timeout_kwargs,
        )
        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages
        try:
            stream = await client.chat.completions.create(
                model=model, messages=all_messages, max_tokens=max_tokens,
                stream=True, **_sampling(kwargs), **tool_kwargs(kwargs),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
                # Tool-call deltas arrive as partial objects keyed by
                # `index` (the id and name first, then argument fragments).
                # Forwarded as one dict event per chunk so the proxy can
                # re-emit them in the same OpenAI chunk shape.
                calls = getattr(delta, "tool_calls", None)
                if calls:
                    yield {"tool_calls": [_stream_tool_call_delta(tc) for tc in calls]}
        except Exception as e:
            raise ProviderError(self.__class__.__name__.lower().replace("provider", ""),
                                "api_error", str(e),
                                retryable=_openai_error_is_retryable(e))


# ── DeepSeek ──────────────────────────────────────────────────────────────────

class DeepSeekProvider(OpenAIProvider):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(api_key, model, base_url="https://api.deepseek.com/v1")
        self.default_model = model


# ── Mistral ───────────────────────────────────────────────────────────────────

class MistralProvider(OpenAIProvider):
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        super().__init__(api_key, model, base_url="https://api.mistral.ai/v1")


# ── OpenRouter ────────────────────────────────────────────────────────────────

class OpenRouterProvider(OpenAIProvider):
    def __init__(self, api_key: str, model: str = "anthropic/claude-sonnet-4-6"):
        super().__init__(api_key, model, base_url="https://openrouter.ai/api/v1")


# ── Grok ─────────────────────────────────────────────────────────────────────

class GrokProvider(OpenAIProvider):
    def __init__(self, api_key: str, model: str = "grok-3"):
        super().__init__(api_key, model, base_url="https://api.x.ai/v1")


# ── Cohere ────────────────────────────────────────────────────────────────────

class CohereProvider(BaseProvider):

    # v2 takes the OpenAI tool shape as-is; only the streamed event names
    # differ. See providers/tools.py.
    supports_tools = True
    supports_tool_stream = True

    def __init__(self, api_key: str, model: str = "command-r-plus"):
        super().__init__(api_key, model)

    def _chat_kwargs(self, kwargs: dict) -> dict:
        """Sampling and tools, shared by both paths so the streaming one
        cannot drift from the other."""
        s = _sampling(kwargs)
        out: dict = {}
        if "temperature" in s:
            out["temperature"] = s["temperature"]
        if "top_p" in s:
            out["p"] = s["top_p"]                # Cohere names top_p as `p`
        if "stop" in s:
            out["stop_sequences"] = _as_stop_list(s["stop"])
        tk = tool_kwargs(kwargs)
        if tk.get("tools"):
            out["tools"] = tk["tools"]
            # v2 has no tool_choice; "none" and "required" cannot be
            # enforced, and pretending otherwise would be worse than the
            # model simply deciding for itself.
        return out

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            import cohere
        except ImportError:
            raise ImportError("pip install cohere")

        client = cohere.AsyncClientV2(api_key=self.api_key)

        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        try:
            resp = await client.chat(model=model,
                                     messages=cohere_messages(all_messages),
                                     max_tokens=max_tokens,
                                     **self._chat_kwargs(kwargs))
            text = resp.message.content[0].text if resp.message.content else ""
            calls = normalize_tool_calls(getattr(resp.message, "tool_calls", None))
            usage = resp.usage
            return LLMResponse(
                text=text,
                input_tokens=usage.tokens.input_tokens if usage else count_messages_tokens(all_messages),
                output_tokens=usage.tokens.output_tokens if usage else count_tokens(text),
                model=model,
                provider="cohere",
                finish_reason=finish_reason(getattr(resp, "finish_reason", None), calls),
                tool_calls=calls,
            )
        except Exception as e:
            # Word-boundary matching, not a bare substring check: "rate" is
            # inside "separate", "moderate", "generate" — the exact
            # collision _RETRYABLE_TEXT's own docstring warns about for
            # OpenAI, just never mirrored here. A malformed-request error
            # mentioning any of those words was marked retryable (three
            # pointless retries before the real error surfaced), and a
            # genuine rate-limit phrased without the literal word "rate"
            # ("too many requests") was marked NOT retryable when it should
            # have been.
            raise ProviderError("cohere", "api_error", str(e),
                                retryable=bool(_RETRYABLE_TEXT.search(str(e))))

    async def chat_stream(self, messages: list[dict], model: str = "",
                          max_tokens: int = 4096, system: str = "", **kwargs):
        """True streaming, text and tool calls.

        This used to raise, so `stream: true` on Cohere was a 501 — v2 has
        had `chat_stream` throughout.
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("pip install cohere")

        model = model or self.default_model
        client = cohere.AsyncClientV2(api_key=self.api_key)
        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        try:
            stream = client.chat_stream(model=model,
                                        messages=cohere_messages(all_messages),
                                        max_tokens=max_tokens,
                                        **self._chat_kwargs(kwargs))
            async for event in stream:
                mapped = cohere_stream_event(event)
                if not mapped:
                    continue
                if "text" in mapped:
                    yield mapped["text"]
                else:
                    yield mapped
        except Exception as e:
            raise ProviderError("cohere", "api_error", str(e),
                                retryable=bool(_RETRYABLE_TEXT.search(str(e))))


# ── Gemini ───────────────────────────────────────────────────────────────────

def _gemini_error_is_retryable(exc: Exception) -> bool:
    """True if a google-genai API error is worth retrying.

    Mirrors _openai_error_is_retryable's shape: the typed exception is
    authoritative when available (APIError.code is the real HTTP status
    the API returned), word-boundary text matching is the fallback.
    "quota"/"429" are kept as an extra Gemini-specific signal even in
    the fallback — some quota errors surface as a plain-text message
    that doesn't set a matching HTTP status.
    """
    try:
        from google.genai import errors
        if isinstance(exc, errors.APIError):
            return exc.code in (408, 429, 500, 502, 503, 504)
    except Exception:
        pass  # SDK missing or shaped differently — fall through to text
    return ("quota" in str(exc).lower() or "429" in str(exc)
            or bool(_RETRYABLE_TEXT.search(str(exc))))


class GeminiProvider(BaseProvider):
    """
    Uses the google-genai SDK — the actively maintained, unified SDK.
    google-generativeai (used until this migration) reached end of life
    in January 2026 and prints a deprecation warning on import; nothing
    about this adapter's SDK dependency was broken before the migration,
    it was heading there.

    Fixed version (carried over from the pre-migration adapter):
    - Full conversation history (not just last message)
    - Native async via the SDK's own .aio client (not run_in_executor)

    Also strictly better than before on two counts the SDK swap enabled:
    - The API key is passed per-Client (genai.Client(api_key=...)) rather
      than through the old SDK's process-global genai.configure() — a
      proxy that may hold different provider credentials per caller had
      no business mutating global SDK state on every request.
    - Retry classification now has a real typed exception with the
      actual HTTP status code (see _gemini_error_is_retryable) instead
      of guessing from message text alone.
    """

    supports_tools = True
    # The SDK carries function_call parts on the streamed chunks, so a
    # streamed tool request does not have to fall back to a buffered call.
    supports_tool_stream = True

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)

    def _config_kwargs(self, kwargs: dict, max_tokens: int,
                       system_instruction: Optional[str]) -> dict:
        """Everything that goes into GenerateContentConfig. Shared so the
        streaming path cannot drift from the non-streaming one, which is
        how it lost temperature/top_p/stop the last time."""
        s = _sampling(kwargs)
        gen_kw: dict = {"max_output_tokens": max_tokens}
        if system_instruction:
            gen_kw["system_instruction"] = system_instruction
        if "temperature" in s:
            gen_kw["temperature"] = s["temperature"]
        if "top_p" in s:
            gen_kw["top_p"] = s["top_p"]
        if "stop" in s:
            gen_kw["stop_sequences"] = _as_stop_list(s["stop"])
        tk = tool_kwargs(kwargs)
        declarations = gemini_tools(tk.get("tools") or [])
        if declarations:
            gen_kw["tools"] = declarations
            config = gemini_tool_config(tk.get("tool_choice"))
            if config:
                gen_kw["tool_config"] = config
        return gen_kw

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        client = genai.Client(
            api_key=self.api_key,
            **({"http_options": {"timeout": int(self.request_timeout * 1000)}}
               if self.request_timeout and self.request_timeout > 0 else {}),
        )

        # Extract system prompt
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        system_instruction = "\n\n".join(sys_parts) if sys_parts else None

        conversation = conversation_messages(messages)

        # generate_content over the whole conversation, not chats.create
        # with a history and one last message: the chat helper takes plain
        # text for the latest turn, which cannot express an assistant turn
        # that asked for a tool or the client's results coming back. Those
        # are content parts, and this is the call that takes them.
        contents = gemini_contents(conversation)

        try:
            gen_kw = self._config_kwargs(kwargs, max_tokens, system_instruction)

            resp = await client.aio.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**gen_kw),
            )
            candidates = getattr(resp, "candidates", None) or []
            text, calls = gemini_response(candidates[0]) if candidates else ("", [])
            if not text and not calls:
                text = getattr(resp, "text", "") or ""
            # Real API-reported usage when the SDK provides it.
            # prompt_token_count is the full effective prompt size
            # (Google's own docs: "includes ... cached content"), so unlike
            # the local estimate below it correctly counts
            # system_instruction — which lives outside `conversation` here
            # the same way Anthropic's system param lives outside `conv`,
            # and was silently missing from every token count that used
            # to fall back to count_messages_tokens(conversation, model)
            # unconditionally, undercounting input_tokens on every call
            # that set a system prompt.
            usage = resp.usage_metadata
            if usage is not None and usage.prompt_token_count:
                input_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
            else:
                input_tokens = count_messages_tokens(conversation, model)
                output_tokens = count_tokens(text, model)
            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="gemini",
                finish_reason=finish_reason(
                    getattr(candidates[0], "finish_reason", None) if candidates else None,
                    calls),
                tool_calls=calls,
            )
        except Exception as e:
            raise ProviderError("gemini", "api_error", str(e),
                                retryable=_gemini_error_is_retryable(e))

    async def chat_stream(self, messages: list[dict], model: str = "",
                          max_tokens: int = 4096, system: str = "", **kwargs):
        """True streaming, text and tool calls.

        This used to raise, so `stream: true` on Gemini was a 501 — the
        SDK has had `generate_content_stream` throughout.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        model = model or self.default_model
        client = genai.Client(
            api_key=self.api_key,
            **({"http_options": {"timeout": int(self.request_timeout * 1000)}}
               if self.request_timeout and self.request_timeout > 0 else {}),
        )

        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        system_instruction = "\n\n".join(sys_parts) if sys_parts else None
        contents = gemini_contents(conversation_messages(messages))
        gen_kw = self._config_kwargs(kwargs, max_tokens, system_instruction)

        try:
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**gen_kw),
            )
            # Gemini sends a whole function_call part rather than argument
            # fragments, so each becomes one delta carrying the complete
            # arguments. The ordinal is ours: Gemini does not number them.
            emitted = 0
            async for chunk in stream:
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    text = getattr(chunk, "text", "") or ""
                    if text:
                        yield text
                    continue
                text, calls = gemini_response(candidates[0])
                if text:
                    yield text
                if calls:
                    yield {"tool_calls": [
                        {"index": emitted + i, "id": c["id"], "type": "function",
                         "function": dict(c["function"])}
                        for i, c in enumerate(calls)
                    ]}
                    emitted += len(calls)
        except Exception as e:
            raise ProviderError("gemini", "api_error", str(e),
                                retryable=_gemini_error_is_retryable(e))


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(BaseProvider):

    supports_tools = True
    # Ollama does not stream tool-call fragments — it sends the finished
    # calls on the last message — but the stream carries them, which is
    # what this flag means: the proxy does not have to fall back to a
    # buffered non-streaming call to get them.
    supports_tool_stream = True

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        super().__init__(api_key="", model=model)
        self._base_url = base_url.rstrip("/")

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx")

        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        s = _sampling(kwargs)
        options = {"num_predict": max_tokens}
        if "temperature" in s:
            options["temperature"] = s["temperature"]
        if "top_p" in s:
            options["top_p"] = s["top_p"]
        if "stop" in s:
            options["stop"] = _as_stop_list(s["stop"])
        payload = {"model": model, "messages": ollama_messages(all_messages),
                   "stream": False, "options": options}
        tk = tool_kwargs(kwargs)
        if tk.get("tools"):
            # Ollama takes OpenAI-shaped declarations; it has no
            # tool_choice, so "none"/"required" cannot be enforced here.
            payload["tools"] = tk["tools"]

        async with httpx.AsyncClient(timeout=self.request_timeout or None) as client:
            try:
                r = await client.post(f"{self._base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                message = data.get("message", {})
                text = message.get("content", "") or ""
                calls = ollama_tool_calls(message.get("tool_calls"))
                return LLMResponse(
                    text=text,
                    input_tokens=data.get("prompt_eval_count", count_messages_tokens(all_messages)),
                    output_tokens=data.get("eval_count", count_tokens(text)),
                    model=model,
                    provider="ollama",
                    finish_reason=finish_reason(data.get("done_reason"), calls),
                    tool_calls=calls,
                )
            except Exception as e:
                raise ProviderError("ollama", "api_error", str(e),
                                    retryable=_ollama_error_is_retryable(e))

    async def chat_stream(self, messages: list[dict], model: str = "",
                          max_tokens: int = 4096, system: str = "", **kwargs):
        """Streaming passthrough for local Ollama models."""
        import json as _json

        import httpx
        model = model or self.default_model
        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages
        # Same option mapping as _call(): the streaming path dropped
        # temperature/top_p/stop on the floor, so a client got different
        # sampling depending on whether it asked for a stream.
        s = _sampling(kwargs)
        options = {"num_predict": max_tokens}
        if "temperature" in s:
            options["temperature"] = s["temperature"]
        if "top_p" in s:
            options["top_p"] = s["top_p"]
        if "stop" in s:
            options["stop"] = _as_stop_list(s["stop"])
        payload = {"model": model, "messages": ollama_messages(all_messages),
                   "stream": True, "options": options}
        tk = tool_kwargs(kwargs)
        if tk.get("tools"):
            # The streaming payload carried no tools at all, so a client
            # that asked for tools AND a stream got a model that could not
            # see them — the same silent drop the non-streaming path had.
            payload["tools"] = tk["tools"]
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout or None) as client:
                async with client.stream("POST", f"{self._base_url}/api/chat",
                                         json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        data = _json.loads(line)
                        message = data.get("message", {}) or {}
                        chunk = message.get("content", "")
                        if chunk:
                            yield chunk
                        # Ollama sends finished calls on one message rather
                        # than fragments, so each becomes a single delta
                        # carrying the whole arguments string. The proxy
                        # assembles by index either way.
                        calls = ollama_tool_calls(message.get("tool_calls"))
                        if calls:
                            yield {"tool_calls": [
                                {"index": i, "id": c["id"], "type": "function",
                                 "function": dict(c["function"])}
                                for i, c in enumerate(calls)
                            ]}
                        if data.get("done"):
                            break
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError("ollama", "api_error", str(e),
                                retryable=_ollama_error_is_retryable(e))


# ── Registry ──────────────────────────────────────────────────────────────────

# Model-name prefixes that identify which provider a model belongs to.
# Used only to warn about an obvious provider/model mismatch.
_MODEL_FAMILY_HINTS = {
    "claude": ("anthropic", "claude"),
    "gpt-": ("openai", "gpt"),
    "o1-": ("openai", "gpt"),
    "gemini": ("gemini",),
    "deepseek": ("deepseek",),
    "mistral": ("mistral",),
    "grok": ("grok",),
    "command-r": ("cohere",),
}


def _warn_on_model_provider_mismatch(provider: str, model: str) -> None:
    """Warn when default_model clearly belongs to a different provider.

    `default_model` defaults to a Claude model, so switching only
    `provider` to openai (a one-line change, and the obvious one to make)
    sent "claude-sonnet-4-6" to the OpenAI API and produced an opaque
    model-not-found error from the SDK with nothing pointing at the
    actual cause. The `model or "<default>"` fallbacks in the registry
    below never help, because default_model is never empty.
    """
    m = (model or "").lower()
    for prefix, owners in _MODEL_FAMILY_HINTS.items():
        if m.startswith(prefix) and provider not in owners:
            logger.warning(
                "default_model=%r looks like a %s model, but provider=%r. "
                "The request will be sent to %s and will most likely fail "
                "with an unknown-model error — set default_model to one of "
                "%s's models.",
                model, owners[0], provider, provider, provider,
            )
            return


def build_provider(settings, model: Optional[str] = None) -> BaseProvider:
    """Build the correct provider from settings. `model` overrides
    settings.default_model on the same provider (used by the extraction
    pass when graph_checkpoint.extraction_model pins one)."""
    provider = settings.provider.lower()
    key = settings.get_api_key_for_provider(provider)
    model = model or settings.default_model
    _warn_on_model_provider_mismatch(provider, model)

    mapping = {
        "anthropic": lambda: AnthropicProvider(key, model),
        "claude": lambda: AnthropicProvider(key, model),
        "openai": lambda: OpenAIProvider(key, model),
        "gpt": lambda: OpenAIProvider(key, model),
        "deepseek": lambda: DeepSeekProvider(key, model or "deepseek-chat"),
        "mistral": lambda: MistralProvider(key, model or "mistral-large-latest"),
        "grok": lambda: GrokProvider(key, model or "grok-3"),
        "cohere": lambda: CohereProvider(key, model or "command-r-plus"),
        "gemini": lambda: GeminiProvider(key, model or "gemini-1.5-pro"),
        "ollama": lambda: OllamaProvider(model or "llama3"),
        "openrouter": lambda: OpenRouterProvider(key, model),
    }

    if provider not in mapping:
        raise ValueError(f"Unknown provider: {provider!r}. Valid: {list(mapping)}")

    instance = mapping[provider]()
    # request_timeout is a Settings field (config/settings.py), not read
    # from the environment here — getattr covers a caller that passes
    # something settings-shaped without it, falling back to the class
    # default rather than raising.
    instance.request_timeout = getattr(settings, "request_timeout",
                                       instance.request_timeout)
    return instance
