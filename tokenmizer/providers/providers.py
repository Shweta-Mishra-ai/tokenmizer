"""
Provider adapters for all supported LLMs.

Fixed in this version:
- Gemini: full conversation history passed (not just last message)
- Gemini: proper async (not run_in_executor)
- asyncio.get_event_loop() → asyncio.get_running_loop()
- Lazy imports — no SDK required at import time
- Normalized ProviderError across all providers
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from tokenmizer.core.errors import ProviderError
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

# Sampling params forwarded from the proxy request to every provider.
# Each provider maps these to its own SDK's naming.
_SAMPLING_KEYS = ("temperature", "top_p", "stop")


def _sampling(kwargs: dict) -> dict:
    """Extract only recognized sampling params from arbitrary kwargs."""
    return {k: kwargs[k] for k in _SAMPLING_KEYS if kwargs.get(k) is not None}


def _as_stop_list(stop) -> list[str]:
    return [stop] if isinstance(stop, str) else list(stop)


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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseProvider(ABC):

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key
        self.default_model = model
        self._retry_delays = [1.0, 2.0, 4.0]

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


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicProvider(BaseProvider):

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        super().__init__(api_key, model)

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Separate system messages from conversation
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        conv = [m for m in messages if m.get("role") != "system"]
        system_text = "\n\n".join(sys_parts) if sys_parts else None

        try:
            kwargs_clean = _sampling(kwargs)
            # Anthropic SDK uses `stop_sequences`, not OpenAI's `stop`
            if "stop" in kwargs_clean:
                kwargs_clean["stop_sequences"] = _as_stop_list(kwargs_clean.pop("stop"))
            if system_text:
                # Anthropic prompt caching — system prompts >200 tokens cached at 90% discount
                # on repeated calls. Cache persists for 5 minutes.
                if len(system_text) > 800:  # ~200 tokens
                    kwargs_clean["system"] = [
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                else:
                    kwargs_clean["system"] = system_text

            if stream:
                full_text = ""
                input_tokens = count_messages_tokens(conv, model)
                async with client.messages.stream(
                    model=model, messages=conv, max_tokens=max_tokens, **kwargs_clean
                ) as s:
                    async for chunk in s.text_stream:
                        full_text += chunk
                output_tokens = count_tokens(full_text, model)
                return LLMResponse(text=full_text, input_tokens=input_tokens,
                                   output_tokens=output_tokens, model=model, provider="anthropic")

            resp = await client.messages.create(
                model=model, messages=conv, max_tokens=max_tokens, **kwargs_clean
            )
            text = resp.content[0].text if resp.content else ""
            return LLMResponse(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=model,
                provider="anthropic",
                finish_reason=resp.stop_reason or "stop",
            )
        except anthropic.RateLimitError as e:
            raise ProviderError("anthropic", "rate_limit", str(e), retryable=True, retry_after=60.0)
        except anthropic.APIStatusError as e:
            retryable = e.status_code in (500, 502, 503, 529)
            raise ProviderError("anthropic", f"http_{e.status_code}", str(e), retryable=retryable)


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(BaseProvider):

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
        )

        all_messages = messages[:]
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        try:
            sampling = _sampling(kwargs)  # OpenAI SDK accepts temperature/top_p/stop natively
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
            return LLMResponse(
                text=choice.message.content or "",
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=model,
                provider="openai",
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception as e:
            err = str(e).lower()
            retryable = "rate" in err or "server" in err or "timeout" in err
            raise ProviderError("openai", "api_error", str(e), retryable=retryable)


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

    def __init__(self, api_key: str, model: str = "command-r-plus"):
        super().__init__(api_key, model)

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
            s = _sampling(kwargs)
            cohere_kw = {}
            if "temperature" in s:
                cohere_kw["temperature"] = s["temperature"]
            if "top_p" in s:
                cohere_kw["p"] = s["top_p"]          # Cohere names top_p as `p`
            if "stop" in s:
                cohere_kw["stop_sequences"] = _as_stop_list(s["stop"])
            resp = await client.chat(model=model, messages=all_messages,
                                     max_tokens=max_tokens, **cohere_kw)
            text = resp.message.content[0].text if resp.message.content else ""
            usage = resp.usage
            return LLMResponse(
                text=text,
                input_tokens=usage.tokens.input_tokens if usage else count_messages_tokens(all_messages),
                output_tokens=usage.tokens.output_tokens if usage else count_tokens(text),
                model=model,
                provider="cohere",
            )
        except Exception as e:
            raise ProviderError("cohere", "api_error", str(e), retryable="rate" in str(e).lower())


# ── Gemini — FIXED ─────────────────────────────────────────────────────────────

class GeminiProvider(BaseProvider):
    """
    Fixed version:
    - Full conversation history (not just last message)
    - Native async via generate_content_async (not run_in_executor)
    - asyncio.get_running_loop() instead of deprecated get_event_loop()
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)

    async def _call(self, messages, model, max_tokens, stream, system, **kwargs) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        genai.configure(api_key=self.api_key)

        # Extract system prompt
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        if system:
            sys_parts.insert(0, system)
        system_instruction = "\n\n".join(sys_parts) if sys_parts else None

        conversation = [m for m in messages if m.get("role") != "system"]

        m_instance = genai.GenerativeModel(
            model,
            **({"system_instruction": system_instruction} if system_instruction else {}),
        )

        # Build full history (all turns except last)
        history = []
        for msg in conversation[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        last_msg = conversation[-1]["content"] if conversation else ""

        try:
            chat = m_instance.start_chat(history=history)
            s = _sampling(kwargs)
            gen_kw = {"max_output_tokens": max_tokens}
            if "temperature" in s:
                gen_kw["temperature"] = s["temperature"]
            if "top_p" in s:
                gen_kw["top_p"] = s["top_p"]
            if "stop" in s:
                gen_kw["stop_sequences"] = _as_stop_list(s["stop"])
            # Use native async — not run_in_executor
            resp = await chat.send_message_async(
                last_msg,
                generation_config=genai.GenerationConfig(**gen_kw),
            )
            text = resp.text or ""
            input_tokens = count_messages_tokens(conversation, model)
            output_tokens = count_tokens(text, model)
            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="gemini",
            )
        except Exception as e:
            retryable = "quota" in str(e).lower() or "429" in str(e)
            raise ProviderError("gemini", "api_error", str(e), retryable=retryable)


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(BaseProvider):

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
        payload = {"model": model, "messages": all_messages, "stream": False,
                   "options": options}

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(f"{self._base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                text = data.get("message", {}).get("content", "")
                return LLMResponse(
                    text=text,
                    input_tokens=data.get("prompt_eval_count", count_messages_tokens(all_messages)),
                    output_tokens=data.get("eval_count", count_tokens(text)),
                    model=model,
                    provider="ollama",
                )
            except Exception as e:
                raise ProviderError("ollama", "api_error", str(e), retryable=True)


# ── Registry ──────────────────────────────────────────────────────────────────

def build_provider(settings) -> BaseProvider:
    """Build the correct provider from settings."""
    provider = settings.provider.lower()
    key = settings.get_api_key_for_provider(provider)
    model = settings.default_model

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

    return mapping[provider]()
