"""
The extraction call: how HybridExtractor.llm_extract talks to a provider.

One function, used by the proxy's background extraction and by
benchmarks/extraction_models, so the benchmark measures exactly the call
production makes.
"""
from __future__ import annotations

from tokenmizer.graph_memory.patterns import EXTRACTION_SCHEMA


def extraction_fn(provider):
    """`provider_fn` for HybridExtractor.extract(), bound to `provider`.

    For Ollama it asks for the reply constrained to EXTRACTION_SCHEMA, at
    temperature 0, with a reasoning model's thinking turned off — what turns
    a small local model from "usually JSON" into "always JSON". Other
    providers get the plain call they always got: OpenAI's reasoning models,
    for one, reject a temperature parameter, and the extra keys are ignored
    by adapters that do not read them.
    """
    from tokenmizer.providers.providers import OllamaProvider

    local = isinstance(provider, OllamaProvider)

    async def _call(messages, system="", max_tokens=600):
        extra = ({"temperature": 0, "json_schema": EXTRACTION_SCHEMA, "think": False}
                 if local else {})
        r = await provider.chat(messages=messages, system=system,
                                max_tokens=max_tokens, **extra)
        return {"text": r.text}

    return _call
