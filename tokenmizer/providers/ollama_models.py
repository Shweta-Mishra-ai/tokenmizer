"""
Choosing the local model that extracts session memory through Ollama.

WHY LOCAL
---------
Extraction reads every turn of a session — code, logs, the decisions behind
them — to decide what to remember. Sending that to a hosted model is a
second copy of the conversation leaving the machine, often to a cheaper
third-party model than the one chatting. A model served by Ollama on the
same host keeps it local and costs nothing per call.

WHICH MODEL
-----------
`graph_checkpoint.extraction_model: auto` asks Ollama what is installed and
takes the first match from PREFERENCE. The order is a STARTING POINT, not a
measurement: it reflects what these model families are generally good at —
following a JSON schema, short factual spans, no chattiness — weighed
against what runs on an ordinary machine. It was not measured on this
project's corpus, because the environment it was written in could not run
Ollama.

The measurement is yours to run, on your hardware, against the labelled
corpus the extractor is scored on:

    python -m benchmarks.extraction_models            # every installed model
    python -m benchmarks.extraction_models --models qwen2.5:7b,llama3.1:8b

It reports extraction F1 per category with and without each model, JSON
failures and seconds per call, and says which to set. Pin the winner with
`extraction_model: <name>`.

Rough sizing, for choosing what to pull (download / RAM in use):
  ~3B  (qwen2.5:3b, llama3.2:3b)   ~2 GB / ~4 GB    — runs on any laptop, CPU-only is fine
  ~7B  (qwen2.5:7b, qwen3:8b)      ~5 GB / ~8 GB    — the usual sweet spot
  ~14B (qwen2.5:14b, phi4)         ~9 GB / ~16 GB   — better recall; wants a GPU or patience
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Preferred extraction models, best first among those installed. See the
# module docstring: an informed default, to be confirmed by the benchmark.
PREFERENCE: tuple[str, ...] = (
    "qwen2.5:14b", "qwen3:14b", "phi4:14b", "gemma3:12b",
    "qwen2.5:7b", "qwen3:8b", "llama3.1:8b", "gemma3:4b",
    "qwen2.5:3b", "qwen3:4b", "llama3.2:3b",
)

# What to suggest pulling when nothing usable is installed.
RECOMMENDED_PULL = "qwen2.5:7b"

# Families that think out loud before answering. Extraction wants the JSON,
# not the reasoning: thinking spends the output budget and minutes of CPU,
# so it is switched off for these (Ollama's `think: false`).
_THINKING = re.compile(r"^(?:qwen3|deepseek-r1|gpt-oss|magistral|phi4-reasoning)", re.I)

# Installed models that cannot hold a conversation.
_NOT_CHAT = re.compile(r"embed|bge|nomic|minilm|e5-|rerank|clip|llava|vision", re.I)


def is_thinking_model(name: str) -> bool:
    return bool(_THINKING.match(name or ""))


def _base(name: str) -> str:
    """'qwen2.5:7b-instruct-q4_K_M' -> 'qwen2.5:7b'; 'llama3.1' -> 'llama3.1:latest'."""
    name = name.strip().lower()
    if ":" not in name:
        return f"{name}:latest"
    family, tag = name.split(":", 1)
    return f"{family}:{tag.split('-', 1)[0]}"


def installed_models(base_url: str, timeout: float = 3.0) -> list[str] | None:
    """Model names Ollama has pulled, or None if it could not be asked."""
    try:
        import httpx
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m.get("name") or m.get("model") for m in r.json().get("models", [])
                if m.get("name") or m.get("model")]
    except Exception as e:
        logger.warning("Could not list Ollama models at %s (%s: %s)", base_url, type(e).__name__, e)
        return None


def pick_model(installed: list[str], configured: str = "") -> tuple[str | None, str]:
    """(model to use, why) — None when nothing usable is installed.

    A configured name other than "auto" is used as given if it is installed;
    if it is not, that is said, and the call fails rather than silently
    falling back to a different model than the one asked for.
    """
    configured = (configured or "").strip()
    chat = [m for m in installed if not _NOT_CHAT.search(m)]
    if configured and configured.lower() != "auto":
        if any(_base(m) == _base(configured) or m == configured for m in installed):
            return configured, "configured"
        return None, (f"extraction_model {configured!r} is not installed in Ollama — "
                      f"run: ollama pull {configured}")
    by_base = {_base(m): m for m in chat}
    for want in PREFERENCE:
        if want in by_base:
            return by_base[want], f"auto: {want} is the highest-ranked installed model"
    if chat:
        return chat[0], (f"auto: no ranked model installed, using {chat[0]!r}; "
                         f"for better extraction run: ollama pull {RECOMMENDED_PULL}")
    return None, f"no chat model installed in Ollama — run: ollama pull {RECOMMENDED_PULL}"
