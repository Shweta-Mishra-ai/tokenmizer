"""TokenMizer configuration — Pydantic Settings with env var support."""
from __future__ import annotations

from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CompressionSettings(BaseSettings):
    enabled: bool = True
    engine: Literal["llmlingua2", "heuristic", "none"] = "heuristic"
    ratio: float = Field(default=0.5, ge=0.1, le=1.0)
    min_tokens_to_compress: int = 300


class MemorySettings(BaseSettings):
    enabled: bool = True
    max_tokens_before_summary: int = 4000
    recent_turns_verbatim: int = 10


class GraphCheckpointSettings(BaseSettings):
    enabled: bool = True
    trigger_at_percent: float = Field(default=0.85, ge=0.5, le=0.99)
    storage_dir: str = "./checkpoints"
    max_resume_tokens: int = 400
    use_llm_extraction: bool = False  # set True for 80%+ recall (needs API key, ~$0.001/turn)
    extraction_model: str = ""        # leave empty = auto-pick cheapest model for your provider
    min_confidence: float = 0.65      # minimum validation confidence threshold


class RoutingSettings(BaseSettings):
    enabled: bool = False
    simple_model: str = "claude-haiku-4-5"
    medium_model: str = "claude-sonnet-4-6"
    complex_model: str = "claude-sonnet-4-6"
    complexity_threshold: float = 0.6


class CacheSettings(BaseSettings):
    enabled: bool = True
    similarity_threshold: float = 0.92
    ttl_seconds: int = 3600
    max_size: int = 10_000
    # "session" (default): every cached prompt is scoped to its session_id,
    # never shared across sessions — safe by default for hosted/team use.
    # "shared": non-sensitive prompts are shared globally across sessions
    # (higher hit rate, but requires trusting the sensitivity heuristic in
    # semantic_cache/cache.py::_is_session_sensitive — see TM-03). Opt in
    # explicitly; do not flip this without understanding that heuristic's
    # documented limits.
    share_scope: Literal["session", "shared"] = "session"


class TerseOutputSettings(BaseSettings):
    enabled: bool = True
    level: Literal["lite", "full", "ultra"] = "full"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TOKENMIZER_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Provider — synced exactly with providers/registry.py
    provider: Literal[
        "anthropic", "claude",
        "openai", "gpt",
        "deepseek",
        "mistral",
        "grok",
        "cohere",
        "gemini",
        "ollama",
        "openrouter",
    ] = "anthropic"

    default_model: str = "claude-sonnet-4-6"

    # API keys (prefer env vars over config file)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    grok_api_key: str = ""
    deepseek_api_key: str = ""
    mistral_api_key: str = ""
    cohere_api_key: str = ""
    openrouter_api_key: str = ""

    # State backend
    state_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    api_key: str = ""  # TOKENMIZER_API_KEY — empty = dev mode (no auth)

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Sub-configs
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    graph_checkpoint: GraphCheckpointSettings = Field(default_factory=GraphCheckpointSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    terse_output: TerseOutputSettings = Field(default_factory=TerseOutputSettings)

    # Server
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8000

    def get_api_key_for_provider(self, provider: str) -> str:
        mapping = {
            "anthropic": self.anthropic_api_key,
            "claude": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gpt": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "grok": self.grok_api_key,
            "deepseek": self.deepseek_api_key,
            "mistral": self.mistral_api_key,
            "cohere": self.cohere_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "",
        }
        return mapping.get(provider, "")

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        import logging
        import os
        logger = logging.getLogger(__name__)
        yaml_path = os.environ.get("TOKENMIZER_CONFIG", "tokenmizer.yaml")
        if os.path.exists(yaml_path):
            try:
                _settings = Settings.from_yaml(yaml_path)
            except Exception as e:
                # FIXED: previously this silently discarded the user's
                # entire config file and fell back to hardcoded defaults
                # with ZERO indication anything went wrong. The defaults
                # are dev-mode-permissive: no API key required, CORS may
                # be wider than intended, state backend is in-memory (no
                # Redis). An operator who sets a real config — including
                # security-relevant fields like `api_key` or
                # `cors_origins` — could end up running with none of that
                # applied, with no error, no warning, nothing. This is a
                # security-relevant failure mode disguised as "graceful
                # fallback." Logging at `error` (not silent) means a typo
                # in tokenmizer.yaml is visible at startup instead of
                # discovered later as "wait, why does this accept
                # unauthenticated requests?"
                logger.error(
                    f"Failed to load config from {yaml_path}: {e}. "
                    "Falling back to hardcoded defaults — this means any "
                    "settings in your YAML file (including api_key, "
                    "cors_origins, state_backend) are NOT applied. Fix the "
                    "YAML file and restart."
                )
                _settings = Settings()
        else:
            _settings = Settings()
    return _settings
