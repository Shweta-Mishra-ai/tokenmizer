"""TokenMizer configuration — Pydantic Settings with env var support."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Nested sub-configs ────────────────────────────────────────────────────────
#
# FIXED (TM-08): these used to subclass BaseSettings directly with no
# env_prefix of their own, which means each one independently read BARE
# environment variables — e.g. TerseOutputSettings() would pick up a
# plain `LEVEL` or `ENABLED` env var from the host process. Those are
# generic enough names that a CI system or shell profile could set them
# for unrelated reasons and silently reconfigure the product (a stray
# ENABLED=false in the environment could disable compression, memory, AND
# terse output simultaneously, with zero log line). They are nested value
# objects, not independently-configurable top-level settings — the outer
# Settings object already provides TOKENMIZER_-prefixed, __-nested env
# var access to every field on these (e.g.
# TOKENMIZER_TERSE_OUTPUT__LEVEL=ultra), so plain pydantic.BaseModel is
# both correct and sufficient.


class CompressionSettings(BaseModel):
    enabled: bool = True
    engine: Literal["llmlingua2", "heuristic", "none"] = "heuristic"
    ratio: float = Field(default=0.5, ge=0.1, le=1.0)
    min_tokens_to_compress: int = 300


class MemorySettings(BaseModel):
    enabled: bool = True
    max_tokens_before_summary: int = 4000
    recent_turns_verbatim: int = 10


class GraphCheckpointSettings(BaseModel):
    enabled: bool = True
    trigger_at_percent: float = Field(default=0.85, ge=0.5, le=0.99)
    storage_dir: str = "./checkpoints"
    max_resume_tokens: int = 400
    use_llm_extraction: bool = False  # set True for 80%+ recall (needs API key, ~$0.001/turn)
    extraction_model: str = ""        # leave empty = auto-pick cheapest model for your provider
    min_confidence: float = 0.65      # minimum validation confidence threshold


class RoutingSettings(BaseModel):
    enabled: bool = False
    simple_model: str = "claude-haiku-4-5"
    medium_model: str = "claude-sonnet-4-6"
    complex_model: str = "claude-sonnet-4-6"
    complexity_threshold: float = 0.6


class CacheSettings(BaseModel):
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


class TerseOutputSettings(BaseModel):
    enabled: bool = True
    level: Literal["lite", "full", "ultra"] = "full"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TOKENMIZER_",
        env_nested_delimiter="__",
        env_file=".env",
        # FIXED (TM-06 / closes #28): was "ignore" — a misspelled YAML key
        # (e.g. `api_keys:` instead of `api_key:`) parsed cleanly and
        # silently discarded the value, with no exception and no log
        # line. "forbid" raises the same way a YAML syntax error already
        # did, so a typo is caught by the SAME fail-closed logic in
        # get_settings() below instead of needing separate handling.
        extra="forbid",
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
    # Was "0.0.0.0" (all interfaces) — the CLI's `serve` command didn't
    # even read this field until this fix (see cli.py), so the old
    # default was inert. Now that it's wired in, localhost-only is the
    # safe default; the documented Docker deployment path is unaffected
    # since the Dockerfile always passes --host 0.0.0.0 explicitly.
    proxy_host: str = "127.0.0.1"
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


class ConfigSecurityError(RuntimeError):
    """Raised when TOKENMIZER_ENV=production and the effective config is
    unsafe to boot with — either it failed to load at all, or it loaded
    fine but landed on permissive defaults (e.g. no api_key). Refusing to
    start is the point: a production deployment must never silently run
    more permissively than the operator configured."""


_settings: Settings | None = None


def _is_production() -> bool:
    """Read directly from the environment, not from a Settings field —
    this must be checkable BEFORE we know whether Settings() can even be
    constructed (a config load failure is exactly the case this guards
    against), so it can't depend on the object it's deciding how to
    handle. Matches the flag name issue #28 itself proposed."""
    import os
    return os.environ.get("TOKENMIZER_ENV", "").strip().lower() == "production"


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        import logging
        import os
        logger = logging.getLogger(__name__)
        production = _is_production()
        yaml_path = os.environ.get("TOKENMIZER_CONFIG", "tokenmizer.yaml")

        if os.path.exists(yaml_path):
            try:
                loaded = Settings.from_yaml(yaml_path)
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
                #
                # FIXED further (closes #28): in production, logging is
                # not enough — a log line nobody is watching at 3am is
                # not a safety control. TOKENMIZER_ENV=production now
                # refuses to start at all rather than fall back.
                if production:
                    logger.error(
                        f"Failed to load config from {yaml_path}: {e}. "
                        "TOKENMIZER_ENV=production — refusing to start "
                        "rather than fall back to permissive defaults."
                    )
                    raise ConfigSecurityError(
                        f"Refusing to start: TOKENMIZER_ENV=production and "
                        f"{yaml_path} failed to load ({type(e).__name__}: {e}). "
                        "Fix the config file, or explicitly unset "
                        "TOKENMIZER_ENV (or set it to a non-'production' "
                        "value) to accept dev-mode permissive defaults "
                        "instead."
                    ) from e
                logger.error(
                    f"Failed to load config from {yaml_path}: {e}. "
                    "Falling back to hardcoded defaults — this means any "
                    "settings in your YAML file (including api_key, "
                    "cors_origins, state_backend) are NOT applied. Fix the "
                    "YAML file and restart."
                )
                loaded = Settings()
        else:
            loaded = Settings()

        # Fail-closed floor: even when loading succeeded WITHOUT any
        # exception, a production deployment must never boot
        # unauthenticated. This covers the gap issue #28's literal ask
        # (parse-failure fallback) didn't: an operator who simply never
        # set api_key at all — no typo, no load error, just a genuine
        # config gap — would otherwise get a perfectly "successful" boot
        # into the same unauthenticated state.
        if production and not loaded.api_key:
            logger.error(
                "TOKENMIZER_ENV=production but no api_key is configured — "
                "refusing to start unauthenticated."
            )
            raise ConfigSecurityError(
                "Refusing to start: TOKENMIZER_ENV=production but no "
                "api_key is set (TOKENMIZER_API_KEY env var, or api_key in "
                "tokenmizer.yaml). Set an API key, or explicitly unset "
                "TOKENMIZER_ENV to run in development mode."
            )

        _settings = loaded
    return _settings
