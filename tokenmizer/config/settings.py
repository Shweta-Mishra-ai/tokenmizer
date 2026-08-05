"""TokenMizer configuration — Pydantic Settings with env var support."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Nested sub-configs ────────────────────────────────────────────────────────
#
# These are plain pydantic.BaseModel, NOT BaseSettings. A BaseSettings
# subclass without its own env_prefix reads BARE environment variables —
# TerseOutputSettings() would pick up a plain `LEVEL` or `ENABLED` from
# the host process, so a stray ENABLED=false could silently disable
# compression, memory and terse output at once. They are nested value
# objects: the outer Settings already exposes every field via
# TOKENMIZER_-prefixed, __-nested vars (TOKENMIZER_TERSE_OUTPUT__LEVEL).


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
    """NOT IMPLEMENTED — no code reads any field below.

    Complexity-based model routing was advertised as a pipeline layer
    (the proxy even reports a `savings["routing"]` figure, hardcoded to
    0), but there has never been an implementation: nothing reads
    `enabled`, `simple_model`, `medium_model`, `complex_model`, or
    `complexity_threshold`. Setting `enabled: true` does nothing at all
    and produces no warning.

    The fields are kept so that existing tokenmizer.yaml files carrying a
    `routing:` block still load (Settings uses extra="forbid", so
    deleting them would turn every such config into a hard startup
    failure). get_settings() logs a warning if routing.enabled is true,
    so nobody is left believing a switch did something.
    """
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
    # semantic_cache/cache.py::_is_session_sensitive). Opt in
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
        # "forbid", not "ignore": a misspelled key would otherwise parse
        # cleanly and discard the value with no exception and no log line.
        # Forbidding raises the same way a YAML syntax error does, so a
        # typo hits the same fail-closed path in get_settings() below.
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
    # Additional accepted credentials. Each one is a SEPARATE principal
    # for session-ownership purposes, so different callers get genuinely
    # isolated sessions (see security/ownership.py). With this empty, the
    # deployment is single-tenant: every caller shares one principal and
    # therefore one session namespace — which is the honest description
    # of what a single shared api_key has always meant.
    api_keys: List[str] = []

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Rate limiting: trust X-Forwarded-For?
    #
    # Rate limits key on request.client.host, which behind a load
    # balancer, ingress, or CDN is the PROXY's address — so every client
    # in the world shares one 60/min bucket and one noisy tenant starves
    # everyone. The fix is to read the forwarded client address, but
    # X-Forwarded-For is caller-supplied and trivially spoofed, so
    # trusting it unconditionally would hand every client an unlimited
    # supply of fresh buckets. It must therefore be opt-in, and only
    # enabled when a proxy you control actually overwrites the header.
    trust_proxy_headers: bool = False
    # How many right-hand entries of X-Forwarded-For are your own proxies.
    # With one load balancer this is 1: the last entry is what your LB
    # appended, and the entry before it is the real client.
    trusted_proxy_hops: int = 1

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
        """Load settings from YAML, with environment variables taking
        precedence over the file — the behaviour tokenmizer.yaml's own
        header has always promised ("Environment variables always
        override this file") and that docker-compose.yml depends on.

        The trap this avoids: in pydantic-settings, values passed to
        __init__ are the HIGHEST priority source — above env vars — so a
        plain `cls(**data)` makes the YAML win every conflict. Since the
        shipped tokenmizer.yaml is COPY'd into the Docker image and sets
        provider/state_backend/cors_origins, the matching TOKENMIZER_*
        variables would be inert.

        So: drop from the YAML payload any key the environment also sets,
        letting those fall through to the env source underneath.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"{path} must contain a YAML mapping at the top level, "
                f"got {type(data).__name__}"
            )
        return cls(**_strip_env_overridden(data))


_ENV_PREFIX = "TOKENMIZER_"
_ENV_NESTED_DELIM = "__"


def _strip_env_overridden(data: dict, _prefix: str = _ENV_PREFIX) -> dict:
    """Remove YAML keys that an environment variable also sets, so the env
    value wins. Mirrors the env_prefix / env_nested_delimiter scheme
    declared in Settings.model_config.

    `provider: anthropic` in YAML is dropped when TOKENMIZER_PROVIDER is
    set; `graph_checkpoint.trigger_at_percent` is dropped when either
    TOKENMIZER_GRAPH_CHECKPOINT (the whole object) or
    TOKENMIZER_GRAPH_CHECKPOINT__TRIGGER_AT_PERCENT (the one field) is
    set. Nested dicts are pruned per-key rather than wholesale, so
    setting one nested env var does not discard its YAML siblings.
    """
    import os
    out: dict = {}
    for key, value in data.items():
        env_name = f"{_prefix}{key}".upper()
        if env_name in os.environ:
            continue  # env sets this outright — let the env source supply it
        if isinstance(value, dict):
            pruned = _strip_env_overridden(value, f"{env_name}{_ENV_NESTED_DELIM}")
            if pruned or not value:
                out[key] = pruned
            # if every sub-key was env-overridden, omit the parent entirely
            continue
        out[key] = value
    return out


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
                # A config that fails to load must never fall back
                # silently. The defaults are dev-mode permissive — no API
                # key required, in-memory state — so an operator whose
                # YAML has a typo would otherwise run with none of their
                # security settings applied and no indication why.
                # Logging at error makes it visible at startup; in
                # production, logging is not a safety control, so
                # TOKENMIZER_ENV=production refuses to start instead.
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

        # Warn about settings that are accepted but not implemented, so a
        # config value can never quietly mean nothing. See
        # RoutingSettings' docstring.
        if loaded.routing.enabled:
            logger.warning(
                "routing.enabled is set, but complexity-based model routing "
                "is NOT IMPLEMENTED — no request will be routed differently. "
                "The setting is accepted only so existing config files keep "
                "loading. Remove it to avoid confusion."
            )

        _settings = loaded
    return _settings
