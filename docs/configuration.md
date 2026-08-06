# Configuration

Every setting, where it can be set, and which ones fail loudly. Precedence is **environment → `tokenmizer.yaml` → defaults**.

---

# Configuration

```yaml
# tokenmizer.yaml
provider: anthropic
default_model: claude-sonnet-4-6

graph_checkpoint:
  enabled: true
  trigger_at_percent: 0.85
  use_llm_extraction: false     # true = hybrid LLM+heuristic extraction
                                # (needs a provider key, ~$0.001/turn;
                                # requires v0.3.2+ — see CHANGELOG)

compression:
  enabled: true

cache:
  enabled: true
  max_size: 10000

state_backend: memory           # memory | redis — see note below
```

## Environment

Every setting has an environment variable. **Environment variables
override `tokenmizer.yaml`** — they genuinely do as of v0.5.0; before
that the YAML file silently won every conflict, so any `TOKENMIZER_*`
variable whose key also appeared in the file was ignored.

Precedence, highest first: **environment → `tokenmizer.yaml` → defaults.**

Top-level settings take the prefix directly. Nested ones use a double
underscore for the dot: `graph_checkpoint.trigger_at_percent` becomes
`TOKENMIZER_GRAPH_CHECKPOINT__TRIGGER_AT_PERCENT`. Setting the parent
(`TOKENMIZER_GRAPH_CHECKPOINT`, as JSON) replaces the whole object.

| Variable | Default | What it does |
|---|---|---|
| `TOKENMIZER_CONFIG` | `tokenmizer.yaml` | Path to the config file |
| `TOKENMIZER_ENV` | *(unset)* | `production` refuses to start on an unsafe config — no API key, or a wide-open bind. Anything else is development mode |
| `TOKENMIZER_API_KEY` | *(empty)* | Client auth for TokenMizer itself. Empty = no auth, single shared principal |
| `TOKENMIZER_PROVIDER` | `anthropic` | Upstream provider |
| `TOKENMIZER_DEFAULT_MODEL` | `claude-sonnet-4-6` | Used when the request names no model |
| `TOKENMIZER_<PROVIDER>_API_KEY` | *(empty)* | Upstream key — `ANTHROPIC`, `OPENAI`, `GEMINI`, `GROK`, `DEEPSEEK`, `MISTRAL`, `COHERE`, `OPENROUTER` |
| `TOKENMIZER_PROXY_HOST` | `127.0.0.1` | Bind address. `0.0.0.0` accepts remote connections — set an API key first |
| `TOKENMIZER_TRUST_PROXY_HEADERS` | `false` | Read `X-Forwarded-For` for rate-limit identity. Only enable behind a proxy you control; otherwise callers can forge it |
| `TOKENMIZER_TRUSTED_PROXY_HOPS` | `1` | How many proxies sit in front of you |
| `TOKENMIZER_GRAPH_CHECKPOINT__ENABLED` | `true` | Graph memory on/off |
| `TOKENMIZER_GRAPH_CHECKPOINT__TRIGGER_AT_PERCENT` | `0.85` | Auto-checkpoint at this share of the context window |
| `TOKENMIZER_GRAPH_CHECKPOINT__STORAGE_DIR` | `./checkpoints` | Where the SQLite database lives |
| `TOKENMIZER_GRAPH_CHECKPOINT__MAX_RESUME_TOKENS` | `400` | Budget for the injected resume block |
| `TOKENMIZER_GRAPH_CHECKPOINT__USE_LLM_EXTRACTION` | `false` | Hybrid LLM + heuristic extraction (needs a key, ~$0.001/turn) |
| `TOKENMIZER_CACHE__ENABLED` | `true` | Semantic cache |
| `TOKENMIZER_CACHE__SIMILARITY_THRESHOLD` | `0.92` | How close a hit must be |
| `TOKENMIZER_COMPRESSION__ENABLED` | `true` | Prompt compression |
| `TIKTOKEN_CACHE_DIR` | *(unset)* | Where tiktoken looks for its BPE vocabulary. Set it, and pre-download, to run without egress — the Docker image does this at build time |

Two variables are worth calling out because they fail loudly rather than
quietly: `TOKENMIZER_ENV=production` **refuses to start** on an unsafe
config instead of warning, and `TOKENMIZER_TRUST_PROXY_HEADERS` changes
who the rate limiter thinks you are — enabling it in front of an
untrusted network lets any caller reset their own limit.

> **`state_backend: redis` is not wired up.** `tokenmizer/state/backend.py`
> has no callers — nothing reads from or writes to Redis. All durable
> state (graph memory, checkpoints, session ownership) is SQLite under
> `storage_dir`. The setting is accepted so existing configs keep
> loading; it does not change behaviour.

## Not implemented, despite being configurable

| Setting | Status |
|---|---|
| `routing.*` | No implementation. `savings.routing` is always 0. Setting `enabled: true` logs a warning and changes nothing. |
| `state_backend: redis` | Accepted, unused (see above). |

---

# Supported Providers

Model strings pass through unchanged — the newest models work out of the box:
`claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5`,
GPT-4o/o-series, Gemini 1.5/2.0, and any Ollama/OpenRouter model.

| Provider | Env var |
|---|---|
| Anthropic (Claude) | `TOKENMIZER_ANTHROPIC_API_KEY` |
| OpenAI | `TOKENMIZER_OPENAI_API_KEY` |
| Google Gemini | `TOKENMIZER_GEMINI_API_KEY` |
| DeepSeek | `TOKENMIZER_DEEPSEEK_API_KEY` |
| Mistral | `TOKENMIZER_MISTRAL_API_KEY` |
| Grok (xAI) | `TOKENMIZER_GROK_API_KEY` |
| Cohere | `TOKENMIZER_COHERE_API_KEY` |
| OpenRouter | `TOKENMIZER_OPENROUTER_API_KEY` |
| Ollama | No key — free, local |

---


---

[← Back to the README](../README.md)
