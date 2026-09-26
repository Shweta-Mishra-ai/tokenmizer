# Configuration

Every setting, where it can be set, and which ones fail loudly. Precedence is **environment → `tokenmizer.yaml` → defaults**.

---

## Example config

```yaml
# tokenmizer.yaml
provider: anthropic
default_model: claude-sonnet-4-6

# Send a client's model name somewhere else. Exact match, applied once,
# empty by default. The response carries `tokenmizer.model_mapped_from`
# whenever a substitution happened.
model_map:
  gpt-4: claude-sonnet-4-6
  gpt-3.5-turbo: claude-haiku-4-5

graph_checkpoint:
  enabled: true
  trigger_at_percent: 0.85
  use_llm_extraction: false     # true = hybrid LLM+heuristic extraction on the
                                # same provider and model configured for chat
  extraction_model: ""          # pin a different model of that provider for
                                # extraction; empty = default_model

compression:
  enabled: true

cache:
  enabled: true
  max_size: 10000               # entries
  max_bytes: 268435456          # 256 MiB — the bound that is actually memory
  max_entry_bytes: 1048576      # one answer bigger than this is not cached

state_backend: memory           # memory | sqlite — see note below
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
| `TOKENMIZER_MODEL_MAP` | `{}` | Client model name → the model actually sent upstream, as JSON. Exact match, applied once, reported back in `tokenmizer.model_mapped_from` |
| `TOKENMIZER_STATE_BACKEND` | `memory` | Where the rate limiter's token buckets live: `memory` or `sqlite`. See the note below — `memory` enforces the limit once **per worker** |
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
| `TOKENMIZER_CACHE__SIMILARITY_THRESHOLD` | `0.92` | How close a hit must be. Similarity is necessary but not sufficient: a candidate that clears this bar is still refused when it disagrees with the query on polarity (enable/disable) or on a literal (a number, a filename, a flag, an identifier), because cosine barely moves on exactly those. `rejected_unsafe` in `/api/cache/stats` counts the refusals |
| `TOKENMIZER_CACHE__MAX_SIZE` | `10000` | Cap on cached **entries** |
| `TOKENMIZER_CACHE__MAX_BYTES` | `268435456` | Cap on cached **bytes** (256 MiB). An entry holds a whole response, so the entry cap alone is not a memory bound — 10,000 answers of 60 KB is 579 MiB measured. Whichever bound binds first wins |
| `TOKENMIZER_CACHE__MAX_ENTRY_BYTES` | `1048576` | A single response larger than this is served but not cached, so one huge answer cannot evict the whole cache |
| `TOKENMIZER_CACHE__MAX_SEMANTIC_SCAN` | `2000` | Most-recent entries compared on a cache miss. The semantic layer is an O(n) loop on the request path; this bounds its worst case |
| `TOKENMIZER_REQUEST_TIMEOUT` | `120` | Seconds an upstream call may hang. Every vendor SDK defaults to 600, which on a proxy holds a request, its session lock and its extraction slot for ten minutes. `0` restores the SDK default |
| `TOKENMIZER_COMPRESSION__ENABLED` | `true` | Prompt compression |
| `TOKENMIZER_COMPRESSION__ENGINE` | `heuristic` | `heuristic` or `llmlingua2`. The ML engine needs `pip install tokenmizer[compression]`; without it the heuristic runs and says so |
| `TOKENMIZER_COMPRESSION__MIN_TOKENS_TO_COMPRESS` | `300` | Below this a prompt is passed through untouched — compressing a short prompt costs more than it saves |
| `TOKENMIZER_DOMAIN` | `coding` | Which vocabulary the extractor runs: `coding`, `research`, `ops` or `product`. Adds phrasings for that domain; `coding` is what has always run. See [benchmarks.md](https://github.com/Shweta-Mishra-ai/tokenmizer/blob/main/docs/benchmarks.md) for each pack's measured numbers |
| `TOKENMIZER_API_KEYS` | *(empty)* | Extra accepted credentials, as JSON. **Each one is a separate principal**, so callers get isolated sessions. With this empty the deployment is single-tenant: every caller shares one principal and one session namespace |
| `TOKENMIZER_CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8000"]` | Browser origins allowed to call the proxy, as JSON. Widen it only to origins you control — a wildcard lets any page a user visits spend their key |
| `TOKENMIZER_PROXY_PORT` | `8000` | Port the CLI's `serve` binds to, alongside `TOKENMIZER_PROXY_HOST` |
| `TOKENMIZER_MEMORY__MAX_TOKENS_BEFORE_SUMMARY` | `4000` | Conversation size at which older turns are summarised rather than carried verbatim |
| `TOKENMIZER_MEMORY__RECENT_TURNS_VERBATIM` | `10` | How many recent turns always survive windowing untouched |
| `TOKENMIZER_GRAPH_CHECKPOINT__MIN_CONFIDENCE` | `0.65` | Extraction confidence below which a fact is not written to the graph |
| `TOKENMIZER_GRAPH_CHECKPOINT__SEMANTIC_RETRIEVAL` | `auto` | Rank injected context by embeddings. `auto` = on when the model actually loads, resolved once at startup; `true`/`false` pin it |
| `TOKENMIZER_GRAPH_CHECKPOINT__CROSS_SESSION_RECALL` | `false` | Let a session retrieve facts from the same principal's other sessions |
| `TOKENMIZER_CACHE__TTL_SECONDS` | `3600` | How long a cached answer stays valid |
| `TOKENMIZER_CACHE__SHARE_SCOPE` | `session` | `session` scopes every cached prompt to its session. `shared` lets non-sensitive prompts cross sessions for a higher hit rate — read the sensitivity heuristic in `semantic_cache/cache.py` before enabling it |
| `TOKENMIZER_COMPRESSION__RATIO` | `0.5` | Target share of the prompt to keep. Only the `llmlingua2` engine honours it; the heuristic engine drops what it can prove is safe to drop and ignores a ratio |
| `TOKENMIZER_MEMORY__ENABLED` | `true` | Conversation windowing and summarisation. Off means every turn is sent verbatim |
| `TOKENMIZER_GRAPH_CHECKPOINT__EXTRACTION_MODEL` | *(empty)* | Pin a different model **of the configured provider** for extraction — usually a cheaper one. Empty uses `default_model` |
| `TOKENMIZER_PREFERENCES__ENABLED` | `false` | Cross-session preference memory. Off by default on purpose — see the section below before enabling it |
| `TOKENMIZER_PREFERENCES__MAX_ITEMS` | `4` | Preference lines injected into the system prompt |
| `TOKENMIZER_PREFERENCES__MAX_CHARS` | `400` | Total size of those lines |
| `TOKENMIZER_TERSE_OUTPUT__ENABLED` | `true` | Inject the output-style instruction at all |
| `TOKENMIZER_TERSE_OUTPUT__LEVEL` | `full` | How hard `lite`/`full`/`ultra` push, both in the injected instruction and in how much boilerplate the response trimmer strips. Ignored for the injected prompt when `style: minimal`, which is a single fixed instruction |
| `TOKENMIZER_TERSE_OUTPUT__STYLE` | `terse` | Output-style prompt injected each turn: `terse`, `minimal`, or `off` |
| `TIKTOKEN_CACHE_DIR` | *(unset)* | Where tiktoken looks for its BPE vocabulary. Set it, and pre-download, to run without egress — the Docker image does this at build time |

Two variables are worth calling out because they fail loudly rather than
quietly: `TOKENMIZER_ENV=production` **refuses to start** on an unsafe
config instead of warning, and `TOKENMIZER_TRUST_PROXY_HEADERS` changes
who the rate limiter thinks you are — enabling it in front of an
untrusted network lets any caller reset their own limit.

> **Set `state_backend: sqlite` if you run more than one worker.** The
> rate limiter keeps its token buckets in process memory by default,
> which means `--workers 4` enforces your configured limit four times
> over — 60 requests a minute becomes 240. `sqlite` puts the buckets in
> `storage_dir`, where every worker on the host shares one count, at the
> cost of one small transaction per request. Neither option spans hosts:
> several machines behind a load balancer need the limit at the load
> balancer.
>
> **`state_backend: redis` was never implemented.** Nothing has ever read
> it, so it behaves as `memory`. The value is still accepted so existing
> configs load, and now logs a warning naming `sqlite` as the option that
> covers the same deployment.

## Preferences — habits that outlive a session

```yaml
preferences:
  enabled: false     # OFF by default; read the note
  max_items: 4       # lines injected into the system prompt
  max_chars: 400     # and their total size
```

The session graph remembers what you decided about a project. This
remembers what is true of *you* — "keep it brief", "always TypeScript" —
per principal, in `storage_dir`, and injects a few lines into the system
prompt.

> **Off by default on purpose.** The failure mode of a preference memory
> is not forgetting; it is remembering something that was never a
> preference and repeating it in every prompt you send for the rest of
> the year. The detector is a set of regexes and it will have false
> positives. `GET /api/preferences` shows exactly what is remembered and
> the exact text injected; `DELETE /api/preferences` (optionally
> `?key=...`) forgets it. Secrets, env vars and complaints ("I hate this
> bug") are excluded by construction, but do not mistake that for a
> guarantee.

## Not implemented, despite being configurable

| Setting | Status |
|---|---|
| `routing.*` | **Deprecated, removed next release.** Never implemented. Replaced by `model_map`. A config carrying the block still loads and logs a deprecation warning. |
| `state_backend: redis` | Accepted, never implemented; behaves as `memory` and warns. Use `sqlite` (see above). |

---

## Supported Providers

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

[← Back to the README](../README.md)
