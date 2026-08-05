<div align="center">
  <img src="docs/assets/logo.svg" width="140" alt="TokenMizer"/>

  <h1>TokenMizer</h1>

  <p><strong>Your AI forgets why. TokenMizer remembers.</strong></p>

  <p>
    An OpenAI-compatible proxy that builds a <b>knowledge graph</b> of your
    session — decisions, files, errors, goals — and replays it when the<br/>
    context window runs out. Not a summary: a queryable graph that knows
    <i>"we switched from MongoDB to PostgreSQL, and here is why."</i>
  </p>

  <p>
    <sub>One line to adopt · works with Claude, GPT, Gemini, Grok, DeepSeek, Mistral, Cohere, Ollama · MIT</sub>
  </p>

  <p>
    <a href="https://pypi.org/project/tokenmizer"><img src="https://img.shields.io/pypi/v/tokenmizer?color=7c6af7&style=flat-square" alt="PyPI"/></a>
    <a href="https://pypi.org/project/tokenmizer"><img src="https://img.shields.io/pypi/dm/tokenmizer?color=5ee7c8&style=flat-square" alt="Downloads"/></a>
    <a href="https://github.com/Shweta-Mishra-ai/tokenmizer/actions"><img src="https://img.shields.io/github/actions/workflow/status/Shweta-Mishra-ai/tokenmizer/ci.yml?branch=main&style=flat-square&color=4ade80" alt="CI"/></a>
    <a href="https://registry.modelcontextprotocol.io/v0/servers?search=tokenmizer"><img src="https://img.shields.io/badge/MCP%20Registry-published-5ee7c8?style=flat-square" alt="MCP Registry"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4ade80?style=flat-square"/></a>
    <a href="https://github.com/Shweta-Mishra-ai/tokenmizer/stargazers"><img src="https://img.shields.io/github/stars/Shweta-Mishra-ai/tokenmizer?style=flat-square&color=f9d84a" alt="Stars"/></a>
    <a href="https://glama.ai/mcp/servers/Shweta-Mishra-ai/tokenmizer"><img src="https://glama.ai/mcp/servers/Shweta-Mishra-ai/tokenmizer/badges/score.svg" alt="Glama Score"/></a>
  </p>

  <p>
    <a href="#quick-start"><b>Quick Start</b></a> ·
    <a href="#how-tokenmizer-solves-it"><b>How it works</b></a> ·
    <a href="#benchmarks"><b>Benchmarks</b></a> ·
    <a href="#durability--what-happens-when-something-breaks-mid-session"><b>Durability</b></a> ·
    <a href="#claude-code-integration"><b>Claude Code</b></a> ·
    <a href="#contributing"><b>Contributing</b></a>
  </p>

  <img src="docs/assets/demo.gif" width="860" alt="TokenMizer demo: 40-turn session checkpointed at 87% context, resumed next day in 233 tokens"/>
  <br/>
  <sub>Real run: 25-node graph, checkpoint <code>ckpt_21a0959c3ddf</code>, 233-token resume. Regenerate with <code>python scripts/gen_demo_gif.py</code>.</sub>
</div>

---

## The Problem

Every AI session has a context limit. When you hit it:

- The model forgets every decision, rationale, and context built over hours
- You waste 10–30 minutes re-explaining the project every new session
- Large files (CSV, PDF, Excel) eat your entire token budget instantly

## How TokenMizer Solves It

TokenMizer is a **local proxy** between your app and any LLM. Every
request passes through a pipeline that builds a live knowledge graph,
compresses inputs, caches responses, and auto-checkpoints before context
runs out.

```mermaid
flowchart LR
    App["Your app<br/><sub>OpenAI-compatible client</sub>"]
    subgraph TM["TokenMizer :8000"]
        direction TB
        L0["<b>L0</b> File intelligence<br/><sub>CSV · PDF · Excel · JSON → schema + sample</sub>"]
        L1["<b>L1</b> Prompt compression<br/><sub>heuristics; code blocks passed through untouched</sub>"]
        L2["<b>L2</b> Terse-output injection"]
        L4["<b>L4</b> Graph memory<br/><sub>extract → window → inject context</sub>"]
        L3["<b>L3</b> Semantic cache<br/><sub>session-scoped by default</sub>"]
        L5["<b>L5</b> Provider prompt cache<br/><sub>Anthropic, prefixes ≥1024 tokens</sub>"]
        L0 --> L1 --> L2 --> L4 --> L3 --> L5
    end
    LLM["Claude · GPT · Gemini<br/>Grok · DeepSeek · Ollama"]
    DB[("SQLite<br/><sub>graph · checkpoints · ownership</sub>")]

    App -->|"POST /v1/chat/completions"| TM
    TM --> LLM
    LLM -.->|response| TM
    TM -.->|"response + savings"| App
    L4 <-->|"per-row, locked"| DB
```

> **On the layer numbering:** `savings.routing` appears in API responses
> and is always `0`. Complexity-based model routing is **not
> implemented** — see [Not implemented, despite being configurable](#not-implemented-despite-being-configurable).

---

## Architecture

<div align="center">
  <img src="docs/assets/architecture.svg" width="880" alt="TokenMizer architecture: proxy pipeline, graph memory, and SQLite storage"/>
</div>

### What happens on one request

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant P as Proxy
    participant G as GraphMemory
    participant S as SQLite
    participant L as LLM

    C->>P: POST /v1/chat/completions (session_id)
    P->>P: rate limit · auth · session ownership
    P->>P: redact secrets once, at ingestion
    P->>P: L0-L2 file intel · compress · terse
    P->>G: extract nodes from new messages
    G->>S: persist changed rows only (locked)
    G-->>P: relevant context block
    P->>L: compressed messages + graph context
    L-->>P: completion
    alt context >= trigger_at_percent
        P->>S: auto-checkpoint (retry once, report outcome)
    end
    P-->>C: completion + usage + tokenmizer.savings
```

### Decision lifecycle

The graph tracks *why* the current answer is current. A new decision in
an occupied slot supersedes the old one and records the transition, so
`/api/graph/{id}/why` can replay the chain.

```mermaid
stateDiagram-v2
    [*] --> COMPLETED: decision extracted
    COMPLETED --> SUPERSEDED: replaced by a newer decision
    COMPLETED --> CONTESTED: same topic, purpose unclear
    COMPLETED --> INVALIDATED: explicitly rejected
    CONTESTED --> COMPLETED: ambiguity resolved
    SUPERSEDED --> ARCHIVED: after 7 days
    ARCHIVED --> [*]: prunable

    note right of SUPERSEDED
        Kept in history, hidden from resume.
        The transition records trigger,
        reason and evidence.
    end note
    note right of INVALIDATED
        Surfaced in resume as "DO NOT REVISIT"
        so the model does not re-propose it.
    end note
```

### Decision Memory — 4-State Model

| Status | Meaning | In Resume |
|---|---|---|
| 🟢 `ACTIVE` | Current — in effect | ✅ Always |
| 🟡 `SUPERSEDED` | Replaced by newer decision | ⚠️ 7 days |
| 🔴 `INVALIDATED` | Explicitly wrong/cancelled | ⚠️ Always (warning) |
| ⬜ `ARCHIVED` | Superseded >7 days ago — aged out | ❌ Never |

History is **never deleted**. "Why did we switch from React to Next.js?" — always answerable:
ask `GET /api/graph/{session}/why?q=react` (or the `why_decision` MCP tool) and get the full
old → new trail with trigger, reason, and evidence per hop.

### From Storage to Reasoning

The graph doesn't just store facts — it answers questions over them:

| Capability | Endpoint / Tool | What it answers |
|---|---|---|
| **Ontology** | `GET /api/ontology` | The formal vocabulary: node/edge types with semantics, and the status state machine (which lifecycle transitions are legal) |
| **Causal chains** | `GET /api/graph/{id}/why?q=...` · MCP `why_decision` | "Why is X the current choice?" — walks the supersession chain with trigger/reason/evidence per hop |
| **Reasoning view** | `GET /api/graph/{id}/reasoning` | Active decisions per topic, recent changes, decision timeline, and a consistency audit |
| **Consistency audit** | (part of `/reasoning`) | Contradictions the tracker missed, superseded decisions with lost history, dangling references |

All reasoning is deterministic and local — no LLM calls, no extra cost.

---

## Quick Start

<details>
<summary><b>🟢 Complete step-by-step setup (start here if you're new — 5 minutes, no code reading needed)</b></summary>

<br/>

**Step 0 — Check Python** (need 3.10 or newer)

Open a terminal (Windows: press Win, type "PowerShell", Enter · Mac: Cmd+Space, type "Terminal"):

```
python --version
```

You should see `Python 3.10` or higher. If not: install from [python.org/downloads](https://python.org/downloads) (Windows: tick **"Add Python to PATH"** during install).

**Step 1 — Install TokenMizer**

```
pip install "tokenmizer[anthropic,cache]"
```

✅ You should see: `Successfully installed tokenmizer-...`

**Step 2 — Add your API key** (get one at [console.anthropic.com](https://console.anthropic.com) → API Keys)

Windows PowerShell:
```powershell
setx TOKENMIZER_ANTHROPIC_API_KEY "sk-ant-YOUR-KEY"
```
then **close and reopen** the terminal.

Mac/Linux:
```bash
export TOKENMIZER_ANTHROPIC_API_KEY=sk-ant-YOUR-KEY
```

*(No key? Use free local Ollama instead — see "No API key?" below.)*

**Step 3 — Start TokenMizer**

```
tokenmizer serve
```

✅ You should see: `Proxy: http://localhost:8000/v1/chat/completions`
Leave this terminal open — TokenMizer runs here.

**Step 4 — Verify it's alive**

Open [http://localhost:8000](http://localhost:8000) in your browser → the TokenMizer dashboard appears. That's it — the proxy works.

**Step 5 — Connect your tool** (pick yours)

- **Cursor:** Settings → Models → OpenAI API → Base URL: `http://localhost:8000/v1`
- **Claude Desktop / Claude Code:** see [Claude Code Integration](#claude-code-integration) below (copy one JSON block, restart the app)
- **Your own Python code:** see "Use — change one line" below

**Something failed?** `pip` not found → reinstall Python with "Add to PATH". Port 8000 busy → `tokenmizer serve --port 8001`. Anything else → [open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues) with the error text — median response < 1 day.

</details>

### 1. Install

Works on **Windows, macOS, and Linux** (Python 3.10+). Same command everywhere:

```bash
# Recommended
pip install "tokenmizer[anthropic,cache]"

# All providers
pip install "tokenmizer[anthropic,openai,gemini,cohere,cache]"
```

<details>
<summary><b>No API key? Use Ollama (free, local)</b></summary>

```bash
# macOS:   brew install ollama
# Windows: winget install Ollama.Ollama   (or download from ollama.com)
# Linux:   curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3
pip install tokenmizer
# then set provider: ollama in tokenmizer.yaml
```
</details>

### 2. Set your API key

**macOS / Linux (bash, zsh):**
```bash
export TOKENMIZER_ANTHROPIC_API_KEY=sk-ant-...
```

**Windows (PowerShell):**
```powershell
$env:TOKENMIZER_ANTHROPIC_API_KEY = "sk-ant-..."      # current session
setx TOKENMIZER_ANTHROPIC_API_KEY "sk-ant-..."         # persistent (new terminals)
```

Other providers: `TOKENMIZER_OPENAI_API_KEY`, `TOKENMIZER_GEMINI_API_KEY`, etc. — full table in [Supported Providers](#supported-providers).

### 3. Start

```bash
tokenmizer serve
# → Proxy:     http://localhost:8000/v1/chat/completions
# → Dashboard: http://localhost:8000
# → API docs:  http://localhost:8000/docs
```

### 4. Use — change one line

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1",  # ← only this changes
)

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Let's build an auth service"}],
    extra_body={"session_id": "my-project"},  # enables graph memory
)
```

> ✅ **Streaming works** (v0.3+): `stream: true` gives real SSE passthrough for
> Anthropic, OpenAI, DeepSeek, Mistral, OpenRouter, Grok and Ollama. Cursor and
> Continue.dev work with default settings — no config changes needed.

---

## Claude Code Integration

### Option A — Plugin (recommended)

```bash
# Add TokenMizer as a plugin marketplace
/plugin marketplace add Shweta-Mishra-ai/tokenmizer

# Install
/plugin install tokenmizer@Shweta-Mishra-ai/tokenmizer
```

Then use skills directly:

```
/tokenmizer:checkpoint my-project      → save session to graph memory
/tokenmizer:resume my-project          → load previous session (300 tokens)
/tokenmizer:resume my-project full     → full 600-token context
/tokenmizer:analyze /data/sales.csv    → analyze file (99% token savings)
/tokenmizer:stats                      → token savings report
```

### Option B — MCP server (Claude Desktop, Claude Code, Cursor, VS Code, Zed)

mcp-name: io.github.Shweta-Mishra-ai/tokenmizer

Add this `mcpServers` block to your client's MCP config file:

```json
{
  "mcpServers": {
    "tokenmizer": {
      "command": "tokenmizer-mcp",
      "env": { "TOKENMIZER_URL": "http://localhost:8000" }
    }
  }
}
```

Where the config file lives:

| Client | Config file |
|---|---|
| **Claude Desktop** (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Claude Desktop** (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude Code** | `.mcp.json` in your project, or `~/.claude/settings.json` |
| **Cursor** | Settings → MCP → Add server (same JSON) |
| **VS Code / Zed** | their MCP settings — same `command` + `env` |
| **OpenAI Codex CLI** | `~/.codex/config.toml` — TOML format, see below |

<details>
<summary>Codex CLI config (TOML, not JSON)</summary>

```toml
[mcp_servers.tokenmizer]
command = "tokenmizer-mcp"
env = { TOKENMIZER_URL = "http://localhost:8000" }
```
</details>

Then restart the client. Keep `tokenmizer serve` running for the
checkpoint/resume/stats/reasoning tools (file analysis works without it).
If `tokenmizer-mcp` isn't on your PATH, use `"command": "python"`,
`"args": ["-m", "tokenmizer.mcp.server"]` instead.

**Tools exposed (6):** `checkpoint_session`, `resume_session`,
`get_graph_stats`, `analyze_file`, `get_savings_stats`, and
`why_decision` — ask your agent *"why did we pick X?"* and it traces the
decision's supersession chain with reasons and evidence.

---

## Other Tools

**Cursor / Continue.dev / any OpenAI-compatible tool:**
```
API Base URL:  http://localhost:8000/v1
```

---

## Session Resume

```bash
tokenmizer checkpoint my-project
tokenmizer resume my-project
```

```
Goal: Build FastAPI auth service with JWT + PostgreSQL
Done: Project setup | User model | Login endpoint | Fix 422 | 18 tests passing
In progress: Refresh token rotation
Decided: PostgreSQL (concurrent writes) | bcrypt | Redis for refresh tokens
Changed: ~~React~~ → Next.js (better SEO)
Files: api/auth.py, api/models.py, config.py
Continue: Implement token refresh endpoint
```

**247 tokens** replaces **25,000+ tokens** of conversation history.

---

## File Intelligence

```python
from tokenmizer.filters.file_intelligence import FileIntelligence

fi = FileIntelligence()
result = fi.process(open("sales.csv","rb").read(), "sales.csv",
                    token_budget=500, query="which regions underperforming")
# 412,000 tokens → 447 tokens  (99.9% saved)
```

| File | Savings |
|---|---|
| CSV (50k rows) | 99.9% |
| PDF (200 pages) | 98.8% |
| Excel (10 sheets) | 99.7% |
| JSON (1k items) | 95% |

---

## Works Alongside Caveman & CodeBurn

TokenMizer **complements** — does not replace — these tools:

| Tool | What it does |
|---|---|
| **Caveman** | Output tokens shorter (~65%) |
| **CodeBurn** | Input context trimming |
| **TokenMizer** | Graph memory + resume + file intelligence + cache |

> **Tip:** If using Caveman, set `terse_output: enabled: false` in `tokenmizer.yaml` to avoid conflicting system prompts.

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

## Configuration

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

All settings via env vars: `TOKENMIZER_PROVIDER`, `TOKENMIZER_API_KEY`, etc.
**Environment variables override `tokenmizer.yaml`.** (They genuinely do
as of v0.4.1 — before that the YAML file silently won every conflict, so
any `TOKENMIZER_*` variable whose key also appeared in the file was
ignored.)

> **`state_backend: redis` is not wired up.** `tokenmizer/state/backend.py`
> has no callers — nothing reads from or writes to Redis. All durable
> state (graph memory, checkpoints, session ownership) is SQLite under
> `storage_dir`. The setting is accepted so existing configs keep
> loading; it does not change behaviour.

### Not implemented, despite being configurable

| Setting | Status |
|---|---|
| `routing.*` | No implementation. `savings.routing` is always 0. Setting `enabled: true` logs a warning and changes nothing. |
| `state_backend: redis` | Accepted, unused (see above). |

---

## Docker

```bash
# Quick start
docker-compose up tokenmizer

# With a provider key
ANTHROPIC_API_KEY=sk-ant-... docker-compose up

# With proxy auth
TOKENMIZER_API_KEY=strong-key docker-compose up
```

`stop_grace_period` is set to 30s because SIGTERM triggers a shutdown
flush (see [Durability](#durability--what-happens-when-something-breaks-mid-session));
cutting it short is what loses data.

### Running more than one worker

Graph and checkpoint **writes are safe across processes**: storage is
per-row and every persist is a read-modify-write under an OS-level file
lock (`fcntl` / `msvcrt`), so concurrent writers merge and a stale
writer adopts another's deletions instead of reinstating them. Measured
lossless with 4 processes writing one session — see
[Benchmarks](#storage--schema-v2-per-row).

What is **still per-process**, and therefore differs per worker:

| Component | Consequence of >1 worker |
|---|---|
| Rate limiter | Limits apply per worker — the effective limit is ~N× what you configured |
| Analytics | `/api/stats` reflects only the worker that served the request |
| Semantic cache | Cache hit rate drops; each worker warms its own |
| Graph LRU cache | A session's in-memory copy can be briefly stale between writes |

The image ships `--workers 1` for that reason. If you raise it, put a
real rate limiter in front and treat `/api/stats` as per-worker.

> `flock` is unreliable on NFS. Keep `storage_dir` on a local
> filesystem if you run more than one process against it.

---

## Durability — what happens when something breaks mid-session

The point of this tool is that you don't lose context. That has to hold
when things go wrong mid-session, not just when they go right.

| Failure | Behaviour |
|---|---|
| **Graceful stop** (SIGTERM: `docker stop`, k8s rollout, `systemctl restart`) | In-flight background extraction is drained (up to 10s), then every cached graph is force-persisted before exit. Nothing accepted is lost. |
| **Hard kill** (SIGKILL, OOM, power loss) | A background flush runs every `FLUSH_INTERVAL_SECONDS` (30s), so the worst case is bounded to the last 30s of graph activity. Anything already checkpointed is unaffected. |
| **Database refuses writes** | A graph that fails to persist is **kept in memory** instead of being evicted, so nothing is dropped before it has been saved. The cache deliberately runs over its cap until the write succeeds. |
| **Cache eviction during a request** | Sessions with an in-flight request are never selected for eviction, so a request cannot have its graph persisted and detached out from under it. |
| **Transient DB lock** (`database is locked`) | Treated as contention, not corruption: nothing is deleted, and a graph that failed to *read* refuses to *write* over the stored row rather than replacing it with an empty one. |
| **Corrupt database file** | The file is **quarantined by rename** (`graph_memory.db.corrupt-<timestamp>`), never deleted, and recovery is scoped to the affected session's row where possible. Recover with `sqlite3 <quarantined> '.recover' \| sqlite3 graph_memory.db`. |
| **One corrupt row** | Costs that node or edge only — the rest of the session loads normally. |
| **Anything was actually lost** | Surfaced as `data_loss_detected` in `GET /api/graph/{id}` and as `persist_failures` in `GET /api/stats` — queryable, not just a log line. |

Both SQLite databases are shared by every session in a `storage_dir`,
which is why "delete the file and start fresh" is never the recovery
path: it would discard every other session too.

### Storage layout

Graph state is stored **one row per node and per edge** (`graph_nodes`,
`graph_edges`, `graph_meta`). A persist writes only what changed —
adding one node to a 151-node graph writes 1 row, and a turn that changes
nothing writes none.

Databases written before v0.4.2 used a single JSON blob per session.
They are migrated automatically the first time each session is opened,
one session at a time. The old row is kept, not deleted, so a downgrade
still finds the data it expects as of the moment of migration — changes
made after upgrading are lost if you roll back, which is what a rollback
means. Nothing needs to be run by hand.

## Session isolation

A session is claimed by the first API key that uses it, and only that key
can read or modify it afterwards (`GET /api/graph/{id}`, `/api/resume/{id}`,
`POST /api/checkpoint`, `POST /api/decision/invalidate`, and the chat
endpoint itself). Requests for someone else's session return 404 rather
than 403, so the endpoints can't be used to probe which session names
exist.

```yaml
api_key: primary-key            # the deployment credential
api_keys:                       # additional credentials...
  - second-team-key             # ...each of which is its own principal
```

* **No key configured (default):** every caller is the same principal —
  local single-user use, unchanged.
* **One key:** every caller is the same principal. This is a
  **single-tenant** deployment: everyone holding the key shares one
  session namespace.
* **Multiple keys:** sessions are genuinely isolated per key.

Since `session_id` is chosen by the client, don't treat it as a secret;
isolation comes from the credential, not from the id being hard to guess.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy |
| `/api/resume/{id}` | GET | Get resume context |
| `/api/checkpoint` | POST | Manual checkpoint |
| `/api/decision/invalidate` | POST | Mark decision as invalid |
| `/api/graph/{id}` | GET | Session graph stats |
| `/api/graph/{id}/html` | GET | **Interactive graph page** — decision-history timeline, supersession arcs, type/status filters, search, zoom/pan, PNG export. Zero external dependencies (works offline) |
| `/api/graph/{id}/why?q=` | GET | **Reasoning:** causal chain behind a decision (old → new with trigger/reason/evidence) |
| `/api/graph/{id}/reasoning` | GET | **Reasoning view:** active decisions by topic, recent changes, consistency audit |
| `/api/ontology` | GET | Machine-readable graph ontology (types, relations, status state machine) |
| `/api/stats` | GET | Token savings analytics |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## Security

- API key auth — `TOKENMIZER_API_KEY` (constant-time comparison)
- Secret/PII redaction applied once at ingestion, before graph storage,
  checkpoint storage, and every LLM call (main chat and the background
  extraction model). Patterns cover Anthropic/OpenAI/Google/GitHub/AWS/
  Slack/Stripe/JWT/OpenRouter/HF/xAI keys, URL-embedded credentials
  (`postgres://user:pass@host`), and generic `key=`/`password=`
  assignments. Best-effort by nature — an unrecognized format with no
  keyword context can still slip through. The checkpoint layer
  independently re-redacts what it persists (defense in depth).
- Session-isolated cache (sensitive data never shared across sessions)
- Basic prompt-injection keyword filter — catches copy-pasted jailbreak
  templates only; **not** a security boundary against a motivated
  adversary. See [SECURITY.md](SECURITY.md#prompt-injection-basic-keyword-filter-read-the-scope)
  for exactly what it does and doesn't catch.
- CORS restricted to configured origins by default

---

## Benchmarks

Every number below comes from a committed runner you can execute
yourself. Nothing here is hand-written; if a figure and a runner
disagree, the runner is right and the README is a bug.

```bash
python -m benchmarks.eval                            # extraction P/R/F1
python -m benchmarks.eval --errors                   # every miss, every false positive
python -m benchmarks.eval --corpus DIR               # score YOUR sessions
python -m benchmarks.checkpoint_accuracy.runner_v2   # graph vs summary
python -m benchmarks.persistence.runner              # storage + concurrency
pytest tests/ -q                                     # 521 tests
```

### Extraction quality — precision, recall and F1

`python -m benchmarks.eval` scores extraction against a labelled corpus:
**8 sessions, 82 turns, 97 labelled items, 8 domains** (Go, Rust, Python,
TypeScript, React, SQL, CI, ML). Measured on v0.6.0:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Files | 92% | 91% | **91%** |
| Errors | 83% | 92% | **87%** |
| Completed tasks | 67% | 86% | **75%** |
| Pending tasks | 50% | 86% | **63%** |
| Decisions | 50% | 74% | **60%** |
| | | **macro F1** | **75%** |

**Precision is reported, not just recall.** An extractor that emits the
whole transcript as one node scores 100% recall; that is why recall-only
extraction numbers should be distrusted, including earlier ones of ours.

Label quality, scored separately because a correct-but-sprawling label
still wastes resume budget: 8% truncated mid-word, 5% spanning more than
one sentence, mean length 32 characters.

Decisions are the weakest category on precision — the five decision
passes over-fire, producing roughly one spurious decision for each real
one. That is the next thing to fix, and it is visible here rather than
hidden behind an average.

**This corpus is entirely synthetic.** Eight hand-written sessions are a
directional signal, not a claim about your workload. To get a number that
describes yours, label a few of your own sessions in the documented
format and run `python -m benchmarks.eval --corpus /path/to/them`.

### Memory quality — graph vs a plain summary

`benchmarks/checkpoint_accuracy/runner_v2.py`, n=3 synthetic sessions:
the graph preserves **89%** of labelled information against **79%** for a
plain-summary baseline (Δ +10%), in an average resume block of **249
tokens** versus ~1,500+ tokens of raw history. The advantage is
concentrated in decision recall; on tasks it ties the baseline.

### Storage — schema v2 (per-row)

`benchmarks/persistence/runner.py`, measured on v0.5.0:

| Metric | v1 (one blob per session) | v2 (per-row) |
|---|---|---|
| Rows written to add 1 node to a 50-node graph | 51 | **1** (−98.0%) |
| …to a 100-node graph | 101 | **1** (−99.0%) |
| …to a 200-node graph | 201 | **1** (−99.5%) |
| Rows written when a turn changes nothing | 100 | **0** |

Persist latency, one added node on a 200-node graph: **median 9.3 ms,
p95 11.7 ms**.

Concurrency (4 OS processes writing one session, 25 nodes each):
**100/100 nodes persisted, zero lost.** A stale writer holding a
pre-prune view of the graph no longer reinstates the rows another worker
deleted.

Enable `use_llm_extraction: true` for hybrid extraction (LLM + heuristic merge).

**On LLM/hybrid recall numbers — read this before trusting any percentage
here:** earlier versions of this README quoted "90-100% hybrid recall"
sourced from `runner_v3.py`'s `MockLLMProvider`. That mock sampled its
fake output directly from the same ground-truth dict used to *score*
recall — circular by construction, guaranteed to look good regardless of
what the real extraction logic did. It measured nothing about actual LLM
extraction quality. That number has been removed rather than replaced
with a better-sounding one we can't back up.

What `runner_v3.py` now actually does:
- **Default mode** verifies `HybridExtractor.merge()`'s logic contract
  against fixtures with deliberately known overlap (corroborated /
  LLM-only / heuristic-only items) — confirms merge never drops an item
  either source found, and applies confidence tiers (0.95 corroborated,
  0.80 LLM-only, 0.65 heuristic-only) correctly. This is a real,
  non-circular check, but it's a logic-contract test, not a recall
  measurement.
- **`--live` mode** calls a real configured provider (`ANTHROPIC_API_KEY`
  or `OPENAI_API_KEY`) and scores its actual output against ground truth.
  This is the only path that produces a number meaningful enough to put
  in a table. Run it yourself — we're not publishing a live-mode number
  here because n=3 sessions is too small a sample to generalize, and
  publishing one without a large, ongoing benchmark would just be
  swapping one unsubstantiated number for another.

Heuristic-only numbers above (76-100%) ARE real, deterministic,
reproducible measurements — `runner_v2.py` runs actual heuristic
extraction against actual ground truth with no LLM and no mocking
involved, which is why those numbers are presented with confidence
and the LLM ones currently are not.

---

## Why TokenMizer and not X?

Engineers ask this every time. Honest answers:

**Why not just use Git history?**
Git stores *what changed*, not *why you decided to change it*. You can't ask Git "what did we decide about auth?" or "why did we switch from MySQL to PostgreSQL?" TokenMizer stores decisions with trigger, reason, and evidence — not diffs.

**Why not RAG (retrieval-augmented generation)?**
RAG retrieves *relevant chunks* — it doesn't model *decision state*. If you switched from bcrypt to Argon2 mid-session, RAG might retrieve both and confuse the model about which is current. TokenMizer tracks decision supersession explicitly: old decision is marked `SUPERSEDED`, new decision is `ACTIVE`. Resume context only includes current state.

**Why not a plain summary at the start of each session?**
Summaries lose structure. You can't query "all superseded decisions" or "what triggered the auth change" from a blob of text. Our benchmark shows graph memory preserves +5% more information than a summary baseline — and unlike summaries, the graph is queryable, editable, and grows incrementally without re-summarizing everything each turn.

**Why not Mem0 or Zep?**
Mem0 and Zep store *facts* ("user prefers Python"). TokenMizer stores *decisions with rationale* — the full causal chain: what was decided, what replaced it, why, what evidence triggered the change, and how confidence shifted. If you need "remember my name across sessions," use Mem0. If you need "remember that we switched from PostgreSQL to SQLite because of cost, and here's the evidence," use TokenMizer.

**Why not just a longer context window?**
Longer context = higher cost + slower inference + model attention dilution on long histories. TokenMizer compresses a 50-turn session into ~246 tokens of structured context — not by summarizing, but by extracting what actually matters: goals, active decisions, current tasks, recent errors.

---

## CLI

```bash
tokenmizer serve [--port 8000]
tokenmizer checkpoint <session-id>
tokenmizer resume <session-id> [--level standard|full|critical]
tokenmizer stats
```

> **Note on file analysis:** `/tokenmizer:analyze` (used from inside Claude
> Code, see [Claude Code Integration](#claude-code-integration) above) is
> real and works — it's a plugin skill (`.claude-plugin/skills/analyze/`)
> that calls `FileIntelligence` directly via an inline Python snippet,
> independent of the CLI/API layer. What does **not** exist is a bare
> `tokenmizer analyze <file>` terminal command or a `/api/analyze` HTTP
> endpoint — useful if you want file analysis from a plain shell or a
> non-Claude-Code tool (Cursor, a script, curl, etc.) rather than inside
> Claude Code specifically. Found during a documentation accuracy pass:
> an earlier version of this README listed `tokenmizer analyze <file>` in
> this CLI section as if it were a `cli.py` command — it never was.
> Removed from here rather than left in place pointing at something that
> would fail. Tracked as a real, wanted gap — contributions adding a
> `/api/analyze` endpoint + thin CLI wrapper (following the existing
> pattern in `cli.py`) are welcome.

---

## Roadmap

| Version | Focus |
|---|---|
| **v0.3** | SSE streaming passthrough (checkpoint on stream close) |
| **v0.4** | Graph ontology · deterministic reasoning API (`why`, `impact`, consistency checks) |
| **v0.5** | Per-row storage schema · cross-process write safety · session ownership · durability guarantees |
| v0.6 | Cross-session memory · embedding-based edge linking · a real precision/recall eval harness for extraction |
| Research | Real-transcript benchmark suite → paper ([tokenmizer-research](https://github.com/Shweta-Mishra-ai/tokenmizer-research)) |

Have a use case that doesn't fit? [Open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose) — extraction misses have their own issue template.

---

## Contributing

Contributions welcome — this project merges fast (median PR review < 1 day).

```bash
git clone https://github.com/Shweta-Mishra-ai/tokenmizer
cd tokenmizer
pip install -e ".[dev]"
pytest tests/ -v && ruff check tokenmizer/     # 495 tests, must stay green
python scripts/mcp_e2e_check.py                # full-pipeline e2e check
```

**Highest-impact areas right now:**

1. **Graph extraction quality** — real-world transcripts where extraction misses tasks/decisions (file an [extraction-miss issue](.github/ISSUE_TEMPLATE/extraction_miss.md) even if you don't fix it — the failing transcript itself is the contribution)
2. **Decision tracker edge cases** — negation, semantic opposites, and same-decision matching are an active area (see recent merges below)
3. **Reasoning and ontology** (`graph_memory/reasoning.py`, `graph_memory/ontology.py`) — new in v0.4, still growing
4. **Benchmark sessions** — add a real session + ground truth to `benchmarks/`

Every PR runs the full CI gauntlet (tests × 3 Python versions on Linux, one Python version on Windows, lint, Docker build). See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [TESTING.md](TESTING.md) for the test architecture.

### Contributors

Thanks to everyone who has sent a fix upstream:

- [**@0xfroOty**](https://github.com/0xfroOty) — negated-decision handling in the decision tracker ([#22](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/22)), `OutputTrimmer` level alignment ([#25](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/25)), streaming cache-hit analytics ([#31](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/31))
- [**@pollychen-lab**](https://github.com/pollychen-lab) — graph node IDs derived from stored (truncated) labels ([#21](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/21)), semantic-opposite decision detection ([#26](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/26))
- [**@floze-the-genius**](https://github.com/floze-the-genius) — dashboard stats authentication fix ([#35](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/35))

Open a PR — [CONTRIBUTING.md](CONTRIBUTING.md) covers setup and review expectations.

---

## Support the project

TokenMizer is built and maintained by one person. If it saved you tokens, time, or a lost session:

- ⭐ **[Star the repo](https://github.com/Shweta-Mishra-ai/tokenmizer)** — the single best way to help others find it
- 🐛 [Report a bug](https://github.com/Shweta-Mishra-ai/tokenmizer/issues) — especially extraction misses
- 📣 Share your before/after token numbers (`tokenmizer stats`) — real usage data shapes the roadmap

---

## License

MIT © [Shweta Mishra](https://github.com/Shweta-Mishra-ai)

---

<div align="center">
  <sub>Built for developers who spend too much time re-explaining their projects to AI.</sub>
  <br/><br/>
  <a href="https://github.com/Shweta-Mishra-ai/tokenmizer/stargazers"><img src="https://img.shields.io/github/stars/Shweta-Mishra-ai/tokenmizer?style=flat-square&color=f9d84a&label=%E2%AD%90%20Star%20on%20GitHub" alt="GitHub stars"/></a>
</div>
