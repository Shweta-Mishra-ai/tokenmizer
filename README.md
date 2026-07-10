<div align="center">
  <img src="docs/assets/logo.svg" width="140" alt="TokenMizer"/>

  <h1>TokenMizer</h1>

  <p><strong>Keep your AI context alive across sessions.</strong></p>

  <p>
    Graph-backed memory · session checkpointing · intelligent compression<br/>
    Drop-in proxy for Claude, GPT, Gemini, Grok, DeepSeek, Ollama — any LLM.
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

TokenMizer is a **local proxy** between your app and any LLM. Every request goes through a pipeline that builds a live knowledge graph, compresses inputs, caches responses, and auto-checkpoints before context runs out.

```
Your App  →  TokenMizer (:8000)  →  Claude / GPT / Gemini / any LLM
                    │
          ┌─────────┴──────────────┐
          │   6-Layer Pipeline     │
          │   L0  File Intel       │  CSV/PDF/Excel → schema + sample
          │   L1  Compression      │  15–40% input reduction
          │   L2  Output Trim      │  5–15% output reduction
          │   L3  Semantic Cache   │  100% on repeated queries
          │   L4  Graph Memory     │  session continuity
          │   L5  Prompt Cache     │  90% on repeated system prompts
          └────────────────────────┘
```

---

## Architecture

<div align="center">
  <img src="docs/assets/architecture.svg" width="860" alt="Architecture"/>
</div>

### Decision Memory — 4-State Model

| Status | Meaning | In Resume |
|---|---|---|
| 🟢 `ACTIVE` | Current — in effect | ✅ Always |
| 🟡 `SUPERSEDED` | Replaced by newer decision | ⚠️ 7 days |
| 🔴 `INVALIDATED` | Explicitly wrong/cancelled | ⚠️ Always (warning) |
| ⬜ `ARCHIVED` | Superseded >7 days ago — aged out | ❌ Never |

History is **never deleted**. "Why did we switch from React to Next.js?" — always answerable.

> Honesty note (2026-07-10 audit): before v0.3.2, `ARCHIVED` was advertised
> here but **unreachable** — the decay/prune/query logic all handled it, yet
> no code path ever set it. Superseded decisions now age into `ARCHIVED`
> automatically after 7 days (`GraphMemory.ARCHIVE_SUPERSEDED_AFTER_DAYS`),
> which is what makes the "⚠️ 7 days" row above actually true.

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
checkpoint/resume/stats tools (file analysis works without it).
If `tokenmizer-mcp` isn't on your PATH, use `"command": "python"`,
`"args": ["-m", "tokenmizer.mcp.server"]` instead.

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
                                # (needs a provider key, ~$0.001/turn).
                                # NOTE: before v0.3.2 this flag silently did
                                # nothing — the call site passed provider_fn
                                # to the wrong function and raised TypeError
                                # on every turn, falling back to heuristics.
                                # Fixed + regression-tested; see CHANGELOG.

compression:
  enabled: true

cache:
  enabled: true
  max_size: 10000

state_backend: memory           # memory | redis (production)
```

All settings via env vars: `TOKENMIZER_PROVIDER`, `TOKENMIZER_API_KEY`, etc.

---

## Docker

```bash
# Quick start
docker-compose up tokenmizer

# With Redis (production)
ANTHROPIC_API_KEY=sk-ant-... docker-compose up

# With proxy auth
TOKENMIZER_API_KEY=strong-key docker-compose up
```

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
| `/api/stats` | GET | Token savings analytics |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## Security

- API key auth — `TOKENMIZER_API_KEY` (constant-time comparison)
- Secret/PII redaction applied once at ingestion, before graph storage,
  checkpoint storage, AND every LLM call (main chat *and* the background
  extraction model — these are separate, the redaction gap between them
  was a real bug, now fixed). Patterns cover Anthropic/OpenAI/Google/
  GitHub/AWS/Slack/Stripe/JWT/OpenRouter/HF/xAI keys, URL-embedded
  credentials (`postgres://user:pass@host` — a 2026-07 audit gap, fixed),
  and generic `key=`/`password=` assignments. Best-effort by nature —
  an unrecognized format with no keyword context can still slip through.
  The checkpoint layer independently re-redacts what it persists
  (defense-in-depth, added in the same audit).
- Session-isolated cache (sensitive data never shared across sessions)
- Basic prompt-injection keyword filter — catches copy-pasted jailbreak
  templates only; **not** a security boundary against a motivated
  adversary. See [SECURITY.md](SECURITY.md#prompt-injection-basic-keyword-filter-read-the-scope)
  for exactly what it does and doesn't catch.
- CORS restricted to configured origins by default

---

## Benchmarks

```bash
python benchmarks/checkpoint_accuracy/runner_v2.py
pytest tests/ -v
```

**Benchmark v2 — Graph vs plain Summary (3 sessions, heuristic-only,
measured 2026-07-02 on v0.2.4):**

| Method | Task Recall | Decision Recall | File Recall | Info Preserved |
|--------|-------------|-----------------|-------------|----------------|
| TokenMizer Graph | 76% | 85% | 100% | **87%** |
| Plain Summary baseline | 76% | 70% | 92% | 79% |
| **Δ advantage** | 0% | **+15%** | **+8%** | **+8%** |

Avg resume size: **254 tokens** vs ~1,500+ tokens of raw history.
(n=3 synthetic sessions — small sample; treat as directional, reproduce
with the command above.)

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
| v0.4 | Cross-session memory · embedding-based edge linking |
| v0.5 | Per-node storage schema (scale past 200-node graphs) |
| Research | Real-transcript benchmark suite → paper ([tokenmizer-research](https://github.com/Shweta-Mishra-ai/tokenmizer-research)) |

Have a use case that doesn't fit? [Open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues/new/choose) — extraction misses have their own issue template.

---

## Contributing

Contributions welcome — this project merges fast (median PR review < 1 day).

```bash
git clone https://github.com/Shweta-Mishra-ai/tokenmizer
cd tokenmizer
pip install -e ".[dev]"
pytest tests/ -v && ruff check tokenmizer/     # 262 tests, must stay green
python scripts/mcp_e2e_check.py                # full-pipeline e2e check
```

**Highest-impact areas right now:**

1. **Graph extraction quality** — real-world transcripts where extraction misses tasks/decisions (file an [extraction-miss issue](.github/ISSUE_TEMPLATE/extraction_miss.md) even if you don't fix it — the failing transcript itself is the contribution)
2. **SSE streaming** (v0.3 headline feature)
3. **Benchmark sessions** — add a real session + ground truth to `benchmarks/`

Every PR runs the full CI gauntlet (tests × 3 Python versions, lint, Docker build). See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [TESTING.md](TESTING.md) for the test architecture.

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
