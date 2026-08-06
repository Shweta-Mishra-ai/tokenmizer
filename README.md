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
    <a href="https://github.com/sponsors/Shweta-Mishra-ai"><img src="https://img.shields.io/badge/sponsor-%E2%9D%A4-db61a2?style=flat-square" alt="Sponsor"/></a>
  </p>

  <p>
    <a href="#quick-start"><b>Quick start</b></a> ·
    <a href="#use-it-from-your-tools"><b>Claude Code &amp; MCP</b></a> ·
    <a href="docs/architecture.md"><b>Architecture</b></a> ·
    <a href="docs/benchmarks.md"><b>Benchmarks</b></a> ·
    <a href="docs/configuration.md"><b>Configuration</b></a> ·
    <a href="docs/api.md"><b>API &amp; CLI</b></a> ·
    <a href="CONTRIBUTING.md"><b>Contributing</b></a>
  </p>

  <img src="docs/assets/demo.gif" width="860" alt="TokenMizer demo: 40-turn session checkpointed at 87% context, resumed next day in 233 tokens"/>
  <br/>
  <sub>Real run: 25-node graph, checkpoint <code>ckpt_21a0959c3ddf</code>, 233-token resume. Regenerate with <code>python scripts/gen_demo_gif.py</code>.</sub>
</div>

---

---

## The problem

Every AI session has a context limit. When you hit it, the model forgets
every decision and every rationale built over hours of work, and you
spend the first ten minutes of the next session re-explaining the
project.

Summarising the history does not fix this. A summary tells you *what*
was decided; it loses *why*, and it loses what was rejected — so the
model happily re-proposes the thing you moved off three sessions ago.

## How it works

TokenMizer is a local proxy between your app and any LLM. Every request
passes through a pipeline that builds a live knowledge graph, compresses
inputs, caches responses, and checkpoints before the context runs out.

```mermaid
flowchart LR
    App["Your app<br/><sub>OpenAI-compatible client</sub>"]
    subgraph TM["TokenMizer :8000"]
        direction TB
        L0["<b>L0</b> File intelligence"]
        L1["<b>L1</b> Prompt compression"]
        L2["<b>L2</b> Terse-output injection"]
        L4["<b>L4</b> Graph memory<br/><sub>extract → window → inject</sub>"]
        L3["<b>L3</b> Semantic cache"]
        L5["<b>L5</b> Provider prompt cache"]
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

The graph is not a summary. It is typed nodes and edges — decisions,
tasks, files, errors, goals — with a lifecycle, so a decision that gets
replaced is marked superseded rather than deleted. The resume block is a
filtered projection of it: active decisions, open work, unresolved
errors, in a few hundred tokens.

→ [**Architecture**](docs/architecture.md) — the request sequence, the
data model, and the decision lifecycle.

## Quick start

```bash
pip install "tokenmizer[anthropic,cache]"
export TOKENMIZER_ANTHROPIC_API_KEY=sk-ant-...
tokenmizer serve
```

Then change one line in your client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1",   # ← only this changes
)

resp = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Continue where we left off"}],
    extra_body={"session_id": "my-project"},   # ← optional, enables memory
)
```

Everything else is unchanged: same request shape, same response shape,
plus a `tokenmizer` block reporting what was saved.

<details>
<summary><b>Windows, Ollama, Docker, and the full step-by-step</b></summary>

**Windows (PowerShell)**

```powershell
$env:TOKENMIZER_ANTHROPIC_API_KEY = "sk-ant-..."   # this session
setx TOKENMIZER_ANTHROPIC_API_KEY "sk-ant-..."     # persistent
```

**No API key?** Ollama runs locally and free:

```bash
ollama pull llama3
pip install tokenmizer
# then set `provider: ollama` in tokenmizer.yaml
```

**Docker**

```bash
docker compose up -d
```

Full installation notes, every provider's environment variable, and the
configuration reference are in
[**docs/configuration.md**](docs/configuration.md) and
[**docs/deployment.md**](docs/deployment.md).
</details>

## Use it from your tools

Three ways in, depending on where you work. All three talk to the same
graph, so a session checkpointed from Claude Code resumes in the CLI.

### Claude Code — plugin

```
/plugin marketplace add Shweta-Mishra-ai/tokenmizer
/plugin install tokenmizer@Shweta-Mishra-ai/tokenmizer
```

Then, in any session:

```
/tokenmizer:checkpoint my-project      save the session to graph memory
/tokenmizer:resume my-project          load it back (~300 tokens)
/tokenmizer:analyze data/sales.csv     digest a large file
/tokenmizer:stats                      token savings report
```

### Claude Desktop, Cursor, VS Code, Zed — MCP server

<!-- mcp-name: io.github.Shweta-Mishra-ai/tokenmizer -->

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

| Client | Where that goes |
|---|---|
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Code | `.mcp.json` in the project, or `~/.claude/settings.json` |
| Cursor | Settings → MCP → Add server, same JSON |
| VS Code / Zed | their MCP settings, same `command` and `env` |
| Codex CLI | `~/.codex/config.toml` — TOML, see [docs/api.md](docs/api.md) |

Restart the client afterwards. Keep `tokenmizer serve` running for the
checkpoint, resume, stats and reasoning tools; file analysis works
without it. If `tokenmizer-mcp` is not on your PATH, use
`"command": "python", "args": ["-m", "tokenmizer.mcp.server"]`.

**Six tools:** `checkpoint_session`, `resume_session`, `get_graph_stats`,
`get_savings_stats`, `analyze_file`, and `why_decision` — ask your agent
*"why did we pick X?"* and it walks the supersession chain with the
reason and evidence for each hop.

### Anything else — the proxy

Any OpenAI-compatible client works by pointing `base_url` at
`http://localhost:8000/v1`, as in the quick start above. That covers
Continue.dev, Aider, LangChain, LlamaIndex, the OpenAI SDKs in every
language, and `curl`.

→ [**API & CLI reference**](docs/api.md) — every endpoint, every command,
every MCP tool.

## What a resume looks like

```
Goal: Build FastAPI auth service with JWT + PostgreSQL
Done: Project setup | User model | Login endpoint | Fix 422 | 18 tests passing
In progress: Refresh token rotation
Decided: PostgreSQL (concurrent writes) | bcrypt | Redis for refresh tokens
Changed: ~~React~~ → Next.js (better SEO)
Files: api/auth.py, api/models.py, config.py
Continue: Implement token refresh endpoint
```

A few hundred tokens in place of the whole conversation. The `Changed:`
line is the part a summary loses — and asking
`GET /api/graph/{session}/why?q=react` replays the full chain with the
trigger, the reason and the evidence for each hop.

## Measured

`python -m benchmarks.eval` scores extraction against a labelled corpus
of 14 sessions, 6 of them real transcripts:

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Files | 98% | 100% | **99%** |
| Pending tasks | 100% | 90% | **95%** |
| Errors | 93% | 96% | **94%** |
| Decisions | 90% | 95% | **92%** |
| Completed tasks | 92% | 90% | **91%** |
| | | **macro F1** | **94%** |

**Precision is reported, not just recall.** An extractor that emits the
whole transcript as one node scores 100% recall, which is why
recall-only extraction numbers should be distrusted — including our own
earlier ones.

Scored separately by origin, because hand-written fixtures are easier
than real transcripts and a single headline hides that: **synthetic 95%,
real 90%.** Treat 90% as the number that describes real sessions. n=14
is a small sample and the same person wrote every label.

→ [**Benchmarks**](docs/benchmarks.md) — memory quality against a
plain-summary baseline, storage, and how to score your own sessions.

## What is not implemented

Two settings are accepted by the config and do nothing. They are listed
here rather than left to be discovered:

| Setting | Status |
|---|---|
| `routing.*` | No implementation. `savings.routing` is always `0`. Enabling it logs a warning and changes nothing. |
| `state_backend: redis` | Accepted and unused. `tokenmizer/state/backend.py` has no callers; all durable state is SQLite. |

## Documentation

| | |
|---|---|
| [**Architecture**](docs/architecture.md) | Request pipeline, graph data model, decision lifecycle, file intelligence |
| [**Configuration**](docs/configuration.md) | Every setting, environment variables, precedence, providers |
| [**API & CLI**](docs/api.md) | Endpoints, commands, MCP tools, Claude Code integration |
| [**Deployment**](docs/deployment.md) | Docker, multiple workers, durability, session isolation, security |
| [**Benchmarks**](docs/benchmarks.md) | Extraction quality, memory quality, storage, running your own |
| [**Comparisons**](docs/comparisons.md) | Mem0, Zep, longer context windows, and the roadmap |
| [**Contributing**](CONTRIBUTING.md) | Setup, layer rules, and how to improve extraction |
| [**Changelog**](CHANGELOG.md) · [**Security**](SECURITY.md) | Release history and how to report a vulnerability |

## Contributing

```bash
git clone https://github.com/Shweta-Mishra-ai/tokenmizer
cd tokenmizer
pip install -e ".[dev]"
pytest tests/ -q && ruff check tokenmizer/     # 584 tests, must stay green
```

**The most valuable contribution is a session where extraction got it
wrong.** The eval corpus is 14 sessions and the same person wrote every
label in it — that is the honest ceiling on what the numbers above can
tell you about *your* workload, and the only way past it is transcripts
nobody here wrote. Label a few of your own in the format documented in
[`benchmarks/eval/corpus.py`](benchmarks/eval/corpus.py) and open a PR,
or [open an issue](https://github.com/Shweta-Mishra-ai/tokenmizer/issues)
with the turn that was missed. Redact freely — the shape of the prose is
what matters, not its content.

[CONTRIBUTING.md](CONTRIBUTING.md) covers setup, the layer rules, and how
to run the eval harness.

### Contributors

Thanks to everyone who has sent a fix upstream:

- [**@0xfroOty**](https://github.com/0xfroOty) — negated-decision handling in the decision tracker ([#22](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/22)), `OutputTrimmer` level alignment ([#25](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/25)), streaming cache-hit analytics ([#31](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/31))
- [**@pollychen-lab**](https://github.com/pollychen-lab) — graph node IDs derived from stored (truncated) labels ([#21](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/21)), semantic-opposite decision detection ([#26](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/26))
- [**@floze-the-genius**](https://github.com/floze-the-genius) — dashboard stats authentication fix ([#35](https://github.com/Shweta-Mishra-ai/tokenmizer/pull/35))

## Support

A [⭐ star](https://github.com/Shweta-Mishra-ai/tokenmizer) is how other
people find it. If it saved you real time and you would like to put
something behind it,
[sponsorship](https://github.com/sponsors/Shweta-Mishra-ai) is open —
entirely optional.

## License

MIT © [Shweta Mishra](https://github.com/Shweta-Mishra-ai)
