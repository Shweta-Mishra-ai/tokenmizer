# API and CLI reference

Every HTTP endpoint, every CLI command, and the MCP tools. The endpoint table is checked against the route decorators by `tests/unit/test_version_consistency.py`, in both directions — a documented endpoint that does not exist fails the build, and so does a live endpoint missing from the table.

---

# API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy |
| `/api/resume/{id}` | GET | Get resume context |
| `/api/checkpoint` | POST | Manual checkpoint |
| `/api/analyze` | POST | File → token-budgeted digest (CSV/JSON/PDF/Excel/logs/code) |
| `/api/checkpoints/{id}` | GET | List a session's checkpoints |
| `/api/graph/{id}/viz` | GET | Graph as D3-compatible JSON |
| `/api/graph/{id}/history` | GET | Graph state at a point in time |
| `/api/graph/{id}/transitions` | GET | Decision transitions, newest first |
| `/api/graph/{id}/obsidian` | GET | Obsidian Canvas export |
| `/api/cache/stats` | GET | Semantic cache hit rate and utilisation |
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

# CLI

```bash
tokenmizer serve [--port 8000]
tokenmizer checkpoint <session-id>
tokenmizer resume <session-id> [--level standard|full|critical]
tokenmizer stats
```

> **File analysis, three ways.** `FileIntelligence` turns a large CSV /
> JSON / PDF / Excel / log / code file into a token-budgeted digest, and
> is reachable from all three surfaces:
>
> | Surface | Use it when |
> |---|---|
> | `tokenmizer analyze <file>` | A plain shell, a script, CI. Runs locally — no server, no API key. |
> | `POST /api/analyze` | Another tool, curl, a remote client. Content is sent inline, never a server-side path. |
> | `/tokenmizer:analyze` | Inside Claude Code (plugin skill). |
>
> ```bash
> tokenmizer analyze data.csv --token-budget 300
> tokenmizer analyze big.json --raw > digest.txt
> ```
>
> The endpoint takes `content` inline rather than a path on purpose: the
> server is often a container or a remote host, so a client-side path
> means nothing to it — and accepting one would be an arbitrary-file-read
> primitive against the server.

---

# Claude Code Integration

## Option A — Plugin (recommended)

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

## Option B — MCP server (Claude Desktop, Claude Code, Cursor, VS Code, Zed)

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

# Other Tools

**Cursor / Continue.dev / any OpenAI-compatible tool:**
```
API Base URL:  http://localhost:8000/v1
```

---


---

[← Back to the README](../README.md)
