# API and CLI reference

Every HTTP endpoint, every CLI command, and the MCP tools. The endpoint table is checked against the route decorators by `tests/unit/test_version_consistency.py`, in both directions — a documented endpoint that does not exist fails the build, and so does a live endpoint missing from the table.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy, including `tools` / `tool_choice` and `role: "tool"` messages (see below) |
| `/api/resume/{id}` | GET | Resume context. Built live from the graph whenever it is newer than the latest checkpoint (or there is none) — `source: "live_graph"` — else from the checkpoint. 404 only when both are empty |
| `/api/checkpoint` | POST | Manual checkpoint. Optional JSON body `{"messages": [{role, content}, ...]}` is extracted into the graph first, so a checkpoint made outside the proxy (MCP tool, CLI) populates the session |
| `/api/analyze` | POST | File → token-budgeted digest (CSV/JSON/PDF/Excel/logs/code) |
| `/api/checkpoints/{id}` | GET | List a session's checkpoints |
| `/api/graph/{id}/viz` | GET | Graph as D3-compatible JSON |
| `/api/graph/{id}/history` | GET | Graph state at a point in time |
| `/api/graph/{id}/transitions` | GET | Decision transitions, newest first |
| `/api/graph/{id}/obsidian` | GET | Obsidian Canvas export |
| `/api/cache/stats` | GET | Semantic cache hit rate and utilisation |
| `/api/preferences` | GET | What this principal's habits are remembered as, and the exact text injected into the system prompt. `{"enabled": false}` when `preferences.enabled` is off |
| `/api/preferences` | DELETE | Forget one (`?key=...`) or all of this principal's preferences |
| `/api/decision/invalidate` | POST | Mark decision as invalid |
| `/api/sessions` | GET | The caller's sessions with node counts, last activity and a link to each graph page. Scoped by ownership: one API key never sees another's sessions |
| `/api/graph/{id}` | GET | Session graph stats |
| `/api/graph/{id}/html` | GET | **Interactive graph page** — nodes grouped and colored by detected community with a toggleable community panel, click-to-inspect node detail, decision-history timeline, supersession arcs, search, fit/zoom/pan, PNG export; an empty graph explains why. Zero external dependencies (works offline) |
| `/api/graph/{id}/why?q=` | GET | **Reasoning:** causal chain behind a decision (old → new with trigger/reason/evidence) |
| `/api/graph/{id}/reasoning` | GET | **Reasoning view:** active decisions by topic, recent changes, consistency audit |
| `/api/ontology` | GET | Machine-readable graph ontology (types, relations, status state machine) |
| `/api/stats` | GET | Token savings analytics |
| `/health` | GET | Liveness AND durability. `status` is `degraded`, not `ok`, when a write has failed, a session's stored graph could not be read, storage is not durable, or memory was displaced by corruption recovery; the counters behind it are in the body. No API key required |
| `/docs` | GET | Swagger UI |

### Tool calling

Send the OpenAI shape and get it back:

```python
resp = client.chat.completions.create(
    model="claude-sonnet-4-6",                       # any supported provider's model
    messages=[{"role": "user", "content": "weather in Pune?"}],
    tools=[{"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    extra_body={"session_id": "my-project"},
)
call = resp.choices[0].message.tool_calls[0]          # finish_reason == "tool_calls"
# run the tool, then continue the loop:
messages += [resp.choices[0].message,
             {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 31}'}]
```

| Provider | `tools` | Streamed tool calls |
|---|---|---|
| OpenAI, DeepSeek, Mistral, OpenRouter, Grok | forwarded as-is | fragments, as the API sends them |
| Anthropic | translated (`input_schema`, `tool_use` / `tool_result` blocks) | fragments (`input_json_delta`) |
| Cohere | v2 takes the OpenAI shape; tool results become document parts | fragments (`tool-call-delta`); no `tool_choice` |
| Ollama | translated (`arguments` as a dict) | whole calls on the last message; no `tool_choice` |
| Gemini | translated (`function_declarations`, `function_call` / `function_response` parts) | whole calls per chunk |

Every provider's deltas come out in the OpenAI chunk shape, indexed by
tool call — so a client assembles them the same way whatever is behind
the proxy. Anthropic indexes content blocks rather than calls and Gemini
numbers nothing at all; both are mapped here rather than passed on.

What the pipeline does with tool traffic: tool turns are redacted like any
other message and otherwise passed through untouched (no compression, no
history pruning); windowing keeps a tool call and its results together; a
tool-call answer is never cached or output-trimmed; a request that ends on
a tool result is never served from the cache. The deprecated
`functions` / `function_call` fields are ignored with a warning.

---

## CLI

```bash
tokenmizer serve [--port 8000]
tokenmizer checkpoint [session-id]      # default: the working directory name
tokenmizer resume [session-id] [--level standard|full|critical]
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

<!-- mcp-name: io.github.Shweta-Mishra-ai/tokenmizer -->

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

Then restart the client.

**The proxy is optional for most tools.** `checkpoint_session`,
`resume_session`, `get_graph_stats`, `why_decision` and `analyze_file`
fall back to reading and writing the SQLite store directly when nothing
answers at `TOKENMIZER_URL`, and say so in the reply. Point
`TOKENMIZER_STORAGE_DIR` at the same directory as
`graph_checkpoint.storage_dir` if you have moved it (default:
`./checkpoints`). Only `get_savings_stats` needs the proxy, because
savings are measured on requests that pass through it.

Only a transport failure falls back. A 401, 403 or 404 means the proxy is
running and refused, and answering from local storage would bypass the
session-ownership boundary it was enforcing — so those stay errors.

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

[← Back to the README](../README.md)
