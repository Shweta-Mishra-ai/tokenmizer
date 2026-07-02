#!/usr/bin/env python3
"""
TokenMizer MCP Server

Exposes TokenMizer as an MCP (Model Context Protocol) server so:
  - Claude.ai can use it directly as a connected tool
  - Claude Code can call it via .mcp.json
  - Any MCP-compatible client (Cursor, Zed, VS Code) can integrate it

Tools exposed:
  - checkpoint_session    save current session to graph memory
  - resume_session        get resume context for a previous session
  - get_graph_stats       see what's in the knowledge graph
  - analyze_file          run file intelligence on any file
  - get_savings_stats     see token savings analytics

Run standalone:
  python3 -m tokenmizer.mcp.server

Or via MCP stdio transport (for Claude Code .mcp.json):
  {
    "mcpServers": {
      "tokenmizer": {
        "command": "python3",
        "args": ["-m", "tokenmizer.mcp.server"],
        "env": { "TOKENMIZER_URL": "http://localhost:8000" }
      }
    }
  }
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

TOKENMIZER_URL = os.environ.get("TOKENMIZER_URL", "http://localhost:8000")
TOKENMIZER_API_KEY = os.environ.get("TOKENMIZER_API_KEY", "")


# ── MCP Tool definitions ──────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "checkpoint_session",
        "description": (
            "Save the current AI session to TokenMizer's graph memory. "
            "Creates a checkpoint that can be resumed later with full context. "
            "Use this when: finishing a work session, before switching tasks, "
            "or when the conversation is getting long."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Unique identifier for this session (e.g. 'my-project-auth')",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about what was accomplished",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "resume_session",
        "description": (
            "Get the resume context for a previous session. "
            "Returns a compact summary of what was done, decided, and what's pending. "
            "Inject this into the system prompt when starting a new session on the same project."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to resume",
                },
                "level": {
                    "type": "string",
                    "enum": ["critical", "standard", "full"],
                    "description": "critical=~100 tokens, standard=~300 tokens, full=~600 tokens",
                    "default": "standard",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "get_graph_stats",
        "description": (
            "See the knowledge graph stats for a session: "
            "how many tasks, decisions, files, and errors are tracked."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to inspect",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "analyze_file",
        "description": (
            "Analyze a large file (CSV, Excel, PDF, JSON) and return a token-efficient summary. "
            "Instead of pasting thousands of rows into the chat, use this to get schema, "
            "statistics, and sample data in ~300-500 tokens."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to analyze",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Max tokens for the summary (default: 500)",
                    "default": 500,
                },
                "query": {
                    "type": "string",
                    "description": "What you want to know about the file (improves relevance)",
                    "default": "",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "get_savings_stats",
        "description": "Get token savings analytics — how many tokens were saved today/this week.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if TOKENMIZER_API_KEY:
        h["Authorization"] = f"Bearer {TOKENMIZER_API_KEY}"
    return h


def _get(path: str) -> dict:
    try:
        import httpx
        r = httpx.get(f"{TOKENMIZER_URL}{path}", headers=_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except ImportError:
        return {"error": "pip install httpx  — required for MCP server"}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict) -> dict:
    try:
        import httpx
        r = httpx.post(f"{TOKENMIZER_URL}{path}", json=body, headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except ImportError:
        return {"error": "pip install httpx  — required for MCP server"}
    except Exception as e:
        return {"error": str(e)}


# ── Tool handlers ─────────────────────────────────────────────────────────────

def handle_checkpoint_session(args: dict) -> str:
    session_id = args["session_id"]
    result = _post(f"/api/checkpoint?session_id={session_id}", {})
    if "error" in result:
        return f"❌ Checkpoint failed: {result['error']}"
    return (
        f"✅ Session '{session_id}' checkpointed\n"
        f"Checkpoint ID: {result.get('checkpoint_id', 'unknown')}\n"
        f"Nodes in graph: {result.get('node_count', 0)}\n"
        f"Resume size: {result.get('resume_tokens', 0)} tokens\n\n"
        f"Resume context:\n{result.get('resume_standard', '')}"
    )


def handle_resume_session(args: dict) -> str:
    session_id = args["session_id"]
    level = args.get("level", "standard")
    result = _get(f"/api/resume/{session_id}?level={level}")
    if "error" in result:
        return f"❌ Resume failed: {result['error']}"
    ctx = result.get("resume_context", "")
    tokens = result.get("token_count", 0)
    if not ctx:
        # A checkpoint EXISTS (the API 404s via "error" above when none does) —
        # its graph was just empty at checkpoint time. Saying "no checkpoint
        # found" here would be wrong and send the user debugging the wrong
        # thing.
        return (
            f"Checkpoint {result.get('checkpoint_id', '?')} exists for "
            f"'{session_id}' but its graph was empty (no session activity "
            f"had been recorded when it was created). Chat through the proxy "
            f"with this session_id, then checkpoint again."
        )
    return (
        f"[TokenMizer Resume — session: {session_id} — {tokens} tokens]\n\n"
        f"{ctx}\n\n"
        f"[Paste the above into your system prompt to resume this session]"
    )


def handle_get_graph_stats(args: dict) -> str:
    session_id = args["session_id"]
    result = _get(f"/api/graph/{session_id}")
    if "error" in result:
        return f"❌ Graph stats failed: {result['error']}"
    by_type = result.get("by_type", {})
    by_status = result.get("by_status", {})
    lines = [
        f"Graph for session: {session_id}",
        f"Total nodes: {result.get('node_count', 0)}",
        f"Total edges: {result.get('edge_count', 0)}",
        f"Messages processed: {result.get('processed_messages', 0)}",
        "",
        "Node types:",
    ]
    for t, count in sorted(by_type.items()):
        lines.append(f"  {t}: {count}")
    lines.append("\nNode statuses:")
    for s, count in sorted(by_status.items()):
        lines.append(f"  {s}: {count}")
    return "\n".join(lines)


def handle_analyze_file(args: dict) -> str:
    file_path = args["file_path"]
    token_budget = args.get("token_budget", 500)
    query = args.get("query", "")

    try:
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            return f"❌ File not found: {file_path}"

        content = path.read_bytes()
        from tokenmizer.filters.file_intelligence import FileIntelligence
        fi = FileIntelligence()
        result = fi.process(content, path.name, token_budget=token_budget, query=query)
        return (
            f"[File Analysis: {path.name}]\n"
            f"Type: {result.file_type} | "
            f"Original: {result.original_tokens:,} tokens | "
            f"Extracted: {result.extracted_tokens} tokens | "
            f"Saved: {result.savings_pct:.0f}%\n\n"
            f"{result.content}"
        )
    except Exception as e:
        return f"❌ File analysis error: {e}"


def handle_get_savings_stats(args: dict) -> str:
    result = _get("/api/stats")
    if "error" in result:
        return f"❌ Stats failed: {result['error']}"
    d = result.get("daily", {})
    w = result.get("weekly", {})
    breakdown = result.get("layer_breakdown", {})
    lines = [
        "TokenMizer Savings Report",
        "─" * 30,
        f"Today:   {d.get('tokens_saved', 0):,} tokens saved "
        f"({d.get('savings_pct', 0):.1f}%) — ${d.get('cost_saved_usd', 0):.4f}",
        f"Week:    {w.get('tokens_saved', 0):,} tokens saved "
        f"({w.get('savings_pct', 0):.1f}%) — ${w.get('cost_saved_usd', 0):.4f}",
        "",
        "Savings by layer:",
    ]
    for layer, saved in sorted(breakdown.items()):
        lines.append(f"  {layer}: {saved:,} tokens")
    return "\n".join(lines)


# ── MCP stdio transport ───────────────────────────────────────────────────────

def handle_tool_call(name: str, arguments: dict) -> str:
    handlers = {
        "checkpoint_session": handle_checkpoint_session,
        "resume_session": handle_resume_session,
        "get_graph_stats": handle_get_graph_stats,
        "analyze_file": handle_analyze_file,
        "get_savings_stats": handle_get_savings_stats,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(arguments)
    except Exception as e:
        return f"Tool error: {e}"


def run_stdio_server():
    """
    MCP stdio transport — reads JSON-RPC from stdin, writes to stdout.
    This is the standard MCP server protocol.
    """

    def send(obj: dict):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def send_error(req_id: Any, code: int, message: str):
        send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "initialize":
            from tokenmizer import __version__
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "tokenmizer", "version": __version__},
                },
            })

        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result_text = handle_tool_call(tool_name, arguments)
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": result_text.startswith("❌"),
                },
            })

        elif method == "notifications/initialized":
            pass  # acknowledge, no response needed

        else:
            send_error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    run_stdio_server()
