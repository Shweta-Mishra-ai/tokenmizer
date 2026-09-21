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
  - why_decision          trace why a decision is the current choice
                          (supersession chain with reasons/evidence)

Run standalone:
  tokenmizer-mcp

Or via MCP stdio transport (Claude Code .mcp.json, Cursor, VS Code, Zed,
Codex — every client takes the same shape):
  {
    "mcpServers": {
      "tokenmizer": {
        "command": "tokenmizer-mcp",
        "env": { "TOKENMIZER_URL": "http://localhost:8000" }
      }
    }
  }

`tokenmizer-mcp` is the console script declared in pyproject.toml. Do not
substitute a bare `python3`: on Windows that name is the Microsoft Store
alias and the server never starts.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

TOKENMIZER_URL = os.environ.get("TOKENMIZER_URL", "http://localhost:8000")
TOKENMIZER_API_KEY = os.environ.get("TOKENMIZER_API_KEY", "")
# Where the graph lives when these tools read it directly. Matches the
# proxy's own default (graph_checkpoint.storage_dir), so a local read sees
# exactly what a running proxy would have written.
TOKENMIZER_STORAGE_DIR = os.environ.get("TOKENMIZER_STORAGE_DIR", "./checkpoints")

# The MCP server is installed by a plugin; the proxy is a separate process
# nobody started. So the single most likely state on a fresh install is
# "tokenmizer serve is not running", and five of the six tools answered
# nothing but an explanation of that. They now fall back to reading and
# writing the SQLite store directly — the same store the proxy uses, under
# the same cross-process lock — and say which path answered.
#
# Only a TRANSPORT failure falls back. A 401, a 403 or a 404 means the
# proxy is there and said no, and quietly answering from local storage
# would bypass the session-ownership boundary it was enforcing.
OFFLINE = "offline"


def _local_note(detail: str = "") -> str:
    return ("\n\n[Answered from local graph memory at "
            f"{TOKENMIZER_STORAGE_DIR} — TokenMizer is not running at "
            f"{TOKENMIZER_URL}. Start it with `tokenmizer serve` for the "
            f"proxy's analytics and cross-session features.{detail}]")


def _is_offline(result: dict) -> bool:
    return isinstance(result, dict) and result.get(OFFLINE) is True


def _memory(session_id: str):
    from tokenmizer.agents import Memory

    return Memory(session_id, storage_dir=TOKENMIZER_STORAGE_DIR)


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
                # The conversation so far. This tool runs beside the
                # conversation rather than through the proxy, so without
                # it the checkpoint has nothing to extract from and the
                # session is recorded with zero nodes.
                "messages": {
                    "type": "array",
                    "description": (
                        "The conversation to save, as {role, content} objects "
                        "(user/assistant). Decisions, tasks, files and errors "
                        "are extracted from it into the session's graph memory."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
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
    {
        "name": "why_decision",
        "description": (
            "Reason over the session's decision history: why is something "
            "the current choice? Traces the supersession chain (old → new "
            "with trigger, reason, and evidence per hop) for decisions "
            "matching the query, and reports the currently active choice. "
            "Use when the user asks 'why did we pick X', 'what happened to "
            "Y', or 'what was the previous approach'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session whose decision history to query",
                },
                "query": {
                    "type": "string",
                    "description": "Substring of the decision to explain (e.g. 'react', 'postgres')",
                },
            },
            "required": ["session_id", "query"],
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
        logger.warning(f"GET {path} against the proxy failed: {e}")
        return {"error": _explain(e), OFFLINE: _is_transport_error(e)}


def _is_transport_error(exc: Exception) -> bool:
    """True if the proxy could not be reached at all.

    A connect failure or a timeout means nothing answered. An HTTP status
    means something did, and its answer stands — see the OFFLINE note.
    """
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                           httpx.ReadTimeout, httpx.TimeoutException))


def _explain(exc: Exception) -> str:
    """Turn a transport failure into something the reader can act on.

    Every graph tool here talks to the proxy, and the proxy is a separate
    process the plugin install does not start. So the single most likely
    outcome of a fresh install is every tool answering
    "[Errno 111] Connection refused" — which names a socket, not a next step,
    and reads as a broken plugin rather than a stopped server. cli.py already
    solved exactly this for the same failure; MCP clients deserve the same
    sentence.

    Also keeps SDK exception text out of the reply: it routinely embeds the
    request URL and query string, which for these endpoints contains the
    session id. Full detail still goes to the log.
    """
    try:
        import httpx
    except ImportError:
        return f"{type(exc).__name__}: {exc}"

    if isinstance(exc, httpx.ConnectError):
        return (
            f"TokenMizer is not running at {TOKENMIZER_URL}. Start it with "
            "`tokenmizer serve`, then try again. (Set TOKENMIZER_URL if it "
            "runs somewhere else.) File analysis works without the server; "
            "checkpoint, resume, graph and savings tools need it."
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"TokenMizer did not respond within the timeout at {TOKENMIZER_URL}. "
            "It may still be starting up, or busy with a long request."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return (
                f"TokenMizer rejected the request ({code}). Set TOKENMIZER_API_KEY "
                "in the MCP server's env to match the proxy's api_key."
            )
        if code == 404:
            return "Not found — nothing has been checkpointed under that session id yet."
        return f"TokenMizer returned HTTP {code}."
    return f"{type(exc).__name__}"


def _post(path: str, body: dict) -> dict:
    try:
        import httpx
        r = httpx.post(f"{TOKENMIZER_URL}{path}", json=body, headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except ImportError:
        return {"error": "pip install httpx  — required for MCP server"}
    except Exception as e:
        logger.warning(f"POST {path} against the proxy failed: {e}")
        return {"error": _explain(e), OFFLINE: _is_transport_error(e)}


# ── Tool handlers ─────────────────────────────────────────────────────────────
#
# Every handler returns (text, is_error). The error flag is structural —
# never inferred from the text — so validation failures and crashes are
# reported to MCP clients with isError: true regardless of message content.


class ToolInputError(ValueError):
    """Client sent arguments that fail the tool's declared inputSchema."""


def _require_str(args: dict, key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ToolInputError(
            f"Missing or invalid required argument '{key}' (expected a "
            f"non-empty string, got {type(val).__name__}: {val!r})"
        )
    return val


def handle_checkpoint_session(args: dict) -> tuple[str, bool]:
    session_id = _require_str(args, "session_id")
    # session_id must never be interpolated raw into the
    # query string — a session_id containing '&', space, or other
    # reserved URL characters would produce a malformed or misdirected
    # request. quote() is used consistently across every handler below,
    # matching the one handler (why_decision) that already did this.
    messages = args.get("messages")
    if messages is not None and not isinstance(messages, list):
        return "Invalid 'messages': expected an array of {role, content} objects.", True
    result = _post(
        f"/api/checkpoint?session_id={quote(session_id, safe='')}",
        {"messages": messages or []},
    )
    if _is_offline(result):
        return _local_checkpoint(session_id, messages or [])
    if "error" in result:
        return f"Checkpoint failed: {result['error']}", True
    return (
        f"Session '{session_id}' checkpointed\n"
        f"Checkpoint ID: {result.get('checkpoint_id', 'unknown')}\n"
        f"Nodes in graph: {result.get('node_count', 0)}\n"
        f"Resume size: {result.get('resume_tokens', 0)} tokens\n\n"
        f"Resume context:\n{result.get('resume_standard', '')}"
    ), False


def _local_checkpoint(session_id: str, messages: list) -> tuple[str, bool]:
    """Write the transcript into the graph without the proxy."""
    try:
        memory = _memory(session_id)
        gained = memory.add([
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages if isinstance(m, dict)
        ]) if messages else 0
        memory.save()
        stats = memory.stats()
        context = memory.context(token_budget=400)
    except Exception as e:
        logger.exception("Local checkpoint failed")
        return (f"Checkpoint failed, and TokenMizer is not running at "
                f"{TOKENMIZER_URL} to fall back on: {type(e).__name__}: {e}"), True
    return (
        f"Session '{session_id}' saved to graph memory\n"
        f"Nodes in graph: {stats.get('node_count', 0)} (+{gained} from this call)\n\n"
        f"Resume context:\n{context}" + _local_note()
    ), False


def handle_resume_session(args: dict) -> tuple[str, bool]:
    session_id = _require_str(args, "session_id")
    level = args.get("level", "standard")
    if level not in ("critical", "standard", "full"):
        raise ToolInputError(
            f"Invalid 'level': {level!r} (expected critical|standard|full)"
        )
    result = _get(f"/api/resume/{quote(session_id, safe='')}?level={level}")
    if _is_offline(result):
        return _local_resume(session_id, level)
    if "error" in result:
        return f"Resume failed: {result['error']}", True
    ctx = result.get("resume_context", "")
    tokens = result.get("token_count", 0)
    if not ctx:
        # The API 404s (via "error" above) only when there is neither a
        # checkpoint nor any node, so reaching here means the session is
        # known but nothing in it survived extraction. Saying "no
        # checkpoint found" would send the user debugging the wrong thing.
        return (
            f"Session '{session_id}' exists but its memory has nothing to "
            f"resume from (no decisions, tasks, files or errors were "
            f"extracted). Chat through the proxy with this session_id, or "
            f"checkpoint with the conversation as `messages`, then try again."
        ), False
    source = "graph memory (live)" if result.get("source") == "live_graph" \
        else f"checkpoint {result.get('checkpoint_id', '?')}"
    return (
        f"[TokenMizer Resume — session: {session_id} — {tokens} tokens — "
        f"from {source}]\n\n"
        f"{ctx}\n\n"
        f"[Paste the above into your system prompt to resume this session]"
    ), False


# Budgets match the proxy's three resume tiers so the local answer is the
# same size as the remote one.
_LEVEL_BUDGET = {"critical": 120, "standard": 400, "full": 700}


def _local_resume(session_id: str, level: str) -> tuple[str, bool]:
    try:
        memory = _memory(session_id)
        context = memory.context(token_budget=_LEVEL_BUDGET.get(level, 400))
        nodes = memory.stats().get("node_count", 0)
    except Exception as e:
        logger.exception("Local resume failed")
        return (f"Resume failed, and TokenMizer is not running at "
                f"{TOKENMIZER_URL} to fall back on: {type(e).__name__}: {e}"), True
    if not context:
        return (
            f"Nothing to resume for '{session_id}': its graph memory holds "
            f"{nodes} node(s) at {TOKENMIZER_STORAGE_DIR}. Chat through the "
            f"proxy with this session_id, or pass the conversation to "
            f"checkpoint_session as `messages`."
        ), False
    return (
        f"[TokenMizer Resume — session: {session_id} — from local graph memory]\n\n"
        f"{context}\n\n"
        f"[Paste the above into your system prompt to resume this session]"
        + _local_note()
    ), False


def handle_get_graph_stats(args: dict) -> tuple[str, bool]:
    session_id = _require_str(args, "session_id")
    result = _get(f"/api/graph/{quote(session_id, safe='')}")
    if _is_offline(result):
        try:
            result = _memory(session_id).stats()
            offline = True
        except Exception as e:
            logger.exception("Local graph stats failed")
            return (f"Graph stats failed, and TokenMizer is not running at "
                    f"{TOKENMIZER_URL}: {type(e).__name__}: {e}"), True
    else:
        offline = False
    if "error" in result:
        return f"Graph stats failed: {result['error']}", True
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
    return "\n".join(lines) + (_local_note() if offline else ""), False


def handle_analyze_file(args: dict) -> tuple[str, bool]:
    file_path = _require_str(args, "file_path")
    token_budget = args.get("token_budget", 500)
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) \
            or token_budget <= 0:
        raise ToolInputError(
            f"Invalid 'token_budget': {token_budget!r} (expected a positive integer)"
        )
    query = args.get("query", "")
    if not isinstance(query, str):
        raise ToolInputError(f"Invalid 'query': expected string, got {type(query).__name__}")

    try:
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}", True

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
        ), False
    except Exception as e:
        logger.warning(f"analyze_file failed for {file_path}: {type(e).__name__}: {e}")
        return f"File analysis error: {type(e).__name__}: {e}", True


def handle_why_decision(args: dict) -> tuple[str, bool]:
    session_id = _require_str(args, "session_id")
    query = _require_str(args, "query")
    result = _get(f"/api/graph/{quote(session_id, safe='')}/why?q={quote(query)}")
    offline = _is_offline(result)
    if offline:
        try:
            result = _memory(session_id).why(query)
        except Exception as e:
            logger.exception("Local reasoning query failed")
            return (f"Reasoning query failed, and TokenMizer is not running "
                    f"at {TOKENMIZER_URL}: {type(e).__name__}: {e}"), True
    if "error" in result:
        return f"Reasoning query failed: {result['error']}", True

    matches = result.get("matches", [])
    chain = result.get("chain", [])
    current = result.get("current")

    if not matches:
        return (
            f"No decision matching '{query}' found in session '{session_id}'. "
            f"Try a shorter substring, or use get_graph_stats to see what "
            f"the graph contains." + (_local_note() if offline else "")
        ), False

    lines = [f"Decision trail for '{query}' — session: {session_id}", ""]
    if chain:
        for t in chain:
            lines.append(f"  replaced: {t['from_label']}")
            hop = f"      └─ replaced by: {t['to_label']}"
            if t.get("reason"):
                hop += f" — {t['reason']}"
            lines.append(hop)
            if t.get("evidence"):
                lines.append(f"         evidence: {t['evidence']}")
    else:
        lines.append("  (no supersessions — this decision has never changed)")
    lines.append("")
    if current:
        lines.append(f"  current:  {current['label']}"
                     + (f" — {current['summary']}" if current.get("summary") else ""))
    else:
        lines.append("  Note: no active decision on this topic (superseded or "
                     "invalidated without replacement).")
    return "\n".join(lines) + (_local_note() if offline else ""), False


def handle_get_savings_stats(args: dict) -> tuple[str, bool]:
    result = _get("/api/stats")
    if _is_offline(result):
        # Analytics are recorded by the proxy as requests pass through it.
        # There is nothing local to read: a session checkpointed straight
        # into the graph never went through a pipeline that could save
        # anything. Say that rather than reporting zeros as a result.
        return (
            f"No savings to report: TokenMizer is not running at "
            f"{TOKENMIZER_URL}, and savings are measured on requests that "
            f"pass through the proxy. Graph memory still works without it "
            f"(checkpoint, resume, graph stats and why all read "
            f"{TOKENMIZER_STORAGE_DIR} directly). Start `tokenmizer serve` "
            f"and route your client at it to see savings."
        ), False
    if "error" in result:
        return f"Stats failed: {result['error']}", True
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
    return "\n".join(lines), False


# ── MCP stdio transport ───────────────────────────────────────────────────────

def handle_tool_call(name: str, arguments: Any) -> tuple[str, bool]:
    """Dispatch a tools/call. Returns (text, is_error) — never raises."""
    handlers = {
        "checkpoint_session": handle_checkpoint_session,
        "resume_session": handle_resume_session,
        "get_graph_stats": handle_get_graph_stats,
        "analyze_file": handle_analyze_file,
        "get_savings_stats": handle_get_savings_stats,
        "why_decision": handle_why_decision,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name!r}. Available: {sorted(handlers)}", True
    if not isinstance(arguments, dict):
        return (
            f"Invalid arguments for tool '{name}': expected a JSON object, "
            f"got {type(arguments).__name__}"
        ), True
    try:
        return handler(arguments)
    except ToolInputError as e:
        logger.warning(f"Tool '{name}' rejected input: {e}")
        return f"Invalid input for tool '{name}': {e}", True
    except Exception as e:
        logger.exception(f"Tool '{name}' crashed")
        return f"Tool '{name}' internal error: {type(e).__name__}: {e}", True


def _handle_request(req: dict, send) -> None:
    """Handle one parsed JSON-RPC request object."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    is_notification = "id" not in req

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
        result_text, is_error = handle_tool_call(tool_name, arguments)
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": is_error,
            },
        })

    elif method.startswith("notifications/"):
        pass  # notifications never get a response

    elif not is_notification:
        send({"jsonrpc": "2.0", "id": req_id,
              "error": {"code": -32601, "message": f"Method not found: {method}"}})


def run_stdio_server():
    """
    MCP stdio transport — reads JSON-RPC from stdin, writes to stdout.
    Newline-delimited JSON framing (one object per line, both directions).

    Robustness contract: no input may terminate the read loop. Malformed
    JSON returns -32700, non-object messages return -32600, and handler
    exceptions return -32603 — each with a log line. Logging goes to
    stderr; stdout carries only protocol frames.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("TOKENMIZER_MCP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s tokenmizer-mcp: %(message)s",
    )

    def send(obj: dict):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def send_error(req_id: Any, code: int, message: str):
        send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    logger.info(f"tokenmizer MCP stdio server started (proxy: {TOKENMIZER_URL})")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"Dropping malformed JSON line ({e}): {line[:200]!r}")
            send_error(None, -32700, f"Parse error: {e}")
            continue

        if not isinstance(req, dict):
            logger.warning(f"Dropping non-object JSON-RPC message: {line[:200]!r}")
            send_error(None, -32600, "Invalid request: expected a JSON object")
            continue

        try:
            _handle_request(req, send)
        except (BrokenPipeError, OSError):
            # stdout is gone — client disconnected mid-write; nothing left
            # to reply to. Exit the loop cleanly.
            logger.info("Client disconnected (broken pipe); shutting down")
            return
        except Exception as e:
            logger.exception(f"Unhandled error in method {req.get('method')!r}")
            try:
                send_error(req.get("id"), -32603,
                           f"Internal error: {type(e).__name__}: {e}")
            except Exception:
                logger.error("Could not send error response; shutting down")
                return

    logger.info("stdin closed; MCP server exiting")


if __name__ == "__main__":
    run_stdio_server()
