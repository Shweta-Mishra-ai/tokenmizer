"""
MCP server end-to-end check.

Spawns the real MCP stdio server as a subprocess, performs the MCP
handshake (initialize → initialized → tools/list), then exercises real
tool calls against a running TokenMizer proxy (TOKENMIZER_URL) and a
local file analysis (no proxy needed).

Run:  python scripts/mcp_e2e_check.py
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _start_proxy_in_thread(port: int) -> None:
    """Run the real FastAPI app via uvicorn in a daemon thread."""
    import threading
    import time

    import uvicorn

    from tokenmizer.api.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):  # wait up to 5s for startup
        if server.started:
            return
        time.sleep(0.1)
    raise RuntimeError("proxy did not start within 5s")


def _seed_session_graph(session_id: str) -> None:
    """Put real nodes into the session graph BEFORE the proxy starts, using
    the same storage dir the proxy will read — so checkpoint/resume exercise
    the full real pipeline instead of an empty graph."""
    from tokenmizer.config.settings import get_settings
    from tokenmizer.graph_memory.graph import GraphMemory

    storage = get_settings().graph_checkpoint.storage_dir
    g = GraphMemory(session_id, storage_dir=storage)
    g.extract_from_messages([
        {"role": "user", "content": "Let's build a FastAPI auth service with JWT and PostgreSQL"},
        {"role": "assistant", "content": "Decided: PostgreSQL for storage. Completed: project setup. Working on: login endpoint in api/auth.py"},
    ], incremental=False)
    g._persist(force=True)


def main() -> int:
    port = 8765
    _seed_session_graph("mcp-e2e-test")
    _start_proxy_in_thread(port)
    os.environ["TOKENMIZER_URL"] = f"http://127.0.0.1:{port}"
    print(f"proxy up on :{port}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "tokenmizer.mcp.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8",
        env={**os.environ},
    )

    def rpc(method: str, params: dict | None = None, req_id: int | None = 1):
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if req_id is not None:
            msg["id"] = req_id
        if params is not None:
            msg["params"] = params
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        if req_id is None:
            return None
        line = proc.stdout.readline()
        return json.loads(line)

    try:
        # 1. Handshake
        r = rpc("initialize", {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "e2e", "version": "0"}})
        check("initialize returns serverInfo",
              r and r.get("result", {}).get("serverInfo", {}).get("name") == "tokenmizer")
        # Compare against the package version, not a hardcoded sentinel:
        # the old check asserted "not 0.2.3", which silently stopped
        # meaning anything the moment 0.2.4 shipped.
        from tokenmizer import __version__ as _pkg_version
        check("initialize reports the package version",
              r.get("result", {}).get("serverInfo", {}).get("version") == _pkg_version,
              f"expected {_pkg_version}, got {r.get('result', {}).get('serverInfo')}")
        rpc("notifications/initialized", req_id=None)

        # 2. tools/list
        r = rpc("tools/list", req_id=2)
        tools = {t["name"] for t in r.get("result", {}).get("tools", [])}
        expected = {"checkpoint_session", "resume_session", "get_graph_stats",
                    "analyze_file", "get_savings_stats", "why_decision"}
        check("tools/list exposes all 6 tools", tools == expected, str(tools))

        # 3. Local tool (no proxy): analyze_file on a real CSV
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         encoding="utf-8") as f:
            f.write("region,revenue\nnorth,100\nsouth,250\neast,90\nwest,410\n")
            csv_path = f.name
        r = rpc("tools/call", {"name": "analyze_file",
                               "arguments": {"file_path": csv_path}}, req_id=3)
        text = r.get("result", {}).get("content", [{}])[0].get("text", "")
        check("analyze_file returns analysis", "File Analysis" in text, text[:120])
        os.unlink(csv_path)

        # 4. Proxy-backed tools (require TOKENMIZER_URL server running)
        r = rpc("tools/call", {"name": "get_savings_stats", "arguments": {}}, req_id=4)
        text = r.get("result", {}).get("content", [{}])[0].get("text", "")
        check("get_savings_stats reaches proxy", "Savings Report" in text, text[:120])

        r = rpc("tools/call", {"name": "checkpoint_session",
                               "arguments": {"session_id": "mcp-e2e-test"}}, req_id=5)
        text = r.get("result", {}).get("content", [{}])[0].get("text", "")
        check("checkpoint_session creates checkpoint", "checkpointed" in text, text[:160])

        r = rpc("tools/call", {"name": "resume_session",
                               "arguments": {"session_id": "mcp-e2e-test"}}, req_id=6)
        text = r.get("result", {}).get("content", [{}])[0].get("text", "")
        check("resume_session returns context", "TokenMizer Resume" in text, text[:160])

        # Reasoning tool round-trips through the proxy. Whether the test
        # session happens to contain a matching decision or not, a healthy
        # server answers with a trail or a clean "no match" — never an error.
        r = rpc("tools/call", {"name": "why_decision",
                               "arguments": {"session_id": "mcp-e2e-test",
                                             "query": "postgres"}}, req_id=7)
        res = r.get("result", {})
        text = res.get("content", [{}])[0].get("text", "")
        check("why_decision answers without error",
              res.get("isError") is False
              and ("Decision trail" in text or "No decision matching" in text),
              text[:160])

        # 5. Unknown method → JSON-RPC error, not crash
        r = rpc("bogus/method", {}, req_id=8)
        check("unknown method returns -32601 error",
              r.get("error", {}).get("code") == -32601)

    finally:
        proc.stdin.close()
        proc.terminate()

    print()
    if FAILURES:
        print(f"E2E RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("E2E RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
