"""TokenMizer CLI"""
from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Force UTF-8 output streams: Windows consoles default to cp1252, which
# cannot encode the emoji in help/output text and would crash the CLI
# before printing anything.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass  # non-reconfigurable stream (e.g. pytest capture) — harmless

app = typer.Typer(
    name="tokenmizer",
    help="🧠 TokenMizer — Never lose your AI context again.",
    add_completion=False,
)
console = Console()


# ── Shared HTTP helpers for the stats/checkpoint/resume commands ────────────
#
# FIXED: `checkpoint` and `resume` used to call httpx.post/httpx.get with NO
# error handling at all — unlike `stats`, which already wrapped its call and
# printed a clean "Cannot reach server" message. An unreachable server (the
# single most common real-world CLI failure mode) crashed both commands with
# a raw, unhandled traceback instead of a clean error and exit code. Both
# commands now go through the same two helpers `stats` already used the
# pattern for, so all three fail the same way.

def _cli_get(url: str, headers: dict, timeout: float):
    import httpx
    try:
        return httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        console.print(f"[red]Cannot reach server: {e}[/red]")
        raise typer.Exit(1)


def _cli_post(url: str, headers: dict, timeout: float):
    import httpx
    try:
        return httpx.post(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        console.print(f"[red]Cannot reach server: {e}[/red]")
        raise typer.Exit(1)


def _require_fields(data: dict, *fields: str) -> bool:
    """
    FIXED: `checkpoint` and `resume` used to access response fields with
    direct dict keys (data['checkpoint_id'], data["resume_context"]). A
    non-200-non-404 response (auth failure, validation error, upstream 500)
    has a different body shape than a successful one, and direct key access
    on that raised a raw KeyError instead of surfacing what actually went
    wrong. Returns False (and prints a clean message) if any required field
    is missing; caller should typer.Exit(1) in that case.
    """
    missing = [f for f in fields if f not in data]
    if missing:
        console.print(
            f"[red]Unexpected server response — missing field(s): "
            f"{', '.join(missing)}. Raw response: {data!r}[/red]"
        )
        return False
    return True


@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None, help="Bind host (default: proxy_host from tokenmizer.yaml, or 127.0.0.1)"
    ),
    port: Optional[int] = typer.Option(
        None, help="Bind port (default: proxy_port from tokenmizer.yaml, or 8000)"
    ),
    config: Optional[str] = typer.Option(None, help="Path to tokenmizer.yaml"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start TokenMizer proxy + dashboard."""
    import os

    import uvicorn

    if config:
        os.environ["TOKENMIZER_CONFIG"] = config

    # FIXED: host/port used to be hardcoded typer defaults ("0.0.0.0",
    # 8000) completely independent of Settings.proxy_host/proxy_port —
    # tokenmizer.yaml shipped those as documented config, but editing
    # them did nothing at all. Now an explicit --host/--port flag always
    # wins (keeps the Dockerfile's hardcoded `--host 0.0.0.0` unaffected);
    # absent a flag, the config file's value is used.
    from tokenmizer.config.settings import get_settings
    settings = get_settings()
    actual_host = host if host is not None else settings.proxy_host
    actual_port = port if port is not None else settings.proxy_port

    if workers > 1:
        console.print(
            f"[yellow]Warning:[/yellow] --workers {workers} starts multiple "
            "OS processes. TokenMizer's SQLite-backed graph/checkpoint "
            "storage is NOT safe for concurrent multi-process writers to "
            "the same session — the last process to persist a session "
            "silently overwrites what an earlier one wrote (tracked as "
            "issue #27). Until that lands: run a single worker, or route "
            "each session_id to a fixed worker at your load balancer."
        )

    console.print(Panel.fit(
        "[bold green]🧠 TokenMizer[/bold green]\n"
        f"[dim]Proxy:     http://{actual_host}:{actual_port}/v1/chat/completions[/dim]\n"
        f"[dim]Dashboard: http://{actual_host}:{actual_port}[/dim]\n"
        f"[dim]API Docs:  http://{actual_host}:{actual_port}/docs[/dim]\n"
        f"[dim]Health:    http://{actual_host}:{actual_port}/health[/dim]",
        border_style="green",
    ))

    uvicorn.run(
        "tokenmizer.api.app:app",
        host=actual_host,
        port=actual_port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


@app.command()
def stats(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (optional)"),
    server: str = typer.Option("http://localhost:8000", help="TokenMizer server URL"),
    api_key: Optional[str] = typer.Option(None, envvar="TOKENMIZER_API_KEY"),
):
    """Print session/global analytics."""
    from urllib.parse import quote

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{server}/api/stats"
    if session_id:
        # FIXED: session_id used to be interpolated raw into the query
        # string — a session_id containing '&', space, or other reserved
        # URL characters produced a malformed or misdirected request
        # (same class of bug already fixed in the MCP server).
        url += f"?session_id={quote(session_id, safe='')}"

    r = _cli_get(url, headers, timeout=5)
    if r.status_code != 200:
        # FIXED: previously there was no status check at all — a non-200
        # response (e.g. an auth failure) still had `.json()` called on
        # it, which often succeeds and returns something like
        # {"detail": "..."}; `.get("daily", {})` on that silently
        # returned {} and the command printed all-zero stats with no
        # indication anything had gone wrong.
        console.print(f"[red]Error: {r.text}[/red]")
        raise typer.Exit(1)
    data = r.json()

    d = data.get("daily", {})
    console.print(Panel.fit(
        f"[bold]Daily Stats[/bold]\n"
        f"[green]Requests:     {d.get('requests', 0):,}[/green]\n"
        f"[green]Tokens saved: {d.get('tokens_saved', 0):,} ({d.get('savings_pct', 0):.1f}%)[/green]\n"
        f"[yellow]Cost saved:   ${d.get('cost_saved_usd', 0):.4f}[/yellow]",
        border_style="green",
    ))


@app.command()
def checkpoint(
    session_id: str = typer.Argument(..., help="Session ID to checkpoint"),
    server: str = typer.Option("http://localhost:8000"),
    api_key: Optional[str] = typer.Option(None, envvar="TOKENMIZER_API_KEY"),
    level: str = typer.Option("standard", help="Resume level: critical | standard | full"),
):
    """Create a manual checkpoint and show resume context."""
    from urllib.parse import quote

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = _cli_post(
        f"{server}/api/checkpoint?session_id={quote(session_id, safe='')}",
        headers, timeout=30,
    )
    if r.status_code != 200:
        console.print(f"[red]Error: {r.text}[/red]")
        raise typer.Exit(1)

    data = r.json()
    if not _require_fields(data, "checkpoint_id", "resume_tokens"):
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[green]✅ Checkpoint created[/green]\n"
        f"[dim]ID:            {data['checkpoint_id']}[/dim]\n"
        f"[dim]Nodes:         {data.get('node_count', 0)}[/dim]\n"
        f"[dim]Resume tokens: {data['resume_tokens']}[/dim]\n\n"
        f"[bold]Resume context ({level}):[/bold]\n"
        f"{data.get('resume_standard', '')}",
        border_style="green",
    ))


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
    server: str = typer.Option("http://localhost:8000"),
    level: str = typer.Option("standard", help="critical | standard | full"),
    api_key: Optional[str] = typer.Option(None, envvar="TOKENMIZER_API_KEY"),
):
    """Get the resume context for a session checkpoint."""
    from urllib.parse import quote

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = _cli_get(
        f"{server}/api/resume/{quote(session_id, safe='')}?level={level}",
        headers, timeout=10,
    )
    if r.status_code == 404:
        console.print(f"[yellow]No checkpoint found for session: {session_id}[/yellow]")
        raise typer.Exit(1)
    if r.status_code != 200:
        # FIXED: previously ONLY 404 was special-cased — any OTHER
        # failure status (401, 500, ...) fell through to `data =
        # r.json()` and then `data["resume_context"]`, which raised a
        # raw KeyError on that response's different body shape instead
        # of surfacing the actual server error.
        console.print(f"[red]Error: {r.text}[/red]")
        raise typer.Exit(1)

    data = r.json()
    if not _require_fields(data, "resume_context", "token_count"):
        raise typer.Exit(1)

    console.print(Panel(
        data["resume_context"],
        title=f"[green]Resume — {session_id[:16]}... ({data['token_count']} tokens)[/green]",
        border_style="green",
    ))


if __name__ == "__main__":
    app()
