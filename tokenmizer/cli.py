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
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{server}/api/stats"
    if session_id:
        url += f"?session_id={session_id}"

    try:
        data = httpx.get(url, headers=headers, timeout=5).json()
    except Exception as e:
        console.print(f"[red]Cannot reach server: {e}[/red]")
        raise typer.Exit(1)

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
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = httpx.post(
        f"{server}/api/checkpoint?session_id={session_id}",
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        console.print(f"[red]Error: {r.text}[/red]")
        raise typer.Exit(1)

    data = r.json()
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
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = httpx.get(
        f"{server}/api/resume/{session_id}?level={level}",
        headers=headers,
        timeout=10,
    )
    if r.status_code == 404:
        console.print(f"[yellow]No checkpoint found for session: {session_id}[/yellow]")
        raise typer.Exit(1)

    data = r.json()
    console.print(Panel(
        data["resume_context"],
        title=f"[green]Resume — {session_id[:16]}... ({data['token_count']} tokens)[/green]",
        border_style="green",
    ))


if __name__ == "__main__":
    app()
