"""TokenMizer CLI"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="tokenmizer",
    help="🧠 TokenMizer — Never lose your AI context again.",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    config: Optional[str] = typer.Option(None, help="Path to tokenmizer.yaml"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start TokenMizer proxy + dashboard."""
    import os

    import uvicorn

    if config:
        os.environ["TOKENMIZER_CONFIG"] = config

    console.print(Panel.fit(
        "[bold green]🧠 TokenMizer[/bold green]\n"
        f"[dim]Proxy:     http://{host}:{port}/v1/chat/completions[/dim]\n"
        f"[dim]Dashboard: http://{host}:{port}[/dim]\n"
        f"[dim]API Docs:  http://{host}:{port}/docs[/dim]\n"
        f"[dim]Health:    http://{host}:{port}/health[/dim]",
        border_style="green",
    ))

    uvicorn.run(
        "tokenmizer.api.app:app",
        host=host,
        port=port,
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
