"""
The shipped MCP manifests must launch a console script, not an interpreter.

`"command": "python3"` is dead on Windows: python3.exe there is the Microsoft
Store App Execution Alias, so the MCP server never starts and the user gets
"Python was not found; run without arguments to install from the Microsoft
Store" instead of a working plugin. It is also wrong inside a virtualenv on
any platform, where `python3` may resolve outside the env that has tokenmizer
installed.

The `tokenmizer-mcp` entry point declared in pyproject.toml resolves correctly
on all three platforms and inside virtualenvs. These tests keep the manifests
and the entry point from drifting apart — the failure mode is silent at
install time and only shows up as a broken plugin.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

MANIFESTS = [
    ROOT / ".mcp.json",
    ROOT / ".claude-plugin" / "plugin.json",
]

# Anything that resolves through PATH lookup of an interpreter rather than
# through an installed entry point.
_BARE_INTERPRETERS = {"python", "python3", "python.exe", "python3.exe", "py"}


def _project_scripts() -> dict[str, str]:
    """[project.scripts] from pyproject.toml.

    Parsed with a regex rather than tomllib: tomllib is Python 3.11+ and this
    package supports 3.10, and the table is flat `name = "module:attr"` lines,
    which is all a regex needs to be right about.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)", text,
                      re.MULTILINE | re.DOTALL)
    assert match, "[project.scripts] not found in pyproject.toml"
    return dict(re.findall(r'^\s*([\w-]+)\s*=\s*"([^"]+)"', match.group(1),
                           re.MULTILINE))



def _server_blocks(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("mcpServers") or {}).items()


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_does_not_launch_a_bare_interpreter(path):
    for name, block in _server_blocks(path):
        assert block.get("command") not in _BARE_INTERPRETERS, (
            f"{path.name}: server {name!r} launches "
            f"{block['command']!r}, which does not resolve on Windows "
            "(Microsoft Store alias) or inside a virtualenv. Use the "
            "tokenmizer-mcp console script."
        )


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_command_is_a_declared_entry_point(path):
    scripts = _project_scripts()

    for name, block in _server_blocks(path):
        command = block.get("command")
        assert command in scripts, (
            f"{path.name}: server {name!r} runs {command!r}, which is not in "
            f"[project.scripts] ({sorted(scripts)}). Installing the package "
            "would not put it on PATH."
        )


def test_mcp_entry_point_target_is_importable_and_callable():
    """The entry point must name something that actually exists."""
    scripts = _project_scripts()
    module_path, _, attr = scripts["tokenmizer-mcp"].partition(":")

    import importlib
    module = importlib.import_module(module_path)
    assert callable(getattr(module, attr)), (
        f"{scripts['tokenmizer-mcp']} does not resolve to a callable"
    )
