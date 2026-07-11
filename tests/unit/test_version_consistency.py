"""
Version-consistency guard.

WHY THIS EXISTS: version drift has shipped repeatedly in this repo —
the 2026-07-10 audit found `server.json` at 0.2.6 (three releases stale)
and `.claude-plugin/plugin.json` at 0.2.3 (four releases stale) while
pyproject said 0.3.1. Any MCP registry or plugin marketplace reading
those manifests displayed the wrong version. This test makes every
version pin agree with `tokenmizer.__version__` so drift fails CI on the
same push that introduces it.
"""
import json
import re
from pathlib import Path

import tokenmizer

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_matches_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no version field"
    assert m.group(1) == tokenmizer.__version__


def test_server_json_matches_package():
    data = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    versions = []
    if "version" in data:
        versions.append(data["version"])
    for pkg in data.get("packages", []):
        if "version" in pkg:
            versions.append(pkg["version"])
    assert versions, "server.json has no version fields"
    for v in versions:
        assert v == tokenmizer.__version__, (
            f"server.json pins {v}, package is {tokenmizer.__version__} — "
            f"the MCP registry would display the wrong version"
        )


def test_plugin_manifest_matches_package():
    data = json.loads(
        (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert data["version"] == tokenmizer.__version__, (
        f"plugin.json pins {data['version']}, package is "
        f"{tokenmizer.__version__} — plugin marketplaces would display "
        f"the wrong version"
    )


def test_fastapi_app_version_matches_package():
    text = (ROOT / "tokenmizer" / "api" / "app.py").read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', text)
    assert m, "app.py FastAPI(version=...) pin not found"
    assert m.group(1) == tokenmizer.__version__, (
        f"/docs page would show {m.group(1)}, package is {tokenmizer.__version__}"
    )


def test_changelog_has_entry_for_current_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{tokenmizer.__version__}]" in text, (
        f"CHANGELOG.md has no section for {tokenmizer.__version__} — "
        f"retitle the [Unreleased] entry before releasing"
    )
