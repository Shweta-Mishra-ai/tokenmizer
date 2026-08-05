"""
Version-consistency guard.

Every manifest that pins a version (pyproject, server.json, plugin.json,
marketplace.json, the FastAPI app, the changelog) must agree with
`tokenmizer.__version__`. Registries and marketplaces read these files
directly, so any drift ships a wrong version string; this test fails CI
on the push that introduces it.
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


def test_marketplace_manifest_matches_package():
    data = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert data["version"] == tokenmizer.__version__, (
        f"marketplace.json pins {data['version']}, package is "
        f"{tokenmizer.__version__}"
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


def test_no_asset_hardcodes_a_version():
    """No committed asset may bake in a version string.

    The logo SVG carried `v0.2.3` in a badge while the package was on
    0.6.0 — on the README, above the fold, the first thing anyone sees.
    Nothing caught it because nothing looked. Version belongs in exactly
    one place (pyproject) plus badges that render it live from a
    registry; anywhere else it is a stale claim waiting to happen.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    version_like = re.compile(r"\bv?\d+\.\d+\.\d+\b")
    offenders = []

    for asset in list((root / "docs").rglob("*.svg")):
        text = asset.read_text(encoding="utf-8", errors="ignore")
        # Strip XML comments: explaining the past mistake is allowed,
        # rendering a version is not.
        rendered = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        for m in version_like.findall(rendered):
            offenders.append(f"{asset.relative_to(root)}: {m}")

    assert not offenders, (
        "assets must not hardcode a version — it goes stale on every "
        f"release and nothing catches it: {offenders}"
    )


def test_readme_documents_every_endpoint_and_no_others():
    """The API table must match the route decorators, in both directions.

    Found six live endpoints missing from the table, and one documented
    endpoint (`/api/analyze`) that did not exist. Docs drift silently;
    a script does not.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")

    declared = set()
    for py in (root / "tokenmizer" / "api").rglob("*.py"):
        declared |= set(re.findall(
            r'@(?:app|router)\.(?:get|post)\("([^"]+)"', py.read_text(encoding="utf-8")
        ))

    documented = set(re.findall(r"`(/(?:api|v1)/[^`]*?)`", readme))

    def norm(path: str) -> str:
        # {session_id} and {id} name the same thing in prose vs code.
        return re.sub(r"\{[^}]+\}", "{}", path.split("?")[0].rstrip("/"))

    declared_n = {norm(x) for x in declared}
    documented_n = {norm(x) for x in documented}

    phantom = sorted(d for d in documented if norm(d) not in declared_n)
    assert not phantom, f"README documents endpoints that do not exist: {phantom}"

    # `/` and `/health` are conventional and not worth a table row.
    undocumented = sorted(
        d for d in declared
        if norm(d) not in documented_n and d not in ("/", "/health")
    )
    assert not undocumented, f"live endpoints missing from the README: {undocumented}"
