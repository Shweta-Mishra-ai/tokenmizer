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
    """Checked at runtime, on the object /docs and /openapi.json read from.

    Grepping the source for a literal was the previous check. It passed for
    four releases while the literal itself was stale, because a test that
    compares a hardcoded string to another hardcoded string only fails when
    someone updates exactly one of them.
    """
    from tokenmizer.api.app import app
    assert app.version == tokenmizer.__version__, (
        f"/docs would show {app.version}, package is {tokenmizer.__version__}"
    )


def test_fastapi_app_does_not_hardcode_a_version():
    """The version must be derived from `__version__`, not written out.

    A literal here is a string nobody thinks to update at release time.
    Deriving it removes the drift instead of testing for it.
    """
    text = (ROOT / "tokenmizer" / "api" / "app.py").read_text(encoding="utf-8")
    body = text.split("app = FastAPI(", 1)[-1].split(")", 1)[0]
    assert not re.search(r'version\s*=\s*["\']\d+\.\d+\.\d+', body), (
        "app.py pins a literal version — use `version=__version__`"
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
    On the README, above the fold, the first thing anyone sees.
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


def test_changelog_has_no_versions_that_were_never_released():
    """Every `## [x.y.z]` heading must be a version that exists as a release.

    Six headings once sat above the last published one — 0.4.1 through
    0.7.0 — none of which was ever on PyPI. A changelog that lists
    versions nobody can install is worse than no changelog: it makes the
    published version look stale and the release history look forged.

    The rule this enforces: a version heading is created when the release
    is cut, not while the work is in progress. Work in progress belongs
    under the current version's heading.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r'^## \[(\d+\.\d+\.\d+)\]', text, re.MULTILINE)
    assert headings, "CHANGELOG.md has no version headings"

    def as_tuple(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    current = as_tuple(tokenmizer.__version__)
    ahead = [v for v in headings if as_tuple(v) > current]
    assert not ahead, (
        f"CHANGELOG.md documents {ahead} above the package version "
        f"{tokenmizer.__version__}. Those versions do not exist."
    )
    assert headings[0] == tokenmizer.__version__, (
        f"the newest CHANGELOG heading is {headings[0]}, package is "
        f"{tokenmizer.__version__}"
    )


def test_no_source_file_hardcodes_a_stale_version():
    """A version baked into prose or a comment goes stale silently.

    The logo carried `v0.2.3` for four releases before anyone noticed,
    on the README above the fold. This sweeps every tracked text file
    for a `vX.Y.Z` that is neither the current version nor one that was
    genuinely released earlier.
    """
    released_or_current = {tokenmizer.__version__}
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released_or_current |= set(
        re.findall(r'^## \[(\d+\.\d+\.\d+)\]', changelog, re.MULTILINE))

    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {
                ".py", ".md", ".json", ".yaml", ".yml", ".toml", ".svg"}:
            continue
        rel = path.relative_to(ROOT)
        if any(part in {".git", "node_modules", "dist", "build", ".venv"}
               for part in rel.parts):
            continue
        if rel.name == "CHANGELOG.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for found in set(re.findall(r'\bv(\d+\.\d+\.\d+)\b', text)):
            if found not in released_or_current:
                offenders.append(f"{rel}: v{found}")

    assert not offenders, (
        "version strings referring to releases that do not exist:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_readme_test_count_matches_reality():
    """The README quotes a test count next to a command you can run.

    It said 495 while the suite had 578 — a reader's very first command
    disagreeing with the README is a bad first impression, and nothing
    catches it because the number lives in prose. A tolerance is allowed
    so that adding one test does not red the build; a drift of more than
    5% means the number was simply never updated.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    m = re.search(r'(\d+)\s+tests? collected', out)
    assert m, f"could not read the collected count from pytest:\n{out[-500:]}"
    actual = int(m.group(1))

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    quoted = [int(x) for x in re.findall(r'#\s*(\d+) tests', text)]
    assert quoted, "README no longer quotes a test count — drop this guard too"

    for n in quoted:
        assert abs(n - actual) <= max(5, actual * 0.05), (
            f"README says {n} tests, the suite collects {actual}"
        )
