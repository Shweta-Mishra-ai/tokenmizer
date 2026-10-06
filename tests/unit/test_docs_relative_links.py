"""Every relative link and image in the docs points at a file that exists.

`docs/architecture.md` embedded `docs/assets/architecture.svg`. That path is
right in the README, which sits at the repository root, and wrong one
directory down, where it resolves to `docs/docs/assets/...`: the diagram
rendered as a broken image on GitHub. Nothing noticed, because the existing
guard checks the `#fragment` of links to other Markdown files, never whether
an image or any other relative target exists.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNED = [
    ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "TESTING.md",
    ROOT / "SECURITY.md", ROOT / "CHANGELOG.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|(?:src|href)=\"([^\"]+)\"")
_NOT_A_FILE = ("http://", "https://", "mailto:", "#", "data:", "/")


def _relative_targets(path: Path) -> list[str]:
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    targets = []
    for match in _LINK.finditer(text):
        target = (match.group(1) or match.group(2)).strip()
        if target.startswith(_NOT_A_FILE):
            continue
        target = target.split("#")[0].split("?")[0]
        if target:
            targets.append(target)
    return targets


@pytest.mark.parametrize("doc", [p for p in SCANNED if p.exists()], ids=lambda p: p.name)
def test_relative_links_and_images_resolve(doc):
    broken = [t for t in _relative_targets(doc) if not (doc.parent / t).exists()]
    assert not broken, (
        f"{doc.relative_to(ROOT)} links to files that do not exist: {broken}. "
        "A relative path resolves from the file's own directory, not the repo root."
    )


def test_the_guard_would_have_caught_the_original_mistake(tmp_path):
    docs = tmp_path / "docs"
    (docs / "assets").mkdir(parents=True)
    (docs / "assets" / "diagram.svg").write_text("<svg/>")
    page = docs / "page.md"
    page.write_text('<img src="docs/assets/diagram.svg"/> <img src="assets/diagram.svg"/>')

    broken = [t for t in _relative_targets(page) if not (page.parent / t).exists()]

    assert broken == ["docs/assets/diagram.svg"]
