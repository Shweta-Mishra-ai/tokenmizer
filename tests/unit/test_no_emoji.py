"""
No emoji in anything a user of this package sees.

Every surface — CLI output, MCP replies, the dashboard, the plugin skills,
benchmark runners, install scripts, README and docs — is plain text. Emoji
carry nothing the structure does not already carry, and they are the first
thing that makes a tool read as a demo rather than something to depend on.

Typography is not emoji: em dashes, arrows, the ≥ sign and box-drawing
characters (used by the /why decision trail to render a tree, the same way
`tree` and `git log --graph` do) are deliberately allowed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SURFACES = [
    *ROOT.glob("tokenmizer/**/*.py"),
    *ROOT.glob("benchmarks/**/*.py"),
    *ROOT.glob("scripts/*"),
    *ROOT.glob(".claude-plugin/**/*.md"),
    *ROOT.glob(".claude-plugin/**/*.json"),
    ROOT / "README.md", ROOT / "USAGE.md", ROOT / "TESTING.md",
    ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md",
    *ROOT.glob("docs/*.md"),
    ROOT / "tokenmizer.yaml",
]
SURFACES = [p for p in SURFACES if p.is_file() and "__pycache__" not in p.parts]


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return (
        0x1F000 <= o <= 0x1FAFF      # emoticons, symbols, pictographs, extended
        or 0x2600 <= o <= 0x27BF     # misc symbols and dingbats (check/cross marks)
        or 0x2B00 <= o <= 0x2BFF     # misc symbols and arrows (stars)
        or o == 0xFE0F               # variation selector that makes a glyph emoji
    )


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: str(p.relative_to(ROOT)))
def test_surface_has_no_emoji(path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip("binary")
    found = sorted({ch for ch in text if _is_emoji(ch)})
    assert not found, (
        f"{path.relative_to(ROOT)} contains {[hex(ord(c)) for c in found]}; "
        "use words"
    )
