"""
No unreachable statements in the package.

A refactor of _get_cheap_provider() left the whole old function body
sitting after the new `return`. Every test passed — the dead block was
never executed — and nothing in the suite or the linter looked for it.
Code that cannot run is not harmless: it is the next reader's wrong
model of what the function does. This walks every function and fails
on any statement that follows a return/raise/continue/break at the
same block level.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(p for p in (ROOT / "tokenmizer").rglob("*.py") if "__pycache__" not in p.parts)

_TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _unreachable(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, _TERMINAL):
                    lines.append(block[i + 1].lineno)
                    break
    return lines


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_unreachable_statements(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = _unreachable(tree)
    assert not found, f"{path.relative_to(ROOT)}: unreachable code at line(s) {found}"
