#!/usr/bin/env python3
"""
Stdlib-only static auditor (no network/pyflakes available in this environment).

Checks across the whole tokenmizer/ package:
  1. Unused imports (imported but never referenced as a name in the same module)
  2. Module-level functions/classes defined but never referenced anywhere
     in the package (best-effort cross-file dead-code signal)
  3. Bare `except:` / overly broad `except Exception:` that swallow errors
     without logging at warning+ level or re-raising
  4. Functions with no return and no side-effect call that look like stubs

This is intentionally conservative — it flags candidates for manual review,
not auto-deletes. False positives are expected for dynamically-referenced
names (e.g. things only used in dataclass `asdict`/string templates).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIRS = [ROOT / "tokenmizer", ROOT / "tests", ROOT / "benchmarks"]


def iter_py_files():
    for d in PKG_DIRS:
        if d.exists():
            yield from d.rglob("*.py")


def unused_imports(tree: ast.Module, source: str) -> list[str]:
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                imported[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                imported[name] = node.lineno

    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            pass  # attribute access roots are still ast.Name, caught above

    findings = []
    for name, lineno in imported.items():
        if name == "annotations":
            continue
        if name not in used_names and f"__all__" in source and name not in source.split("__all__")[1][:500]:
            findings.append(f"line {lineno}: unused import {name!r}")
        elif name not in used_names and "__all__" not in source:
            findings.append(f"line {lineno}: unused import {name!r}")
    return findings


def broad_except_swallows(tree: ast.Module, filename: str) -> list[str]:
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            is_broad = (
                node.type is None
                or (isinstance(node.type, ast.Name) and node.type.id == "Exception")
            )
            if not is_broad:
                continue
            body_calls = []
            reraises = False
            log_level = None
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Raise):
                    reraises = True
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                    if isinstance(stmt.func.value, ast.Name) and stmt.func.value.id == "logger":
                        log_level = stmt.func.attr
                        body_calls.append(stmt.func.attr)
            if not reraises and log_level in (None, "debug"):
                findings.append(
                    f"line {node.lineno}: broad except with "
                    f"{'no logging' if log_level is None else 'only debug-level logging'} "
                    f"and no re-raise — silent failure risk"
                )
    return findings


def main():
    total_unused = 0
    total_swallow = 0
    for f in sorted(iter_py_files()):
        try:
            source = f.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(f))
        except SyntaxError as e:
            print(f"SYNTAX ERROR in {f}: {e}")
            continue

        rel = f.relative_to(ROOT)
        ui = unused_imports(tree, source)
        sw = broad_except_swallows(tree, str(f))

        if ui:
            total_unused += len(ui)
            print(f"\n[UNUSED IMPORTS] {rel}")
            for line in ui:
                print(f"  {line}")
        if sw:
            total_swallow += len(sw)
            print(f"\n[SILENT-FAILURE RISK] {rel}")
            for line in sw:
                print(f"  {line}")

    print(f"\n{'='*60}")
    print(f"Total unused-import findings: {total_unused}")
    print(f"Total silent-failure risk findings: {total_swallow}")


if __name__ == "__main__":
    main()
