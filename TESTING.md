# Testing

The suite is 600 tests under `pytest`, and it is the source of truth —
if a claim elsewhere in the docs disagrees with what the suite does, the
suite is right and the docs are a bug.

```bash
pip install -e ".[dev]"
pytest tests/ -v              # 600 tests
ruff check tokenmizer/ tests/ # lint, import order
```

CI runs this matrix on every push: Python 3.10–3.13 on Linux, 3.12 on
Windows, plus a Docker build checked with the network removed
(`--network none`) and a migration job that opens a database written by
the previous release. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Coverage floor

`pyproject.toml`'s `[tool.coverage.report]` sets `fail_under` on product
code only — nothing is excluded to inflate the number, and the floor is
kept below the measured figure on purpose so an incidental dip doesn't
red the build while a real collapse still does.

## Extra checks, not part of CI

Two standalone scripts, useful when iterating locally:

```bash
python3 scripts/run_stdlib_tests.py    # zero-dependency subset, for quick sanity checks
python3 scripts/static_audit.py        # naive unused-import / broad-except scanner
```

`static_audit.py` is intentionally blunt and has known false positives —
it does not understand `TYPE_CHECKING`-guarded imports used only in
string type annotations, which account for most of what it flags in this
codebase. Treat its output as a prompt to go look, not as a finding on
its own; `ruff` (in CI, enforced) is the authoritative linter.

## Extraction quality

Correctness of the graph-memory extractor is not a unit-test property —
it is measured against a labelled corpus and reported as precision,
recall and F1, not pass/fail. See
[`docs/benchmarks.md`](docs/benchmarks.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md#improving-extraction).
