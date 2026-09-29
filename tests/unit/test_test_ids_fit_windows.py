"""
A test id must fit in a Windows environment variable.

pytest keeps the id of the running test in an environment variable, and on
Windows that variable cannot exceed 32,767 characters. Parametrizing on a
huge string makes it the test's id: six tests parametrized on 400,000-
character inputs errored on Windows CI, at setup, while passing on Linux.
Parametrize on names and build large inputs inside the test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 8_000        # well under 32,767, so a near miss is caught too


def test_no_test_id_is_long_enough_to_break_on_windows():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/unit"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    ).stdout
    too_long = [line[:100] for line in out.splitlines() if len(line) > LIMIT]
    assert not too_long, (
        f"{len(too_long)} test id(s) over {LIMIT} characters, which error on "
        f"Windows (an environment variable holds at most 32,767): {too_long[:3]}"
    )
