"""
Runs tests/js/dashboard_auth.test.mjs — behavioral tests for the
dashboard's client-side auth-retry logic, executed against the real
script extracted from DASHBOARD_HTML rather than reimplemented.

See GitHub issue #36: the other dashboard test (test_dashboard.py) only
asserts that certain substrings are present in the HTML, which a subtly
wrong implementation would still pass. This shells out to Node (skipping,
not failing, if it isn't on PATH — mirroring this repo's convention of
graceful degradation for optional external tooling) to actually execute
dashboardFetch()/requestApiKey() and verify the concurrent-401, cancel,
and stale-key paths behave as intended.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_JS_TEST = Path(__file__).parent.parent / "js" / "dashboard_auth.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_dashboard_auth_behavior():
    result = subprocess.run(
        ["node", "--test", str(_JS_TEST)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, (
        "dashboard auth JS behavioral tests failed — see stdout/stderr above"
    )
