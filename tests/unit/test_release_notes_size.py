"""The release notes must fit in a GitHub release.

The release workflow publishes to PyPI first and creates the tag and the GitHub
release from the CHANGELOG section afterwards. If that section is longer than
GitHub allows, the release step fails after PyPI has already accepted the
upload: a published version with no release page. This keeps the section in the
limit while it can still be shortened, instead of finding out at that point.
"""
from __future__ import annotations

import re
from pathlib import Path

import tokenmizer

ROOT = Path(__file__).resolve().parents[2]
GITHUB_RELEASE_BODY_LIMIT = 125_000


def _section(text: str, version: str) -> str:
    """The body the release workflow publishes for `version`: the same
    pattern as .github/workflows/release.yml."""
    m = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text, re.MULTILINE | re.DOTALL)
    assert m, f"CHANGELOG.md has no section for {version}"
    return m.group(1).strip()


def test_the_current_versions_notes_fit_in_a_github_release():
    body = _section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
                    tokenmizer.__version__)
    assert len(body) <= GITHUB_RELEASE_BODY_LIMIT, (
        f"the {tokenmizer.__version__} release notes are {len(body):,} characters; GitHub "
        f"allows {GITHUB_RELEASE_BODY_LIMIT:,}. Shorten the CHANGELOG section before "
        f"releasing, or the release step fails after PyPI has the upload."
    )


def test_the_guard_measures_the_section_the_workflow_publishes():
    text = "# Changelog\n\n## [Unreleased]\n\nnot published\n\n## [2.0.0] — d — t\n\nbody\n\n## [1.0.0] — d — t\n\nolder\n"
    assert _section(text, "2.0.0") == "body"
