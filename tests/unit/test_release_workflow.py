"""
Static checks on the release workflow.

This is the one piece of automation whose mistakes cannot be undone. A
version uploaded to PyPI can never be replaced or reused, so a workflow
that publishes the wrong artifact, publishes untested code, or records a
release that never uploaded, produces a permanent wrong answer.

Nothing here runs the workflow — that needs GitHub. These assert the
properties whose absence has historically caused the damage, so a future
edit that removes one fails here rather than on release day.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(job: dict) -> list[dict]:
    return job.get("steps", []) or []


class TestNoScriptInjection:
    """`${{ inputs.x }}` spliced into a `run:` block puts caller-controlled
    text on the command line. In the job holding the PyPI publishing
    identity that is not a style issue."""

    def test_caller_controlled_values_reach_scripts_through_env(self, wf):
        offenders = []
        for job_name, job in wf["jobs"].items():
            for step in _steps(job):
                for m in re.finditer(r"\$\{\{\s*([^}]+?)\s*\}\}", step.get("run", "")):
                    if m.group(1).startswith(("inputs.", "github.event.")):
                        offenders.append(f"{job_name}/{step.get('name')}: {m.group(1)}")
        assert not offenders, (
            "interpolated into a shell script instead of passed via env: "
            + "; ".join(offenders)
        )


class TestPublishOrdering:
    """Publishing is irreversible; everything that can fail must fail
    first, and nothing may claim a release that did not upload."""

    def test_publish_requires_the_build_and_test_job(self, wf):
        assert "build" in (wf["jobs"]["publish"].get("needs") or []), (
            "publish could run without the tests having passed"
        )

    def test_the_tag_is_created_only_after_a_successful_upload(self, wf):
        """A tag can be deleted; a PyPI version cannot. So the repository
        records the release after PyPI accepts it, never before — the
        reverse would leave a tag asserting a release that never
        happened."""
        assert "publish" in (wf["jobs"]["finalise"].get("needs") or [])

    def test_the_manual_path_does_not_duplicate_the_release_path(self, wf):
        """The GitHub Release trigger already has a tag and a release;
        creating them again would fail the run."""
        assert wf["jobs"]["finalise"]["if"] == "github.event_name == 'workflow_dispatch'"


class TestPermissions:
    def test_only_the_publishing_job_holds_the_oidc_token(self, wf):
        for name, job in wf["jobs"].items():
            publishes = "gh-action-pypi-publish" in yaml.dump(job)
            has_token = (job.get("permissions") or {}).get("id-token") == "write"
            assert publishes == has_token, (
                f"{name}: id-token: write and publishing must go together"
            )

    def test_jobs_that_write_refs_ask_for_permission(self, wf):
        for name, job in wf["jobs"].items():
            body = yaml.dump(job)
            if "git push origin" in body or "gh release create" in body:
                assert (job.get("permissions") or {}).get("contents") == "write", (
                    f"{name} writes refs without contents: write"
                )

    def test_declared_outputs_exist(self, wf):
        for name, job in wf["jobs"].items():
            for m in re.finditer(r"needs\.([\w-]+)\.outputs\.([\w-]+)", yaml.dump(job)):
                dep, out = m.groups()
                assert out in (wf["jobs"][dep].get("outputs") or {}), (
                    f"{name} reads needs.{dep}.outputs.{out}, which {dep} does not set"
                )
                assert dep in (job.get("needs") or []), (
                    f"{name} reads outputs of {dep} without needing it"
                )


class TestVersionGate:
    """Both trigger paths must refuse to publish a version that disagrees
    with the package, in either direction."""

    def test_both_triggers_are_wired(self, wf):
        triggers = wf.get("on") or wf.get(True)
        assert "release" in triggers and "workflow_dispatch" in triggers

    def test_the_manual_path_requires_the_version_to_be_typed(self, wf):
        triggers = wf.get("on") or wf.get(True)
        version = triggers["workflow_dispatch"]["inputs"]["version"]
        assert version["required"] is True, (
            "a dispatch with no confirmation makes an accidental click a release"
        )

    def test_the_verify_step_compares_against_the_installed_package(self, wf):
        run = "".join(s.get("run", "") for s in _steps(wf["jobs"]["build"]))
        assert "tokenmizer.__version__" in run
        assert "exit 1" in run


class TestReleaseNotes:
    """The body comes from the CHANGELOG so notes and changelog cannot
    drift. Extracted here exactly as the workflow does it."""

    @staticmethod
    def _extractor(wf) -> str:
        step = next(s for s in _steps(wf["jobs"]["finalise"])
                    if s.get("name") == "Create the GitHub Release")
        return step["run"].split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]

    def _run(self, wf, version, tmp_path, env=None):
        """Run the extractor exactly as the workflow does, in a copy of the
        repo root so the notes.md it writes does not litter the checkout."""
        script = tmp_path / "notes.py"
        script.write_text(self._extractor(wf), encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text(
            (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), encoding="utf-8")
        res = subprocess.run(
            [sys.executable, str(script), version],
            capture_output=True, text=True, cwd=tmp_path,
            env={**os.environ, **(env or {})},
        )
        notes = tmp_path / "notes.md"
        return res, (notes.read_text(encoding="utf-8") if notes.exists() else "")

    def test_extracts_exactly_the_current_versions_section(self, wf, tmp_path):
        import tokenmizer

        res, body = self._run(wf, tokenmizer.__version__, tmp_path)
        assert res.returncode == 0, res.stderr
        assert len(body.strip()) > 500, "release notes came out empty"
        assert "## [" not in body, "bled into an adjacent version's section"

    def test_an_unknown_version_degrades_instead_of_crashing(self, wf, tmp_path):
        """A release whose CHANGELOG section is missing should still
        publish with a pointer, not fail after PyPI already has the
        upload."""
        res, body = self._run(wf, "9.9.9", tmp_path)
        assert res.returncode == 0, res.stderr
        assert "CHANGELOG" in body

    def test_it_does_not_depend_on_the_runner_locale(self, wf, tmp_path):
        """The changelog contains arrows and em-dashes. `print` encodes
        with whatever stdout happens to be, so on a non-UTF-8 runner the
        script died with UnicodeEncodeError — which the Windows leg caught,
        and which in production would have fired AFTER PyPI already had the
        upload. Forcing a legacy codec here reproduces that exactly."""
        import tokenmizer

        res, body = self._run(
            wf, tokenmizer.__version__, tmp_path,
            env={"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
        )
        assert res.returncode == 0, (
            "release notes must not depend on the runner's stdout encoding:\n"
            + res.stderr
        )
        assert "\u2192" in body or len(body) > 500
