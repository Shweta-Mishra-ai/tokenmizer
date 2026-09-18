"""
Habits that outlive a session.

`PreferenceStore` existed for a long time inside `semantic_cache/cache.py`
with no callers: `save()` was never invoked from anywhere, so
`/api/cache/stats` reported a preference field that was permanently the
empty string while implying a working cross-session memory. It is its own
module now, per principal, SQLite-backed, and actually wired in.

The failure mode of a preference memory is not forgetting. It is
remembering something that was never a preference and repeating it in
every prompt you send for the rest of the year. So most of what is pinned
here is what it must NOT remember, what it must not leak between callers,
and that a person can read and delete what it holds.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.preferences import (  # noqa: E402
    PreferenceStore,
    looks_like_a_preference,
    preference_key,
)
from tokenmizer.security.ownership import DEV_PRINCIPAL


@pytest.fixture
def store(tmp_path):
    return PreferenceStore(storage_dir=str(tmp_path))


class TestWhatCounts:

    @pytest.mark.parametrize("text", [
        "I prefer TypeScript over JavaScript for anything new",
        "Please keep it brief, I do not need the preamble",
        "My convention is snake_case for everything in Python",
        "I always use pytest rather than unittest",
        "Remember that I work in UTC+5:30",
    ])
    def test_a_habit_is_a_preference(self, text):
        assert looks_like_a_preference(text)

    @pytest.mark.parametrize("text", [
        # Work, not habit. This is what the session graph is for.
        "The login endpoint returns 422 when email is missing",
        "Let's use Postgres for this service",
        "run the tests",
        # A complaint about today is not a standing instruction.
        "I hate this bug, it has taken all afternoon",
        "I hate that the tests are flaky",
        # Never, whatever else it matches.
        "I always use the password hunter2 for local dev",
        "I prefer to set DATABASE_URL=postgres://localhost in my shell",
        "I always use sk-abc123def456 for testing",
    ])
    def test_everything_else_is_not(self, text):
        assert not looks_like_a_preference(text), text

    def test_a_restated_preference_updates_rather_than_duplicates(self, store):
        store.observe("alice", "I prefer TypeScript for everything")
        store.observe("alice", "I prefer TypeScript with strict mode on")

        assert len(store.all("alice")) == 1, (
            "two lines saying nearly the same thing cost the prompt twice"
        )
        assert "strict" in store.all("alice")[0][1]

    def test_the_key_ignores_scaffolding(self):
        assert preference_key("I prefer TypeScript for everything") == \
               preference_key("I always prefer TypeScript, for everything")


class TestItIsPerPrincipal:
    """The reason the old in-memory version could not be wired up: it was
    process-global, so one caller's habits would have reached another
    caller's prompt."""

    def test_one_principal_never_sees_another(self, store):
        store.observe("alice", "I prefer TypeScript for everything")
        store.observe("bob", "Please keep answers brief and skip the preamble")

        assert [v for _, v in store.all("alice")] == \
               ["I prefer TypeScript for everything"]
        assert "TypeScript" not in store.context("bob")
        assert "brief" not in store.context("alice")

    def test_an_unknown_principal_has_nothing(self, store):
        store.observe("alice", "I prefer TypeScript for everything")

        assert store.all("carol") == []
        assert store.context("carol") == ""


class TestItCanBeReadAndForgotten:

    def test_the_injected_block_is_exactly_what_it_says(self, store):
        store.observe("alice", "I prefer TypeScript for everything")
        block = store.context("alice")

        assert "How this person likes to work" in block
        assert "I prefer TypeScript for everything" in block

    def test_nothing_remembered_means_nothing_injected(self, store):
        assert store.context("alice") == "", (
            "the common case must add nothing to the prompt"
        )

    def test_forget_one(self, store):
        store.observe("alice", "I prefer TypeScript for everything")
        store.observe("alice", "Please keep it brief, skip the preamble")
        key = store.all("alice")[0][0]

        assert store.forget("alice", key) == 1
        assert len(store.all("alice")) == 1

    def test_forget_everything(self, store):
        store.observe("alice", "I prefer TypeScript for everything")
        store.observe("alice", "Please keep it brief, skip the preamble")

        assert store.forget("alice") == 2
        assert store.all("alice") == []

    def test_the_block_is_capped(self, store):
        for i in range(10):
            store.observe("alice", f"I prefer style number {i} for everything always")

        block = store.context("alice", max_items=3, max_chars=400)
        assert len([x for x in block.splitlines() if x.startswith("- ")]) <= 3
        assert len(block) <= 400


class TestItNeverCostsTheCallerTheirAnswer:

    def test_an_unopenable_store_is_inert_not_fatal(self, tmp_path):
        blocked = tmp_path / "file-not-a-dir"
        blocked.write_text("")

        store = PreferenceStore(storage_dir=str(blocked / "sub"))

        assert store.available is False
        assert store.observe("alice", "I prefer TypeScript for everything") is None
        assert store.all("alice") == [] and store.context("alice") == ""

    def test_a_broken_write_returns_none(self, store, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(store, "_connect", boom)

        assert store.observe("alice", "I prefer TypeScript for everything") is None
        assert store.all("alice") == []


class TestThroughTheProxy:

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
        monkeypatch.setattr(app_module.settings.preferences, "enabled", True)
        monkeypatch.setattr(app_module, "_preferences",
                            PreferenceStore(storage_dir=str(tmp_path)))
        with TestClient(app) as c:
            yield c

    def test_off_by_default(self):
        from tokenmizer.config.settings import Settings

        assert Settings().preferences.enabled is False, (
            "a feature that rewrites every prompt is opt-in"
        )

    def test_the_endpoint_reports_what_is_remembered(self, client):
        app_module._preferences.observe(
            DEV_PRINCIPAL, "I prefer TypeScript for everything")

        body = client.get("/api/preferences").json()

        assert body["enabled"] is True
        assert any("TypeScript" in p["value"] for p in body["preferences"])
        assert "TypeScript" in body["injected"], (
            "the endpoint must show the exact text that reaches the model"
        )

    def test_the_endpoint_forgets(self, client):
        app_module._preferences.observe(
            DEV_PRINCIPAL, "I prefer TypeScript for everything")

        assert client.request("DELETE", "/api/preferences").json()["forgotten"] == 1
        assert client.get("/api/preferences").json()["preferences"] == []

    def test_disabled_says_so_rather_than_looking_empty(self, monkeypatch):
        monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
        monkeypatch.setattr(app_module, "_preferences", None)
        with TestClient(app) as c:
            body = c.get("/api/preferences").json()

        assert body["enabled"] is False
        assert "note" in body, (
            "off and empty look identical without one of them saying which"
        )
