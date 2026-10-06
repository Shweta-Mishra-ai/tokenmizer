"""Edge cases found by probing an installed server with hostile input.

1. An empty `messages` list was not rejected. It reached the provider twice
   (the transformed request, then the untransformed retry) and came back as
   a 502 "Provider request failed" instead of a 422 naming the field.
2. A session id that sanitises to a Windows device name (`con`, `nul`,
   `com1`, ...) produced a lock file called `con.<hash>.lock`. Windows
   treats those names as devices whatever follows the first dot. This was
   found by inspection; it was not reproduced on Windows.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tokenmizer.api import app as app_module
from tokenmizer.api.app import ChatRequest, app
from tokenmizer.graph_memory.filelock import _WINDOWS_DEVICE_NAMES, _safe_name
from tokenmizer.security.ownership import OwnershipStore


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
    app_module._graph_cache.clear()
    with TestClient(app) as c:
        yield c


class TestEmptyMessagesAreRejected:

    def test_the_model_requires_at_least_one_message(self):
        with pytest.raises(ValidationError, match="at least 1"):
            ChatRequest(messages=[])

    def test_one_message_is_still_accepted(self):
        assert len(ChatRequest(messages=[{"role": "user", "content": "hi"}]).messages) == 1

    def test_the_endpoint_answers_422_and_never_reaches_the_provider(self, client, monkeypatch):
        called = []

        async def provider_must_not_run(*args, **kwargs):
            called.append(1)
            raise AssertionError("an empty request reached the provider")

        monkeypatch.setattr(app_module, "_call_provider", provider_must_not_run, raising=False)

        r = client.post("/v1/chat/completions", json={"messages": []})

        assert r.status_code == 422, r.text
        assert "messages" in r.text
        assert not called


class TestLockFileNamesAreNeverWindowsDeviceNames:

    @pytest.mark.parametrize("device", sorted(_WINDOWS_DEVICE_NAMES))
    def test_a_device_name_is_not_used_as_the_file_stem(self, device):
        for variant in (str.lower, str.upper, str.title):
            stem = _safe_name(variant(device)).split(".")[0]
            assert stem.upper() not in _WINDOWS_DEVICE_NAMES, (variant(device), stem)

    @pytest.mark.parametrize("session_id", ["con", "nul", "com1", "lpt9", "CON", "Aux"])
    def test_the_name_is_still_distinct_from_a_session_really_named_with_an_underscore(
        self, session_id
    ):
        assert _safe_name(session_id) != _safe_name("_" + session_id)

    @pytest.mark.parametrize("session_id", ["my-project", "session_1", "a" * 200, "x"])
    def test_ordinary_ids_are_unchanged_in_shape(self, session_id):
        name = _safe_name(session_id)
        assert name.endswith(".lock")
        assert not name.startswith("_")

    def test_a_dotted_id_that_is_not_a_device_is_untouched(self):
        # "console" and "nullable" merely start with a device name.
        assert _safe_name("console").startswith("console.")
        assert _safe_name("nullable").startswith("nullable.")
