"""
How long a hung upstream can hold a worker.

Every vendor SDK used here defaults to a 600-second timeout. On a client
library that is a sensible default; on a proxy it is an outage. A hung
upstream holds the request, the session lock, the extraction slot and
the session's place in the graph cache — for ten minutes, per request.
A handful of them is the whole worker, and nothing in the process
recovers until the SDK gives up.

The Ollama adapter already passed `timeout=120`, which is how the gap
was visible at all: one adapter had thought about it and the rest had
inherited a default nobody chose.

`request_timeout` is a `Settings` field (config/settings.py), not a
bespoke `os.environ.get()` in providers.py — that centralisation is a
documented rule (`CONTRIBUTING.md`: no `os.getenv()` outside
config/settings.py), and the reason is exactly what
`test_a_bad_value_is_a_validation_error_like_every_other_setting` below
pins: a malformed value now fails the SAME way a malformed
`cache.max_size` or `terse_output.style` always has, rather than each
setting inventing its own parse-and-warn.
"""
from __future__ import annotations

import pytest

from tokenmizer.config.settings import Settings
from tokenmizer.providers.providers import (
    AnthropicProvider,
    BaseProvider,
    OpenAIProvider,
    build_provider,
)


class TestTheDefaultIsNotTenMinutes:

    def test_there_is_a_default_and_it_is_not_the_sdk_s(self):
        assert BaseProvider.request_timeout == 120.0
        assert Settings().request_timeout == 120.0

    @pytest.mark.parametrize("cls", [AnthropicProvider, OpenAIProvider])
    def test_every_adapter_inherits_it(self, cls):
        assert cls("k")._timeout_kwargs == {"timeout": 120.0}


class TestItIsConfigurable:

    def test_the_env_var_is_read(self, monkeypatch):
        monkeypatch.setenv("TOKENMIZER_REQUEST_TIMEOUT", "30")
        assert Settings().request_timeout == 30.0

    def test_a_bad_value_is_a_validation_error_like_every_other_setting(
            self, monkeypatch):
        """Not a bespoke warn-and-fall-back: this field is parsed by the
        same pydantic machinery as every other TOKENMIZER_* value, so a
        typo here fails exactly the way a typo in cache.max_size does —
        one behaviour to document and reason about, not N of them."""
        import pydantic

        monkeypatch.setenv("TOKENMIZER_REQUEST_TIMEOUT", "soon")
        with pytest.raises(pydantic.ValidationError):
            Settings()

    def test_build_provider_carries_the_setting_onto_the_instance(self):
        s = Settings(provider="anthropic", anthropic_api_key="k",
                     request_timeout=45.0)
        p = build_provider(s)
        assert p.request_timeout == 45.0
        assert p._timeout_kwargs == {"timeout": 45.0}

    def test_zero_restores_the_sdk_default_for_anyone_who_wants_it(self):
        p = AnthropicProvider("k")
        p.request_timeout = 0
        assert p._timeout_kwargs == {}, (
            "an explicit 0 means 'use the SDK default', not 'time out "
            "immediately'"
        )


class TestTheTimeoutActuallyReachesTheClient:

    async def test_anthropic_is_constructed_with_it(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        seen = {}

        class _Messages:
            async def create(self, **kw):
                raise RuntimeError("stop here — construction is the assertion")

        class _Client:
            def __init__(self, **kw):
                seen.update(kw)
                self.messages = _Messages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _Client)
        with pytest.raises(Exception):
            await AnthropicProvider("k")._call(
                [{"role": "user", "content": "hi"}], "claude-sonnet-4-6",
                64, False, "")

        assert seen.get("timeout") == 120.0
