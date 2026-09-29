"""
The two window limits are user-facing settings, so a value that would make
windowing misbehave has to be refused when the config loads, not discovered
mid-session.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tokenmizer.config.settings import MemorySettings


def test_defaults():
    m = MemorySettings()
    assert m.max_tail_tokens == 16000
    assert m.tool_index_tokens == 1500
    assert m.stable_window == "auto"


@pytest.mark.parametrize("field", ["max_tail_tokens", "tool_index_tokens"])
def test_zero_turns_the_feature_off_and_is_accepted(field):
    assert getattr(MemorySettings(**{field: 0}), field) == 0


@pytest.mark.parametrize("field", ["max_tail_tokens", "tool_index_tokens"])
def test_a_negative_limit_is_refused(field):
    with pytest.raises(ValidationError):
        MemorySettings(**{field: -1})


def test_a_bad_stable_window_mode_is_refused():
    with pytest.raises(ValidationError):
        MemorySettings(stable_window="sometimes")


def test_the_environment_can_set_them(monkeypatch):
    from tokenmizer.config.settings import Settings

    monkeypatch.setenv("TOKENMIZER_MEMORY__MAX_TAIL_TOKENS", "8000")
    monkeypatch.setenv("TOKENMIZER_MEMORY__TOOL_INDEX_TOKENS", "0")
    s = Settings()
    assert s.memory.max_tail_tokens == 8000
    assert s.memory.tool_index_tokens == 0


def test_the_window_singleton_takes_its_limits_from_the_settings():
    from tokenmizer.api import app as app_module

    window = app_module._smart_window
    assert window.max_tail_tokens == app_module.settings.memory.max_tail_tokens
    assert window.tool_index_tokens == app_module.settings.memory.tool_index_tokens
