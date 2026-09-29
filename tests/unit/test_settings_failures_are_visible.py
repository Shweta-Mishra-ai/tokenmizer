"""
A failure to read settings must not silently change behaviour.

Three call sites fell back to a default when `get_settings()` raised, and
logged nothing: a broken config quietly switched the extraction patterns,
the resume-block headings, or the set of accepted API keys. The fallbacks
are right (a proxy that stays up beats one that dies on a config read);
being silent about them is not.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def broken_settings(monkeypatch):
    import tokenmizer.config.settings as settings_module

    def boom():
        raise RuntimeError("simulated settings failure")

    monkeypatch.setattr(settings_module, "get_settings", boom)


def test_the_extractor_domain_fallback_says_so(broken_settings, caplog):
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor

    with caplog.at_level(logging.WARNING):
        extractor = get_hybrid_extractor()
    assert extractor is not None
    assert any("configured domain" in r.getMessage() and "simulated settings failure"
               in r.getMessage() for r in caplog.records)


def test_the_resume_vocabulary_fallback_says_so(broken_settings, caplog, tmp_path):
    from tokenmizer.graph_memory.context_block import _vocabulary
    from tokenmizer.graph_memory.graph import GraphMemory

    graph = GraphMemory("vocab-fallback", storage_dir=str(tmp_path))
    graph._domain = None
    with caplog.at_level(logging.WARNING):
        words = _vocabulary(graph)
    assert words
    assert any("configured domain" in r.getMessage() for r in caplog.records)


def test_the_key_list_fallback_says_so_and_stays_closed(monkeypatch, caplog):
    import asyncio

    from fastapi import HTTPException

    import tokenmizer.security.auth as auth_module

    monkeypatch.setattr(auth_module, "_get_configured_key", lambda: "primary-key")

    def boom():
        raise RuntimeError("simulated key list failure")

    monkeypatch.setattr(auth_module, "_get_configured_keys", boom)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as denied:
            asyncio.run(auth_module.verify_api_key(_request("wrong-key")))
    assert denied.value.status_code == 401, "a wrong key must still be refused"
    assert any("full list of API keys" in r.getMessage() for r in caplog.records)


def _request(key: str):
    """A request with real Starlette headers, which are case-insensitive:
    the code asks for "Authorization", and a plain dict would answer only
    to the exact spelling and send the test down the missing-key path."""
    from starlette.requests import Request

    return Request({
        "type": "http", "method": "POST", "path": "/v1/chat/completions",
        "headers": [(b"authorization", f"Bearer {key}".encode())],
        "query_string": b"",
    })
