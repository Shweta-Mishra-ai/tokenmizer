"""
Local-only mode has to be a property of the process, not of today's
defaults: nothing leaves the machine except to hosts the operator named.

These tests use real sockets against a listener on loopback, so they show
what the guard does to actual connections and lookups rather than to a mock
of them. Every test releases the guard; it patches the socket module for
the whole process.
"""
from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import pytest

from tokenmizer.core import privacy
from tokenmizer.core.privacy import NetworkBlocked, PrivacyConfigError


def _settings(provider="ollama", local_only=True, allowed=()):
    return SimpleNamespace(
        provider=provider,
        privacy=SimpleNamespace(local_only=local_only, allowed_hosts=list(allowed)),
    )


@pytest.fixture(autouse=True)
def _release_guard():
    yield
    privacy.release()


@pytest.fixture
def listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                continue

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield srv.getsockname()[1]
    stop.set()
    t.join(2)
    srv.close()


# ── Configuration that contradicts the mode is refused ───────────────────────

@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "mistral"])
def test_a_remote_provider_is_an_error_not_a_warning(provider):
    with pytest.raises(PrivacyConfigError, match="leave this machine"):
        privacy.validate(_settings(provider=provider))


def test_a_local_provider_is_accepted():
    privacy.validate(_settings(provider="ollama"))


def test_your_own_gateway_makes_a_remote_provider_acceptable():
    privacy.validate(_settings(provider="openai", allowed=["llm.internal.example"]))


def test_nothing_is_checked_when_the_mode_is_off():
    assert privacy.enforce(_settings(provider="openai", local_only=False)) is False
    assert privacy._active is None


# ── The guard on real connections ────────────────────────────────────────────

def test_loopback_still_connects(listener):
    privacy.enforce(_settings())
    with socket.create_connection(("127.0.0.1", listener), timeout=2):
        pass
    with socket.create_connection(("localhost", listener), timeout=2):
        pass


def test_an_unlisted_address_is_refused_before_any_packet(listener):
    privacy.enforce(_settings())
    with pytest.raises(NetworkBlocked, match="203.0.113.7"):
        socket.socket().connect(("203.0.113.7", 443))


def test_an_unlisted_hostname_is_refused_at_the_lookup():
    """The name a process asks for is itself information, so the DNS
    lookup is what gets refused, not just the connection that follows."""
    privacy.enforce(_settings())
    with pytest.raises(NetworkBlocked, match="api.example.com"):
        socket.getaddrinfo("api.example.com", 443)


def test_connect_ex_reports_failure_instead_of_connecting():
    privacy.enforce(_settings())
    assert socket.socket().connect_ex(("203.0.113.7", 443)) != 0


def test_a_listed_host_resolves_and_its_addresses_may_connect(listener, monkeypatch):
    calls = []
    real = socket.getaddrinfo

    def fake(host, *a, **k):
        calls.append(host)
        if host == "gateway.internal.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", listener))]
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    privacy.enforce(_settings(provider="openai", allowed=["gateway.internal.example"]))
    socket.getaddrinfo("gateway.internal.example", listener)
    assert "gateway.internal.example" in calls
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("other.example.com", 443)


def test_a_proxy_is_not_allowed_implicitly(monkeypatch):
    """Behind a proxy the destination is invisible to a socket guard, so
    reaching one has to be a decision the operator wrote down."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    privacy.enforce(_settings())
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("proxy.corp.example", 3128)


def test_unix_sockets_are_local_by_construction(tmp_path):
    privacy.enforce(_settings())
    path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    try:
        cli = socket.socket(socket.AF_UNIX)
        cli.connect(path)
        cli.close()
    finally:
        srv.close()


def test_release_restores_the_socket_module(listener):
    original = (socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex)
    privacy.enforce(_settings())
    assert socket.socket.connect is not original[1]
    privacy.release()
    assert (socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex) == original


def test_enforcing_twice_does_not_stack_guards():
    privacy.enforce(_settings())
    first = socket.socket.connect
    privacy.enforce(_settings())
    assert socket.socket.connect is first


# ── The behaviours the mode exists to protect ────────────────────────────────

def test_the_tokenizer_degrades_to_an_estimate_instead_of_dialling_out(monkeypatch, caplog):
    """count_tokens must not fail the request because the vocabulary
    download was refused: it falls back, and says so.

    The download is simulated by a lookup of the host tiktoken really
    fetches from, so this fails without the guard (the lookup would go
    out) and shows the refusal reaching the fallback."""
    import logging

    import tiktoken

    from tokenmizer.core import tokenizer

    reached = []

    def download(*_a, **_k):
        reached.append(1)
        socket.getaddrinfo("openaipublic.blob.core.windows.net", 443)
        raise AssertionError("the lookup was not refused")

    monkeypatch.setattr(tiktoken, "encoding_for_model", download)
    monkeypatch.setattr(tiktoken, "get_encoding", download)
    privacy.enforce(_settings())
    tokenizer._get_encoding.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            n = tokenizer.count_tokens("hello world, this is a sentence", "gpt-4o")
    finally:
        tokenizer._get_encoding.cache_clear()
    assert reached, "the tokenizer never tried to download"
    assert n > 0
    assert any("openaipublic" in r.getMessage() or "local-only" in r.getMessage()
               for r in caplog.records)


def test_the_server_refuses_to_start_on_a_contradictory_config(monkeypatch):
    from fastapi.testclient import TestClient

    from tokenmizer.api import app as app_module

    monkeypatch.setattr(app_module.settings.privacy, "local_only", True)
    monkeypatch.setattr(app_module.settings, "provider", "openai")
    with pytest.raises(PrivacyConfigError):
        with TestClient(app_module.app):
            pass
    assert privacy._active is None


def test_the_server_starts_and_serves_health_in_local_only_mode(monkeypatch):
    from fastapi.testclient import TestClient

    from tokenmizer.api import app as app_module

    monkeypatch.setattr(app_module.settings.privacy, "local_only", True)
    monkeypatch.setattr(app_module.settings, "provider", "ollama")
    with TestClient(app_module.app) as client:
        assert privacy._active is not None
        assert client.get("/health").status_code == 200
    assert privacy._active is None, "the guard outlived the server"
