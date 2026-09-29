"""
Local-only mode: nothing leaves the machine except to hosts the operator named.

TokenMizer's own work — extraction, the graph, windowing, the cache — is
local. What can still reach a network is the chat provider, and, on first
use, two downloads (the tokenizer vocabulary and the embedding model) that
send nothing but do open a connection. "Local by default" is a property of
today's code, not something a config change or a dependency update cannot
break, so this makes it a property of the process.

Two parts:

  validate()  refuses to start on a configuration that contradicts the mode,
              rather than starting and leaking. A remote provider under
              `local_only` is an error, not a warning.

  a guard     that wraps name resolution and connect(). Anything not on the
              allow-list raises NetworkBlocked before a packet is sent, and
              the DNS lookup is refused too, since the hostname a process
              asks for is itself information.

The guard sees hostnames and addresses, not what travels inside a tunnel.
A proxy hides the real destination, and a proxy on loopback (a local relay,
an SSH or corporate tunnel) is the common case and passes the loopback
rule, which would make the whole guard a formality. So while the mode is on,
proxy settings are removed from the process environment unless the operator
has listed the proxy's own host in privacy.allowed_hosts: a listed proxy is
a decision, and everything behind it is then reachable.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Providers that run on the operator's own machine.
LOCAL_PROVIDERS = frozenset({"ollama"})

_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "http_proxy", "https_proxy", "all_proxy")
# Libraries that would otherwise try the network on first use. They degrade
# on a refusal anyway; these make them not try.
_OFFLINE_VARS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


class NetworkBlocked(OSError):
    """A connection or lookup was refused by local-only mode."""


class PrivacyConfigError(ValueError):
    """The configuration contradicts local-only mode."""


def validate(settings) -> None:
    """Raise PrivacyConfigError if `settings` cannot honour local-only mode."""
    if not settings.privacy.local_only:
        return
    provider = settings.provider.lower()
    if provider not in LOCAL_PROVIDERS and not settings.privacy.allowed_hosts:
        raise PrivacyConfigError(
            f"privacy.local_only is on but provider={provider!r} is a remote "
            f"service, so every request would leave this machine. Use "
            f"provider: ollama, or list your own gateway's hostname in "
            f"privacy.allowed_hosts."
        )


class _Guard:
    def __init__(self, allowed_hosts):
        self.hosts = {h.strip().lower().rstrip(".") for h in allowed_hosts if h.strip()}
        self.ips: set[str] = set()
        self.lock = threading.Lock()
        self.saved: tuple | None = None
        self.env_saved: dict[str, str | None] = {}
        self.removed_proxies: list[str] = []

    def _proxy_host(self, value: str) -> str:
        parsed = urlparse(value if "://" in value else "http://" + value)
        return (parsed.hostname or "").lower().rstrip(".")

    def apply_environment(self) -> None:
        """Remove proxies that would hide a destination, and stop libraries
        that would try the network on first use from trying."""
        for var in _PROXY_VARS:
            value = os.environ.get(var)
            if not value or self._proxy_host(value) in self.hosts:
                continue
            self.env_saved[var] = value
            self.removed_proxies.append(var)
            del os.environ[var]
        for var, value in _OFFLINE_VARS.items():
            if var not in os.environ:
                self.env_saved[var] = None
                os.environ[var] = value
        if self.removed_proxies:
            logger.warning(
                "Local-only mode removed proxy settings %s: a proxy hides "
                "the real destination from the network guard. To keep one, "
                "list its host in privacy.allowed_hosts.",
                sorted(set(self.removed_proxies)),
            )

    def restore_environment(self) -> None:
        for var, value in self.env_saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        self.env_saved.clear()
        self.removed_proxies.clear()

    def _is_loopback(self, host: str) -> bool:
        if host in _LOCAL_NAMES:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def host_allowed(self, host) -> bool:
        if host is None or host == "":
            return True          # a wildcard bind, not a destination
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        host = str(host).strip().lower().rstrip(".")
        return self._is_loopback(host) or host in self.hosts or host in self.ips

    def address_allowed(self, address) -> bool:
        if not isinstance(address, tuple) or not address:
            return True          # AF_UNIX path: local by construction
        return self.host_allowed(address[0])

    def install(self) -> None:
        if self.saved is not None:
            return
        guard = self
        orig_gai = socket.getaddrinfo
        orig_connect = socket.socket.connect
        orig_connect_ex = socket.socket.connect_ex

        def getaddrinfo(host, *args, **kwargs):
            if not guard.host_allowed(host):
                raise NetworkBlocked(
                    f"local-only mode refused to resolve {host!r}: not in "
                    f"privacy.allowed_hosts")
            result = orig_gai(host, *args, **kwargs)
            if host and not guard._is_loopback(str(host).lower()):
                with guard.lock:
                    guard.ips.update(r[4][0] for r in result)
            return result

        def connect(sock, address):
            if not guard.address_allowed(address):
                raise NetworkBlocked(
                    f"local-only mode refused to connect to {address[0]!r}: "
                    f"not in privacy.allowed_hosts")
            return orig_connect(sock, address)

        def connect_ex(sock, address):
            if not guard.address_allowed(address):
                return 1         # EPERM, the errno connect_ex reports
            return orig_connect_ex(sock, address)

        self.saved = (orig_gai, orig_connect, orig_connect_ex)
        socket.getaddrinfo = getaddrinfo
        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex

    def uninstall(self) -> None:
        self.restore_environment()
        if self.saved is None:
            return
        socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex = self.saved
        self.saved = None


_active: _Guard | None = None


def enforce(settings) -> bool:
    """Validate, then install the guard. True if local-only mode is on."""
    global _active
    validate(settings)
    if not settings.privacy.local_only:
        return False
    if _active is None:
        _active = _Guard(settings.privacy.allowed_hosts)
        _active.apply_environment()
        _active.install()
        logger.info(
            "Local-only mode on: connections limited to loopback and %s",
            sorted(_active.hosts) or "no other hosts",
        )
    return True


def release() -> None:
    """Remove the guard. Called at shutdown so a process that outlives the
    server (a test run) is not left with a patched socket module."""
    global _active
    if _active is not None:
        _active.uninstall()
        _active = None


def status(settings) -> dict:
    """What an operator can check from outside, since the log line that
    confirms the mode is at a level the server does not print."""
    return {
        "local_only": bool(settings.privacy.local_only),
        "guard_active": _active is not None,
        "allowed_hosts": sorted(_active.hosts) if _active else
                         list(settings.privacy.allowed_hosts),
        "proxies_removed": sorted(set(_active.removed_proxies)) if _active else [],
    }
