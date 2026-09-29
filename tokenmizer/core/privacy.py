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
An HTTP(S) proxy therefore has to be listed explicitly to be reachable, and
once it is, everything behind it is reachable too. That is why proxies are
not allowed implicitly.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import threading

logger = logging.getLogger(__name__)

# Providers that run on the operator's own machine.
LOCAL_PROVIDERS = frozenset({"ollama"})

_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


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
