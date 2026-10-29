"""
HTTP-level protections for a proxy that holds provider keys.

TokenMizer usually runs on the operator's own machine, with no API key
(dev mode) and their Anthropic/OpenAI key loaded. Every web page they
visit can make their browser send requests to http://localhost:8000. Four
consequences were reproducible against the default configuration:

  1. Cross-site writes. `POST /api/decision/invalidate?...` and
     `POST /api/checkpoint?...` carry everything in the query string, so
     they are "simple" requests: the browser sends them with no CORS
     preflight, and any page could invalidate a session's decisions.
  2. DNS rebinding. With the Host header unchecked, a page whose domain
     re-resolves to 127.0.0.1 becomes same-origin with the proxy and can
     READ graphs and resume blocks, not just write.
  3. No body limit. A 60 MB body was read into memory and processed.
  4. No session-id limit. A 20,000-character id was claimed and stored.

The guards below are pure ASGI, so they run before routing and before any
body is parsed. A client that is not a browser (an SDK, curl, an agent)
sends no Origin header and is unaffected by the origin check.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

MAX_SESSION_ID_LENGTH = 256

# "testserver" is the Host Starlette's TestClient sends. It is not a
# registrable domain, so no attacker can point a page at it.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0", "testserver"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _host_only(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("["):                      # [::1]:8000
        return value[: value.find("]") + 1] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _origin_host(origin: str) -> str:
    rest = origin.split("://", 1)[-1]
    return _host_only(rest.split("/", 1)[0])


async def _reject(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class HTTPGuard:
    """Host allowlist, cross-origin write block, body and id limits, and
    browser hardening headers on every response."""

    def __init__(self, app, *, auth_configured, trusted_hosts: Iterable[str],
                 cors_origins: Iterable[str], max_body_bytes: int):
        self.app = app
        # Callables, so a key or host list changed in settings at runtime
        # (tests do this) is honoured without rebuilding the app.
        self._auth_configured = auth_configured
        self._trusted_hosts = trusted_hosts
        self._cors_origins = cors_origins
        self.max_body_bytes = max_body_bytes

    def _host_allowed(self, host: str) -> bool:
        trusted = {h.strip().lower() for h in self._trusted_hosts() if h.strip()}
        if "*" in trusted:
            return True
        if not trusted and self._auth_configured():
            # A key is required for every data route, so a rebinding page
            # without it reads nothing; any Host is fine.
            return True
        return host in _LOOPBACK_HOSTS or host in trusted

    def _origin_allowed(self, origin: str, host: str) -> bool:
        if not origin:
            return True
        if origin == "null":
            return False                           # sandboxed frame, file://
        allowed = {o.rstrip("/").lower() for o in self._cors_origins()}
        if "*" in allowed or origin.rstrip("/").lower() in allowed:
            return True
        # Same origin: the dashboard served by this proxy posting to it.
        return _origin_host(origin) == host

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        host = _host_only(headers.get("host", ""))
        # /health stays reachable by any Host: orchestrator probes address
        # the container by IP or service name, and it holds nothing a
        # rebinding page could use.
        if scope.get("path") != "/health" and not self._host_allowed(host):
            logger.warning("Rejected request for Host %r (not trusted)", host)
            return await _reject(
                send, 421,
                f"Host '{host}' is not trusted. With no api_key set, TokenMizer "
                f"answers only to localhost (this blocks DNS-rebinding attacks "
                f"from web pages). Set api_key, or list this host in "
                f"security.trusted_hosts.")

        method = scope.get("method", "GET").upper()
        origin = headers.get("origin", "")
        if method not in _SAFE_METHODS and origin and not self._origin_allowed(origin, host):
            logger.warning("Rejected cross-origin %s %s from %r", method, scope.get("path"), origin)
            return await _reject(
                send, 403,
                f"Cross-origin {method} from '{origin}' refused. Add the origin to "
                f"cors_origins if this page is yours.")

        sid = _session_id_in(scope)
        if sid is not None and len(sid) > MAX_SESSION_ID_LENGTH:
            return await _reject(
                send, 422, f"session_id is longer than {MAX_SESSION_ID_LENGTH} characters.")

        declared = headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_body_bytes:
            return await _reject(
                send, 413, f"Request body is larger than {self.max_body_bytes} bytes "
                           f"(security.max_request_bytes).")

        received = 0
        too_large = False

        async def limited_receive():
            # Content-Length can be absent (chunked) or wrong; count what
            # actually arrives.
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        started = False

        async def hardened_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                extra = [(b"x-content-type-options", b"nosniff"),
                         (b"referrer-policy", b"no-referrer")]
                ctype = dict(message.get("headers", [])).get(b"content-type", b"")
                if ctype.startswith(b"text/html"):
                    # Nothing here is meant to be framed by another site.
                    extra += [(b"x-frame-options", b"DENY"),
                              (b"content-security-policy", b"frame-ancestors 'none'")]
                message = {**message, "headers": list(message.get("headers", [])) + extra}
            await send(message)

        try:
            await self.app(scope, limited_receive, hardened_send)
        except Exception:
            if not too_large:
                raise
        if too_large and not started:
            await _reject(send, 413, f"Request body is larger than {self.max_body_bytes} bytes.")


def _session_id_in(scope) -> str | None:
    path = scope.get("path", "")
    for prefix in ("/api/graph/", "/api/resume/", "/api/checkpoints/"):
        if path.startswith(prefix):
            return path[len(prefix):].split("/", 1)[0]
    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if "session_id" in qs:
        return qs["session_id"][0]
    for k, v in scope.get("headers", []):
        if k.lower() == b"x-session-id":
            return v.decode("latin-1")
    return None
