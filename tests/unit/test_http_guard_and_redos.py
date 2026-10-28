"""
Regressions for the browser-facing attacks, the quadratic regexes and the
dashboard defects found in the security pass. Each test is written against
behaviour that was reproduced before the fix.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def proxy(monkeypatch):
    from tokenmizer.api import app as app_module
    from tokenmizer.providers.providers import LLMResponse
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(
        text="ok", input_tokens=10, output_tokens=5,
        model="fake-model", provider="fake", latency_ms=1.0,
    ))
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings, "api_keys", [], raising=False)
    monkeypatch.setattr(app_module.settings.security, "trusted_hosts", [])
    with TestClient(app_module.app) as c:
        yield c, app_module


_CHAT = {"messages": [{"role": "user", "content": "hello there"}]}


# ── Cross-site writes ────────────────────────────────────────────────────────

def test_a_foreign_page_cannot_write(proxy):
    c, _ = proxy
    r = c.post("/api/checkpoint?session_id=s1", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = c.post("/v1/chat/completions", json=_CHAT, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_a_client_without_an_origin_is_unaffected(proxy):
    c, _ = proxy
    assert c.post("/v1/chat/completions", json=_CHAT).status_code == 200


def test_the_dashboards_own_origin_may_write(proxy):
    c, _ = proxy
    r = c.post("/v1/chat/completions", json=_CHAT, headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_a_listed_cors_origin_may_write(proxy, monkeypatch):
    c, app_module = proxy
    monkeypatch.setattr(app_module.settings, "cors_origins", ["https://app.example"])
    r = c.post("/v1/chat/completions", json=_CHAT, headers={"Origin": "https://app.example"})
    assert r.status_code == 200


def test_reads_from_a_foreign_origin_are_left_to_cors(proxy):
    c, _ = proxy
    assert c.get("/health", headers={"Origin": "https://evil.example"}).status_code == 200


# ── DNS rebinding ────────────────────────────────────────────────────────────

def test_a_keyless_proxy_answers_only_to_loopback(proxy):
    c, _ = proxy
    r = c.get("/api/stats", headers={"Host": "rebind.evil.example"})
    assert r.status_code == 421
    assert c.get("/api/stats", headers={"Host": "localhost:8000"}).status_code == 200
    assert c.get("/api/stats", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert c.get("/api/stats", headers={"Host": "[::1]:8000"}).status_code == 200


def test_health_answers_any_host_for_container_probes(proxy):
    c, _ = proxy
    assert c.get("/health", headers={"Host": "10.0.3.7:8000"}).status_code == 200


def test_trusted_hosts_are_honoured(proxy, monkeypatch):
    c, app_module = proxy
    monkeypatch.setattr(app_module.settings.security, "trusted_hosts", ["box.lan"])
    assert c.get("/api/stats", headers={"Host": "box.lan:8000"}).status_code == 200
    assert c.get("/api/stats", headers={"Host": "other.lan"}).status_code == 421


def test_with_a_key_any_host_is_accepted(proxy, monkeypatch):
    c, app_module = proxy
    monkeypatch.setattr(app_module.settings, "api_key", "k-123", raising=False)
    r = c.get("/api/stats", headers={"Host": "proxy.corp.example",
                                     "Authorization": "Bearer k-123"})
    assert r.status_code == 200


# ── Size limits ──────────────────────────────────────────────────────────────

def test_an_overlong_session_id_is_refused(proxy):
    c, _ = proxy
    long_id = "a" * 300
    assert c.get(f"/api/graph/{long_id}").status_code == 422
    r = c.post("/v1/chat/completions", json=_CHAT, headers={"X-Session-ID": long_id})
    assert r.status_code == 422
    assert c.post(f"/api/checkpoint?session_id={long_id}").status_code == 422


def _guarded(max_body: int):
    from tokenmizer.security.http_guard import HTTPGuard
    seen = {}

    async def app(scope, receive, send):
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                raise RuntimeError("client went away")
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        seen["body"] = body
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/html")]})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = HTTPGuard(app, auth_configured=lambda: False, trusted_hosts=lambda: [],
                      cors_origins=lambda: [], max_body_bytes=max_body)
    return guard, seen


def _call(guard, headers, chunks):
    sent = []
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
             for i, c in enumerate(chunks)]

    async def receive():
        return queue.pop(0)

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "query_string": b"", "headers": headers}
    asyncio.run(guard(scope, receive, send))
    return sent


def test_a_declared_oversize_body_is_refused_unread():
    guard, seen = _guarded(1024)
    sent = _call(guard, [(b"host", b"localhost"), (b"content-length", b"5000")], [b"x" * 5000])
    assert sent[0]["status"] == 413
    assert "body" not in seen


def test_an_undeclared_oversize_body_is_cut_off():
    guard, seen = _guarded(1024)
    sent = _call(guard, [(b"host", b"localhost")], [b"x" * 800, b"x" * 800])
    assert sent[0]["status"] == 413
    assert "body" not in seen


def test_hardening_headers_are_added():
    guard, _ = _guarded(1024)
    sent = _call(guard, [(b"host", b"localhost")], [b"{}"])
    headers = dict(sent[0]["headers"])
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"referrer-policy"] == b"no-referrer"
    assert headers[b"x-frame-options"] == b"DENY"
    assert b"frame-ancestors 'none'" in headers[b"content-security-policy"]


# ── Dashboard ────────────────────────────────────────────────────────────────

def test_health_reports_the_real_layer_configuration(proxy):
    c, _ = proxy
    layers = c.get("/health").json()["layers"]
    assert layers["semantic_cache"] is False          # the fixture turned it off
    assert set(layers) == {"graph_memory", "semantic_cache", "compression",
                           "file_intelligence", "terse_output", "model_map"}


def test_the_graph_preview_cannot_reach_the_dashboard_origin():
    import re

    from tokenmizer.dashboard.page import DASHBOARD_HTML
    frame = re.search(r'<iframe[^>]*id="preview"[^>]*>', DASHBOARD_HTML).group(0)
    assert 'sandbox="' in frame
    assert "allow-same-origin" not in frame


_DOM_STUB = """
const els = {};
const el = id => (els[id] = els[id] || {id, textContent: '', className: '', innerHTML: '',
  style: {}, dataset: {}, addEventListener() {}, removeAttribute() {}, classList: {add() {}, remove() {}}});
globalThis.document = {getElementById: el, querySelectorAll: () => [], addEventListener() {}};
globalThis.sessionStorage = {getItem: () => '', setItem() {}, removeItem() {}};
globalThis.fetch = () => new Promise(() => {});
globalThis.setInterval = () => 0;
"""


def test_render_health_handles_counts_and_sets_real_badges(tmp_path):
    """/health sends counts; renderHealth called .forEach on them and threw,
    and every layer badge was literal "Active" whatever the config said."""
    import json
    import re
    import shutil
    import subprocess

    from tokenmizer.dashboard.page import DASHBOARD_HTML
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = re.findall(r"<script>(.*?)</script>", DASHBOARD_HTML, re.S)[-1]
    health = {"status": "ok", "sessions_with_data_loss": 2, "provider_failures": {},
              "layers": {"graph_memory": True, "semantic_cache": False, "compression": True,
                         "file_intelligence": True, "terse_output": "auto", "model_map": 0}}
    probe = tmp_path / "probe.js"
    probe.write_text(_DOM_STUB + script + f"""
renderHealth({json.dumps(health)});
console.log(JSON.stringify(Object.fromEntries(
  Object.entries(els).filter(([k]) => k.startsWith('layer-')).map(([k, v]) => [k, v.textContent]))));
""")
    out = subprocess.run([node, str(probe)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    badges = json.loads(out.stdout.strip().splitlines()[-1])
    assert badges == {"layer-graph": "Active", "layer-cache": "Off", "layer-compression": "Active",
                      "layer-files": "Active", "layer-terse": "Auto",
                      "layer-model-map": "Config only"}


# ── Quadratic regexes ────────────────────────────────────────────────────────
# Thresholds are an order of magnitude above the fixed timings and an order
# below the quadratic ones, so a regression fails without the test flaking.

def _elapsed(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def test_extraction_is_linear_without_sentence_breaks():
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
    he = get_hybrid_extractor(None)
    text = "Decided: " * (100_000 // 9)          # was 8.7s
    assert _elapsed(he.heuristic_extract, [{"role": "user", "content": text}]) < 3.0


def test_email_redaction_is_linear():
    from tokenmizer.security.redaction import _EMAIL
    assert _elapsed(lambda: list(_EMAIL.finditer("1." * 100_000))) < 0.5   # was ~8s
    assert _EMAIL.search("write to alice.smith+ops@acme.co.uk today").group(0) \
        == "alice.smith+ops@acme.co.uk"


def test_traceback_pattern_is_bounded():
    from tokenmizer.graph_memory.hybrid_extractor import _MODULE_PATH, _TRACEBACK_LAST_LINE
    text = "AbcError" * 2_000
    assert _elapsed(lambda: [_TRACEBACK_LAST_LINE.match(text, i) for i in range(0, 4000, 8)]) < 1.0
    assert _elapsed(lambda: [_MODULE_PATH.fullmatch("a_" * 8000, 0, i) for i in range(0, 4000, 8)]) < 1.0
    assert _TRACEBACK_LAST_LINE.match("django.db.utils.OperationalError: no such table")
    assert _TRACEBACK_LAST_LINE.match("django.core.exceptions.ImproperlyConfigured: Requested")


def test_trimmer_newline_runs_are_linear():
    from tokenmizer.compression.output_trimmer import _CLOSING_FILLERS, OutputTrimmer
    text = "\n" * 100_000
    assert _elapsed(lambda: [list(p.finditer(text)) for p in _CLOSING_FILLERS]) < 0.5
    out, _ = OutputTrimmer().trim("The fix is in auth.py line 42.\n\nHope this helps!", "full")
    assert "Hope this helps" not in out


def test_env_var_and_owner_secret_checks_are_linear():
    from tokenmizer.preferences import looks_like_a_preference
    from tokenmizer.semantic_cache.cache import SemanticCache
    sc = SemanticCache.__new__(SemanticCache)
    assert _elapsed(looks_like_a_preference, "A" * 100_000) < 0.5
    assert _elapsed(sc._is_session_sensitive, "A" * 100_000) < 0.5
    assert _elapsed(sc._is_session_sensitive, "project " * 20_000) < 0.5
    assert sc._is_session_sensitive("what is the API_KEY = abcd1234 for")
    assert sc._is_session_sensitive("where is our project config stored")
    assert not sc._is_session_sensitive("explain the config file format")


def test_negation_is_still_seen_within_a_clause():
    from tokenmizer.graph_memory.patterns import _is_negated_context, _is_question_context
    text = "We decided not to use MongoDB for this."
    assert _is_negated_context(text, text.index("MongoDB"))
    text = "Should we go with Postgres for this?"
    assert _is_question_context(text, text.index("Postgres"))
    text = "We went with Postgres. Is Redis next?"
    assert not _is_question_context(text, text.index("Postgres"))


def test_supersession_reason_is_its_own_sentence():
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
    he = get_hybrid_extractor(None)
    out = he.heuristic_extract([{"role": "user", "content":
        "TODO: add rate limiting to the API. We switched from moment.js to "
        "date-fns because the bundle was too large."}])
    reasons = [d["reason"] for d in out.decisions if d["label"] == "Use date-fns"]
    assert reasons and "TODO" not in reasons[0] and "bundle" in reasons[0]
