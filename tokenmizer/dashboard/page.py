"""Dashboard HTML — served at GET /"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TokenMizer — Never Lose AI Context</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3e;
    --accent: #7c6af7;
    --accent2: #5ee7c8;
    --text: #e8eaf6;
    --muted: #8b8fa8;
    --green: #4ade80;
    --yellow: #fbbf24;
    --red: #f87171;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; }

  header {
    border-bottom: 1px solid var(--border);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  header .logo { font-size: 1.4rem; font-weight: 700; }
  header .logo span { color: var(--accent); }
  header .tagline { color: var(--muted); font-size: 0.85rem; margin-left: auto; }
  header .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }

  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
  .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 2rem; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }
  .card h3 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.75rem; }
  .stat-value { font-size: 2.2rem; font-weight: 700; color: var(--text); }
  .stat-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
  .stat-value.green { color: var(--green); }
  .stat-value.accent { color: var(--accent); }
  .stat-value.accent2 { color: var(--accent2); }

  .layer-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
  .layer-row:last-child { border-bottom: none; }
  .layer-name { font-size: 0.875rem; }
  .layer-badge { font-size: 0.75rem; padding: 2px 10px; border-radius: 20px; font-weight: 600; }
  .badge-active { background: #1a2e1a; color: var(--green); }
  .badge-cache   { background: #1a2040; color: var(--accent); }
  .badge-graph   { background: #1e1a40; color: #a78bfa; }
  .badge-terse   { background: #2a2010; color: var(--yellow); }
  .badge-routing { background: #1a2a2a; color: var(--accent2); }

  .session-card { margin-bottom: 0.75rem; padding: 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; }
  .session-id { font-size: 0.8rem; color: var(--muted); font-family: monospace; }
  .session-progress { width: 100%; height: 4px; background: var(--border); border-radius: 2px; margin: 0.5rem 0; }
  .session-bar { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }

  .graph-node { display: inline-flex; align-items: center; gap: 0.4rem; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; margin: 3px; }
  .node-task       { background: #1a2e1a; color: var(--green); }
  .node-decision   { background: #1e1a40; color: #a78bfa; }
  .node-file       { background: #1a2040; color: var(--accent); }
  .node-error      { background: #2e1a1a; color: var(--red); }
  .node-dependency { background: #2a2010; color: var(--yellow); }
  .node-environment { background: #1a2a2a; color: var(--accent2); }
  .node-goal       { background: #2a1a2e; color: #e879f9; }
  .node-endpoint   { background: #101a2e; color: #60a5fa; }

  .endpoint-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-family: monospace;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.8;
  }
  .endpoint-box .method { color: var(--green); font-weight: 700; }
  .endpoint-box .path { color: var(--text); }
  .endpoint-box .desc { color: var(--muted); font-size: 0.75rem; }

  .resume-block {
    background: var(--bg);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-family: monospace;
    font-size: 0.8rem;
    line-height: 1.7;
    color: var(--accent2);
    white-space: pre-wrap;
    word-break: break-word;
  }

  footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem 2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 2rem;
  }

  @media (max-width: 768px) {
    .grid-3, .grid-2 { grid-template-columns: 1fr; }
    .container { padding: 1rem; }
  }
</style>
</head>
<body>

<header>
  <div class="status-dot"></div>
  <div class="logo">🧠 Token<span>Mizer</span></div>
  <div class="tagline">Never lose your AI context again.</div>
</header>

<div class="container">

  <!-- Stats row -->
  <div class="grid-3" id="stats-row">
    <div class="card">
      <h3>Requests Today</h3>
      <div class="stat-value accent" id="req-today">—</div>
      <div class="stat-sub">All sessions combined</div>
    </div>
    <div class="card">
      <h3>Tokens Saved Today</h3>
      <div class="stat-value green" id="tokens-saved">—</div>
      <div class="stat-sub" id="savings-pct">Loading…</div>
    </div>
    <div class="card">
      <h3>Cache Hit Rate</h3>
      <div class="stat-value accent2" id="cache-hit-rate">—</div>
      <div class="stat-sub" id="cache-entries">Loading…</div>
    </div>
  </div>

  <div class="grid-2">

    <!-- Pipeline layers -->
    <div class="card">
      <h3>Pipeline Layers</h3>
      <div class="layer-row">
        <span class="layer-name">Graph Memory + Checkpoint</span>
        <span class="layer-badge badge-graph">Core</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Semantic Cache</span>
        <span class="layer-badge badge-cache">Cache</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Prompt Compression</span>
        <span class="layer-badge badge-active">Active</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Terse Output</span>
        <span class="layer-badge badge-terse">Active</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Context Router</span>
        <span class="layer-badge badge-routing">Beta</span>
      </div>
    </div>

    <!-- Live graph nodes (demo) -->
    <div class="card">
      <h3>Graph Node Types</h3>
      <div>
        <span class="graph-node node-goal">🎯 Goal</span>
        <span class="graph-node node-task">✅ Task</span>
        <span class="graph-node node-decision">⚡ Decision</span>
        <span class="graph-node node-file">📄 File</span>
        <span class="graph-node node-error">🐛 Error</span>
        <span class="graph-node node-dependency">📦 Dependency</span>
        <span class="graph-node node-environment">🔧 Environment</span>
        <span class="graph-node node-endpoint">🌐 Endpoint</span>
      </div>
      <div style="margin-top:1rem;font-size:0.8rem;color:var(--muted)">
        Session graphs are built live, survive restarts, and produce a compact resume block when context fills.
      </div>
    </div>
  </div>

  <!-- API Endpoints -->
  <div class="card" style="margin-bottom:2rem">
    <h3>API Endpoints</h3>
    <div class="endpoint-box">
<span class="method">POST</span> <span class="path">/v1/chat/completions</span>  <span class="desc">— drop-in OpenAI-compatible proxy</span>
<span class="method">GET </span> <span class="path">/api/resume/{session_id}</span>  <span class="desc">— get checkpoint resume context</span>
<span class="method">POST</span> <span class="path">/api/checkpoint?session_id=</span>  <span class="desc">— manually trigger checkpoint</span>
<span class="method">GET </span> <span class="path">/api/graph/{session_id}</span>  <span class="desc">— session graph stats</span>
<span class="method">GET </span> <span class="path">/api/stats</span>  <span class="desc">— analytics summary</span>
<span class="method">GET </span> <span class="path">/docs</span>  <span class="desc">— interactive API docs (Swagger)</span>
    </div>
  </div>

  <!-- Resume block example -->
  <div class="card" style="margin-bottom:2rem">
    <h3>Example Resume Block (what gets injected on session resume)</h3>
    <div class="resume-block" id="resume-example">Goal: Build FastAPI auth service with JWT + PostgreSQL
In progress: Refresh token rotation | Rate limiting
Done: Project structure | User model | Auth endpoints | Fix 422 error | Tests (12 passing)
Decided: PostgreSQL (concurrent writes) | bcrypt | Redis for refresh tokens (not DB)
Files: api/auth.py, api/models.py, api/main.py, config.py, tests/test_auth.py
Env: Python 3.12, FastAPI 0.111
Continue from: Add rate limiting to auth endpoints</div>
    <div style="margin-top:0.75rem;font-size:0.8rem;color:var(--muted)" id="resume-tokens">
      ~280 tokens · replaces ~15,000+ tokens of conversation history
    </div>
  </div>

</div>

<footer>
  TokenMizer — open source · <a href="/docs" style="color:var(--accent)">API Docs</a> · <a href="/health" style="color:var(--accent)">Health</a>
</footer>

<script>
const API_KEY_STORAGE_KEY = 'tokenmizer.apiKey';
let apiKeyPrompt = null;
let apiKeyPromptDismissed = false;

function getStoredApiKey() {
  return sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
}

function authOptions(apiKey) {
  return apiKey ? {headers: {Authorization: `Bearer ${apiKey}`}} : {};
}

async function requestApiKey() {
  if (apiKeyPromptDismissed) return null;

  if (!apiKeyPrompt) {
    apiKeyPrompt = Promise.resolve()
      .then(() => window.prompt('TokenMizer API key'))
      .then(value => {
        const key = (value || '').trim();
        if (!key) {
          apiKeyPromptDismissed = true;
          return null;
        }
        sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
        return key;
      })
      .finally(() => {
        apiKeyPrompt = null;
      });
  }

  return apiKeyPrompt;
}

async function dashboardFetch(path) {
  let apiKey = getStoredApiKey();
  let response = await fetch(path, authOptions(apiKey));
  if (response.status !== 401) return response;

  if (apiKey) {
    sessionStorage.removeItem(API_KEY_STORAGE_KEY);
  }
  apiKey = await requestApiKey();
  if (!apiKey) return response;

  return fetch(path, authOptions(apiKey));
}

async function loadStats() {
  try {
    const [statsRes, cacheRes] = await Promise.all([
      dashboardFetch('/api/stats').catch(() => null),
      dashboardFetch('/api/cache/stats').catch(() => null),
    ]);

    if (statsRes && statsRes.ok) {
      const s = await statsRes.json();
      const d = s.daily || {};
      document.getElementById('req-today').textContent = (d.requests || 0).toLocaleString();
      document.getElementById('tokens-saved').textContent = (d.tokens_saved || 0).toLocaleString();
      document.getElementById('savings-pct').textContent =
        d.savings_pct ? `${d.savings_pct.toFixed(1)}% reduction vs original` : 'No requests yet';
    }

    if (cacheRes && cacheRes.ok) {
      const c = await cacheRes.json();
      document.getElementById('cache-hit-rate').textContent =
        c.hit_rate != null ? `${(c.hit_rate * 100).toFixed(1)}%` : '—';
      document.getElementById('cache-entries').textContent =
        `${(c.entries || 0).toLocaleString()} cached entries · ${c.utilization_pct || 0}% full`;
    }
  } catch (e) {
    console.warn('Stats load failed:', e);
  }
}

loadStats();
setInterval(loadStats, 15000);
</script>

</body>
</html>
"""
