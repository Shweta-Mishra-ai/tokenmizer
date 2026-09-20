"""Dashboard HTML — served at GET /"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TokenMizer — Never Lose AI Context</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #151822; --border: #2a2d3e;
    --accent: #7c6af7; --accent2: #5ee7c8; --text: #e8eaf6; --muted: #8b8fa8;
    --faint: #5b6078; --green: #4ade80; --yellow: #fbbf24; --red: #f87171;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Inter', system-ui, -apple-system, sans-serif; min-height: 100vh; }

  header { border-bottom: 1px solid var(--border); padding: 1rem 2rem;
           display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
  header .logo { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em; }
  header .logo span { color: var(--accent); }
  header .tagline { color: var(--muted); font-size: 0.85rem; }
  header .spacer { margin-left: auto; }
  .health { display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.8rem;
            border: 1px solid var(--border); border-radius: 20px; padding: 4px 12px;
            color: var(--muted); cursor: default; }
  .health .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .health.ok .dot { background: var(--green); animation: pulse 2s infinite; }
  .health.ok { color: var(--green); border-color: #1f3a24; }
  .health.bad .dot { background: var(--red); }
  .health.bad { color: var(--red); border-color: #3a1f1f; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

  .container { max-width: 1240px; margin: 0 auto; padding: 2rem; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .grid-2 { display: grid; grid-template-columns: 1.05fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }

  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 1.4rem 1.5rem; }
  .card h3 { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em;
             color: var(--muted); margin-bottom: 0.75rem; display: flex;
             justify-content: space-between; align-items: center; }
  .card h3 em { font-style: normal; text-transform: none; letter-spacing: 0;
                color: var(--faint); font-size: 0.72rem; }
  .stat-value { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
  .stat-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }
  .stat-value.green { color: var(--green); }
  .stat-value.accent { color: var(--accent); }
  .stat-value.accent2 { color: var(--accent2); }

  .layer-row { display: flex; justify-content: space-between; align-items: center;
               padding: 0.45rem 0; border-bottom: 1px solid var(--border); }
  .layer-row:last-child { border-bottom: none; }
  .layer-name { font-size: 0.86rem; }
  .layer-badge { font-size: 0.72rem; padding: 2px 10px; border-radius: 20px; font-weight: 600; }
  .badge-active { background: #1a2e1a; color: var(--green); }
  .badge-cache   { background: #1a2040; color: var(--accent); }
  .badge-graph   { background: #1e1a40; color: #a78bfa; }
  .badge-terse   { background: #2a2010; color: var(--yellow); }
  .badge-routing { background: #201a1a; color: var(--faint); }

  .srow { display: flex; justify-content: space-between; gap: 1rem; padding: 0.5rem 0;
          border-bottom: 1px solid var(--border); }
  .srow:last-child { border-bottom: none; }
  .srow a { color: var(--accent); text-decoration: none; font-weight: 600; font-size: 0.9rem; }
  .srow a:hover { text-decoration: underline; }
  .srow .meta { color: var(--muted); font-size: 0.76rem; margin-top: 2px; }
  .srow .when { color: var(--faint); font-size: 0.76rem; white-space: nowrap; }
  .srow.sel { background: var(--surface2); border-radius: 8px;
              padding-left: 0.5rem; padding-right: 0.5rem; }
  .pick { cursor: pointer; }

  .resume-block { background: var(--bg); border: 1px solid var(--accent); border-radius: 8px;
    padding: 1rem 1.25rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem; line-height: 1.7; color: var(--accent2);
    white-space: pre-wrap; word-break: break-word; min-height: 3rem; }
  .preview { width: 100%; height: 420px; border: 1px solid var(--border);
             border-radius: 10px; background: var(--bg); display: block; }
  .muted { color: var(--muted); font-size: 0.82rem; line-height: 1.6; }
  .faint { color: var(--faint); font-size: 0.78rem; }
  code { background: var(--surface2); padding: 1px 6px; border-radius: 4px;
         font-size: 0.78rem; color: var(--text); font-family: ui-monospace, monospace; }
  .steps { list-style: none; counter-reset: s; }
  .steps li { counter-increment: s; padding: 0.55rem 0 0.55rem 2rem; position: relative;
              border-bottom: 1px solid var(--border); font-size: 0.86rem; }
  .steps li:last-child { border-bottom: none; }
  .steps li::before { content: counter(s); position: absolute; left: 0; top: 0.5rem;
    width: 1.3rem; height: 1.3rem; border-radius: 50%; background: var(--surface2);
    color: var(--accent); font-size: 0.72rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; }
  .warn { border-color: #3a2a1a; background: #1f1a12; }
  .warn h3 { color: var(--yellow); }

  .endpoint-box { background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem 1.25rem; font-family: ui-monospace, monospace; font-size: 0.8rem;
    color: var(--muted); line-height: 1.9; overflow-x: auto; }
  .endpoint-box .method { color: var(--green); font-weight: 700; }
  .endpoint-box .path { color: var(--text); }

  footer { border-top: 1px solid var(--border); padding: 1.5rem 2rem; text-align: center;
           color: var(--muted); font-size: 0.8rem; margin-top: 2rem; }
  footer a { color: var(--accent); }

  @media (max-width: 900px) {
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .grid-2 { grid-template-columns: 1fr; }
    .container { padding: 1rem; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">Token<span>Mizer</span></div>
  <div class="tagline">Never lose your AI context again.</div>
  <div class="spacer"></div>
  <div class="health" id="health"><span class="dot"></span><span id="health-text">checking</span></div>
</header>

<div class="container">

  <!-- Degraded banner: shown only when /health says something was lost. -->
  <div class="card warn" id="degraded" style="display:none;margin-bottom:1.5rem">
    <h3>Attention <em>from /health</em></h3>
    <div class="muted" id="degraded-body"></div>
  </div>

  <div class="grid-4" id="stats-row">
    <div class="card">
      <h3>Requests today</h3>
      <div class="stat-value accent" id="req-today">&mdash;</div>
      <div class="stat-sub">All sessions combined</div>
    </div>
    <div class="card">
      <h3>Tokens saved today</h3>
      <div class="stat-value green" id="tokens-saved">&mdash;</div>
      <div class="stat-sub" id="savings-pct">Loading</div>
    </div>
    <div class="card">
      <h3>Cache hit rate</h3>
      <div class="stat-value accent2" id="cache-hit-rate">&mdash;</div>
      <div class="stat-sub" id="cache-entries">Loading</div>
    </div>
    <div class="card">
      <h3>Sessions remembered</h3>
      <div class="stat-value" id="session-count">&mdash;</div>
      <div class="stat-sub" id="session-nodes">Loading</div>
    </div>
  </div>

  <!-- Getting started: shown only while there is nothing to show. -->
  <div class="card" id="getting-started" style="display:none;margin-bottom:1.5rem">
    <h3>Getting started</h3>
    <ol class="steps">
      <li>Point your client at <code>http://localhost:8000/v1</code> and pass
          <code>session_id</code> in the request body. Every turn builds the graph.</li>
      <li>Not using the proxy? Send a transcript straight in:
          <code>POST /api/checkpoint?session_id=my-project</code> with a
          <code>{"messages": [...]}</code> body. The MCP tool and
          <code>tokenmizer checkpoint</code> do exactly this.</li>
      <li>Come back here. Your sessions, their graphs and the resume block
          they produce appear below.</li>
    </ol>
    <div class="faint" style="margin-top:0.9rem">
      Nothing extracted from a session you did send? The default extractor keys on
      explicit phrasing (<code>decided to use X</code>, <code>fixed Y</code>,
      <code>working on Z</code>, file paths). For free-form chat set
      <code>graph_checkpoint.use_llm_extraction: true</code>; a local model server
      works with no API key.
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>Sessions <em id="sessions-hint"></em></h3>
      <div id="sessions" style="font-size:0.85rem">
        <span class="muted">Loading</span>
      </div>
    </div>

    <div class="card">
      <h3>Pipeline layers</h3>
      <div class="layer-row">
        <span class="layer-name">Graph memory + checkpoint</span>
        <span class="layer-badge badge-graph">Core</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Semantic cache</span>
        <span class="layer-badge badge-cache">Cache</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Prompt compression</span>
        <span class="layer-badge badge-active">Active</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Terse output</span>
        <span class="layer-badge badge-terse">Active</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Model routing</span>
        <span class="layer-badge badge-routing" title="routing.* is accepted by the config but nothing implements it yet">Not implemented</span>
      </div>
      <div class="faint" style="margin-top:0.9rem">
        Savings per layer are in <code>/api/stats</code> and on every chat response
        under <code>tokenmizer.savings</code>.
      </div>
    </div>
  </div>

  <!-- The live resume block for the selected session. This card used to
       show a hard-coded example, so it said the same thing on a fresh
       install and on a deployment with a thousand sessions. -->
  <div class="card" style="margin-bottom:1.5rem">
    <h3>Resume block <em id="resume-meta"></em></h3>
    <div class="resume-block" id="resume-example"><span class="muted">Select a session to see what it would inject.</span></div>
    <div class="faint" style="margin-top:0.75rem" id="resume-note">
      This is the block injected into the system prompt when the session resumes.
    </div>
  </div>

  <div class="card" style="margin-bottom:1.5rem">
    <h3>Session graph <em id="preview-meta"></em></h3>
    <iframe class="preview" id="preview" title="Session graph"></iframe>
    <div class="faint" style="margin-top:0.75rem">
      Nodes filled by detected community, ringed by type. Open the full page for the
      timeline view, decision history and PNG export.
    </div>
  </div>

  <div class="card">
    <h3>API endpoints</h3>
    <div class="endpoint-box">
<span class="method">POST</span> <span class="path">/v1/chat/completions</span>  &mdash; drop-in OpenAI-compatible proxy, tools included
<span class="method">POST</span> <span class="path">/api/checkpoint?session_id=</span>  &mdash; checkpoint, optionally with a transcript
<span class="method">GET </span> <span class="path">/api/resume/{session_id}</span>  &mdash; resume context, live from the graph
<span class="method">GET </span> <span class="path">/api/graph/{session_id}/html</span>  &mdash; interactive session graph
<span class="method">GET </span> <span class="path">/api/graph/{session_id}/why?q=</span>  &mdash; why a decision is the current one
<span class="method">GET </span> <span class="path">/api/sessions</span>  &mdash; your sessions
<span class="method">GET </span> <span class="path">/api/stats</span>  &mdash; savings analytics
<span class="method">GET </span> <span class="path">/health</span>  &mdash; liveness and durability counters
<span class="method">GET </span> <span class="path">/docs</span>  &mdash; interactive API docs (Swagger)
    </div>
  </div>

</div>

<footer>
  TokenMizer &middot; open source &middot; <a href="/docs">API docs</a> &middot;
  <a href="/health">Health</a> &middot;
  <a href="https://github.com/Shweta-Mishra-ai/tokenmizer">GitHub</a>
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

function esc(text) {
  return String(text).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// A link cannot carry the Authorization header the other dashboard
// requests send, so with api_key set the graph page answered 401 straight
// from the dashboard. Without a stored key the anchor navigates natively
// (dev mode, middle-click, copy-link all keep working). With one, the tab
// is opened synchronously inside the click — after an await the user
// gesture is gone and popup blocking eats it — then the page is fetched
// with the key and written into that tab (or this one, where popups are
// blocked). The page is self-contained, so nothing in it needs auth, and
// writing the document avoids a blob: navigation, which embedded browsers
// refuse. The key never goes in a URL.
async function openGraph(ev) {
  const link = ev.target.closest('a[data-graph]');
  if (!link || !getStoredApiKey()) return;
  ev.preventDefault();
  const tab = window.open('', '_blank');
  try {
    const r = await dashboardFetch(link.getAttribute('href'));
    if (!r.ok) { if (tab) tab.close(); return; }
    const html = await r.text();
    const doc = tab ? tab.document : document;
    doc.open(); doc.write(html); doc.close();
  } catch (e) {
    if (tab) tab.close();
    console.warn('Graph page load failed:', e);
  }
}

let selectedSession = null;
let knownSessions = [];

function renderSessions(payload) {
  const el = document.getElementById('sessions');
  const sessions = (payload && payload.sessions) || [];
  knownSessions = sessions;
  document.getElementById('getting-started').style.display = sessions.length ? 'none' : '';
  document.getElementById('session-count').textContent = sessions.length.toLocaleString();
  const totalNodes = sessions.reduce((a, s) => a + (s.node_count || 0), 0);
  document.getElementById('session-nodes').textContent =
    sessions.length ? `${totalNodes.toLocaleString()} nodes remembered` : 'None yet';

  if (!sessions.length) {
    el.innerHTML = '<span class="muted">No sessions yet. The steps above take about a minute.</span>';
    document.getElementById('sessions-hint').textContent = '';
    return;
  }
  document.getElementById('sessions-hint').textContent = 'click one to inspect';
  el.innerHTML = sessions.slice(0, 12).map(s => {
    const t = s.by_type || {};
    const parts = ['decision','task','error','file'].filter(k => t[k])
      // A literal character, not an entity: this string goes through esc()
      // with the session id, so "&middot;" would render as its own source.
      .map(k => `${t[k]} ${k}${t[k] === 1 ? '' : 's'}`).join(' · ');
    const when = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString() : '';
    const sel = s.session_id === selectedSession ? ' sel' : '';
    return `<div class="srow pick${sel}" data-session="${esc(s.session_id)}">` +
      `<div><a href="/api/graph/${encodeURIComponent(s.session_id)}/html" data-graph>` +
      `${esc(s.session_id)}</a>` +
      `<div class="meta">${esc(parts || (s.node_count + ' nodes'))}</div></div>` +
      `<div class="when">${esc(when)}</div></div>`;
  }).join('');

  if (!selectedSession || !sessions.some(s => s.session_id === selectedSession)) {
    selectSession(sessions[0].session_id);
  }
}

async function selectSession(id) {
  if (!id) return;
  selectedSession = id;
  document.querySelectorAll('.srow').forEach(r =>
    r.classList.toggle('sel', r.dataset.session === id));
  document.getElementById('preview-meta').textContent = id;
  await Promise.all([loadResume(id), loadPreview(id)]);
}

async function loadResume(id) {
  const box = document.getElementById('resume-example');
  const meta = document.getElementById('resume-meta');
  const note = document.getElementById('resume-note');
  try {
    const r = await dashboardFetch('/api/resume/' + encodeURIComponent(id));
    if (!r.ok) {
      box.innerHTML = '<span class="muted">Nothing to resume from yet for this session.</span>';
      meta.textContent = ''; return;
    }
    const d = await r.json();
    box.textContent = d.resume_context || '';
    meta.textContent = `${id} — ${d.token_count} tokens`;
    note.textContent = d.source === 'live_graph'
      ? 'Built live from the session graph. This is what would be injected right now.'
      : `From checkpoint ${d.checkpoint_id || ''}. This is what would be injected on resume.`;
  } catch (e) {
    box.innerHTML = '<span class="muted">Could not load the resume block.</span>';
  }
}

// The graph page is self-contained, so it renders from srcdoc with no
// second authenticated request from inside the frame.
async function loadPreview(id) {
  const frame = document.getElementById('preview');
  try {
    const r = await dashboardFetch('/api/graph/' + encodeURIComponent(id) + '/html');
    if (!r.ok) { frame.removeAttribute('srcdoc'); return; }
    frame.srcdoc = await r.text();
  } catch (e) {
    frame.removeAttribute('srcdoc');
  }
}

function renderHealth(h) {
  const pill = document.getElementById('health');
  const text = document.getElementById('health-text');
  const banner = document.getElementById('degraded');
  if (!h) {
    pill.className = 'health'; text.textContent = 'unreachable';
    banner.style.display = 'none'; return;
  }
  const ok = h.status === 'ok';
  pill.className = 'health ' + (ok ? 'ok' : 'bad');
  text.textContent = ok ? 'healthy' : 'degraded';
  if (ok) { banner.style.display = 'none'; return; }

  const lines = [];
  const f = h.persist_failures || {};
  Object.keys(f).sort().forEach(k => lines.push(
    `<b>${esc(k)}</b> failed ${f[k]} time${f[k] === 1 ? '' : 's'} since start.`));
  (h.sessions_with_unreadable_graph || []).forEach(s => lines.push(
    `Stored graph for <code>${esc(s)}</code> could not be read. Nothing was lost; it retries.`));
  (h.sessions_without_durable_storage || []).forEach(s => lines.push(
    `<code>${esc(s)}</code> is running without durable storage — it will not survive a restart.`));
  (h.sessions_with_data_loss || []).forEach(s => lines.push(
    `Memory for <code>${esc(s)}</code> was displaced during database recovery.`));
  if (h.checkpoint_storage_broken) lines.push('Checkpoint storage could not be initialised.');
  if (h.checkpoint_data_loss) lines.push('The checkpoint database was quarantined after corruption.');
  lines.push('Full detail is in the server log and under <code>/api/stats</code>.');
  document.getElementById('degraded-body').innerHTML = lines.map(l => `<div>${l}</div>`).join('');
  banner.style.display = '';
}

async function loadStats() {
  try {
    const [statsRes, cacheRes, sessionsRes, healthRes] = await Promise.all([
      dashboardFetch('/api/stats').catch(() => null),
      dashboardFetch('/api/cache/stats').catch(() => null),
      dashboardFetch('/api/sessions').catch(() => null),
      fetch('/health').catch(() => null),
    ]);

    renderHealth(healthRes && healthRes.ok ? await healthRes.json() : null);

    if (sessionsRes && sessionsRes.ok) {
      renderSessions(await sessionsRes.json());
    } else {
      renderSessions(null);
    }

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

document.getElementById('sessions').addEventListener('click', ev => {
  if (ev.target.closest('a[data-graph]')) return openGraph(ev);
  const row = ev.target.closest('.srow');
  if (row && row.dataset.session) selectSession(row.dataset.session);
});
loadStats();
setInterval(loadStats, 15000);
</script>

</body>
</html>
"""
