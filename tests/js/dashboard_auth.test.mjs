// Behavioral tests for the dashboard's client-side auth retry logic
// (getStoredApiKey/authOptions/requestApiKey/dashboardFetch), extracted
// verbatim from the real DASHBOARD_HTML string and executed in a Node vm
// context with mocked sessionStorage/window.prompt/fetch.
//
// Closes GitHub issue #36: the existing Python test
// (tests/unit/test_dashboard.py) only asserts that certain substrings
// appear in DASHBOARD_HTML — it would still pass if the logic inside
// those functions were subtly wrong, as long as the right tokens were
// present somewhere in the file. This file actually runs the shipped
// code and checks its behavior for the three paths the issue named:
// concurrent 401s sharing one prompt, a cancelled prompt staying
// dismissed for the rest of the session, and a stale stored key being
// dropped before re-prompting.
//
// Run directly: node tests/js/dashboard_auth.test.mjs
// Wired into the suite via tests/unit/test_dashboard_auth_js.py, which
// skips (not fails) if `node` isn't on PATH.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pagePyPath = path.join(__dirname, '../../tokenmizer/dashboard/page.py');
const pageSrc = readFileSync(pagePyPath, 'utf8');

const scriptStart = pageSrc.indexOf('const API_KEY_STORAGE_KEY');
const scriptEnd = pageSrc.indexOf('async function loadStats');
assert.ok(scriptStart !== -1 && scriptEnd !== -1 && scriptEnd > scriptStart,
  'could not locate the auth-retry block in tokenmizer/dashboard/page.py — ' +
  'has DASHBOARD_HTML been restructured?');
const authScript = pageSrc.slice(scriptStart, scriptEnd);

/**
 * Builds a fresh vm context with mocked sessionStorage/window/fetch and
 * the real dashboardFetch/requestApiKey/getStoredApiKey/authOptions
 * loaded into it. Each test gets its own context so module-level state
 * (apiKeyPrompt, apiKeyPromptDismissed) never leaks between tests.
 */
function makeContext({ promptResponses = [], responses = [] } = {}) {
  const store = new Map();
  const fetchCalls = [];
  const promptCalls = { count: 0 };
  let responseIdx = 0;

  const sandbox = {
    sessionStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, v),
      removeItem: (k) => store.delete(k),
    },
    window: {
      prompt: (msg) => {
        const value = promptResponses[promptCalls.count];
        promptCalls.count += 1;
        return value === undefined ? null : value;
      },
    },
    fetch: (fetchPath, opts) => {
      fetchCalls.push({ path: fetchPath, opts });
      const status = responses[Math.min(responseIdx, responses.length - 1)];
      responseIdx += 1;
      return Promise.resolve({ status, ok: status >= 200 && status < 300 });
    },
    console,
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(authScript, context);
  return { context, store, fetchCalls, promptCalls };
}

test('stale-key path: a stored key that gets 401 is removed before re-prompting, and the retry uses the new key', async () => {
  const { context, store, fetchCalls, promptCalls } = makeContext({
    promptResponses: ['new-key'],
    responses: [401, 200],
  });
  store.set('tokenmizer.apiKey', 'old-key');

  const result = await vm.runInContext("dashboardFetch('/api/stats')", context);

  assert.equal(promptCalls.count, 1, 'expected exactly one prompt for the stale key');
  assert.equal(fetchCalls.length, 2, 'expected an initial fetch plus one retry');
  assert.equal(fetchCalls[0].opts.headers.Authorization, 'Bearer old-key');
  assert.equal(fetchCalls[1].opts.headers.Authorization, 'Bearer new-key',
    'retry must use the freshly-prompted key, not the stale one');
  assert.equal(store.get('tokenmizer.apiKey'), 'new-key',
    'the stale key must be gone from storage, replaced by the new one');
  assert.equal(result.status, 200);
});

test('concurrent-401 path: two requests that both 401 at once share a single prompt', async () => {
  const { context, fetchCalls, promptCalls } = makeContext({
    promptResponses: ['shared-key'],
    responses: [401, 401, 200, 200],
  });

  const [a, b] = await vm.runInContext(
    "Promise.all([dashboardFetch('/api/stats'), dashboardFetch('/api/cache/stats')])",
    context
  );

  assert.equal(promptCalls.count, 1,
    `two concurrent 401s must share one prompt, got ${promptCalls.count} prompts`);
  assert.equal(fetchCalls.length, 4, 'each request retries once after the shared prompt resolves');
  assert.equal(a.status, 200);
  assert.equal(b.status, 200);
});

test('cancel path: dismissing the prompt once suppresses prompting for the rest of the session', async () => {
  const { context, fetchCalls, promptCalls } = makeContext({
    promptResponses: [''], // user cancels / submits empty
    responses: [401, 401],
  });

  const first = await vm.runInContext("dashboardFetch('/api/stats')", context);
  assert.equal(first.status, 401, 'no retry happens when the user cancels — the original 401 is returned');
  assert.equal(promptCalls.count, 1);

  const second = await vm.runInContext("dashboardFetch('/api/cache/stats')", context);
  assert.equal(second.status, 401);
  assert.equal(promptCalls.count, 1,
    'a later 401 in the same session must not prompt again after a cancel');
  assert.equal(fetchCalls.length, 2,
    'neither call retries — a dismissed prompt must not trigger a fetch with a key that was never obtained');
});

test('no-401 path: dashboardFetch does not prompt or retry when the response is not 401', async () => {
  const { fetchCalls, promptCalls, context } = makeContext({
    responses: [200],
  });

  const result = await vm.runInContext("dashboardFetch('/api/stats')", context);

  assert.equal(result.status, 200);
  assert.equal(fetchCalls.length, 1);
  assert.equal(promptCalls.count, 0);
});
