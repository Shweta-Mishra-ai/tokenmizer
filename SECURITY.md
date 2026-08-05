# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.5.x | ✅ Active |
| < 0.5 | ❌ Not supported — upgrade |

Only the latest minor release receives security fixes. TokenMizer is a
single-maintainer project; backporting to older lines is not something
that can be promised, so it is not promised.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: open a private GitHub Security Advisory at  
`https://github.com/Shweta-Mishra-ai/tokenmizer/security/advisories/new`

Include:
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (optional)

You will receive a response within 72 hours.

---

## Security Model

TokenMizer runs **locally on your machine**. It does not send data to any third-party service beyond the LLM provider you configure.

### What leaves your machine

- LLM requests — sent to your configured provider (Anthropic, OpenAI, etc.) after secret redaction
- Nothing else

### What stays local

- Graph memory (SQLite on disk)
- Checkpoints (SQLite on disk)
- Cache entries (in-process memory)
- All analytics

---

## Secret Redaction

Before any content is stored in the graph, checkpoints, or sent to the LLM, it passes through the redaction layer which strips:

- Anthropic API keys (`sk-ant-*`)
- OpenAI API keys (`sk-*`, `sk-proj-*`)
- Google API keys (`AIza*`)
- GitHub tokens (`ghp_*`, `ghs_*`)
- Generic secret patterns (`password=`, `token=`, `secret=`)
- Email addresses
- Bearer tokens
- Database connection strings

Redacted values are replaced with `[REDACTED]` before storage or transmission.

---

## Cache Privacy

**Default (`cache.share_scope: session`): nothing is ever shared across
sessions.** Every cached prompt — sensitive-looking or not — is scoped to
its `session_id` (or a private bucket if no session_id is given). This is
the safe default for hosted or multi-tenant use: a five-regex heuristic
cannot enumerate everything that might be confidential to a given
session, so the default does not rely on it.

**Opt-in (`cache.share_scope: shared`): non-sensitive prompts are shared
globally** across sessions for a higher cache hit rate. Even with this
enabled, prompts matching the sensitivity heuristic below are still
always session-scoped — the opt-in only affects prompts that DON'T match
it:

- API keys, passwords, tokens, connection strings, and anything matching
  known secret patterns are always excluded from sharing
- Project-specific content, code, long prompts with embedded data, and
  anything matching the privacy heuristics in
  `semantic_cache/cache.py::_is_session_sensitive` stay session-scoped

Only enable `shared` if you understand and accept that heuristic's
limits — it's a best-effort filter, not a guarantee, and a
misclassified prompt under `shared` mode can leak across sessions in a
way it cannot under the default.

---

## Authentication

When `TOKENMIZER_API_KEY` is set, all non-health API endpoints require:

```
Authorization: Bearer <key>
```
or
```
X-API-Key: <key>
```

Key comparison uses `hmac.compare_digest` — constant-time, immune to timing attacks.

In development mode (no key set), the proxy accepts all requests. **Do not expose the proxy to the internet without setting an API key.**

### Production mode: `TOKENMIZER_ENV=production`

By default (`TOKENMIZER_ENV` unset, or anything other than `production`),
a config load failure or a missing `api_key` logs an error and falls back
to permissive dev-mode defaults — convenient for local development, but a
config typo in a real deployment could otherwise boot unauthenticated
with only a log line as evidence.

Set `TOKENMIZER_ENV=production` to make TokenMizer **fail closed**
instead: it refuses to start at all (raises, non-zero exit) if:
- `tokenmizer.yaml` fails to parse, or contains an unrecognized key (a
  typo like `api_kye:` is a hard error, not a silently-dropped value —
  see `config/settings.py`'s `extra="forbid"`), or
- the resulting configuration has no `api_key` set at all, even if the
  YAML file itself loaded and validated perfectly fine.

Recommended for any deployment reachable outside your own machine.

---

## Prompt Injection (Basic Keyword Filter — Read the Scope)

Incoming requests are scanned against a denylist of common, unsophisticated
injection phrasings:
- "ignore all previous instructions"
- "print your system prompt"
- "bypass your restrictions"
- a handful of similar copy-pasted jailbreak templates (see
  `tokenmizer/security/middleware.py` for the full list)

**What this catches:** literal copy-pasted jailbreak templates from the
open web — the laziest, most common attempts.

**What this does NOT catch:** paraphrased injection, non-English injection,
encoded payloads (base64/unicode tricks), injection split across multiple
turns, or anything not matching the literal pattern list. This is a regex
denylist, not a trained classifier or a semantic detector. Treat it as one
weak, optional speed bump — not a security boundary. If your threat model
includes a motivated adversary, you need structural defenses (e.g. fencing
untrusted content away from instructions, minimizing what the LLM is
privileged to do regardless of its context) in addition to this filter,
not instead of it.

Matched requests return `400 Bad Request` (corrected from an earlier
version of this doc/code that incorrectly returned `429 Too Many Requests`
— 429 implies "retry later," which is wrong here; the request is rejected,
not rate-limited) and are logged at `warning` level.

---

## CORS

By default, CORS is restricted to configured origins (`cors_origins` in `tokenmizer.yaml`). The default is **not** `*`.

To add your frontend:
```yaml
cors_origins:
  - "http://localhost:3000"
  - "https://yourdomain.com"
```

---

## Graph Storage

Graph data and checkpoints are stored in SQLite files inside `./checkpoints/`. These files contain extracted project information — tasks, decisions, file names.

**They do not contain:**
- Full message content (only extracted structured facts)
- API keys or credentials (redacted before extraction)
- Raw user input (only normalized, structured graph nodes)

**Encryption at rest is NOT implemented.** There is no
`encrypt_storage` setting; an earlier version of this document showed
one, which was never real. Use filesystem- or volume-level encryption
(LUKS, FileVault, an encrypted Docker volume) if your data needs it.

Files are created with the process umask and are not further restricted.
On a shared host, set a restrictive umask or place `storage_dir` on a
directory only the service user can read.

---

## Session Isolation

A session is **claimed by the first credential that uses it**, and only
that credential can read or modify it afterwards. This applies to the
chat endpoint and to every session-scoped route
(`/api/graph/{id}`, `/api/resume/{id}`, `/api/checkpoint`,
`/api/decision/invalidate`, `/api/graph/{id}/transitions`, …).

```yaml
api_key: primary-key      # the deployment credential
api_keys:                 # additional credentials...
  - second-team-key       # ...each its own principal
```

| Configuration | Isolation |
|---|---|
| No key (default) | Single implicit principal — local, single-user |
| One key | Single principal. **Single-tenant**: everyone holding the key shares one session namespace |
| Multiple keys | Sessions are isolated per key |

Two properties worth stating explicitly:

- **`session_id` is not a secret.** Clients choose it in the request
  body. Isolation comes from the credential, never from the id being
  hard to guess. Do not put anything sensitive in a session id.
- **Denials return 404, not 403.** A distinguishable 403 would turn the
  session routes into an oracle for which session names exist.

Principals are stored as a SHA-256 prefix of the credential, so the
ownership table never contains a usable key.

---

## Known Limitations

Current, and deliberately listed rather than discovered:

- **Graph storage is not encrypted at rest.** Anyone with read access to
  `storage_dir` can read every session's graph and checkpoints. Use
  filesystem-level encryption if that matters for your data.
- **Redaction is best-effort.** It is pattern matching. A credential in
  an unrecognised format with no keyword context can reach the provider
  and the graph. It reduces exposure; it does not eliminate it.
- **The prompt-injection filter is a keyword denylist**, not a security
  boundary — see the section above for exactly what it misses.
- **Rate limits are per worker.** With more than one worker the
  effective limit is roughly N× what you configured; put a real limiter
  in front if that matters.
- **No audit log.** There is no record of which principal read which
  session.
- **No key rotation support.** Changing `api_key` orphans sessions
  claimed under the old one; they become unreachable rather than
  reassigned.
