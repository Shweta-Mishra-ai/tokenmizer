# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x (alpha) | ✅ Active |

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
- Cache entries (in-memory or Redis)
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

The semantic cache applies three-tier scoping:

**Blocked (never cached):**
- API keys, passwords, tokens, connection strings
- Content matching known secret patterns

**Session-scoped (isolated per session_id):**
- Project-specific content
- Code from current session
- Long prompts with embedded data
- Any content matching privacy heuristics

**Cross-session shared:**
- Generic how-to queries
- Explanations of public concepts
- Short, non-sensitive prompts

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

---

## Prompt Injection

Incoming requests are scanned for common injection patterns:
- "ignore all previous instructions"
- "print your system prompt"
- "bypass your restrictions"

Detected injection attempts return `429 Too Many Requests` and are logged.

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

For encryption at rest (coming in v0.2.0), set:
```yaml
encrypt_storage: true
encryption_key: ""  # 32-byte base64 key from env
```

---

## Known Limitations (Alpha)

- No multi-user isolation yet — all sessions share the same SQLite database
- Graph storage is not encrypted at rest in v0.1.0
- Session IDs are not authenticated — any caller who knows a session_id can access its graph

These are tracked and planned for v0.2.0.
