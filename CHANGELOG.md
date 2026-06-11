# Changelog

## [1.0.0] — 2026-06-07

### First stable release

#### New in v1.0.0

**HybridExtractor — 3-pass extraction pipeline**
- Pass 1: LLM-powered structured extraction (JSON schema, ~85-90% recall)
- Pass 2: Enhanced heuristic sweep (file paths, decisions, errors, dependencies)
- Pass 3: Confidence-weighted merge — corroborated items reach 0.95 confidence
- Result: ~88-92% recall vs ~45-55% heuristic-only

**Streaming support**
- `/v1/chat/completions` now accepts `stream=true`
- SSE (Server-Sent Events) compatible with all OpenAI-SDK clients
- Works with Cursor, Continue.dev, Claude Code

**Rate limiting**
- Token-bucket rate limiter: 60 req/min per client, burst of 10
- Stale bucket eviction prevents memory growth
- Returns `Retry-After` header on 429

**Bug fixes**
- Session locks: replaced unbounded dict with LRU-bounded (max 1000 sessions)
- CHANGELOG recall numbers now match actual benchmark measurements
- Version aligned across pyproject.toml, __init__.py, and app.py

**Installer**
- Universal one-line installer: `curl -fsSL .../install.sh | bash`
- Auto-detects OS (macOS/Linux/WSL), Python version, installs if missing
- Auto-detects provider from 9 env vars (Anthropic, OpenAI, Gemini, DeepSeek,
  Mistral, Grok, Cohere, OpenRouter, Ollama)
- Prompts for API key interactively if none found
- Writes tokenmizer.yaml + .mcp.json automatically

---

## [0.1.0] — 2025

### Initial alpha release

**Known limitations in 0.1.0 (fixed in 1.0.0)**
- Heuristic extraction: ~45–55% task recall, ~70–80% file recall
- No streaming support
- Session lock dict unbounded (memory leak on long-running servers)
- No rate limiting on proxy endpoint
- Version mismatch across files
