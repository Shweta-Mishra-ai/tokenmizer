# Deployment

Running TokenMizer beyond `tokenmizer serve` on a laptop: containers, more than one worker, what survives a crash, and how sessions are kept apart from each other.

---

# Docker

```bash
# Quick start
docker-compose up tokenmizer

# With a provider key
ANTHROPIC_API_KEY=sk-ant-... docker-compose up

# With proxy auth
TOKENMIZER_API_KEY=strong-key docker-compose up
```

`stop_grace_period` is set to 30s because SIGTERM triggers a shutdown
flush (see [Durability](#durability--what-happens-when-something-breaks-mid-session));
cutting it short is what loses data.

## Running more than one worker

Graph and checkpoint **writes are safe across processes**: storage is
per-row and every persist is a read-modify-write under an OS-level file
lock (`fcntl` / `msvcrt`), so concurrent writers merge and a stale
writer adopts another's deletions instead of reinstating them. Measured
lossless with 4 processes writing one session — see
[Benchmarks](benchmarks.md).

What is **still per-process**, and therefore differs per worker:

| Component | Consequence of >1 worker |
|---|---|
| Rate limiter | Limits apply per worker — the effective limit is ~N× what you configured |
| Analytics | `/api/stats` reflects only the worker that served the request |
| Semantic cache | Cache hit rate drops; each worker warms its own |
| Graph LRU cache | A session's in-memory copy can be briefly stale between writes |

The image ships `--workers 1` for that reason. If you raise it, put a
real rate limiter in front and treat `/api/stats` as per-worker.

> `flock` is unreliable on NFS. Keep `storage_dir` on a local
> filesystem if you run more than one process against it.

---

# Durability — what happens when something breaks mid-session

The point of this tool is that you don't lose context. That has to hold
when things go wrong mid-session, not just when they go right.

| Failure | Behaviour |
|---|---|
| **Graceful stop** (SIGTERM: `docker stop`, k8s rollout, `systemctl restart`) | In-flight background extraction is drained (up to 10s), then every cached graph is force-persisted before exit. Nothing accepted is lost. |
| **Hard kill** (SIGKILL, OOM, power loss) | A background flush runs every `FLUSH_INTERVAL_SECONDS` (30s), so the worst case is bounded to the last 30s of graph activity. Anything already checkpointed is unaffected. |
| **Database refuses writes** | A graph that fails to persist is **kept in memory** instead of being evicted, so nothing is dropped before it has been saved. The cache deliberately runs over its cap until the write succeeds. |
| **Cache eviction during a request** | Sessions with an in-flight request are never selected for eviction, so a request cannot have its graph persisted and detached out from under it. |
| **Transient DB lock** (`database is locked`) | Treated as contention, not corruption: nothing is deleted, and a graph that failed to *read* refuses to *write* over the stored row rather than replacing it with an empty one. |
| **Corrupt database file** | The file is **quarantined by rename** (`graph_memory.db.corrupt-<timestamp>`), never deleted, and recovery is scoped to the affected session's row where possible. Recover with `sqlite3 <quarantined> '.recover' \| sqlite3 graph_memory.db`. |
| **One corrupt row** | Costs that node or edge only — the rest of the session loads normally. |
| **Anything was actually lost** | Surfaced as `data_loss_detected` in `GET /api/graph/{id}` and as `persist_failures` in `GET /api/stats` — queryable, not just a log line. |

Both SQLite databases are shared by every session in a `storage_dir`,
which is why "delete the file and start fresh" is never the recovery
path: it would discard every other session too.

## Storage layout

Graph state is stored **one row per node and per edge** (`graph_nodes`,
`graph_edges`, `graph_meta`). A persist writes only what changed —
adding one node to a 151-node graph writes 1 row, and a turn that changes
nothing writes none.

Databases written before v0.5.0 used a single JSON blob per session.
They are migrated automatically the first time each session is opened,
one session at a time. The old row is kept, not deleted, so a downgrade
still finds the data it expects as of the moment of migration — changes
made after upgrading are lost if you roll back, which is what a rollback
means. Nothing needs to be run by hand.

# Session isolation

A session is claimed by the first API key that uses it, and only that key
can read or modify it afterwards (`GET /api/graph/{id}`, `/api/resume/{id}`,
`POST /api/checkpoint`, `POST /api/decision/invalidate`, and the chat
endpoint itself). Requests for someone else's session return 404 rather
than 403, so the endpoints can't be used to probe which session names
exist.

```yaml
api_key: primary-key            # the deployment credential
api_keys:                       # additional credentials...
  - second-team-key             # ...each of which is its own principal
```

* **No key configured (default):** every caller is the same principal —
  local single-user use, unchanged.
* **One key:** every caller is the same principal. This is a
  **single-tenant** deployment: everyone holding the key shares one
  session namespace.
* **Multiple keys:** sessions are genuinely isolated per key.

Since `session_id` is chosen by the client, don't treat it as a secret;
isolation comes from the credential, not from the id being hard to guess.

---

# Security

- API key auth — `TOKENMIZER_API_KEY` (constant-time comparison)
- Secret/PII redaction applied once at ingestion, before graph storage,
  checkpoint storage, and every LLM call (main chat and the background
  extraction model). Patterns cover Anthropic/OpenAI/Google/GitHub/AWS/
  Slack/Stripe/JWT/OpenRouter/HF/xAI keys, URL-embedded credentials
  (`postgres://user:pass@host`), and generic `key=`/`password=`
  assignments. Best-effort by nature — an unrecognized format with no
  keyword context can still slip through. The checkpoint layer
  independently re-redacts what it persists (defense in depth).
- Session-isolated cache (sensitive data never shared across sessions)
- Basic prompt-injection keyword filter — catches copy-pasted jailbreak
  templates only; **not** a security boundary against a motivated
  adversary. See [SECURITY.md](../SECURITY.md)
  for exactly what it does and doesn't catch.
- CORS restricted to configured origins by default

---


---

[← Back to the README](../README.md)

## Releasing

Two ways, both running the same checks and publishing the same artifacts.

**From the Actions tab** — Actions → *Release to PyPI* → Run workflow, and
type the version. The typed version is compared against
`tokenmizer.__version__`, so a stray click cannot publish: getting it
wrong fails the run before anything is built. This path creates the tag
and the GitHub Release for you.

**From a GitHub Release** — create one whose tag is `v<version>`. The tag
is the source of truth and is checked the same way.

Either way the run: refuses a version already on PyPI (a version can
never be replaced), runs the full suite and ruff, builds an sdist and a
wheel, runs `twine check`, and uploads through PyPI Trusted Publishing —
no API token is stored anywhere.

The ordering is deliberate. Publishing is the irreversible step, so the
tag and the GitHub Release are created **after** PyPI accepts the upload,
never before. A tag is deletable; a published version is not. If tagging
fails, PyPI is still correct and the tag can be added by hand — the
reverse would leave the history asserting a release that never happened.

`tests/unit/test_release_workflow.py` holds these properties: publish
depends on the tests, the tag depends on the publish, caller input
reaches scripts through `env:` rather than string interpolation, and only
the publishing job carries the OIDC token.

---

[← Back to the README](../README.md)
