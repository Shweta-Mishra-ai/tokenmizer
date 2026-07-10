# TokenMizer Demo Script / Storyboard

Target length: **2.5–3 minutes.** Four acts: checkpoint → resume → graph
visualization → live MCP tool calls from Claude Code. Every command below
was dry-run on 2026-07-10 against this exact codebase — nothing here is
aspirational.

## Prep (before recording — not in the video)

```powershell
# 1. Clean slate: delete old state so the demo graph is exactly what you build
Remove-Item -Recurse -Force .\checkpoints -ErrorAction SilentlyContinue

# 2. Two terminals side by side. Left = server, right = your commands.
#    Font size 16+, dark theme. Browser window ready on a second half of screen.

# 3. Left terminal — start the proxy and LEAVE IT VISIBLE (its log lines
#    during checkpoint are part of the show):
tokenmizer serve
```

Optional but recommended: set a real provider key so Act 1 can use live
chat; if you don't want live LLM calls in the demo, the curl-based variant
in Act 1 works with no key at all.

## Act 1 — Build up a session worth saving (~45s)

**Say:** "I've been working on an auth service all afternoon. My AI session
knows every decision I made — and it's about to hit the context limit."

Right terminal — feed the session through the proxy (paste one by one so
the server log visibly reacts; each is one `curl`):

```bash
SID=demo-auth-service

curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Goal: build a FastAPI auth service with JWT and PostgreSQL\"}],\"session_id\":\"$SID\",\"model\":\"claude-sonnet-4-6\"}" > /dev/null

curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Decided: use bcrypt for password hashing. Completed the login endpoint in api/auth.py, 18 tests passing.\"}],\"session_id\":\"$SID\",\"model\":\"claude-sonnet-4-6\"}" > /dev/null

curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Decided: use React for the frontend.\"}],\"session_id\":\"$SID\",\"model\":\"claude-sonnet-4-6\"}" > /dev/null

# THE MONEY LINE — a decision gets REVERSED. Say it out loud as you paste:
curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Actually, switch from React to Next.js for the frontend. Better SEO, benchmarked LCP 40 percent faster.\"}],\"session_id\":\"$SID\",\"model\":\"claude-sonnet-4-6\"}" > /dev/null
```

**Say:** "That last message reversed an earlier decision. Watch what
TokenMizer does with that."

> No API key? Add `"provider": "ollama"` config, or skip the provider
> entirely — extraction happens on the request side, so even failed
> upstream calls still build the graph. Verify with:
> `curl -s http://localhost:8000/api/graph/demo-auth-service | python -m json.tool`

## Act 2 — Checkpoint + resume (~40s)

```bash
tokenmizer checkpoint demo-auth-service
```

**Camera focus:** the green panel — checkpoint ID, node count, and the
resume context. **Say:** "My whole afternoon, checkpointed."

Now the punchline. **Say:** "Next morning. New session. Instead of
re-explaining the project for ten minutes:"

```bash
tokenmizer resume demo-auth-service
```

**Camera focus:** the resume block, especially the token count in the
title and the `Changed: React → Next.js` line. **Say:** "~250 tokens.
Notice it doesn't just remember the decisions — it remembers that Next.js
*replaced* React, so the model can't get confused by the stale choice."

## Act 3 — The graph (~40s)

Open in the browser:

```
http://localhost:8000/api/graph/demo-auth-service/html
```

Choreography (in this order — it's a story, not a tour):

1. Let the force layout settle (~2s). Nodes glow by type; the legend chips
   are bottom-left.
2. **Point at the dashed red arc** between the React and Next.js nodes.
   Say: "That's a supersession — the graph knows WHY this edge exists."
3. **Click the entry in the Decision history panel** (right side): the
   old decision is struck through, the new one bold, the reason under it.
   Clicking spotlights both nodes and dims everything else. Say: "Every
   reversal in the session, with trigger and evidence, one click away."
4. Press Escape to reset, click **"Active only"** — the superseded React
   node vanishes. Say: "Current truth only."
5. Type "bcrypt" in the search box — one node stays lit.
6. Click **⬇ PNG** — a shareable image downloads. Say: "And it's one file,
   no CDN, works offline — this page IS the export."

## Act 4 — Live MCP from Claude Code (~45s)

Prereq: `.mcp.json` in the project you open with Claude Code (this repo
ships one):

```json
{
  "mcpServers": {
    "tokenmizer": {
      "command": "python3",
      "args": ["-m", "tokenmizer.mcp.server"],
      "env": { "TOKENMIZER_URL": "http://localhost:8000" }
    }
  }
}
```

1. Open Claude Code in the repo. Type: **"Resume my tokenmizer session
   demo-auth-service"** — Claude calls the `resume_session` MCP tool; the
   tool result shows the same ~250-token context. Say: "Same memory, now
   inside my coding agent."
2. Type: **"What's in the knowledge graph for demo-auth-service?"** —
   Claude calls `get_graph_stats`; node/status breakdown appears.
3. (Optional, if time) **"Analyze data/sales.csv with tokenmizer"** —
   `analyze_file` returns schema + sample in a few hundred tokens.
   Prepare any CSV beforehand.
4. Close: split-screen the graph page + Claude Code. Say: "Checkpoint,
   resume, see the reasoning, use it from any MCP client. pip install
   tokenmizer."

## Failure modes to avoid on camera

- **Don't reuse a session id from prior takes** — old state reloads from
  SQLite and the graph shows stale nodes. Delete `./checkpoints` between
  takes (prep step 1).
- The resume panel says "graph was empty" if you checkpoint before Act 1's
  messages — order matters.
- If port 8000 is taken: `tokenmizer serve --port 8001` and adjust every
  URL (including `.mcp.json`'s `TOKENMIZER_URL`).
- Claude Code caches MCP server processes — restart it if you restarted
  the proxy, or the first tool call may hit a dead connection.
- On Windows, run the curls in Git Bash (as written) or convert to
  `Invoke-RestMethod` — cmd.exe quoting will mangle the JSON.
