---
name: checkpoint
description: Save the current session to TokenMizer graph memory. Creates a persistent checkpoint with all tasks, decisions, files, and errors — resumable in any future session. Use when user says "save", "checkpoint", "remember this", "I'm done for today", or session is getting long.
---

Save the current session to TokenMizer graph memory.

## What to do

1. Ask the user for a session ID if not provided (suggest a slug based on what you're working on, e.g. "auth-service", "data-pipeline", "my-project")
2. Call the TokenMizer checkpoint API:

```bash
curl -s -X POST "http://localhost:8000/api/checkpoint?session_id=$ARGUMENTS" \
  -H "Content-Type: application/json"
```

3. Show the user what was saved:
   - Checkpoint ID
   - Number of nodes in graph
   - Resume token count
   - The resume context block

## If TokenMizer is not running

Tell the user to start it first:
```bash
tokenmizer serve
# or
python3 -m tokenmizer.api.app
```

## Session ID rules
- Use lowercase slugs: `auth-service` not `Auth Service`
- Keep it short and meaningful
- Same ID across sessions for the same project

## Example output to show user

```
✅ Session 'auth-service' saved

Checkpoint: ckpt_a3f9b2
Nodes: 14 (6 tasks, 4 decisions, 3 files, 1 error)
Resume size: 247 tokens

Resume context:
Goal: Build FastAPI auth service with JWT
Done: Project setup | User model | Login endpoint | Fix 422
In progress: Refresh token rotation
Decided: PostgreSQL | bcrypt | Redis for tokens
Files: api/auth.py, api/models.py, config.py
Continue: Implement token refresh endpoint
```

If $ARGUMENTS is empty, ask: "What should I call this session? (e.g. my-project)"
