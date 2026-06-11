---
name: resume
description: Load a previous session from TokenMizer graph memory. Returns a compact context block (100-600 tokens) covering goals, completed work, decisions, open tasks, and files. Inject this as system context to continue exactly where you left off. Use when user says "resume", "continue from last time", "load my project", "what did we do on X", or starts a session on a known project.
---

Load a previous session from TokenMizer graph memory.

## What to do

1. Get the session ID from $ARGUMENTS (or ask the user if not provided)
2. Determine the level from $ARGUMENTS:
   - `critical` = ~100 tokens, only open blockers + key decisions
   - `standard` = ~300 tokens, normal resume (default)
   - `full` = ~600 tokens, everything including env, schemas, endpoints

3. Call the TokenMizer resume API:

```bash
SESSION_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
LEVEL=$(echo "$ARGUMENTS" | awk '{print $2}')
LEVEL=${LEVEL:-standard}

curl -s "http://localhost:8000/api/resume/${SESSION_ID}?level=${LEVEL}"
```

4. Inject the returned `resume_context` as a system message at the top of the conversation — NOT as a user message.

5. Tell the user: "Loaded [X] tokens of context for '[session-id]'. Continuing from: [next_action]"

## Format for system injection

```
[TokenMizer — session: {session_id}]
{resume_context}
[End of session context — continue from here]
```

## If no checkpoint found

Tell the user: "No checkpoint found for '[session-id]'. Either the session hasn't been checkpointed yet, or TokenMizer isn't running."

Suggest: `tokenmizer checkpoint {session-id}` to save the current session first.

## Examples of $ARGUMENTS

- `auth-service` → load standard resume for auth-service
- `auth-service full` → load full 600-token resume
- `auth-service critical` → load critical 100-token resume
- (empty) → ask user which project to resume
