---
name: resume
description: Load a previous session from TokenMizer graph memory. Returns a compact context block (100-600 tokens) covering goals, completed work, decisions, open tasks, and files. Inject this as system context to continue exactly where you left off. Use when user says "resume", "continue from last time", "load my project", "what did we do on X", or starts a session on a known project.
---

Load a previous session from TokenMizer graph memory.

## Session ID

Do not ask for one. The first word of `$ARGUMENTS` is the session ID. If
`$ARGUMENTS` is empty, TokenMizer derives the ID from the working directory
name, the same way `checkpoint` does, and prints the ID it chose.

Ask only if the command exits 2, which means the directory has no usable name
to derive from. Then ask exactly: `Which session ID should I resume?`

## Level

The second word of `$ARGUMENTS`, if present:

| Level | Size | Contents |
|---|---|---|
| `critical` | ~100 tokens | open blockers and key decisions only |
| `standard` | ~300 tokens | the default |
| `full` | ~600 tokens | adds environment, schemas, endpoints |

## What to do

```bash
tokenmizer resume $ARGUMENTS
```

Then inject the returned resume context as a **system** message at the top of
the conversation, never as a user message, in this wrapper:

```
[TokenMizer — session: {session_id}]
{resume_context}
[End of session context — continue from here]
```

## Output

Report in exactly this shape:

```
Resumed auth-service — 247 tokens, standard
Next: implement the token refresh endpoint
```

Take `Next` from the resume context's continue line. If it has none, omit the
line rather than inventing one. Add no commentary, and do not address the
operator by name.

## If no checkpoint is found

The command exits 1 and prints what happened. Relay that, then:

```
Nothing saved for auth-service yet. Save the current session first:
  tokenmizer checkpoint auth-service
```
