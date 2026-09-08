---
name: checkpoint
description: Save the current session to TokenMizer graph memory. Creates a persistent checkpoint with all tasks, decisions, files, and errors — resumable in any future session. Use when user says "save", "checkpoint", "remember this", "I'm done for today", or session is getting long.
---

Save the current session to TokenMizer graph memory.

## Session ID

Do not ask for one. `$ARGUMENTS`, if given, is the session ID. If it is empty,
TokenMizer derives the ID from the working directory name — stable across
sessions on the same project, and the convention the docs already recommend
(`~/projects/auth-service` → `auth-service`). The command prints the ID it
chose.

Ask only if the command exits 2, which means the directory has no usable name
to derive from (a filesystem root, or a generic folder such as `tmp`). Then
ask exactly: `Which session ID should this be saved under?`

## What to do

```bash
tokenmizer checkpoint $ARGUMENTS
```

Relay the command's output. Do not restate it in your own words, do not add
commentary or congratulation, and do not address the operator by name.

## Output

Report the result in exactly this shape, filled from the command's output:

```
Checkpoint saved — auth-service
  checkpoint     ckpt_a3f9b2
  nodes          14 (6 tasks, 4 decisions, 3 files, 1 error)
  resume cost    247 tokens

Resume with: tokenmizer resume auth-service
```

If the command exits non-zero, show its error text unchanged. It already says
what to do — an unreachable server prints the `tokenmizer serve` instruction
itself.

## Notes

- Session IDs are lowercase slugs: `auth-service`, not `Auth Service`.
- Use the same ID across sessions for the same project. That is what makes a
  later resume find anything.
- The server must be running for this command; `tokenmizer serve` starts it.
