"""
Extraction patterns — regex vocabulary and small pure text-analysis
helpers used by HybridExtractor's heuristic pass.

Extracted from hybrid_extractor.py to keep that file focused on the
extraction PIPELINE (LLM pass, heuristic orchestration, merge/dedup);
this one is DATA — the regex patterns and tiny stateless functions the
pipeline applies. Pure code motion, no behavior change: every name here
is re-exported by hybrid_extractor.py (see its imports), so existing
code that does `from tokenmizer.graph_memory.hybrid_extractor import
_clip` or similar keeps working unchanged.

Keep tests/unit/test_extraction_quality.py's F1 numbers as the
regression guard for any change here — that's what actually verifies
these patterns still work, a line-count split cannot.
"""
from __future__ import annotations

import re

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a technical memory extractor for AI coding sessions.
Extract ALL structured information from conversation messages.
Respond ONLY with valid JSON. No explanation, no markdown fences.

{
  "goals": ["string — the main objective of this session"],
  "tasks_done": ["string — completed work items, be specific"],
  "tasks_wip": ["string — work currently in progress"],
  "tasks_todo": ["string — planned but not started"],
  "decisions": [{"label": "string — what was decided", "reason": "string — why"}],
  "files": ["string — actual filenames or paths only, e.g. api/auth.py"],
  "errors": ["string — bugs, exceptions, failures encountered"],
  "dependencies": ["string — libraries, packages, tools added"],
  "environments": ["string — runtime versions, e.g. Python 3.12, Node 20"],
  "endpoints": ["string — API routes, e.g. POST /api/auth/login"],
  "schemas": ["string — data models, DB tables, e.g. users table"],
  "superseded": [{"old": "string — old decision", "new": "string — new decision"}]
}

CRITICAL RULES:
- decisions: extract ANY technology choice, architecture choice, or "going with X" statement
  Examples: "Use PostgreSQL", "bcrypt for passwords", "Redis for sessions", "JWT tokens"
  Look for: "decided:", "going with", "will use", "chose", "using X for Y", "switched to"
- tasks_done: extract ALL completed work. Look for: "completed:", "done:", "fixed:", "implemented:", "created:"
- files: extract EVERY filename mentioned with extension (.py, .js, .ts, .yaml, .json etc)
- superseded: when user or assistant says "switching from X to Y" or "instead of X, use Y"
- Max 20 items per category
- If nothing found for a category, use []
- NEVER fabricate — only extract what is explicitly stated"""


EXTRACTION_USER_TEMPLATE = """Extract from these conversation messages:

{messages_text}

Respond with JSON only."""


# ── Heuristic patterns (enhanced) ────────────────────────────────────────────

# Leading dot allowed: `.github/workflows/ci.yml` and `.env.example` are
# ordinary paths, and requiring an alphanumeric first character silently
# excluded every dotfile directory.
_FILE_PATH = re.compile(
    r'(?:^|[\s\'\"`(])((?:\.?[a-zA-Z0-9_\-]+)(?:/[a-zA-Z0-9_\-\.]+){1,6}\.[a-zA-Z]{1,6})',
    re.MULTILINE,
)
# Extension list covers the files people actually name in these sessions.
# It previously omitted .txt, .mod, .lock and .cfg, so `requirements.txt`,
# `go.mod` and `Cargo.lock` — three of the most-mentioned files in any
# Python, Go or Rust session — were never extracted at all.
#
# The stem may itself contain dots: `vite.config.ts`, `tailwind.config.js`,
# `docker-compose.override.yml`, `jest.setup.ts`. A single-segment stem
# matched only the last two parts, so the file was recorded as `config.ts`
# — a name that exists in almost every frontend repo and identifies none.
#
# The extra segments are capped at three, and the cap is load-bearing. An
# unbounded `(?:\.[\w-]+)*` reads the whole of `word.word.word…` from every
# starting position before backtracking to look for an extension: 3.5
# seconds on the 15 KB adversarial payload the scan-cost tests use, on the
# hot path of a proxy that scans whatever a caller sends. Bounded, each
# position costs at most four segments. No real filename has more.
_FILE_COMMON = re.compile(
    # Starts only where a token starts: `\b` also holds after every `-`,
    # so `a-a-a-…` was re-read from each of its 2,000 hyphens.
    r'(?<![\w\-])((?:[\w\-]+(?:\.[\w\-]+){0,3}\.(?:'
    r'py|pyi|js|mjs|cjs|ts|tsx|jsx|vue|svelte|'
    r'go|mod|sum|rs|java|kt|swift|scala|rb|php|cs|cpp|cc|c|h|hpp|ex|exs|'
    r'yaml|yml|json|toml|ini|cfg|conf|env|lock|txt|md|rst|'
    r'sh|bash|zsh|ps1|sql|proto|graphql|tf|tfvars|'
    r'html|css|scss|less|xml|csv|tsv|'
    r'db|sqlite|sqlite3'
    r'))\b)',
    re.IGNORECASE,
)

# Build and tooling files that have no extension at all. Every pattern above
# keys on a dot, so `Dockerfile` and `Makefile` — named in almost every infra
# session — could not be extracted by any of them. Case-sensitive on purpose:
# lowercase "makefile" in prose is usually the noun, not the file.
#
# The directory is part of the name when one is given: `fastlane/Fastfile`
# and `deploy/Dockerfile` are specific files, and a bare `Fastfile` does not
# say which. The Ruby/iOS toolchain's files (Fastfile, Podfile, …) are named
# as often in a mobile session as Dockerfile is in an infra one.
_FILE_EXTENSIONLESS = re.compile(
    r'(?<![\w/.\-])((?:\.?[\w\-]+/){0,6}'
    r'(?:Dockerfile|Makefile|Procfile|Jenkinsfile|Gemfile|Rakefile|Vagrantfile|'
    r'Brewfile|Justfile|Caddyfile|Containerfile|Tiltfile|Earthfile|Pipfile|'
    r'Fastfile|Appfile|Matchfile|Snapfile|Podfile|Cartfile|Dangerfile|'
    r'Berksfile|Guardfile|Capfile|CODEOWNERS|MANIFEST\.in))\b'
)

# JavaScript libraries are named `<name>.js` in prose — "switched from
# moment.js to date-fns" — and read exactly like a file to the patterns
# above. Only the bare name is excluded; `src/lib/moment.js` has a directory
# and is a file whatever it is called.
_LIBRARY_NOT_FILE = frozenset({
    "moment.js", "node.js", "next.js", "nuxt.js", "vue.js", "react.js",
    "angular.js", "ember.js", "backbone.js", "three.js", "chart.js", "d3.js",
    "express.js", "day.js", "p5.js", "paper.js", "anime.js", "video.js",
    "highlight.js", "marked.js", "alpine.js", "solid.js", "knockout.js",
    "handlebars.js", "socket.io.js", "pdf.js", "fabric.js", "leaflet.js",
    "mapbox-gl.js", "hammer.js", "lodash.js", "underscore.js", "jquery.js",
    "require.js", "ext.js", "meteor.js", "sails.js", "koa.js", "hapi.js",
    "nest.js", "gatsby.js", "remix.js", "svelte.js", "preact.js", "deno.js",
    "bun.js", "tone.js", "matter.js", "pixi.js", "babylon.js", "phaser.js",
})


def is_library_name(name: str) -> bool:
    """True if `name` is a JavaScript library written as `<name>.js`, not a file."""
    return "/" not in name and name.lower() in _LIBRARY_NOT_FILE

# ── Decision patterns — 5 passes ─────────────────────────────────────────────

# A capture that stops at the end of the sentence it started in.
#
# The plain `(.{5,80})` ran straight past the full stop, so a message
# stating two decisions — "Decided: X. Decided: Y." — produced ONE match
# spanning both. _clip() then kept the first clause and Y was gone for
# good: finditer does not re-scan inside a span it already consumed, so
# the second keyword was never even looked at. Measured on the corpus,
# that silently dropped one decision in every multi-decision turn.
#
# The lookahead is what makes this safe for labels that legitimately
# contain dots — `moment.js`, `React.lazy`, `Python 3.12`, `go.mod` —
# where the dot is not followed by whitespace or end-of-string.
#
# The 80-character ceiling is a budget, not a place to stop reading, so the
# span ends on a word boundary: without that, a sentence that runs past it
# was cut mid-token and the label shipped as "...confusion matri" or
# "...destroyed memory is queryab". The second branch is the fallback for a
# span with no boundary inside the budget at all (one very long token, like
# a URL), where a hard cut is better than dropping the fact entirely.
_SPAN_CHAR = r'(?:(?![.!?](?=\s|$))[^\n])'
_CLAUSE_SPAN = r'(' + _SPAN_CHAR + r'{5,80}(?!\w)|' + _SPAN_CHAR + r'{5,80})'

# What separates a keyword from its capture: a colon or dash header ("Done:
# X", "Done — X", "Done - X") or whitespace. NOT a bare hyphen: `[\s:\-]+`
# read the first half of a hyphenated compound as the keyword — "a
# FIXED-width lookbehind" became the completed task "width lookbehind",
# "PENDING-task recall is 19%" a to-do, "BUILT-in" a completion.
_SEP = r'(?:\s*[:\u2014\u2013]\s*|\s+-\s+|\s+)'

# Pass 1: explicit verb ("decided:", "going with", "will use")
#
# `picked(?! up)`: "picked up" is a different verb. "Somewhere in there we
# picked up duplicate rows from a non-unique key" reports a bug the session
# acquired, and reading it as a choice filed that bug as the DECISION "up
# duplicate rows … and it's been a pain" — wrong type and a broken label.
#
# The leading `\b` is load-bearing. Without it the verbs matched inside
# other words: "excessive allocations cAUSING GC pressure" recorded the
# decision "GC pressure" (`using`), and "TODO: watCHOS companion app"
# recorded "companion app" (`chose?`).
_DECISION = re.compile(
    r'\b(?:decided?|going with|will use|chose?|switching? to|opted for|settled on|'
    r'picked(?!\s+up\b)|sticking with|selected?|using|went with|we.ll use|let.s use|'
    # A proposal is how a choice is usually put in conversation: "I think we
    # should use sqlc for type-safe database access". Pass 3 caught only the
    # bare name ("Use sqlc") and dropped the purpose that makes it a
    # decision. Questions ("Should we use X?") are excluded by the question
    # guard every decision pass applies.
    r'should (?:use|go with|adopt|switch to|stick with|standardi[sz]e on)|'
    # The base form of "go with" needs a subject or a modal in front of it —
    # "we'll go with Postgres" chooses, "go with the flow" does not. "Opting
    # for" is the progressive of "opted for", and a stated preference is a
    # choice put mildly: "I'd prefer bcrypt", "we'd rather use Redis".
    r"(?:we'?ll|i'?ll|we'?d|i'?d|we will|i will|we should|i would|we would) go with|"
    r'opt(?:ing|s)? for|'
    r"(?:i'?d|we'?d|i|we) (?:prefer|would prefer|rather (?:have|use|go with))|"
    r'leaning toward|recommends?|recommended|'
    # Probed against phrasings the corpus does not use: "standardise on",
    # "let's do", "moving (everything) to", "consolidate on" are how people
    # commit to a choice without saying "decided". "moving to" carries an
    # optional object between verb and destination — "moving everything to
    # Postgres" — which `switching to` never needed.
    r'standardi[sz](?:e|ing) on|let.s (?:do|go)|consensus is|committing to|'
    r'went ahead with|locked in(?: on)?|'
    r'mov(?:e|ing) (?:(?:everything|all|it|over) )?to|consolidat(?:e|ing) on)'
    + _SEP + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# Pass 2: header format ("Decision: X", "Tech choice: X")
_DECISION_HEADER = re.compile(
    # `[ \t]*`, not `\s*`, after the line start: `\s*` also eats newlines,
    # so from every one of 4,000 blank lines it ran to the end of the run
    # and back — 1.3 s on a message of blank lines.
    r'(?:^|\n)[ \t]*(?:decision|tech choice|architecture choice|approach|stack|'
    # The word people type when closing a discussion is rarely "decision".
    r'agreed|final call|final choice|final decision|verdict|conclusion|'
    r'going with|settled|locked in|going forward|consensus)\s*[:\-]\s*'
    + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# "OK, Kafka it is." — the choice is the subject, the verb comes after,
# and nothing else in the sentence marks it. The capitalised-name
# requirement keeps this to proper nouns; "it is" after a common word is
# ordinary prose ("that's how it is").
_DECISION_IT_IS = re.compile(
    r'\b([A-Z][\w.+-]{1,30}(?:\s+[A-Z][\w.+-]{1,30})?)\s+it\s+is[.!,]',
)

# Pass 3: known tech names — expanded (was missing bcrypt, slowapi, etc.)
_DECISION_FOR = re.compile(
    r'\b((?:'
    r'postgres(?:ql)?|redis|sqlite|mongodb|mysql|mariadb|cassandra|dynamodb|supabase|'
    r'fastapi|flask|django|express|nestjs|next\.?js|nuxt|react|vue|svelte|angular|'
    r'docker|kubernetes|k8s|terraform|ansible|'
    r'jwt|oauth|oauth2|openid|bcrypt|argon2|passlib|'
    r'graphql|rest|grpc|websocket|'
    r'celery|rabbitmq|kafka|bull|'
    r'nginx|gunicorn|uvicorn|caddy|traefik|'
    r'pytest|jest|vitest|cypress|playwright|'
    r'sqlalchemy|prisma|drizzle|typeorm|'
    r'pydantic|zod|yup|marshmallow|'
    r'slowapi|authlib|httpx|aiohttp|'
    r'openai|anthropic|gemini|langchain|llamaindex|'
    r'sqlc|dbt|nats|'
    r'pnpm|uv|ruff|kong|airflow|'
    # Probed against current tooling the list had drifted behind. The gate
    # (_tech_mention_is_a_decision) still applies to every name here, so a
    # name only becomes a decision with choosing context around it.
    r'vite|webpack|esbuild|turbopack|bun|deno|tailwind|astro|remix|htmx|'
    r'mongo|firebase|planetscale|neon|clickhouse|duckdb|elasticsearch|opensearch|'
    r'meilisearch|typesense|pgvector|qdrant|weaviate|pinecone|chroma|milvus|'
    r'sentry|datadog|grafana|prometheus|opentelemetry|otel|temporal|helm|argo|'
    r'pulumi|vercel|netlify|cloudflare|lambda|ecs|eks|gke|'
    r'tokio|axum|actix|gin|fiber|spring|rails|laravel|phoenix|hono|trpc|'
    r'zustand|redux|tanstack|mypy|pyright|black|isort|poetry|pytorch|torch|'
    r'tensorflow|jax|transformers|vllm|ollama|litellm|langgraph|crewai|'
    r'styled-components|cypress'
    # `\b` alone is satisfied by the dot in `React.lazy`, so "Decided: code
    # splitting with React.lazy" produced BOTH the real decision and a bare
    # "Use React" beside it. A tech name followed by a dot and more word
    # characters is part of a longer identifier — React.lazy, redis.conf,
    # torch.nn — and naming it is not choosing it. Only a dot is guarded:
    # `postgres-15` and `bert-base-uncased` are how versions are written,
    # and those ARE the choice.
    r')\b(?!\.\w)(?:(?!\s+(?:to|with|for)\s+\w)[^.!?\n,—\-]){0,40})',
    re.IGNORECASE,
)

# Pass 3 matches a bare technology name anywhere in a message, which is
# not the same thing as a decision. "missing email validation in the
# LoginRequest Pydantic model" mentions Pydantic; nobody decided anything.
# Measured, this single pass produced most of the spurious decisions.
#
# A tech mention counts as a decision only with supporting context:
#   - a choosing verb shortly before it ("decided: Redis", "we'll use X"), or
#   - a purpose clause right after it ("Redis FOR refresh tokens"), which
#     is the shape a decision takes when stated without a verb.
_DECISION_CONTEXT_BEFORE = re.compile(
    # `\b` for the same reason as _DECISION: "because" ends in "use", which
    # made every technology named after it look chosen.
    r"\b(?:decided?|decision|going with|will use|we'?ll use|let'?s use|chose|"
    r"choosing|opted for|settled on|picked(?! up)|sticking with|selected|switch(?:ing)? to|"
    r"moved? to|migrat\w+ to|adopt(?:ed|ing)?|use|using|with|"
    r"leaning toward|recommends?|recommended|"
    # "Should talk gRPC to the inventory service" states a choice without a
    # choosing verb. Restricted to verbs of adoption: a bare "should" also
    # heads work items ("should fix the Redis timeout"), which are tasks,
    # not decisions. Questions are excluded upstream by _is_question_context.
    r"should (?:talk|speak|use|run on|target|go with|adopt|be on))\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_DECISION_PURPOSE_AFTER = re.compile(
    r"^\s*(?:for|as|to handle|to store|to manage|instead of|over)\b",
    re.IGNORECASE,
)
# "migrate 40M rows FROM MySQL TO Postgres" — MySQL is what is being left
# behind, not what was chosen. Only the destination is a decision.
_MIGRATION_SOURCE = re.compile(
    r"\b(?:from|away from|off of|out of|replacing|instead of|drop(?:ping)?|"
    r"deprecat\w+|retir\w+)\s*$",
    re.IGNORECASE,
)


# A clause that opens with a work-item header. "Completed: project scaffold
# with Vite" and "Working on: rate limiting using slowapi" mention a
# technology inside a statement about WORK, and the weak choosing cues
# (`with`, `using`) fire on both. The clause is a task; the technology
# named in it is how the task was done, not a decision that was made.
# Measured: this single confusion produced two of the four spurious
# decisions on the corpus.
_TASK_HEADER_AT_CLAUSE_START = re.compile(
    r'^\s*(?:completed|finished|done|implemented|fixed|added|built|shipped|'
    r'created|wrote|updated|deployed|working on|implementing|building|'
    r'currently|in progress|todo|next|pending|wip)\s*[:\-]',
    re.IGNORECASE,
)


# A clause that opens by reporting work, with or without a header colon —
# "Finished up dark mode using CSS custom properties". The `using` names how
# the work was done, not a choice; see _TASK_HEADER_AT_CLAUSE_START for the
# header form of the same guard.
_WORK_CLAUSE_START = re.compile(
    r"^\s*(?:(?:i|we)(?:'ve| have)?\s+)?(?:just\s+|also\s+)?(?:re-?)?"
    r"(?:completed|finished|done|implemented|fixed|added|built|shipped|wired up|"
    r"set up|created|wrote|written|updated|deployed|refactored|migrated|"
    r"wrapped up|got|working on|implementing|building|adding|writing)\b",
    re.IGNORECASE,
)


def _clause_start(content: str, pos: int) -> int:
    for i in range(pos - 1, -1, -1):
        if content[i] in ".!?\n":
            return i + 1
    return 0


def _tech_mention_is_a_decision(content: str, start: int, end: int) -> bool:
    """True if a bare technology name at [start:end] is stated as a choice."""
    before = content[max(0, start - 40):start]
    after = content[end:end + 40]
    if _MIGRATION_SOURCE.search(before):
        return False        # the thing being migrated away from
    clause = content[_clause_start(content, start):start]
    if _TASK_HEADER_AT_CLAUSE_START.match(clause):
        return False        # named inside a work item, not chosen
    return bool(_DECISION_CONTEXT_BEFORE.search(before)
                or _DECISION_PURPOSE_AFTER.match(after))


# Pass 4: passive/implicit — "bcrypt with cost factor 12", "JWT expires in 15m"
_DECISION_PASSIVE = re.compile(
    r'\b((?:'
    r'postgres(?:ql)?|redis|sqlite|mongodb|mysql|'
    r'fastapi|flask|django|express|'
    r'jwt|bcrypt|argon2|oauth|'
    r'docker|kubernetes|nginx|gunicorn|uvicorn|'
    r'celery|rabbitmq|kafka|'
    r'slowapi|pydantic|sqlalchemy|prisma'
    r'))\s+(?:is|are|will be|handles?|with|has|provides?|expires?)',
    re.IGNORECASE,
)

# NEGATION CHECK — required by every decision pass above.
#
# Without it, "We are NOT using Redis" matches Pass 1's verb list
# ("using") and Pass 3's tech-name list ("redis") independently, both
# blind to the preceding "NOT", and both extract "Use Redis" — the
# literal opposite of what was said. That compounds with
# SmartMessageWindow, which replaces older turns with the graph's context
# block: the original sentence is dropped and only the fabricated
# "Decided: Use Redis" remains in what the model sees.
#
# Scoped to the current CLAUSE (back to the nearest sentence boundary),
# not the whole message — an unrelated negation in an earlier, different
# sentence ("The old code didn't have caching. Use Redis for the
# session cache.") must not suppress a later, legitimate decision.
_NEGATION_WORDS = re.compile(
    r"\b(?:not|never|no|avoid(?:ed|ing)?|without|"
    r"don'?t|doesn'?t|didn'?t|won'?t|wouldn'?t|can'?t|couldn'?t|shouldn'?t|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t)\b",
    re.IGNORECASE,
)


def _is_negated_context(content: str, match_start: int) -> bool:
    """True if `match_start` in `content` falls inside a negated clause."""
    clause_start = 0
    for i in range(match_start - 1, -1, -1):
        if content[i] in ".!?\n":
            clause_start = i + 1
            break
    return bool(_NEGATION_WORDS.search(content[clause_start:match_start]))


# "Should we go with Postgres or Redis for this?" matches Pass 1's "go
# with" and Pass 3's "Postgres"/"Redis" — a question weighing options,
# not a decision made. Scoped forward to the end of the current clause:
# the match sits mid-question, so the terminator that disambiguates it
# is ahead, not behind.
def _is_question_context(content: str, match_start: int) -> bool:
    """True if the clause containing `match_start` ends in a question mark."""
    clause_end = len(content)
    for i in range(match_start, len(content)):
        if content[i] in ".!?\n":
            clause_end = i
            break
    return content[clause_end:clause_end + 1] == "?"


# Pass 5: config decisions — "expires in 15 minutes", "cost factor 12"
_DECISION_CONFIG = re.compile(
    r'\b(?:expire[sd]? in|cost factor|timeout of|limit of|max(?:imum)? of|'
    r'requests? per|connections? per|workers?)\s+(\d+[^.!?\n]{0,40})',
    re.IGNORECASE,
)

# ── Supersession ──────────────────────────────────────────────────────────────
#
# Two phrasings state the same fact with the operands in OPPOSITE orders,
# and reading both with one pattern produced a false decision rather than
# merely missing one:
#
#   forward — "switched FROM moment.js TO date-fns"   (old, then new)
#   reverse — "date-fns INSTEAD OF moment.js"         (new, then old)
#
# A single pattern listing "instead of" alongside "switched from" and then
# requiring a trailing "to/with/for" read "Next.js instead of React for
# better SEO" as old="React", new="better SEO" — recording a decision to
# use "better SEO" and superseding the real one with it. The same pattern
# could not match "date-fns instead of moment.js" at all, because nothing
# follows the old side. Hence three patterns and one operand cleaner.
#
# `/` and `@` belong in an operand: real dependency names carry them
# (`cenkalti/backoff`, `@scope/pkg`, `psf/black`).
_OPERAND = r'[\w][\w\s\./@\-]{2,40}'

# "switched from X to Y", "moved from X to Y", "migrating away from X to Y"
_SUPERSEDED = re.compile(
    r'(?:switch(?:ed|ing)?|mov(?:ed|ing)|migrat(?:ed|ing))\s+'
    r'(?:away\s+)?from\s+(' + _OPERAND + r')\s+(?:to|over\s+to|onto)\s+(' + _OPERAND + r')',
    re.IGNORECASE,
)

# "replaced X with Y", "replacing X by Y"
_SUPERSEDED_REPLACE = re.compile(
    r'replac(?:ed|ing|es)?\s+(' + _OPERAND + r')\s+(?:with|by)\s+(' + _OPERAND + r')',
    re.IGNORECASE,
)

# "Y instead of X" is deliberately NOT a supersession.
#
# It states ONE decision and the alternative it was chosen over, in one
# sentence, at one moment — "quarantine by rename instead of unlink" is a
# single choice, and the labelled corpus records it as one decision with
# that whole phrase as its label. Reading it as a change over time
# produced three decisions per sentence (the phrase plus a node for each
# side), a chain that ran in a circle, and — because the old pattern
# listed "instead of" alongside "switched from" and then demanded a
# trailing "to/with/for" — labels like "better SEO" recorded as the
# technology chosen. "instead of", "rather than" and "in place of" appear
# constantly in ordinary technical prose ("quoting only the macro rather
# than the file"), so matching them here cost 28 points of decision
# precision on the corpus for no transition anyone could trust.
#
# A transition needs evidence that the state actually CHANGED: something
# was in use, and then it was not. That is what the two patterns above say
# and what this one cannot.

# An operand runs on into the clause that explains it — "React for better
# SEO", "moment.js because it ships every locale". Everything from the
# first connective on is rationale, not the name of the thing chosen.
_OPERAND_TAIL = re.compile(
    r'(?<!\s)\s+(?:for|because|since|so|as|which|that|and|but|due|given|after|'
    r'when|while|to)\b.*$',
    re.IGNORECASE | re.DOTALL,
)


# A sentence boundary inside an operand: the same lookahead the clause
# patterns use, so `moment.js` and `3.12` keep their dots while
# "pglogical. Created infra/dms_task.tf" stops at the full stop.
_OPERAND_SENTENCE = re.compile(r"[.!?;](?=\s|$)|\s[\u2014\u2013]\s")


# "Replaced THE PATTERN with ..." — an article followed by one ordinary
# word is a common noun, not the name of a thing that can be adopted or
# dropped. "Replaced the hand-rolled retry loop with cenkalti/backoff" is
# the same shape with three words and IS a real change, so the test is
# the word count, not the article alone.
_COMMON_NOUN_OPERAND = re.compile(r"^(?:a|an|the)\s+[\w\-]+$", re.IGNORECASE)


def _supersede_operand(text: str) -> str:
    """One side of a supersession, trimmed to the thing being named.

    Returns "" when the span is a common noun phrase rather than a name.
    """
    s = " ".join((text or "").split())
    cut = _OPERAND_SENTENCE.search(s)
    if cut:
        s = s[:cut.start()]
    s = _OPERAND_TAIL.sub("", s).strip(" ,;:.\u2014-")
    if _COMMON_NOUN_OPERAND.match(s):
        return ""
    s = re.sub(r"^(?:a|an|the)\s+", "", s, flags=re.IGNORECASE)
    return s.strip(" ,;:.\u2014-")


_COMMON_PHRASE = re.compile(
    r"^\s*(?:a|an|the|this|that|our|my|its|their|each|every)\s+[a-z][a-z \-]*$")


def _is_common_phrase(span: str) -> bool:
    """An article-led, all-lowercase phrase with no identifier in it:
    no capital, digit, dot, slash or `@` — "the longer label", not "the
    Redis cache" or "a v2 client"."""
    head = _OPERAND_TAIL.sub("", " ".join((span or "").split()))
    cut = _OPERAND_SENTENCE.search(head)
    head = head[:cut.start()] if cut else head
    return bool(_COMMON_PHRASE.match(head.strip(" ,;:.\u2014-")))


def find_supersessions(content: str) -> list[tuple[str, str, int, int]]:
    """Every "X was replaced by Y" the text states, as
    (old, new, match_start, match_end).

    Deduplicated on (old, new) so two phrasings of one change in the same
    message do not produce two transitions.
    """
    found: list[tuple[str, str, int, int]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in (_SUPERSEDED, _SUPERSEDED_REPLACE):
        for m in pattern.finditer(content or ""):
            old = _supersede_operand(m.group(1))
            new = _supersede_operand(m.group(2))
            if len(old) < 2 or len(new) < 2 or old.lower() == new.lower():
                continue
            # Neither side names anything: "replacing an error fragment
            # with the longer label" describes an edit, not a change of
            # technology or approach, and recording it made the resume
            # report "Changes: 'Use error fragment' -> 'Use longer label'".
            # One named side is enough to keep it ("switching from
            # cenkalti/backoff to a hand-rolled retry loop").
            if _is_common_phrase(m.group(1)) and _is_common_phrase(m.group(2)):
                continue
            key = (old.lower(), new.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append((old, new, m.start(), m.end()))
    return found


# ── Evidence extraction patterns ──────────────────────────────────────────────

# Numeric metrics with context — "latency 340ms", "score was 61"
_EVIDENCE_NUMBER = re.compile(
    # `(?<![\d.])`: a match may only start where a number starts. Without
    # it, a 4,000-digit run was re-scanned from every digit (1.3 s).
    r'(?<![\d.])(\d+(?:\.\d+)?\s*(?:ms|s|seconds?|minutes?|hours?|'
    r'%|percent|'
    r'mb|gb|tb|kb|'
    r'rpm|rps|req/s|'
    r'\$/(?:month|mo|year|yr)|'
    r'x\s+(?:faster|slower|larger|smaller)'
    r')[^.!?\n]{0,40})',
    re.IGNORECASE,
)

# Bare score/rating — "score was 61", "rating of 4.5"
_EVIDENCE_SCORE = re.compile(
    r'(?:score|rating|result|grade)\s+(?:was|is|of|:)\s*(\d+(?:\.\d+)?(?:/\d+)?[^.!?\n]{0,30})',
    re.IGNORECASE,
)

# Dollar amounts — "$50/month", "costs $200"
_EVIDENCE_COST = re.compile(
    r'(\$\d+(?:\.\d+)?(?:/(?:month|mo|year|yr|day))?[^.!?\n]{0,20})',
)


# Direct quotes from user — "user said X", "you mentioned X"
_EVIDENCE_QUOTE = re.compile(
    r'(?:you said|user said|you mentioned|you told me|per your requirement|'
    r'as you noted|because you|since you want|you need)'
    r'\s+["\']?(.{10,100}?)["\']?(?:\.|,|$)',
    re.IGNORECASE,
)

# Standards/recommendations — "OWASP recommends", "industry standard"
_EVIDENCE_STANDARD = re.compile(
    r'(?:OWASP|RFC|ISO|IEEE|NIST|W3C|MDN|Google|Lighthouse|'
    r'industry standard|best practice|recommended|specification)'
    r'[^.!?\n]{0,60}',
    re.IGNORECASE,
)

# Trailing fragments that mean a capture was cut where a clause
# continued, so the label ends on a dangling connective.
_DANGLING_TAIL = re.compile(
    r"(?<![\s,;:—-])[\s,;:—-]+(?:and|or|but|with|for|to|in|on|at|by|from|the|a|an|of|"
    r"that|which|when|while|so|then|using|via)\s*$",
    re.IGNORECASE,
)
# A clause ends at terminal punctuation FOLLOWED BY whitespace or the
# end of the span. The lookahead is load-bearing: a bare `[.!?;]` also
# matches the dot in `evaluate.py`, `requirements.txt` and `Python 3.12`,
# which silently truncated those labels to "evaluate", "requirements"
# and "Python 3".
# Minimum length a clipped label must reach before a clause boundary is
# allowed to end it. Chosen by sweeping it against the eval corpus
# (`python -m benchmarks.eval --sweep`, and the table in the CHANGELOG)
# rather than by feel:
#
#   min_chars   macro F1   truncated   multi-sentence
#           8        74%          6%              3%
#          22        75%          8%              5%     <- chosen
#          34        79%         17%             14%
#          48        79%         24%             20%
#
# F1 keeps climbing past 22, but only by letting labels sprawl again —
# which is the defect this clipping was added to fix. 22 keeps
# essentially all of the readability win (baseline was 23% truncated,
# 27% multi-sentence) while taking most of the accuracy gain. Anyone
# preferring a different point on that curve can re-run the sweep; the
# point is that the number is now defensible from a table.
_MIN_CLAUSE_CHARS = 22
_CLAUSE_END = re.compile(r"[.!?;](?=\s|$)|\s[—–]\s|\n")


_ONLY_PATHS = re.compile(
    r"^(?:the\s+)?[\w./\-]+\.[a-zA-Z]{1,6}"
    r"(?:\s*(?:,|and)\s*[\w./\-]+\.[a-zA-Z]{1,6})*$",
    re.IGNORECASE,
)

# "Fixed the backfill timeout" names a real failure, just in the past
# tense. Strip the repair verb and keep the failure — an error that was
# resolved is still part of the session's history, and dropping it loses
# the reason the code looks the way it does.
_FIX_PREFIX = re.compile(
    r"^(?:fix(?:ed|es)?|resolv(?:ed|es)?|patch(?:ed)?|repair(?:ed)?|"
    r"correct(?:ed)?|address(?:ed)?|clos(?:ed)?|eliminat(?:ed)?)\s+"
    r"(?:the\s+|a\s+|an\s+)?",
    re.IGNORECASE,
)

# "Let's work on a memory leak", "still chasing down the flaky upload test":
# the verb says what the session is DOING about the failure, and it is not
# part of the failure's name. Kept apart from _FIX_PREFIX on purpose — that
# prefix also marks the error resolved, and investigating is not fixing.
_INVESTIGATION_PREFIX = re.compile(
    r"^(?:(?:start(?:ed|ing)?|keep|kept)\s+)?(?:work(?:ing|ed)?\s+on|"
    r"look(?:ing|ed)?\s+(?:at|into)|investigat\w+|debugg?\w*|dig(?:ging)?\s+into|"
    r"chas(?:e|ed|ing)\s+down|track(?:ed|ing)?\s+down|on)\s+",
    re.IGNORECASE,
)

# A symptom word can also name part of the SOLUTION rather than the
# problem: "Fixed by adding a 5 second context timeout" is a timeout
# being introduced on purpose. An additive verb immediately before the
# match is the tell.
_SOLUTION_VERB = re.compile(
    r"\b(?:add(?:ing|ed)?|introduc(?:ing|ed)?|set(?:ting)?|configur(?:ing|ed)?|"
    r"enabl(?:ing|ed)?|impos(?:ing|ed)?|appl(?:ying|ied))\s+"
    r"(?:an?\s+|the\s+)?(?:[\w.-]+\s+){0,3}$",
    re.IGNORECASE,
)


# A captured task that opens on an article or a preposition is the tail of
# a sentence whose verb the pattern consumed: "Removed |the dependency from
# go.mod|", "Fixed |by adding a 5 second timeout|". The label lands in the
# resume block, which is the thing the product exists to produce, and
# "Done: the dependency from go.mod" does not say what happened to it.
# "to" is deliberately absent: it opens a purpose clause, not an object.
# "Removed |to prove the bake worked|" reads worse with the verb restored,
# not better, because the verb already had its object elsewhere.
_FRAGMENT_OPENER = re.compile(
    r"^(?:the|a|an|by|from|with|in|on|at|into|onto|via|using|after|"
    r"before|during|over|under)\b",
    re.IGNORECASE,
)


def restore_verb(verb: str, label: str) -> str:
    """Put the matched verb back in front of a fragment label.

    Only when the label reads as a fragment: "virtual environment setup"
    is already a statement and gains nothing from "Completed" in front of
    it, while "the dependency from go.mod" is not a statement at all.
    """
    if not label or not verb or not _FRAGMENT_OPENER.match(label):
        return label
    verb = verb.strip()
    if not verb:
        return label
    return verb[0].upper() + verb[1:].lower() + " " + label


def _is_only_paths(text: str) -> bool:
    """True if `text` is nothing but filenames.

    "Updated src/App.tsx and src/routes.tsx" is a file list; recording it
    as a completed task duplicates the file nodes and says nothing about
    what was done.
    """
    return bool(_ONLY_PATHS.match((text or "").strip()))


_SENTENCE_BOUNDARY = re.compile(r'[.!?](?=\s)')


def _drop_leading_sentence(text: str) -> str:
    """Drop everything up to the last sentence boundary inside `text`.

    The subject windows in the error patterns walk backwards a few words to
    find what failed, and a word may legitimately end in a dot (`moment.js`,
    `go.mod`), so they cannot simply refuse to cross one. When the dot really
    was a full stop the window steps into the previous sentence and the label
    comes out as "per worker. Also flock is unreliable" — two half-thoughts,
    the first of them irrelevant.

    Solving this inside the regex needs an atomic group, which Python 3.10
    does not have, and the non-atomic equivalent backtracks catastrophically
    (see `_TOKEN`). Trimming after the match is O(n) and does the same job.
    """
    s = (text or "").strip()
    last = None
    for m in _SENTENCE_BOUNDARY.finditer(s):
        last = m
    return s[last.end():].strip() if last else s


def _sentence_index(text: str, position: int) -> int:
    """Which sentence of `text` contains `position` (0-based)."""
    return sum(1 for m in _SENTENCE_BOUNDARY.finditer(text) if m.end() <= position)


def _content_words(text: str) -> int:
    """How much a label actually says — distinct words over two characters."""
    return len({w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2})


def _clip(text: str, max_chars: int = 90) -> str:
    """Trim a captured span to one readable clause.

    The extraction patterns capture a fixed width of whatever follows a
    keyword — `(.{5,80})` — which has no idea where the thought ends. In
    practice that produced labels cut mid-word, labels running across
    three sentences ("...updated api/models.py. Login endpoint working
    now."), and several overlapping labels for one fact. Measured on the
    eval corpus before this existed: 27% of labels truncated mid-word,
    26% spanning more than one sentence.

    Cut at the first clause terminator, fall back to the last whole word
    inside the budget, then drop a dangling connective so the label reads
    as a statement rather than the first half of one.
    """
    s = " ".join((text or "").split())
    if not s:
        return ""

    # Cut at the first clause boundary that leaves a label with enough
    # substance to identify what it refers to. Taking the FIRST boundary
    # unconditionally reduced "Fixed: 422 error - missing email
    # validation in LoginRequest" to "422 error", which is short, tidy,
    # and no longer says which validation broke. Measured on the eval
    # corpus, cutting at >=8 chars cost 5 points of completed-task F1
    # and 6 of decision F1 relative to this bound.
    for m in _CLAUSE_END.finditer(s):
        if m.start() >= _MIN_CLAUSE_CHARS:
            s = s[:m.start()]
            break

    if len(s) > max_chars:
        cut = s[:max_chars]
        space = cut.rfind(" ")
        s = cut[:space] if space >= 12 else cut

    s = _DANGLING_TAIL.sub("", s).strip(" ,;:—-")

    # A parenthetical the cut landed inside — "Redis for refresh token
    # storage (not DB" — reads as a typo on every surface that shows the
    # label. Drop the open parenthetical when what precedes it still
    # identifies the fact; close it otherwise.
    if s.count("(") > s.count(")"):
        head = s[:s.rfind("(")].rstrip(" ,;:—-")
        s = head if len(head) >= _MIN_CLAUSE_CHARS // 2 else s + ")"
    return s


_TASK_DONE = re.compile(
    # Past participles are listed explicitly. `wrote?` only ever matched
    # "wrot"/"wrote" — never "written", which is how most completion is
    # actually narrated ("I've written the connection pool").
    #
    # Every verb here is past tense, with no optional final `d`. Writing them
    # as `migrated?`/`removed?`/`fixed?` also matched the PRESENT tense, so
    # "We need to migrate 40M rows to Postgres" — a statement of intent in the
    # opening turn — was recorded as finished work. Present tense is the one
    # reliable signal that something has not happened yet, and spending it to
    # save four characters cost precision on every session that opens by
    # describing the goal.
    #
    # Anchored at a word start, with `re-` allowed: "UNfinished work" and
    # "UNdone" are the opposite of completion and matched as it, while
    # "rebuilt the index" and "redeployed" are completions.
    #
    # The particle after a phrasal verb belongs to the verb: "finished UP
    # dark mode", "cleaned UP the fixtures" were recorded as "up dark mode".
    r'\b(?:re-?)?(?:completed|finished|done|implemented|fixed|added|built|shipped|'
    r'wired up|set up|created|wrote|written|updated|deployed|resolved|'
    r'merged|refactored|cleaned|migrated|restructured|removed|'
    r'switched|replaced)(?:\s+(?:up|out|off)\b)?'
    + _SEP + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# ── Guards against a completion verb that is not a completion ───────────────
#
# _TASK_DONE matches a past-tense verb followed by a span. Two ways that
# reads work into a sentence that describes none.

# The verb is an adjective on one of the ontology's own nouns. "vanished
# from completed tasks and appeared as a spurious file" recorded "tasks and
# appeared as a spurious file" as finished work; so did "completed tasks F1
# dropped from 77 to 75". Talking ABOUT the categories is exactly what a
# session reviewing its own extraction does, and it was the largest single
# source of spurious completed tasks on the real-transcript corpus.
# A capture that opens on a preposition or an auxiliary is the tail of a
# clause whose verb was consumed, not a task: "I'll start |by checking what
# is available|", "the numbers I'd written |were from round 2|", "the
# pattern I just wrote |(a leading \\b …)|".
_NOT_A_TASK_START = re.compile(
    r"^\s*(?:[(\[{]|(?:by|with|to|from|for|on|in|at|of|as|than|about|"
    r"is|are|was|were|be|been|being|has|have|had|will|would|can|could|should|"
    r"may|might|must|do|does|did)\b)",
    re.IGNORECASE,
)

# The same category nouns followed by a figure or a colon: a line of a
# status report ("tasks: 2% -> 32%", "task recall 19%"), not work. Narrower
# than _CATEGORY_NOUN on purpose — "error boundaries around each route" is
# a to-do that happens to start with one of the words.
_CATEGORY_STAT = re.compile(
    r'^(?:tasks?|decisions?|errors?|files?|items?|goals?|endpoints?|schemas?|'
    r'nodes?|labels?)\b\s*(?:[:|]|\d|recall\b|precision\b|f1\b)',
    re.IGNORECASE,
)

_CATEGORY_NOUN = re.compile(
    r'^(?:tasks?|decisions?|errors?|files?|items?|goals?|endpoints?|'
    r'schemas?|nodes?|labels?)\b',
    re.IGNORECASE,
)

# The clause opens by saying the work is NOT done, and the completion verb
# belongs to a subordinate phrase inside it: "Working on a CI step that runs
# the image with the network removed to prove the bake worked" is one piece
# of outstanding work, not a finished task called "to prove the bake
# worked". This is the mirror of _COMPLETION_LEAD, which stops finished work
# being read as a to-do.
_WIP_LEAD = re.compile(
    r'\b(?:working on|currently|in progress|about to|going to|planning to|'
    r'need(?:s|ed)? to|still|next up|todo|to do|blocked on)\b'
    r'[^.!?\n]{0,90}$',
    re.IGNORECASE,
)

# A completion verb earlier in the same clause. See the WIP/TODO guards.
_COMPLETION_LEAD = re.compile(
    r'\b(?:completed|finished|done|fixed|resolved|solved|implemented|shipped|'
    r'landed|merged)\b\s*(?:[:\-—]|by|via|with)?\s*[^.!?\n]{0,30}$',
    re.IGNORECASE,
)

# A capture that opens with a conjunction is the tail of someone else's
# sentence, not a statement: "Rows this instance created but never persisted
# are untouched" yields "but never persisted are untouched".
_LEADING_CONNECTIVE = re.compile(
    r'^(?:but|and|or|so|because|which|while|that|then|though|although|however)\b',
    re.IGNORECASE,
)

# Passive completion — "rate limiting is implemented", "tests are passing"
#
# The subject must END on a non-space. As `[^\n]{5,50}?\s+`, a run of
# whitespace could be split between the subject and the `\s+` in every way
# at every starting position: a message holding 5,000 spaces took 16.5
# seconds in this one pattern — any caller could stall the proxy with
# padding. Ending the subject on `\S` leaves exactly one split.
_TASK_DONE_PASSIVE = re.compile(
    r'((?:(?![.!?](?=\s|$))[^\n]){4,49}?\S)\s+(?:is|are)\s+'
    r'(?:working|ready|done|complete|live|passing|'
    r'implemented|deployed|fixed|resolved|working now|up and running)',
    re.IGNORECASE,
)

# Anchored at a word start, `re-` allowed, as _TASK_DONE is: without the
# anchor "reWRITING half of conftest.py along the way" matched `writing`
# and recorded "half of conftest.py along the way" as work in progress.
_TASK_WIP = re.compile(
    r'\b(?:re-?)?(?:working on|implementing|building|currently\s+\w+ing|adding|integrating|'
    r'setting up|configuring|writing|debugging|investigating)'
    + _SEP + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# A gerund after an aspectual verb is finished, not ongoing: "ended up
# rewriting half the module", "spent the afternoon debugging the parser".
_PAST_ASPECT_LEAD = re.compile(
    r"\b(?:ended up|wound up|finished|stopped|quit|gave up|spent\s+[\w ]{0,30}?)\s+$",
    re.IGNORECASE,
)

# A completion verb followed straight by a preposition is intransitive —
# "the retry fix LANDED IN the last commit", "a breaking change SHIPPED IN a
# minor version" — and what follows is where or when, not what was done.
# Read as a transitive verb it produced labels like "Landed in in the last
# commit". `to` is not here: "Deployed to staging" is a whole report.
_INTRANSITIVE_TAIL = re.compile(
    r"^(?:in|on|at|into|with|from|off|over|during|after|before|last|yesterday|"
    r"earlier|today|this morning|this afternoon|already|successfully|fine|cleanly)\b",
    re.IGNORECASE,
)

# `missing` needs a guard the other openers do not: "was missing dependency
# in useEffect" is a past-tense diagnosis of something already fixed, and
# recording it as outstanding work tells the next session to go redo it.
# Only an unqualified "missing X" is a TODO.
_TASK_TODO = re.compile(
    r'(?:will add|will implement|next:|todo:|will do|need to add|planning to|'
    r'should add|still need|not yet|'
    # "a crash from a missing Info.plist key", "Bug: a missing statistical
    # test" — after an article or a preposition, `missing` describes the
    # CAUSE of a defect, and reading it as a to-do sent the next session off
    # to add a statistical test as if it were planned work.
    r'(?<!was )(?<!were )(?<!is )(?<!are )(?<!\bfrom )(?<!\ba )(?<!\ban )'
    r'(?<!\bthe )(?<!\bby )(?<!\bof )(?<!\bwith )(?<!\bto )missing|'
    # `pending` only as a header ("Pending: X"). As a plain word it is an
    # adjective as often as not — "pending recall dropped 3 items",
    # "pending requests" — and each of those became a to-do.
    r'pending(?=\s*:)|next step)'
    + _SEP + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# ── Conversational forms ─────────────────────────────────────────────────────
#
# Everything above keys on a marker: a header ("Done:", "TODO:"), a verb
# that opens the clause ("implemented", "going with"). That is how a status
# report is written, and it is not how most of a working session is. The
# same facts arrive as a subject followed by what happened to it — "the
# retry wrapper is in and working", "rate limiting is next on the list",
# "Postgres felt like the right call" — or as the object of a phrasal verb
# ("wrapped up X", "got X working", "haven't started X"). Measured on the
# 100-session external benchmark, these shapes were most of what the
# heuristic pass missed in every category but files.
#
# Every capture here is either the object of a verb (read forwards to the
# end of the clause) or the subject of a predicate (read backwards with
# clause_subject() below). Subject captures are bounded to one clause, so
# a sentence can never contribute more than its own subject.

# A clause starts after a sentence boundary, a comma-like break, or a
# conjunction that opens a new clause ("…, but X won out").
_CLAUSE_BREAK = re.compile(
    r"[.!?](?=\s)|[,;:\n(]|\s[—–-]\s|\b(?:but|while|because|though|although|"
    r"whereas|unless|and then|so)\b",
    re.IGNORECASE,
)

# Discourse openers and stance frames that sit in front of a subject but are
# not part of it: "In the end Postgres was simpler", "I think Redis makes
# more sense". Removed repeatedly, since they stack ("so honestly maybe").
_SUBJECT_LEAD = re.compile(
    r"^(?:and|but|so|then|also|now|well|ok(?:ay)?|yeah|yes|honestly|frankly|"
    r"overall|ultimately|eventually|finally|in the end|at this point|for now|"
    r"maybe|perhaps|probably|possibly|i guess|it seems|"
    r"(?:i|we|the team|everyone|they) (?:think|thinks|thought|feel|feels|felt|"
    r"agree|agreed|decided|reckon|believe|figured)(?: that)?"
    r")\b[,\s]*",
    re.IGNORECASE,
)


def clause_subject(content: str, end: int, max_chars: int = 110) -> str:
    """The subject of the clause that ends at `end` — the text between the
    nearest clause break before it and `end`, minus discourse openers.

    Bounded backwards to `max_chars` and to one clause, so it cannot pull a
    previous sentence (or the first half of a compound one) into the label.
    """
    start = max(0, end - max_chars)
    seg = content[start:end]
    last = None
    for m in _CLAUSE_BREAK.finditer(seg):
        last = m
    if last is not None:
        seg = seg[last.end():]
    elif start > 0:
        # The window cut into the middle of a long clause; its first word
        # may be partial and the subject is not reliably recoverable.
        seg = seg.split(" ", 1)[1] if " " in seg else ""
    seg = seg.strip(" \t\"'`*_")
    prev = None
    while prev != seg:
        prev = seg
        seg = _SUBJECT_LEAD.sub("", seg, count=1).strip()
    return seg


# ── Completed ────────────────────────────────────────────────────────────────

# Phrasal completion verbs. "Wrapped up the OpenAPI schema" and "knocked out
# the pagination endpoints" are how completion is narrated in conversation;
# none of them is in _TASK_DONE's list of past participles.
_TASK_DONE_PHRASAL = re.compile(
    r'\b(?:wrapped up|finished up|knocked out|tidied up|sorted out|squared away|'
    r'landed|nailed down|closed out|crossed off|checked off|ticked off)'
    + _SEP + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# "got X working", "got the migration merged": completion stated as the
# RESULT state of the object. The state list is what "finished" looks like
# for software. The object is lazy and bounded to one clause.
_TASK_DONE_GOT = re.compile(
    r"\bgot\s+((?:(?![.!?](?=\s|$))[^\n,;]){4,80}?)\s+"
    r"(?:working|done|handled|sorted(?: out)?|fixed|merged|passing|running|"
    r"deployed|landed|finished|shipped|wired up|set up|in place|squared away|"
    r"green|over the line|across the line)\b",
    re.IGNORECASE,
)

# The subject form: "<X> is in and working", "<X> is now in place",
# "<X> has been merged". The predicate list is the state of finished work;
# the subject is read backwards with clause_subject(). `in` alone is only a
# completion when nothing locational follows it ("the fix is in." / "is in
# and working") — "most of it is in tailwind.config.js" is a location.
_TASK_DONE_STATE = re.compile(
    r"(?:\s(?:is|are|was|were|has been|have been|got)|'s)\s+(?:now\s+|finally\s+|all\s+)?"
    r"(?:done|finished|complete|completed|in place|sorted|handled|merged|landed|"
    r"shipped|live|taken care of|out of the way|behind us|wired up|set up|"
    r"in and working|in and tested|in(?=\s*(?:[.,;!]|$|now\b)))",
    re.IGNORECASE | re.MULTILINE,
)

# Checklist markers. A checked box or a check mark is the most explicit
# completion marker there is, and it is how an assistant renders a task
# list (Claude Code's own todo list prints a ballot box, checked or not).
_TASK_DONE_CHECK = re.compile(
    r'(?:^|\n)[ \t]*(?:[-*\u2022][ \t]+)?(?:\[[xX]\]|\u2612|\u2705|\u2714\ufe0f?|\u2713)[ \t]*'
    r'((?:(?![.!?](?=\s|$))[^\n]){4,100})',
    re.MULTILINE,
)
_TASK_TODO_CHECK = re.compile(
    r'(?:^|\n)[ \t]*(?:[-*\u2022][ \t]+)?(?:\[ \]|\u2610|\u2b1c)[ \t]*'
    r'((?:(?![.!?](?=\s|$))[^\n]){4,100})',
    re.MULTILINE,
)

# ── Pending ──────────────────────────────────────────────────────────────────

# Section headers for outstanding work, at the start of a line or sentence.
# _TASK_TODO carries "todo:"/"next:" already; these are the rest of the
# vocabulary people use for the same list.
_TASK_TODO_HEADER = re.compile(
    r'(?:^|\n|(?<=[.!?]))[ \t]*(?:[-*\u2022][ \t]+)?(?:\*\*|__)?'
    r'(?:to[ -]?do|next steps?|next up|up next|remaining|still to do|left to do|'
    r'open items?|follow[- ]?ups?|backlog|outstanding|not (?:yet )?started|'
    r'action items?|still needed|deferred)'
    r'(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*'
    r'((?:(?![.!?](?=\s|$))[^\n]){4,100})',
    re.IGNORECASE | re.MULTILINE,
)

# The same list fronted as a clause: "Up next is the GDPR export", "Next on
# the list is rotating the keys" — the inverse of the subject form in
# _TASK_TODO_STATE below.
_TASK_TODO_FRONTED = re.compile(
    r"(?:^|\n|(?<=[.!?]))[ \t]*(?:up next|next up|next on (?:the|my|our) list)"
    r"\s+(?:is|are)\s+" + _CLAUSE_SPAN,
    re.IGNORECASE | re.MULTILINE,
)

# Negated completion: the work is named as not yet done. "Haven't started
# the audit log yet", "didn't get to the backfill".
_TASK_TODO_NOT_DONE = re.compile(
    r"\b(?:haven'?t|have not|hasn'?t|has not|didn'?t|did not|never)\s+(?:yet\s+)?"
    r"(?:started(?:\s+on)?|begun(?:\s+on)?|touched|gotten (?:to|around to)|got (?:to|around to)|"
    r"done|tackled|looked at|picked up|written|added|implemented|finished|"
    r"addressed|dealt with)\s+" + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# Deferral: "we'll circle back to X", "I keep meaning to get to X",
# "we still have to do X". The work is named as put off, which stays true
# until it is done — the extractor reads these over the whole session.
_TASK_TODO_DEFER = re.compile(
    r"\b(?:(?:circle|come|get) back to|revisit(?:ing)?|"
    r"(?:keep|kept|been|am|was) meaning to(?: get to)?|"
    r"(?:we|i) still (?:need|have) to(?: do| get to)?)\s+" + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# Intent: "I'll tackle X tomorrow", "next I'll pick up X". Unlike deferral,
# this is also how an assistant narrates the step it is about to take ("I'll
# add logging here"), which is done a turn later and rarely reported as
# done — so the extractor reads these from the recent window only, like
# WIP. The verb list is work verbs only: "I'll explain the reasoning" is
# not a task.
_TASK_TODO_INTENT = re.compile(
    r"\b(?:(?:i'?ll|we'?ll|i will|we will|i'?m going to|we'?re going to|"
    r"(?:i|we) (?:plan|intend|need|have) to|next,? (?:i'?ll|we'?ll|we|i)|"
    r"(?:we|i) should)\s+(?:also\s+|still\s+|then\s+)?"
    r"(?:tackle|pick up|handle|look (?:at|into)|get (?:to|around to)|work on|"
    r"implement|write|build|set up|wire up|finish|do|start(?: on)?|address|"
    r"migrate|split|refactor|clean up|backfill|document|add|move|port|"
    r"replace|introduce|investigate|profile|benchmark|audit|review))"
    r"\s+" + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# The subject form: "<X> is next on the list", "<X> is still outstanding",
# "<X> can wait until after the demo", "<X> hasn't been done yet".
_TASK_TODO_STATE = re.compile(
    r"\s(?:is|are)\s+(?:still\s+|also\s+)?(?:next(?: on the list| up)?(?=\s*[.,;!]|\s+on\b|\s+up\b|\s*$)|"
    r"up next|outstanding|still open|pending|not (?:yet )?(?:started|done)|"
    r"on the (?:list|backlog|to-?do list|roadmap|radar|plate)|"
    r"(?:the )?next (?:thing|step|task|item)|left to do|still to do)"
    r"|\s(?:hasn'?t|has not|haven'?t|have not) been (?:started|done|touched|"
    r"implemented|written|added|addressed|picked up)"
    r"|\s(?:can|will|should) wait\b"
    r"|\s(?:still )?needs? (?:doing|to be done|to happen|work)\b"
    r"|\s(?:has|have) to happen\b"
    r"|\sremains? (?:open|to be done|outstanding|pending)\b",
    re.IGNORECASE,
)

# ── Decisions ────────────────────────────────────────────────────────────────

# A user's imperative is a decision when it opens a sentence: "Use Postgres
# for the orders store.", "Build it with Alembic for migrations." — the user
# is directing the work. The same sentence in an ASSISTANT turn is usually
# an instruction to the reader ("Use `npm run dev` to start the server"),
# so the extractor applies this to user turns only.
_DECISION_IMPERATIVE = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:please\s+|ok(?:ay)?,?\s+|yes,?\s+|then\s+)?"
    r"(?:use|go with|stick with|switch to|adopt|pick(?!\s+up\b)|choose|prefer|"
    r"build (?:it|this|them|that) (?:with|on|using)|do (?:it|this|that) (?:with|using))"
    r"\s+" + _CLAUSE_SPAN,
    re.IGNORECASE | re.MULTILINE,
)

# Evaluative choice: the option is the subject and the choosing is done by
# the predicate — "<X> felt like the right call", "<X> won out in the end",
# "<X> made the most sense here". The adjective and noun lists are the
# ordinary vocabulary of preferring one option over another.
_DECISION_EVALUATIVE = re.compile(
    r"\s(?:(?:felt|feels|seemed|seems|looked|looks|sounded|sounds)\s+like|"
    r"is|was|would be|will be|ended up being|turned out to be|remains)\s+"
    r"(?:the|a|our)\s+(?:right|best|better|obvious|safer|safest|sensible|"
    r"pragmatic|simplest|cleanest|natural|correct|clear|winning|preferred)\s+"
    r"(?:call|choice|option|bet|fit|approach|move|pick|winner|way forward|way to go|answer)\b"
    r"|\s(?:won out|wins out|made the most sense|makes the most sense|makes more sense|"
    r"made more sense|is the way forward|is the way to go|is worth (?:trying|a shot|a try|considering)|"
    r"would (?:probably |likely )?(?:fit|work) better|fits better|works better)\b"
    r"|\s(?:was|is|would be)\s+(?:simpler|cleaner|cheaper|safer|easier|faster|better)"
    r"\s+than\s+(?:the\s+)?(?:alternatives?|other options?|the rest|anything else)\b",
    re.IGNORECASE,
)

# A proposal put as a question: "Can we do this with Celery?", "How about
# Redis for the queue?". Not a decision by itself — see
# HybridExtractor._accepted_proposals, which takes it as one only when the
# next assistant turn opens by agreeing.
_PROPOSAL = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:(?:can|could|shall|should) (?:we|i|you) "
    r"(?:(?:do|build|handle|implement|write|run|approach|solve) (?:this|it|that|them) "
    r"(?:with|using|on|in|via)|use|go with|try|switch to|adopt|move to)|"
    r"how about|what about|why not(?: just)?(?: use)?)\s+"
    r"((?:(?![.!?](?=\s|$))[^\n]){3,90})\?",
    re.IGNORECASE | re.MULTILINE,
)

# The next assistant turn opening by agreeing. A reply that agrees and then
# qualifies ("Sure, but I'd use X instead") is not acceptance of the
# proposal, so a contrastive word in the first sentence voids it.
_ACCEPTANCE = re.compile(
    r"^\s*(?:yes|yep|yeah|sure|sounds good|agreed|agree|ok(?:ay)?|makes sense|"
    r"good (?:call|idea|point|choice)|will do|fine by me|perfect|great|absolutely|"
    r"definitely|let'?s do (?:it|that)|going with (?:that|it)|running with (?:it|that)|"
    r"that works|works for me|on it|done)\b",
    re.IGNORECASE,
)
_ACCEPTANCE_VOID = re.compile(
    r"\b(?:but|however|instead|rather|although|though|actually)\b", re.IGNORECASE,
)


# Errors are matched two ways, because they are stated two ways.
#
# The original single pattern required an error KEYWORD followed by the
# description ("Error: <text>"), from a vocabulary of exception class
# names and HTTP codes. Real transcripts mostly do neither: "a port
# collision in the integration tests", "an OOM on the Windows runner",
# "CUDA out of memory", "the backfill is timing out". Measured on the
# eval corpus, that pattern recalled 1 of 12 labelled errors.

# Named exceptions, tracebacks and HTTP status codes. Captures the token
# ITSELF plus trailing context, so "422 error" survives rather than being
# reduced to "error".
# Bare symptom words with no subject carry no information and would
# otherwise be emitted as nodes on their own.
_ERROR_STOPWORDS = frozenset({
    "error", "errors", "exception", "failed", "failure", "broken",
    "crash", "crashed", "race", "timeout", "hangs", "flaky", "panic",
    "regression", "regressions", "leak", "leaks", "collision", "collisions",
    "deadlock", "deadlocks", "segfault", "segfaults",
})

# The stopword check runs on the label with any leading determiner removed:
# a user turn reading only "any regressions" is a question, not a failure,
# and it reached the graph purely because the determiner made the label
# longer than the bare stopword.
_ERROR_DETERMINER = re.compile(
    r'^(?:any|some|no|the|a|an|these|those|this|that|more|other|'
    # Leading discourse adverbs are not part of the failure's name: a
    # subject window that starts one word too early yields "Also flock is
    # unreliable" where the label is "flock is unreliable".
    r'also|additionally|furthermore|moreover|however|besides)\s+',
    re.IGNORECASE,
)

# An exception named as one that is CAUGHT is part of the handling code, not
# a failure that happened: "catches only ImportError" is the diagnosis of a
# bug, and recording ImportError as an error of the session is wrong.
_ERROR_HANDLED = re.compile(
    r'\b(?:catch(?:es|ing)?|caught|except|excluding|handles?|handled|handling|'
    r'raises?|raising|swallow(?:s|ed|ing)?)\s+(?:only\s+|just\s+)?$',
    re.IGNORECASE,
)

# One word of a subject or object window.
#
# This is deliberately the permissive `[\w./-]+` and NOT the "may contain a
# dot but must not end on one" form (`[\w/-]+(?:\.[\w/-]+)*`) that it looks
# like it should be. That form is ambiguous — a run like `word.word.word` can
# be split many ways — and nesting it inside the `{0,3}` window repeats made
# the error patterns backtrack catastrophically: 6.3 seconds on a 15 KB
# message built from `"word."`, on the hot path of a proxy that scans
# whatever a caller sends. Python 3.10 has no atomic groups to fence it with,
# so the boundary problem is solved after the match instead, by
# `_drop_leading_sentence`.
_TOKEN = r'[\w./-]+'

# The few words in front of the thing that broke — "a port collision", "the
# later save discards…". Written as a FLAT bounded run of characters, not as
# a repeat of whole tokens.
#
# `(?:[\w./-]+\s+){0,3}` reads better and is a latent denial of service: the
# inner `+` can give back inside every token, and nesting that in a `{0,3}`
# repeat makes the alternatives multiply. Scanning a 15 KB message built from
# `"word."` took 6.3 seconds — on the hot path of a proxy that scans whatever
# a caller sends it. The flat form backtracks at most `n` times instead of
# `n**3`, and the same message now takes ~60 ms.
#
# Cost of the flat form: the window can end mid-phrase or step over a full
# stop, which `_drop_leading_sentence` cleans up after the match.
# The trailing `\s` is load-bearing, not tidiness. Without it the window can
# end mid-word and the vocabulary that follows matches INSIDE a word: `race`
# inside "All of them trace", `hang` inside "single-user use is unchanged".
# Ending the window on whitespace puts the vocabulary at a word boundary by
# construction, which is what the nested-token form gave for free.
def _subject_window(max_chars: int, extra: str = "") -> str:
    return r"(?:[\w./\-" + extra + r" ]{0," + str(max_chars - 1) + r"}\s)?"


def _object_run(max_chars: int) -> str:
    """A trailing object window. No boundary requirement — an object cut
    mid-word costs label quality, not correctness, and `_clip` trims it."""
    return r"[\w./\- ]{0," + str(max_chars) + r"}"

# Trailing context stops at a clause boundary (`,` `;`) as well as at sentence
# end. Running to the next full stop meant one match swallowed the errors named
# after it: in "OperationalError subclasses DatabaseError, and OperationalError
# covers database is locked" a single greedy match consumed the whole sentence,
# and `database is locked` was unreachable because finditer does not return
# overlapping matches.
#
# A status code needs evidence that it was *received*, not merely named. A bare
# `[45]\d{2}` treated "Decided: 404 rather than 403" — a design choice — as two
# failures. So the code must either follow a production verb ("every request
# returns 500", where the subject sits BEFORE the code) or be followed by
# error/response/status. Named exceptions need neither: the class name is
# already the failure.
_ERROR_TYPED = re.compile(
    # A qualified exception path is the normal way libraries name their
    # exceptions in a traceback — `psycopg2.errors.UniqueViolation`,
    # `sqlalchemy.exc.IntegrityError`, `requests.exceptions.ConnectionError`.
    # The bare-class branch below could not reach any of them: the class is
    # preceded by a dotted module path, and UniqueViolation does not end in
    # Error or Exception at all. Anchored on an `errors`/`exceptions`/`exc`
    # module segment so an ordinary dotted attribute (`config.Settings`) is
    # not mistaken for a failure. Flat run, no nested quantifier — see _TOKEN.
    r'\b((?:[\w.]{0,40}\.(?:errors?|exceptions?|exc)\.[A-Z]\w+\b|'
    r'[A-Z]\w*(?:Error|Exception)\b|Traceback|'
    # Runtime error codes and runtime-level failures that carry no
    # Error/Exception suffix: POSIX errno names (ECONNREFUSED, ENOENT,
    # EADDRINUSE), Node's ERR_* codes, a Go panic, an unhandled promise
    # rejection. Each is unambiguous on its own — nobody writes ECONNREFUSED
    # in prose about anything but a failure — so no context is required.
    r'E[A-Z]{4,14}\b|ERR_[A-Z_]{3,30}\b|OOMKilled|'
    r'panic:|runtime error:|[Uu]nhandled (?:promise rejection|exception|error)|'
    r'[Uu]ncaught (?:exception|error|TypeError|ReferenceError))'
    r'(?:[\s:-]+[^.!?,;\n]{0,60})?)',
)

# HTTP status failures, split out of _ERROR_TYPED so this half can be
# case-insensitive.
#
# _ERROR_TYPED must stay case-SENSITIVE: its `[A-Z]\w*(?:Error|Exception)`
# branch is what distinguishes the class `TimeoutError` from the English word
# "error", and IGNORECASE there would match "an error occurred" as an
# exception class. But that sensitivity also applied to the production verbs,
# so a status report that OPENED a sentence — "Getting a 500 from
# /api/orders", "Returns 502 under load" — never matched, because the verb
# was capitalised. Mid-sentence reports matched and sentence-initial ones did
# not, which is not a distinction anyone intends.
#
# Scoped inline flags — `(?i:...)` — would express this in one pattern, but
# they need Python 3.11 and this package supports 3.10.
#
# The guard the original carried is preserved: a bare code is not a failure.
# "Decided: 404 rather than 403" is a design choice, so the code must either
# follow a production verb or be followed by error/response/status.
_ERROR_STATUS = re.compile(
    r'\b(' + _subject_window(30) +
    r'(?:returns?|returning|returned|throws?|throwing|threw|gives?|got|getting|'
    r'receives?|received|responds? with|responded with|fails? with|'
    # A determiner between the verb and the code is the ordinary way this is
    # written ("getting a 500", "got an HTTP 502"); requiring \s+ straight
    # onto the digits missed every one of them.
    # `s?`: "started returning 503s" pluralises the code.
    r'failing with)\s+(?:an?\s+|the\s+)?(?:HTTP\s*)?[45]\d{2}s?\b'
    r'(?:[\s:-]+[^.!?,;\n]{0,60})?)',
    re.IGNORECASE,
)

_ERROR_STATUS_NAMED = re.compile(
    r'\b((?:HTTP\s*)?[45]\d{2}\s+(?:error|errors|response|status|'
    # The standard reason phrase is the other way a status is named as a
    # failure — "502 Bad Gateway from nginx" — and it is as unambiguous as
    # the word "error" after the code. "404 rather than 403" still has
    # neither and is still a design choice.
    r'bad gateway|internal server error|not found|unauthori[sz]ed|forbidden|'
    r'gateway timeout|service unavailable|too many requests|'
    r'unprocessable (?:entity|content)|request timeout|conflict|'
    r'method not allowed|bad request|payload too large)\b'
    r'(?:[\s:-]+[^.!?,;\n]{0,60})?)',
    re.IGNORECASE,
)

# Data-integrity failures are stated as an adjective in front of the thing that
# is broken ("one corrupt header"), which the symptom vocabulary below cannot
# reach — it only looks backwards for a subject.
_ERROR_INTEGRITY = re.compile(
    r'\b((?:corrupt(?:ed)?|malformed|truncated|unreadable|mismatched|'
    r'unparseable)\s+' + _TOKEN + r'(?:\s+' + _TOKEN + r')?)',
    re.IGNORECASE,
)

# Data-loss verbs. A whole class of defect is stated as plain prose about what
# the system does to your data — "the later save discards everything the
# earlier one added", "one bad read permanently deletes everyone's memory" —
# with no exception name, no status code and no symptom noun to key on. These
# are the failures most worth carrying into a resume and none of the patterns
# above could see them.
#
# Only forms that describe what the system DID: `deletes`/`deleted` but not
# the imperative `delete`, so "Actually drop that library" stays an
# instruction. A trailing object is required, so "zero lost" is not a failure.
_ERROR_DAMAGE = re.compile(
    r'\b(' + _subject_window(40, extra="'") +
    r'(?:deletes|deleted|discards|discarded|dropped|loses|lost|'
    r'overwrites|overwrote|clobbers|clobbered|wipes|wiped|'
    r'resurrects|resurrected|reinstates|reinstating|reinstated|'
    r'corrupts|corrupted|silently (?:ignores|drops|fails))'
    r'\s+[^.!?,;\n]{3,50})',
    re.IGNORECASE,
)

# "no longer resurrects a prune" describes the fix. A general negation check
# cannot be used here: "WebSocket message NOT triggering re-render" and "NO
# dial timeout" are real failures whose names contain a negator. Only the
# phrases that mean *this used to happen and no longer does* are excluded.
# The label is preceded by a statement that the failure was fixed:
# "Fixed: 422 error — ...", "Resolved the timeout by ...". Distinct from
# _ALREADY_FIXED, which means "not an error at all any more" and excludes
# the match: here the error is real, was hit, and is now resolved — which
# is exactly what a resume must say so the next session does not go
# looking for a bug that is gone.
_FIX_LEAD = re.compile(
    r"\b(?:fixed|resolved|patched|solved|corrected|repaired|addressed|"
    r"eliminated|closed)\b\s*[:\-—]?\s*(?:the\s+|a\s+|an\s+|this\s+|that\s+)?$",
    re.IGNORECASE,
)

_ALREADY_FIXED = re.compile(
    r'\b(?:no longer|not any ?more|already fixed|since fixed)\b',
    re.IGNORECASE,
)

# Marks the second clause as a CONSEQUENCE of the first rather than a second
# item in a list. See the sentence-level dedup in _extract_one_message.
_CAUSAL_LINK = re.compile(
    r'\b(?:so|so that|which (?:meant|means|left|made)|therefore|hence|thus|'
    r'as a result|with the result that|leaving|meaning)\b',
    re.IGNORECASE,
)

# Vulnerability classes are named, not described, and were invisible to every
# other pattern. No trailing context: the class name IS the label, and the
# clause after it is prose about the fix.
#
# Acronyms stay case-sensitive — lowercase `rce` and `xss` occur inside
# ordinary words far more often than they occur as vulnerabilities. The
# spelled-out phrases are case-insensitive, because prose does not
# capitalise "sql injection".
_ERROR_VULN = re.compile(
    r'\b(IDOR|XSS|CSRF|SSRF|RCE|SQLi|TOCTOU|CVE-\d{4}-\d+|'
    r'(?i:SQL injection|path traversal|privilege escalation|'
    r'session fixation|open redirect|prototype pollution))\b',
)

# Silent failure — the system reporting health it does not have.
#
# "The persistence_broken flag stayed False, so stats reported healthy over an
# empty database" is the worst class of bug there is: nothing raises, nothing
# logs, and the monitoring says everything is fine. It is also invisible to
# every other pattern here, because the vocabulary of failure never appears —
# the whole sentence is made of words that normally mean success.
#
# Keyed on a reporting verb followed by a health claim. The claim is what
# makes it a defect rather than good news: a turn only bothers to write
# "reported healthy" when the point is that it was not.
_ERROR_FALSE_HEALTH = re.compile(
    r'\b(' + _subject_window(30) +
    r'(?:reported|reports|returned|showed|shows|said|says|stayed|remained|'
    r'still (?:read|reads|showed|shows))\s+'
    r'(?:healthy|green|ok|okay|fine|clean|success\w*|passing|valid|'
    r'False|True|zero|empty|0)\b' + _object_run(40) + r')',
    re.IGNORECASE,
)

# Misclassification. "any task whose label ends in a file path was retyped as
# a FILE node" — the data is not lost or corrupt, it is filed under the wrong
# thing, which is why nothing errors and the symptom shows up somewhere else
# entirely. The `re`/`mis`/`wrongly` prefix is required: a bare "typed as" is
# ordinary description ("the field is typed as a string").
_ERROR_MISCLASSIFIED = re.compile(
    r'\b((?:re|mis|wrongly\s+|incorrectly\s+|silently\s+)'
    r'(?:typed|classified|label(?:l?ed)?|categor(?:ised|ized)|routed|parsed|'
    r'mapped|counted|attributed)\s+as\s+' + _object_run(40) + r')',
    re.IGNORECASE,
)

# Absence defects. "missing X", "no dial timeout", "lacks a retry" is one of
# the most common shapes a defect takes in engineering prose, and none of the
# patterns above could see it: there is no exception, no status code and no
# symptom noun, only the thing that should have been there and was not.
#
# `no` is excluded on purpose — "with no downtime" and "no extra library" are
# requirements and design notes, not failures. So is `absent`, which in this
# prose is nearly always predicative ("only runs when tiktoken is absent"),
# a condition rather than a defect.
#
# The thing missing must be a noun: `(?!\w*ly\b)` rejects "absent entirely".
#
# The missing thing is a short noun phrase, not one word: "missing readiness
# gates", "a missing statistical test". Stopping at one token recorded
# "missing readiness" and "missing statistical", which name nothing. Up to
# two more words are taken, never a function word — so "missing email
# validation in LoginRequest" still ends its phrase at "in".
_ABSENCE_WORD = (r'(?:\s+(?!(?:in|on|from|for|and|or|but|is|are|was|were|so|to|'
                 r'the|a|an|when|after|because|which|that)\b)' + _TOKEN + r')')
_ERROR_ABSENCE = re.compile(
    r'\b((?:missing|lacks|lacking|never set|unset)'
    r'\s+(?:a|an|the)?\s*(?!\w*ly\b)' + _TOKEN + _ABSENCE_WORD + r'{0,2}'
    r'(?:\s+(?:in|on|from|for)\s+' + _TOKEN + r')?)',
    re.IGNORECASE,
)

# Inert code. "There is a char/4 fallback but it is unreachable" states that a
# code path exists and never runs — a defect with no failure event to name, so
# nothing else here can reach it. The subject window is wider than the other
# patterns because the thing that is dead is usually named a clause earlier
# ("a char/4 fallback but it is unreachable").
_ERROR_INERT = re.compile(
    r'\b(' + _subject_window(50) + r'(?:is|are|was|were)\s+'
    r'(?:unreachable|unreliable|dead code|silently ignored|'
    r'never (?:called|reached|run|hit|used|fired)|'
    r'not (?:reached|called|persisted|applied|enforced)))',
    re.IGNORECASE,
)

# Symptom vocabulary, with the noun phrase that precedes it — the subject
# is what identifies the failure ("teardown race", not "race").
# The plainest way anybody reports a defect — "the build fails", "CI is
# failing on the Windows runner", "Docker build broke after the base image
# bump" — had no pattern at all. _ERROR_SYMPTOM keys on a vocabulary of
# named symptoms (deadlock, segfault, timeout) and _ERROR_TYPED needs an
# exception class or a status code, so a bare failure verb was invisible.
# Probed against ordinary phrasings rather than the corpus, this was the
# single largest source of missed errors.
#
# Same flat-run construction as the patterns above, for the same
# backtracking reason — no nested quantifiers.
#
# `fixed`/`resolved` are deliberately absent: those are repair verbs,
# handled separately. The past participle "broken" is included but
# "breaking" is not — "breaking change" is a description of an intended
# change, not a failure.
_ERROR_FAILING = re.compile(
    r'\b(' + _subject_window(40) +
    r'(?:fails?|failed|failing|broke|broken|errored|erroring|'
    r'blew up|fell over|went red)'
    # The object clause is REQUIRED, not optional. Without it the pattern
    # matched a bare noun ("cache the fail") and every fragmentary restatement
    # of a failure already captured elsewhere ("Three fail", "21 percent of
    # the test suite fails") — measured on the corpus, that cost 13 points of
    # error precision for no recall. A failure worth carrying into a resume
    # says what it happened to: "fails with exit code 1", "failing on the
    # Windows runner", "broke after the base image bump".
    r'\s+(?:with|on|in|at|during|after|because of|halfway)\s+'
    + _object_run(40) + r')',
    re.IGNORECASE,
)

# "failed" in front of a noun is an adjective, not a report: "background
# retry for failed webhook deliveries" is a FEATURE that handles failures,
# and "Haven't started a rollback job for failed deploys yet" is a to-do.
# _ERROR_FAILED_SUBJECT read both as the failure "…failed deploys yet".
# The word before the verb decides it: a preposition, a quantifier or a verb
# that takes failures as its object ("retries failed requests") makes it
# attributive. "The failed deploy from last night" is deliberately NOT in
# the list — a definite article usually points at one real incident.
_ATTRIBUTIVE_BEFORE_FAILED = frozenset({
    "for", "of", "on", "with", "to", "from", "into", "about", "against",
    "all", "any", "every", "each", "some", "no", "many", "few", "several",
    "retry", "retries", "retrying", "rerun", "reruns", "re-run", "replay",
    "replays", "requeue", "requeues", "resend", "resends", "skip", "skips",
    "handle", "handles", "handling", "log", "logs", "logging", "count",
    "counts", "track", "tracks", "tracking", "clean", "cleans", "clear",
    "clears", "drop", "drops", "collect", "collects", "reprocess",
    "reprocesses", "surface", "surfaces", "alert", "alerts", "list", "lists",
})


# One fixed-width negative lookbehind per word, so the check costs nothing
# unless the engine is already standing on a candidate verb.
_NOT_ATTRIBUTIVE = "".join(
    r"(?<!\b" + re.escape(w) + r" )" for w in sorted(_ATTRIBUTIVE_BEFORE_FAILED)
)


# The past-tense forms with a named subject — "Deploy failed, rolled back",
# "The migration errored out halfway", "the cron job stopped running" —
# are complete reports on their own; the object clause _ERROR_FAILING
# demands exists to keep out the bare noun ("the fail") and the bare
# present tense ("Three fail"), neither of which these forms can be. The
# subject is REQUIRED here (at least one word before the verb) so a
# sentence-initial "Failed." fragment does not qualify.
#
# The subject is a FLAT bounded run — `[\w./\- ]{1,40}\s` — not
# `[\w./\-]+(?: [\w./\-]+){0,5}`. The token-repeat form is the latent
# denial of service _TOKEN's comment above describes: a first draft of this
# pattern used it and took 1.1 seconds on the 15 KB `"word."` payload the
# ReDoS test feeds it, against ~4 ms for every neighbouring pattern. The
# flat run costs a subject that may open mid-phrase, which
# _drop_leading_sentence already cleans up for the other patterns.
_ERROR_FAILED_SUBJECT = re.compile(
    r'\b([\w./\- ]{1,40}\s'
    r'(?:' + _NOT_ATTRIBUTIVE + r'failed|errored(?: out)?|crashed|died|'
    r'stopped (?:running|working|responding|processing)|'
    r'(?:is|are|got|gets|was|were) (?:stuck|hung|wedged|unresponsive)|'
    r'exits? (?:with )?(?:code )?[1-9]\d{0,2}|exit code [1-9]\d{0,2}|'
    r'exited (?:with )?(?:code )?[1-9]\d{0,2}|'
    r'(?:is|are|went|was|were) (?:down|offline|unreachable)|'
    r'keeps? (?:crashing|restarting|rebalancing|dropping|failing|timing out)|'
    r'(?:can.?t|cannot|could not|couldn.?t|fails? to) (?:find|locate|resolve|connect|open|load|reach))'
    r'(?:\s+[^.!?,;\n]{0,40})?)',
    re.IGNORECASE,
)

# Defect words that are also the names of ordinary things. "baseline
# logistic regression at 0.71 AUC" is a model, "the regression suite" is a
# test suite; neither is a regression anyone has to fix.
_NOT_A_DEFECT = re.compile(
    r'\b(?:(?:logistic|linear|ridge|lasso|polynomial|poisson|quantile|'
    r'isotonic|kernel|bayesian|stepwise|multivariate|ordinal)\s+regression|'
    r'regression\s+(?:tests?|suites?|testing|models?|coefficients?|analysis)|'
    # A soft delete is a feature — a row marked deleted instead of removed —
    # and "soft deletes on the tenant table" reads to _ERROR_DAMAGE as data
    # being deleted.
    r'soft[- ]delet\w*)\b',
    re.IGNORECASE,
)

# The explicit header form: "Bug: X", "Error: X", "Issue: X". It is the
# plainest way anyone labels a defect — in a status update, a commit
# message, a test report or tool output — and no pattern keyed on it: a
# header whose description happened to contain no exception name, status
# code or symptom word ("Bug: the evaluation set overlapping the training
# window") produced nothing. The header is the evidence, so the
# description needs none of its own.
#
# Anchored to the start of a line or sentence so an inline mention ("the
# issue: we never retried") is left to the other patterns. Bullets, bold
# and a leading emoji are accepted because that is how an assistant writes
# a status list. Plural headers ("Errors:") are lists, and the first item
# is taken the same way.
# The line/sentence start is `(?:^|\n|(?<=[.!?]))[ \t]*` — ONE quantifier over
# the whitespace. `(?<=[.!?])\s+` followed by `[ \t]*` could split a run of
# spaces between the two in every possible way, and mask_mentions leaves
# exactly such runs where it blanks a quote: 0.85 s on a 79 KB fuzz input
# in this pattern alone.
_ERROR_HEADER = re.compile(
    r'(?:^|\n|(?<=[.!?]))[ \t]*(?:[-*\u2022][ \t]+)?(?:\*\*|__)?'
    r'(?:bugs?|issues?|errors?|problems?|exceptions?|failures?|blockers?|'
    r'regressions?|incidents?|defects?|root cause|symptoms?|known issues?|'
    r'hit a snag|snag)'
    r'(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*'
    r'((?:(?![.!?](?=\s|$))[^\n]){5,120})',
    re.IGNORECASE | re.MULTILINE,
)

# A header whose "description" says there is nothing to report.
_NO_DEFECT = re.compile(
    r'^(?:none|n/?a|nil|nothing|no (?:errors?|issues?|problems?|bugs?)|'
    r'0|zero|not (?:yet )?known|tbd|unknown)\b',
    re.IGNORECASE,
)

# Verbs that introduce a problem as their object. "Ran into connection pool
# exhaustion", "hit a snag with the token refresh", "we tripped over a race
# in the fixture" — the verb says the object went wrong, so the object needs
# no failure vocabulary of its own. "picked up" and "came across" are not
# here: both are as often about finding a ticket or a library.
_ERROR_ENCOUNTER = re.compile(
    r'\b(?:ran into|run(?:ning)? into|runs into|'
    r'hit (?:a|an|another) (?:snag|problem|issue|bug|wall|error|edge case)'
    r'(?:\s*(?::|with|in|on|—|-))?|'
    r'hitting (?:a|an|another) (?:snag|problem|issue|bug|wall|error)(?:\s*(?::|with|in|on))?|'
    r'encounter(?:ed|ing|s)?|tripped (?:over|on)|stumbled (?:on|onto|across|over))'
    r'\s+((?:(?![.!?](?=\s|$))[^\n,;]){4,90})',
    re.IGNORECASE,
)

# "We're seeing X", "getting X", "noticed X": verbs of observation. Unlike
# the encounter verbs, what is observed is as often good news ("we're
# seeing a 20% speedup", "getting 200 OK now") as a defect, so the object
# must itself name something wrong. The vocabulary is the ordinary English
# of defects — not a list of this corpus's bugs — and deliberately broad,
# because the verb has already narrowed the sentence to an observation.
_DEFECT_WORD = re.compile(
    r'\b(?:errors?|fail\w*|crash\w*|timeouts?|timing out|leak\w*|races?|flak\w*|'
    r'duplicat\w+|missing|stale|drift\w*|skew\w*|spikes?|spiking|growth|growing|'
    r'loss|lost|lag\w*|nan|exceptions?|panics?|deadlocks?|slow\w*|latency|'
    r'regress\w*|bias\w*|corrupt\w*|mismatch\w*|overlap\w*|contaminat\w+|'
    r'inconsisten\w+|non-?determinis\w+|unbounded|blocking|blocked|churn|thrash\w*|'
    r'warnings?|oom\w*|[45]\d\d|[45]xx|denied|refused|reject\w*|invalid|broken|'
    r'wrong|incorrect|cycles?|traversal|injection|vulnerab\w+|off-by-one|'
    r'overflow\w*|underflow|hang\w*|stuck|freez\w+|jank\w*|flicker\w*|drop\w*|'
    r'zombie|orphan\w*|conflicts?|collisions?|contention|starv\w+|throttl\w+|'
    r'dangling|use-after-free|segfault\w*|violations?|unhandled|uncaught|'
    r'exhaust\w*|saturat\w+|backlog|retry storms?|dupes?|bugs?|issues?|problems?)\b',
    re.IGNORECASE,
)
#
# Two more shapes share the gate. "There's a race in the fixture teardown" —
# the existential is how a defect is most often introduced in speech, and
# as often introduces anything else ("there's a helper for that"). And
# "we picked up X" in the sense of acquiring it: "picked up a memory leak
# somewhere in the refactor" versus "picked up the ticket".
_ERROR_OBSERVED = re.compile(
    r"(?:\b(?:we|i|users|customers|they|people|everyone)(?:'re|'m| are| am)?\s+"
    r"(?:still\s+|now\s+|also\s+)?(?:seeing|getting|hitting|noticing|observing)|"
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:seeing|getting|noticed|noticing|observed)|"
    r"\b(?:we|i|they|users)\s+(?:noticed|observed|saw|got|hit|picked up)|"
    r"\b(?:there'?s|there is|there are|there was|there were)(?:\s+(?:now|still|also))?)"
    r"\s+((?:(?![.!?](?=\s|$))[^\n,;]){4,90})",
    re.IGNORECASE | re.MULTILINE,
)

# The same inability reported with no subject at all — "Can't connect to
# Redis from the worker pod" opens the sentence. _ERROR_FAILED_SUBJECT
# requires a subject to keep out fragments; this form is anchored to the
# start of a sentence instead, which is its own guarantee that it is a
# report and not a clause of something else.
_ERROR_CANNOT_INITIAL = re.compile(
    r'(?:^|[.!?][ \t]+|\n[ \t]*)((?:can.?t|cannot|could not|couldn.?t|unable to|failed to)\s+'
    r'(?:find|locate|resolve|connect|open|load|reach|start|bind|write|read|parse)'
    r'\s+[^.!?,;\n]{3,50})',
    re.IGNORECASE,
)

# Test-runner and CI summaries: "3 failed, 41 passed", "1 error". The count
# is the whole report.
_ERROR_COUNT = re.compile(
    r'\b((?:[1-9]\d{0,4}) (?:failed|failures?|errors?)(?:, \d+ passed)?)\b',
    re.IGNORECASE,
)

# A latency regression stated as two numbers: "takes 12 seconds, used to
# take 200ms", "went from 200ms to 12s". The COMPARISON is the failure and
# it is required: "the dashboard takes 4.2s to first paint, needs to be
# under 1.5s" is a measurement and a target — the goal of a performance
# session, which the corpus labels as such — not a report that something
# got worse. A first draft made the comparison optional and recorded that
# goal as an error.
_ERROR_SLOWER = re.compile(
    r'\b(' + _subject_window(40) +
    r'(?:(?:now )?takes|taking|took|jumped to|regressed to|climbed to|slowed to)'
    r'\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|min)'
    r'[^.!?\n]{0,40}?(?:used to|it was|was|instead of|down from|up from)\s+[^.!?\n]{0,30}'
    r'|' + _subject_window(40) +
    r'(?:went|regressed|slowed) from\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|min)'
    r'\s+to\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|min))',
    re.IGNORECASE,
)

_ERROR_SYMPTOM = re.compile(
    r'\b(' + _subject_window(40) +
    r'(?:out of memory|oom|segfaults?|segmentation fault|stack overflow|'
    r'deadlocks?|race condition|race|collisions?|memory leaks?|'
    r'timing out|timed out|times out|timeouts?|hangs?|hanging|flaky|'
    r'panics?|crash(?:es|ed|ing)?|regressions?|'
    r'nil pointer(?:\s+dereference)?|null pointer(?:\s+(?:exception|dereference))?|'
    r'infinite loop|not triggering|borrow checker error|fails? intermittently|'
    r'gc pressure|garbage collection pressure|poison messages?|consumer lag|'
    r'schema drift|partition skew|goroutine leaks?|connection churn|'
    r'thundering herd|'
    # User-visible symptoms named as nouns: what a person reports before
    # anyone has a stack trace.
    r'stale (?:data|reads?|cache|results?)|blank (?:page|screen)|white screen|'
    r'spinner forever|never (?:loads?|finishes|returns|completes)|'
    r'silently (?:stopped|fails?|drops?|ignores?)|dropped under load)'
    # `after` and `when` are how the trigger is usually named ("timeout
    # after 30s", "crashes when the input is empty"); they were missing
    # from the connective list, so the object was cut off.
    r'(?:\s+(?:in|on|from|between|during|under|while|across|after|when|at)\s+' + _object_run(30) +
    r'|\s+(?:halting|blocking|breaking|rejecting|overwhelming|causing|growing|'
    r'putting|discarding)' + _object_run(30) + r')?)',
    re.IGNORECASE,
)

#
# "adding" counts only in the header form ("Adding: redis"). As a plain verb
# it adds anything — "adding agentic tool-call support" recorded the
# dependency "agentic".
_DEPENDENCY = re.compile(
    r'(?:(?:pip install|pip add|npm install|npm add|yarn add|pnpm add|poetry add|'
    r'cargo add|go get|installed?)\s*[:\-]?|adding\s*:)'
    r'\s+([a-zA-Z][a-zA-Z0-9_\-]{2,40})',
    re.IGNORECASE,
)

_ENV = re.compile(
    r'(?:Python|Node\.?js?|npm|pip|Docker|Redis|PostgreSQL|SQLite|MongoDB|'
    r'Linux|macOS|Windows|Ubuntu|Debian)\s+([\d\.]+\+?)',
    re.IGNORECASE,
)

_GOAL_OPENERS = re.compile(
    r'(?:goal|objective|trying to|want to|need to|building|creating|'
    r'let.s build|we.re building|the plan is|we need to build|task is)'
    # Same run-past-the-full-stop bug the decision and task captures had:
    # a goal in the opening turn ran into the sentence after it.
    r'\s*[:\-]?\s*((?:(?![.!?](?=\s|$))[^\n]){10,120})',
    re.IGNORECASE,
)

# ── Endpoint / schema patterns ────────────────────────────────────────────────
#
# ENDPOINT and SCHEMA are full node types (graph.py creates nodes
# for them, to_context_block() has dedicated sections, the LLM extraction
# prompt asks for them) but the heuristic extractor had NO patterns to
# ever populate ExtractedData.endpoints/.schemas. Since
# use_llm_extraction defaults to False, this meant these node types could
# never be created at all in the shipped default configuration — a gap
# distinct from (and upstream of) the separately-fixed bug where
# _deduplicate() dropped these fields even when something DID populate
# them.

# "POST /api/auth/login", "GET /api/users/:id", etc.
_ENDPOINT = re.compile(
    r'\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[\w\-/{}:.]+)',
)
# A span that is a list of routes and nothing else (parentheticals like
# "(returns JWT)" and separators allowed): see the task pass.
_ENDPOINT_ONLY = re.compile(
    r'^(?:\s*(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/[\w\-/{}:.]+'
    r'(?:\s*\([^)]{0,40}\))?\s*[,;]?\s*(?:and\s+)?)+\.?\s*$',
    re.IGNORECASE,
)

# Header format: "Schema: users table — id (UUID PK), email (unique)..."
_SCHEMA_HEADER = re.compile(
    r'(?:^|\n)[ \t]*schema[ \t]*[:\-][ \t]*(.{5,120})',
    re.IGNORECASE,
)

# Inline "X table" mention — "a new sessions table to track logins".
# Deliberately excludes common non-identifier words immediately before
# "table" (generic phrasing like "the table below") via _SCHEMA_STOP_WORDS
# below, and is negation-checked the same way decisions are  — a
# statement like "No refresh_tokens table needed" must not produce a
# schema node claiming that table exists.
_SCHEMA_TABLE = re.compile(
    r'\b([a-z][a-z0-9_]*)\s+table\b',
    re.IGNORECASE,
)
_SCHEMA_STOP_WORDS = frozenset({
    "the", "a", "an", "this", "that", "data", "lookup", "routing",
    "truth", "below", "above", "following", "same", "new",
})


# ── Language ─────────────────────────────────────────────────────────────────
#
# Every prose pattern in this module is English. Run on another language —
# or on the Hindi-English mix a lot of developers actually type — they do
# not fail quietly, they misfire: "paisa ni bada key add karogi paid but
# pending all task and task recall" became the pending task "all task and
# task recall and all hidden bugs…", because `pending` is an English word
# in a sentence that is not English. What does carry across languages is
# structure: file paths, exception names, status codes, checkboxes, a
# "TODO:" header. looks_english() decides which of the two a message gets.
#
# The test is the PRESENCE of another language's function words, not the
# absence of English ones: terse English status lines ("Created: db.py.
# Files: api/orders.py.") contain no English function words at all, while
# Hinglish carries plenty of them ("I want", "and", "all"). Calibrated on
# 6,162 English messages across four labelled corpora: none misclassified.
_ENGLISH_FW = frozenset("""the a an and or but of to in on at for with from by as is are was
were be been being it its this that these those we i you he she they them our my your their
us me not no do does did done have has had will would can could should so if then than there
here what which who when where why how all any some more most other into over after before just
also now up out about via per each both only very""".split())

# Function words of the languages developers most often mix with English
# or write in instead: Hindi/Urdu in Latin script, Spanish, Portuguese,
# French, German, Italian. Each is chosen to be rare as an English word or
# identifier — "die", "man", "come", "de" are left out for that reason.
_FOREIGN_FW = frozenset("""hai hain ka ki ke ko mein karo karna kar nahi nahin aur bhi abhi kya
sab chahiye chaiye hona hua mujhe mujha humko tumko bahut bhut lekin agar toh kuch sirf jaise
wala wali raha rahi liye yeh ye woh hum tum aap kiya karogi karoge dena lena hoga hogi sakta
sakti apna apni hota hoti tha thi jo ma ab itna kaise kyun
el los las del que por para con una está necesitamos pero como más también después porque
não uma com os das dos mas você
la les des est une pour avec pas nous vous dans sur mais être devons avons sommes vers avant
der das und ist nicht mit für auf wir müssen sind oder auch noch
il gli della sono anche questo""".split())

_FENCE_OR_INLINE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)
_WORD = re.compile(r"[^\W\d_]+")


def looks_english(text: str) -> bool:
    """True unless `text` reads as another language (see "Language" above).

    Code, inline or fenced, is ignored: identifiers are not prose in any
    language. A message that is mostly non-Latin script is not English.
    """
    t = _FENCE_OR_INLINE.sub(" ", text or "")
    letters = [c for c in t if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > 0.3:
        return False
    toks = _WORD.findall(t.lower())
    if not toks:
        return True
    foreign = sum(w in _FOREIGN_FW for w in toks)
    english = sum(w in _ENGLISH_FW for w in toks)
    if foreign >= 2 and foreign * 2 >= english:
        return False
    return not (len(toks) >= 6 and foreign / len(toks) >= 0.15)


# ── Mentions ─────────────────────────────────────────────────────────────────
#
# A quoted phrase, a table row and a code block are MENTIONS: an assistant
# explaining a bug quotes the sentence that triggers it ("retry for failed
# deliveries" read as an error), a status table restates numbers ("tasks |
# 53% | same"), a code block is source. Run on a real 612-message agent
# session, most of what the prose patterns extracted came from exactly
# these three places. mask_mentions() blanks them to spaces — keeping every
# offset valid for the passes that read context around a match — before
# the prose patterns run. File patterns still read the original text.
#
# Two exceptions keep real errors reachable. A code block often holds a
# pasted log, so a line in it that reads as an error line — a traceback's
# exception, `FAILED tests/…`, `error: …` — is left visible. And a quoted
# span naming an exception or a status code is the error message itself
# ('I get "TypeError: x is undefined"'), so it is left visible too.
_FENCED = re.compile(r"(```|~~~)[^\n]*\n.*?(?:\1|\Z)", re.DOTALL)
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.MULTILINE)
_QUOTED = re.compile(r'"[^"\n]{1,300}"|“[^”\n]{1,300}”')
_LOG_ERROR_LINE = re.compile(
    r"^\s*(?:[\w.]*(?:Error|Exception)\b\s*:|FAILED\s|ERROR\s|E\s{2,}|"
    r"error(?:\[\w+\])?:|fatal:|panic:|npm ERR!|Uncaught\s)",
    re.IGNORECASE,
)
_QUOTED_ERROR = re.compile(r"\b[A-Z]\w*(?:Error|Exception)\b|\b[45]\d{2}\b|\bE[A-Z]{4,14}\b")


def _spaces(m: re.Match) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def _fence_keep_log_lines(m: re.Match) -> str:
    return "\n".join(ln if _LOG_ERROR_LINE.match(ln) else " " * len(ln)
                     for ln in m.group(0).split("\n"))


def _quote_unless_error(m: re.Match) -> str:
    return m.group(0) if _QUOTED_ERROR.search(m.group(0)) else " " * len(m.group(0))


def mask_mentions(text: str) -> str:
    """`text` with quoted spans, table rows and code blocks blanked to
    spaces (same length, newlines kept). See "Mentions" above. Each kind
    is one linear substitution pass."""
    if not text:
        return text
    if "```" in text or "~~~" in text:
        text = _FENCED.sub(_fence_keep_log_lines, text)
    if "|" in text:
        text = _TABLE_ROW.sub(_spaces, text)
    if '"' in text or "\u201c" in text:
        text = _QUOTED.sub(_quote_unless_error, text)
    return text


# ── Shared technology vocabulary ──────────────────────────────────────────────
#
# One spelling per technology, so two modules that both need to know
# "postgres" and "PostgreSQL" are the same thing cannot drift apart:
# graph.py expands query and label tokens with it, decision_tracker.py
# folds labels onto it before deciding whether two decisions are the same.
# Without the second use, "PostgreSQL for order storage" and "Postgres for
# orders" were two decisions on one topic — which the tracker then reported
# as an unresolved conflict in every resume block.
CANONICAL_TECH = {
    "postgresql": "postgres", "psql": "postgres", "pg": "postgres",
    "mongodb": "mongo",
    "nextjs": "next", "next.js": "next",
    "nodejs": "node", "node.js": "node",
    "typescript": "ts", "javascript": "js",
    "kubernetes": "k8s",
    "golang": "go",
    "postgres": "postgres",
}


def canonical_word(word: str) -> str:
    """One spelling for a technology name, with a plural folded off.

    "orders" and "order" are the same noun in a decision label, and a
    label that says "PostgreSQL" is naming what another says as
    "Postgres".
    """
    w = (word or "").lower().strip(".,!?:;()[]")
    if w in CANONICAL_TECH:
        return CANONICAL_TECH[w]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return CANONICAL_TECH.get(w, w)
