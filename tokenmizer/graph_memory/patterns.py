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
_FILE_COMMON = re.compile(
    r'\b((?:[\w\-]+\.(?:'
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
_FILE_EXTENSIONLESS = re.compile(
    r'\b(Dockerfile|Makefile|Procfile|Jenkinsfile|Gemfile|Rakefile|Vagrantfile|'
    r'Brewfile|Justfile|Caddyfile|CODEOWNERS|MANIFEST\.in)\b'
)

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
_CLAUSE_SPAN = r'((?:(?![.!?](?=\s|$))[^\n]){5,80})'

# Pass 1: explicit verb ("decided:", "going with", "will use")
_DECISION = re.compile(
    r'(?:decided?|going with|will use|chose?|switching? to|opted for|settled on|'
    r'picked|sticking with|selected?|using|went with|we.ll use|let.s use|'
    r'leaning toward|recommends?|recommended)'
    r'[\s:\-]+' + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# Pass 2: header format ("Decision: X", "Tech choice: X")
_DECISION_HEADER = re.compile(
    r'(?:^|\n)\s*(?:decision|tech choice|architecture choice|approach|stack)\s*[:\-]\s*'
    + _CLAUSE_SPAN,
    re.IGNORECASE,
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
    r'pnpm|uv|ruff|kong|airflow'
    r')\b(?:(?!\s+(?:to|with|for)\s+\w)[^.!?\n,—\-]){0,40})',
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
    r"(?:decided?|decision|going with|will use|we'?ll use|let'?s use|chose|"
    r"choosing|opted for|settled on|picked|sticking with|selected|switch(?:ing)? to|"
    r"moved? to|migrat\w+ to|adopt(?:ed|ing)?|use|using|with|"
    r"leaning toward|recommends?|recommended)\s*[:\-]?\s*$",
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


def _tech_mention_is_a_decision(content: str, start: int, end: int) -> bool:
    """True if a bare technology name at [start:end] is stated as a choice."""
    before = content[max(0, start - 40):start]
    after = content[end:end + 40]
    if _MIGRATION_SOURCE.search(before):
        return False        # the thing being migrated away from
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

_SUPERSEDED = re.compile(
    r'(?:'
    r'switched?\s+(?:from|away\s+from)|'
    r'switching?\s+from|'
    r'replacing|'
    r'instead\s+of|'
    r'moved?\s+from|'
    r'migrat\w+\s+(?:from|away\s+from)|'
    r'dropping|'
    r'no\s+longer\s+using|'
    r'replaced?\s+\w+\s+with|'
    r'moving\s+(?:away\s+from|from)'
    r')'
    r'\s+(\w[\w\s\.\-]{2,30}?)\s+(?:to|with|for)\s+(\w[\w\s\.\-]{2,30})',
    re.IGNORECASE,
)

# ── Evidence extraction patterns ──────────────────────────────────────────────

# Numeric metrics with context — "latency 340ms", "score was 61"
_EVIDENCE_NUMBER = re.compile(
    r'(\d+(?:\.\d+)?\s*(?:ms|s|seconds?|minutes?|hours?|'
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
    r"[\s,;:—-]+(?:and|or|but|with|for|to|in|on|at|by|from|the|a|an|of|"
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
    r'(?:completed|finished|done|implemented|fixed|added|built|shipped|'
    r'wired up|set up|created|wrote|written|updated|deployed|resolved|'
    r'merged|refactored|cleaned|migrated|restructured|removed|'
    r'switched|replaced)'
    r'[\s:\-]+' + _CLAUSE_SPAN,
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
_TASK_DONE_PASSIVE = re.compile(
    r'((?:(?![.!?](?=\s|$))[^\n]){5,50}?)\s+(?:is|are)\s+'
    r'(?:working|ready|done|complete|live|passing|'
    r'implemented|deployed|fixed|resolved|working now|up and running)',
    re.IGNORECASE,
)

_TASK_WIP = re.compile(
    r'(?:working on|implementing|building|currently\s+\w+ing|adding|integrating|'
    r'setting up|configuring|writing|debugging|investigating)'
    r'[\s:\-]+' + _CLAUSE_SPAN,
    re.IGNORECASE,
)

# `missing` needs a guard the other openers do not: "was missing dependency
# in useEffect" is a past-tense diagnosis of something already fixed, and
# recording it as outstanding work tells the next session to go redo it.
# Only an unqualified "missing X" is a TODO.
_TASK_TODO = re.compile(
    r'(?:will add|will implement|next:|todo:|will do|need to add|planning to|'
    r'should add|still need|not yet|(?<!was )(?<!were )(?<!is )(?<!are )missing|'
    r'pending|next step)'
    r'[\s:\-]+' + _CLAUSE_SPAN,
    re.IGNORECASE,
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
    r'\b((?:[A-Z]\w*(?:Error|Exception)\b|Traceback|'
    + _subject_window(30) +
    r'(?:returns?|returning|returned|throws?|throwing|threw|gives?|got|getting|'
    r'receives?|received|responds? with|responded with|fails? with|'
    r'failing with)\s+(?:HTTP\s*)?[45]\d{2}\b|'
    r'(?:HTTP\s*)?[45]\d{2}\s+(?:error|errors|response|status)\b)'
    r'(?:[\s:-]+[^.!?,;\n]{0,60})?)',
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
_ERROR_ABSENCE = re.compile(
    r'\b((?:missing|lacks|lacking|never set|unset)'
    r'\s+(?:a|an|the)?\s*(?!\w*ly\b)' + _TOKEN +
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
    r'thundering herd)'
    r'(?:\s+(?:in|on|from|between|during|under|while|across)\s+' + _object_run(30) +
    r'|\s+(?:halting|blocking|breaking|rejecting|overwhelming|causing|growing|'
    r'putting|discarding)' + _object_run(30) + r')?)',
    re.IGNORECASE,
)

_DEPENDENCY = re.compile(
    r'(?:pip install|pip add|npm install|npm add|yarn add|pnpm add|poetry add|'
    r'cargo add|go get|adding|installed?)'
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

# Header format: "Schema: users table — id (UUID PK), email (unique)..."
_SCHEMA_HEADER = re.compile(
    r'(?:^|\n)\s*schema\s*[:\-]\s*(.{5,120})',
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
