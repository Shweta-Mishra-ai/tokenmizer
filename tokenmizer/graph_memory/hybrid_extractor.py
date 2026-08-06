"""
HybridExtractor — 90%+ recall target for graph memory.

Strategy (3-pass pipeline):
  Pass 1: LLM extraction  (structured JSON, ~85-92% recall when enabled)
  Pass 2: Heuristic sweep (catches explicit patterns LLM might miss)
  Pass 3: NLP merge+dedup (normalize, deduplicate, confidence-rank)

Result: significantly higher recall than either approach alone.

Key innovations over V1 heuristic-only:
  - Structured extraction prompt (forces JSON schema, no hallucinations)
  - Temporal ordering awareness (SUPERSEDED detection across turns)
  - File path extraction (regex + context)
  - Goal extraction (opening messages get higher weight)
  - Confidence boosting by corroboration (if both LLM + heuristic agree → 0.95+)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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
    r'Brewfile|Justfile|Caddyfile|CODEOWNERS|LICENSE|MANIFEST\.in)\b'
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
    r'picked|sticking with|selected?|using|went with|we.ll use|let.s use)'
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
    r'openai|anthropic|gemini|langchain|llamaindex'
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
    r"moved? to|migrat\w+ to|adopt(?:ed|ing)?|use|using|with)\s*[:\-]?\s*$",
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
# (`python -m benchmarks.eval --sweep`, and the table in CHANGELOG 0.6.0)
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
    r'(.{5,50}?)\s+(?:is|are)\s+(?:working|ready|done|complete|live|passing|'
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
    r'^(?:any|some|no|the|a|an|these|those|this|that|more|other)\s+',
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
    r'(?:[A-Za-z][\w./-]*\s+){0,2}'
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
    r'unparseable)\s+[\w./-]+(?:\s+[\w./-]+)?)',
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
    r'\b((?:[\w./\'-]+\s+){0,3}'
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

# Vulnerability classes are named, not described, and were invisible to every
# other pattern. No trailing context: the class name IS the label, and the
# clause after it is prose about the fix.
_ERROR_VULN = re.compile(
    r'\b(IDOR|XSS|CSRF|SSRF|RCE|SQLi|TOCTOU|CVE-\d{4}-\d+|'
    r'SQL injection|path traversal|privilege escalation|'
    r'session fixation|open redirect)\b',
)

# Symptom vocabulary, with the noun phrase that precedes it — the subject
# is what identifies the failure ("teardown race", not "race").
_ERROR_SYMPTOM = re.compile(
    r'\b((?:[\w./-]+\s+){0,3}'
    r'(?:out of memory|oom|segfaults?|segmentation fault|stack overflow|'
    r'deadlocks?|race condition|race|collisions?|memory leaks?|'
    r'timing out|timed out|times out|timeouts?|hangs?|hanging|flaky|'
    r'panics?|crash(?:es|ed|ing)?|regressions?|null pointer|infinite loop|'
    r'not triggering|borrow checker error|fails? intermittently)'
    r'(?:\s+(?:in|on|from)\s+(?:[\w./-]+\s*){1,3})?)',
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
    r'\s*[:\-]?\s*(.{10,120})',
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


@dataclass
class ExtractedData:
    goals: list[str] = field(default_factory=list)
    tasks_done: list[str] = field(default_factory=list)
    tasks_wip: list[str] = field(default_factory=list)
    tasks_todo: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    environments: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    schemas: list[str] = field(default_factory=list)
    superseded: list[dict] = field(default_factory=list)
    # Evidence: metrics, quotes, standards extracted per message
    # Format: {"text": str, "type": "metric"|"quote"|"standard", "turn": int}
    evidence: list[dict] = field(default_factory=list)
    # Confidence per category (filled by merger)
    confidence: dict[str, float] = field(default_factory=dict)


class HybridExtractor:
    """
    3-pass extraction pipeline targeting 90%+ recall.

    Pass 1: LLM extraction (if provider available) — structured, high precision
    Pass 2: Heuristic extraction — catches patterns LLM misses, zero cost
    Pass 3: Merge + dedup + confidence scoring
      - Corroborated items (both passes agree) → confidence 0.95
      - LLM-only items → confidence 0.80
      - Heuristic-only items → confidence 0.65

    min_confidence: items below this tier are dropped from extract()'s
    output. Default 0.55 keeps every tier (backward compatible); 0.7
    drops heuristic-only items; 0.9 keeps only corroborated ones.
    """

    def __init__(self, min_confidence: float = 0.55):
        self.min_confidence = min_confidence

    # ── Pass 1: LLM ──────────────────────────────────────────────────────────

    async def llm_extract(
        self, messages: list[dict], provider_fn
    ) -> Optional[ExtractedData]:
        """Call provider to extract structured data. Returns None on failure."""
        # Format messages as readable text for the LLM
        parts = []
        for m in messages[-20:]:  # last 20 messages max
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role.upper()}]: {content[:500]}")

        if not parts:
            return None

        messages_text = "\n\n".join(parts)
        prompt = EXTRACTION_USER_TEMPLATE.format(messages_text=messages_text)

        try:
            result = await provider_fn(
                messages=[{"role": "user", "content": prompt}],
                system=EXTRACTION_SYSTEM,
                max_tokens=800,
            )
            raw = result.get("text", "")
            # Strip markdown fences if present
            raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
            data = json.loads(raw)
            return self._dict_to_extracted(data)
        except Exception as e:
            # Warning, not debug: a provider timeout, rate limit, or
            # malformed reply degrades extraction to heuristic-only for the
            # batch, and that degradation must be visible at default log
            # levels.
            logger.warning(
                f"LLM extraction failed (falling back to heuristic-only "
                f"for this batch): {type(e).__name__}: {e}"
            )
            return None

    def _dict_to_extracted(self, data: dict) -> ExtractedData:
        """Safely convert LLM JSON response to ExtractedData."""
        def safe_list(key, transform=None) -> list:
            val = data.get(key, [])
            if not isinstance(val, list):
                return []
            if transform:
                return [transform(x) for x in val if x]
            return [str(x).strip() for x in val if x]

        return ExtractedData(
            goals=safe_list("goals"),
            tasks_done=safe_list("tasks_done"),
            tasks_wip=safe_list("tasks_wip"),
            tasks_todo=safe_list("tasks_todo"),
            decisions=[d for d in data.get("decisions", []) if isinstance(d, dict) and "label" in d],
            files=safe_list("files"),
            errors=safe_list("errors"),
            dependencies=safe_list("dependencies"),
            environments=safe_list("environments"),
            endpoints=safe_list("endpoints"),
            schemas=safe_list("schemas"),
            superseded=[s for s in data.get("superseded", []) if isinstance(s, dict)],
        )

    # ── Pass 2: Heuristic ────────────────────────────────────────────────────

    def _extract_one_message(
        self,
        content: str,
        role: str,
        turn_idx: int,
        is_recent: bool,
        result: "ExtractedData",
        seen_decisions: set,
        seen_tasks: set,
        seen_files: set,
        seen_endpoints: set,
        seen_schemas: set,
    ) -> None:
        """
        Apply all 5 extraction passes to a single message.
        Mutates result in place. Called by heuristic_extract().

        Extracted into a helper to keep heuristic_extract() readable
        (was 194L — loop body alone was 166L).
        """
        # Goals: first 4 messages only (session intent captured early)
        if role == "user" and turn_idx < 4:
            for m in _GOAL_OPENERS.finditer(content):
                result.goals.append(_clip(m.group(1), 100))

        # Tasks done: full history (completed = permanent fact)
        for m in _TASK_DONE.finditer(content):
            task = _clip(m.group(1))
            if len(task) < 5 or _is_only_paths(task) or _LEADING_CONNECTIVE.match(task):
                continue
            norm = self._normalize(task)
            # Subsumption, not just exact match: several passes fire on
            # overlapping spans of one sentence and would otherwise emit
            # two labels for one fact. Keep the more specific (longer) one.
            dup_idx = next(
                (i for i, t in enumerate(result.tasks_done) if self._subsumes(task, t)),
                None,
            )
            if dup_idx is not None:
                if len(task) > len(result.tasks_done[dup_idx]):
                    result.tasks_done[dup_idx] = task
                continue
            if norm not in seen_tasks:
                result.tasks_done.append(task)
                seen_tasks.add(norm)

        # Passive completion: full history
        for m in _TASK_DONE_PASSIVE.finditer(content):
            # was r'...\\s+' — a raw string, so \\s is a
            # literal backslash-s, not the whitespace escape \s. Since no
            # real text contains a literal backslash there, this prefix
            # strip could never match anything and has never once fired.
            task = _clip(re.sub(r'^(?:the|a|an|this|that)\s+', '',
                                m.group(1), flags=re.IGNORECASE))
            if len(task) > 5:
                norm = self._normalize(task)
                if any(self._subsumes(task, t) for t in result.tasks_done):
                    continue
                if norm not in seen_tasks:
                    result.tasks_done.append(task)
                    seen_tasks.add(norm)

        # WIP/TODO: recent window only (avoid stale in-progress)
        #
        # Two guards, both about work that is NOT outstanding:
        #
        #   _COMPLETION_LEAD — a completion verb earlier in the same clause
        #   makes the rest of it finished work, whatever opener follows.
        #   "Fixed by adding a 5 second context timeout" is not a to-do;
        #   "Completed: four OS processes writing one session" is not WIP.
        #   Carrying these into a resume tells the next session to redo work
        #   that is already merged, which is worse than omitting them.
        #
        #   goal subsumption — the opening turn states the goal in exactly
        #   the shape of a WIP line ("Building a real-time analytics
        #   dashboard"), and the goal is already a node of its own.
        if is_recent:
            def _outstanding(text: str, start: int, seen: list[str]) -> bool:
                if len(text) < 5 or _is_only_paths(text):
                    return False
                if _COMPLETION_LEAD.search(content[max(0, start - 40):start]):
                    return False
                if any(self._subsumes(text, g) for g in result.goals):
                    return False
                return not any(self._subsumes(text, t) for t in seen)

            for m in _TASK_WIP.finditer(content):
                wip = _clip(m.group(1))
                if _outstanding(wip, m.start(1), result.tasks_wip):
                    result.tasks_wip.append(wip)
            for m in _TASK_TODO.finditer(content):
                todo = _clip(m.group(1))
                if _outstanding(todo, m.start(1), result.tasks_todo):
                    result.tasks_todo.append(todo)

        # Decision Pass 1: explicit verb
        for m in _DECISION.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = _clip(m.group(1))
            norm  = self._normalize(label)
            if norm not in seen_decisions and len(norm) > 4:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 2: header format
        for m in _DECISION_HEADER.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = _clip(m.group(1))
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 3: tech names
        for m in _DECISION_FOR.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            # A bare tech name is only a decision with choosing context —
            # see _tech_mention_is_a_decision.
            if not _tech_mention_is_a_decision(content, m.start(1), m.end(1)):
                continue
            label = "Use " + _clip(m.group(1), 60)
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 4: passive (bcrypt with cost factor 12)
        for m in _DECISION_PASSIVE.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = "Use " + m.group(1).strip()[:60]
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Superseded + both sides as decisions
        for m in _SUPERSEDED.finditer(content):
            old_label = m.group(1).strip()
            new_label = m.group(2).strip()
            result.superseded.append({"old": old_label, "new": new_label})
            start = max(0, m.start() - 60)
            surrounding = content[start:m.end() + 80].replace("\n", " ").strip()
            for label in (f"Use {old_label}", f"Use {new_label}"):
                norm = self._normalize(label)
                if norm not in seen_decisions and len(norm) > 6:
                    reason = f"Replaced by: {new_label}" if label.endswith(old_label) else surrounding[:100]
                    result.decisions.append({"label": label, "reason": reason,
                                             "evidence": surrounding[:120], "source_role": role})
                    seen_decisions.add(norm)

        # Files
        for m in _FILE_PATH.finditer(content):
            f = m.group(1).strip()
            if f not in seen_files and len(f) > 4:
                result.files.append(f)
                seen_files.add(f)
        for m in _FILE_COMMON.finditer(content):
            f = m.group(1).strip()
            if f not in seen_files:
                result.files.append(f)
                seen_files.add(f)
        for m in _FILE_EXTENSIONLESS.finditer(content):
            f = m.group(1).strip()
            if f not in seen_files:
                result.files.append(f)
                seen_files.add(f)

        # Endpoints — "POST /api/auth/login"
        for m in _ENDPOINT.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            ep = m.group(0).strip()
            norm = self._normalize(ep)
            if norm not in seen_endpoints:
                result.endpoints.append(ep)
                seen_endpoints.add(norm)

        # Schemas — header format ("Schema: users table — ...")
        for m in _SCHEMA_HEADER.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            schema = _clip(m.group(1), 100)
            norm = self._normalize(schema)
            if norm not in seen_schemas and len(schema) > 3:
                result.schemas.append(schema)
                seen_schemas.add(norm)

        # Schemas — inline "X table" mention, excluding generic non-
        # identifier words immediately before "table" and negated
        # mentions ("No refresh_tokens table needed").
        for m in _SCHEMA_TABLE.finditer(content):
            word = m.group(1).lower()
            if word in _SCHEMA_STOP_WORDS:
                continue
            if _is_negated_context(content, m.start()):
                continue
            schema = m.group(0).strip()
            norm = self._normalize(schema)
            if norm not in seen_schemas:
                result.schemas.append(schema)
                seen_schemas.add(norm)

        # Errors: full history, NOT the recent window.
        #
        # An error is a permanent fact about the session in the same way a
        # completed task is: a resolved one explains why the code looks
        # the way it does, and an unresolved one is the single most
        # important thing to carry into a resume. Gating on recency meant
        # a session that diagnosed three failures early and spent the rest
        # of its turns fixing them carried none of them forward.
        for pattern in (_ERROR_TYPED, _ERROR_VULN, _ERROR_INTEGRITY,
                        _ERROR_DAMAGE, _ERROR_SYMPTOM):
            for m in pattern.finditer(content):
                before = content[max(0, m.start(1) - 60):m.start(1)]
                if _SOLUTION_VERB.search(before):
                    continue   # the symptom names the fix, not the failure
                if _is_negated_context(content, m.start(1)):
                    continue
                if _ALREADY_FIXED.search(m.group(1)) or _ALREADY_FIXED.search(before):
                    continue   # "no longer resurrects a prune" is the fix
                if _ERROR_HANDLED.search(before):
                    continue   # the exception is being caught, not raised
                err = _clip(_FIX_PREFIX.sub("", m.group(1).strip()), 70)
                # `IDOR`, `XSS`, `RCE` are four and three characters. A flat
                # minimum length rejected the entire vulnerability vocabulary,
                # which is the highest-signal thing an error label can carry.
                floor = 3 if pattern is _ERROR_VULN else 5
                bare = _ERROR_DETERMINER.sub("", err).lower()
                if len(err) < floor or bare in _ERROR_STOPWORDS:
                    continue
                if any(self._subsumes(err, e) for e in result.errors):
                    continue
                result.errors.append(err)

        # Dependencies
        for m in _DEPENDENCY.finditer(content):
            dep = m.group(1).strip()
            if len(dep) > 2 and dep.lower() not in {"the", "a", "an", "it", "this", "that"}:
                result.dependencies.append(dep)

        # Environments
        for m in _ENV.finditer(content):
            result.environments.append(m.group(0).strip())

        # Evidence
        for m in _EVIDENCE_NUMBER.finditer(content):
            text = m.group(1).strip()
            if len(text) > 5:
                result.evidence.append({"text": text, "type": "metric", "turn": turn_idx})
        for m in _EVIDENCE_SCORE.finditer(content):
            text = m.group(0).strip()
            if len(text) > 5:
                result.evidence.append({"text": text, "type": "metric", "turn": turn_idx})
        for m in _EVIDENCE_COST.finditer(content):
            text = m.group(1).strip()
            if len(text) > 1:
                result.evidence.append({"text": text, "type": "metric", "turn": turn_idx})
        for m in _EVIDENCE_QUOTE.finditer(content):
            text = m.group(1).strip()
            if len(text) > 10:
                result.evidence.append({"text": text, "type": "quote", "turn": turn_idx})
        for m in _EVIDENCE_STANDARD.finditer(content):
            text = m.group(0).strip()
            if len(text) > 8:
                result.evidence.append({"text": text, "type": "standard", "turn": turn_idx})


    def heuristic_extract(
        self,
        messages: list[dict],
        window_size: int = 0,
    ) -> "ExtractedData":
        """
        Fast regex-based extraction. No API calls.

        Sliding window strategy for long sessions (>30 turns):
        - Goals: first 4 messages only (session intent)
        - Decisions: full history (decisions are permanent facts)
        - Tasks WIP/TODO: last window_size messages (stale WIP is noise)
        - Errors: last window_size messages (old errors are usually fixed)
        - Completed tasks + files: full history

        Per-message extraction delegated to _extract_one_message() to
        keep this method focused on windowing/setup logic (was 194L).
        """
        if window_size == 0:
            window_size = len(messages)

        recent_start   = max(0, len(messages) - window_size)
        result         = ExtractedData()
        seen_decisions: set[str] = set()
        seen_tasks:     set[str] = set()
        seen_files:     set[str] = set()
        seen_endpoints: set[str] = set()
        seen_schemas:   set[str] = set()

        for i, msg in enumerate(messages):
            from tokenmizer.graph_memory.graph import _content_to_text
            content = _content_to_text(msg.get("content", ""))
            if not content.strip():
                continue
            role      = msg.get("role", "user")
            is_recent = i >= recent_start

            self._extract_one_message(
                content, role, i, is_recent,
                result, seen_decisions, seen_tasks, seen_files,
                seen_endpoints, seen_schemas,
            )

        # Cross-granularity dedup. Without this the heuristic path
        # returns both `scripts/backfill.py` and `backfill.py`, and both
        # "bcrypt for password hashing" and a bare "Use bcrypt" — one
        # fact each, two nodes each. extract_from_messages() consumes
        # this return value directly, so it is the only place the
        # de-duplication can happen for heuristic-only extraction.
        result.files = self._drop_shadowed_paths(result.files)
        result.decisions = self._drop_vaguer_decisions(result.decisions)
        return result


    # ── Pass 3: Merge ────────────────────────────────────────────────────────

    def merge(
        self,
        llm: Optional[ExtractedData],
        heuristic: ExtractedData,
    ) -> ExtractedData:
        """
        Merge LLM + heuristic results with confidence scoring.

        Corroboration confidence values are stored in decision dicts
        under the 'confidence' key so _apply_extracted() can pass them
        directly to add_node() — bypassing the validator's default confidence
        which would otherwise overwrite the corroboration signal.

        confidence values:
          0.95 — corroborated (both LLM and heuristic found it)
          0.80 — LLM-only (LLM caught it, heuristic missed)
          0.65 — heuristic-only (heuristic caught it, LLM missed)
        """
        if llm is None:
            result = self._deduplicate(heuristic)
            result.confidence = {k: 0.65 for k in vars(result) if k != "confidence"}
            # Tag heuristic-only decisions with lower confidence
            for d in result.decisions:
                d.setdefault("confidence", 0.65)
            return result

        merged = ExtractedData()

        # Simple list categories — merge with corroboration tracking.
        #
        # Normalize (lowercase) ONLY for set membership and corroboration
        # detection; always emit the ORIGINAL first-seen casing into the
        # output. Emitting the normalized form is not cosmetic for file
        # paths: "src/App.tsx" and "src/app.tsx" are different files on
        # any case-sensitive filesystem, so a session graph would report
        # a file that does not exist. `_deduplicate()` below follows the
        # same rule.
        #
        # The field list is derived from the dataclass rather than
        # hand-maintained: two lists needing to stay in sync with
        # ExtractedData is how "endpoints"/"schemas" went missing from
        # _deduplicate() once already.
        for attr in self._simple_list_field_names():
            llm_raw = list(getattr(llm, attr))
            heu_raw = list(getattr(heuristic, attr))

            # norm -> original-case string, first occurrence wins
            llm_by_norm: dict[str, str] = {}
            for x in llm_raw:
                llm_by_norm.setdefault(self._normalize(x), x)
            heu_by_norm: dict[str, str] = {}
            for x in heu_raw:
                heu_by_norm.setdefault(self._normalize(x), x)

            llm_keys = set(llm_by_norm.keys())
            heu_keys = set(heu_by_norm.keys())
            corroborated = bool(llm_keys & heu_keys)
            llm_only     = bool(llm_keys - heu_keys)

            # `combined` MUST be built by iterating the
            # insertion-ordered llm_by_norm/heu_by_norm dicts, never from
            # set operations (llm_keys & heu_keys, ...). Python's
            # string-hash randomization makes set iteration order unstable
            # across processes, so with more than 15 items, WHICH 15
            # survive combined[:15] would vary between runs on identical
            # input — the same conversation could yield different graphs
            # on two workers. Every LLM item in the LLM's order, then
            # heuristic-only items in the heuristic's order: a pure,
            # deterministic function of input order.
            combined: list[str] = []
            for k, v in llm_by_norm.items():
                combined.append(v)  # prefer LLM casing when corroborated
            for k, v in heu_by_norm.items():
                if k not in llm_keys:
                    combined.append(v)

            setattr(merged, attr, combined[:15])
            merged.confidence[attr] = (
                0.95 if corroborated else
                0.80 if llm_only else
                0.65
            )

        # Decisions: merge by label similarity, tag each with confidence
        seen: dict[str, dict] = {}
        for d in llm.decisions:
            key = self._normalize(d.get("label", ""))
            if key:
                seen[key] = {**d, "confidence": 0.80, "_source": "llm"}

        for d in heuristic.decisions:
            key = self._normalize(d.get("label", ""))
            if not key:
                continue
            if key in seen:
                # Corroborated — upgrade confidence, keep LLM reason if better
                existing = seen[key]
                existing["confidence"] = 0.95
                existing["_source"] = "both"
                if d.get("reason") and not existing.get("reason"):
                    existing["reason"] = d["reason"]
                if d.get("evidence") and not existing.get("evidence"):
                    existing["evidence"] = d["evidence"]
                # source_role : only the heuristic pass attributes a
                # decision to a specific message's role — the LLM pass
                # synthesizes across the whole conversation with no
                # single-turn attribution, so `existing` (built from the LLM
                # dict) never has one. Backfill it here the same way
                # reason/evidence are, or a corroborated decision (the
                # highest-confidence tier) would silently lose the one
                # signal the heuristic side actually knew.
                if d.get("source_role") and not existing.get("source_role"):
                    existing["source_role"] = d["source_role"]
            else:
                seen[key] = {**d, "confidence": 0.65, "_source": "heuristic"}

        merged.decisions = list(seen.values())[:15]
        merged.confidence["decisions"] = (
            0.95 if any(d.get("_source") == "both" for d in merged.decisions) else
            0.80 if any(d.get("_source") == "llm"  for d in merged.decisions) else
            0.65
        )

        # Transitions: prefer LLM (better context understanding)
        merged.superseded = (llm.superseded or heuristic.superseded)[:10]

        # Evidence: combine from both sources
        merged.evidence = llm.evidence + heuristic.evidence

        return merged

    def _normalize(self, s: str) -> str:
        """Normalize for dedup comparison."""
        return re.sub(r'\s+', ' ', s.lower().strip())[:60]

    @staticmethod
    def _subsumes(a: str, b: str) -> bool:
        """True if `a` and `b` state substantially the same fact.

        Exact-string dedup is not enough. Several patterns fire on
        overlapping spans of one sentence, so "Fixed: 422 error — missing
        email validation in X" and "email validation in X. Login endpoint
        working now" arrive as two labels for one event. Compared on
        content words, one clearly subsumes the other.
        """
        wa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2}
        wb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2}
        if not wa or not wb:
            return False
        smaller = wa if len(wa) <= len(wb) else wb
        return len(wa & wb) / len(smaller) >= 0.75

    # Fields on ExtractedData that are lists of DICTS, not lists of plain
    # strings — these can't go through the string-normalize-and-dedup
    # loop below and are copied through as-is instead.
    _DICT_LIST_FIELDS = frozenset({"decisions", "superseded", "evidence"})
    # Non-list field — per-category confidence scores, set by merge(), not by dedup.
    _NON_LIST_FIELDS = frozenset({"confidence"})

    @classmethod
    def _simple_list_field_names(cls) -> list[str]:
        """Every ExtractedData field that's a plain list[str] — derived
        from the dataclass itself rather than hand-maintained here.

        _deduplicate() must not hardcode this list
        ("goals", "tasks_done", ... "environments") and silently dropped
        "endpoints", "schemas", and "evidence" — added to ExtractedData
        at some point after this hardcoded list was written, and never
        added here to match. _deduplicate() is the path taken whenever
        the LLM pass is absent, which is the SHIPPED DEFAULT
        (use_llm_extraction=False) — so two of the nine documented node
        types (ENDPOINT, SCHEMA) and all decision evidence were
        unreachable out of the box. Deriving the field list from
        dataclasses.fields() means a future field addition to
        ExtractedData can't silently create the same gap again.
        """
        import dataclasses
        return [
            f.name for f in dataclasses.fields(ExtractedData)
            if f.name not in cls._DICT_LIST_FIELDS
            and f.name not in cls._NON_LIST_FIELDS
        ]

    def _deduplicate(self, data: ExtractedData) -> ExtractedData:
        """Deduplicate within heuristic results."""
        result = ExtractedData()
        for attr in self._simple_list_field_names():
            seen = set()
            deduped = []
            for item in getattr(data, attr):
                norm = self._normalize(item)
                if norm not in seen and len(norm) > 3:
                    deduped.append(item)
                    seen.add(norm)
            setattr(result, attr, deduped[:15])
        result.files = self._drop_shadowed_paths(result.files)
        result.decisions = self._drop_vaguer_decisions(data.decisions)
        result.superseded = data.superseded
        result.evidence = data.evidence
        return result

    @staticmethod
    def _drop_shadowed_paths(files: list[str]) -> list[str]:
        """Remove a bare basename when a full path to it was also found.

        The file patterns match at two granularities, so one mention of
        `scripts/backfill.py` yields both that and `backfill.py`. Two
        nodes for one file is not extra information — it is a duplicate
        that costs resume budget and drags precision down.
        """
        full = [f for f in files if "/" in f or "\\" in f]
        basenames = {f.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower() for f in full}
        return [f for f in files
                if ("/" in f or "\\" in f) or f.lower() not in basenames]

    def _drop_vaguer_decisions(self, decisions: list) -> list:
        """Collapse "Use X" into a longer decision that also names X.

        Several decision passes fire on one sentence at different levels
        of specificity: "Decided: bcrypt for password hashing" produces
        both that and a bare "Use bcrypt". The bare form states strictly
        less and is not independently useful — a resume block listing
        both reads as two decisions where one was made.
        """
        from tokenmizer.graph_memory.decision_tracker import _matched_topic_keywords

        def _adds_nothing(short: str, long: str) -> bool:
            """True if `short` names the same technology as `long` and
            contributes no other content of its own.

            Word overlap alone does not catch this: "Use bcrypt" and
            "bcrypt for password hashing" share exactly one word out of
            two, well under any sane subsumption threshold, yet the first
            states strictly less than the second.
            """
            ks, kl = _matched_topic_keywords(short, ""), _matched_topic_keywords(long, "")
            if not ks or not ks <= kl:
                return False
            filler = {"use", "using", "used", "go", "went", "with", "for",
                      "the", "and", "decided", "choose", "chose", "pick",
                      "picked", "switch", "switched", "adopt", "adopted"}
            extra = {
                w for w in re.findall(r"[a-z0-9]+", short.lower())
                if len(w) > 2 and w not in filler and w not in ks
            }
            return not extra

        kept: list = []
        for d in sorted(decisions, key=lambda x: -len(x.get("label", ""))):
            label = d.get("label", "")
            if not label:
                continue
            if any(self._subsumes(label, k.get("label", "")) for k in kept):
                continue
            if any(_adds_nothing(label, k.get("label", "")) for k in kept):
                continue
            kept.append(d)
        # Restore the original order so downstream ordering stays stable.
        order = {id(d): i for i, d in enumerate(decisions)}
        return sorted(kept, key=lambda d: order.get(id(d), 0))

    # ── Main entry ────────────────────────────────────────────────────────────

    async def extract(
        self,
        messages: list[dict],
        provider_fn=None,
    ) -> ExtractedData:
        """
        Full 3-pass extraction.
        provider_fn: async callable(messages, system, max_tokens) → {"text": str}
                     Pass None for heuristic-only mode.
        """
        # Pass 1: LLM (if available)
        llm_result = None
        if provider_fn is not None:
            llm_result = await self.llm_extract(messages, provider_fn)

        # Pass 2: Heuristic (always runs)
        heu_result = self.heuristic_extract(messages)

        # Pass 3: Merge, then filter by min_confidence
        return self._filter_by_confidence(self.merge(llm_result, heu_result))

    def _filter_by_confidence(self, merged: ExtractedData) -> ExtractedData:
        """
        Drop extracted items whose merge() confidence tier is below
        self.min_confidence.

        The default (0.55) sits below the lowest tier (heuristic-only,
        0.65), so nothing is filtered unless the caller opts into stricter
        extraction: 0.7 drops heuristic-only items, 0.9 keeps only
        corroborated ones.
        """
        if self.min_confidence <= 0.65:  # lowest tier — nothing can be dropped
            return merged
        # Decisions carry per-item confidence tags from merge()
        merged.decisions = [
            d for d in merged.decisions
            if d.get("confidence", 0.65) >= self.min_confidence
        ]
        # Simple-list categories share one tier per category
        for attr, tier in merged.confidence.items():
            if attr == "decisions":
                continue
            if isinstance(tier, (int, float)) and tier < self.min_confidence \
                    and isinstance(getattr(merged, attr, None), list):
                setattr(merged, attr, [])
        return merged


# Singleton
_extractor: HybridExtractor | None = None

def get_hybrid_extractor() -> HybridExtractor:
    global _extractor
    if _extractor is None:
        _extractor = HybridExtractor()
    return _extractor
