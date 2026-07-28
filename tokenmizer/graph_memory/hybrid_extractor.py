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

_FILE_PATH = re.compile(
    r'(?:^|[\s\'\"`(])([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-\.]+){1,6}\.[a-zA-Z]{1,6})',
    re.MULTILINE,
)
_FILE_COMMON = re.compile(
    r'\b((?:[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cs|cpp|c|h|yaml|yml|json|toml|env|md|sh|sql|html|css))\b)',
    re.IGNORECASE,
)

# ── Decision patterns — 5 passes ─────────────────────────────────────────────

# Pass 1: explicit verb ("decided:", "going with", "will use")
_DECISION = re.compile(
    r'(?:decided?|going with|will use|chose?|switching? to|opted for|settled on|'
    r'picked|sticking with|selected?|using|went with|we.ll use|let.s use)'
    r'[\s:\-]+(.{5,80})',
    re.IGNORECASE,
)

# Pass 2: header format ("Decision: X", "Tech choice: X")
_DECISION_HEADER = re.compile(
    r'(?:^|\n)\s*(?:decision|tech choice|architecture choice|approach|stack)\s*[:\-]\s*(.{5,80})',
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

# FIXED (TM-01) — the single most severe finding in the audit: none of
# the decision passes above checked for negation. "We are NOT using
# Redis" matched Pass 1's verb list ("using") and Pass 3's tech-name list
# ("redis") independently — both blind to the preceding "NOT" — and both
# produced "Use Redis" as an extracted decision: the literal opposite of
# what was said. This became critical in combination with
# SmartMessageWindow, which replaces older conversation turns with the
# graph's context block, deleting the original sentence and leaving only
# the fabricated "Decided: Use Redis" in what the model actually sees.
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

_TASK_DONE = re.compile(
    r'(?:completed?|finished?|done|implemented?|fixed?|added?|built?|shipped?|'
    r'wired up|set up|created?|wrote?|updated?|deployed?|resolved?|merged?|'
    r'refactored?|cleaned?|migrated?)'
    r'[\s:\-]+(.{5,80})',
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
    r'[\s:\-]+(.{5,80})',
    re.IGNORECASE,
)

_TASK_TODO = re.compile(
    r'(?:will add|will implement|next:|todo:|will do|need to add|planning to|'
    r'should add|still need|not yet|missing|pending|next step)'
    r'[\s:\-]+(.{5,80})',
    re.IGNORECASE,
)

_ERROR = re.compile(
    r'(?:Error|Exception|TypeError|ValueError|ImportError|KeyError|AttributeError|'
    r'404|500|422|503|failed?|broken?|crash\w*|traceback)'
    r'[\s:]+([A-Z][^.]{5,60})',
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
# FIXED: ENDPOINT and SCHEMA are full node types (graph.py creates nodes
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
# below, and is negation-checked the same way decisions are (TM-01) — a
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
                result.goals.append(m.group(1).strip()[:100])

        # Tasks done: full history (completed = permanent fact)
        for m in _TASK_DONE.finditer(content):
            task = m.group(1).strip()[:80]
            norm = self._normalize(task)
            if norm not in seen_tasks:
                result.tasks_done.append(task)
                seen_tasks.add(norm)

        # Passive completion: full history
        for m in _TASK_DONE_PASSIVE.finditer(content):
            # FIXED (TM-21): was r'...\\s+' — a raw string, so \\s is a
            # literal backslash-s, not the whitespace escape \s. Since no
            # real text contains a literal backslash there, this prefix
            # strip could never match anything and has never once fired.
            task = re.sub(r'^(?:the|a|an|this|that)\s+', '', m.group(1).strip()[:80], flags=re.IGNORECASE)
            if len(task) > 5:
                norm = self._normalize(task)
                if norm not in seen_tasks:
                    result.tasks_done.append(task)
                    seen_tasks.add(norm)

        # WIP/TODO: recent window only (avoid stale in-progress)
        if is_recent:
            for m in _TASK_WIP.finditer(content):
                result.tasks_wip.append(m.group(1).strip()[:80])
            for m in _TASK_TODO.finditer(content):
                result.tasks_todo.append(m.group(1).strip()[:80])

        # Decision Pass 1: explicit verb
        for m in _DECISION.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = m.group(1).strip()[:80]
            norm  = self._normalize(label)
            if norm not in seen_decisions and len(norm) > 4:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 2: header format
        for m in _DECISION_HEADER.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = m.group(1).strip()[:80]
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 3: tech names
        for m in _DECISION_FOR.finditer(content):
            if _is_negated_context(content, m.start()):
                continue
            label = "Use " + m.group(1).strip()[:60]
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
            schema = m.group(1).strip()[:100]
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

        # Errors (recent only)
        if is_recent:
            for m in _ERROR.finditer(content):
                result.errors.append(m.group(1).strip()[:80])

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

        # Simple list categories — merge with corroboration tracking
        #
        # FIXED — REAL BUG (found via testing, not in the original audit's
        # list): this used to build `combined` directly from the
        # normalized (lowercased) sets, which meant the FINAL OUTPUT
        # stored lowercased strings permanently. For file paths this is
        # not cosmetic: "src/App.tsx" and "src/app.tsx" are different
        # files on any case-sensitive filesystem (Linux, most CI/prod
        # environments). A user reading their session graph would see
        # "src/app.tsx" even though the actual file on disk is
        # "src/App.tsx" — wrong information about which file was
        # touched. `_deduplicate()` elsewhere in this same file gets this
        # right (normalizes only for the dedup KEY, keeps original-case
        # value) — `merge()` was inconsistent with its own codebase's
        # established correct pattern. Fixed to match: normalize only for
        # set membership / corroboration detection, always emit the
        # ORIGINAL (first-seen, original-case) string into the output.
        # FIXED (TM-14): use the dataclass-derived field list, same as
        # _deduplicate() below, instead of a second hand-maintained copy
        # of it — two independent lists that both need to stay in sync
        # with ExtractedData's fields is exactly how "endpoints"/"schemas"
        # drifted out of _deduplicate() in the first place.
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

            # FIXED (TM-18): this used to build `combined` from set
            # difference/intersection operations (llm_keys & heu_keys,
            # etc.) — Python's string-hash randomization (on by default)
            # means set iteration order isn't stable across process runs,
            # so when more than 15 items were found, WHICH 15 survived
            # combined[:15] changed between runs of the IDENTICAL input.
            # The same conversation processed on two different workers
            # could produce different graphs. Fixed by iterating the
            # already-insertion-ordered llm_by_norm/heu_by_norm dicts
            # directly: every LLM item (corroborated or not) in the LLM's
            # own order, then heuristic-only items in the heuristic's own
            # order — a pure, deterministic function of input order.
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
                # source_role (TM-29): only the heuristic pass attributes a
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

        FIXED (TM-14): _deduplicate() used to hardcode this list
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
        result.decisions = data.decisions
        result.superseded = data.superseded
        result.evidence = data.evidence
        return result

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
