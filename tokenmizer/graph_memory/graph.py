"""
Graph Memory — the core of TokenMizer's context continuity.

Key fixes over V3:
- Node deduplication by normalized label+type
- LLM-powered extraction (haiku/gpt-4o-mini) with heuristic fallback
- Full message history extraction (not just last 10)
- Incremental extraction (skip already-processed messages)
- New node types: ENVIRONMENT, GOAL, TEST, ENDPOINT, SCHEMA
- Graph pruning / aging
- Secret redaction on every write
- SQLite persistence (survives restarts)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
def _get_validator():
    from tokenmizer.graph_memory.validator import get_validator
    return get_validator()


# ── Node / Edge types ────────────────────────────────────────────────────────

class NodeType(str, Enum):
    TASK = "task"
    FILE = "file"
    DECISION = "decision"
    ERROR = "error"
    CONCEPT = "concept"
    DEPENDENCY = "dependency"
    API = "api"
    PROJECT = "project"
    AGENT = "agent"
    # V4 additions
    ENVIRONMENT = "environment"   # runtime env, versions, infra
    GOAL = "goal"                 # top-level session objective
    TEST = "test"                 # test file / test result
    ENDPOINT = "endpoint"         # HTTP endpoint definition
    SCHEMA = "schema"             # data model / DB schema


class NodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"       # active — shown in resume (GREEN)
    FAILED = "failed"
    SUPERSEDED = "superseded"     # replaced by newer decision (YELLOW) — kept in history
    INVALIDATED = "invalidated"   # explicitly wrong/cancelled (RED) — kept as warning
    ARCHIVED = "archived"         # old but valid, not relevant now (GRAY)
    MODIFIED = "modified"         # alias for SUPERSEDED — backward compat


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    IMPLEMENTS = "implements"
    FIXES = "fixes"
    BLOCKS = "blocks"
    PART_OF = "part_of"
    SUPERSEDES = "supersedes"


@dataclass
class MemoryNode:
    id: str
    type: NodeType
    label: str
    status: NodeStatus = NodeStatus.PENDING
    summary: str = ""
    importance: float = 0.5       # 0–1, used in pruning
    confidence: float = 0.7       # 0–1, from GraphValidator
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)   # when this fact became true
    valid_until: float = field(default=0.0)                # 0.0 = currently valid
    _evicted: bool = field(default=False, repr=False)

    def is_valid_at(self, t: float) -> bool:
        """True if this node was active at time t."""
        return self.valid_from <= t and (self.valid_until == 0.0 or self.valid_until > t)

    def touch(self) -> None:
        self.updated_at = time.time()
        self.importance = min(1.0, self.importance + 0.05)

    def age_days(self) -> float:
        return (time.time() - self.updated_at) / 86400


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0


# ── Heuristic extractor (fallback) ──────────────────────────────────────────

# ── Task patterns — broad coverage ──────────────────────────────────────────
# Explicit completion verbs
_TASK_DONE_EXPLICIT = re.compile(
    r'(?:completed?|finished?|done|implemented?|created?|added?|fixed?|wrote?|updated?|refactored?|built?|set up|set-up|wired up|hooked up|deployed?|shipped?)\s*[:\-]?\s*(.{8,80})',
    re.IGNORECASE,
)
# Passive / state phrases: "X is ready", "X is working", "X now works"
_TASK_DONE_PASSIVE = re.compile(
    r'(.{5,60})\s+(?:is|are|has been|have been)\s+(?:ready|done|complete|completed|finished|working|fixed|resolved|set up|passing)',
    re.IGNORECASE,
)
# "X now works/passes/runs"
_TASK_DONE_NOW = re.compile(
    r'(.{5,60})\s+now\s+(?:works?|passes?|runs?|compiles?|deploys?|connects?)',
    re.IGNORECASE,
)
_TASK_WIP = re.compile(
    r'(?:working on|implementing|building|creating|adding|writing|fixing|updating|setting up|integrating)\s+(.{8,80})',
    re.IGNORECASE,
)
_TASK_TODO = re.compile(
    r'(?:next(?:\s+step)?(?:\s+is)?|need to|should|will|todo|plan to|going to|about to|still need)\s+(?:implement|create|add|write|fix|update|build|set up|configure|test|deploy|integrate)\s+(.{8,80})',
    re.IGNORECASE,
)

# ── Decision patterns — comprehensive ────────────────────────────────────────
# Explicit decision verbs
_DECISION_EXPLICIT = re.compile(
    r'(?:decided?|chose?|selected?|going with|will use|switched? to|settled on|opted for|went with|sticking with|picked)\s+(.{5,80})',
    re.IGNORECASE,
)
# "X makes more sense", "X is the better choice", "better to use X"
_DECISION_REASONING = re.compile(
    r'(?:(.{5,60})\s+(?:makes? more sense|is (?:the )?better|is (?:the )?best|would work better|is (?:more )?appropriate))',
    re.IGNORECASE,
)
# "X over Y", "X instead of Y", "X rather than Y"
_DECISION_COMPARISON = re.compile(
    r'([\w][\w\s]{1,40})\s+(?:over|instead of|rather than|vs\.?|versus)\s+([\w][\w\s]{1,40})',
    re.IGNORECASE,
)
# "Using X for Y", "X for the Y"
_DECISION_USING = re.compile(
    r'(?:using|use)\s+([\w][\w\s\-]{3,40})\s+(?:for|as|to handle|to manage)\s+(.{5,60})',
    re.IGNORECASE,
)
# "Switching to X", "switched to X", "moving to X", "migrating to X"
_DECISION_SWITCH = re.compile(
    r'(?:switch(?:ing|ed)?\s+to|mov(?:ing|ed)\s+to|migrat(?:ing|ed)\s+to|'
    r'replac(?:ing|ed)\s+.*?\s+with|'
    r'use\s+([\w][\w\.\-]{1,30})\s+(?:not|instead\s+of))\s+([\w][\w\.\s\-]{1,40})',
    re.IGNORECASE,
)
# "Switching from X to Y"
_DECISION_FROM_TO = re.compile(
    r'(?:switch(?:ing|ed)|mov(?:ing|ed)|migrat(?:ing|ed))\s+from\s+([\w][\w\.\s\-]{1,30})\s+to\s+([\w][\w\.\s\-]{1,40})',
    re.IGNORECASE,
)
# Tech name + rationale: "FastAPI because async", "PostgreSQL for concurrent writes"
_DECISION_TECH = re.compile(
    r'\b(fastapi|flask|django|postgresql|mysql|sqlite|mongodb|redis|jwt|oauth|docker|kubernetes|'
    r'react|vue|angular|typescript|python|nodejs|go|rust|aws|gcp|azure|railway|vercel|'
    r'bcrypt|argon2|celery|kafka|rabbitmq|elasticsearch|nginx|gunicorn|uvicorn)\b',
    re.IGNORECASE,
)

_FILE = re.compile(r'[\w][\w\-/]*\.(?:py|js|ts|jsx|tsx|go|rs|java|rb|cpp|c|h|css|html|json|yaml|yml|toml|md|txt|sh|env|sql|proto|graphql)')
_DEP = re.compile(r'(?:pip install|npm install|yarn add|npm i|require|from)\s+([\w\-]+(?:[>=<~!]+[\d.]+)?)', re.IGNORECASE)
# Also catch "added X>=version to requirements" and "X==version" patterns
_DEP_ADDED = re.compile(r'(?:added?|install(?:ing|ed)?|includ(?:ing|ed)?)\s+([\w\-]+[>=<~!]+[\d][\d.]*)', re.IGNORECASE)
_DEP_INLINE = re.compile(r'\b([a-z][a-z0-9\-]{2,25})[>=<]{1,2}[\d]+[\d.]*\b', re.IGNORECASE)
_ENV = re.compile(r'\b(?:python|node|npm|yarn|docker|postgres|postgresql|redis|mongodb|mysql|sqlite|nginx|ubuntu|debian|macos)\s*(?:v?[\d][\d.]*)?\b', re.IGNORECASE)
_GOAL = re.compile(
    r'(?:(?:let\'?s?|i\'?m?|we\'?re?)\s+(?:build|create|make|develop|implement|design|working on|starting)\s+(.{10,120})|'
    r'(?:the goal|objective|purpose|aim)\s+(?:is|here is|of this)\s+(?:to\s+)?(.{10,120})|'
    r'(?:building|creating|developing|implementing)\s+(?:a |an |the )?(.{10,100})\s+(?:using|with|in|for))',
    re.IGNORECASE,
)
_ERROR = re.compile(
    r'(?:error|exception|bug|issue|problem|failing?|crashed?|broke?n?|traceback|'
    r'TypeError|ValueError|KeyError|AttributeError|ImportError|SyntaxError|'
    r'404|422|500|502|503)\s*[:\-]?\s*(.{5,100})',
    re.IGNORECASE,
)


def _heuristic_extract(messages: list[dict]) -> dict:
    """
    Extract structured facts using regex patterns.
    Improved: catches passive completions, comparison decisions,
    tech-name decisions, broader goal language, and errors.
    Used as primary when use_llm_extraction=False, fallback otherwise.
    """
    tasks_done: list[dict] = []
    tasks_wip:  list[dict] = []
    tasks_todo: list[dict] = []
    decisions:  list[dict] = []
    files:      list[str]  = []
    deps:       list[str]  = []
    envs:       list[str]  = []
    goals:      list[str]  = []
    errors:     list[dict] = []

    # Track tech names mentioned — used to build decisions below
    tech_mentions: list[str] = []

    for msg in messages:
        text = msg.get("content", "")
        role = msg.get("role", "assistant")
        if not text:
            continue

        # ── Tasks ────────────────────────────────────────────────────────────

        # Explicit completion verbs: "implemented X", "fixed X", "created X"
        for m in _TASK_DONE_EXPLICIT.finditer(text):
            label = m.group(1).strip()[:80]
            if label:
                tasks_done.append({"label": label, "status": "completed"})

        # Passive completions: "X is ready", "X is working", "tests are passing"
        for m in _TASK_DONE_PASSIVE.finditer(text):
            label = m.group(1).strip()[:80]
            if label and len(label) > 4:
                tasks_done.append({"label": label, "status": "completed"})

        # "X now works / passes"
        for m in _TASK_DONE_NOW.finditer(text):
            label = m.group(1).strip()[:80]
            if label and len(label) > 4:
                tasks_done.append({"label": label, "status": "completed"})

        # In progress
        for m in _TASK_WIP.finditer(text):
            label = m.group(1).strip()[:80]
            if label:
                tasks_wip.append({"label": label, "status": "in_progress"})

        # Pending / todo
        for m in _TASK_TODO.finditer(text):  # _TASK_TODO = module-level compiled regex
            label = m.group(1).strip()[:80]
            if label:
                tasks_todo.append({"label": label, "status": "pending"})

        # ── Decisions ────────────────────────────────────────────────────────

        # Explicit: "decided to use X", "going with X", "went with X"
        for m in _DECISION_EXPLICIT.finditer(text):
            label = m.group(1).strip()[:80]
            if label:
                decisions.append({"label": label, "rationale": ""})

        # Reasoning: "X makes more sense", "X is the better choice"
        for m in _DECISION_REASONING.finditer(text):
            label = m.group(1).strip()[:80]
            if label and len(label) > 5:
                decisions.append({"label": label, "rationale": "makes more sense"})

        # Comparison: "X over Y", "X instead of Y"
        for m in _DECISION_COMPARISON.finditer(text):
            chosen = m.group(1).strip()[:50]
            rejected = m.group(2).strip()[:50]
            if chosen and rejected:
                decisions.append({
                    "label": f"Use {chosen}",
                    "rationale": f"over {rejected}",
                })

        # "Using X for Y"
        for m in _DECISION_USING.finditer(text):
            tech = m.group(1).strip()[:40]
            purpose = m.group(2).strip()[:60]
            if tech and purpose:
                decisions.append({"label": f"Use {tech} for {purpose}", "rationale": ""})

        # "Switching to X" / "use X not Y"
        for m in _DECISION_SWITCH.finditer(text):
            g1 = (m.group(1) or "").strip()  # chosen (the "use X" part)
            g2 = (m.group(2) or "").strip()  # rejected (the "not Y" part)
            chosen = g1 if g1 else g2
            rejected = g2 if g1 else ""
            if chosen and len(chosen) > 1:
                label = f"Use {chosen}"
                rationale = f"instead of {rejected}" if rejected and rejected != chosen else ""
                decisions.append({"label": label, "rationale": rationale})

        # "Switching from X to Y"
        for m in _DECISION_FROM_TO.finditer(text):
            rejected = m.group(1).strip()[:40]
            chosen = m.group(2).strip()[:40]
            if chosen:
                decisions.append({"label": f"Use {chosen}", "rationale": f"replacing {rejected}"})

        # Tech name mentions — collect for context
        for m in _DECISION_TECH.finditer(text):
            tech_mentions.append(m.group(0).lower())

        # ── Files ─────────────────────────────────────────────────────────────
        files.extend(m.group(0) for m in _FILE.finditer(text))

        # ── Dependencies ──────────────────────────────────────────────────────
        _STDLIB = {"os","sys","re","json","time","math","io","typing","abc",
                   "enum","path","str","int","bool","list","dict","set","tuple"}
        for m in _DEP.finditer(text):
            dep = m.group(1).strip()
            if dep and len(dep) > 1 and dep.split(">=")[0].split("==")[0] not in _STDLIB:
                deps.append(dep)
        # "Added stripe>=7.0.0", "installed redis==5.0.0"
        for m in _DEP_ADDED.finditer(text):
            dep = m.group(1).strip()
            if dep and len(dep) > 1:
                deps.append(dep)
        # Inline version pins: "stripe>=7.0.0", "fastapi>=0.111"
        for m in _DEP_INLINE.finditer(text):
            pkg = m.group(1).strip()
            if pkg not in _STDLIB and len(pkg) > 2:
                deps.append(m.group(0).strip())

        # ── Environment ───────────────────────────────────────────────────────
        for m in _ENV.finditer(text):
            env = m.group(0).strip()
            if env and len(env) > 2:
                envs.append(env)

        # ── Goals ─────────────────────────────────────────────────────────────
        # Goals appear in both user AND assistant messages
        for m in _GOAL.finditer(text):
            # Pattern has multiple groups — find first non-None
            label = next((g for g in m.groups() if g), None)
            if label:
                goals.append(label.strip()[:120])

        # ── Errors ────────────────────────────────────────────────────────────
        for m in _ERROR.finditer(text):
            label = m.group(1).strip()[:80] if m.lastindex else m.group(0).strip()[:80]
            if label and len(label) > 4:
                # Resolved if assistant message (assistant is fixing it)
                resolved = role == "assistant"
                errors.append({"label": label, "resolved": resolved})

    # ── Clean and de-duplicate ───────────────────────────────────────────────

    def _clean_label(raw: str) -> str:
        """
        Trim a raw regex capture to a clean, concise label.
        Cuts at REAL sentence boundaries (uppercase after period),
        preserves file paths and technical terms.
        """
        if not raw:
            return ""
        # Only cut at sentence-ending punctuation followed by uppercase
        # This preserves "backend/models.py line 45" but cuts "Fixed X. Also did Y."
        import re as _re
        sentence_end = _re.search(r'[.!?]\s+[A-Z]', raw)
        if sentence_end:
            raw = raw[:sentence_end.start() + 1]
        # Cut at newlines (always a boundary)
        if "\n" in raw:
            raw = raw[:raw.index("\n")]
        # Remove leading articles/conjunctions
        raw = re.sub(r"^(?:the |a |an |and |but |so |also |i've |i have |i am |i'm )", "", raw, flags=re.IGNORECASE)
        # Remove trailing incomplete phrases
        raw = re.sub(r"\s+(?:and|with|in|for|to|the|a|an)\s*$", "", raw, flags=re.IGNORECASE)
        return raw.strip()[:80]

    def _dedup(items: list[dict], key: str = "label") -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for item in items:
            label = _clean_label(item.get(key, ""))
            if not label or len(label) < 5:
                continue
            k = label.lower()[:50]
            if k not in seen:
                seen.add(k)
                item[key] = label  # use cleaned label
                out.append(item)
        return out

    # Clean decision labels too
    def _clean_decisions(items: list[dict]) -> list[dict]:
        out = []
        seen: set[str] = set()
        for d in items:
            label = _clean_label(d.get("label", ""))
            if not label or len(label) < 5:
                continue
            k = label.lower()[:40]
            if k not in seen:
                seen.add(k)
                out.append({"label": label, "rationale": d.get("rationale", "")})
        return out

    return {
        "tasks": _dedup(tasks_done + tasks_wip + tasks_todo),
        "decisions": _clean_decisions(decisions),
        "files": list(dict.fromkeys(files))[:25],
        "dependencies": list(dict.fromkeys(deps))[:25],
        "environment": list(dict.fromkeys(envs))[:15],
        "goals": [_clean_label(g) for g in dict.fromkeys(goals) if g][:5],
        "errors": _dedup(errors),
        "tech_mentions": list(dict.fromkeys(tech_mentions))[:20],
    }


# ── LLM extractor ────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a project memory extractor. Read conversation turns and extract structured facts.

Return ONLY this JSON — no markdown, no explanation, no extra text:
{
  "tasks": [{"label": "verb + specific object (e.g. Implement JWT refresh tokens)", "status": "completed|pending|in_progress|failed"}],
  "decisions": [{"label": "what was chosen (e.g. Use Redis for sessions)", "rationale": "why (e.g. faster than DB)"}],
  "files": ["exact/file/path.py"],
  "errors": [{"label": "specific error (e.g. 422 on login endpoint)", "resolved": true}],
  "dependencies": ["package>=version"],
  "environment": ["Python 3.12", "FastAPI 0.111"],
  "goals": ["top-level session goal (e.g. Build FastAPI auth service with JWT)"],
  "endpoints": ["METHOD /path (e.g. POST /api/auth/login)"],
  "schemas": ["Model: fields (e.g. User: id, email, password_hash)"]
}

TASK rules — capture ALL of these as "completed":
- Explicit: "I implemented X", "created X", "fixed X", "added X"
- Passive: "X is ready", "X is working", "X is done", "tests are passing"
- State change: "X now works", "X now passes", "X is set up"
- Status: "that bug is resolved", "the endpoint is live"

DECISION rules — capture ALL of these:
- Explicit choices: "going with X", "decided to use X", "chose X", "went with X"
- Comparisons: "X over Y", "X instead of Y", "X rather than Y"
- Reasoning: "X makes more sense", "X is better here", "better to use X"
- Tech selection: "using FastAPI because async", "PostgreSQL for concurrent writes"
- ALWAYS fill rationale — even "simpler", "faster", "team preference" counts

GOAL rules:
- Extract from FIRST user message if they describe what they are building
- "Let's build X", "I'm working on X", "We need to create X", "Building X with Y"
- One goal per session — the top-level objective

Return empty arrays for categories with nothing found.
Quality over quantity — 3 accurate nodes beats 10 noisy ones."""


async def _llm_extract(messages: list[dict], provider_fn) -> dict:
    """
    Use cheap model (haiku / gpt-4o-mini) to extract structured facts.
    ~$0.001 per extraction call. Much more accurate than heuristics.
    """
    # Take last 3 messages as the "current turn"
    recent = messages[-3:] if len(messages) >= 3 else messages
    conversation_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m.get('content','')[:2000]}" for m in recent
    )

    try:
        result = await provider_fn(
            messages=[{"role": "user", "content": conversation_text}],
            system=_EXTRACTION_SYSTEM,
            max_tokens=600,
        )
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        # Strip any accidental markdown fences
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM extraction failed ({e}), using heuristic fallback")
        return _heuristic_extract(messages[-3:])


# ── Graph ────────────────────────────────────────────────────────────────────

class GraphMemory:
    """
    In-process graph with SQLite persistence.
    Survives process restarts. One DB file per storage_dir.
    """

    def __init__(self, session_id: str, storage_dir: str = "./checkpoints"):
        self.session_id = session_id
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []
        self._processed_hashes: set[str] = set()
        self._schema_version = 1  # increment when storage format changes  # for incremental extraction
        self._db_path = Path(storage_dir) / "graph_memory.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._safe_init_db()
        self._load()

    def _safe_init_db(self) -> None:
        """Initialize DB, deleting corrupt file if necessary."""
        try:
            self._init_db()
        except Exception:
            logger.warning(f"DB corrupt or unreadable — recreating: {self._db_path}")
            try:
                self._db_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                self._init_db()
            except Exception as e:
                logger.error(f"Cannot initialize DB after cleanup: {e}")

    # ── DB ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graphs (
                    session_id TEXT PRIMARY KEY,
                    nodes_json TEXT NOT NULL,
                    edges_json TEXT NOT NULL,
                    processed_hashes TEXT NOT NULL DEFAULT '[]',
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def _persist(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO graphs
                       (session_id, nodes_json, edges_json, processed_hashes, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        self.session_id,
                        json.dumps([asdict(n) for n in self._nodes.values()]),
                        json.dumps([asdict(e) for e in self._edges]),
                        json.dumps(list(self._processed_hashes)),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Graph persist failed for {self.session_id}: {e}")

    def _load(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT nodes_json, edges_json, processed_hashes FROM graphs WHERE session_id=?",
                    (self.session_id,),
                ).fetchone()
            if not row:
                return
            nodes_data = json.loads(row[0])
            edges_data = json.loads(row[1])
            self._processed_hashes = set(json.loads(row[2]))

            for nd in nodes_data:
                nd.pop("_evicted", None)
                n = MemoryNode(**{k: v for k, v in nd.items() if k != "_evicted"})
                self._nodes[n.id] = n
            for ed in edges_data:
                self._edges.append(MemoryEdge(**ed))
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning(f"Corrupted DB for {self.session_id} — starting fresh: {e}")
            self._nodes = {}
            self._edges = []
            self._processed_hashes = set()
            # Re-initialize the DB file
            try:
                self._db_path.unlink(missing_ok=True)
                self._init_db()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Graph load failed for {self.session_id}: {e}")

    # ── Nodes ────────────────────────────────────────────────────────────────

    def _node_id(self, node_type: str, label: str) -> str:
        normalized = f"{node_type}:{label.lower().strip()}"
        return hashlib.sha1(normalized.encode()).hexdigest()[:12]

    def _normalize_label(self, label: str) -> str:
        return label.lower().strip().rstrip(".,!?")

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        status: NodeStatus = NodeStatus.PENDING,
        summary: str = "",
        importance: float = 0.5,
    ) -> str:
        from tokenmizer.security.redaction import redact_node
        label, summary = redact_node(label, summary)

        norm = self._normalize_label(label)
        node_id = self._node_id(node_type.value, norm)

        if node_id in self._nodes:
            # Dedup: update existing node instead of creating duplicate
            existing = self._nodes[node_id]
            existing.touch()
            # Only upgrade status (completed > in_progress > pending)
            status_rank = {
                NodeStatus.PENDING: 0,
                NodeStatus.IN_PROGRESS: 1,
                NodeStatus.COMPLETED: 2,
                NodeStatus.FAILED: 3,
                NodeStatus.ARCHIVED: 4,
                NodeStatus.SUPERSEDED: 5,
                NodeStatus.MODIFIED: 5,    # alias for SUPERSEDED
                NodeStatus.INVALIDATED: 6,
            }
            if status_rank.get(status, 0) > status_rank.get(existing.status, 0):
                existing.status = status
            if summary and not existing.summary:
                existing.summary = summary
            return node_id

        # Validate before inserting — reject noise and low-confidence nodes
        validator = _get_validator()
        result = validator.validate(
            label=label,
            node_type=node_type.value,
            summary=summary,
        )
        if not result.accepted:
            logger.debug(f"Node rejected: {label!r} ({result.rejection_reason})")
            return ""  # empty string = rejected, callers must check

        # Apply type correction if validator detected mismatch
        if result.corrected_type:
            try:
                node_type = NodeType(result.corrected_type)
                node_id = self._node_id(node_type.value, norm)
            except ValueError:
                pass  # keep original type if correction is unknown

        node = MemoryNode(
            id=node_id,
            type=node_type,
            label=label[:120],
            status=status,
            summary=summary[:300],
            importance=importance,
            confidence=result.confidence,
        )
        self._nodes[node_id] = node

        # Decision contradiction detection:
        # If this is a new DECISION, check if any existing decision covers
        # the same topic — if so, mark it as MODIFIED (superseded).
        # Old decisions are KEPT in graph (for history/rollback) but
        # marked MODIFIED so they are excluded from resume context.
        if node_type == NodeType.DECISION and status == NodeStatus.COMPLETED:
            try:
                from tokenmizer.graph_memory.decision_tracker import (
                    find_contradicting_decisions,
                )
                to_supersede = find_contradicting_decisions(
                    label, summary, self._nodes
                )
                for old_id in to_supersede:
                    if old_id != node_id and old_id in self._nodes:
                        old_node = self._nodes[old_id]
                        old_node.status = NodeStatus.SUPERSEDED
                        old_node.valid_until = time.time()  # closed-world timestamp
                        old_node.summary = (
                            f"Superseded by: {label[:60]}"
                            + (f" — {summary[:40]}" if summary else "")
                        )
                        # Add supersedes edge
                        self.add_edge(node_id, old_id, EdgeType.SUPERSEDES, weight=1.0)
                        logger.info(
                            f"Decision superseded: {old_node.label!r} → {label!r}"
                        )
            except Exception as e:
                logger.debug(f"Decision contradiction check failed (non-fatal): {e}")

        return node_id

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0) -> None:
        # No duplicate edges
        for e in self._edges:
            if e.source_id == source_id and e.target_id == target_id and e.type == edge_type:
                return
        self._edges.append(MemoryEdge(source_id=source_id, target_id=target_id,
                                       type=edge_type, weight=weight))

    # ── Extraction ───────────────────────────────────────────────────────────

    def _msg_hash(self, msg: dict) -> str:
        return hashlib.sha1(msg.get("content", "")[:500].encode()).hexdigest()[:16]

    def extract_from_messages(
        self,
        messages: list[dict],
        incremental: bool = True,
        extracted_data: dict | None = None,
    ) -> None:
        """
        Update graph from messages.

        Pipeline:
          1. If extracted_data is provided (from LLM/HybridExtractor) — use it directly.
          2. Otherwise run _heuristic_extract() as fallback.
        """
        if incremental:
            new_messages = [m for m in messages
                           if self._msg_hash(m) not in self._processed_hashes]
            if not new_messages:
                return
        else:
            new_messages = messages

        # Use provided data (from HybridExtractor or LLM) or fall back to heuristic
        data = extracted_data if extracted_data is not None else _heuristic_extract(new_messages)
        self._apply_extracted(data, new_messages)

        for m in new_messages:
            self._processed_hashes.add(self._msg_hash(m))

        self._persist()

    def _apply_extracted(self, data: dict, messages: list[dict]) -> None:
        """
        Apply extracted structured data to the graph.

        Edge rule: edges are created only between semantically related nodes,
        NOT by accident-of-order (previous version used task_ids[-3:] which
        linked any task to any file extracted in the same message — wrong).

        Relationship logic:
          - decision → task: only if decision label shares ≥1 meaningful word with task
          - task → file: only if file name appears in task label or vice versa
          - file → endpoint: only if endpoint label shares a path segment with file name
        """
        # Collect accepted node IDs by type for relationship inference
        goal_ids: list[str] = []
        task_ids: list[str] = []
        file_ids: list[str] = []
        decision_ids: list[str] = []

        # Goals
        for goal in data.get("goals", []):
            if goal:
                nid = self.add_node(NodeType.GOAL, goal, NodeStatus.IN_PROGRESS, importance=1.0)
                if nid:
                    goal_ids.append(nid)

        # Tasks
        status_map = {
            "completed": NodeStatus.COMPLETED,
            "in_progress": NodeStatus.IN_PROGRESS,
            "failed": NodeStatus.FAILED,
        }
        for t in data.get("tasks", []):
            label = t.get("label", "")
            if not label or len(label) < 5:
                continue
            status = status_map.get(t.get("status", "pending"), NodeStatus.PENDING)
            importance = 0.8 if status == NodeStatus.COMPLETED else 0.6
            nid = self.add_node(NodeType.TASK, label, status, importance=importance)
            if nid:
                task_ids.append(nid)
                # Tasks are part of the session goal
                for gid in goal_ids:
                    self.add_edge(nid, gid, EdgeType.PART_OF)

        # Decisions — linked to tasks that share vocabulary
        for d in data.get("decisions", []):
            label = d.get("label", "")
            if not label or len(label) < 5:
                continue
            summary = d.get("rationale", "")
            nid = self.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED,
                                summary=summary, importance=0.9)
            if nid:
                decision_ids.append(nid)
                # Link to tasks only if they share meaningful vocabulary
                decision_words = self._meaningful_words(label)
                for tid in task_ids:
                    task_node = self._nodes.get(tid)
                    if task_node:
                        task_words = self._meaningful_words(task_node.label)
                        if decision_words & task_words:
                            self.add_edge(nid, tid, EdgeType.RELATED_TO)

        # Files — linked to tasks only if file name appears in task description
        for f in data.get("files", []):
            if not f or len(f) < 3:
                continue
            nid = self.add_node(NodeType.FILE, f, NodeStatus.IN_PROGRESS, importance=0.7)
            if nid:
                file_ids.append(nid)
                file_stem = f.split("/")[-1].split(".")[0].lower()
                for tid in task_ids:
                    task_node = self._nodes.get(tid)
                    if task_node and file_stem and file_stem in task_node.label.lower():
                        self.add_edge(tid, nid, EdgeType.IMPLEMENTS)

        # Errors
        for e in data.get("errors", []):
            label = e.get("label", "")
            if not label:
                continue
            status = NodeStatus.COMPLETED if e.get("resolved") else NodeStatus.FAILED
            importance = 0.5 if e.get("resolved") else 0.9
            err_nid = self.add_node(NodeType.ERROR, label, status, importance=importance)
            if err_nid:
                # Link error to file if file name is in error description
                for fid in file_ids:
                    file_node = self._nodes.get(fid)
                    if file_node and file_node.label.split("/")[-1] in label:
                        self.add_edge(err_nid, fid, EdgeType.RELATED_TO)

        # Dependencies (no edges — standalone nodes)
        for dep in data.get("dependencies", []):
            if dep and len(dep) > 1:
                self.add_node(NodeType.DEPENDENCY, dep, NodeStatus.COMPLETED, importance=0.6)

        # Environment (no edges — standalone nodes)
        for env in data.get("environment", []):
            if env:
                self.add_node(NodeType.ENVIRONMENT, env, NodeStatus.COMPLETED, importance=0.8)

        # Endpoints — linked to files only when they share a path segment
        for ep in data.get("endpoints", []):
            if not ep:
                continue
            ep_nid = self.add_node(NodeType.ENDPOINT, ep, NodeStatus.COMPLETED, importance=0.7)
            if ep_nid:
                ep_parts = set(ep.lower().replace("/", " ").split())
                for fid in file_ids:
                    file_node = self._nodes.get(fid)
                    if file_node:
                        file_parts = self._meaningful_words(file_node.label)
                        if ep_parts & file_parts:
                            self.add_edge(fid, ep_nid, EdgeType.IMPLEMENTS)

        # Schemas
        for schema in data.get("schemas", []):
            if schema:
                self.add_node(NodeType.SCHEMA, schema, NodeStatus.COMPLETED, importance=0.7)

    _STOP_WORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "will", "would", "could", "should",
        "this", "that", "it", "we", "i", "you", "they", "use", "using",
    })

    def _meaningful_words(self, text: str) -> frozenset:
        """Extract meaningful words from text for semantic edge linking."""
        words = set(text.lower().split())
        # Remove stop words, punctuation, and very short words
        return frozenset(
            w.strip(".,!?:;()[]") for w in words
            if len(w) > 3 and w not in self._STOP_WORDS
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, task: str, top_k: int = 12) -> list[MemoryNode]:
        """Keyword + importance ranked retrieval."""
        words = set(task.lower().split())
        scored: list[tuple[float, MemoryNode]] = []

        for node in self._nodes.values():
            if node._evicted:
                continue
            # Skip archived/superseded nodes in retrieval — they're historical noise
            if node.status in (NodeStatus.ARCHIVED, NodeStatus.SUPERSEDED,
                               NodeStatus.MODIFIED, NodeStatus.INVALIDATED):
                continue
            node_words = set(node.label.lower().split())
            overlap = len(words & node_words) / max(1, len(words))
            # Boost by importance and recency
            recency = 1.0 / (1.0 + node.age_days() * 0.1)
            score = overlap * 0.6 + node.importance * 0.3 + recency * 0.1
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def query_at_time(self, task: str, at_time: float, top_k: int = 12) -> list[MemoryNode]:
        """
        Return nodes that were ACTIVE at a specific point in time.
        Enables: "What did we decide last Tuesday?"
        """
        results = self.query(task, top_k=top_k * 2)
        return [n for n in results if n.is_valid_at(at_time)][:top_k]

    # ── Prune ────────────────────────────────────────────────────────────────

    def prune(
        self,
        max_nodes: int = 200,
        max_age_days: float = 60.0,
    ) -> int:
        """Remove low-importance, old, completed nodes. Preserve decisions, envs, goals."""
        preserve_types = {NodeType.GOAL, NodeType.SCHEMA}
        # Decisions are kept even when old — history matters
        # But ARCHIVED/SUPERSEDED decisions can be pruned after max_age_days
        cutoff = time.time() - max_age_days * 86400
        # Superseded decisions expire faster (30 days default)
        superseded_cutoff = time.time() - min(max_age_days, 30) * 86400
        candidates: list[tuple[float, str]] = []

        for nid, node in self._nodes.items():
            if node.type in preserve_types:
                continue
            # ACTIVE decisions and environments: keep unless very old
            if node.type in (NodeType.DECISION, NodeType.ENVIRONMENT):
                if node.status == NodeStatus.COMPLETED and node.updated_at < cutoff:
                    score = node.importance * 0.1  # low score = prune first
                    candidates.append((score, nid))
                elif node.status in (NodeStatus.SUPERSEDED, NodeStatus.MODIFIED,
                                     NodeStatus.ARCHIVED) and node.updated_at < superseded_cutoff:
                    candidates.append((0.0, nid))  # prune superseded decisions after 30d
                continue
            # All other nodes: prune if old and completed
            if node.status in (NodeStatus.COMPLETED, NodeStatus.FAILED,
                               NodeStatus.ARCHIVED) and node.updated_at < cutoff:
                score = node.importance * (node.updated_at / (time.time() + 1))
                candidates.append((score, nid))

        if len(self._nodes) <= max_nodes:
            return 0

        candidates.sort()
        to_prune = len(self._nodes) - max_nodes
        pruned = 0

        for _, nid in candidates[:to_prune]:
            del self._nodes[nid]
            self._edges = [e for e in self._edges
                           if e.source_id != nid and e.target_id != nid]
            pruned += 1

        if pruned:
            self._persist()
            logger.info(f"Graph pruned {pruned} nodes for session {self.session_id}")

        return pruned

    # ── Context block ────────────────────────────────────────────────────────

    def to_context_block(self, token_budget: int = 400) -> str:
        """
        Build tiered resume context block.
        Respects token_budget — truncates lower-priority sections first.
        """
        sections: list[str] = []

        goals = [n for n in self._nodes.values()
                 if n.type == NodeType.GOAL and not n._evicted]
        if goals:
            sections.append("Goal: " + " | ".join(g.label for g in goals[:2]))

        open_tasks = [n for n in self._nodes.values()
                      if n.type == NodeType.TASK
                      and n.status in (NodeStatus.PENDING, NodeStatus.IN_PROGRESS)
                      and not n._evicted]
        open_tasks.sort(key=lambda x: x.importance, reverse=True)
        if open_tasks:
            sections.append("In progress: " + " | ".join(t.label for t in open_tasks[:5]))

        done = [n for n in self._nodes.values()
                if n.type == NodeType.TASK
                and n.status == NodeStatus.COMPLETED
                and not n._evicted]
        done.sort(key=lambda x: x.updated_at, reverse=True)
        if done:
            sections.append("Done: " + " | ".join(t.label for t in done[:8]))

        # Active decisions — shown prominently
        decisions = [
            n for n in self._nodes.values()
            if n.type == NodeType.DECISION
            and n.status == NodeStatus.COMPLETED   # GREEN — active only in resume
            and not n._evicted
        ]
        decisions.sort(key=lambda x: x.importance, reverse=True)
        if decisions:
            parts = []
            for d in decisions[:6]:
                entry = d.label
                if d.summary and "Superseded by" not in d.summary:
                    entry += f" ({d.summary[:60]})"
                parts.append(entry)
            sections.append("Decided: " + " | ".join(parts))

        # Superseded decisions — shown briefly for context (max 2, recently changed only)
        # YELLOW: recently superseded decisions — shown briefly, fade after 7 days
        # RED: invalidated decisions — always show as warning
        superseded = [
            n for n in self._nodes.values()
            if n.type == NodeType.DECISION
            and n.status in (NodeStatus.SUPERSEDED, NodeStatus.MODIFIED)
            and n.age_days() < 7
            and not n._evicted
        ]
        invalidated = [
            n for n in self._nodes.values()
            if n.type == NodeType.DECISION
            and n.status == NodeStatus.INVALIDATED
            and not n._evicted
        ]
        superseded.sort(key=lambda x: x.updated_at, reverse=True)
        if superseded[:2]:
            labels = [f"~~{n.label[:40]}~~" for n in superseded[:2]]
            sections.append("Changed: " + " | ".join(labels))
        if invalidated[:2]:
            labels = [f"[INVALID] {n.label[:40]}" for n in invalidated[:2]]
            sections.append("Invalidated: " + " | ".join(labels))

        files = [n for n in self._nodes.values()
                 if n.type == NodeType.FILE and not n._evicted]
        if files:
            sections.append("Files: " + ", ".join(f.label for f in files[:10]))

        env_nodes = [n for n in self._nodes.values()
                     if n.type == NodeType.ENVIRONMENT and not n._evicted]
        if env_nodes:
            sections.append("Env: " + ", ".join(e.label for e in env_nodes[:6]))

        errors = [n for n in self._nodes.values()
                  if n.type == NodeType.ERROR
                  and n.status == NodeStatus.FAILED
                  and not n._evicted]
        if errors:
            sections.append("Open issues: " + " | ".join(e.label for e in errors[:3]))

        block = "\n".join(sections)

        # Trim to budget (rough token estimate)
        from tokenmizer.core.tokenizer import count_tokens
        while count_tokens(block) > token_budget and sections:
            sections.pop()
            block = "\n".join(sections)

        return block

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        from tokenmizer.core.dto import GraphStatsDTO
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        confidences: list[float] = []
        for n in self._nodes.values():
            by_type[n.type.value] = by_type.get(n.type.value, 0) + 1
            by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
            confidences.append(n.confidence)
        avg_confidence = round(sum(confidences) / max(1, len(confidences)), 3)
        dto = GraphStatsDTO(
            session_id=self.session_id,
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            by_type=by_type,
            by_status=by_status,
            processed_messages=len(self._processed_hashes),
            avg_confidence=avg_confidence,
        )
        # Return as dict for JSON serialization — DTO used for type safety at boundary
        from dataclasses import asdict
        return asdict(dto)
