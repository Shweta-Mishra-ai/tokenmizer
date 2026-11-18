"""
Turn archive — the verbatim record of what windowing stops sending.

WHY THIS EXISTS
---------------
Windowing replaces old turns with a context block built from the graph.
The graph holds what the extractor has a node type for: goals, decisions,
tasks, files, errors. Everything else in those turns — a figure, an exact
error string, a snippet, a constraint stated in passing, the reason
behind a choice that was never phrased as one — left the session for
good at the first cut. Nothing could bring it back, because nothing kept
it.

This keeps it. Every turn that is no longer in the request is stored as it
was received (already redacted at ingestion), once, and searched on later
turns. When a question is clearly about something only an archived turn
says, the matching passage is quoted back inside a fenced block, within a
token budget. The archive is also readable through the API, so a person
can see exactly what was said even when no model is asked.

WHAT IT IS NOT
--------------
Not a second context window: at most one excerpt per turn, and only when
the question shares distinctive words with an archived turn. Keyword
matching (SQLite FTS5, or a LIKE fallback where FTS5 is not compiled in)
will miss a paraphrase; the graph and its semantic retrieval cover that
side. It errs towards saying nothing, because an irrelevant excerpt costs
tokens on every turn that triggers it.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

# Per turn. A pasted log can be megabytes; the archive keeps the head,
# which is where the question about it usually points, and says so.
_MAX_TURN_CHARS = 200_000

# Words never worth matching on.
_STOP = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its me my no not of on or our should so than that the
their them then there these they this to us was we were what when where which
who why will with would you your yes ok okay please thanks just also about
again more some any all one two use using used make made get got need want
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*[A-Za-z0-9]|[A-Za-z0-9]")


def _terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for w in _WORD.findall(text.lower()):
        if len(w) > 2 and w not in _STOP:
            seen.setdefault(w)
    return list(seen)


def _distinctive(word: str) -> bool:
    """A word that pins a passage on its own: a number, a version, a path,
    an identifier. Ordinary words need company."""
    return any(c.isdigit() for c in word) or any(c in word for c in "._/-")


def _render(message: dict) -> str:
    """A message as the prose a person would read, tool calls included."""
    from tokenmizer.graph_memory.helpers import _content_to_text
    text = _content_to_text(message.get("content"))
    calls = message.get("tool_calls") or []
    rendered = []
    for c in calls:
        fn = (c.get("function") or {}) if isinstance(c, dict) else {}
        rendered.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    if rendered:
        text = (text + "\n" if text else "") + "[called tools: " + "; ".join(rendered) + "]"
    return text


class TurnArchive:
    """SQLite-backed, per-session, append-only store of dropped turns."""

    _KNOWN_MAX_SESSIONS = 2_000

    def __init__(self, storage_dir: str):
        self._db_path = Path(storage_dir) / "turn_archive.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # session -> hashes already stored, so a request whose windowed-out
        # prefix is unchanged writes nothing.
        self._known: OrderedDict[str, set[str]] = OrderedDict()
        self._fts = False
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            conn.close()
            raise
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT NOT NULL,
                    msg_hash   TEXT NOT NULL,
                    seq        INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, msg_hash)
                )""")
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5("
                    "content, session_id UNINDEXED, msg_hash UNINDEXED)")
                self._fts = True
            except sqlite3.OperationalError:
                logger.info("SQLite has no FTS5 here — the turn archive falls back "
                            "to LIKE matching, which is slower on long sessions")

    # ── write ──────────────────────────────────────────────────────────────

    def add_dropped(self, session_id: str, raw_messages: list[dict],
                    sent_messages: list[dict]) -> int:
        """Store every message in `raw_messages` that is not in what was
        sent. Returns how many new turns were written."""
        from tokenmizer.graph_memory.helpers import _content_to_text
        sent = {_content_to_text(m.get("content")) for m in sent_messages}
        with self._lock:
            known = self._known.get(session_id)
        if known is None:
            known = self._load_known(session_id)
        rows = []
        for seq, m in enumerate(raw_messages):
            if m.get("role") == "system":
                continue
            text = _render(m)
            if not text.strip() or _content_to_text(m.get("content")) in sent:
                continue
            h = hashlib.sha1(f"{m.get('role')}\x1f{text}".encode(
                "utf-8", "surrogatepass")).hexdigest()[:20]
            if h in known:
                continue
            if len(text) > _MAX_TURN_CHARS:
                text = text[:_MAX_TURN_CHARS] + f"\n[... {len(text) - _MAX_TURN_CHARS} characters not archived]"
            rows.append((session_id, h, seq, str(m.get("role", "")), text, time.time()))
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany("INSERT OR IGNORE INTO turns VALUES (?,?,?,?,?,?)", rows)
            if self._fts:
                conn.executemany(
                    "INSERT INTO turns_fts (content, session_id, msg_hash) VALUES (?,?,?)",
                    [(r[4], r[0], r[1]) for r in rows])
        known.update(r[1] for r in rows)
        return len(rows)

    def _load_known(self, session_id: str) -> set[str]:
        with self._connect() as conn:
            known = {r[0] for r in conn.execute(
                "SELECT msg_hash FROM turns WHERE session_id=?", (session_id,))}
        with self._lock:
            self._known[session_id] = known
            self._known.move_to_end(session_id)
            while len(self._known) > self._KNOWN_MAX_SESSIONS:
                self._known.popitem(last=False)
        return known

    def delete_session(self, session_id: str) -> int:
        with self._connect() as conn:
            n = conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,)).rowcount
            if self._fts:
                conn.execute("DELETE FROM turns_fts WHERE session_id=?", (session_id,))
        with self._lock:
            self._known.pop(session_id, None)
        return n

    def count(self, session_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM turns WHERE session_id=?",
                                (session_id,)).fetchone()[0]

    # ── read ───────────────────────────────────────────────────────────────

    def search(self, session_id: str, query: str, limit: int = 5) -> list[dict]:
        """Archived turns of `session_id` ranked by how many of the query's
        terms they contain, best first. Each: seq, role, content, matched."""
        terms = _terms(query)
        if not terms:
            return []
        with self._connect() as conn:
            if self._fts:
                match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms[:16])
                try:
                    rows = conn.execute(
                        "SELECT t.seq, t.role, t.content FROM turns_fts f "
                        "JOIN turns t ON t.session_id=f.session_id AND t.msg_hash=f.msg_hash "
                        "WHERE turns_fts MATCH ? AND f.session_id=? ORDER BY bm25(turns_fts) LIMIT 40",
                        (match, session_id)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            else:
                like = " OR ".join("lower(content) LIKE ?" for _ in terms[:16])
                rows = conn.execute(
                    f"SELECT seq, role, content FROM turns WHERE session_id=? AND ({like}) "
                    f"ORDER BY seq DESC LIMIT 40",
                    (session_id, *[f"%{t}%" for t in terms[:16]])).fetchall()
        scored = []
        for seq, role, content in rows:
            low = content.lower()
            matched = [t for t in terms if t in low]
            if matched:
                scored.append((len(matched), seq, role, content, matched))
        scored.sort(key=lambda r: (r[0], r[1]), reverse=True)
        return [{"seq": seq, "role": role, "content": content, "matched": matched}
                for _, seq, role, content, matched in scored[:limit]]

    def recall(self, session_id: str, query: str, budget_tokens: int,
               model: str = "gpt-4o", sent_messages: list[dict] | None = None) -> str:
        """A verbatim excerpt from the archived turn that best answers
        `query`, within `budget_tokens`, or "" when no turn is a clear
        match.

        Clear means: the turn contains a distinctive term of the question
        (a number, a path, an identifier) or at least two of its ordinary
        terms. One shared common word is how an irrelevant excerpt gets
        billed on every turn.
        """
        from tokenmizer.core.tokenizer import count_tokens
        from tokenmizer.graph_memory.helpers import _content_to_text

        hits = self.search(session_id, query, limit=3)
        payload = "\n".join(_content_to_text(m.get("content")) for m in (sent_messages or []))
        for hit in hits:
            matched = hit["matched"]
            if not (any(_distinctive(t) for t in matched) or len(matched) >= 2):
                continue
            excerpt = _excerpt(hit["content"], matched, budget_tokens, model)
            if not excerpt or excerpt in payload:
                continue
            header = f"[{hit['role']} turn {hit['seq'] + 1}, no longer in the conversation above]"
            text = f"{header}\n{excerpt}"
            if count_tokens(text, model) <= budget_tokens + 20:
                return text
        return ""


def _excerpt(content: str, matched: list[str], budget_tokens: int, model: str) -> str:
    """The lines around the densest cluster of matched terms, grown outward
    until the budget is spent. Whole lines only: half a line of code is
    worse than none."""
    from tokenmizer.core.tokenizer import count_tokens
    lines = content.split("\n")
    if count_tokens(content, model) <= budget_tokens:
        return content
    scores = [sum(1 for t in matched if t in line.lower()) for line in lines]
    best = max(range(len(lines)), key=lambda i: scores[i])
    lo = hi = best
    used = count_tokens(lines[best], model)
    if used > budget_tokens:
        # One enormous line: cut it on characters around the first match.
        line = lines[best]
        at = max(0, min(line.lower().find(t) for t in matched if t in line.lower()) - 200)
        return "…" + line[at:at + budget_tokens * 3] + "…"
    while True:
        grew = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(lines) and not (lo <= j <= hi):
                cost = count_tokens(lines[j], model) + 1
                if used + cost <= budget_tokens:
                    used += cost
                    lo, hi = min(lo, j), max(hi, j)
                    grew = True
        if not grew:
            break
    out = "\n".join(lines[lo:hi + 1])
    return ("…\n" if lo > 0 else "") + out + ("\n…" if hi < len(lines) - 1 else "")
