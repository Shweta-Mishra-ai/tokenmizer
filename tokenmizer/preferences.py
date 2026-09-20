"""
The habits that outlive a session.

A session graph remembers what you decided *about this project*. It does
not, and should not, remember that you want short answers — that is true
of you, not of the repository, and it has to survive starting a new
session in a different directory.

This is the store for that. It is deliberately narrow, because the
failure mode of a preference memory is not forgetting, it is remembering
something that was never a preference and putting it in every prompt you
send for the rest of the year. Three guards:

  - a sentence must match one of the signal patterns AND none of the
    exclusions (secrets, env vars, project-specific asides)
  - the store is keyed by PRINCIPAL, so one caller's preferences never
    reach another's prompt (the reason the earlier in-memory version
    could not be wired up — it was process-global)
  - it is off unless `preferences.enabled` is set, and capped at a
    handful of short lines, so the cost to a prompt is bounded and an
    operator opts in with their eyes open

Previously this class lived in `semantic_cache/cache.py`, instantiated by
`SemanticCache` and never written to by anything: `save()` had no callers
anywhere, so `/api/cache/stats` used to report a preference field that
was permanently the empty string. It is its own module now because it has
nothing to do with caching responses.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# What a habit sounds like. Kept from the original implementation, which
# was the one part of it that had been thought through.
_SIGNALS = [
    re.compile(r'\b(?:always|prefer|like|want|use)\b.{3,60}\b'
               r'(?:format|style|language|approach|pattern|framework)\b', re.I),
    # `.{0,40}` and not `.{2,40}`: "keep it brief" has one space between
    # the two halves, so the most natural phrasing of the most common
    # preference was the one thing this could not match.
    re.compile(r'\b(?:be|keep it|make it|stay)\b.{0,40}?\b'
               r'(?:brief|concise|short|simple|direct|formal|casual)\b', re.I),
    re.compile(r'\b(?:my|our)\b.{0,30}?\b'
               r'(?:preference|style|convention|standard|default)\b.{0,60}?\b(?:is|are)\b',
               re.I),
    re.compile(r'\b(?:i\s+(?:prefer|like|use|always|hate|avoid))\b.{3,80}', re.I),
    re.compile(r'\bremember\s+(?:that\s+)?(?:i|my|we)\b.{3,100}', re.I),
]

# Never a preference, whatever else it matches.
_EXCLUSIONS = [
    re.compile(r'\b(?:password|secret|key|token|credential)\b', re.I),
    re.compile(r'\b(?:project|client|customer|company)\b.{3,30}\b'
               r'(?:specific|only|internal)\b', re.I),
    re.compile(r'[A-Z_]{5,}\s*=\s*\S'),          # env var
    re.compile(r'sk-|ghp_|AIza'),                 # API key shapes
    # A complaint is not a standing instruction. "I hate this bug" and "I
    # hate that the tests are flaky" are about today; the signal list
    # above cannot tell them from "I hate verbose output" on its own.
    re.compile(r'\bi\s+hate\s+(?:this|that|it|these|those|when)\b', re.I),
]

# Words that carry no identity, so a key built from them would collide
# with every other preference.
_STOP = {
    "the", "a", "an", "and", "or", "but", "for", "with", "that", "this",
    "please", "always", "never", "prefer", "like", "want", "use", "using",
    "keep", "make", "stay", "remember", "should", "would", "could", "our",
    "my", "me", "you", "your", "i", "we", "is", "are", "be", "it", "to",
    "in", "on", "of", "when", "then", "than", "not", "dont", "do",
}

_MAX_VALUE_CHARS = 160
_MAX_PER_PRINCIPAL = 12

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    principal  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (principal, key)
)
"""


def looks_like_a_preference(text: str) -> bool:
    """True if `text` states a habit that should outlive this session."""
    if not text or len(text) < 8:
        return False
    for block in _EXCLUSIONS:
        if block.search(text):
            return False
    return any(p.search(text) for p in _SIGNALS)


def preference_key(text: str) -> str:
    """A stable slug, so restating a preference UPDATES it.

    "I prefer TypeScript" and later "I prefer TypeScript strict mode"
    must not become two lines that say nearly the same thing in every
    prompt — the whole cost of this feature is the prompt it writes.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
             if len(w) > 2 and w not in _STOP]
    return "-".join(words[:3]) or "general"


class PreferenceStore:
    """Per-principal, SQLite-backed, shared by every worker and session.

    Every method is best-effort: a preference is a nicety, and nothing
    here may cost a caller their answer. Failures are logged, never
    raised.
    """

    def __init__(self, storage_dir: str = "./checkpoints"):
        self._db_path = Path(storage_dir) / "preferences.db"
        self.available = self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0,
                               check_same_thread=False)
        # Losing the race to set the journal mode is not a failure: it is a
        # property of the file and persists once any process has set it.
        # Same fix as the shared rate limiter, for the same startup race.
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL"):
            try:
                conn.execute(f"PRAGMA {pragma}")
            except sqlite3.OperationalError:
                pass
        return conn

    def _init(self) -> bool:
        """Create the table, retrying while the file is merely busy.

        Every worker runs `CREATE TABLE IF NOT EXISTS` at startup, so on a
        cold start with several workers some of them meet a locked
        database. Treating the first failure as final disabled the store
        for that worker's whole life.
        """
        delay = 0.05
        last: Exception | None = None
        for _ in range(6):
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
                return True
            except sqlite3.OperationalError as e:
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 0.8)
            except Exception as e:
                last = e
                break
        logger.warning(
            "Preference store unavailable (%s) — preferences will not be "
            "remembered across sessions; nothing else is affected.", last,
        )
        return False

    def observe(self, principal: str, text: str) -> str | None:
        """Record `text` if it states a preference. Returns the key, or None.

        The caller passes a user turn; this decides whether it was about
        the person or about the work.
        """
        if not self.available or not looks_like_a_preference(text):
            return None
        key = preference_key(text)
        value = " ".join(text.split())[:_MAX_VALUE_CHARS]
        # Restating a preference must UPDATE it, and the key alone cannot
        # tell: "I prefer TypeScript for everything" and "I prefer
        # TypeScript with strict mode on" have different first-three
        # words. Overlap can, and it is the same subsumption rule the
        # graph uses to fold two labels for one fact.
        existing = self._overlapping(principal, text)
        if existing:
            key = existing
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO preferences (principal, key, value, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(principal, key) DO UPDATE SET "
                    "value = excluded.value, updated_at = excluded.updated_at",
                    (principal, key, value, time.time()),
                )
                # Oldest out first: a preference nobody has restated in a
                # year is less likely to be current than one from today.
                conn.execute(
                    "DELETE FROM preferences WHERE principal = ? AND key NOT IN ("
                    "  SELECT key FROM preferences WHERE principal = ? "
                    "  ORDER BY updated_at DESC LIMIT ?)",
                    (principal, principal, _MAX_PER_PRINCIPAL),
                )
        except Exception as e:
            logger.warning("Could not save a preference for %r: %s", principal, e)
            return None
        return key

    @staticmethod
    def _significant(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", text.lower())
                if len(w) > 2 and w not in _STOP]

    def _overlapping(self, principal: str, text: str) -> str | None:
        """The key of an existing preference this one is a restatement of.

        Two rules, because either alone gets a case wrong:

          same topic word — the first significant word is what a
            preference is ABOUT, so "I prefer TypeScript for everything"
            and "I prefer TypeScript with strict mode" are one preference
            stated twice, although they share only one word in three.
          heavy overlap — "keep it brief" and "please keep answers brief
            and skip the preamble" share their topic and most of their
            content.

        Overlap alone would merge "I prefer dark mode" with "I prefer
        strict mode" at any threshold low enough to catch the first case.
        """
        words = self._significant(text)
        if not words:
            return None
        head, wordset = words[0], set(words)
        for key, value in self.all(principal):
            other_words = self._significant(value)
            if not other_words:
                continue
            if other_words[0] == head:
                return key
            other = set(other_words)
            smaller = wordset if len(wordset) <= len(other) else other
            if len(wordset & other) / len(smaller) >= 0.6:
                return key
        return None

    def all(self, principal: str) -> list[tuple[str, str]]:
        """This principal's preferences, most recently stated first."""
        if not self.available:
            return []
        try:
            with self._connect() as conn:
                return [
                    (row[0], row[1]) for row in conn.execute(
                        "SELECT key, value FROM preferences WHERE principal = ? "
                        "ORDER BY updated_at DESC", (principal,))
                ]
        except Exception as e:
            logger.warning("Could not read preferences for %r: %s", principal, e)
            return []

    def context(self, principal: str, max_items: int = 4,
                max_chars: int = 400) -> str:
        """The block to put in the system prompt, or "" when there is
        nothing to say — which must stay the common case."""
        items = self.all(principal)[:max_items]
        if not items:
            return ""
        lines = ["[How this person likes to work]"]
        used = len(lines[0])
        for _key, value in items:
            if used + len(value) + 3 > max_chars:
                break
            lines.append(f"- {value}")
            used += len(value) + 3
        return "\n".join(lines) if len(lines) > 1 else ""

    def forget(self, principal: str, key: str | None = None) -> int:
        """Drop one preference, or all of this principal's.

        A memory with no way to say "stop remembering that" is a liability,
        not a feature.
        """
        if not self.available:
            return 0
        try:
            with self._connect() as conn:
                if key:
                    cur = conn.execute(
                        "DELETE FROM preferences WHERE principal = ? AND key = ?",
                        (principal, key))
                else:
                    cur = conn.execute(
                        "DELETE FROM preferences WHERE principal = ?", (principal,))
                return cur.rowcount or 0
        except Exception as e:
            logger.warning("Could not forget preferences for %r: %s", principal, e)
            return 0
