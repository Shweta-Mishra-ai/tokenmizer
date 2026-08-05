"""
Semantic Cache
==============
Three-layer lookup:
  1. Exact hash match (O(1))
  2. Embedding similarity (cosine ≥ threshold, optional)
  3. Cache miss

Session isolation:
  Sensitive prompts (API keys, DB URLs, project data) are scoped
  to session_id — never shared cross-session.
  Generic prompts (how-to questions, explanations) are shared.

LRU eviction, TTL expiry, eviction metrics included.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from tokenmizer.core.tokenizer import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    key: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    created_at: float
    hit_count: int = 0
    _evicted: bool = False
    # Scope this entry was stored under: "__shared__" (readable by any
    # session) or a specific session_id / "__private__" (readable only by
    # that same session). Recorded on the entry itself — not just implied
    # by which key it was filed under — so the semantic-similarity lookup
    # (which iterates ALL entries, not just one key) can enforce the same
    # scoping rule the exact-match lookup does. Without it, a near-miss
    # query can return another session's private entry purely because
    # cosine similarity cleared the threshold, bypassing scope entirely.
    scope: str = "__shared__"

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds

    def touch(self) -> None:
        self.hit_count += 1


# ── Embedding engine (lazy, optional) ────────────────────────────────────────

class EmbeddingEngine:
    """Sentence-transformers based embeddings. Falls back to None if not installed."""

    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self):
        self._model = None
        self._initialized = False

    def _load(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer loaded for semantic cache")
        except ImportError:
            logger.info("sentence-transformers not installed "
                        "— semantic cache uses exact match only")

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def embed(self, text: str):
        self._load()
        if self._model is None:
            return None
        return self._model.encode(text[:1000], normalize_embeddings=True)

    @staticmethod
    def cosine(a, b) -> float:
        if a is None or b is None:
            return 0.0
        import numpy as np
        return float(np.dot(a, b))

    @classmethod
    def get(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = EmbeddingEngine()
        return cls._instance


# ── Cache ────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Three-layer cache:
    1. Exact match (hash lookup) — O(1)
    2. Semantic similarity (embedding cosine) — requires sentence-transformers
    3. Miss

    LRU eviction: when full, removes least-recently-used entry.
    """

    def __init__(
        self,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_size: int = 10_000,
        share_scope: str = "session",
    ):
        """
        share_scope:
          "session" (default) — every prompt is scoped to its session_id
            (or "__private__" if none given), regardless of whether it
            looks sensitive. Nothing is ever shared across sessions unless
            explicitly opted in.
          "shared" — non-sensitive prompts (per _is_session_sensitive)
            are shared globally across sessions; sensitive-looking ones
            stay session-scoped regardless. The sensitivity gate is a
            floor share_scope cannot override.
        """
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.share_scope = share_scope
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: dict[str, object] = {}  # key → embedding
        self._embedder = EmbeddingEngine.get()
        self._eviction_count = 0
        self._hit_exact = 0
        self._hit_semantic = 0
        self._miss = 0
        self._preference_store = PreferenceStore()

    def _key(self, prompt: str, scope: str = "__shared__") -> str:
        """Include scope in key — session-specific entries don't collide across sessions."""
        data = f"{scope}:{prompt}"
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._exact:
            return
        lru_key, _ = self._exact.popitem(last=False)
        self._embeddings.pop(lru_key, None)
        self._eviction_count += 1

    def get(self, prompt: str, session_id: str = "") -> Optional[CacheEntry]:
        """
        Look up cache.
        Checks session-scoped key first (using session_id, or "__private__"
        when none was given — mirrors set()'s own default scope so a
        caller that never passes session_id is still self-consistent),
        then the shared key. Same-session hits always found; cross-session
        hits only for entries explicitly stored under share_scope="shared".
        """
        # Try session-scoped (or private, if no session_id) key first
        session_key = self._key(prompt, session_id or "__private__")
        if session_key in self._exact:
            entry = self._exact[session_key]
            if not entry.is_expired(self.ttl_seconds):
                self._exact.move_to_end(session_key)
                entry.touch()
                self._hit_exact += 1
                return entry

        # Try shared key
        key = self._key(prompt, "__shared__")

        # 1. Exact match
        if key in self._exact:
            entry = self._exact[key]
            if entry.is_expired(self.ttl_seconds):
                del self._exact[key]
                self._embeddings.pop(key, None)
                self._miss += 1
                return None
            self._exact.move_to_end(key)  # mark as recently used
            entry.touch()
            self._hit_exact += 1
            return entry

        # 2. Semantic match
        #
        # Only entries visible to THIS caller are eligible: shared ones,
        # or ones scoped to this exact session_id. Scanning every entry
        # regardless of scope would let cosine similarity hand a caller
        # another session's private entry.
        if self._embedder.available:
            query_emb = self._embedder.embed(prompt)
            best_score = 0.0
            best_key = None

            for k, entry in self._exact.items():
                if entry.is_expired(self.ttl_seconds):
                    continue
                if entry.scope != "__shared__" and entry.scope != (session_id or "__private__"):
                    continue
                emb = self._embeddings.get(k)
                if emb is None:
                    continue
                score = EmbeddingEngine.cosine(query_emb, emb)
                if score > best_score:
                    best_score = score
                    best_key = k

            if best_key and best_score >= self.threshold:
                entry = self._exact[best_key]
                self._exact.move_to_end(best_key)
                entry.touch()
                self._hit_semantic += 1
                return entry

        self._miss += 1
        return None

    # Patterns that indicate session-specific content — never cross-session cache
    _SENSITIVE_PATTERNS = [
        re.compile(r'sk-ant-|sk-proj-|sk-[A-Za-z0-9]{20,}|ghp_|AIza|Bearer\s', re.I),
        re.compile(r'\b(my|our)\s+(name|email|password|key|secret|token|api[_\s]key)\b', re.I),
        re.compile(r'(?:DATABASE_URL|REDIS_URL|MONGO_URL|postgres://|mysql://|redis://)', re.I),
        re.compile(r'[A-Z_]{4,}\s*=\s*\S{4,}'),
        re.compile(r'\b(project|company|client|internal|private)\b.*?\b(config|secret|key|token)\b', re.I | re.S),
    ]

    def _is_session_sensitive(self, prompt: str) -> bool:
        """True if prompt contains session-specific content — never share cross-session."""
        for pat in self._SENSITIVE_PATTERNS:
            if pat.search(prompt):
                return True
        # Long prompts with code/data are likely session-specific
        if len(prompt) > 800 and ("\n" in prompt or "```" in prompt):
            return True
        return False

    def set(
        self,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        session_id: str = "",
    ) -> None:
        """
        Store a cache entry.

        Scoping rules (safe by default):
        - Default (`share_scope="session"`): EVERY prompt is scoped to
          session_id (or "__private__" if none given), sensitive or not.
          Nothing is shared across sessions unless explicitly opted in.
        - Opt-in (`share_scope="shared"`): non-sensitive prompts are
          shared globally, exactly like the old default behavior — but
          the sensitivity gate is a floor this can never override, so a
          prompt that looks like it contains secrets/PII/business-
          specific content stays session-scoped regardless.

        A misclassified-as-"not sensitive" prompt under share_scope="session"
        is still safe — it's session-scoped either way. Under
        share_scope="shared" a misclassification could still leak, which is
        exactly why "shared" is opt-in rather than the default: the
        five-regex sensitivity heuristic cannot enumerate everything that
        might be confidential to someone else's business.
        """
        is_sensitive = self._is_session_sensitive(prompt)

        if self.share_scope == "shared" and not is_sensitive:
            # Explicit opt-in AND not flagged sensitive: shared globally,
            # so any session's get() can find it.
            scope = "__shared__"
        else:
            # Default, or sensitive content regardless of share_scope:
            # scoped to this session only, never shared.
            scope = session_id or "__private__"

        # Scoped key: sensitive/session-only responses keyed by session,
        # shared ones keyed by content alone.
        key = self._key(prompt, scope)

        # Evict if at capacity
        while len(self._exact) >= self.max_size:
            self._evict_lru()

        entry = CacheEntry(
            key=key,
            prompt=prompt[:500],
            response=response,
            input_tokens=input_tokens or count_tokens(prompt),
            output_tokens=output_tokens or count_tokens(response),
            created_at=time.time(),
            scope=scope,
        )
        self._exact[key] = entry
        self._exact.move_to_end(key)

        # Store embedding for semantic lookup
        if self._embedder.available:
            emb = self._embedder.embed(prompt)
            if emb is not None:
                self._embeddings[key] = emb

    def invalidate(self, prompt: str, session_id: str = "") -> int:
        """Remove cached entries for `prompt`. Returns how many were removed.

        Note the scope trap: set() files entries under the session's
        scope by default, so building a single "__shared__" key here
        would match nothing and silently remove nothing.

        With a session_id, the session-scoped entry and the shared one
        are both removed. Without one, every scope holding this prompt is
        removed — "invalidate this prompt" should mean it, and a caller
        who cannot name the scope still needs the stale answer gone.
        """
        removed = 0
        if session_id:
            candidates = [self._key(prompt, session_id), self._key(prompt, "__shared__")]
        else:
            # Scope is part of the key hash and can't be reversed, so
            # match on the stored prompt instead. Entries store a 500-char
            # prefix, so compare against the same prefix.
            needle = prompt[:500]
            candidates = [k for k, e in self._exact.items() if e.prompt == needle]
        for key in candidates:
            if self._exact.pop(key, None) is not None:
                removed += 1
            self._embeddings.pop(key, None)
        return removed

    def clear(self) -> None:
        self._exact.clear()
        self._embeddings.clear()

    def stats(self) -> dict:
        total = self._hit_exact + self._hit_semantic + self._miss
        hit_rate = (self._hit_exact + self._hit_semantic) / max(1, total)
        return {
            "entries": len(self._exact),
            "max_size": self.max_size,
            "utilization_pct": round(len(self._exact) / self.max_size * 100, 1),
            "evictions": self._eviction_count,
            "hit_rate": round(hit_rate, 3),
            "hit_exact": self._hit_exact,
            "hit_semantic": self._hit_semantic,
            "miss": self._miss,
            "semantic_available": self._embedder.available,
        }


# ── Preference / Habit Store ──────────────────────────────────────────────────

class PreferenceStore:
    """
    Stores user habits and preferences ONLY — never session content.

    Your thinking (correct):
      Cache should NOT save project-specific content across sessions.
      It SHOULD remember things like:
        - "I prefer concise answers"
        - "Always use TypeScript"
        - "Response format: bullet points"
        - "My timezone is UTC+5:30"

    This is separate from SemanticCache (which caches LLM responses).
    PreferenceStore caches USER PREFERENCES that apply across all sessions.

    What it NEVER stores:
        - Passwords, API keys, secrets
        - Project-specific data
        - Code from a specific session
        - Anything that looks like PII

    What it STORES:
        - Communication preferences ("be brief", "use examples")
        - Technical preferences ("TypeScript", "snake_case", "pytest")
        - Format preferences ("bullet points", "numbered lists")
        - Style preferences ("formal", "casual")
    """

    # Patterns that indicate a preference/habit worth remembering
    _PREFERENCE_SIGNALS = [
        re.compile(r'\b(?:always|prefer|like|want|use)\b.{3,60}\b(?:format|style|language|approach|pattern|framework)\b', re.I),
        re.compile(r'\b(?:be|keep it|make it|stay)\b.{2,40}\b(?:brief|concise|short|simple|direct|formal|casual)\b', re.I),
        re.compile(r'\b(?:my|our)\b.{2,30}\b(?:preference|style|convention|standard|default)\b.{2,60}(?:is|are)\b', re.I),
        re.compile(r'\b(?:i\s+(?:prefer|like|use|always|hate|avoid))\b.{3,80}', re.I),
        re.compile(r'\bremember\s+(?:that\s+)?(?:i|my|we)\b.{3,100}', re.I),
    ]

    # NEVER treat these as preferences (too specific / sensitive)
    _NOT_PREFERENCE = [
        re.compile(r'\b(?:password|secret|key|token|credential)\b', re.I),
        re.compile(r'\b(?:project|client|customer|company)\b.{3,30}\b(?:specific|only|internal)\b', re.I),
        re.compile(r'[A-Z_]{5,}\s*=\s*\S'),  # env var
        re.compile(r'sk-|ghp_|AIza'),          # API key patterns
    ]

    def __init__(self):
        self._prefs: dict[str, str] = {}   # key → preference text

    def is_preference(self, text: str) -> bool:
        """True if text expresses a habit or preference worth remembering."""
        for block in self._NOT_PREFERENCE:
            if block.search(text):
                return False
        return any(p.search(text) for p in self._PREFERENCE_SIGNALS)

    def save(self, key: str, value: str) -> None:
        """Save a preference. key should be a short slug like 'response_style'."""
        if not self.is_preference(value):
            return  # silent reject — not a preference
        self._prefs[key.lower().strip()] = value.strip()[:200]

    def get(self, key: str) -> str:
        return self._prefs.get(key.lower().strip(), "")

    def all(self) -> dict[str, str]:
        return dict(self._prefs)

    def to_system_context(self) -> str:
        """Format stored preferences as a system context block."""
        if not self._prefs:
            return ""
        lines = ["[User preferences]"]
        for k, v in self._prefs.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
