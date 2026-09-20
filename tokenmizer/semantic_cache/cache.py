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
    # Digest of the conversation the prompt was asked IN (every message
    # before the final user turn). A cached answer is only a correct
    # answer to the same question asked in the same state: "continue",
    # "yes", "try again" and "run the tests" recur constantly inside one
    # coding session and mean something different every time. Keyed on
    # the prompt alone, the second "continue" of a session was served the
    # first one's answer for the whole TTL. Empty string = single-turn
    # prompt with no prior conversation, which is where the cache earns
    # its keep.
    context: str = ""

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
        """Load the embedding model, or leave it None. Never raises.

        Catching ImportError alone was NOT enough, and the gap is the same
        one `core.tokenizer._get_encoding` documents for tiktoken:
        sentence-transformers ships no weights. `SentenceTransformer(...)`
        downloads them from huggingface.co on first use, so with the
        package installed but the model not cached — an air-gapped host,
        an egress proxy, a Hub outage, a rate limit — it raises OSError.

        That escaped `_load()`, and `embed()` is reached from the cache
        lookup on the request path, so a Hugging Face problem became an
        exception on every request that consulted the cache. The semantic
        cache is an optimisation; it must degrade to exact-match, never
        fail the request that was only trying to use it.

        The failure is recorded (`_initialized` is set before the attempt),
        so this costs one try per process rather than a fresh network
        timeout on every lookup.

        To avoid the network entirely, pre-fetch the model at image build
        time and point HF_HOME at it — see the Dockerfile.
        """
        if self._initialized:
            return
        self._initialized = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            logger.info("sentence-transformers not installed "
                        "— semantic cache uses exact match only")
            return

        try:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer loaded for semantic cache")
        except Exception as e:
            logger.warning(
                "Could not load the embedding model (%s: %s) — semantic "
                "cache falls back to exact match. Pre-fetch the model at "
                "image build time and set HF_HOME to avoid the network.",
                type(e).__name__, e,
            )

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def embed(self, text: str):
        self._load()
        if self._model is None:
            return None
        return self._model.encode(text[:1000], normalize_embeddings=True)

    def embed_batch(self, texts: list[str]):
        """Batch version of embed() — one model call instead of N.

        Every caller that ranks several candidates against a query (graph
        recall, and the decision-conflict/error-dedup work planned to
        reuse this same engine) needs this same batching, so it lives here
        once instead of being reimplemented per caller.
        """
        self._load()
        if self._model is None or not texts:
            return None
        return self._model.encode([t[:1000] for t in texts], normalize_embeddings=True)

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

    def _key(self, prompt: str, scope: str = "__shared__", context: str = "") -> str:
        """Include scope and conversation context in the key — session-
        specific entries don't collide across sessions, and the same
        prompt asked in a different conversation state never collides
        with an earlier answer (see CacheEntry.context)."""
        data = f"{scope}:{context}:{prompt}"
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    @staticmethod
    def conversation_fingerprint(messages: list[dict]) -> str:
        """Digest of the conversation a prompt is asked in: every message
        except the final user turn. Callers pass the result as `context`
        to get()/set(). Pure function of role+content, so two clients
        holding the same history produce the same fingerprint.

        Returns "" for a conversation with no prior turns, so a single-
        turn prompt keys exactly as it always has."""
        prior = list(messages)
        if prior and prior[-1].get("role") == "user":
            prior = prior[:-1]
        if not prior:
            return ""
        h = hashlib.sha256()
        for m in prior:
            h.update(str(m.get("role", "")).encode())
            h.update(b"\x1f")
            content = m.get("content", "")
            h.update((content if isinstance(content, str) else str(content)).encode())
            h.update(b"\x1e")
        return h.hexdigest()[:24]

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._exact:
            return
        lru_key, _ = self._exact.popitem(last=False)
        self._embeddings.pop(lru_key, None)
        self._eviction_count += 1

    def get(self, prompt: str, session_id: str = "",
            context: str = "") -> Optional[CacheEntry]:
        """
        Look up cache.
        Checks session-scoped key first (using session_id, or "__private__"
        when none was given — mirrors set()'s own default scope so a
        caller that never passes session_id is still self-consistent),
        then the shared key. Same-session hits always found; cross-session
        hits only for entries explicitly stored under share_scope="shared".
        """
        # Try session-scoped (or private, if no session_id) key first
        session_key = self._key(prompt, session_id or "__private__", context)
        if session_key in self._exact:
            entry = self._exact[session_key]
            if not entry.is_expired(self.ttl_seconds):
                self._exact.move_to_end(session_key)
                entry.touch()
                self._hit_exact += 1
                return entry

        # Try shared key
        key = self._key(prompt, "__shared__", context)

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
                # Same rule as the exact key: a near-identical question in a
                # different conversation state is a different question.
                if entry.context != context:
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
        context: str = "",
    ) -> None:
        """
        Store a cache entry. `context` is the conversation fingerprint
        (see conversation_fingerprint) — pass the same value to get().

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
        key = self._key(prompt, scope, context)

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
            context=context,
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
        # Scope and conversation context are part of the key hash and
        # can't be reversed, so match on the stored prompt instead — in
        # every conversation state, since "invalidate this prompt" means
        # every answer to it. Entries store a 500-char prefix, so compare
        # against the same prefix.
        needle = prompt[:500]
        if session_id:
            candidates = [k for k, e in self._exact.items()
                          if e.prompt == needle and e.scope in (session_id, "__shared__")]
        else:
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

# PreferenceStore used to live here: a detector, a store and a context
# formatter for habits that outlive a session ("keep it brief", "always
# TypeScript"). It had no callers — `save()` was never invoked from
# anywhere — so `/api/cache/stats` reported a preference field that was
# permanently the empty string.
#
# It is now `tokenmizer/preferences.py`, per-principal and SQLite-backed,
# and wired into the request path behind `preferences.enabled`. It does
# not belong beside a response cache: one remembers an answer to a
# question, the other remembers something about a person.
