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


# ── Decisive tokens: the second test a semantic hit must pass ────────────────
#
# A semantic hit serves ANOTHER prompt's answer. That makes a false
# positive here worse than a miss: a miss costs one upstream call, a false
# positive returns a confident, fluent answer to a question nobody asked.
#
# Cosine similarity over sentence embeddings cannot carry that decision
# alone, for two reasons that are properties of the method rather than of
# any threshold:
#
#   - Negation and polarity barely move the vector. "How do I enable the
#     cache?" and "How do I disable the cache?" differ in one token, share
#     every other, and land close together — the answers are opposites.
#   - Decisive literals are a small part of a long sentence. "roll back
#     migration 003" and "roll back migration 004" differ in one digit.
#
# There is a third, local reason: embed() truncates to text[:1000], so two
# long prompts that differ only after the thousandth character embed
# IDENTICALLY — cosine 1.0, whatever the threshold is set to.
#
# So similarity decides that two prompts are ABOUT the same thing, and
# this decides whether they are the same QUESTION about it. Same shape as
# the output trimmer's content check: a match on form is confirmed
# against substance before anything is deleted or reused.
#
# Being too strict costs a cache hit — one upstream call. Being too loose
# costs a wrong answer. This errs strict on purpose.

# Polarity is compared by CLASS, not by token, so a genuine paraphrase
# still hits: "enable the cache" and "turn the cache on" are both
# positive, while "enable" and "disable" are not.
_POLARITY = {
    "enable": 1, "enabled": 1, "on": 1, "add": 1, "adding": 1, "create": 1,
    "start": 1, "starting": 1, "include": 1, "including": 1, "allow": 1,
    "turn on": 1, "switch on": 1, "keep": 1, "show": 1,
    "disable": -1, "disabled": -1, "off": -1, "remove": -1, "removing": -1,
    "delete": -1, "deleting": -1, "stop": -1, "stopping": -1,
    "exclude": -1, "excluding": -1, "deny": -1, "drop": -1, "revert": -1,
    "undo": -1, "roll back": -1, "rollback": -1, "hide": -1,
    "not": -1, "no": -1, "never": -1, "without": -1, "cannot": -1,
    "don't": -1, "doesn't": -1, "isn't": -1, "won't": -1,
}
_MULTIWORD_POLARITY = [w for w in _POLARITY if " " in w]

# A number, a version, a code span, a path, a dotted or snake_cased
# identifier, a flag. These must match EXACTLY — a paraphrase keeps its
# literals, so requiring them costs nothing a paraphrase would want.
# Every repeat here is BOUNDED, and the snake_case branch uses a class
# that excludes "_" inside the inner repeat. Both matter: this runs on
# the request path against a caller-supplied prompt.
#
# The first version was quadratic — measured 4x the time for 2x the
# input, 0.79 s on a 16 KB prompt. Two causes. Unbounded repeats before
# a required literal (`[\w-]+\.` then an extension) make the engine
# rescan the whole run from every start position. And `\w+(?:_\w+)+`
# nests two repeats over overlapping character sets — `\w` already
# includes "_" — so the engine can split a single run between them in
# exponentially many ways, the textbook catastrophic shape.
_LITERAL = re.compile(
    r"""
      `[^`\n]{1,200}`
    | \b\d[\w.]{0,40}                     # 003, 1.2.3, 5xx, 120ms
    | \b[\w-]{1,64}\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sql|ya?ml|json|toml|md|sh|txt|csv)\b
    | (?:^|[\s(])[/~][\w./-]{1,120}       # /etc/hosts, ~/.config
    | (?:^|\s)--?[a-z][\w-]{0,60}         # --force, -v
    | \b[^\W_]{1,40}(?:_[^\W_]{1,40}){1,8}\b   # snake_case identifiers
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _decisive_profile(prompt: str) -> tuple:
    """The part of a prompt that similarity is not allowed to smooth over.

    Returns (net polarity, frozenset of literals). Two prompts may only
    share a cached answer when both agree.
    """
    lowered = prompt.lower()
    polarity = 0
    for phrase in _MULTIWORD_POLARITY:
        hits = lowered.count(phrase)
        if hits:
            polarity += _POLARITY[phrase] * hits
            lowered = lowered.replace(phrase, " ")
    for word in re.findall(r"[a-z']+", lowered):
        polarity += _POLARITY.get(word, 0)
    literals = frozenset(m.group(0).strip().lower()
                         for m in _LITERAL.finditer(prompt))
    return polarity, literals


def _profile_of(entry) -> tuple:
    """The entry's decisive profile, computed once and kept."""
    if entry._profile is None:
        entry._profile = _decisive_profile(entry.prompt)
    return entry._profile


def _same_question(a: str, b: str) -> bool:
    """True when a semantic match is safe to serve.

    Convenience for callers comparing a single pair. The scan loop in
    get() does NOT use this: it profiles the query once and compares
    each candidate against that, because profiling the query inside the
    loop repeats identical work up to max_semantic_scan times.
    """
    return _decisive_profile(a) == _decisive_profile(b)


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

    # The decisive-token profile of `prompt`, memoised on first use. The
    # semantic scan compares every surviving candidate against the query's
    # profile, so without this the same candidate is re-profiled on every
    # lookup that reaches it — max_semantic_scan of them per miss, all on
    # the request path. A prompt never changes once stored, so this is
    # computed at most once per entry for the life of the process.
    _profile: tuple | None = None

    # Bytes this entry costs, measured once at construction. The cache is
    # bounded by this as well as by entry count; see SemanticCache.
    nbytes: int = 0

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
        max_bytes: int = 256 * 1024 * 1024,
        max_entry_bytes: int = 1024 * 1024,
        max_semantic_scan: int = 2_000,
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
        # A cap on entries is not a cap on memory: an entry holds a whole
        # LLM response. 10,000 entries at 60 KB is 600 MB resident, and
        # the only thing `stats()` used to report was "utilization_pct:
        # 100", which reads as a full cache rather than as a gigabyte.
        # Both bounds are enforced and both are reported.
        self.max_bytes = max_bytes
        # A per-entry cap above the total cap is not a cap: the eviction
        # loop would empty the cache for one entry and then store it over
        # budget anyway. Clamp, and say so — an operator who set these two
        # values inconsistently gets a working proxy and a log line, not a
        # bound that silently is not one.
        if max_entry_bytes > max_bytes:
            logger.warning(
                "cache.max_entry_bytes (%d) exceeds cache.max_bytes (%d) — "
                "clamping the per-entry cap to the total, which is the most "
                "one entry can occupy in any case.",
                max_entry_bytes, max_bytes,
            )
            max_entry_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self.max_semantic_scan = max_semantic_scan
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: dict[str, object] = {}  # key → embedding
        self._embedder = EmbeddingEngine.get()
        self._bytes = 0
        self._eviction_count = 0
        self._evicted_for_bytes = 0
        self._rejected_too_large = 0
        self._expired_swept = 0
        self._sets_since_sweep = 0
        self._hit_exact = 0
        self._hit_semantic = 0
        # Semantic candidates cleared for similarity but refused by
        # the decisive-token check. An operator watching this climb
        # is watching wrong answers NOT being served.
        self._rejected_unsafe = 0
        self._miss = 0

    def _key(self, prompt: str, scope: str = "__shared__", context: str = "") -> str:
        """Include scope and conversation context in the key — session-
        specific entries don't collide across sessions, and the same
        prompt asked in a different conversation state never collides
        with an earlier answer (see CacheEntry.context)."""
        data = f"{scope}:{context}:{prompt}"
        return hashlib.sha256(data.encode("utf-8", "surrogatepass")).hexdigest()[:24]

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
            h.update(str(m.get("role", "")).encode("utf-8", "surrogatepass"))
            h.update(b"\x1f")
            content = m.get("content", "")
            h.update((content if isinstance(content, str) else str(content)).encode("utf-8", "surrogatepass"))
            h.update(b"\x1e")
        return h.hexdigest()[:24]

    # Rough per-entry overhead that is not the response text: the key,
    # the truncated prompt, the dataclass itself, and the embedding vector
    # when there is one (384 float32s for the default model). Approximate
    # on purpose — the point is a bound an operator can reason about, not
    # an exact heap measurement, and sys.getsizeof on every set() would
    # cost more than it tells anyone.
    _ENTRY_OVERHEAD = 700
    _EMBEDDING_BYTES = 384 * 4

    def _measure(self, prompt: str, response: str) -> int:
        return (len(response.encode("utf-8", "replace"))
                + len(prompt[:500].encode("utf-8", "replace"))
                + self._ENTRY_OVERHEAD)

    def _drop(self, key: str) -> None:
        """Remove one entry and give back its bytes. Every removal goes
        through here, so `_bytes` cannot drift away from the contents."""
        entry = self._exact.pop(key, None)
        if entry is None:
            return
        self._bytes -= entry.nbytes
        if self._embeddings.pop(key, None) is not None:
            self._bytes -= self._EMBEDDING_BYTES
        if self._bytes < 0:  # belt and braces: never report a negative size
            self._bytes = 0

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._exact:
            return
        lru_key = next(iter(self._exact))
        self._drop(lru_key)
        self._eviction_count += 1

    def _sweep_expired(self) -> int:
        """Drop entries past their TTL.

        Expiry used to be checked only on lookup, so an entry nobody asks
        for again was never reclaimed — it sat until LRU pressure reached
        it. On a low-traffic proxy with a large cache that is an hour of
        dead responses held for nothing. A full scan is O(n) but runs once
        every `_SWEEP_EVERY` sets, so the amortised cost is a handful of
        timestamp comparisons per request.
        """
        now = time.time()
        dead = [k for k, e in self._exact.items()
                if (now - e.created_at) > self.ttl_seconds]
        for k in dead:
            self._drop(k)
        self._expired_swept += len(dead)
        return len(dead)

    _SWEEP_EVERY = 256

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
                self._drop(key)
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
            # Once per lookup, not once per candidate — see the loop below.
            query_profile = _decisive_profile(prompt)
            best_score = 0.0
            best_key = None

            # Newest first, and bounded. This loop is O(n) Python on the
            # MISS path — the path a request takes when the cache did not
            # help it — so at max_size=10,000 it was ten thousand dict
            # lookups and dot products added to the latency of every
            # uncached request. Walking from the most-recently-used end
            # and stopping after max_semantic_scan bounds that; what it
            # skips is the coldest tail, which is also the least likely
            # to match.
            scanned = 0
            for k in reversed(self._exact):
                if scanned >= self.max_semantic_scan:
                    break
                scanned += 1
                entry = self._exact[k]
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
                # Similarity says these are ABOUT the same thing; this says
                # whether they are the same QUESTION about it. See
                # _decisive_profile: polarity and literals are exactly what
                # cosine smooths over, and serving the wrong answer costs
                # more than the extra upstream call a rejection costs.
                #
                # query_profile is computed ONCE, above the loop. Calling
                # _same_question(prompt, ...) here instead re-profiled the
                # query on every candidate — up to max_semantic_scan
                # times for one lookup, all of it identical work, all of
                # it on the request path.
                if _profile_of(entry) != query_profile:
                    self._rejected_unsafe += 1
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

        nbytes = self._measure(prompt, response)
        if nbytes > self.max_entry_bytes:
            # Serve it, do not remember it. One 40 MB answer would
            # otherwise evict the entire useful cache to store a single
            # entry that is unlikely ever to be asked for again. Counted,
            # not silent: a rising rejected_too_large in /api/cache/stats
            # is how an operator learns the per-entry cap is wrong for
            # their traffic rather than wondering why the hit rate fell.
            self._rejected_too_large += 1
            logger.debug(
                "Response of %d bytes exceeds max_entry_bytes=%d — served "
                "but not cached", nbytes, self.max_entry_bytes,
            )
            return

        # Replacing an existing key must give its bytes back first, or the
        # total drifts upward by the old entry's size on every overwrite.
        if key in self._exact:
            self._drop(key)

        self._sets_since_sweep += 1
        if self._sets_since_sweep >= self._SWEEP_EVERY:
            self._sets_since_sweep = 0
            self._sweep_expired()

        # Evict if at capacity — on EITHER bound. Leave room for the
        # embedding too, so a cache full of vectors still lands under the
        # byte budget rather than a fraction over it.
        incoming = nbytes + (self._EMBEDDING_BYTES
                             if self._embedder.available else 0)
        while len(self._exact) >= self.max_size:
            self._evict_lru()
        while self._exact and self._bytes + incoming > self.max_bytes:
            self._evict_lru()
            self._evicted_for_bytes += 1

        entry = CacheEntry(
            key=key,
            prompt=prompt[:500],
            response=response,
            input_tokens=input_tokens or count_tokens(prompt),
            output_tokens=output_tokens or count_tokens(response),
            created_at=time.time(),
            scope=scope,
            context=context,
            nbytes=nbytes,
        )
        self._exact[key] = entry
        self._exact.move_to_end(key)
        self._bytes += nbytes

        # Store embedding for semantic lookup
        if self._embedder.available:
            emb = self._embedder.embed(prompt)
            if emb is not None:
                self._embeddings[key] = emb
                self._bytes += self._EMBEDDING_BYTES

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
            if key in self._exact:
                self._drop(key)
                removed += 1
        return removed

    def clear(self) -> None:
        self._exact.clear()
        self._embeddings.clear()
        self._bytes = 0

    def stats(self) -> dict:
        total = self._hit_exact + self._hit_semantic + self._miss
        hit_rate = (self._hit_exact + self._hit_semantic) / max(1, total)
        return {
            "entries": len(self._exact),
            "max_size": self.max_size,
            "utilization_pct": round(len(self._exact) / self.max_size * 100, 1),
            # What the entry count does not tell an operator: how much
            # memory this is. A cache at 12% of its entry cap can be at
            # 100% of its byte cap, and only one of those two numbers
            # explains the resident size of the process.
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "bytes_pct": round(self._bytes / max(1, self.max_bytes) * 100, 1),
            "evictions": self._eviction_count,
            "evicted_for_bytes": self._evicted_for_bytes,
            "rejected_too_large": self._rejected_too_large,
            "expired_swept": self._expired_swept,
            "hit_rate": round(hit_rate, 3),
            "hit_exact": self._hit_exact,
            "hit_semantic": self._hit_semantic,
            "rejected_unsafe": self._rejected_unsafe,
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
