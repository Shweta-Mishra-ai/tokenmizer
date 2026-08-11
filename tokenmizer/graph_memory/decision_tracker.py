"""
Decision Topic Classifier
tokenmizer/graph_memory/decision_tracker.py

Problem being solved:
  User says "use PostgreSQL" → Decision node created
  Later says "actually use MySQL instead" → NEW Decision node created
  Now graph has BOTH. Resume shows BOTH. LLM gets confused.

Solution:
  Every new decision is classified into a topic bucket.
  If an existing decision covers the same topic → mark it MODIFIED (superseded).
  Resume shows only ACTIVE decisions. History preserved in graph for rollback.

Topic detection approach:
  1. Keyword matching on known tech categories (fast, no LLM needed)
  2. Word overlap for unknown topics (fallback)

This runs on every add_node(NodeType.DECISION, ...) call.
Zero external dependencies. Zero LLM calls. ~0.1ms per check.

Also hosts _is_same_error/_semantic_same_error (used by
add_node(NodeType.ERROR, ...) in graph.py) — the same "is this text A
the same underlying thing as text B" fuzzy-matching problem, just for
error labels instead of decision labels, so it lives alongside
_is_same_decision/_semantic_same_slot rather than in a second file.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Topic taxonomy ────────────────────────────────────────────────────────────
# Maps keywords → topic bucket name
# When two decisions share a bucket → one supersedes the other

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    # Databases
    "database":         ["postgresql", "postgres", "mysql", "sqlite", "mongodb",
                         "dynamodb", "cassandra", "cockroachdb", "mariadb",
                         "database", "db choice", "storage backend", "data store"],
    "cache_backend":    ["redis", "memcached", "valkey", "dragonfly",
                         "cache backend", "caching layer", "session store"],
    "search":           ["elasticsearch", "opensearch", "meilisearch", "typesense",
                         "algolia", "search engine", "full-text"],

    # Auth
    "auth_mechanism":   ["jwt", "session", "cookie", "oauth", "saml", "paseto",
                         "auth token", "authentication method", "token type"],
    "password_hashing": ["bcrypt", "argon2", "scrypt", "pbkdf2", "password hash"],

    # Frameworks
    "web_framework":    ["fastapi", "flask", "django", "express", "hono", "gin", "fiber",
                         "rails", "laravel", "spring", "nestjs", "nest.js",
                         "web framework", "backend framework", "api framework"],
    "frontend":         ["react", "vue", "angular", "svelte", "nextjs", "next.js",
                         "nuxt", "remix", "astro", "gatsby", "vite",
                         "frontend framework", "ui framework", "frontend", "client side"],
    "orm":              ["sqlalchemy", "tortoise", "peewee", "prisma", "typeorm",
                         "sequelize", "orm", "query builder"],

    # Infrastructure
    "deployment":       ["docker", "kubernetes", "k8s", "railway", "render",
                         "heroku", "fly.io", "aws", "gcp", "azure", "vercel",
                         "netlify", "deployment platform", "hosting"],
    "queue":            ["celery", "arq", "rq", "kafka", "rabbitmq", "sqs",
                         "task queue", "message queue", "job queue"],
    "storage":          ["s3", "cloudinary", "gcs", "azure blob", "minio",
                         "file storage", "object storage", "media storage"],

    # Language / runtime
    # Bare "go" is intentionally absent: the imperative verb ("Go with X")
    # is far more common in decision phrasing than the language name.
    # Go is matched only via unambiguous forms ("golang" and the bigrams).
    "language":         ["python", "typescript", "javascript", "golang", "rust",
                         "java", "kotlin", "programming language",
                         "in go", "use go", "go language", "go backend",
                         "go service", "go rewrite"],
    "runtime":          ["node", "deno", "bun", "python version", "runtime"],

    # Architecture
    "api_style":        ["rest", "graphql", "grpc", "trpc", "websocket",
                         "api design", "api style"],
    "architecture":     ["monolith", "microservice", "serverless", "modular",
                         "architecture", "system design"],
    "testing":          ["pytest", "jest", "vitest", "cypress", "playwright",
                         "testing framework", "test runner"],

    # Keep this vocabulary in sync with hybrid_extractor's decision
    # patterns: a tech name known to extraction but not classification
    # produces decisions that never participate in supersession.
    "backend_platform": ["supabase", "firebase", "appwrite", "pocketbase",
                         "convex", "amplify", "backend platform", "baas"],
    "auth_provider":    ["clerk", "auth0", "supertokens", "keycloak",
                         "cognito", "okta", "nextauth", "next auth",
                         "firebase auth", "auth provider"],
    "payments":         ["stripe", "paddle", "braintree", "lemonsqueezy",
                         "razorpay", "payment provider", "payment gateway"],
    "observability":    ["sentry", "datadog", "grafana", "prometheus",
                         "honeycomb", "new relic", "error tracking",
                         "monitoring tool"],
    "state_management": ["redux", "zustand", "mobx", "recoil", "jotai",
                         "pinia", "state management"],
    "package_manager":  ["npm", "pnpm", "yarn", "poetry", "pipenv",
                         "package manager"],
    "styling":          ["tailwind", "styled components", "css framework",
                         "sass", "css modules"],
}

# Reverse map: keyword → topic
_KEYWORD_TO_TOPIC: dict[str, str] = {}
for _topic, _keywords in _TOPIC_KEYWORDS.items():
    for _kw in _keywords:
        _KEYWORD_TO_TOPIC[_kw.lower()] = _topic


def _classify_ordered(label: str, summary: str = "") -> list[str]:
    """
    All topic buckets matched by a decision, in match order (bigrams first,
    then single words, each in word order), deduplicated.

    Multi-topic statements ("Use FastAPI with SQLAlchemy and PostgreSQL")
    must report every topic they touch; contradiction detection compares
    topic sets, so dropping topics here disables supersession for them.

    Bigrams are matched first and consume their words so that a phrase
    match ("session store" → cache_backend) does not also produce a
    spurious single-word match ("session" → auth_mechanism).
    """
    label = label.rstrip(".,!?;:")
    text = (label + " " + summary).lower()
    # Remove punctuation for matching — but preserve version numbers
    text = re.sub(r"[^\w\s\.]", " ", text)
    # "next.js" → "nextjs" for matching
    text = text.replace("next.js", "nextjs").replace("node.js", "nodejs")
    words = text.split()

    topics: list[str] = []
    consumed = [False] * len(words)

    # Bigrams first (consume both words on match)
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in _KEYWORD_TO_TOPIC:
            topic = _KEYWORD_TO_TOPIC[bigram]
            if topic not in topics:
                topics.append(topic)
            consumed[i] = consumed[i + 1] = True

    # Single words on whatever the bigrams didn't consume
    for i, word in enumerate(words):
        if consumed[i]:
            continue
        if word in _KEYWORD_TO_TOPIC:
            topic = _KEYWORD_TO_TOPIC[word]
            if topic not in topics:
                topics.append(topic)

    return topics


def classify_topics(label: str, summary: str = "") -> set[str]:
    """
    Classify a decision into ALL topic buckets it touches.

    Examples:
      "Use PostgreSQL for storage"                → {"database"}
      "Use FastAPI with SQLAlchemy and Postgres"  → {"web_framework", "orm", "database"}
      "Some custom thing"                         → set()
    """
    return set(_classify_ordered(label, summary))


def classify_topic(label: str, summary: str = "") -> Optional[str]:
    """
    Backward-compatible singular form: the primary (first-matched) topic,
    or None if nothing matched. Prefer classify_topics() in new code —
    multi-topic decisions lose information here by construction.
    """
    ordered = _classify_ordered(label, summary)
    return ordered[0] if ordered else None


# ── Slot extraction  ───────────────────────────────────────────────────
#
# Topic-bucket overlap alone is too coarse to decide "this new decision
# replaces that old one": "Use PostgreSQL for primary user data" and "Use
# SQLite for the local offline cache" both classify as "database", but
# they're plausibly two independent, complementary decisions, not a
# reversal. The "slot" is what's left of a label after removing every
# word that's part of ANY topic-taxonomy keyword (the tech-name/category
# vocabulary that DOESN'T distinguish purpose) and common stopwords — the
# descriptive remainder is a proxy for "what role does this choice play,"
# and comparing it is what separates a genuine same-purpose swap from two
# decisions that merely share a category.

_SLOT_STOP_WORDS = frozenset({
    "use", "using", "used", "for", "the", "a", "an", "with", "as", "to",
    "and", "or", "of", "in", "on", "is", "are", "this", "that", "we",
    "our", "instead", "switch", "switching", "switched", "go", "went",
    "chose", "choose", "decided", "will", "from", "away",
})

# Single-word topic keywords (e.g. "jwt", "redis", "postgresql") — an
# unambiguous category/tech-name match wherever it appears. Excludes
# multi-word keywords like "auth token" or "api style": those are only
# stripped when the exact bigram occurs CONTIGUOUSLY in the text being
# compared (see _slot_words). Splitting a bigram into individual words
# and stripping each independently (an earlier version of this fix) was
# wrong: "auth" and "api" are each part of some OTHER topic's bigram
# keyword ("auth token", "api style") but are also genuinely descriptive
# words on their own — stripping them unconditionally erased the exact
# signal that distinguishes "Enable JWT auth for the API" from "Disable
# JWT auth for the API" and broke that real regression test.
_SINGLE_WORD_TOPIC_KEYWORDS: frozenset = frozenset(
    kw for keywords in _TOPIC_KEYWORDS.values() for kw in keywords if " " not in kw
)


def _slot_words(label: str) -> frozenset:
    """
    Descriptive words in the LABEL ONLY, minus topic-taxonomy vocabulary
    and stopwords — see module comment above.

    Deliberately excludes the rationale/summary: that's the WHY a
    decision was made, which naturally differs between any two decisions
    regardless of whether they're about the same purpose ("good for
    concurrent writes" vs "team is more familiar with it" — two
    completely different rationales for what is, in fact, a genuine
    same-purpose database swap). Including it polluted the comparison
    and produced false CONTESTED results for ordinary decisions with a
    stated reason on each side.
    """
    text = re.sub(r"[^\w\s]", " ", label.lower())
    words = text.split()
    consumed = [False] * len(words)
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in _KEYWORD_TO_TOPIC:
            consumed[i] = consumed[i + 1] = True
    return frozenset(
        w for i, w in enumerate(words)
        if not consumed[i]
        and len(w) > 2
        and w not in _SLOT_STOP_WORDS
        and w not in _SINGLE_WORD_TOPIC_KEYWORDS
    )


def _matched_topic_keywords(label: str, summary: str) -> frozenset:
    """
    The actual keyword tokens/phrases (not topic BUCKET names) matched in
    label+summary — e.g. {"postgresql"} for "Use PostgreSQL 16 with
    pgvector". Two decisions naming the SAME specific keyword are a
    refinement of one choice ("PostgreSQL" -> "PostgreSQL 16 with
    pgvector"), not a swap between alternatives — that must always count
    as the same slot regardless of how much their surrounding descriptive
    text overlaps, since a version bump's descriptive words (e.g.
    "pgvector") have no reason to resemble the original's ("storage").
    """
    label = label.rstrip(".,!?;:")
    text = (label + " " + summary).lower()
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = text.replace("next.js", "nextjs").replace("node.js", "nodejs")
    words = text.split()
    matched: set[str] = set()
    consumed = [False] * len(words)
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in _KEYWORD_TO_TOPIC:
            matched.add(bigram)
            consumed[i] = consumed[i + 1] = True
    for i, w in enumerate(words):
        if not consumed[i] and w in _KEYWORD_TO_TOPIC:
            matched.add(w)
    return frozenset(matched)


# Stricter than reasoning.py's 0.55 recall bar — that one only widens
# what gets SHOWN to a query, a low-stakes ranking decision where a false
# positive is a slightly-too-long results list. This one decides whether
# two decisions get SUPERSEDED (data mutation): a false positive here
# would silently merge two decisions that should have stayed CONTESTED
# and visible, the exact failure TM-09 (see test_contested_decisions.py)
# already fixed once for the lexical case — a semantic false positive
# would reopen the same hole through a different door. A false negative
# just leaves the pair CONTESTED, same as today, so the bar favors
# precision over recall.
_SLOT_SEMANTIC_THRESHOLD = 0.72


def _semantic_same_slot(new_label: str, existing_label: str) -> bool:
    """Extra signal for _same_slot's ambiguous case: same topic bucket,
    but lexical slot-word overlap says "probably not the same purpose."
    A purpose can be phrased with zero shared words ("primary datastore"
    vs "main persistence layer" both mean the same slot as "cache" vs
    "cold storage" do NOT) — word overlap alone can't tell those apart,
    which is exactly what genuinely complementary same-topic decisions
    ("primary user data" vs "local offline cache") rely on to stay
    distinct. Only ever used to turn an existing False into True (see
    _same_slot) — never the reverse, so it can only ADD supersessions
    that lexical matching missed, never suppress one lexical matching
    already found correctly.

    No-op (False) whenever the embedding model isn't available — same
    graceful-degradation contract as reasoning.py's _semantic_matches,
    reusing the same engine.
    """
    if not new_label.strip() or not existing_label.strip():
        return False
    from tokenmizer.semantic_cache.cache import EmbeddingEngine
    engine = EmbeddingEngine.get()
    if not engine.available:
        return False
    embs = engine.embed_batch([new_label, existing_label])
    if embs is None:
        return False
    return EmbeddingEngine.cosine(embs[0], embs[1]) >= _SLOT_SEMANTIC_THRESHOLD


def _same_slot(new_label: str, new_summary: str, existing_label: str, existing_summary: str) -> bool:
    """
    True if two same-topic decisions are confidently about the SAME
    purpose/role (a genuine replacement), False if they're merely in the
    same category (ambiguous — see find_contested_decisions).

    Two decisions naming the same specific keyword (see
    _matched_topic_keywords) are always the same slot — a refinement of
    one choice, not a swap. Otherwise, if EITHER side has no descriptive
    vocabulary (e.g. a bare "Use Supabase" / "Switch to Firebase" swap
    with no further qualifying context), there's no signal to compare
    against — default to treating it as the same slot, matching this
    mechanism's original, simpler design intent for plain tech swaps.
    Otherwise require at least half of the smaller side's descriptive
    words to overlap — or, failing that, that the two labels are
    semantically close enough to be the same purpose phrased differently
    (see _semantic_same_slot; a no-op when no embedding model is
    available, so this falls back to exactly today's word-overlap-only
    behavior on a host without one).
    """
    new_keywords = _matched_topic_keywords(new_label, new_summary)
    existing_keywords = _matched_topic_keywords(existing_label, existing_summary)
    if new_keywords & existing_keywords:
        return True

    new_slot = _slot_words(new_label)
    existing_slot = _slot_words(existing_label)
    if not new_slot or not existing_slot:
        return True
    overlap = len(new_slot & existing_slot) / min(len(new_slot), len(existing_slot))
    if overlap >= 0.5:
        return True
    return _semantic_same_slot(new_label, existing_label)


def find_contradicting_decisions(
    new_label: str,
    new_summary: str,
    existing_nodes: dict,  # dict[str, MemoryNode]
) -> list[str]:
    """
    Find existing decision nodes that CONFIDENTLY cover the same purpose
    as the new decision. Returns list of node IDs to mark as SUPERSEDED
    (history preserved, never deleted).

    Only returns decisions that:
    1. Are currently COMPLETED (active)
    2. Cover the same topic bucket AND the same descriptive "slot" (see
       _same_slot) — topic overlap alone is not sufficient
    3. Are NOT the same decision (not a duplicate)

    Same-topic decisions that DON'T pass the slot check are not silently
    ignored — see find_contested_decisions(), which returns those
    separately so callers can flag rather than destroy them.

    Args:
        new_label: label of the incoming decision
        new_summary: rationale of the incoming decision
        existing_nodes: current graph nodes

    Returns:
        List of node IDs to supersede
    """
    from tokenmizer.graph_memory.graph import NodeStatus, NodeType

    new_topics = classify_topics(new_label, new_summary)

    # If topic unknown, use word overlap as fallback
    if not new_topics:
        return _find_by_word_overlap(new_label, existing_nodes)

    to_supersede = []
    for node_id, node in existing_nodes.items():
        if node.type != NodeType.DECISION:
            continue
        if node.status not in (NodeStatus.COMPLETED,):
            continue  # already superseded/archived/invalidated — skip

        # Set intersection: a multi-topic decision is contradicted by a new
        # decision touching any of its topics. This supersedes the whole
        # node even if its other topics still hold — granularity is one
        # node per decision statement, and the transition record preserves
        # both labels, whereas leaving it active would present a stale
        # choice as current.
        existing_topics = classify_topics(node.label, node.summary)
        if existing_topics & new_topics:
            if _is_same_decision(new_label, node.label):
                continue  # same decision — not a contradiction
            if _same_slot(new_label, new_summary, node.label, node.summary):
                to_supersede.append(node_id)

    return to_supersede


def find_contested_decisions(
    new_label: str,
    new_summary: str,
    existing_nodes: dict,  # dict[str, MemoryNode]
    exclude_ids: frozenset = frozenset(),
) -> list[str]:
    """
    Find existing ACTIVE decisions that share a topic with the new
    decision but were NOT confident enough to supersede (see
    find_contradicting_decisions / _same_slot). These should be marked
    CONTESTED alongside the new decision rather than silently left as-is
    (which would hide the ambiguity) or silently superseded (which would
    destroy potentially-correct information on weak evidence).

    exclude_ids: node IDs already claimed by find_contradicting_decisions
    for this same new decision — never double-classify a node as both
    superseded and contested.
    """
    from tokenmizer.graph_memory.graph import NodeStatus, NodeType

    new_topics = classify_topics(new_label, new_summary)
    if not new_topics:
        return []

    contested = []
    for node_id, node in existing_nodes.items():
        if node_id in exclude_ids:
            continue
        if node.type != NodeType.DECISION:
            continue
        if node.status not in (NodeStatus.COMPLETED,):
            continue

        existing_topics = classify_topics(node.label, node.summary)
        if not (existing_topics & new_topics):
            continue
        if _is_same_decision(new_label, node.label):
            continue
        # Reaching here means topics overlapped but _same_slot said no —
        # ambiguous, not confidently a replacement.
        if not _same_slot(new_label, new_summary, node.label, node.summary):
            contested.append(node_id)

    return contested


def _find_by_word_overlap(
    new_label: str,
    existing_nodes: dict,
    overlap_threshold: float = 0.6,
) -> list[str]:
    """
    Fallback: find decisions with high word overlap (same topic, unknown
    category). Only used when topic classification returns empty.

    Below the word-overlap threshold, also tries _semantic_same_slot as a
    second signal — the exact same gap _same_slot's ambiguous branch has
    (a decision about an unlisted technology can be phrased with zero
    words in common with an existing one about the same thing), just
    reached from the topic-unknown path instead of the topic-known one.
    Fixing one and not the other would leave the identical bug alive
    behind a different condition.
    """
    from tokenmizer.graph_memory.graph import NodeStatus, NodeType

    _STOP = frozenset({"use", "using", "the", "a", "an", "for", "to", "in",
                       "on", "with", "and", "or", "of", "is", "are", "we",
                       "our", "this", "that", "it", "be", "have", "will"})

    new_words = {w for w in re.sub(r"[^\w]", " ", new_label.lower()).split()
                 if w not in _STOP and len(w) > 2}

    if not new_words:
        return []

    to_supersede = []
    for node_id, node in existing_nodes.items():
        if node.type != NodeType.DECISION:
            continue
        if node.status != NodeStatus.COMPLETED:
            continue

        existing_words = {w for w in re.sub(r"[^\w]", " ", node.label.lower()).split()
                          if w not in _STOP and len(w) > 2}
        if not existing_words:
            continue

        if _is_same_decision(new_label, node.label):
            continue
        overlap = len(new_words & existing_words) / max(len(new_words), len(existing_words))
        if overlap >= overlap_threshold or _semantic_same_slot(new_label, node.label):
            to_supersede.append(node_id)

    return to_supersede


def _names_competing_alternatives(label_a: str, label_b: str) -> bool:
    """
    True if both labels name a technology from the SAME topic bucket but
    name DIFFERENT ones — i.e. they are competing alternatives for one
    slot ("MySQL" vs "MongoDB" for the database), not two phrasings of
    one decision.

    This is the discriminating signal that flat word overlap destroys.
    Two decisions in the same slot differ by exactly one word — the
    technology name — which is the entire semantic content of the
    decision. Every other word (use / for / the / user / database) is
    shared scaffold. The longer and more natural the label, the higher
    the overlap ratio, so `_is_same_decision`'s threshold alone gets
    MORE wrong as labels get more descriptive:

        "Use MySQL"                       vs "Use MongoDB"                        -> 0.50  (correctly distinct)
        "Use MySQL for the user database" vs "Use MongoDB for the user database"  -> 0.83  (wrongly merged)

    When this returns True the callers must NOT treat the pair as a
    duplicate — the supersession path is what should handle it.
    """
    kw_a = _matched_topic_keywords(label_a, "")
    kw_b = _matched_topic_keywords(label_b, "")
    if not kw_a or not kw_b:
        return False          # no tech vocabulary on one side — no signal

    # Compare only the keywords each side has that the other does NOT.
    # Shared keywords are usually the category word rather than the
    # choice ("Use MySQL for the user *database*" and "Use MongoDB for
    # the user *database*" both match the generic `database` keyword as
    # well as their product name), so a plain intersection test reads
    # every same-slot swap as a match and defeats the whole check.
    exclusive_a = kw_a - kw_b
    exclusive_b = kw_b - kw_a
    if not exclusive_a or not exclusive_b:
        # One side only refines the other ("Use PostgreSQL" ->
        # "Use PostgreSQL 16 with pgvector"): same choice, more detail.
        return False

    topics_a = {_KEYWORD_TO_TOPIC[k] for k in exclusive_a if k in _KEYWORD_TO_TOPIC}
    topics_b = {_KEYWORD_TO_TOPIC[k] for k in exclusive_b if k in _KEYWORD_TO_TOPIC}
    return bool(topics_a & topics_b)


def _is_same_decision(label_a: str, label_b: str) -> bool:
    """True if two labels are essentially the same decision (dedup check)."""

    # Competing alternatives in one slot are never the same decision, no
    # matter how much of the surrounding sentence they share. Checked
    # before the overlap heuristics below precisely because those
    # heuristics get this case backwards — see
    # _names_competing_alternatives.
    if _names_competing_alternatives(label_a, label_b):
        return False

    def _norm(s: str) -> str:
        s = s.lower().rstrip(".,!?;:")
        # Normalize common tech name variants
        s = re.sub(r"[^\w\s]", " ", s)
        s = s.replace("next js", "nextjs").replace("node js", "nodejs")
        s = s.replace("type script", "typescript").replace("java script", "javascript")
        return re.sub(r"\s+", " ", s).strip()

    a, b = _norm(label_a), _norm(label_b)

    if a == b:
        return True

    words_a = set(a.split())
    words_b = set(b.split())

    if not words_a or not words_b:
        return False

    negations = {
        "not",
        "no",
        "never",
        "dont",
        "don't",
    }
    opposite_terms = (
        ("enable", "disable"),
        ("enabled", "disabled"),
        ("allow", "block"),
        ("allowed", "blocked"),
        ("permit", "forbid"),
        ("permitted", "forbidden"),
        ("use", "avoid"),
        ("add", "remove"),
        ("include", "exclude"),
    )

    neg_a = bool(words_a & negations)
    neg_b = bool(words_b & negations)

    # Don't merge decisions if one is negated and the other isn't.
    if neg_a != neg_b:
        return False

    for positive, negative in opposite_terms:
        a_positive = positive in words_a
        a_negative = negative in words_a
        b_positive = positive in words_b
        b_negative = negative in words_b
        if (a_positive and b_negative and not a_negative and not b_positive) or (
            a_negative and b_positive and not a_positive and not b_negative
        ):
            return False

    # Containment: "Use React" and "use React for the frontend." are the
    # same decision even though flat word overlap (2/5) is far below the
    # threshold. The extractor can emit both variants from one message.
    # The smaller set must have >= 2 words — a single shared word
    # ("postgresql") is a subset of many genuinely different decisions.
    smaller, larger = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    if len(smaller) >= 2 and smaller <= larger:
        return True
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap >=   0.82


# ── Error dedup (Phase 3) ─────────────────────────────────────────────────
#
# Same dedup problem _is_same_decision solves, unaddressed for errors:
# graph.add_node() only merges DECISION near-duplicates, so the same
# underlying failure mentioned twice in different words ("connection to
# the DB times out" ... two turns later ... "DB connection timeout after
# 30s on checkout") became two separate ERROR nodes. context_block.py's
# "Open issues" section is capped at the top 3 by importance, so one real
# bug silently pushed out a genuinely different one instead of both being
# recognized as the same node with importance reinforced by a second hit.

_ERROR_CLASS_NAME = re.compile(r'\b[A-Z]\w*(?:Error|Exception)\b')


def _named_error_classes_conflict(label_a: str, label_b: str) -> bool:
    """True if both labels name a specific exception/error class and the
    classes differ (TypeError vs ValueError) — the error equivalent of
    _names_competing_alternatives for decisions. Checked first in both
    _is_same_error and _semantic_same_error: near-identical error text
    that differs only in the exception class name ("TypeError: x is not
    a function" vs "ReferenceError: x is not a function") shares almost
    every other word, so without this guard it would clear the overlap
    threshold below and merge two genuinely different failures into one
    node — the same precision hole _names_competing_alternatives closes
    for decisions, just reachable from the error path instead.
    """
    classes_a = {m.upper() for m in _ERROR_CLASS_NAME.findall(label_a)}
    classes_b = {m.upper() for m in _ERROR_CLASS_NAME.findall(label_b)}
    if not classes_a or not classes_b:
        return False       # no named class on one side — no signal either way
    return not (classes_a & classes_b)


def _is_same_error(label_a: str, label_b: str) -> bool:
    """True if two error labels describe the same underlying failure,
    phrased identically or near-identically. Mirrors _is_same_decision's
    containment/overlap tail, minus the enable/disable-style negation
    handling — an error either is or isn't the same failure; there's no
    "opposite" of an error the way there's an opposite of a decision.
    """
    if _named_error_classes_conflict(label_a, label_b):
        return False

    def _norm(s: str) -> str:
        s = s.lower().rstrip(".,!?;:")
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    a, b = _norm(label_a), _norm(label_b)
    if a == b:
        return True

    words_a, words_b = set(a.split()), set(b.split())
    if not words_a or not words_b:
        return False

    smaller, larger = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    if len(smaller) >= 2 and smaller <= larger:
        return True
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap >= 0.82


# Same precision-favoring bar as Phase 2's _SLOT_SEMANTIC_THRESHOLD, for
# the same reason: this decides a MERGE (discarding one of two error
# nodes), a data mutation, not a ranking. A false negative just leaves
# two nodes instead of one — a minor "Open issues" clutter, not a wrong
# claim about the codebase's state — so the bar favors precision.
_ERROR_SEMANTIC_THRESHOLD = 0.72


def _semantic_same_error(label_a: str, label_b: str) -> bool:
    """Extra signal for _is_same_error's miss case: the same failure
    restated with little or no shared vocabulary (a stack-trace fragment
    the first time, a plain-English restatement two turns later). Named-
    exception-class conflicts are refused here too — see
    _named_error_classes_conflict — a semantic false positive on two
    truly different exception types would reopen the same hole through
    this path instead. No-op (False) whenever the embedding model isn't
    installed, reusing the same EmbeddingEngine reasoning.py's semantic
    recall and _semantic_same_slot already use.
    """
    if _named_error_classes_conflict(label_a, label_b):
        return False
    if not label_a.strip() or not label_b.strip():
        return False
    from tokenmizer.semantic_cache.cache import EmbeddingEngine
    engine = EmbeddingEngine.get()
    if not engine.available:
        return False
    embs = engine.embed_batch([label_a, label_b])
    if embs is None:
        return False
    return EmbeddingEngine.cosine(embs[0], embs[1]) >= _ERROR_SEMANTIC_THRESHOLD
