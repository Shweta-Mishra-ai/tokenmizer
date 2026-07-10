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
    # AUDIT FIX (2026-07-10): bare "go" removed — the imperative verb
    # ("Go with tRPC") collided with Go-the-language and misclassified
    # API-style decisions as language decisions. Go is now detected only
    # via unambiguous forms: "golang", or the bigrams below.
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

    # AUDIT FIX (2026-07-10): vocabulary gaps — hybrid_extractor.py's
    # decision regexes knew tech names (supabase, clerk, ...) this file
    # didn't, so those decisions classified as None and supersession
    # silently never fired ("Use Supabase" → "Switch to Firebase" left
    # both looking active). Buckets below close the drift.
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

    AUDIT FIX (2026-07-10): the old classifier returned on the FIRST
    single-word hit, so "Use FastAPI with SQLAlchemy and PostgreSQL"
    classified only as web_framework — a later "switch Postgres to SQLite"
    was never detected as contradicting it, leaving a stale DB decision
    active in resume context. All matched topics are now returned.

    Bigrams are matched first and consume their words so that e.g.
    "session store" (cache_backend) doesn't also leak a spurious
    auth_mechanism match from the bare word "session".
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


def find_contradicting_decisions(
    new_label: str,
    new_summary: str,
    existing_nodes: dict,  # dict[str, MemoryNode]
) -> list[str]:
    """
    Find existing decision nodes that cover the same topic as the new decision.
    Returns list of node IDs to mark as SUPERSEDED (history preserved, never deleted).

    Only returns decisions that:
    1. Are currently COMPLETED (active)
    2. Cover the same topic bucket
    3. Are NOT the same decision (not a duplicate)

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

        # AUDIT FIX (2026-07-10): topic comparison is now SET INTERSECTION,
        # not equality of a single first-match topic. A multi-topic decision
        # ("FastAPI + SQLAlchemy + PostgreSQL") is contradicted by a new
        # decision touching ANY of its topics ("switch Postgres to SQLite").
        # Trade-off (deliberate): the whole node is superseded even though
        # its other topics (FastAPI) may still hold — the graph's granularity
        # is one node per decision *statement*, and the supersession
        # transition records both labels, so no information is lost; the
        # alternative (leaving it active) showed stale decisions as current.
        existing_topics = classify_topics(node.label, node.summary)
        if existing_topics & new_topics:
            # Same topic — check it's not the same decision
            if not _is_same_decision(new_label, node.label):
                to_supersede.append(node_id)

    return to_supersede


def _find_by_word_overlap(
    new_label: str,
    existing_nodes: dict,
    overlap_threshold: float = 0.6,
) -> list[str]:
    """
    Fallback: find decisions with high word overlap (same topic, unknown category).
    Only used when topic classification returns None.
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

        overlap = len(new_words & existing_words) / max(len(new_words), len(existing_words))
        if overlap >= overlap_threshold and not _is_same_decision(new_label, node.label):
            to_supersede.append(node_id)

    return to_supersede


def _is_same_decision(label_a: str, label_b: str) -> bool:
    """True if two labels are essentially the same decision (dedup check)."""
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
    # AUDIT FIX (2026-07-10, round 2): containment. "Use React" and
    # "use React for the frontend." are the same decision, but flat word
    # overlap scores them 2/5 = 0.4 — far below the 0.82 threshold. The
    # extractor can emit both variants from ONE message (different regex
    # passes capture different spans), and without this check they became
    # two nodes where one superseded the other — a self-supersession that
    # polluted the resume context with a bogus "Changed:" line.
    # Guard: the smaller set needs >= 2 words, so "Use PostgreSQL" is NOT
    # collapsed into "Switch from PostgreSQL to SQLite" ({postgresql}
    # alone would be a subset of many genuinely different decisions).
    smaller, larger = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    if len(smaller) >= 2 and smaller <= larger:
        return True
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap >= 0.82
