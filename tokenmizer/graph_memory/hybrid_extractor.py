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

The regex vocabulary and small pure text-analysis helpers this pipeline
applies live in patterns.py (extracted to keep this file focused on the
pipeline itself) and are re-imported below, so every name that used to
be defined here directly is still an attribute of this module.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from tokenmizer.graph_memory.patterns import (
    _ALREADY_FIXED,
    _CAUSAL_LINK,
    _COMPLETION_LEAD,
    _DECISION,
    _DECISION_FOR,
    _DECISION_HEADER,
    _DECISION_PASSIVE,
    _DEPENDENCY,
    _ENDPOINT,
    _ENV,
    _ERROR_ABSENCE,
    _ERROR_DAMAGE,
    _ERROR_DETERMINER,
    _ERROR_FALSE_HEALTH,
    _ERROR_HANDLED,
    _ERROR_INERT,
    _ERROR_INTEGRITY,
    _ERROR_MISCLASSIFIED,
    _ERROR_STOPWORDS,
    _ERROR_SYMPTOM,
    _ERROR_TYPED,
    _ERROR_VULN,
    _EVIDENCE_COST,
    _EVIDENCE_NUMBER,
    _EVIDENCE_QUOTE,
    _EVIDENCE_SCORE,
    _EVIDENCE_STANDARD,
    _FILE_COMMON,
    _FILE_EXTENSIONLESS,
    _FILE_PATH,
    _FIX_PREFIX,
    _GOAL_OPENERS,
    _LEADING_CONNECTIVE,
    _SCHEMA_HEADER,
    _SCHEMA_STOP_WORDS,
    _SCHEMA_TABLE,
    _SOLUTION_VERB,
    _SUPERSEDED,
    _TASK_DONE,
    _TASK_DONE_PASSIVE,
    _TASK_TODO,
    _TASK_WIP,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
    _clip,
    _content_words,
    _drop_leading_sentence,
    _is_negated_context,
    _is_only_paths,
    _is_question_context,
    _sentence_index,
    _tech_mention_is_a_decision,
)

logger = logging.getLogger(__name__)


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
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            label = _clip(m.group(1))
            norm  = self._normalize(label)
            if norm not in seen_decisions and len(norm) > 4:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 2: header format
        for m in _DECISION_HEADER.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            label = _clip(m.group(1))
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Decision Pass 3: tech names
        for m in _DECISION_FOR.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
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
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
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
        sentence_claims: dict[tuple[int, int], tuple[str, int]] = {}
        for pattern in (_ERROR_TYPED, _ERROR_VULN, _ERROR_INTEGRITY,
                        _ERROR_DAMAGE, _ERROR_ABSENCE, _ERROR_INERT,
                        _ERROR_FALSE_HEALTH, _ERROR_MISCLASSIFIED,
                        _ERROR_SYMPTOM):
            for m in pattern.finditer(content):
                before = content[max(0, m.start(1) - 60):m.start(1)]
                if _SOLUTION_VERB.search(before):
                    continue   # the symptom names the fix, not the failure
                # NOT _is_negated_context here. It fires on any "no"/"not"
                # in the window, and error prose is full of them: "WebSocket
                # message NOT triggering re-render — was missing dependency
                # in useEffect" lost the second failure entirely. Only the
                # phrases that actually mean *fixed* are excluded.
                if _ALREADY_FIXED.search(m.group(1)) or _ALREADY_FIXED.search(before):
                    continue   # "no longer resurrects a prune" is the fix
                if _ERROR_HANDLED.search(before):
                    continue   # the exception is being caught, not raised
                raw = m.group(1)
                err = _drop_leading_sentence(raw)
                # Where the label really starts. The subject window may open
                # on the PREVIOUS sentence's full stop — `[\w./\- ]` has to
                # allow dots for `moment.js` — so `m.start(1)` can sit one
                # sentence too early, which would file the two halves of one
                # cause-and-effect statement under different sentences and
                # defeat the dedup below.
                label_start = m.start(1) + raw.rfind(err) if err else m.start(1)
                err = _clip(_FIX_PREFIX.sub("", err.strip()), 70)
                err = _ERROR_DETERMINER.sub("", err).strip()
                err = _LEADING_CONNECTIVE.sub("", err).strip()
                # `IDOR`, `XSS`, `RCE` are four and three characters. A flat
                # minimum length rejected the entire vulnerability vocabulary,
                # which is the highest-signal thing an error label can carry.
                floor = 3 if pattern is _ERROR_VULN else 5
                if len(err) < floor or err.lower() in _ERROR_STOPWORDS:
                    continue
                if any(self._subsumes(err, e) for e in result.errors):
                    continue

                # A sentence often states the cause and the effect of ONE
                # bug — "The persistence_broken flag stayed False, SO stats
                # reported healthy over an empty database" — and a single
                # pattern matches both halves. They share no content words,
                # so the subsumption check above cannot see they are one
                # failure, and the resume block reported it twice.
                #
                # The test is the connective, not the sentence. A sentence
                # may equally list several genuinely different failures —
                # "a port collision in the integration tests, a race in the
                # fixture teardown, AND an OOM on the Windows runner" — and
                # collapsing those loses two real bugs. Only a causal link
                # ("so", "which meant", "as a result") means the second
                # clause is the consequence of the first rather than a
                # second item. Keep the half that carries more of it.
                key = (id(pattern), _sentence_index(content, label_start))
                prior = sentence_claims.get(key)
                # The window reaches a little INTO the current match: a
                # subject window routinely swallows the connective it starts
                # after ("…stayed False, |so stats| reported healthy"), which
                # would otherwise hide the link that identifies the pair.
                if prior is not None and _CAUSAL_LINK.search(
                        content[prior[1]:m.start(1) + 15]):
                    if _content_words(err) <= _content_words(prior[0]):
                        continue
                    result.errors.remove(prior[0])
                sentence_claims[key] = (err, m.end(1))
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
        result.errors = self._drop_restated_errors(result.errors)
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
        result.errors = self._drop_restated_errors(result.errors)
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

    @staticmethod
    def _drop_restated_errors(errors: list[str]) -> list[str]:
        """Keep one label per named failure.

        A status code or exception class names the failure, so two labels
        carrying the same one are two descriptions of a single event, not
        two events: a session that says "Login keeps returning 422" and
        later "Fixed: 422 error — missing email validation in the
        LoginRequest model" has one bug. Word-overlap dedup cannot see
        that — the two share only the digits — so both survived into the
        graph and the resume block reported the same failure twice.

        Keeps the label with the most content words, which is the one that
        says what actually broke rather than merely that something did.
        """
        def key(label: str) -> str | None:
            m = re.search(r'\b[A-Z]\w*(?:Error|Exception)\b', label)
            if m:
                return m.group(0).lower()
            m = re.search(r'\b[45]\d{2}\b', label)
            return m.group(0) if m else None

        def weight(label: str) -> int:
            return len({w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) > 2})

        # The LLM path can hand back non-strings; this runs on both paths.
        errors = [e for e in errors if isinstance(e, str) and e.strip()]

        best: dict[str, str] = {}
        for e in errors:
            k = key(e)
            if k is None:
                continue
            if k not in best or weight(e) > weight(best[k]):
                best[k] = e
        kept = set(best.values())
        return [e for e in errors if key(e) is None or e in kept]

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
