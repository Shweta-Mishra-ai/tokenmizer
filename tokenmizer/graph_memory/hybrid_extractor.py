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

from tokenmizer.graph_memory.domains import get_pack, normalize_domain
from tokenmizer.graph_memory.helpers import tool_signals
from tokenmizer.graph_memory.patterns import (
    _ACCEPTANCE,
    _ACCEPTANCE_VOID,
    _ALREADY_FIXED,
    _CATEGORY_NOUN,
    _CATEGORY_STAT,
    _CAUSAL_LINK,
    _CLAUSE_END,
    _COMPLETION_LEAD,
    _DECISION,
    _DECISION_EVALUATIVE,
    _DECISION_FOR,
    _DECISION_HEADER,
    _DECISION_IMPERATIVE,
    _DECISION_IT_IS,
    _DECISION_IT_IS_PHRASE,
    _DECISION_PASSIVE,
    _DEFECT_WORD,
    _DEPENDENCY,
    _ENDPOINT,
    _ENDPOINT_ONLY,
    _ENV,
    _ERROR_ABSENCE,
    _ERROR_ALERT,
    _ERROR_CANNOT_INITIAL,
    _ERROR_CAUSE,
    _ERROR_COUNT,
    _ERROR_DAMAGE,
    _ERROR_DETERMINER,
    _ERROR_ENCOUNTER,
    _ERROR_FAILED_SUBJECT,
    _ERROR_FAILING,
    _ERROR_FALSE_HEALTH,
    _ERROR_HANDLED,
    _ERROR_HEADER,
    _ERROR_INCIDENT_WAS,
    _ERROR_INERT,
    _ERROR_INTEGRITY,
    _ERROR_MISCLASSIFIED,
    _ERROR_OBSERVED,
    _ERROR_RECURRED,
    _ERROR_RED_STATE,
    _ERROR_SLOWER,
    _ERROR_STATUS,
    _ERROR_STATUS_NAMED,
    _ERROR_STOPWORDS,
    _ERROR_SYMPTOM,
    _ERROR_TRACED,
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
    _FIX_LEAD,
    _FIX_PREFIX,
    _GOAL_OPENERS,
    _HYPOTHETICAL_FAILURE,
    _INTRANSITIVE_TAIL,
    _INVESTIGATION_PREFIX,
    _LEADING_CONNECTIVE,
    _NEGATION_WORDS,
    _NO_DEFECT,
    _NOT_A_CHOICE_BEFORE_IT_IS,
    _NOT_A_DEFECT,
    _NOT_A_TASK_START,
    _PAST_ASPECT_LEAD,
    _PROPOSAL,
    _SCHEMA_HEADER,
    _SCHEMA_STOP_WORDS,
    _SCHEMA_TABLE,
    _SOLUTION_VERB,
    _TASK_DONE,
    _TASK_DONE_BEHIND_US,
    _TASK_DONE_CHECK,
    _TASK_DONE_CROSSED_OFF,
    _TASK_DONE_GOT,
    _TASK_DONE_MANAGED,
    _TASK_DONE_NO_WORRY,
    _TASK_DONE_PASSIVE,
    _TASK_DONE_PHRASAL,
    _TASK_DONE_POSTFIX,
    _TASK_DONE_PREMISE,
    _TASK_DONE_STATE,
    _TASK_DONE_WENT,
    _TASK_TODO,
    _TASK_TODO_CHECK,
    _TASK_TODO_DEFER,
    _TASK_TODO_FRONTED,
    _TASK_TODO_HEADER,
    _TASK_TODO_INTENT,
    _TASK_TODO_NOT_DONE,
    _TASK_TODO_OWED,
    _TASK_TODO_PARKED,
    _TASK_TODO_REMIND,
    _TASK_TODO_STATE,
    _TASK_WIP,
    _WIP_LEAD,
    _WORK_CLAUSE_START,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
    _clause_start,
    _clip,
    _content_words,
    _drop_leading_sentence,
    _is_negated_context,
    _is_only_paths,
    _is_question_context,
    _sentence_index,
    _tech_mention_is_a_decision,
    clause_subject,
    find_supersessions,
    is_library_name,
    looks_english,
    mask_mentions,
    restore_verb,
)

logger = logging.getLogger(__name__)


def _parse_json_object(raw: str) -> Optional[dict]:
    """The first JSON object in a model reply, or None.

    The prompt says "JSON only", and most replies comply, but a smaller or
    local model (the ones people run extraction on to keep it cheap) often
    wraps the object in a sentence or a fenced block anyway. json.loads on
    the whole reply then fails and the batch — already paid for — is thrown
    away as an llm_extraction silent failure. Fences are stripped first;
    then, if the reply still does not parse as a whole, the outermost
    {...} span is tried on its own.
    """
    text = re.sub(r"```(?:json)?\s*|```", "", raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# A contraction's tail. The subject windows are character runs that exclude
# the apostrophe, so a window that starts right after one opens on the
# tail: "We haven't started a rollback job" produced the label "t started a
# rollback job", "Let's work on" produced "s work on".
_CONTRACTION_TAIL = re.compile(r"(?:s|t|re|ve|ll|d|m)\b\s*", re.IGNORECASE)


def _word_bounded(content: str, start: int, end: int) -> tuple[int, int]:
    """Widen or narrow [start:end] so it begins and ends on whole words.

    The bounded windows the error patterns use (`{0,40}`, `{0,60}`) stop
    where their budget runs out, which is usually inside a word: labels
    shipped as "crash from a missing NSMotionUsageDescrip" and "stale data
    from a silently fail". A truncated label also defeats the dedup, which
    compares whole words — the cut label and the one it was cut from look
    like two different failures. At the other end, a window that opens
    just after an apostrophe starts on the contraction's tail (see
    _CONTRACTION_TAIL). O(length of one word) either way.
    """
    if start > 0 and content[start - 1] in "'\u2019":
        tail = _CONTRACTION_TAIL.match(content, start, end)
        if tail:
            start = tail.end()
    elif start > 0 and (content[start - 1].isalnum() or (
            content[start - 1] == "-" and start > 1 and content[start - 2].isalnum())):
        # The span opens inside a word — usually the second half of a
        # hyphenated one, since the subject windows accept `-`: "To|-dos
        # from early …" was labelled "dos from early …". Take the whole
        # word back.
        while start > 0 and (content[start - 1].isalnum() or content[start - 1] in "-_"):
            start -= 1
    if 0 < end < len(content) and content[end - 1].isalnum():
        # Through the rest of the word — and through a `.`, `/` or `-` that
        # sits INSIDE a name (`ci/lint.yml`), never a sentence-ending one.
        while end < len(content) and (
                content[end].isalnum() or content[end] == "_" or (
                    content[end] in "./-" and end + 1 < len(content)
                    and content[end + 1].isalnum())):
            end += 1
    return start, max(start, end)


# A subject introduced by a subordinating conjunction is a condition, not a
# report: "once the migration is merged we can deploy", "if Redis is the
# better option". The predicate patterns read the subject backwards and
# would otherwise take the condition as a statement of fact.
_CONDITIONAL_LEAD = re.compile(
    r"^(?:when|whenever|once|after|before|until|till|if|unless|as soon as|"
    r"whether|in case|assuming|provided)\b",
    re.IGNORECASE,
)

# What an intent or not-done capture carries past the task itself: "to get
# to migrating the class components", "the audit log yet, soon though",
# "X once the current thing lands", "X but other things keep jumping the
# queue". Stripped for the label; the task is what is left.
_TODO_LEAD = re.compile(r"^(?:to\s+)?(?:get (?:around )?to\s+|do\s+)?", re.IGNORECASE)
_TODO_TAIL = re.compile(
    r"(?<!\s)\s+(?:yet|once|until|but|tomorrow|today|tonight|later|soon|eventually|"
    r"next (?:week|sprint)|this (?:week|afternoon|evening))\b.*$",
    re.IGNORECASE,
)


# Markdown bold around a label: "**Decision:** Postgres", "**TODO:** the
# backfill". It is how an assistant formats a status list, and every header
# pattern anchors on the keyword being at the start of the line — the `**`
# in front of it defeated all of them at once. Only `**` is removed, not
# `__`, which is part of Python names (`__init__.py`). The inner text must
# start and end on a non-space, so `f(**a, **b)` is left alone.
_EMPHASIS = re.compile(r"\*\*(?=\S)([^*\n]{1,120}?)(?<=\S)\*\*")


def _strip_emphasis(text: str) -> str:
    return _EMPHASIS.sub(r"\1", text) if "**" in text else text


# A failure whose subject is a negative quantifier did not happen: "Nothing
# failed on the last run", "No tests failed", "None of the jobs errored".
# Measured on held-out v2 that one distractor alone was 27 false errors.
_NEGATIVE_SUBJECT = re.compile(
    r"(?:nothing|none|no one|nobody|zero|neither|no\s+\w+(?:\s+\w+)?\s+"
    r"(?:failed|fails|errored|crashed|broke))\b",
    re.IGNORECASE,
)

_FIX_DESCRIPTION = re.compile(
    r"(?:(?:fixed|resolved|solved|mitigated|worked around|patched)\s+)?(?:it\s+)?by\s+"
    r"(?:adding|introducing|setting|configuring|enabling|applying|using|switching|"
    r"raising|lowering|increasing|bumping|wrapping)\b",
    re.IGNORECASE,
)

# An observed object that denies there is anything wrong.
_NONE_LEAD = re.compile(r"(?:no|not|none|nothing|zero|never|0)\b", re.IGNORECASE)


# "my edit adding the new shapes to the scanner didn't apply": the gerund
# phrase is the subject of a sentence about something else, not a statement
# of work in progress. Only the gerund's own clause is examined, up to the
# first connector, so a negated verb in a clause of its own is still work:
# "adding retries that didn't exist before", "adding a retry wrapper so
# that the result doesn't get lost".
_CLAUSE_CONNECTOR = re.compile(
    r"[,;:(]|\b(?:so|that|which|who|whom|whose|because|since|when|where|if|unless|"
    r"until|while|and|but|or|once|after|before)\b",
    re.IGNORECASE,
)
_NEGATED_VERB = re.compile(
    r"\b(?:didn't|doesn't|don't|wasn't|isn't|aren't|weren't|won't|can't|couldn't|"
    r"did not|does not|do not|was not|is not)\b",
    re.IGNORECASE,
)


def _gerund_is_subject(obj: str) -> bool:
    cut = _CLAUSE_CONNECTOR.search(obj)
    return _NEGATED_VERB.search(obj[:cut.start()] if cut else obj) is not None


_TRAILING_AUX = re.compile(
    r"\s+(?:was|were|is|are|has been|have been|had been|got|been|just|finally)$",
    re.IGNORECASE,
)


def _todo_label(text: str) -> str:
    return _TODO_TAIL.sub("", _TODO_LEAD.sub("", text.strip())).strip(" ,;:—-")


# The subject of an intransitive completion verb has to be the WORK: "the
# retry fix landed in the last commit". A person ("…attribution I wrote in
# the report") or a clause opening on a gerund ("Verifying the commit
# attribution I wrote…") is not something that got done.
_PERSON_FINAL = re.compile(r"\b(?:i|we|you|he|she|they|someone|somebody|me|us)\s*$", re.IGNORECASE)
_GERUND_INITIAL = re.compile(r"^\s*\w+ing\b(?!\s+(?:is|are|was|were)\b)", re.IGNORECASE)


def _is_work_subject(label: str) -> bool:
    # "There's a breaking change shipped in a minor version": an
    # existential introduces something, it does not name finished work.
    return (_is_subject_label(label) and not _PERSON_FINAL.search(label)
            and not _GERUND_INITIAL.match(label)
            and not re.match(r"there(?:'s| is| are| was| were)\b", label, re.IGNORECASE))


def _is_subject_label(label: str) -> bool:
    """A subject captured backwards is only a label if it names something:
    at least four characters (two for a capitalised name), at most fourteen
    words, and not a pronoun."""
    words = label.split()
    # Short names are names: "SQS", "Go", "S3". A capital is the evidence;
    # a short lowercase word is usually a fragment.
    long_enough = len(label) >= 4 or (len(label) >= 2 and label != label.lower())
    return (long_enough and 1 <= len(words) <= 14
            and label.lower().strip(" .,") not in _NOT_A_SUBJECT)


# Pronouns and quantifiers: grammatical subjects that name nothing a resume
# could carry. "Everything is in place" is not a task called "Everything".
_NOT_A_SUBJECT = frozenset({
    "it", "this", "that", "these", "those", "they", "them", "there", "which",
    "we", "i", "you", "he", "she", "everyone", "nobody", "no one",
    "everything", "anything", "something", "nothing", "all", "all of it",
    "all of this", "the rest", "that part", "this part", "the first part",
    "each", "both", "either", "neither", "one", "the other", "the latter",
    "the former", "things", "stuff", "so far so good", "that one", "this one",
})


# Order is precedence: the first pattern to claim a span keeps it.
# _ERROR_STATUS / _ERROR_STATUS_NAMED sit directly after _ERROR_TYPED
# because they were split out of it — moving them to the end of the tuple
# let _ERROR_SYMPTOM claim status-code spans first and cost 7 points of
# error F1 on the corpus.
#
# _ERROR_HEADER goes first of all: "Bug: <description>" states both that
# this is a defect and exactly where its description starts and ends, so no
# pattern that guesses a subject window should get to claim a shorter,
# vaguer slice of the same sentence first. _ERROR_ENCOUNTER and
# _ERROR_OBSERVED follow it for the same reason: their capture starts
# exactly where the problem does ("ran into |X|", "we're seeing |X|"),
# where _ERROR_SYMPTOM's subject window, reading backwards from a symptom
# word, took "seeing a flaky" out of "We're seeing a flaky test failing".
_ALL_ERROR_PATTERNS = (
    _ERROR_HEADER, _ERROR_ENCOUNTER, _ERROR_OBSERVED, _ERROR_ALERT, _ERROR_CAUSE,
    _ERROR_TYPED, _ERROR_STATUS, _ERROR_STATUS_NAMED,
    _ERROR_VULN, _ERROR_INTEGRITY,
    _ERROR_DAMAGE, _ERROR_ABSENCE, _ERROR_INERT,
    _ERROR_FALSE_HEALTH, _ERROR_MISCLASSIFIED,
    _ERROR_SYMPTOM, _ERROR_FAILING,
    _ERROR_FAILED_SUBJECT, _ERROR_COUNT, _ERROR_SLOWER,
    _ERROR_CANNOT_INITIAL,
    _ERROR_TRACED, _ERROR_RED_STATE, _ERROR_INCIDENT_WAS, _ERROR_RECURRED,
)

# Keyword prefilter for the patterns that cost the most.
#
# The subject-window patterns read up to forty characters behind every word
# boundary before they test their keyword, so they pay that price on every
# message whether or not the message contains the keyword at all. Profiled
# over ~490 KB of corpus and real-session text, five of them were half the
# extraction time. Each entry lists substrings at least one of which every
# match must contain (the pattern's own keywords, lower-cased and cut to a
# common stem); a message containing none of them cannot match and the
# pattern is skipped. A superset by construction — and checked: extraction
# output is identical with and without this table on every corpus, a real
# 900-message session and the fuzz inputs (tests/unit/test_agentic_robustness.py).
_PREFILTER: dict = {
    id(_ERROR_FAILED_SUBJECT): (
        "fail", "errored", "crash", "died", "stopped", "stuck", "hung", "wedged",
        "unresponsive", "exit", "down", "offline", "unreachable", "keep", "can",
        "could not", "couldn"),
    id(_ERROR_SYMPTOM): (
        "memory", "oom", "segfault", "segmentation", "overflow", "deadlock", "race",
        "collision", "timing out", "timed out", "times out", "timeout", "hang", "flaky",
        "panic", "crash", "regression", "pointer", "infinite loop", "not triggering",
        "borrow checker", "intermittently", "pressure", "poison", "lag", "drift", "skew",
        "goroutine", "churn", "thundering", "stale", "blank", "white screen", "spinner",
        "never", "silently", "dropped under load"),
    id(_ERROR_SLOWER): ("tak", "took", "jumped", "regressed", "climbed", "slowed", "went"),
    id(_ERROR_DAMAGE): (
        "delet", "discard", "dropped", "lose", "lost", "overwr", "clobber", "wipe",
        "resurrect", "reinstat", "corrupt", "silently"),
    id(_ERROR_FALSE_HEALTH): ("report", "return", "show", "said", "says", "stayed",
                              "remained", "still"),
    id(_ERROR_FAILING): ("fail", "broke", "errored", "erroring", "blew up", "fell over",
                         "went red"),
    id(_ERROR_INERT): ("unreachable", "unreliable", "dead code", "silently ignored",
                       "never", "not reached", "not called", "not persisted",
                       "not applied", "not enforced"),
    id(_TASK_DONE_PASSIVE): ("working", "ready", "done", "complete", "live", "passing",
                             "implemented", "deployed", "fixed", "resolved", "running"),
}


def _may_match(pattern, low: str) -> bool:
    keys = _PREFILTER.get(id(pattern))
    return keys is None or any(k in low for k in keys)


# The patterns that key on a name rather than on English prose: an
# exception class, a status code, a vulnerability class, a "Bug:" header.
# These are what a message in another language still carries.
_STRUCTURAL_ERROR_PATTERNS = (
    _ERROR_HEADER, _ERROR_TYPED, _ERROR_STATUS_NAMED, _ERROR_VULN,
)


@dataclass
class ExtractedData:
    goals: list[str] = field(default_factory=list)
    tasks_done: list[str] = field(default_factory=list)
    tasks_wip: list[str] = field(default_factory=list)
    tasks_todo: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Errors the transcript says were fixed, by label (a subset of
    # `errors`). Kept as a parallel list rather than a flag on each entry
    # so `errors` stays a plain list of strings for the merge/dedup code;
    # _extracted_to_dict folds the two into {"label", "resolved"}.
    resolved_errors: list[str] = field(default_factory=list)
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

    def __init__(self, min_confidence: float = 0.55, domain: str | None = None):
        self.min_confidence = min_confidence
        # The domain pack's families run IN ADDITION to the coding ones,
        # so a pack can only add recall and a coding session is
        # bit-for-bit what it was. See graph_memory/domains.py.
        self.pack = get_pack(domain)

    # ── Pass 1: LLM ──────────────────────────────────────────────────────────

    async def llm_extract(
        self, messages: list[dict], provider_fn
    ) -> Optional[ExtractedData]:
        """Call provider to extract structured data. Returns None on failure."""
        # Format messages as readable text for the LLM.
        #
        # Every message shape the heuristic pass reads, the model reads too:
        # the text of content-block lists (Anthropic, multimodal) — which
        # were dropped whole when only a plain-string content counted — and
        # the tool calls and failed tool results an agent's work is mostly
        # made of, summarised as one line each so the model sees WHAT was
        # edited and WHAT failed without the payloads.
        from tokenmizer.graph_memory.graph import _content_to_text
        parts = []
        for m in messages[-20:]:  # last 20 messages max
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            text = _content_to_text(m.get("content", "")).strip()
            files, errors = tool_signals(m)
            extra = [f"[edited {f}]" for f in files[:10]] + [f"[tool error: {e}]" for e in errors[:5]]
            if text or extra:
                body = (text[:500] + (" " if text and extra else "") + " ".join(extra)).strip()
                parts.append(f"[{role.upper()}]: {body}")

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
            data = _parse_json_object(raw)
            if data is None:
                raise ValueError(
                    f"no JSON object in the model's reply (first 120 chars: "
                    f"{raw[:120]!r})"
                )
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
        proposals: list | None = None,
        original: str | None = None,
        english: bool = True,
    ) -> None:
        """
        Apply all 5 extraction passes to a single message.

        `proposals`, when given, collects what a USER turn put forward
        without deciding ("Can we do this with Celery?", "Redis would
        probably fit better") — heuristic_extract records them as decisions
        only if the next assistant turn accepts. See _accepted_proposals.
        Mutates result in place. Called by heuristic_extract().

        Extracted into a helper to keep heuristic_extract() readable
        (was 194L — loop body alone was 166L).
        """
        # `content` is what the prose patterns read (mentions masked — see
        # patterns.mask_mentions); `original` is the text as sent, which
        # file, endpoint and schema names are read from.
        original = content if original is None else original
        if not english:
            self._extract_language_neutral(content, original, result, seen_tasks,
                                           seen_decisions, seen_files,
                                           seen_endpoints, seen_schemas, role)
            return

        # Goals: first 4 messages only (session intent captured early)
        if role == "user" and turn_idx < 4:
            # The domain pack's openers run FIRST and claim their sentence.
            # "The goal this quarter is X" matches the product opener and
            # the generic one, and the generic one — which knows nothing
            # about "this quarter" — leaves that fragment at the front of
            # the label. The more specific pattern should win the sentence,
            # so a later match overlapping a span already taken is skipped
            # rather than added as a second goal for the same sentence.
            taken: list[tuple[int, int]] = []
            for pattern in (*self.pack.goal_openers, _GOAL_OPENERS):
                for m in pattern.finditer(content):
                    if any(m.start() < end and start < m.end()
                           for start, end in taken):
                        continue
                    taken.append((m.start(), m.end()))
                    result.goals.append(_clip(m.group(1), 100))

        # Tasks done: full history (completed = permanent fact)
        for m in _TASK_DONE.finditer(content):
            raw_task = m.group(1)
            # "Implemented: POST /a, POST /b, POST /c" is three pieces of
            # finished work, not one task whose label is a route list cut
            # off at 80 chars (which also left the last route truncated).
            # One task per route, read from the whole sentence rather than
            # the capture, and verb-prefixed so the validator keeps it a
            # TASK; the route itself is also an ENDPOINT node, and the
            # graph links the two.
            if _ENDPOINT_ONLY.match(raw_task.strip()):
                sentence = content[m.start(1):]
                stop = _CLAUSE_END.search(sentence)
                sentence = sentence[:stop.start()] if stop else sentence
                routes = [r.group(0).rstrip(".") for r in _ENDPOINT.finditer(sentence)]
                if len(routes) >= 2:
                    verb = re.match(r"\w+", m.group(0))
                    prefix = (verb.group(0).capitalize() + " ") if verb else ""
                    for route in routes:
                        label = prefix + route
                        norm = self._normalize(label)
                        if norm not in seen_tasks:
                            result.tasks_done.append(label)
                            seen_tasks.add(norm)
                    continue
            # "completed tasks F1 dropped from 77 to 75" — the verb is an
            # adjective on one of our own category nouns, and the sentence
            # reports a measurement, not finished work.
            if _CATEGORY_NOUN.match(raw_task.lstrip()):
                continue
            # "Working on a CI step ... with the network removed to prove
            # the bake worked" — the clause already said this is not done.
            if _WIP_LEAD.search(content[max(0, m.start() - 110):m.start()]):
                continue
            # "Once the migration is merged we can deploy" — a condition on
            # future work, and the capture after the verb is that work.
            if _CONDITIONAL_LEAD.match(
                    content[_clause_start(content, m.start()):m.start()].strip()):
                continue
            # A defect report is not finished work: "Bug: a breaking change
            # shipped in a minor version".
            if _ERROR_HEADER.match(content, _clause_start(content, m.start())):
                continue
            verb = re.match(r"\w+", m.group(0))
            if raw_task.lstrip()[:1] in "([{" or re.match(
                    r"\s*(?:is|are|was|were|be|been|has|have|had|will|would|can|"
                    r"could|should|may|might|must|by(?!\s+\w+ing\b))\b", raw_task,
                    re.IGNORECASE):
                # "the pattern I just wrote (a leading …)", "written were
                # from", and a passive agent: "a test set written by someone
                # else" names who did it, not what was done. "Fixed by adding
                # a timeout" is the method, and stays.
                continue
            if _INTRANSITIVE_TAIL.match(raw_task.strip()):
                # "The retry fix landed in the last commit": the task is the
                # subject, not the prepositional phrase after the verb. A
                # passive leaves its auxiliary in the window: "The incident
                # was | resolved within an hour".
                subject = _TRAILING_AUX.sub("", clause_subject(content, m.start()))
                if not _is_work_subject(subject) or _CONDITIONAL_LEAD.match(subject) \
                        or _NEGATION_WORDS.search(subject):
                    continue
                task = _clip(subject)
            else:
                task = restore_verb(verb.group(0) if verb else "", _clip(raw_task))
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
        for m in (_TASK_DONE_PASSIVE.finditer(content)
                  if _may_match(_TASK_DONE_PASSIVE, content.lower()) else ()):
            # was r'...\\s+' — a raw string, so \\s is a
            # literal backslash-s, not the whitespace escape \s. Since no
            # real text contains a literal backslash there, this prefix
            # strip could never match anything and has never once fired.
            # "When the backfill is done, re-enable the cron" is a condition,
            # not a report that the backfill is done.
            if _CONDITIONAL_LEAD.match(m.group(1).strip()):
                continue
            task = _clip(re.sub(r'^(?:the|a|an|this|that)\s+', '',
                                m.group(1), flags=re.IGNORECASE))
            if len(task) > 5 and _is_subject_label(task):
                norm = self._normalize(task)
                if any(self._subsumes(task, t) for t in result.tasks_done):
                    continue
                if norm not in seen_tasks:
                    result.tasks_done.append(task)
                    seen_tasks.add(norm)

        # Conversational completion: a checked box, a phrasal verb ("wrapped
        # up X"), a result state on the object ("got X working"), or the
        # subject of a finished-state predicate ("X is in and working").
        # See "Conversational forms" in patterns.py.
        def _add_done(task: str) -> None:
            if len(task) < 5 or _is_only_paths(task) or _LEADING_CONNECTIVE.match(task) \
                    or _CATEGORY_NOUN.match(task):
                return
            if any(self._subsumes(task, t) for t in result.tasks_done):
                return
            norm = self._normalize(task)
            if norm not in seen_tasks:
                result.tasks_done.append(task)
                seen_tasks.add(norm)

        for pattern in (_TASK_DONE_CHECK, _TASK_DONE_PHRASAL, _TASK_DONE_GOT,
                        _TASK_DONE_MANAGED, _TASK_DONE_POSTFIX, _TASK_DONE_NO_WORRY,
                        _TASK_DONE_CROSSED_OFF, _TASK_DONE_BEHIND_US):
            for m in pattern.finditer(content):
                if pattern is not _TASK_DONE_CHECK and (
                        _is_negated_context(content, m.start())
                        or _is_question_context(content, m.start())):
                    continue
                task = _clip(m.group(1))
                if pattern is _TASK_DONE_PHRASAL:
                    if _INTRANSITIVE_TAIL.match(task):
                        # Intransitive, as in _TASK_DONE: the subject is the task.
                        subject = clause_subject(content, m.start())
                        if _is_work_subject(subject) and not _CONDITIONAL_LEAD.match(subject) \
                                and not _NEGATION_WORDS.search(subject):
                            _add_done(_clip(subject))
                        continue
                    verb = re.match(r"\w+(?: \w+)?", m.group(0))
                    task = restore_verb(verb.group(0) if verb else "", task)
                _add_done(task)
        # "With X sorted, …", "Once X was in, …": the premise is finished work.
        for m in _TASK_DONE_PREMISE.finditer(content):
            if _is_question_context(content, m.start()):
                continue
            premise = m.group(1) or m.group(2)
            if premise and _is_subject_label(premise) and not _NEGATION_WORDS.search(premise):
                _add_done(_clip(premise))
        for m in _TASK_DONE_WENT.finditer(content):
            if _is_question_context(content, m.start()):
                continue
            subject = clause_subject(content, m.start())
            if _CONDITIONAL_LEAD.match(subject) or _NEGATION_WORDS.search(subject) \
                    or not _is_work_subject(subject):
                continue
            _add_done(_clip(subject))
        for m in _TASK_DONE_STATE.finditer(content):
            if _is_question_context(content, m.start()):
                continue
            subject = clause_subject(content, m.start())
            # "No tests are in place" says the opposite of what it matches.
            if _CONDITIONAL_LEAD.match(subject) or _NEGATION_WORDS.search(subject) \
                    or not _is_subject_label(subject):
                continue
            _add_done(_clip(subject))

        # WIP: recent window only (avoid stale in-progress).
        # TODO: full history — see the note above the TODO passes below.
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
        def _outstanding(text: str, start: int, seen: list[str]) -> bool:
            if len(text) < 5 or _is_only_paths(text):
                return False
            # "Pending tasks: 2% -> 32%" talks ABOUT the category; "I'll
            # start by checking …" has no task in it.
            if _CATEGORY_STAT.match(text) or _NOT_A_TASK_START.match(text):
                return False
            if _COMPLETION_LEAD.search(content[max(0, start - 40):start]):
                return False
            if any(self._subsumes(text, g) for g in result.goals):
                return False
            return not any(self._subsumes(text, t) for t in seen)

        if is_recent:
            for m in _TASK_WIP.finditer(content):
                if _PAST_ASPECT_LEAD.search(content[max(0, m.start() - 40):m.start()]):
                    continue
                if _gerund_is_subject(m.group(1)):
                    continue
                wip = _clip(m.group(1))
                if _outstanding(wip, m.start(1), result.tasks_wip):
                    result.tasks_wip.append(wip)
            # Narrated intent is WIP-like: see _TASK_TODO_INTENT.
            for m in _TASK_TODO_INTENT.finditer(content):
                if _is_question_context(content, m.start()):
                    continue
                todo = _clip(_todo_label(m.group(1)))
                if _outstanding(todo, m.start(1), result.tasks_todo):
                    result.tasks_todo.append(todo)

        # TODO: full history, unlike WIP.
        #
        # "Working on X" goes stale — twenty turns later X is usually done or
        # dropped without anyone saying so, which is what the recency window
        # is for. A to-do does not: "Still need: a load test against the order
        # path" stays true until something says otherwise, and a long session
        # is exactly where it matters that the resume still carries it.
        # Windowing to-dos to the last 20 messages dropped every one stated
        # early in a session over 30 messages. What DOES end a to-do is its
        # completion, and that is reconciled where it can be seen across
        # extraction calls — GraphMemory.add_node merges a completed task into
        # the pending node it finishes.
        for pattern in (_TASK_TODO, _TASK_TODO_HEADER, _TASK_TODO_CHECK,
                        _TASK_TODO_NOT_DONE, _TASK_TODO_DEFER, _TASK_TODO_FRONTED,
                        _TASK_TODO_REMIND, _TASK_TODO_PARKED,
                        _TASK_TODO_OWED):
            for m in pattern.finditer(content):
                todo = _clip(_todo_label(m.group(1)))
                if _outstanding(todo, m.start(1), result.tasks_todo):
                    result.tasks_todo.append(todo)
        for m in _TASK_TODO_STATE.finditer(content):
            if _is_question_context(content, m.start()):
                continue
            subject = clause_subject(content, m.start())
            if _CONDITIONAL_LEAD.match(subject) or _NEGATION_WORDS.search(subject) \
                    or not _is_subject_label(subject):
                continue
            todo = _clip(_todo_label(subject))
            if _outstanding(todo, m.start(), result.tasks_todo):
                result.tasks_todo.append(todo)

        # Decision Pass 1: explicit verb
        for m in _DECISION.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            # "Finished up dark mode using CSS custom properties": `using`
            # names how reported work was done — see _WORK_CLAUSE_START.
            if m.group(0)[:5].lower() == "using" and _WORK_CLAUSE_START.match(
                    content[_clause_start(content, m.start()):m.start()]):
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

        # Decision Pass 2b: "Kafka it is." — the choice precedes the verb.
        for m in _DECISION_IT_IS.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            label = "Use " + m.group(1).strip()
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        def _add_decision(label: str, source_role: str) -> None:
            norm = self._normalize(label)
            if norm not in seen_decisions and len(norm) > 4:
                result.decisions.append({"label": label, "reason": "", "source_role": source_role})
                seen_decisions.add(norm)

        # Decision Pass 2c: a user's sentence-initial imperative — "Use
        # Postgres for the orders store.", "Build it with Alembic." User turns
        # only; see _DECISION_IMPERATIVE.
        if role == "user":
            for m in _DECISION_IMPERATIVE.finditer(content):
                if _is_negated_context(content, m.start(1)) or _is_question_context(content, m.start(1)):
                    continue
                _add_decision(_clip(m.group(1)), role)

        # Decision Pass 2d: the option as the subject of an evaluative
        # predicate — "Postgres felt like the right call". Stated by the
        # assistant, that is the decision; stated by the user it is a
        # proposal until the assistant agrees.
        for m in _DECISION_EVALUATIVE.finditer(content):
            if _is_question_context(content, m.start()):
                continue
            subject = clause_subject(content, m.start())
            # Negation is scoped to this clause, not the sentence: in "It
            # wasn't an obvious choice, but Postgres won out", the negation
            # belongs to the first clause and Postgres still won.
            if _NEGATION_WORDS.search(subject) or _NEGATION_WORDS.search(m.group(0)):
                continue
            if _CONDITIONAL_LEAD.match(subject) or not _is_subject_label(subject):
                continue
            label = _clip(subject)
            # A one-word subject is a bare name ("SQS made the most sense"),
            # labelled the way every other pass labels one: "Use SQS".
            if len(label.split()) == 1:
                label = "Use " + label
            if role == "user":
                if proposals is not None:
                    proposals.append(label)
            else:
                _add_decision(label, role)

        # Decision Pass 2e: "…, the managed Postgres it is." The noun-phrase
        # form of Pass 2b; same roles as 2d.
        for m in _DECISION_IT_IS_PHRASE.finditer(content):
            option = m.group(1).strip()
            if _NOT_A_CHOICE_BEFORE_IT_IS.search(option) or _NEGATION_WORDS.search(option) \
                    or not _is_subject_label(option):
                continue
            label = _clip(option)
            if len(label.split()) == 1:
                label = "Use " + label
            if role == "user":
                if proposals is not None:
                    proposals.append(label)
            else:
                _add_decision(label, role)

        if role == "user" and proposals is not None:
            for m in _PROPOSAL.finditer(content):
                label = _clip(m.group(1))
                if len(label) >= 3:
                    proposals.append(label)

        # Decision Pass 3: tech names
        for m in _DECISION_FOR.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            # A bare tech name is only a decision with choosing context —
            # see _tech_mention_is_a_decision.
            if not _tech_mention_is_a_decision(content, m.start(1), m.end(1)):
                continue
            # The capture runs on past the name, and a negation there is
            # about the name: "Kafka wasn't the right call for this" became
            # the decision "Use Kafka wasn't the right call". The clause
            # check above only looks behind the match.
            if _NEGATION_WORDS.search(m.group(1)):
                continue
            label = "Use " + _clip(m.group(1), 60)
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Domain pack — the phrasings a research, ops or product session
        # uses for the same shapes. Empty for coding, so this loop is a
        # no-op on the default path. Each pattern captures one group, the
        # label; the same guards the coding passes use apply, because the
        # noise a pack picks up is the same noise.
        for pattern in self.pack.decisions:
            for m in pattern.finditer(content):
                if _is_negated_context(content, m.start()) or \
                        _is_question_context(content, m.start()):
                    continue
                label = _clip(m.group(1))
                norm = self._normalize(label)
                if norm not in seen_decisions and len(norm) > 4:
                    result.decisions.append(
                        {"label": label, "reason": "", "source_role": role})
                    seen_decisions.add(norm)

        for pattern in self.pack.tasks_done:
            for m in pattern.finditer(content):
                task = _clip(m.group(1))
                if len(task) < 5 or _is_only_paths(task) or \
                        _LEADING_CONNECTIVE.match(task) or _CATEGORY_NOUN.match(task):
                    continue
                norm = self._normalize(task)
                if norm not in seen_tasks and not any(
                        self._subsumes(task, t) for t in result.tasks_done):
                    result.tasks_done.append(task)
                    seen_tasks.add(norm)

        if is_recent:
            for pattern in self.pack.tasks_wip:
                for m in pattern.finditer(content):
                    wip = _clip(m.group(1))
                    if len(wip) < 5 or _is_only_paths(wip) or \
                            _COMPLETION_LEAD.search(content[max(0, m.start() - 40):m.start()]):
                        continue
                    if any(self._subsumes(wip, g) for g in result.goals):
                        continue
                    if not any(self._subsumes(wip, t) for t in result.tasks_wip):
                        result.tasks_wip.append(wip)

            for pattern in self.pack.errors:
                for m in pattern.finditer(content):
                    err = _clip(m.group(1))
                    if len(err) < 5 or _is_negated_context(content, m.start()):
                        continue
                    if not any(self._subsumes(err, e) for e in result.errors):
                        result.errors.append(err)

        # Decision Pass 4: passive (bcrypt with cost factor 12)
        #
        # Gated by _tech_mention_is_a_decision for the same reason Pass 3 is:
        # this pass also infers a decision from a bare technology name, and a
        # name being mentioned is not a name being chosen. It was the only
        # bare-tech pass without the gate, and its `with` branch turned
        # "migrate 40M rows from MySQL to Postgres with no downtime" — a
        # statement of the problem — into the decision "Use Postgres".
        for m in _DECISION_PASSIVE.finditer(content):
            if _is_negated_context(content, m.start()) or _is_question_context(content, m.start()):
                continue
            if not _tech_mention_is_a_decision(content, m.start(1), m.end(1)):
                continue
            label = "Use " + m.group(1).strip()[:60]
            norm  = self._normalize(label)
            if norm not in seen_decisions:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)

        # Superseded + both sides as decisions. See find_supersessions:
        # "switched from A to B" and "B instead of A" state the same change
        # with the operands in opposite orders, and reading both with one
        # pattern recorded the rationale as the replacement decision.
        for old_label, new_label, m_start, m_end in find_supersessions(content):
            result.superseded.append({"old": old_label, "new": new_label})
            start = max(0, m_start - 60)
            surrounding = content[start:m_end + 80].replace("\n", " ").strip()
            for label in (f"Use {old_label}", f"Use {new_label}"):
                norm = self._normalize(label)
                if norm not in seen_decisions and len(norm) > 6:
                    reason = f"Replaced by: {new_label}" if label.endswith(old_label) else surrounding[:100]
                    result.decisions.append({"label": label, "reason": reason,
                                             "evidence": surrounding[:120], "source_role": role})
                    seen_decisions.add(norm)

        self._extract_structure(original, result, seen_files, seen_endpoints, seen_schemas)
        self._extract_errors(content, result, _ALL_ERROR_PATTERNS)

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


    def _extract_language_neutral(self, content: str, original: str, result: "ExtractedData",
                                  seen_tasks: set, seen_decisions: set, seen_files: set,
                                  seen_endpoints: set, seen_schemas: set, role: str) -> None:
        """What a message in another language still says in a form these
        patterns can read: checkboxes, "TODO:"/"Decision:" headers, file and
        route names, exception names and status codes. See
        patterns.looks_english for why nothing else is attempted."""
        for m in _TASK_DONE_CHECK.finditer(content):
            task = _clip(m.group(1))
            norm = self._normalize(task)
            if len(task) >= 5 and norm not in seen_tasks:
                result.tasks_done.append(task)
                seen_tasks.add(norm)
        for pattern in (_TASK_TODO_CHECK, _TASK_TODO_HEADER):
            for m in pattern.finditer(content):
                todo = _clip(m.group(1))
                if len(todo) >= 5 and not any(self._subsumes(todo, t) for t in result.tasks_todo):
                    result.tasks_todo.append(todo)
        for m in _DECISION_HEADER.finditer(content):
            label = _clip(m.group(1))
            norm = self._normalize(label)
            if norm not in seen_decisions and len(norm) > 4:
                result.decisions.append({"label": label, "reason": "", "source_role": role})
                seen_decisions.add(norm)
        self._extract_structure(original, result, seen_files, seen_endpoints, seen_schemas)
        self._extract_errors(content, result, _STRUCTURAL_ERROR_PATTERNS)

    def _extract_structure(self, original: str, result: "ExtractedData",
                           seen_files: set, seen_endpoints: set, seen_schemas: set) -> None:
        """Files, endpoints and schemas. These are names, not prose, so they
        are read from the original text — quotes and code blocks included —
        and in any language."""
        # Files
        for m in _FILE_COMMON.finditer(original):
            if is_library_name(m.group(1).strip()):
                seen_files.add(m.group(1).strip())   # never a file; see _LIBRARY_NOT_FILE
        for m in _FILE_PATH.finditer(original):
            f = m.group(1).strip()
            if f not in seen_files and len(f) > 4:
                result.files.append(f)
                seen_files.add(f)
        for m in _FILE_COMMON.finditer(original):
            f = m.group(1).strip()
            if f not in seen_files:
                result.files.append(f)
                seen_files.add(f)
        for m in _FILE_EXTENSIONLESS.finditer(original):
            f = m.group(1).strip()
            if f not in seen_files:
                result.files.append(f)
                seen_files.add(f)

        # Endpoints — "POST /api/auth/login"
        for m in _ENDPOINT.finditer(original):
            if _is_negated_context(original, m.start()):
                continue
            ep = m.group(0).strip().rstrip(".")
            norm = self._normalize(ep)
            if norm not in seen_endpoints:
                result.endpoints.append(ep)
                seen_endpoints.add(norm)

        # Schemas — header format ("Schema: users table — ...")
        for m in _SCHEMA_HEADER.finditer(original):
            if _is_negated_context(original, m.start()):
                continue
            schema = _clip(m.group(1), 100)
            norm = self._normalize(schema)
            if norm not in seen_schemas and len(schema) > 3:
                result.schemas.append(schema)
                seen_schemas.add(norm)

        # Schemas — inline "X table" mention, excluding generic non-
        # identifier words immediately before "table" and negated
        # mentions ("No refresh_tokens table needed").
        for m in _SCHEMA_TABLE.finditer(original):
            word = m.group(1).lower()
            if word in _SCHEMA_STOP_WORDS:
                continue
            if _is_negated_context(original, m.start()):
                continue
            schema = m.group(0).strip()
            norm = self._normalize(schema)
            if norm not in seen_schemas:
                result.schemas.append(schema)
                seen_schemas.add(norm)

    def _extract_errors(self, content: str, result: "ExtractedData", patterns) -> None:
        """Run the error patterns over `content` in `patterns` order, which
        is precedence (see _ALL_ERROR_PATTERNS)."""
        # Errors: full history, NOT the recent window.
        #
        # An error is a permanent fact about the session in the same way a
        # completed task is: a resolved one explains why the code looks
        # the way it does, and an unresolved one is the single most
        # important thing to carry into a resume. Gating on recency meant
        # a session that diagnosed three failures early and spent the rest
        # of its turns fixing them carried none of them forward.
        sentence_claims: dict[tuple[int, int], tuple[str, int]] = {}
        low = content.lower()
        for pattern in patterns:
            if not _may_match(pattern, low):
                continue
            for m in pattern.finditer(content):
                if pattern is _ERROR_HEADER and _NO_DEFECT.match(m.group(1).strip(" *_")):
                    continue   # "Errors: none"
                if pattern is _ERROR_OBSERVED and not _DEFECT_WORD.search(m.group(1)):
                    continue   # "we're seeing a 20% speedup" is good news
                if pattern in (_ERROR_RED_STATE, _ERROR_INCIDENT_WAS, _ERROR_RECURRED) \
                        and not _DEFECT_WORD.search(m.group(1)):
                    continue   # "CI is red — rerunning", "then run it again"
                if pattern is _ERROR_TRACED and not _DEFECT_WORD.search(m.group(0)):
                    continue   # "tracked the ticket down to the billing team"
                if pattern in (_ERROR_OBSERVED, _ERROR_ENCOUNTER) and _NONE_LEAD.match(m.group(1).lstrip()):
                    continue   # "there's no error", "seeing zero failures"
                if _NOT_A_DEFECT.search(m.group(1)):
                    continue   # "logistic regression" is a model
                if pattern not in _STRUCTURAL_ERROR_PATTERNS and "    " in m.group(1):
                    # The span runs across a blanked mention (mask_mentions
                    # leaves spaces where a quote was): the sentence is ABOUT
                    # the quoted text — "I found earlier: "…" read as an error".
                    continue
                if m.group(1).lstrip().startswith("Traceback (most recent"):
                    continue   # the header; the exception line below it names the failure
                if pattern not in _STRUCTURAL_ERROR_PATTERNS and _HYPOTHETICAL_FAILURE.search(
                        content[max(0, m.start(1) - 20):m.end(1)]):
                    continue   # "…unless it would fail on the old code"
                if pattern not in _STRUCTURAL_ERROR_PATTERNS and _NEGATIVE_SUBJECT.match(
                        content[_clause_start(content, m.start(1)):m.end(1)].lstrip(" -*\t")):
                    continue   # "Nothing failed on the last run"
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
                span_start, span_end = _word_bounded(content, m.start(1), m.end(1))
                raw = content[span_start:span_end]
                err = _drop_leading_sentence(raw)
                # "Fixed by adding a 5 second context timeout" describes the
                # fix; the timeout is the solution, not the failure. The
                # solution-verb check above only sees text BEFORE the match,
                # and a subject window can start at the fix clause itself.
                if _FIX_DESCRIPTION.match(err):
                    continue
                # Where the label really starts. The subject window may open
                # on the PREVIOUS sentence's full stop — `[\w./\- ]` has to
                # allow dots for `moment.js` — so `m.start(1)` can sit one
                # sentence too early, which would file the two halves of one
                # cause-and-effect statement under different sentences and
                # defeat the dedup below.
                label_start = span_start + raw.rfind(err) if err else span_start
                err = _clip(_INVESTIGATION_PREFIX.sub(
                    "", _FIX_PREFIX.sub("", err.strip())), 70)
                err = _ERROR_DETERMINER.sub("", err).strip()
                err = _LEADING_CONNECTIVE.sub("", err).strip()
                # `IDOR`, `XSS`, `RCE` are four and three characters. A flat
                # minimum length rejected the entire vulnerability vocabulary,
                # which is the highest-signal thing an error label can carry.
                # The label, not the clause: the subject window can open in
                # the previous sentence, so the clause test above can miss
                # "…deploy. Nothing failed on the last run".
                if pattern not in _STRUCTURAL_ERROR_PATTERNS and _NEGATIVE_SUBJECT.match(err):
                    continue
                floor = 3 if pattern is _ERROR_VULN else 5
                if len(err) < floor or err.lower() in _ERROR_STOPWORDS:
                    continue
                # One failure, two labels: keep the more specific. A pattern
                # later in the order can recover the whole description of a
                # failure an earlier one only clipped a piece of.
                dup = next((i for i, e in enumerate(result.errors)
                            if self._subsumes(err, e)), None)
                if dup is not None:
                    if _content_words(err) > _content_words(result.errors[dup]):
                        result.errors[dup] = err
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
                # "Fixed: <error>" / "resolved the <error>" — the failure
                # happened and is over. Recorded so the graph marks it
                # resolved instead of carrying it into every resume as an
                # open bug. The fix prefix stripped from the label above is
                # the same signal when it sat inside the captured span.
                if _FIX_LEAD.search(before) or _FIX_PREFIX.match(raw.strip()):
                    result.resolved_errors.append(err)
                key = (id(pattern), _sentence_index(content, label_start))
                prior = sentence_claims.get(key)
                # The window reaches a little INTO the current match: a
                # subject window routinely swallows the connective it starts
                # after ("…stayed False, |so stats| reported healthy"), which
                # would otherwise hide the link that identifies the pair.
                if prior is not None and _CAUSAL_LINK.search(
                        content[prior[1]:span_start + 15]):
                    if _content_words(err) <= _content_words(prior[0]):
                        continue
                    if prior[0] in result.errors:
                        result.errors.remove(prior[0])
                sentence_claims[key] = (err, span_end)
                result.errors.append(err)

    def heuristic_extract(
        self,
        messages: list[dict],
        window_size: int = 0,
        prior_message: dict | None = None,
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

        prior_message: the message immediately before `messages[0]`, when
        the caller extracts incrementally (GraphMemory passes it). A user's
        proposal and the assistant's "Sounds good" can arrive in different
        extraction calls; without the prior message the acceptance would
        have nothing to accept.
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

        from tokenmizer.graph_memory.graph import _content_to_text

        # Proposals from the most recent user turn, waiting for the next
        # assistant turn to accept or ignore them.
        open_proposals: list[str] = []
        if prior_message and prior_message.get("role") == "user":
            prior_text = _strip_emphasis(_content_to_text(prior_message.get("content", "")))
            if prior_text.strip() and looks_english(prior_text):
                self._extract_one_message(
                    mask_mentions(prior_text), "user", -1, False, ExtractedData(), set(),
                    set(), set(), set(), set(), proposals=open_proposals,
                    original=prior_text,
                )

        for i, msg in enumerate(messages):
            role      = msg.get("role", "user")
            # Tool calls and tool results first: an agent's tool-only turn
            # has no text at all and still edited a file or hit an error.
            tool_files, tool_errors = tool_signals(msg)
            for f in tool_files:
                if f not in seen_files:
                    result.files.append(f)
                    seen_files.add(f)
            for e in tool_errors:
                if not any(self._subsumes(e, x) for x in result.errors):
                    result.errors.append(e)
            original = _strip_emphasis(_content_to_text(msg.get("content", "")))
            if not original.strip():
                continue
            content = mask_mentions(original)
            is_recent = i >= recent_start

            if role == "assistant" and open_proposals:
                for label in self._accepted_proposals(open_proposals, content):
                    norm = self._normalize(label)
                    if norm not in seen_decisions:
                        result.decisions.append(
                            {"label": label, "reason": "", "source_role": "user"})
                        seen_decisions.add(norm)
            if role != "system":
                open_proposals = []

            self._extract_one_message(
                content, role, i, is_recent,
                result, seen_decisions, seen_tasks, seen_files,
                seen_endpoints, seen_schemas,
                proposals=open_proposals if role == "user" else None,
                original=original, english=looks_english(original),
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


    @staticmethod
    def _accepted_proposals(proposals: list[str], reply: str) -> list[str]:
        """The proposals a reply accepts: all of them if it opens by
        agreeing ("Sounds good, going with that."), none otherwise.

        A reply that agrees and then qualifies in the same sentence ("Sure,
        but I'd use Kafka instead") is a counter-proposal, not acceptance.
        """
        if not _ACCEPTANCE.match(reply):
            return []
        first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0]
        if _ACCEPTANCE_VOID.search(first):
            return []
        return list(proposals)

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
            filler = {"use", "using", "used", "go", "went", "with", "for",
                      "the", "and", "decided", "choose", "chose", "pick",
                      "picked", "switch", "switched", "adopt", "adopted"}
            # The bare "Use <name>" the tech-name passes synthesise, where
            # every word of the name also appears in the longer label: "Use
            # sqlc" beside "sqlc for type-safe database access", "Use
            # Pydantic v2" beside "Pydantic v2 for request validation". The
            # topic-keyword check below only knows the technologies in its
            # own list, so a name outside it (sqlc, Pydantic, Helm) kept its
            # duplicate.
            if short.startswith("Use "):
                name = {w for w in re.findall(r"[a-z0-9]+", short[4:].lower())}
                long_words = set(re.findall(r"[a-z0-9]+", long.lower()))
                if name and name <= long_words and len(long) > len(short):
                    return True
            ks, kl = _matched_topic_keywords(short, ""), _matched_topic_keywords(long, "")
            if not ks or not ks <= kl:
                return False
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


# One extractor per domain pack. Packs are stateless and the patterns are
# compiled once at import, so caching them costs nothing and keeps the
# common case a dict lookup.
_extractors: dict[str, HybridExtractor] = {}


def get_hybrid_extractor(domain: str | None = None) -> HybridExtractor:
    if domain is None:
        try:
            from tokenmizer.config.settings import get_settings
            domain = get_settings().domain
        except Exception:
            domain = "coding"
    # Shared with get_pack via domains.normalize_domain: this line and
    # that one were the same expression written twice, and a type guard
    # added to one of them left this one still raising AttributeError.
    key = normalize_domain(domain)
    if key not in _extractors:
        _extractors[key] = HybridExtractor(domain=key)
    return _extractors[key]
