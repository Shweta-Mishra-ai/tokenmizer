"""
Domain packs: the same ontology, a different vocabulary.

Goal / task / decision / file / error is a *coding* ontology, and every
regex family in `patterns.py` is a coding phrasing. Point the proxy at a
research session, an incident review or a product discussion and it
extracts almost nothing — not because the shapes are wrong but because
nobody says "Decided:" or "Fixed:" in those rooms. They say "The
hypothesis is", "Root cause:", "the customer asked for".

Two ways to close that, and this is deliberately the cheaper one.

The expensive way is a new node type per domain — hypothesis, finding,
alert, requirement — which means a new colour slot, a new lane, a new
section in the resume block and a new row in every consumer, times four
domains. The ontology would grow to thirty types, of which a given
session uses five.

The cheap way, taken here: the SHAPES a session has are already the five
this ontology holds — something you are trying to establish, the steps
you took, the calls you made, the things that went wrong, the artifacts
you referenced. A pack adds the phrasings that name those shapes in one
domain, and the words the resume block uses for them. Nothing downstream
changes, and a coding session is bit-for-bit what it was.

    coding    Goal      Working on   Done       Decided    Open issues
    research  Question  Investigating Found     Concluded  Contradictions
    ops       Incident  Mitigating   Done       Decided    Symptoms
    product   Outcome   In flight    Shipped    Decided    Blockers

What a pack is NOT: a replacement for the coding patterns. Its families
run in addition to them, so a research session that also names a file
still gets the file. A pack only ever adds recall.

Each pack ships with its own labelled corpus under
`benchmarks/eval/corpus/`, because the coding numbers are only
trustworthy because that corpus exists, and a pack without one is a claim
rather than a measurement.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from tokenmizer.graph_memory.patterns import _SPAN_CHAR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainPack:
    """Extra phrasings and resume wording for one kind of session.

    Every pattern here captures exactly one group: the label. That is the
    contract `hybrid_extractor` relies on, and the reason these are built
    from `_CLAUSE_SPAN` rather than hand-rolled — the clause span is what
    ends a label on a word boundary.
    """
    name: str
    goal_openers: tuple[re.Pattern, ...] = ()
    decisions: tuple[re.Pattern, ...] = ()
    tasks_done: tuple[re.Pattern, ...] = ()
    tasks_wip: tuple[re.Pattern, ...] = ()
    errors: tuple[re.Pattern, ...] = ()
    # Resume block headers. A missing key keeps the coding word.
    vocabulary: dict = field(default_factory=dict)


# The same clause span the coding patterns use — a budget of 80
# characters that ends on a word boundary, with a hard cut as the fallback
# for a span with no boundary inside it.
_SPAN = f'(?:{_SPAN_CHAR}{{5,80}}(?!\\w)|{_SPAN_CHAR}{{5,80}})'


def _p(prefix: str) -> re.Pattern:
    """Lead phrase, then the clause — the label is the clause alone.

    For a lead that is pure scaffolding: "the research question is X" and
    "concluded that X" both mean X, and repeating the lead in the label
    would spend resume budget on a word the section header already says.
    """
    return re.compile(prefix + r'[\s:,\-]+(' + _SPAN + ')', re.IGNORECASE)


def _pl(prefix: str) -> re.Pattern:
    """Lead phrase, then the clause — the label is BOTH.

    For a lead that carries the meaning. "This contradicts the vendor
    benchmark" reduced to "the vendor benchmark" is not an error, it is a
    noun; "the sample is too small at 120 questions" reduced to "at 120
    questions" is not anything. The coding patterns solve the same problem
    with `restore_verb`, which puts the verb back after clipping.
    """
    return re.compile('(' + prefix + r'[\s:,\-]+' + _SPAN + ')', re.IGNORECASE)


def _plo(prefix: str) -> re.Pattern:
    """Like `_pl`, but the clause after the lead is optional.

    For a lead that is already a complete statement: "traffic is
    recovered" and "this slipped" say the whole thing, and requiring a
    tail meant the sentence had to run on for the pattern to fire at all —
    which is exactly the sentences that do not.
    """
    return re.compile('(' + prefix + r'(?:[\s:,\-]+' + _SPAN + r')?)',
                      re.IGNORECASE)


# ── research ──────────────────────────────────────────────────────────────────
# A literature review, an experiment log, an analysis. The shapes: the
# question being answered (goal), what is being investigated (wip), what
# was established (done), what was concluded (decision), and the results
# that contradict the rest (error — a finding that breaks the story is
# exactly the thing a resume must not drop).

RESEARCH = DomainPack(
    name="research",
    goal_openers=(
        _p(r'(?:the\s+)?(?:research\s+)?question\s+is'),
        _p(r'(?:i(?:\'m| am)\s+)?(?:trying\s+to\s+(?:work\s+out|understand|establish)|'
           r'investigating\s+whether|looking\s+into\s+whether)'),
        _p(r'(?:the\s+)?hypothesis(?:\s+is)?'),
    ),
    decisions=(
        _p(r'(?:we\s+)?conclude(?:d)?(?:\s+that)?'),
        _p(r'(?:the\s+)?(?:evidence|data)\s+(?:supports?|suggests?|shows?)'),
        _p(r'(?:i(?:\'m| am)\s+)?confident\s+that'),
        _p(r'(?:so|therefore)\s+(?:we|i)\s+(?:will|are\s+going\s+to)\s+(?:use|adopt|go\s+with)'),
    ),
    tasks_done=(
        _pl(r'(?:we\s+)?(?:found|established|confirmed|replicated|measured|showed)'),
        _pl(r'(?:the\s+)?(?:experiment|run|trial|survey)\s+(?:showed|gave|produced)'),
        _pl(r'(?:read|reviewed|went\s+through)'),
    ),
    tasks_wip=(
        _pl(r'(?:still\s+)?(?:investigating|analysing|analyzing|reviewing|'
            r'reading\s+up\s+on|testing\s+whether)'),
        _pl(r'open\s+question'),
    ),
    errors=(
        _pl(r'(?:this\s+)?contradicts'),
        _pl(r'(?:the\s+)?(?:result|finding)s?\s+(?:do\s+not|don\'t|did\s+not|didn\'t)\s+'
            r'(?:replicate|hold|support)'),
        _plo(r'(?:no|not)\s+(?:statistically\s+)?significant'),
        _pl(r'(?:the\s+)?(?:sample|data)\s+is\s+(?:too\s+small|biased|incomplete)'),
    ),
    vocabulary={
        "goal": "Question", "wip": "Investigating", "done": "Found",
        "decided": "Concluded", "errors": "Contradictions", "files": "Sources",
    },
)

# ── ops ───────────────────────────────────────────────────────────────────────
# An incident, a migration, a rollout. The shapes: what is broken
# (goal — the incident IS the objective), what is being done about it
# right now, what has been done, what was decided, and the symptoms.

OPS = DomainPack(
    name="ops",
    goal_openers=(
        _pl(r'(?:we(?:\'re| are)\s+)?(?:seeing|getting)\s+(?:an?\s+)?(?:incident|outage|'
            r'page|alert)'),
        _pl(r'(?:sev|p)[0-4]\b'),
        _plo(r'(?:production|prod|the\s+cluster|the\s+service)\s+is\s+(?:down|degraded|'
             r'failing|unavailable)'),
    ),
    decisions=(
        _pl(r'(?:the\s+)?root\s+cause(?:\s+is|\s+was)?'),
        _pl(r'(?:we\s+)?(?:will\s+)?mitigat(?:e|ed|ing)\s+by'),
        _pl(r'(?:rolling|rolled)\s+back\s+to'),
        _p(r'(?:the\s+)?(?:runbook|procedure)\s+(?:says|is)'),
    ),
    tasks_done=(
        _pl(r'(?:we\s+)?(?:restarted|failed\s+over|drained|scaled|patched|'
            r'rolled\s+out|paged|escalated)'),
        _plo(r'(?:service|traffic|the\s+cluster)\s+(?:is\s+)?(?:recovered|restored|healthy)'),
    ),
    tasks_wip=(
        _pl(r'(?:still\s+)?(?:monitoring|draining|failing\s+over|waiting\s+for)'),
        _pl(r'(?:follow[-\s]?up|action\s+item)'),
    ),
    errors=(
        _pl(r'(?:error|exception)\s+rate\s+(?:is|at|spiked\s+to)'),
        _pl(r'(?:latency|p9[59])\s+(?:is|at|spiked\s+to|jumped\s+to)'),
        _pl(r'(?:we(?:\'re| are)\s+)?(?:dropping|losing|timing\s+out\s+on)'),
        _pl(r'(?:the\s+)?alert\s+(?:fired|is\s+firing)\s+(?:for|on)'),
    ),
    vocabulary={
        "goal": "Incident", "wip": "Mitigating", "done": "Done",
        "decided": "Decided", "errors": "Symptoms",
    },
)

# ── product ───────────────────────────────────────────────────────────────────
# A planning or discovery session. The shapes: the outcome wanted, what
# is in flight, what shipped, what was decided, and what is blocking.

PRODUCT = DomainPack(
    name="product",
    goal_openers=(
        _p(r'(?:the\s+)?(?:goal|outcome|objective)\s+(?:this\s+\w+\s+)?is'),
        _p(r'(?:we\s+)?want\s+(?:users|customers|people)\s+to'),
        _p(r'(?:the\s+)?(?:problem|job\s+to\s+be\s+done)\s+is'),
    ),
    decisions=(
        _pl(r'(?:we(?:\'re| are)\s+)?(?:cutting|descoping|deprioriti[sz]ing)'),
        _pl(r'(?:the\s+)?(?:requirement|acceptance\s+criteri(?:a|on))\s+is'),
        _p(r'(?:we\s+)?agreed(?:\s+that)?'),
        _pl(r'(?:owner|owned\s+by|dri)\s+is'),
    ),
    tasks_done=(
        _pl(r'(?:we\s+)?(?:shipped|launched|released|rolled\s+out)'),
        _pl(r'(?:customers?|users?)\s+(?:said|reported|asked\s+for)'),
    ),
    tasks_wip=(
        _pl(r'(?:in\s+flight|in\s+discovery|in\s+review|drafting|spec(?:c?ing|\'?d)?)'),
        _pl(r'(?:next\s+up|up\s+next|this\s+(?:sprint|quarter))'),
    ),
    errors=(
        _pl(r'(?:we(?:\'re| are)\s+)?blocked\s+(?:on|by)'),
        _plo(r'(?:this\s+)?(?:slipped|missed\s+the)'),
        _pl(r'(?:churn|complaints?|support\s+tickets?)\s+(?:is|are|went)\s+up'),
    ),
    vocabulary={
        "goal": "Outcome", "wip": "In flight", "done": "Shipped",
        "decided": "Decided", "errors": "Blockers",
    },
)

CODING = DomainPack(name="coding")

PACKS: dict[str, DomainPack] = {
    "coding": CODING,
    "research": RESEARCH,
    "ops": OPS,
    "product": PRODUCT,
}


# Bad domain values already warned about, so a long session does not log
# once per turn for one mistake made once at construction. Keyed on the
# repr because the offending value need not be hashable.
_warned_domains: set[str] = set()


def normalize_domain(name) -> str:
    """The canonical pack key for `name`, or `"coding"`.

    One function, because there were two copies of
    `(name or "coding").strip().lower()` — here and in
    `get_hybrid_extractor` — and adding a type guard to only one of them
    left the other still raising `AttributeError` on the same public
    call. A rule written twice is a rule enforced once.

    `name` arrives from `GraphMemory(domain=...)`, which is a public
    export, so it is whatever the caller passed. A non-string used to
    reach `.strip()` and raise three frames below the call that caused
    it — the same deferred failure the corpus loader was fixed for. An
    unknown TYPE is an unknown name: it degrades to coding like any
    other, but says so once, because unlike a typo'd string a
    non-string is unambiguously a programming error.
    """
    if name is not None and not isinstance(name, str):
        # repr() is taken ONCE, here, and the result is what both the
        # dedupe key and the log message use. Passing the raw value to
        # the logger's %r instead looks equivalent and is not: logging
        # formats lazily, so a __repr__ that raises would escape this
        # guard and surface from inside the logging machinery — turning
        # a degraded fallback back into the crash it exists to prevent.
        try:
            shown = repr(name)
        except Exception:
            shown = f"<unreprable {type(name).__name__}>"
        key = f"{type(name).__name__}:{shown}"[:120]
        if key not in _warned_domains:
            _warned_domains.add(key)
            logger.warning(
                "domain must be a string or None, got %s (%s) — falling "
                "back to the coding pack", type(name).__name__, shown,
            )
        return "coding"
    return (name or "coding").strip().lower()


def get_pack(name) -> DomainPack:
    """The pack for `name`, or the coding pack — which adds nothing, so an
    unknown name degrades to today's behaviour rather than to nothing."""
    return PACKS.get(normalize_domain(name), CODING)
