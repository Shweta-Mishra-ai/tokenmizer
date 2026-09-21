"""
What the ontology did not catch, from the turns windowing is about to drop.

The problem this exists for: once a session crosses
`memory.max_tokens_before_summary`, every turn older than the protected
tail is replaced by the graph's context block. Anything the extractor did
not turn into a node is gone from the conversation for the rest of the
session — and the extractor is measured at 91% on real transcripts, with
everything outside the ontology (a constraint the user stated once, a
budget, a preference, a deadline) outside that number entirely.

Those are exactly the facts a session is most annoyed to lose. "Keep the
bundle under 500KB", "we cannot use anything AGPL", "the customer demo is
on the 14th" are each one sentence, said once, and none of them is a
task, a decision, a file or an error.

The selector is deliberately narrow. A sentence is kept only if it
carries a hard signal — a number with a unit, a constraint verb, a
prohibition — AND is not already represented by a node the graph holds.
Keeping anything else would spend the resume budget restating what the
ontology already says, which is worse than losing the sentence: the
budget is the scarce thing, and the graph's own sections are better than
prose at the same job.

Written by this heuristic selector by default. With `use_llm_extraction`
on, the chat model is a strictly better sentence selector and can replace
`select_sentences` without changing anything else here.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tokenmizer.graph_memory.patterns import _clip

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

# A sentence ends at .!? followed by space or end. Matches patterns.py's
# treatment of dotted identifiers (moment.js, Python 3.12, go.mod), which
# must not split a sentence.
_SENTENCE = re.compile(r'[^.!?\n]+(?:[.!?](?!\w)|\n|$)')

# A quantity with a unit, or a version. This is the strongest single
# signal that a sentence carries a fact the ontology has no node for.
_QUANTITY = re.compile(
    r'\b\d+(?:\.\d+)?\s*'
    r'(?:kb|mb|gb|tb|ms|s|sec|secs|seconds?|m|min|mins|minutes?|h|hrs?|hours?|'
    r'days?|weeks?|months?|%|percent|rps|qps|req/s|k|m|bn|usd|eur|gbp|\$)\b',
    re.IGNORECASE,
)

# "must", "cannot", "never" — a rule the session has to keep obeying after
# the turn that stated it has been windowed away.
_CONSTRAINT = re.compile(
    r'\b(?:must(?:\s+not)?|cannot|can\'t|may\s+not|should(?:\s+not)?|shall\s+not|'
    r'never|always|required?|requires?|mandatory|forbidden|not\s+allowed|'
    r'ha(?:s|ve)\s+to|need(?:s|ed)?\s+to|got\s+to|'
    r'no\s+more\s+than|at\s+least|at\s+most|under|over|within|by\s+the\s+end|'
    r'deadline|due\s+(?:by|on)|budget|limit(?:ed)?\s+to|cap(?:ped)?\s+at|'
    r'prefer(?:s|red)?|avoid|don\'t\s+use|do\s+not\s+use|stick\s+to|'
    # A gate someone has to pass through is a rule the same way a
    # number is: "nothing ships without the on-call engineer approving".
    r'sign[-\s]?off|approv(?:e|es|ed|al)|nothing\s+\w+\s+without)\b',
    re.IGNORECASE,
)

# Sentences that are conversational filler carry no fact whatever else
# they match. "You should always run the tests" is advice about advice.
_FILLER = re.compile(
    r'^(?:ok|okay|sure|thanks|thank you|got it|sounds good|yes|no|yep|nope|'
    r'right|great|perfect|nice|cool|hmm|well)\b[\s,.!]*$',
    re.IGNORECASE,
)

_MIN_CHARS = 20
_MAX_CHARS = 120


# One sentence often states two rules: "this must be done by the 14th —
# the demo is that morning — and nothing ships without the on-call
# engineer approving it". Clipping that to one clause keeps the deadline
# and loses the approval gate, so each clause is weighed on its own.
_CLAUSE = re.compile(r'\s*(?:[—;]|,\s+(?:and|but|so)\s+)\s*')


def _sentences(messages: list[dict]) -> list[str]:
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue            # tool results and content-part lists
        for s in _SENTENCE.finditer(content):
            for part in _CLAUSE.split(s.group(0)):
                text = " ".join(part.split()).strip(" .!?")
                if _MIN_CHARS <= len(text) and not _FILLER.match(text):
                    out.append(text)
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def select_sentences(messages: list[dict], known: list[str],
                     limit: int = 4) -> list[str]:
    """The kept sentences, most recent first.

    `known` is every label the graph already holds. A sentence covered by
    one of them is dropped: the node says it better and the section it
    lives in is already in the resume block.
    """
    known_tokens = [_tokens(k) for k in known if k]
    picked: list[tuple[int, str]] = []

    for sentence in _sentences(messages):
        quantity = bool(_QUANTITY.search(sentence))
        constraint = bool(_CONSTRAINT.search(sentence))
        if not (quantity or constraint):
            continue

        words = _tokens(sentence)
        if len(words) < 4:
            continue
        # Already a node: not "similar to", but genuinely covered — most
        # of what this sentence says is in that label.
        if any(kt and len(words & kt) / len(words) >= 0.6 for kt in known_tokens):
            continue
        # ...or covered by something already picked, which happens when a
        # point is restated across turns.
        if any(len(words & _tokens(p)) / len(words) >= 0.7 for _, p in picked):
            continue

        # A quantity is worth more than a modal verb: "under 500KB" is a
        # number someone has to hit, "we should probably look at it" is not.
        picked.append((2 if quantity else 1, _clip(sentence, _MAX_CHARS)))

    # Most recent last in the transcript, and the newest statement of a
    # rule is the one in force — so take from the end, strongest first
    # within what is left.
    picked.reverse()
    picked.sort(key=lambda p: -p[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, text in picked:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def summarise_span(graph: "GraphMemory", dropped: list[dict],
                   limit: int = 4) -> str:
    """The note for the turns `dropped`, or "" when there is nothing worth
    keeping — which is the common case and must stay cheap."""
    known = [n.label for n in graph._nodes.values() if not n._evicted]
    return " | ".join(select_sentences(dropped, known, limit=limit))
