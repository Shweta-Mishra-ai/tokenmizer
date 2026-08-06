"""
Matching and scoring primitives for the extraction eval harness.

Extraction is judged against a labelled corpus. That needs a defensible
answer to one question — *when does an extracted string count as the
ground-truth item?* — because every headline number downstream inherits
whatever this decides.

Three rules, deliberately explicit:

1. **Matching is asymmetric and expectation-anchored.** A ground-truth
   item is "found" when some extracted string covers it. Coverage is
   measured over the GROUND-TRUTH item's tokens, not over the extracted
   string's, so an extractor cannot win recall by emitting one enormous
   label that happens to contain every keyword — that label would cover
   many items but score terribly on precision and on label quality.

2. **Precision is scored on the same relation, reversed.** An extracted
   item is "correct" when it covers some ground-truth item. Reporting
   recall without this is how extraction benchmarks flatter themselves:
   emitting the entire transcript as one node scores 100% recall.

3. **Nothing is fuzzy beyond word overlap.** No embeddings, no LLM judge.
   A judge model would make the harness non-deterministic and would put
   the thing being measured inside the measuring device.

Label quality is measured separately from correctness, because an
extractor can be perfectly accurate and still produce labels nobody
wants in a resume block — truncated mid-word, spanning three sentences,
or five near-duplicates of one fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "it", "this",
    "that", "we", "our", "i", "you", "they", "from", "by", "as",
})

_WORD = re.compile(r"[a-z0-9][a-z0-9._/-]*")


def tokens(text: str) -> frozenset:
    """Content words of `text`, lowercased, stopwords removed.

    A path yields BOTH the whole path and its segments: `api/auth.py`
    contributes `api/auth.py`, `api`, `auth` and `auth.py`. Keeping only
    the whole path makes an expectation like "auth endpoints"
    unmatchable against `POST /api/auth/register`; keeping only the
    segments lets a bare `api` match any path. Emitting both lets the
    coverage ratio decide.
    """
    out = set()
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or len(w) <= 1:
            continue
        out.add(w)
        if "/" in w or "." in w:
            for part in re.split(r"[/.]", w):
                if part and part not in _STOP and len(part) > 1:
                    out.add(part)
    return frozenset(out)


def covers(candidate: str, target: str, threshold: float = 0.6) -> bool:
    """True if `candidate` covers `target`'s meaning.

    Coverage is the fraction of TARGET tokens present in CANDIDATE. The
    denominator is always the target, which is what makes the relation
    asymmetric — see rule 1 in the module docstring.
    """
    t = tokens(target)
    if not t:
        return False
    c = tokens(candidate)
    if not c:
        return False
    # Substring match on a multi-word target is decisive: "fix 422" inside
    # "fixed 422 error in the login endpoint" is unambiguous.
    if len(target) >= 6 and target.lower().strip() in (candidate or "").lower():
        return True
    return len(t & c) / len(t) >= threshold


@dataclass
class CategoryScore:
    """Precision/recall/F1 for one category of one session, with the
    actual misses and spurious items retained — a bare percentage tells
    you that extraction is imperfect but not what to fix."""
    category: str
    expected: int
    extracted: int
    true_positives: int
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.true_positives / self.expected if self.expected else 1.0

    @property
    def precision(self) -> float:
        matched = self.extracted - len(self.spurious)
        return matched / self.extracted if self.extracted else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score(category: str, extracted: list[str], expected: list[str],
          threshold: float = 0.6) -> CategoryScore:
    extracted = [e for e in extracted if e and e.strip()]

    missed = [
        want for want in expected
        if not any(covers(got, want, threshold) for got in extracted)
    ]
    spurious = [
        got for got in extracted
        if not any(covers(got, want, threshold) for want in expected)
    ]
    return CategoryScore(
        category=category,
        expected=len(expected),
        extracted=len(extracted),
        true_positives=len(expected) - len(missed),
        missed=missed,
        spurious=spurious,
    )


# ── Label quality ────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"[.!?]\s")


@dataclass
class LabelQuality:
    """How usable the extracted labels are, independent of whether they
    are correct. A resume block is read by a model with a token budget;
    a correct-but-sprawling label costs budget and reads as noise."""
    count: int
    mean_chars: float
    truncated: int          # ends mid-word — the tail was cut arbitrarily
    multi_sentence: int     # spans a sentence boundary
    near_duplicates: int    # pairs where one label subsumes another

    @property
    def truncated_pct(self) -> float:
        return 100.0 * self.truncated / self.count if self.count else 0.0

    @property
    def multi_sentence_pct(self) -> float:
        return 100.0 * self.multi_sentence / self.count if self.count else 0.0


def label_quality(labels: list[str]) -> LabelQuality:
    labels = [x for x in labels if x and x.strip()]
    if not labels:
        return LabelQuality(0, 0.0, 0, 0, 0)

    truncated = 0
    for x in labels:
        s = x.rstrip()
        # A label that stops on a word character, with no terminal
        # punctuation, and is long enough to have been clipped by a
        # fixed-width capture rather than simply being short.
        if len(s) >= 60 and s and s[-1].isalnum():
            truncated += 1

    multi = sum(1 for x in labels if _SENTENCE_END.search(x))

    dupes = 0
    toks = [tokens(x) for x in labels]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = toks[i], toks[j]
            if not a or not b:
                continue
            smaller = a if len(a) <= len(b) else b
            if len(a & b) / len(smaller) >= 0.8:
                dupes += 1

    return LabelQuality(
        count=len(labels),
        mean_chars=round(sum(len(x) for x in labels) / len(labels), 1),
        truncated=truncated,
        multi_sentence=multi,
        near_duplicates=dupes,
    )
