"""
Regression tests for TM-01: the heuristic extractor must not turn a
NEGATED statement into a positive decision.

Bug (the most severe finding in the audit): none of the decision-
extraction passes checked for negation. "We are NOT using Redis" matched
Pass 1's verb list ("using") and Pass 3's tech-name list ("redis")
independently, both blind to the preceding "NOT", and both produced
"Use Redis" as an extracted decision — the literal opposite of what was
said. Reproduced:

    input:  "We are NOT using Redis. Do not use MongoDB either."
    output: [{"label": "Use Redis"}, {"label": "Use MongoDB either"}, ...]

This became critical in combination with SmartMessageWindow, which
replaces older conversation turns with the graph's context block —
deleting the original sentence and leaving only the fabricated "Decided:
Use Redis" in what the model sees.

Fix: a clause-scoped negation check runs before any decision pattern
(all 4 heuristic passes) is allowed to emit. It looks backward from the
match to the start of the current clause (the nearest sentence boundary)
and rejects the match if a negation word appears in that span — so an
unrelated negation in an EARLIER, different sentence doesn't suppress a
later, legitimate decision.
"""
from __future__ import annotations

from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor

extractor = HybridExtractor(min_confidence=0.50)


def _labels(messages):
    result = extractor.heuristic_extract(messages)
    return [d.get("label", "") for d in result.decisions]


class TestNegatedStatementsProduceNoDecision:

    def test_not_using_x_is_not_extracted_as_use_x(self):
        messages = [{"role": "user", "content": "We are NOT using Redis for this."}]
        labels = _labels(messages)
        assert not any("redis" in l.lower() for l in labels), (
            f"a negated statement was extracted as a positive decision: {labels}"
        )

    def test_do_not_use_x_is_not_extracted(self):
        messages = [{"role": "user", "content": "Do not use MongoDB for this project."}]
        labels = _labels(messages)
        assert not any("mongodb" in l.lower() for l in labels), (
            f"'do not use X' was extracted as a decision: {labels}"
        )

    def test_the_original_reproduced_bug_case(self):
        """The exact reproduction from the audit — both sentences must
        produce zero decisions referencing the negated technologies."""
        messages = [{
            "role": "user",
            "content": "We are NOT using Redis. Do not use MongoDB either.",
        }]
        labels = _labels(messages)
        assert not any("redis" in l.lower() for l in labels), labels
        assert not any("mongodb" in l.lower() for l in labels), labels

    def test_never_use_x_is_not_extracted(self):
        messages = [{"role": "user", "content": "We should never use raw SQL string formatting."}]
        labels = _labels(messages)
        assert not any("sql" in l.lower() for l in labels), labels

    def test_avoid_x_is_not_extracted(self):
        messages = [{"role": "user", "content": "Avoid using pickle for untrusted data."}]
        labels = _labels(messages)
        assert not any("pickle" in l.lower() for l in labels), labels

    def test_without_x_is_not_extracted(self):
        messages = [{"role": "user", "content": "We deployed without Docker this time."}]
        labels = _labels(messages)
        assert not any("docker" in l.lower() for l in labels), labels


class TestUnrelatedEarlierNegationDoesNotSuppressLaterDecision:
    """The negation check is clause-scoped — it must not become an
    over-broad 'if the word not appears ANYWHERE in the message, drop
    every decision in it' filter, which would just trade one accuracy
    bug for another."""

    def test_negation_in_earlier_sentence_does_not_suppress_later_decision(self):
        messages = [{
            "role": "user",
            "content": (
                "The old code didn't have any caching at all. "
                "Decided to use Redis for the session cache."
            ),
        }]
        labels = _labels(messages)
        assert any("redis" in l.lower() for l in labels), (
            f"a negation in an unrelated EARLIER sentence suppressed a "
            f"legitimate later decision — the scope must be per-clause, "
            f"not per-message: {labels}"
        )


class TestExistingLegitimateDecisionsStillExtracted:
    """Guard against the fix being so broad it breaks ordinary,
    non-negated decision phrasing already covered by other tests."""

    def test_instead_of_phrasing_still_extracts_the_chosen_option(self):
        """'Decided to use bcrypt instead of argon2' — bcrypt IS the
        decision; 'instead of' here doesn't negate the item before it."""
        messages = [{
            "role": "user",
            "content": "Decided to use bcrypt instead of argon2 for password hashing.",
        }]
        labels = _labels(messages)
        assert any("bcrypt" in l.lower() for l in labels), (
            f"'instead of' phrasing wrongly suppressed the actual decision: {labels}"
        )

    def test_switching_from_x_to_y_still_extracts_both_sides(self):
        messages = [{
            "role": "user",
            "content": "Switching from React to Next.js for better SEO.",
        }]
        result = extractor.heuristic_extract(messages)
        assert len(result.superseded) > 0

    def test_plain_positive_decision_unaffected(self):
        messages = [{"role": "user", "content": "Use PostgreSQL for the primary datastore."}]
        labels = _labels(messages)
        assert any("postgresql" in l.lower() for l in labels), labels
