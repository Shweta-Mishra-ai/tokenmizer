"""
Output Trimmer — removes LLM verbosity without touching information.

LLMs (especially frontier models) have trained-in habits that waste tokens:
  - "Certainly! I'd be happy to help with that."  (+8 tokens, zero info)
  - "In summary, ..." at the end (restates what was just said)
  - "Let me know if you need anything else!" (+10 tokens every response)
  - Excessive caveats and disclaimers on simple tasks

This trimmer removes ONLY structural filler — never content. That claim is
load-bearing: this is the one stage that edits the text a person reads,
after the model has already written it, and a deletion here is invisible —
there is nothing to compare the answer against.

It used to be false in four ways, all reproducible at the default level:

  - `^` with re.MULTILINE matched the start of ANY line, so `Sure! ` was
    deleted from inside a fenced code block — corrupting code the reader
    then pastes.
  - The same anchor cut the first word off a real sentence mid-answer:
    "Sure, the second is slower, but it is correct under concurrency."
  - `Is there anything (?:else|more)[^.!?]*[.!?]$` ate a genuine question
    about specific state — "Is there anything else in the 003 batch you
    want rolled back?" — so the reader never saw that they were asked.
  - On `ultra`, the "In summary, ..." rule deleted the paragraph carrying
    the only numbers in the answer.

Both halves of the fix already existed elsewhere in this package and were
simply never applied here. `CodeBlockGuard` (engine.py) routes code around
the lossy prompt-side stages; `_FILLER` there was anchored to a sentence
start after it was found corrupting words mid-token. The response path got
neither. It now gets both, plus one rule the prompt side does not need:
a filler phrase is only filler if what goes with it carries no
information, so anything holding a number, a code span, a path or a URL
stays.

Average savings: 5-15% on verbose models.
"""
from __future__ import annotations

import re

from tokenmizer.compression.engine import CodeBlockGuard
from tokenmizer.core.tokenizer import count_tokens

# ── Filler patterns ───────────────────────────────────────────────────────────
# Ordered: most specific first.
#
# Anchored with \A, not ^ with re.MULTILINE: an opening pleasantry is only
# an opening if it opens the RESPONSE. Anywhere else it is a sentence.
_OPENING_FILLERS = [
    re.compile(r"\ACertainly[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\AOf course[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\AAbsolutely[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\ASure[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\AGreat question[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\AThat'?s? (?:a )?(?:great|good|excellent|interesting) question[!,.]?\s+", re.IGNORECASE),
    re.compile(r"\AI(?:'d| would) be happy to (?:help|assist)[^.\n]*\.\s*", re.IGNORECASE),
    re.compile(r"\AI(?:'d| would) love to (?:help|assist)[^.\n]*\.\s*", re.IGNORECASE),
    re.compile(r"\AI understand(?: that)? you(?:'re| are)[^.\n]*\.\s*", re.IGNORECASE),
    re.compile(r"\AThank you for (?:your )?(?:question|asking|reaching out)[^.\n]*\.\s*", re.IGNORECASE),
]

# Closing boilerplate. Each is additionally gated by _carries_information()
# below, so a sign-off that happens to carry a fact is kept whole.
_CLOSING_FILLERS = [
    re.compile(r"\n+Let me know if (?:you(?:'d like| need| have))[^.!?]*[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+Feel free to (?:ask|reach out)[^.!?]*[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+(?:Don't hesitate|Please don't hesitate) to (?:ask|reach out)[^.!?]*[.!?]\s*$", re.IGNORECASE),
    # "anything else" must be followed by the generic continuation. Without
    # that, "Is there anything else in the 003 batch you want rolled back?"
    # matched and a real question was deleted.
    re.compile(
        r"\n+Is there anything (?:else|more)"
        r"(?: (?:I can (?:help|do|assist)|you(?:'d| would) like|you need|"
        r"you want))?[^.!?]{0,40}[.!?]\s*$",
        re.IGNORECASE),
    re.compile(r"\n+Hope (?:this|that) helps?[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+I hope (?:this|that) (?:answer|explanation|helps?)[^.!?]*[.!?]\s*$", re.IGNORECASE),
]

_INLINE_REDUNDANCIES = [
    # "In summary, ..." paragraphs that just restate the answer
    re.compile(r"\n+In summary[,:]?\s*[^\n]{0,200}\n+", re.IGNORECASE),
    re.compile(r"\n+To summarize[,:]?\s*[^\n]{0,200}\n+", re.IGNORECASE),
    re.compile(r"\n+In conclusion[,:]?\s*[^\n]{0,200}\n+", re.IGNORECASE),
    re.compile(r"\n+To recap[,:]?\s*[^\n]{0,200}\n+", re.IGNORECASE),
    # Excessive disclaimer on simple code/math tasks
    re.compile(r"\n+Note: This (?:code|implementation|solution) (?:is|should be) (?:tested|reviewed)[^.]*\.\s*\n", re.IGNORECASE),
]

# What makes a fragment worth keeping even though it matched a filler
# pattern: a number, a code span, a path, a URL, or a flag. A sign-off
# carrying any of these is not a sign-off.
# Bounded repeats, for the same reason as semantic_cache._LITERAL: an
# unbounded repeat in front of a required literal makes the engine
# rescan the run from every start position, which is quadratic. The
# fragments reaching this are usually short, but a closing-filler match
# carries `[^.!?]*` and a response need not contain sentence
# punctuation, so "usually short" is not a bound.
_INFORMATION = re.compile(
    r"""
      \d                                  # any digit: 003, 2.1%, p99, 120ms
    | `[^`\n]{1,200}`                     # an inline code span
    | https?://                           # a URL
    | (?:^|[\s(])[/~][\w./-]{1,120}       # an absolute or ~ path
    | \b[\w-]{1,64}\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sql|ya?ml|json|toml|md|sh)\b
    | (?:^|\s)--?[a-z][\w-]{0,60}         # a command-line flag
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def _carries_information(fragment: str) -> bool:
    """True when a filler match is not safe to delete.

    The filler patterns describe a SHAPE ("Is there anything else …"),
    and a shape cannot tell boilerplate from a real question about real
    state. This is the second test: if the text the pattern swallowed
    names a number, a file, a path, a flag or a URL, it is answering or
    asking something specific and is kept whole, however boilerplate it
    looks.
    """
    return bool(_INFORMATION.search(fragment))


def _sub_if_contentless(pattern: re.Pattern, text: str, repl: str = "",
                        count: int = 0) -> str:
    """re.sub, but a match that carries information is left in place."""
    return pattern.sub(
        lambda m: m.group(0) if _carries_information(m.group(0)) else repl,
        text, count=count,
    )


class OutputTrimmer:

    def trim(self, text: str, level: str = "full") -> tuple[str, int]:
        """
        Remove structural filler from LLM output.

        Args:
            text: raw LLM response
            level: "lite" (openings only) | "full" | "ultra"

        Returns:
            (trimmed_text, tokens_saved)
        """
        if not text or len(text) < 20:
            return text, 0

        original_tokens = count_tokens(text)

        # Prose only. Fenced blocks and inline spans pass through
        # byte-identical, so no rule below can reach inside code — the same
        # guarantee CodeBlockGuard already gives the prompt-side stages.
        segments = CodeBlockGuard.segment(text)
        prose_indices = [i for i, (is_code, _) in enumerate(segments) if not is_code]
        parts = [seg for _, seg in segments]

        if prose_indices:
            # An opening pleasantry can only open the FIRST prose segment,
            # and only when nothing precedes it.
            first = prose_indices[0]
            if first == 0:
                for pat in _OPENING_FILLERS:
                    parts[first] = pat.sub("", parts[first], count=1)

            if level in ("full", "ultra"):
                # A closing sign-off can only close the LAST prose segment,
                # and only when no code follows it.
                last = prose_indices[-1]
                if last == len(parts) - 1:
                    for pat in _CLOSING_FILLERS:
                        parts[last] = _sub_if_contentless(pat, parts[last])

            if level == "ultra":
                # Inline redundancies (only on ultra — risky otherwise)
                for i in prose_indices:
                    for pat in _INLINE_REDUNDANCIES:
                        parts[i] = _sub_if_contentless(pat, parts[i], "\n\n")

        # Normalize multiple blank lines — per prose segment, never over the
        # joined result, or a blank line inside a fenced block would go too.
        for i in prose_indices:
            parts[i] = re.sub(r"\n{3,}", "\n\n", parts[i])
        result = "".join(parts).strip()

        saved = max(0, original_tokens - count_tokens(result))
        return result, saved
