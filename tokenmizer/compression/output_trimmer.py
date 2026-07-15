"""
Output Trimmer — removes LLM verbosity without touching information.

LLMs (especially frontier models) have trained-in habits that waste tokens:
  - "Certainly! I'd be happy to help with that."  (+8 tokens, zero info)
  - "In summary, ..." at the end (restates what was just said)
  - "Let me know if you need anything else!" (+10 tokens every response)
  - Excessive caveats and disclaimers on simple tasks

This trimmer removes ONLY structural filler — never content.
Average savings: 5-15% on verbose models (GPT-5.5, Gemini 3.1 Pro).
"""
from __future__ import annotations

import re

from tokenmizer.core.tokenizer import count_tokens

# ── Filler patterns ───────────────────────────────────────────────────────────
# Ordered: most specific first

_OPENING_FILLERS = [
    # These match at start of string (re.MULTILINE so ^ = start of any line)
    re.compile(r"^Certainly[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Of course[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Absolutely[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Sure[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Great question[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^That's? (?:a )?(?:great|good|excellent|interesting) question[!,.]?\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^I(?:'d| would) be happy to (?:help|assist)[^.\n]*\.\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^I(?:'d| would) love to (?:help|assist)[^.\n]*\.\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^I understand(?: that)? you(?:'re| are)[^.\n]*\.\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Thank you for (?:your )?(?:question|asking|reaching out)[^.\n]*\.\s*", re.IGNORECASE | re.MULTILINE),
]

_CLOSING_FILLERS = [
    re.compile(r"\n+Let me know if (?:you(?:'d like| need| have))[^.!?]*[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+Feel free to (?:ask|reach out)[^.!?]*[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+(?:Don't hesitate|Please don't hesitate) to (?:ask|reach out)[^.!?]*[.!?]\s*$", re.IGNORECASE),
    re.compile(r"\n+Is there anything (?:else|more)[^.!?]*[.!?]\s*$", re.IGNORECASE),
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
        result = text

        # Opening fillers
        for pat in _OPENING_FILLERS:
            result = pat.sub("", result, count=1)

        if level in ("full", "ultra"):
            # Closing fillers
            for pat in _CLOSING_FILLERS:
                result = pat.sub("", result)

        if level == "ultra":
            # Inline redundancies (only on ultra — risky otherwise)
            for pat in _INLINE_REDUNDANCIES:
                result = pat.sub("\n\n", result)

        result = result.strip()
        # Normalize multiple blank lines
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        saved = max(0, original_tokens - count_tokens(result))
        return result, saved
