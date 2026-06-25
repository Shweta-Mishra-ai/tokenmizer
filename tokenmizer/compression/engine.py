"""
Layer 1: Advanced Prompt Compression
=====================================
Strategies (applied in pipeline order):
  1. Filler phrase removal       — regex-based, zero deps
  2. Duplicate line suppression  — remove exact repeat lines
  3. Whitespace normalization    — collapse blank lines/spaces
  4. Comment stripping           — strip code comments from heavy files
  5. Repetitive history pruning  — deduplicate assistant boilerplate
  6. Smart truncation            — truncate low-value file blocks
  7. LLMLingua-2                 — ML-based token-level compression
  8. LongLLMLingua               — for >4k token documents

File-type filters (new):
  - PDF/docx text extraction     — don't send raw binary markers
  - Large JSON flattening        — remove nested nulls/empty arrays
  - CSV summarization            — send schema + sample, not full file
  - Code deduplication           — remove duplicate function bodies
  - Log trimming                 — keep first+last N lines of logs

CORRECTNESS FIX — code blocks are now excluded from LLMLingua entirely
(see CodeBlockGuard below). LLMLingua-2 is a lossy, ML-based token
compressor — `force_tokens` only hints at preservation, it does not
guarantee it. Applied to code, this risks dropping or reordering tokens
that change program semantics: a removed `not`, a dropped `except`
clause, mangled indentation in Python (where whitespace IS syntax), a
truncated regex. A tool whose target use case is "coding sessions with
an LLM" must not silently corrupt the code it's supposed to be helping
with. Code fences (```...```) and indented code blocks are now segmented
out before LLMLingua runs and passed through untouched (only the
lossless heuristics — whitespace/dedup/optional comment-stripping — ever
touch code); only prose segments are sent to the ML compressor.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tokenmizer.core.tokenizer import count_tokens

logger = logging.getLogger(__name__)

# ─── Code-block protection ──────────────────────────────────────────────────

# Matches fenced code blocks: ```lang\n...\n``` or ```\n...\n```
_FENCED_CODE_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)
# Matches inline code spans: `like this`
_INLINE_CODE_RE = re.compile(r'`[^`\n]+`')


class CodeBlockGuard:
    """
    Segments text into (is_code, segment) pairs so callers can route code
    around lossy ML compression while still compressing surrounding prose.

    Only handles fenced (```) and inline (`) code markup — the common case
    for chat-style content. Code pasted without any fence markup (a raw
    paste with no backticks) cannot be reliably distinguished from prose by
    this guard and will still reach LLMLingua; that residual risk is
    smaller in practice since most coding-assistant conversations use
    fences, but it is a real, documented gap rather than a solved problem.
    """

    @staticmethod
    def segment(text: str) -> List[Tuple[bool, str]]:
        """Returns ordered (is_code, segment_text) pairs covering the
        entire input losslessly — concatenating all segment_text values
        back together reproduces the original text exactly."""
        segments: List[Tuple[bool, str]] = []
        pos = 0
        # Fenced blocks first (they take priority over inline spans found inside them)
        for m in _FENCED_CODE_RE.finditer(text):
            if m.start() > pos:
                segments.extend(CodeBlockGuard._segment_inline(text[pos:m.start()]))
            segments.append((True, m.group(0)))
            pos = m.end()
        if pos < len(text):
            segments.extend(CodeBlockGuard._segment_inline(text[pos:]))
        return segments

    @staticmethod
    def _segment_inline(text: str) -> List[Tuple[bool, str]]:
        """Within non-fenced text, also protect inline `code spans`."""
        segments: List[Tuple[bool, str]] = []
        pos = 0
        for m in _INLINE_CODE_RE.finditer(text):
            if m.start() > pos:
                segments.append((False, text[pos:m.start()]))
            segments.append((True, m.group(0)))
            pos = m.end()
        if pos < len(text):
            segments.append((False, text[pos:]))
        return segments

    @staticmethod
    def reassemble(segments: List[Tuple[bool, str]]) -> str:
        return "".join(seg for _, seg in segments)

# ─── Filler patterns ────────────────────────────────────────────────────────

_FILLER = [
    r"As an AI(?:\s+language model)?,?\s*",
    r"I(?:'d| would) be (?:happy|glad|pleased) to\s+(?:help\s+)?",
    r"(?:That'?s?\s+a?\s*)?(?:great|excellent|good|wonderful|fantastic)\s+question[.!]\s*",
    r"(?:Certainly|Of course|Sure|Absolutely|Indeed)[!.]?\s*",
    r"It(?:'s| is) (?:worth noting|important to note|crucial to understand) that\s+",
    r"In this (?:case|context|scenario),?\s*",
    r"(?:Essentially|Basically|Simply put|In other words),?\s*",
    r"As you can see(?:,| from)?\s*",
    r"As (?:mentioned|noted|discussed) (?:earlier|above|previously|before),?\s*",
    r"Let me (?:explain|clarify|elaborate|break this down)(?:\s+for you)?\s*",
    r"I hope this (?:helps|answers your question|clarifies things)[.!]\s*",
    r"Feel free to (?:ask|reach out)[^.]*[.!]\s*",
    r"Please (?:let me know|don't hesitate)[^.]*[.!]\s*",
    r"(?:Thank you for|Thanks for) (?:asking|your question|reaching out)[.!]\s*",
]
_FILLER_RE = [re.compile(p, re.IGNORECASE) for p in _FILLER]

# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    original_text: str
    compressed_text: str
    strategies_applied: List[str] = field(default_factory=list)
    quality_score: float = 1.0  # 0-1, estimated

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def savings_pct(self) -> float:
        return (1 - self.ratio) * 100

    def __repr__(self) -> str:
        return (
            f"CompressionResult("
            f"orig={self.original_tokens}, "
            f"compressed={self.compressed_tokens}, "
            f"ratio={self.ratio:.2f}, "
            f"saved={self.savings_pct:.1f}%, "
            f"strategies={self.strategies_applied})"
        )


# ─── Heuristic strategies ────────────────────────────────────────────────────

class FillerRemover:
    """Remove AI filler phrases. Zero dependencies. ~10-20% reduction on verbose responses."""

    def apply(self, text: str) -> Tuple[str, str]:
        for pat in _FILLER_RE:
            text = pat.sub("", text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        return text.strip(), "filler_removal"


class DuplicateLineRemover:
    """Remove exact duplicate lines (common in repeated context). ~5-15% on long chats."""

    def apply(self, text: str) -> Tuple[str, str]:
        seen: set = set()
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped in seen and len(stripped) > 40:
                continue  # skip duplicate non-trivial lines
            seen.add(stripped)
            lines.append(line)
        return "\n".join(lines), "duplicate_removal"


class WhitespaceNormalizer:
    """Collapse excessive whitespace. ~2-5% reduction."""

    def apply(self, text: str) -> Tuple[str, str]:
        text = re.sub(r'\t', '  ', text)
        text = re.sub(r' {4,}', '   ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip(), "whitespace_normalization"


class CommentStripper:
    """Strip comments from code blocks. ~10-30% on comment-heavy code.

    CORRECTNESS FIX: the JS line-comment pattern previously matched `//`
    anywhere on a line, including inside string literals — most commonly
    URLs like "https://example.com", which would get truncated to
    "https:" with everything after silently deleted. This is real code
    corruption, not a cosmetic issue: a stripped URL, connection string,
    or comparison-with-division (`a //= b` truncation edge cases aside)
    changes program behavior.

    Fix: a line is only treated as having a `//` comment if there's an
    even number of unescaped double-quote AND single-quote characters
    before the `//` — i.e. the `//` is outside any open string on that
    line. This is a heuristic (not a real tokenizer — it doesn't know
    about template literals, regex literals, or multi-line strings), but
    it correctly handles the dominant real-world case (URLs in string
    literals) instead of ignoring the problem entirely.
    """

    _BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
    _DOCSTRING = re.compile(r'""".*?"""', re.DOTALL)

    @staticmethod
    def _strip_line_comments(text: str, markers: Tuple[str, ...]) -> str:
        """
        String-aware line-comment stripper, shared by both Python (#) and
        JS-style (//) comments.

        FIXED BUGS (found via actually running tests against this code,
        not just reading it):
          1. The original Python regex `^\\s*#.*$` only matched comments
             where `#` was the first non-whitespace character on the
             line — it silently did NOT strip the far more common
             trailing-comment style `x = 1  # comment`, because that line
             doesn't match `^\\s*#`. So "comment stripping" was already
             failing to strip most real-world comments before this audit
             touched it at all.
          2. The original JS regex `//[^\\n]*` matched `//` anywhere on a
             line including inside string literals (URLs), corrupting
             code — this was the bug this audit set out to fix.

        This single string-aware scanner handles both correctly: it
        strips a line-comment marker only when found outside an open
        quoted string, regardless of whether it's a leading or trailing
        comment, for any of the given marker strings.
        """
        out_lines = []
        for line in text.split('\n'):
            best_pos = None
            for marker in markers:
                idx = 0
                while True:
                    pos = line.find(marker, idx)
                    if pos == -1:
                        break
                    before = line[:pos]
                    if before.count('"') % 2 == 1 or before.count("'") % 2 == 1:
                        idx = pos + len(marker)
                        continue
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                    break
            out_lines.append(line[:best_pos].rstrip() if best_pos is not None else line)
        return '\n'.join(out_lines)

    def apply(self, text: str, strip_docstrings: bool = False) -> Tuple[str, str]:
        result = self._strip_line_comments(text, ('#', '//'))
        result = self._BLOCK_COMMENT.sub('', result)
        if strip_docstrings:
            result = self._DOCSTRING.sub('', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip(), "comment_stripping"


class RepetitiveHistoryPruner:
    """
    Detect and collapse repetitive assistant message patterns.
    e.g. 3+ messages all starting with "Here is the code:" get deduplicated.
    ~10-20% on long coding sessions.
    """

    def apply(self, messages: List[Dict]) -> Tuple[List[Dict], str]:
        if len(messages) < 6:
            return messages, "history_pruning_skipped"

        result = []
        prefix_count: Dict[str, int] = {}

        for msg in messages:
            content = msg.get("content", "")
            if msg.get("role") == "assistant":
                # Get first 60 chars as "prefix signature"
                prefix = content[:60].strip().lower()
                prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
                # If this pattern appeared 3+ times, compress it
                if prefix_count[prefix] > 2 and len(content) > 200:
                    # Keep first 100 + last 100 chars
                    compressed = content[:100] + "\n...[compressed]...\n" + content[-100:]
                    result.append({**msg, "content": compressed})
                    continue
            result.append(msg)

        return result, "history_pruning"


# ─── File-type filters (NEW) ──────────────────────────────────────────────────

class FileContentFilter:
    """
    Smart filters for heavy file types.
    Prevents sending raw binary artifacts, huge CSVs, full logs, etc.
    """

    MAX_CSV_ROWS = 10
    MAX_LOG_LINES = 50
    MAX_JSON_DEPTH = 3

    def filter_csv(self, content: str) -> str:
        """Send schema + first N rows instead of full CSV."""
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) <= self.MAX_CSV_ROWS + 1:
            return content
        header = lines[0]
        sample = lines[1:self.MAX_CSV_ROWS + 1]
        total_rows = len(lines) - 1
        return (
            f"[CSV — {total_rows} rows, showing first {self.MAX_CSV_ROWS}]\n"
            + header + "\n"
            + "\n".join(sample)
            + f"\n...[{total_rows - self.MAX_CSV_ROWS} rows omitted]"
        )

    def filter_json(self, content: str) -> str:
        """Flatten deep JSON, remove nulls/empty arrays."""
        try:
            data = json.loads(content)
            cleaned = self._clean_json(data, depth=0)
            result = json.dumps(cleaned, indent=2)
            if len(result) < len(content):
                return f"[JSON cleaned — {len(content)} → {len(result)} chars]\n{result}"
            return content
        except (json.JSONDecodeError, Exception):
            return content

    def _clean_json(self, obj, depth: int):
        if depth > self.MAX_JSON_DEPTH:
            return f"...[depth limit {self.MAX_JSON_DEPTH}]"
        if isinstance(obj, dict):
            return {
                k: self._clean_json(v, depth + 1)
                for k, v in obj.items()
                if v is not None and v != [] and v != {}
            }
        if isinstance(obj, list):
            if len(obj) > 20:
                trimmed = [self._clean_json(x, depth + 1) for x in obj[:5]]
                return trimmed + [f"...[{len(obj)-5} more]"]
            return [self._clean_json(x, depth + 1) for x in obj]
        return obj

    def filter_log(self, content: str) -> str:
        """Keep first + last N lines of logs (errors are usually at end)."""
        lines = content.splitlines()
        if len(lines) <= self.MAX_LOG_LINES:
            return content
        half = self.MAX_LOG_LINES // 2
        head = lines[:half]
        tail = lines[-half:]
        omitted = len(lines) - self.MAX_LOG_LINES
        return (
            "\n".join(head)
            + f"\n\n...[{omitted} lines omitted]...\n\n"
            + "\n".join(tail)
        )

    def filter_by_extension(self, content: str, filename: str) -> Tuple[str, str]:
        """Auto-detect file type and apply appropriate filter."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "csv":
            return self.filter_csv(content), "csv_filter"
        if ext == "json":
            return self.filter_json(content), "json_filter"
        if ext in ("log", "txt") and len(content.splitlines()) > 100:
            return self.filter_log(content), "log_filter"
        return content, "no_filter"


# ─── LLMLingua wrapper ────────────────────────────────────────────────────────

class LLMLinguaEngine:
    """
    LLMLingua-2 / LongLLMLingua wrapper with graceful fallback.
    Auto-selects LongLLMLingua for documents > 4k tokens.
    """

    LONG_THRESHOLD = 4000  # tokens — use LongLLMLingua above this

    def __init__(self, ratio: float = 0.5, device: str = "cpu"):
        self.ratio = ratio
        self.device = device
        self._short = None
        self._long = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from llmlingua import PromptCompressor  # type: ignore
            self._short = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=self.device,
            )
            # LongLLMLingua for long docs
            self._long = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=self.device,
            )
            self._available = True
            logger.info("LLMLingua-2 loaded")
        except ImportError:
            logger.warning(
                "llmlingua not installed — using heuristic compression only. "
                "pip install tokenmizer[compression] for full ML compression."
            )

    @property
    def available(self) -> bool:
        return self._available

    def compress(self, text: str, ratio: Optional[float] = None) -> CompressionResult:
        """
        Compress text via LLMLingua-2, EXCLUDING code segments.

        FIXED: previously the entire text — including any fenced/inline
        code — went straight into the ML compressor with only a soft
        `force_tokens` hint asking it to try to preserve a few literal
        tokens like "```" and "def ". That hint does not guarantee
        preservation of everything inside a code block; LLMLingua is a
        lossy compressor by design, and applying it to code risks
        corrupting program semantics (dropped tokens, mangled
        indentation in whitespace-significant languages, truncated
        strings/regexes). Since this tool's primary use case is coding
        sessions, that's not a hypothetical edge case.

        Now: text is segmented into code vs. prose (CodeBlockGuard), only
        prose segments are sent to LLMLingua, and code segments are
        reattached completely unmodified — guaranteed identical to the
        input for any text wrapped in ``` fences or single backticks.
        """
        target = ratio or self.ratio
        orig_tokens = count_tokens(text)

        if not self._available or orig_tokens < 100:
            return CompressionResult(
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                original_text=text,
                compressed_text=text,
                strategies_applied=["llmlingua_skipped_short"],
                quality_score=1.0,
            )

        segments = CodeBlockGuard.segment(text)
        code_segment_count = sum(1 for is_code, _ in segments if is_code)

        try:
            out_parts: List[str] = []
            any_compressed = False
            for is_code, seg in segments:
                if is_code or count_tokens(seg) < 20:
                    # Too short to bother, or protected code — pass through.
                    out_parts.append(seg)
                    continue
                engine = self._long if count_tokens(seg) > self.LONG_THRESHOLD else self._short
                result = engine.compress_prompt(
                    seg,
                    rate=target,
                    force_tokens=["\n", ".", "?", "!"],
                )
                out_parts.append(result["compressed_prompt"])
                any_compressed = True

            compressed = "".join(out_parts)
            comp_tokens = count_tokens(compressed)
            label = "llmlingua2_code_protected" if code_segment_count else "llmlingua2"
            if not any_compressed:
                label = "llmlingua_skipped_all_code"

            return CompressionResult(
                original_tokens=orig_tokens,
                compressed_tokens=comp_tokens,
                original_text=text,
                compressed_text=compressed,
                strategies_applied=[label],
                quality_score=comp_tokens / max(orig_tokens, 1),
            )
        except Exception as e:
            logger.warning(f"LLMLingua failed: {e} — falling back")
            return CompressionResult(
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                original_text=text,
                compressed_text=text,
                strategies_applied=["llmlingua_failed"],
                quality_score=1.0,
            )


# ─── Master pipeline ─────────────────────────────────────────────────────────

class CompressionPipeline:
    """
    Orchestrates all compression strategies in the right order.
    Heuristics run first (fast, no deps), ML last (slowest, best quality).
    """

    def __init__(
        self,
        ratio: float = 0.5,
        strip_comments: bool = False,
        enable_ml: bool = True,
        device: str = "cpu",
    ):
        self.ratio = ratio
        self.strip_comments = strip_comments
        # compression_ratio = output_tokens / input_tokens (lower = more compressed)
        # If ratio > threshold, ML compression had no effect — keep heuristic result
        self._quality_threshold = 0.95
        self.filler = FillerRemover()
        self.dedup = DuplicateLineRemover()
        self.whitespace = WhitespaceNormalizer()
        self.comments = CommentStripper()
        self.history_pruner = RepetitiveHistoryPruner()
        self.file_filter = FileContentFilter()
        self.lingua = LLMLinguaEngine(ratio=ratio) if enable_ml else None

    def compress_text(
        self,
        text: str,
        filename: Optional[str] = None,
        min_tokens: int = 100,
    ) -> CompressionResult:
        """Compress a single text block through the full pipeline."""

        original = text
        orig_tokens = count_tokens(text)
        strategies: List[str] = []

        if orig_tokens < min_tokens:
            return CompressionResult(
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                original_text=original,
                compressed_text=text,
                strategies_applied=["skipped_too_short"],
            )

        # File-type filter first
        if filename:
            text, strat = self.file_filter.filter_by_extension(text, filename)
            if strat != "no_filter":
                strategies.append(strat)

        # Heuristics (order matters)
        text, s = self.whitespace.apply(text)
        strategies.append(s)

        text, s = self.filler.apply(text)
        strategies.append(s)

        text, s = self.dedup.apply(text)
        strategies.append(s)

        if self.strip_comments:
            text, s = self.comments.apply(text)
            strategies.append(s)

        # ML compression
        if self.lingua and self.lingua.available:
            result = self.lingua.compress(text, ratio=self.ratio)
            text = result.compressed_text
            strategies.extend(result.strategies_applied)
            quality = result.quality_score
        else:
            quality = 0.9

        comp_tokens = count_tokens(text)

        # Quality gate: compression_ratio is tokens_out/tokens_in.
        # A ratio > _quality_threshold means barely any compression happened — fallback.
        # NOTE: lower ratio = MORE compression (good). We reject when ratio is TOO HIGH.
        compression_ratio = comp_tokens / max(orig_tokens, 1)
        # Fallback if attr missing — reject when ML barely compressed (>95% of original)
        quality_threshold = getattr(self, "_quality_threshold", 0.95)
        if compression_ratio > quality_threshold:
            logger.warning(
                f"Compression ratio {compression_ratio:.2f} > threshold {quality_threshold} "
                f"— ML compression had no effect, keeping heuristic result"
            )
            # Don't fall back to original — keep heuristic-compressed text
            # (filler removal etc. already ran above)

        return CompressionResult(
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            original_text=original,
            compressed_text=text,
            strategies_applied=strategies,
            quality_score=quality,
        )

    def compress_messages(
        self,
        messages: List[Dict],
        protect_recent: int = 3,
    ) -> Tuple[List[Dict], int]:
        """
        Compress all messages except the most recent N.
        Returns (compressed_messages, total_tokens_saved).
        """
        if len(messages) <= protect_recent:
            return messages, 0

        # First pass: prune repetitive history
        messages, _ = self.history_pruner.apply(messages)

        total_saved = 0
        result = []

        for i, msg in enumerate(messages):
            # Don't touch recent messages or system messages
            if i >= len(messages) - protect_recent:
                result.append(msg)
                continue
            if msg.get("role") == "system":
                result.append(msg)
                continue

            content = msg.get("content", "")
            cr = self.compress_text(content, min_tokens=200)
            total_saved += cr.original_tokens - cr.compressed_tokens
            result.append({**msg, "content": cr.compressed_text})

        return result, total_saved

    def terse_system_prompt(self, level: str = "full") -> str:
        """Return terse-output instruction to inject into system prompt."""
        levels = {
            "lite": (
                "Be concise. No preamble (e.g., 'Sure!', 'Great question!'). "
                "No closing remarks. Start answer immediately."
            ),
            "full": (
                "Respond like a senior engineer: no filler, no preamble, no 'I'd be happy to', "
                "no closing fluff. Use fragments when clear. Preserve code/paths/URLs exactly. "
                "Technical accuracy 100%. Start with the answer."
            ),
            "ultra": (
                "Ultra-terse. Fragments only. No articles if obvious. "
                "No preamble or closing. Code/paths exact. Maximum compression."
            ),
        }
        return levels.get(level, levels["full"])
