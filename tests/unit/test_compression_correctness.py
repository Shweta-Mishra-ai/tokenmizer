"""
Unit tests — compression engine correctness fixes.

Covers two real bugs found in the audit:

1. LLMLingua (lossy ML compression) was applied to the ENTIRE message
   text, including fenced/inline code, with only a soft `force_tokens`
   hint asking it to try to preserve a few literal strings. That is not
   a guarantee, and applying lossy compression to code risks silently
   corrupting program semantics. Fix: CodeBlockGuard segments text into
   code vs. prose; only prose reaches LLMLingua, code passes through
   byte-for-byte unchanged.

2. CommentStripper's JS line-comment regex (`//[^\\n]*`) matched `//`
   anywhere on a line, including inside string literals — most commonly
   URLs like "https://example.com", silently truncating them. Fix: only
   treat `//` as a comment marker when it's outside any open quoted
   string on that line.
"""
import pytest

from tokenmizer.compression.engine import CodeBlockGuard, CommentStripper


class TestCodeBlockGuard:

    def test_round_trip_is_lossless(self):
        """Segmenting and reassembling must reproduce the exact original
        text — any divergence here means we'd be corrupting content
        even before LLMLingua gets involved."""
        sample = (
            "Some prose here.\n\n"
            "```python\ndef foo(x):\n    return x + 1\n```\n\n"
            "More prose with `inline_code` in it.\n\n"
            "Final paragraph."
        )
        segments = CodeBlockGuard.segment(sample)
        assert CodeBlockGuard.reassemble(segments) == sample

    def test_fenced_code_block_detected(self):
        text = "Explanation.\n```python\nx = 1\n```\nMore text."
        segments = CodeBlockGuard.segment(text)
        code_segments = [s for is_code, s in segments if is_code]
        assert any("x = 1" in s for s in code_segments)

    def test_inline_code_detected(self):
        text = "Use the `requests` library for this."
        segments = CodeBlockGuard.segment(text)
        code_segments = [s for is_code, s in segments if is_code]
        assert any("requests" in s for s in code_segments)

    def test_prose_not_marked_as_code(self):
        text = "This is plain prose with no code at all in it whatsoever."
        segments = CodeBlockGuard.segment(text)
        assert all(not is_code for is_code, _ in segments)

    def test_url_inside_fenced_code_survives_segmentation(self):
        """The actual real-world failure case this whole fix targets:
        code containing a URL must come out of segmentation completely
        unchanged, ready to skip LLMLingua entirely."""
        text = '```js\nconst url = "https://api.example.com/v1/users";\n```'
        segments = CodeBlockGuard.segment(text)
        code_segments = [s for is_code, s in segments if is_code]
        assert len(code_segments) == 1
        assert "https://api.example.com/v1/users" in code_segments[0]


class TestCommentStripperURLBug:
    """
    FIXED BUG: stripping JS-style `//` comments previously also stripped
    everything after `//` inside string literals, since the old regex
    (`//[^\\n]*`) had no concept of "inside a string." A URL like
    "https://example.com" would be silently truncated to "https:" with
    the rest of the line deleted — real code corruption, not cosmetic.
    """

    def setup_method(self):
        self.stripper = CommentStripper()

    def test_url_in_double_quoted_string_survives(self):
        code = 'const url = "https://example.com/api"; // fetch data'
        result, _ = self.stripper.apply(code)
        assert "https://example.com/api" in result
        assert "fetch data" not in result  # the actual comment IS stripped

    def test_url_in_single_quoted_string_survives(self):
        code = "const url = 'https://test.com/v2'; // comment"
        result, _ = self.stripper.apply(code)
        assert "https://test.com/v2" in result

    def test_real_comment_still_stripped(self):
        code = "const x = 5; // this is a real comment"
        result, _ = self.stripper.apply(code)
        assert result == "const x = 5;"

    def test_multiple_urls_and_comments_on_different_lines(self):
        code = (
            'const a = "https://one.com"; // comment one\n'
            'const b = "https://two.com"; // comment two'
        )
        result, _ = self.stripper.apply(code)
        assert "https://one.com" in result
        assert "https://two.com" in result
        assert "comment one" not in result
        assert "comment two" not in result

    def test_python_comments_unaffected(self):
        code = "x = 1  # this should still be removed\ny = 2"
        result, _ = self.stripper.apply(code)
        assert "should still be removed" not in result
        assert "y = 2" in result

    def test_block_comments_unaffected(self):
        code = "x = 1; /* block comment */ y = 2;"
        result, _ = self.stripper.apply(code)
        assert "block comment" not in result

    def test_trailing_python_comment_now_stripped(self):
        """FIXED PRE-EXISTING BUG (found via testing, not present in the
        original audit's bug list — discovered while writing tests for
        the JS-comment fix): the original _PYTHON_COMMENT regex
        (`^\\s*#.*$`) only matched comments where `#` was the FIRST
        non-whitespace char on the line. Trailing comments like
        `x = 1  # comment` — the more common real-world style — were
        never stripped at all, silently. 'Comment stripping' was already
        failing on the dominant case before this audit touched the file."""
        code = "x = 1  # this should be removed\ny = 2"
        result, _ = self.stripper.apply(code)
        assert "this should be removed" not in result
        assert "y = 2" in result

    def test_fstring_url_with_trailing_comment(self):
        """Combined stress case: an f-string containing a URL (with `//`)
        AND a trailing `#` comment on the same line. Both must be handled
        correctly — URL preserved, comment stripped."""
        code = 'url = f"https://x.com/{id}"  # fetch user'
        result, _ = self.stripper.apply(code)
        assert "https://x.com/{id}" in result
        assert "fetch user" not in result

    def test_hex_color_with_hash_not_treated_as_comment(self):
        """A `#` inside a string (e.g. a CSS hex color) must not be
        mistaken for a Python comment marker — same string-awareness
        fix that protects URLs must also protect this case."""
        code = 'const color = "#FF0000"; // red color'
        result, _ = self.stripper.apply(code)
        assert "#FF0000" in result
        assert "red color" not in result

    def test_no_comment_present_text_unchanged(self):
        code = "x = 1\ny = 2"
        result, _ = self.stripper.apply(code)
        assert result == "x = 1\ny = 2"


class TestHeuristicsDoNotTouchCode:
    """The heuristic stages ran on the whole message, code included.

    CodeBlockGuard existed to route fenced code around LLMLingua, but
    whitespace normalisation, filler removal, duplicate-line removal and
    comment stripping all ran on the raw text first. Measured on the default
    configuration: a Python function in any message older than the last
    three had every indentation level collapsed to a single space — the
    model was shown syntactically invalid code — and repeated log lines in a
    pasted trace were deleted as "duplicates". Both are information loss on
    the input, on the default path.
    """

    PADDING = "\nSome padding prose so the message clears the minimum size. " * 12

    def _compress(self, text: str) -> str:
        from tokenmizer.compression.engine import CompressionPipeline
        return CompressionPipeline().compress_text(text, min_tokens=10).compressed_text

    def test_python_indentation_is_preserved_in_a_fenced_block(self):
        code = (
            "Here is the function:\n\n```python\n"
            "def process(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        if item.valid:\n"
            "            total += item.value\n"
            "        else:\n"
            "            log.warning('skipped')\n"
            "    return total\n"
            "```\n" + self.PADDING
        )
        out = self._compress(code)
        assert "        if item.valid:" in out
        assert "            total += item.value" in out
        assert "        else:" in out

    def test_tabs_are_preserved_in_a_fenced_block(self):
        """Makefiles require tabs; Go is tab-indented by convention."""
        text = "```makefile\nbuild:\n\tgo build ./...\n\ttest -f bin/app\n```\n" + self.PADDING
        assert "\tgo build" in self._compress(text)

    def test_repeated_log_lines_are_kept_in_a_fenced_block(self):
        line = "2026-09-11 10:00:01 WARNING skipping 42 because it failed validation"
        text = f"Log output:\n\n```\n{line}\n{line}\n{line}\n```\n" + self.PADDING
        assert self._compress(text).count(line) == 3

    def test_inline_code_spans_are_preserved(self):
        text = ("Run `pip install   tokenmizer[cache]` and set `x    =   1`.\n"
                + self.PADDING)
        out = self._compress(text)
        assert "`pip install   tokenmizer[cache]`" in out
        assert "`x    =   1`" in out

    def test_comment_stripping_does_not_reach_a_fenced_block(self):
        from tokenmizer.compression.engine import CompressionPipeline
        text = ("```python\n"
                "x = 1  # this comment is the actual question\n"
                "url = 'https://example.com/a#b'\n"
                "```\n" + self.PADDING)
        out = CompressionPipeline(strip_comments=True).compress_text(
            text, min_tokens=10).compressed_text
        assert "# this comment is the actual question" in out
        assert "https://example.com/a#b" in out

    def test_prose_is_still_compressed(self):
        """The guard must not switch the heuristics off — filler removal on
        prose is the whole point of the layer."""
        text = ("Certainly! I'd be happy to help. Great question! "
                "The answer is forty-two.\n" + self.PADDING)
        out = self._compress(text)
        assert "Certainly" not in out
        assert "forty-two" in out

    def test_fenced_and_prose_round_trip_is_exact_for_code(self):
        """Whatever happens to prose, every code segment must come back
        byte-identical."""
        from tokenmizer.compression.engine import CodeBlockGuard
        text = ("Intro   with    spaces.\n```js\nconst a = 1;   // keep\n"
                "const b = 'http://x/y';\n```\nOutro.\n" + self.PADDING)
        out = self._compress(text)
        original_code = [s for is_code, s in CodeBlockGuard.segment(text) if is_code]
        for segment in original_code:
            assert segment in out, f"code segment altered: {segment!r}"


class TestFillerRemovalDoesNotCorruptWords:
    """The interjection pattern had no word boundary, so it matched inside
    words: "pressure" became "pres", "measure" became "mea", "ensure" became
    "en", on every older prose message the model was shown."""

    @pytest.mark.parametrize("sentence", [
        "Make sure the pressure gauge reads zero before you measure.",
        "We need to ensure the index exists, indeed it is required.",
        "The insurer absolutely refused; the closure was certain.",
    ])
    def test_words_containing_an_interjection_survive(self, sentence):
        from tokenmizer.compression.engine import FillerRemover
        out, _ = FillerRemover().apply(sentence)
        for word in ("pressure", "measure", "ensure", "insurer", "closure", "certain"):
            if word in sentence:
                assert word in out, f"{word!r} corrupted: {out!r}"

    def test_a_leading_interjection_is_still_removed(self):
        from tokenmizer.compression.engine import FillerRemover
        out, _ = FillerRemover().apply("Sure! The answer is forty-two.")
        assert out == "The answer is forty-two."

    def test_an_interjection_after_a_sentence_break_is_removed(self):
        from tokenmizer.compression.engine import FillerRemover
        out, _ = FillerRemover().apply("First point. Certainly, the second point.")
        assert "Certainly" not in out
        assert "second point" in out


class TestHistoryPrunerKeepsDistinctReplies:
    """Three assistant replies opening with the same 60 characters were
    treated as one repeated message, and every one after the second was cut
    to its first and last 100 characters. In a coding session that opening
    is "Here's the updated auth.py:" and the middle is the code. The pruner
    also ran before the protect_recent window, so the newest reply was cut
    too."""

    def _session(self, bodies):
        msgs = []
        for i, body in enumerate(bodies):
            msgs.append({"role": "user", "content": f"update auth.py for case {i}"})
            msgs.append({"role": "assistant",
                         "content": f"Here's the updated auth.py:\n\n```python\n{body}```"})
        return msgs

    def test_same_opening_different_code_is_kept_whole(self):
        from tokenmizer.compression.engine import RepetitiveHistoryPruner
        # Same opening line in every reply — what a real session looks like —
        # with the differing code further down, past the 60-char prefix.
        bodies = ["def login(user):\n    token = issue(user)\n"
                  + f"    step_{i} = {i}\n" * 15 for i in range(3)]
        out, _ = RepetitiveHistoryPruner().apply(self._session(bodies))
        for i, msg in enumerate(m for m in out if m["role"] == "assistant"):
            assert "...[compressed]..." not in msg["content"]
            assert msg["content"].count(f"step_{i} = {i}") == 15

    def test_a_verbatim_repeat_is_still_collapsed(self):
        """What the pruner is for: the same reply sent again."""
        from tokenmizer.compression.engine import RepetitiveHistoryPruner
        same = "def v():\n" + "    x = 1\n" * 30
        out, _ = RepetitiveHistoryPruner().apply(self._session([same, same, same]))
        replies = [m["content"] for m in out if m["role"] == "assistant"]
        assert "...[compressed]..." not in replies[0]
        assert any("repeat of" in r or "...[compressed]..." in r for r in replies[1:])

    def test_pruning_never_touches_the_last_reply(self):
        from tokenmizer.compression.engine import CompressionPipeline
        same = "def v():\n" + "    x = 1\n" * 30
        out, _ = CompressionPipeline().compress_messages(
            self._session([same, same, same, same]), protect_recent=3)
        assert out[-1]["content"].count("x = 1") == 30
