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
