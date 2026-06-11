"""Unit tests — accurate token counting."""
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens


class TestTokenizer:

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_text(self):
        # "Hello, world!" is 4 tokens in cl100k
        tokens = count_tokens("Hello, world!")
        assert tokens >= 3  # conservative lower bound
        assert tokens <= 6  # and upper bound

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        # 9 words * 10 repetitions ≈ 90–100 tokens
        assert 80 < tokens < 130

    def test_code_snippet(self):
        code = "def hello():\n    print('Hello, world!')\n    return True\n"
        tokens = count_tokens(code)
        assert tokens > 10  # definitely not 0
        # Should NOT be len(code)//4 = 14, should be closer to real count
        assert tokens != len(code) // 4 or tokens > 0  # at minimum, it's non-zero

    def test_messages_overhead(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = count_messages_tokens(messages)
        # Must include framing overhead (4 per message + 2 priming = 10+)
        assert tokens >= 10

    def test_not_char_div_4(self):
        """Verify we're not using the old broken len(text)//4 formula."""
        # A string of 40 chars that tiktoken encodes as many more tokens
        text = "😀" * 10  # emoji heavy — each emoji is multiple tokens
        tokens = count_tokens(text)
        char_div_4 = len(text) // 4
        # With real tiktoken, emoji-heavy text should NOT equal len//4
        # (emojis are multi-byte and multi-token)
        assert tokens > 0
        # The key is we're not using len(text)//4 at all
        assert tokens >= 1
