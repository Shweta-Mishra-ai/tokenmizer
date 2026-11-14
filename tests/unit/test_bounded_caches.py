"""Process-wide caches are bounded by what they hold, not only by entries.

A long-running proxy sees every message of every session. A memo capped by
entry count alone grows with the size of what clients send: measured over
600 turns of one agent session, the message-word memo in api/injection.py
(an lru_cache keyed on the whole text) grew 38 MB.
"""
from __future__ import annotations

from tokenmizer.api import injection
from tokenmizer.compression import window as W
from tokenmizer.core.bounded_cache import MISSING, BoundedCache
from tokenmizer.security import redaction as R


def test_bounded_by_weight_and_by_entries():
    c: BoundedCache[str] = BoundedCache(max_entries=3, max_weight=10, weight=len)
    for i, v in enumerate(["aaaa", "bbbb", "cccc"]):
        c.put(BoundedCache.key(str(i)), v)
    assert c.total_weight <= 10 and len(c) == 2          # weight evicted the oldest
    assert c.get(BoundedCache.key("0")) is MISSING
    for i in range(3, 10):
        c.put(BoundedCache.key(str(i)), "x")
    assert len(c) == 3 and c.total_weight == 3          # entries evicted


def test_a_value_larger_than_the_budget_is_not_held():
    c: BoundedCache[str] = BoundedCache(10, 5, len)
    c.put(BoundedCache.key("big"), "x" * 6)
    assert len(c) == 0 and c.total_weight == 0


def test_none_is_a_value_not_a_miss():
    c: BoundedCache[str | None] = BoundedCache(10, 5, lambda v: len(v or ""))
    c.put(BoundedCache.key("k"), None)
    assert c.get(BoundedCache.key("k")) is None


def test_the_redaction_memo_is_bounded_by_characters():
    R._memo_clear()
    for i in range(400):
        R.redact(f"password={i:06d}secretvalue " + "x" * 60_000)
    assert R._memo.total_weight <= R._MEMO_MAX_TOTAL_CHARS
    R._memo_clear()


def test_a_memoised_redaction_is_still_correct():
    R._memo_clear()
    text = "token=abcdef1234567890 and more"
    assert R.redact(text) == R.redact(text)
    assert "abcdef1234567890" not in R.redact(text)
    R._memo_clear()


def test_the_message_word_memo_holds_no_text_and_stays_bounded():
    injection._WORDS_CACHE.clear()
    for i in range(300):
        injection._content_words(" ".join(f"word{i}_{j}" for j in range(20_000)))
    assert injection._WORDS_CACHE.total_weight <= injection._WORDS_CACHE.max_weight
    assert injection._content_words("Use bcrypt for passwords in auth.py") >= {"bcrypt", "auth"}
    injection._WORDS_CACHE.clear()


def test_the_path_memo_is_bounded_by_paths():
    W._PATHS_CACHE.clear()
    for i in range(400):
        W._paths_in("\n".join(f"src/p{i}/m{j}.py" for j in range(3000)))
    assert W._PATHS_CACHE.total_weight <= W._PATHS_CACHE.max_weight
    W._PATHS_CACHE.clear()
