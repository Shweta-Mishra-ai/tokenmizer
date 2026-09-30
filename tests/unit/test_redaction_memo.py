"""
Redaction is memoised on the text's digest.

A client resends the whole conversation on every request, and redaction
ran 22 regex passes over all of it each time. On a real 51-request agent
session that was about 210 ms per request, on the event loop, so every
other request in flight waited for it. Redaction is a pure function of the
text, so a text seen before is not scanned again.

What must not change: the output, for every input, first time or repeated.
"""
from __future__ import annotations

import pytest

from tokenmizer.security import redaction as R

SECRET = "sk-ant-api03-" + "A" * 40


@pytest.fixture(autouse=True)
def _fresh_memo():
    R._memo_clear()
    yield
    R._memo_clear()


@pytest.fixture
def scans(monkeypatch):
    calls = []
    real = R._scrub

    def counting(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(R, "_scrub", counting)
    return calls


def test_a_text_seen_before_is_not_scanned_again(scans):
    text = "a long tool result " * 500
    assert R.redact(text) == text
    assert R.redact(text) == text
    assert len(scans) == 1


def test_a_repeated_secret_is_still_redacted(scans):
    text = f"key is {SECRET} ok"
    first = R.redact(text)
    second = R.redact(text)
    assert SECRET not in first and first == second == R._scrub(text)
    assert len(scans) == 2          # two redact() calls scanned once, plus the check


def test_a_clean_text_does_not_shadow_a_different_one():
    clean = "nothing to see here"
    assert R.redact(clean) == clean
    dirty = clean + " " + SECRET
    assert SECRET not in R.redact(dirty)


def test_output_matches_the_unmemoised_scan_for_varied_inputs():
    samples = ["", "x", SECRET, f"Bearer {'a' * 30}", "mail me at a.b@example.com",
               "postgres://user:hunter22@db/prod", "password = s3cretvalue!", "ok " * 1000,
               "\ud800 lone surrogate " + SECRET]
    for s in samples:
        assert R.redact(s) == R._scrub(s)
        assert R.redact(s) == R._scrub(s)    # and from the memo


def test_the_memo_is_bounded(monkeypatch):
    monkeypatch.setattr(R, "_MEMO_MAX_ENTRIES", 50)
    for i in range(500):
        R.redact(f"message number {i}")
    assert len(R._memo) <= 50


def test_a_huge_redacted_text_is_not_retained(monkeypatch):
    """A clean text is remembered as a flag; a redacted one keeps its
    redacted copy, so a very large one is recomputed rather than held."""
    monkeypatch.setattr(R, "_MEMO_MAX_CHARS", 1000)
    big = SECRET + " " + "x" * 5000
    out = R.redact(big)
    assert SECRET not in out
    assert all(v is None or len(v) <= 1000 for v in R._memo.values())


def test_messages_are_redacted_the_same_on_every_resend():
    history = [{"role": "user", "content": f"use {SECRET}"},
               {"role": "tool", "content": [{"type": "text", "text": SECRET}]}]
    first = R.redact_messages(history)
    second = R.redact_messages(history)
    assert first == second
    assert SECRET not in str(first)
    assert history[0]["content"] == f"use {SECRET}"      # input not mutated


def test_concurrent_callers_never_get_an_unredacted_secret(monkeypatch):
    """The memo is shared by the event loop and worker threads. Many threads
    redacting a mix of clean and secret-bearing texts, under constant
    eviction, must each get exactly the unmemoised result."""
    import threading
    monkeypatch.setattr(R, "_MEMO_MAX_ENTRIES", 16)
    texts = [f"turn {i} " + (SECRET if i % 3 == 0 else "clean") for i in range(60)]
    expected = {t: R._scrub(t) for t in texts}
    wrong = []

    def worker(offset):
        for k in range(300):
            t = texts[(offset + k) % len(texts)]
            if R.redact(t) != expected[t]:
                wrong.append(t)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not wrong
    assert len(R._memo) <= 16
