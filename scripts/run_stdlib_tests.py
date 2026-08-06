#!/usr/bin/env python3
"""
Stdlib-only (unittest, no pytest) runner for the subset of tests that
have zero external dependencies (no fastapi/pydantic/tiktoken/llmlingua
required). This environment has no network access, so those packages
cannot be installed — this script proves the fixes that CAN be verified
here actually work, rather than just asserting they do.

Tests requiring fastapi/pydantic/tiktoken (auth.py's FastAPI HTTPException
path, the full API app, tokenizer-dependent compression ratios) are NOT
run here — see TESTING.md for what to run on a machine with
`pip install -e .[dev]`.
"""
import sys
import traceback

sys.path.insert(0, ".")

from tokenmizer.compression.engine import CodeBlockGuard, CommentStripper
from tokenmizer.security.redaction import redact, redact_messages

passed = 0
failed = 0
errors = []


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        errors.append(name)
        print(f"  FAIL  {name}")


def run(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        failed += 1
        errors.append(f"{name}: {e}")
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


print("=" * 70)
print("CodeBlockGuard")
print("=" * 70)


def t1():
    sample = (
        "Some prose here.\n\n"
        "```python\ndef foo(x):\n    return x + 1\n```\n\n"
        "More prose with `inline_code` in it.\n\n"
        "Final paragraph."
    )
    segments = CodeBlockGuard.segment(sample)
    assert CodeBlockGuard.reassemble(segments) == sample, "round-trip not lossless"


run("round_trip_is_lossless", t1)


def t2():
    text = "Explanation.\n```python\nx = 1\n```\nMore text."
    segments = CodeBlockGuard.segment(text)
    code_segments = [s for is_code, s in segments if is_code]
    assert any("x = 1" in s for s in code_segments)


run("fenced_code_block_detected", t2)


def t3():
    text = "Use the `requests` library for this."
    segments = CodeBlockGuard.segment(text)
    code_segments = [s for is_code, s in segments if is_code]
    assert any("requests" in s for s in code_segments)


run("inline_code_detected", t3)


def t4():
    text = '```js\nconst url = "https://api.example.com/v1/users";\n```'
    segments = CodeBlockGuard.segment(text)
    code_segments = [s for is_code, s in segments if is_code]
    assert len(code_segments) == 1
    assert "https://api.example.com/v1/users" in code_segments[0]


run("url_inside_fenced_code_survives_segmentation", t4)

print()
print("=" * 70)
print("CommentStripper (URL-in-string bug fix)")
print("=" * 70)

stripper = CommentStripper()


def t5():
    code = 'const url = "https://example.com/api"; // fetch data'
    result, _ = stripper.apply(code)
    assert "https://example.com/api" in result, f"URL corrupted: {result!r}"
    assert "fetch data" not in result


run("url_in_double_quoted_string_survives", t5)


def t6():
    code = "const url = 'https://test.com/v2'; // comment"
    result, _ = stripper.apply(code)
    assert "https://test.com/v2" in result


run("url_in_single_quoted_string_survives", t6)


def t7():
    code = "const x = 5; // this is a real comment"
    result, _ = stripper.apply(code)
    assert result == "const x = 5;", f"got {result!r}"


run("real_comment_still_stripped", t7)


def t8():
    code = (
        'const a = "https://one.com"; // comment one\n'
        'const b = "https://two.com"; // comment two'
    )
    result, _ = stripper.apply(code)
    assert "https://one.com" in result
    assert "https://two.com" in result
    assert "comment one" not in result
    assert "comment two" not in result


run("multiple_urls_and_comments_different_lines", t8)


def t9():
    code = "x = 1  # this should still be removed\ny = 2"
    result, _ = stripper.apply(code)
    assert "should still be removed" not in result
    assert "y = 2" in result


run("python_comments_unaffected", t9)


def t10():
    code = "x = 1; /* block comment */ y = 2;"
    result, _ = stripper.apply(code)
    assert "block comment" not in result


run("block_comments_unaffected", t10)


def t10b():
    code = "x = 1  # this should be removed\ny = 2"
    result, _ = stripper.apply(code)
    assert "this should be removed" not in result
    assert "y = 2" in result


run("trailing_python_comment_now_stripped", t10b)


def t10c():
    code = 'url = f"https://x.com/{id}"  # fetch user'
    result, _ = stripper.apply(code)
    assert "https://x.com/{id}" in result
    assert "fetch user" not in result


run("fstring_url_with_trailing_comment", t10c)


def t10d():
    code = 'const color = "#FF0000"; // red color'
    result, _ = stripper.apply(code)
    assert "#FF0000" in result
    assert "red color" not in result


run("hex_color_with_hash_not_treated_as_comment", t10d)

print()
print("=" * 70)
print("Redaction (multimodal content handling)")
print("=" * 70)


def t11():
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]
    cleaned = redact_messages(messages)
    assert cleaned[0]["content"] is None


run("none_content_does_not_crash", t11)


def t12():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Here's my key: sk-ant-api03-SECRETVALUE123"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KG=="}},
        ],
    }]
    cleaned = redact_messages(messages)
    text_block = next(b for b in cleaned[0]["content"] if b.get("type") == "text")
    assert "sk-ant" not in text_block["text"], f"key leaked: {text_block['text']!r}"
    assert "[REDACTED]" in text_block["text"]
    image_block = next(b for b in cleaned[0]["content"] if b.get("type") == "image")
    assert image_block["source"]["data"] == "iVBORw0KG==", "image data was corrupted!"


run("multimodal_text_block_is_redacted_image_untouched", t12)


def t13():
    messages = [{
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "1", "name": "search", "input": {}},
            "plain string block with sk-proj-AAAAAAAAAAAAAAAAAAAA1234",
        ],
    }]
    cleaned = redact_messages(messages)
    assert "sk-proj" not in cleaned[0]["content"][1], f"leaked: {cleaned[0]['content'][1]!r}"


run("multimodal_mixed_blocks_does_not_crash", t13)


def t14():
    text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    result = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result


run("aws_access_key_redacted", t14)


def t15():
    slack_token = "xox" + "b-" + "FAKE" + "SLACK" + "TOKEN" + "FOR" + "TESTS" + "ONLY"
    text = f"token: {slack_token}"
    result = redact(text)
    assert slack_token not in result


run("slack_token_redacted", t15)


def t16():
    stripe_key = "sk_" + "live_" + "NOTAREALSTRIPEKEY0001"
    text = f"Stripe secret: {stripe_key}"
    result = redact(text)
    assert stripe_key not in result


run("stripe_key_redacted", t16)


def t17():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    result = redact(f"Authorization token: {jwt}")
    assert jwt not in result


run("jwt_redacted", t17)

print()
print("=" * 70)
print("HybridExtractor.merge() case-preservation fix")
print("=" * 70)

from tokenmizer.graph_memory.hybrid_extractor import ExtractedData, HybridExtractor

_extractor = HybridExtractor(min_confidence=0.50)


def t18():
    llm_data = ExtractedData(files=["src/App.tsx", "src/Utils.ts"])
    heu_data = ExtractedData(files=["src/app.tsx", "src/Other.ts"])
    merged = _extractor.merge(llm_data, heu_data)
    assert any(f in ("src/App.tsx", "src/app.tsx") for f in merged.files), merged.files
    assert "src/Utils.ts" in merged.files
    assert "src/Other.ts" in merged.files
    assert merged.confidence["files"] == 0.95


run("merge_preserves_original_case_on_corroboration", t18)


def t19():
    llm_data = ExtractedData(files=["src/Solo.ts"])
    heu_data = ExtractedData(files=[])
    merged = _extractor.merge(llm_data, heu_data)
    assert merged.files == ["src/Solo.ts"]
    assert merged.confidence["files"] == 0.80


run("merge_llm_only_preserves_case_and_confidence", t19)


def t20():
    """Regression: pre-existing corroboration test must still pass after the fix."""
    llm_data = ExtractedData(
        files=["api/auth.py"],
        decisions=[{"label": "bcrypt for hashing", "reason": ""}],
    )
    heu_data = ExtractedData(
        files=["api/auth.py", "api/models.py"],
        decisions=[{"label": "using bcrypt", "reason": ""}],
    )
    merged = _extractor.merge(llm_data, heu_data)
    assert any("auth" in f for f in merged.files)


run("merge_boosts_corroborated_regression_check", t20)

print()
print("=" * 70)
print("invalidate_decision force=True regression (direct-mutation persist bug)")
print("=" * 70)

import tempfile as _tempfile

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import NodeStatus as _NodeStatusMod
from tokenmizer.graph_memory.types import NodeType as _NodeTypeMod

_NodeType_DECISION = _NodeTypeMod.DECISION
_NodeStatus_INVALIDATED = _NodeStatusMod.INVALIDATED


def _make_graph():
    d = _tempfile.mkdtemp()
    return GraphMemory("dirty-flag-test-session", storage_dir=d), d


def t21():
    g, tmpdir = _make_graph()
    nid = g.add_node(_NodeType_DECISION, "Use PostgreSQL for the database")
    g._persist()
    assert g._dirty is False
    g._nodes[nid].status = _NodeStatus_INVALIDATED
    g._persist()  # BUG PATTERN: no force=True
    reloaded = GraphMemory("dirty-flag-test-session", storage_dir=tmpdir)
    assert reloaded._nodes[nid].status != _NodeStatus_INVALIDATED, (
        "expected the no-force write to be skipped (bug pattern), but it persisted — "
        "_persist()'s dirty-flag behavior may have changed"
    )


def t22():
    g, tmpdir = _make_graph()
    nid = g.add_node(_NodeType_DECISION, "Use PostgreSQL for the database")
    g._persist()
    assert g._dirty is False
    g._nodes[nid].status = _NodeStatus_INVALIDATED
    g._persist(force=True)  # THE FIX
    reloaded = GraphMemory("dirty-flag-test-session", storage_dir=tmpdir)
    assert reloaded._nodes[nid].status == _NodeStatus_INVALIDATED, (
        "invalidate_decision's status change did not survive a reload with force=True"
    )


run("persist_without_force_after_direct_mutation_is_lost", t21)
run("invalidate_decision_pattern_persists_correctly_with_force", t22)

print()
print("=" * 70)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 70)
if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  - {e}")
sys.exit(1 if failed else 0)
