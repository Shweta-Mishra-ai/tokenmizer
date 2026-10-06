"""Gemini-shaped message parts must pass through redaction.

#86 and #88 redacted tool-call *arguments* for OpenAI, Anthropic and Gemini.
These parts still carried a secret straight through to the provider, the graph
extractor and the checkpoints:

* `functionResponse`      a tool's output (the Anthropic `tool_result` twin)
* `executableCode`        code the model wrote
* `codeExecutionResult`   what running it printed
* `{"text": ...}`         a plain Gemini text part, which has no "type" key and
                          which the extractor reads as text, so a secret in
                          one reached extraction as well

Only the field that holds the content is redacted; names, ids, outcomes and
signatures are metadata and stay as sent.
"""
from __future__ import annotations

import copy
import json

import pytest

from tokenmizer.security.redaction import redact_messages

SECRET = "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890"


def _redact_part(part):
    return redact_messages([{"role": "user", "content": [part]}])[0]["content"][0]


class TestEachPartIsRedacted:

    def test_function_response_output(self):
        part = {"functionResponse": {
            "name": "read_file", "id": "call-1",
            "response": {"output": "KEY=" + SECRET, "lines": [SECRET, 3], "ok": True},
        }}
        cleaned = _redact_part(part)
        assert cleaned["functionResponse"] == {
            "name": "read_file", "id": "call-1",
            "response": {"output": "KEY=[REDACTED]", "lines": ["[REDACTED]", 3], "ok": True},
        }

    def test_executable_code(self):
        part = {"executableCode": {"language": "PYTHON", "code": f"key = '{SECRET}'"}}
        cleaned = _redact_part(part)
        assert SECRET not in json.dumps(cleaned)
        assert cleaned["executableCode"]["language"] == "PYTHON"

    def test_code_execution_result(self):
        part = {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": SECRET}}
        cleaned = _redact_part(part)
        assert cleaned["codeExecutionResult"] == {
            "outcome": "OUTCOME_OK", "output": "[REDACTED]"}

    def test_untyped_text_part(self):
        cleaned = _redact_part({"text": f"my key is {SECRET}"})
        assert SECRET not in cleaned["text"]
        assert "[REDACTED]" in cleaned["text"]

    def test_function_call_still_redacted_as_before(self):
        cleaned = _redact_part({"functionCall": {"name": "f", "args": {"k": SECRET}}})
        assert cleaned["functionCall"] == {"name": "f", "args": {"k": "[REDACTED]"}}


class TestNothingElseChanges:

    @pytest.mark.parametrize("part", [
        {"functionResponse": {"name": "read_file", "response": {"output": "plain text", "n": 2}}},
        {"executableCode": {"language": "PYTHON", "code": "print(1)"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}},
        {"text": "nothing secret here"},
        {"functionCall": {"name": "f", "args": {"path": "src/app.py"}}},
    ])
    def test_clean_parts_are_unchanged(self, part):
        assert _redact_part(copy.deepcopy(part)) == part

    def test_metadata_beside_the_payload_is_kept(self):
        part = {"functionResponse": {"name": "n", "response": {"o": SECRET}},
                "thoughtSignature": "keep-this"}
        assert _redact_part(part)["thoughtSignature"] == "keep-this"

    def test_image_data_is_never_touched(self):
        data = "sk-" + "B" * 40
        part = {"type": "image", "source": {"data": data}}
        assert _redact_part(part) == part

    def test_a_non_string_text_value_keeps_its_type(self):
        assert _redact_part({"text": 123}) == {"text": 123}

    def test_a_typed_text_block_behaves_exactly_as_before(self):
        cleaned = _redact_part({"type": "text", "text": f"k={SECRET}"})
        assert cleaned == {"type": "text", "text": "k=[REDACTED]"}


class TestMalformedAndImmutable:

    @pytest.mark.parametrize("part", [
        {"functionResponse": None},
        {"functionResponse": "invalid"},
        {"functionResponse": []},
        {"functionResponse": {"name": "no_response_field"}},
        {"executableCode": {"language": "PYTHON"}},
        {"codeExecutionResult": 5},
    ])
    def test_malformed_parts_pass_through_without_raising(self, part):
        assert _redact_part(copy.deepcopy(part)) == part

    def test_the_callers_messages_are_not_mutated(self):
        messages = [{"role": "user", "content": [
            {"functionResponse": {"name": "n", "response": {"o": SECRET}}},
            {"executableCode": {"language": "PYTHON", "code": SECRET}},
            {"codeExecutionResult": {"outcome": "OK", "output": SECRET}},
            {"text": SECRET},
        ]}]
        original = copy.deepcopy(messages)
        cleaned = redact_messages(messages)
        assert messages == original
        assert SECRET not in json.dumps(cleaned)
