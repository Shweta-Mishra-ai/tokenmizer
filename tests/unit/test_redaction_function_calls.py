"""Legacy OpenAI and Gemini tool arguments must pass through redaction."""

import copy
import json
import unittest

from tokenmizer.security.redaction import redact_messages


SECRET = "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890"


class TestFunctionCallRedaction(unittest.TestCase):
    def test_legacy_arguments_are_redacted_and_remain_valid_json(self):
        arguments = {"nested": [{"value": SECRET}], "count": 3, "enabled": True}
        message = {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "write_file", "arguments": json.dumps(arguments)},
        }

        cleaned = redact_messages([message])[0]

        self.assertIsNone(cleaned["content"])
        self.assertEqual(cleaned["function_call"]["name"], "write_file")
        self.assertEqual(
            json.loads(cleaned["function_call"]["arguments"]),
            {"nested": [{"value": "[REDACTED]"}], "count": 3, "enabled": True},
        )

    def test_clean_legacy_arguments_keep_their_original_formatting(self):
        arguments = '{  "path": "src/app.py", "line": 7  }'
        message = {"function_call": {"name": "read_file", "arguments": arguments}}

        cleaned = redact_messages([message])[0]

        self.assertEqual(cleaned["function_call"]["arguments"], arguments)

    def test_malformed_legacy_json_is_still_scrubbed(self):
        arguments = '{"value":"' + SECRET + '"'
        message = {"function_call": {"name": "broken", "arguments": arguments}}

        cleaned = redact_messages([message])[0]["function_call"]["arguments"]

        self.assertNotIn(SECRET, cleaned)
        self.assertIn("[REDACTED]", cleaned)

    def test_structured_legacy_arguments_preserve_types(self):
        message = {"function_call": {"arguments": {"value": SECRET, "count": 2}}}

        cleaned = redact_messages([message])[0]

        self.assertEqual(
            cleaned["function_call"]["arguments"], {"value": "[REDACTED]", "count": 2}
        )

    def test_gemini_nested_arguments_are_redacted_without_changing_metadata(self):
        message = {
            "role": "model",
            "content": [{
                "functionCall": {
                    "name": "write_file",
                    "id": "call-1",
                    "args": {"nested": [SECRET, {"count": 3}], "enabled": False},
                },
                "thoughtSignature": "keep-this-metadata",
            }],
        }

        cleaned = redact_messages([message])[0]["content"][0]

        self.assertEqual(cleaned["thoughtSignature"], "keep-this-metadata")
        self.assertEqual(cleaned["functionCall"], {
            "name": "write_file",
            "id": "call-1",
            "args": {"nested": ["[REDACTED]", {"count": 3}], "enabled": False},
        })

    def test_clean_gemini_values_are_unchanged(self):
        call = {"name": "read_file", "args": {"path": "src/app.py", "line": 7}}

        cleaned = redact_messages([{"content": [{"functionCall": call}]}])[0]

        self.assertEqual(cleaned["content"][0]["functionCall"], call)

    def test_redaction_does_not_mutate_either_input_shape(self):
        messages = [
            {"function_call": {"arguments": json.dumps({"value": SECRET})}},
            {"content": [{"functionCall": {"args": {"nested": [SECRET]}}}]},
        ]
        original = copy.deepcopy(messages)

        cleaned = redact_messages(messages)

        self.assertEqual(messages, original)
        self.assertNotIn(SECRET, json.dumps(cleaned))

    def test_missing_or_malformed_function_calls_pass_through(self):
        for call in (None, "invalid", [], {"name": "no_arguments"}):
            with self.subTest(call=call):
                messages = [
                    {"function_call": call},
                    {"content": [{"functionCall": call}]},
                ]
                cleaned = redact_messages(messages)
                self.assertEqual(cleaned[0]["function_call"], call)
                self.assertEqual(cleaned[1]["content"][0]["functionCall"], call)

    def test_other_tool_shapes_and_image_data_still_work(self):
        image_data = "sk-" + "B" * 40
        messages = [{
            "tool_calls": [{"function": {"arguments": json.dumps({"value": SECRET})}}],
            "content": [
                {"type": "tool_use", "input": {"value": SECRET}},
                {"type": "image", "source": {"data": image_data}},
            ],
        }]

        cleaned = redact_messages(messages)[0]

        self.assertEqual(
            json.loads(cleaned["tool_calls"][0]["function"]["arguments"]),
            {"value": "[REDACTED]"},
        )
        self.assertEqual(cleaned["content"][0]["input"], {"value": "[REDACTED]"})
        self.assertEqual(cleaned["content"][1]["source"]["data"], image_data)
