"""
A malformed `messages` list must fail where it was passed, not three
frames later.

Found by auditing the public surface for caller-contract violations that
produce a deferred failure. `GraphMemory.extract_from_messages(["hi"])`
and `Memory.add("hi")` both died inside `_msg_hash` as

    AttributeError: 'str' object has no attribute 'get'

which names neither the offending element nor the shape expected, and
which — on the proxy's background extraction path, where extraction runs
inside a broad `except` — was counted as a silent failure and never
surfaced at all. The list is the caller's; the error should say so.
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.graph import GraphMemory

_GOOD = {"role": "user", "content": "Building an auth service"}


def _graph(tmp_path) -> GraphMemory:
    return GraphMemory("shape", storage_dir=str(tmp_path))


class TestANonDictMessageIsRejectedAtTheCall:
    def test_a_bare_string_names_the_index_and_the_expected_shape(self, tmp_path):
        with pytest.raises(TypeError) as excinfo:
            _graph(tmp_path).extract_from_messages(["hi"])

        message = str(excinfo.value)
        assert "messages[0]" in message
        assert "str" in message
        assert "content" in message

    def test_the_index_points_at_the_offending_element(self, tmp_path):
        with pytest.raises(TypeError, match=r"messages\[2\]"):
            _graph(tmp_path).extract_from_messages([_GOOD, _GOOD, None])

    def test_it_is_a_type_error_not_an_attribute_error(self, tmp_path):
        """AttributeError reads as a bug in the library; TypeError reads
        as what it is — the caller passed the wrong thing."""
        with pytest.raises(TypeError):
            _graph(tmp_path).extract_from_messages([12])

    def test_nothing_is_written_before_the_rejection(self, tmp_path):
        graph = _graph(tmp_path)
        with pytest.raises(TypeError):
            graph.extract_from_messages([_GOOD, "oops"])

        assert graph._nodes == {}, (
            "the valid prefix must not be half-applied — the call either "
            "extracts the list or rejects it"
        )

    def test_memory_add_gets_the_same_error(self, tmp_path):
        from tokenmizer.agents import Memory

        memory = Memory("shape", storage_dir=str(tmp_path))
        with pytest.raises(TypeError, match=r"messages\[0\]"):
            memory.add(["hi"])


class TestValidInputIsUnaffected:
    def test_an_ordinary_list_still_extracts(self, tmp_path):
        graph = _graph(tmp_path)
        graph.extract_from_messages([
            _GOOD,
            {"role": "assistant",
             "content": "Completed: rate limiting with slowapi on every route"},
        ])
        assert graph._nodes, "the guard must not reject well-formed messages"

    def test_an_empty_list_is_not_an_error(self, tmp_path):
        _graph(tmp_path).extract_from_messages([])

    def test_multimodal_content_parts_still_pass(self, tmp_path):
        """`content` as a list of parts is valid OpenAI/Anthropic shape;
        only the message itself has to be a mapping."""
        graph = _graph(tmp_path)
        graph.extract_from_messages([{
            "role": "user",
            "content": [{"type": "text",
                         "text": "Decided: use Postgres for order storage"}],
        }])
        assert graph._nodes
