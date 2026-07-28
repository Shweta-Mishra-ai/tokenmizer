"""
Regression tests for a gap the user found by inspection: ENDPOINT and
SCHEMA are full node types in the ontology (graph.py creates nodes for
them, to_context_block() has dedicated sections for them, the LLM
extraction prompt asks for them), but the HEURISTIC extractor —
heuristic_extract() / _extract_one_message() — had ZERO regex patterns
to ever populate ExtractedData.endpoints or .schemas from raw text.

Since use_llm_extraction defaults to False, this meant that in the
SHIPPED DEFAULT configuration, ENDPOINT and SCHEMA nodes could never be
created at all, no matter what the conversation said — the feature
existed structurally but nothing fed it. This is distinct from (and
upstream of) the previously-fixed TM-14 bug, where _deduplicate() DROPPED
these fields even when the LLM path DID populate them; this bug is that
the default, no-LLM path never populates them in the first place.

Fix: two new heuristic passes, following the same pattern (and negation-
awareness) as the existing decision-extraction passes:
  - Endpoints: "METHOD /path" mentions (POST /api/auth/login, etc.)
  - Schemas: "Schema: X" header lines, and inline "X table" mentions
"""
from __future__ import annotations

from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor

extractor = HybridExtractor(min_confidence=0.50)


class TestEndpointExtraction:

    def test_single_endpoint_extracted(self):
        messages = [{
            "role": "assistant",
            "content": "Created the login route: POST /api/auth/login",
        }]
        result = extractor.heuristic_extract(messages)
        assert any("POST" in e and "/api/auth/login" in e for e in result.endpoints), (
            f"expected an endpoint mentioning POST /api/auth/login, got: {result.endpoints}"
        )

    def test_multiple_endpoints_in_one_message(self):
        """Exact phrasing already used in this repo's own benchmark
        fixture (benchmarks/checkpoint_accuracy/runner.py)."""
        messages = [{
            "role": "assistant",
            "content": ("Implemented: POST /api/auth/register, POST /api/auth/login "
                       "(returns JWT), POST /api/auth/logout. Files updated: api/auth.py"),
        }]
        result = extractor.heuristic_extract(messages)
        joined = " ".join(result.endpoints)
        assert "/api/auth/register" in joined
        assert "/api/auth/login" in joined
        assert "/api/auth/logout" in joined

    def test_various_http_methods_recognized(self):
        messages = [{
            "role": "assistant",
            "content": "Added GET /api/users/:id, PUT /api/users/:id, and DELETE /api/users/:id",
        }]
        result = extractor.heuristic_extract(messages)
        joined = " ".join(result.endpoints)
        assert "GET" in joined and "PUT" in joined and "DELETE" in joined

    def test_no_endpoints_no_false_positives_on_plain_text(self):
        messages = [{"role": "user", "content": "Let's build an authentication service."}]
        result = extractor.heuristic_extract(messages)
        assert result.endpoints == []


class TestSchemaExtraction:

    def test_schema_header_format_extracted(self):
        """Exact phrasing already used in this repo's own benchmark fixture."""
        messages = [{
            "role": "assistant",
            "content": ("Schema: users table — id (UUID PK), email (unique), "
                       "hashed_password, created_at (timestamp), is_active (bool)."),
        }]
        result = extractor.heuristic_extract(messages)
        assert any("users table" in s.lower() for s in result.schemas), (
            f"expected a schema mentioning 'users table', got: {result.schemas}"
        )

    def test_inline_table_mention_extracted(self):
        messages = [{
            "role": "assistant",
            "content": "Added a new sessions table to track active logins.",
        }]
        result = extractor.heuristic_extract(messages)
        assert any("sessions table" in s.lower() for s in result.schemas), (
            f"expected 'sessions table' to be extracted, got: {result.schemas}"
        )

    def test_negated_table_mention_is_not_extracted(self):
        """The exact false-positive risk this repo's own benchmark
        fixture contains in the very same message as a real schema:
        'Schema: users table ... No refresh_tokens table needed since
        using Redis.' The negated 'no X table needed' must NOT be
        extracted as if a refresh_tokens table exists."""
        messages = [{
            "role": "assistant",
            "content": ("Schema: users table — id (UUID PK), email (unique), "
                       "hashed_password, created_at (timestamp), is_active (bool). "
                       "No refresh_tokens table needed since using Redis."),
        }]
        result = extractor.heuristic_extract(messages)
        assert not any("refresh_tokens" in s.lower() for s in result.schemas), (
            f"a NEGATED table mention ('No refresh_tokens table needed') was "
            f"extracted as if the table exists: {result.schemas}"
        )
        # The real schema in the same message must still come through.
        assert any("users table" in s.lower() for s in result.schemas)

    def test_generic_the_table_is_not_extracted_as_a_schema_name(self):
        """'the table below' or similar generic phrasing must not produce
        a schema node literally named 'the'."""
        messages = [{
            "role": "assistant",
            "content": "See the table below for a summary of results.",
        }]
        result = extractor.heuristic_extract(messages)
        assert not any(s.lower().startswith("the ") for s in result.schemas), (
            f"generic 'the table' phrasing produced a noise schema: {result.schemas}"
        )
