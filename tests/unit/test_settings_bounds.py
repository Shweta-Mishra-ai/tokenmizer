"""
Settings that accepted any number. Each value below was taken at face value
and broke the proxy at runtime instead of at startup.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tokenmizer.config import settings as S


@pytest.mark.parametrize("model, field, bad", [
    (S.CacheSettings, "similarity_threshold", -1.0),   # matched every prompt
    (S.CacheSettings, "similarity_threshold", 1.5),    # silently disabled the layer
    (S.CacheSettings, "max_size", 0),                  # first write spun forever
    (S.CacheSettings, "ttl_seconds", 0),
    (S.CacheSettings, "max_bytes", 0),
    (S.CacheSettings, "max_semantic_scan", -1),
    (S.GraphCheckpointSettings, "min_confidence", 5.0),
    (S.GraphCheckpointSettings, "max_resume_tokens", 0),
    (S.MemorySettings, "recent_turns_verbatim", -1),
    (S.PreferenceSettings, "max_items", -1),
    (S.CompressionSettings, "min_tokens_to_compress", -1),
    (S.RoutingSettings, "complexity_threshold", 7.0),
])
def test_out_of_range_values_fail_at_startup(model, field, bad):
    with pytest.raises(ValidationError):
        model(**{field: bad})


@pytest.mark.parametrize("field, bad", [
    ("request_timeout", 0), ("proxy_port", 99999), ("trusted_proxy_hops", -1),
])
def test_top_level_out_of_range_values_fail(monkeypatch, field, bad):
    monkeypatch.setenv(f"TOKENMIZER_{field.upper()}", str(bad))
    with pytest.raises(ValidationError):
        S.Settings()


def test_defaults_and_edge_values_still_load():
    S.CacheSettings(similarity_threshold=1.0, max_size=1, ttl_seconds=1)
    S.GraphCheckpointSettings(min_confidence=0.0)
    S.CompressionSettings(min_tokens_to_compress=0)


def test_a_cache_with_no_capacity_stores_nothing_and_returns():
    """Used as a library, the cache can still be built with max_size=0:
    set() must return rather than evict from an empty cache forever."""
    from tokenmizer.semantic_cache.cache import SemanticCache
    c = SemanticCache(max_size=0)
    c.set("hello there", "answer", input_tokens=1, output_tokens=1, session_id="s")
    assert c.stats()["entries"] == 0
