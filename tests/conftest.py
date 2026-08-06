import os
from unittest.mock import patch

import pytest

from tokenmizer.semantic_cache.cache import EmbeddingEngine

# The real `_load` downloads a sentence-transformers model from
# huggingface.co on first use. Letting that happen would make the suite
# slow, flaky and dependent on a third party, so it is stubbed out for
# every test by default.
#
# The stub is kept reachable rather than applied blindly with
# `patch(...).start()`, because a blanket patch hid the one code path it
# covered from the entire suite — and that is where an unhandled OSError
# lived: `_load` caught only ImportError, so a Hub outage or an
# air-gapped host raised out of `embed()` on the request path. No test
# could have caught it while `_load` was replaced everywhere.
#
# Ask for `real_embedding_load` in a test to get the genuine
# implementation back for the duration of that test.
_REAL_LOAD = EmbeddingEngine._load
_load_patch = patch.object(EmbeddingEngine, "_load", lambda self: None)
_load_patch.start()


@pytest.fixture
def real_embedding_load(monkeypatch):
    """Restore the real `EmbeddingEngine._load` for one test.

    For tests that assert how model-loading FAILS. Stub the network at
    the `sentence_transformers` import instead — see
    tests/unit/test_cache.py::TestEmbeddingModelIsOptional.
    """
    monkeypatch.setattr(EmbeddingEngine, "_load", _REAL_LOAD)
    return _REAL_LOAD


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set environment variables for tests so no real API calls are made."""
    os.environ.setdefault("TOKENMIZER_ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("TOKENMIZER_STATE_BACKEND", "memory")
