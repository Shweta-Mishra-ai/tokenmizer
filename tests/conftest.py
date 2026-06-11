import os
from unittest.mock import patch

import pytest

# Disable Hugging Face downloads and sentence-transformers model load in tests
patch("tokenmizer.semantic_cache.cache.EmbeddingEngine._load", lambda self: None).start()


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set environment variables for tests so no real API calls are made."""
    os.environ.setdefault("TOKENMIZER_ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("TOKENMIZER_STATE_BACKEND", "memory")
