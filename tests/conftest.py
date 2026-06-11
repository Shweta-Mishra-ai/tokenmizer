import pytest
import tempfile
import os

@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set environment variables for tests so no real API calls are made."""
    os.environ.setdefault("TOKENMIZER_ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("TOKENMIZER_STATE_BACKEND", "memory")
