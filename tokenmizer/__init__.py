"""TokenMizer — Never lose your AI context again."""

__version__ = "0.5.3"
__all__ = ["GraphMemory", "CheckpointManager", "get_settings"]


def __getattr__(name):
    """Lazy imports — avoids pydantic being required at import time for tests."""
    if name == "get_settings":
        from tokenmizer.config.settings import get_settings
        return get_settings
    if name == "Settings":
        from tokenmizer.config.settings import Settings
        return Settings
    if name == "GraphMemory":
        from tokenmizer.graph_memory.graph import GraphMemory
        return GraphMemory
    if name == "CheckpointManager":
        from tokenmizer.checkpoints.manager import CheckpointManager
        return CheckpointManager
    raise AttributeError(f"module 'tokenmizer' has no attribute {name!r}")
