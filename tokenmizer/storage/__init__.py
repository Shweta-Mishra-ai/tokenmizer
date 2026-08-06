"""
Unified storage interface.
tokenmizer/storage/__init__.py

STATUS: this defines a protocol, nothing imports it, and the classes
below do not declare conformance to it. It documents an intended shape
for the storage layer rather than enforcing one — do not read its
presence as evidence that the storage layer is unified.

Current implementations:
  GraphMemory       → SQLite  (tokenmizer/graph_memory/graph.py)
  CheckpointManager → SQLite  (tokenmizer/checkpoints/manager.py)
  StateBackend      → defined but UNUSED (tokenmizer/state/backend.py)

All three now implement the same conceptual interface:
  get(key) → value | None
  set(key, value, ttl?)
  delete(key)
  keys(prefix) → list[str]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class StorageBackend(ABC):
    """
    Minimal key-value storage interface.
    All TokenMizer persistence layers implement this protocol.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return value for key, or None if not found."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value at key. Optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove key. No-op if not found."""
        ...

    @abstractmethod
    def keys(self, prefix: str = "") -> list[str]:
        """Return all keys matching prefix."""
        ...

    def exists(self, key: str) -> bool:
        """Convenience: check if key exists."""
        return self.get(key) is not None
