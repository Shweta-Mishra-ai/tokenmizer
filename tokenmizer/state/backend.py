"""
State backend abstraction.

InMemoryBackend  — development only, dies on restart
RedisBackend     — production, survives restarts, works across workers

Set TOKENMIZER_STATE_BACKEND=redis and TOKENMIZER_REDIS_URL=redis://...
to enable Redis in production.

STATUS (as of the TM-04 audit fix): this module currently has no caller in
api/app.py. Its one previous caller — a `context_used` accumulator tracking
how much context a session had used across turns — was removed outright:
the accumulator design was wrong independent of persistence (it double-
counted every earlier turn's content each time, since each `messages` list
already contains the full running conversation, and it never reflected
windowing/compaction — see api/app.py::_update_graph). Deleting the wrong
counter was the fix; there is no replacement caller yet. This module is
kept because a Redis-backed shared store is the right building block for
future cross-worker coordination (e.g. a distributed session lock once
issue #27's per-row persistence migration lands), not because anything
here is broken. Do not treat its presence as evidence that cross-worker
state sharing is currently wired up anywhere — it isn't.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Returns True if the write succeeded, False otherwise.

        Kept explicit (not None) so a caller can distinguish "write
        happened" from "write silently dropped" — a dropped write in a
        shared cross-worker store should never look identical to success.
        (This backend's one former caller, a context-usage accumulator in
        api/app.py, was removed for unrelated reasons — see this module's
        STATUS note above — but the bool-return contract stands on its own
        merit for whatever the next caller turns out to be.)
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Returns True if the delete succeeded (or key didn't exist),
        False if the backend call itself failed."""
        ...

    @abstractmethod
    def keys(self, prefix: str) -> list[str]: ...


class InMemoryBackend(StateBackend):
    """
    Development backend. NOT suitable for production or multi-worker deployments.
    State is lost on process restart.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        logger.warning(
            "InMemoryBackend: session state will be lost on restart. "
            "Set TOKENMIZER_STATE_BACKEND=redis for production."
        )

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._store[key] = value  # TTL not enforced in-memory
        return True

    def delete(self, key: str) -> bool:
        self._store.pop(key, None)
        return True

    def keys(self, prefix: str) -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]

    def __len__(self) -> int:
        return len(self._store)


class RedisBackend(StateBackend):
    """
    Production backend. Survives restarts, works across uvicorn workers.
    Requires: pip install redis
    """

    def __init__(self, url: str = "redis://localhost:6379/0"):
        try:
            import redis
            self._r = redis.from_url(url, decode_responses=True, socket_connect_timeout=5)
            self._r.ping()
            logger.info(f"Redis backend connected: {url}")
        except ImportError:
            raise ImportError("pip install redis  — required for Redis state backend")
        except Exception as e:
            raise ConnectionError(f"Redis connection failed ({url}): {e}") from e

    def get(self, key: str) -> Optional[Any]:
        try:
            val = self._r.get(key)
            return json.loads(val) if val is not None else None
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                self._r.setex(key, ttl, serialized)
            else:
                self._r.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis SET failed for {key} — write was DROPPED: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            self._r.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for {key}: {e}")
            return False

    def keys(self, prefix: str) -> list[str]:
        try:
            return list(self._r.scan_iter(f"{prefix}*"))
        except Exception as e:
            logger.warning(f"Redis KEYS failed for prefix {prefix}: {e}")
            return []


def get_state_backend(backend_type: str = "memory", redis_url: str = "redis://localhost:6379/0") -> StateBackend:
    if backend_type == "redis":
        return RedisBackend(url=redis_url)
    return InMemoryBackend()
