"""
State backend abstraction.

InMemoryBackend  — development only, dies on restart
RedisBackend     — production, survives restarts, works across workers

Set TOKENMIZER_STATE_BACKEND=redis and TOKENMIZER_REDIS_URL=redis://...
to enable Redis in production.
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

        FIXED: previously returned None unconditionally (success or
        failure looked identical to every caller). The one real caller —
        `_set_context_used()` in api/app.py, which tracks how many tokens
        of context a session has used — had no way to know its write was
        dropped. A dropped write under-counts context usage, which means
        the auto-checkpoint trigger_at_percent threshold could be silently
        missed: the proxy would think a session is using less context than
        it actually is, and the safety-net checkpoint that's supposed to
        fire before the context window overflows simply... wouldn't.
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
