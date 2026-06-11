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
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

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

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._store[key] = value  # TTL not enforced in-memory

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

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

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                self._r.setex(key, ttl, serialized)
            else:
                self._r.set(key, serialized)
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")

    def delete(self, key: str) -> None:
        try:
            self._r.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE failed for {key}: {e}")

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
