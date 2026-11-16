"""A thread-safe LRU cache bounded by what it holds, not only by entries.

Several per-process memos sit on the request path: the paths in a tool
output, the content words of a message, a message's redacted copy. Each was
capped by entry count alone, and an entry's size is whatever a client sent —
a 4 MB tool output kept alive by a memo keyed on its own text. Measured over
600 turns of one agent session, the content-word memo alone grew 38 MB.

Keys here are digests of the text (`BoundedCache.key`), so no text is
retained by its key, and the cache also bounds the summed weight of its
values.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")

MISSING = object()


class BoundedCache(Generic[V]):
    def __init__(self, max_entries: int, max_weight: int, weight: Callable[[V], int]):
        self.max_entries = max_entries
        self.max_weight = max_weight
        self._weight = weight
        self._data: OrderedDict[bytes, V] = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        for part in parts:
            h.update(part.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        return h.digest()

    def get(self, key: bytes, default=MISSING):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return default

    def put(self, key: bytes, value: V) -> None:
        w = self._weight(value)
        if w > self.max_weight:
            return                      # larger than the whole budget
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return
            self._data[key] = value
            self._total += w
            while self._data and (len(self._data) > self.max_entries
                                  or self._total > self.max_weight):
                _, gone = self._data.popitem(last=False)
                self._total -= self._weight(gone)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._total = 0

    def values(self) -> list[V]:
        """A snapshot of the cached values, oldest first."""
        with self._lock:
            return list(self._data.values())

    @property
    def total_weight(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self._data)
