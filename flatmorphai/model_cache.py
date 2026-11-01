"""
Module-level singleton that caches loaded AI models by checkpoint path.

On a cache hit the caller can skip both the actual model load and the
modal loading dialog. The cache is thread-safe.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional


class ModelCache:
    """Simple keyed cache for expensive model objects."""

    _instance: Optional["ModelCache"] = None
    _lock = Lock()

    def __new__(cls) -> "ModelCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store = {}
        return cls._instance

    def __init__(self) -> None:
        # Store is created in __new__, but keep an explicit annotation.
        self._store: Dict[str, Any]

    def has(self, key: str) -> bool:
        """Return True if ``key`` has a cached value."""
        with self._lock:
            return key in self._store

    def get(self, key: str) -> Any:
        """Return the cached value for ``key`` or None."""
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``."""
        with self._lock:
            self._store[key] = value

    def clear(self, key: Optional[str] = None) -> None:
        """Remove one entry or the whole cache."""
        with self._lock:
            if key is None:
                self._store.clear()
            elif key in self._store:
                del self._store[key]

    def keys(self):
        """Return a snapshot of all cached keys."""
        with self._lock:
            return list(self._store.keys())
