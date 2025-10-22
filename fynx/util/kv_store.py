"""
Key-Value Store Implementation
============================

High-performance key-value store with automatic deduplication and caching.
Uses cachetools for optimized LRU caching and smart deduplication strategies.
"""

import concurrent.futures
import hashlib
import json
import pickle
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from cachetools import LRUCache


@dataclass
class ValueWrapper:
    """
    Wrapper for values to make them weak-referenceable.
    This enables deduplication for built-in types like lists and dicts.

    NOTE: We store the actual value object here, so when multiple keys
    point to the same wrapper, they share the same value instance.
    """

    value: Any
    content_hash: int

    def __hash__(self):
        return self.content_hash

    def __eq__(self, other):
        if not isinstance(other, ValueWrapper):
            return False
        # Two wrappers are equal if they wrap the SAME object (by identity)
        return self.value is other.value


class KeyValueStore:
    """
    Key-value store with automatic deduplication and LRU caching.

    Features:
    - O(1) get, set, delete operations
    - Automatic value deduplication for all types (not just hashable)
    - Thread-safe operations
    - LRU caching for frequently accessed data
    - Memory-efficient with automatic cleanup via weak references

    Usage:
        store = KeyValueStore()
        store.set("key1", [1, 2, 3])
        store.set("key2", [1, 2, 3])  # Reuses same list instance
        value = store.get("key1")
        store.delete("key1")
    """

    # Sentinel object for "key not found"
    _MISSING = object()

    def __init__(
        self,
        enable_dedup: bool = True,
        cache_size: int = 10000,
        async_operations: bool = False,
        max_workers: int = 1,
    ):
        """
        Initialize the store.

        Args:
            enable_dedup: Whether to enable value deduplication
            cache_size: Size of the LRU cache for frequently accessed values
            async_operations: Whether to enable async operation support
            max_workers: Maximum number of worker threads for async operations (default: 1) (default: 1)
        """
        # Main key -> ValueWrapper mapping
        self._data: Dict[Any, ValueWrapper] = {}

        # Content hash -> WeakValueDictionary for deduplication
        # Maps hash -> weak ref to ValueWrapper
        self._dedup_index: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        # LRU cache for frequently accessed values (stores actual values, not wrappers)
        self._cache = LRUCache(maxsize=cache_size)

        # Thread safety
        self._lock = threading.RLock()

        # Configuration
        self._enable_dedup = enable_dedup
        self._async_operations = async_operations

        # Async operation support
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            if async_operations
            else None
        )

        # Statistics
        self._stats = {
            "gets": 0,
            "sets": 0,
            "deletes": 0,
            "cache_hits": 0,
            "dedup_saves": 0,
        }

    def _is_hashable(self, value: Any) -> bool:
        """Check if a value is hashable (and thus deduplicatable)."""
        try:
            hash(value)
            return True
        except TypeError:
            return False

    def _compute_hash(self, value: Any) -> int:
        """
        Compute a content-based hash for any value type.
        """
        try:
            # Try standard hash first for hashable types
            return hash(value)
        except TypeError:
            # For unhashable types, use content-based hashing
            try:
                # Try JSON serialization for common types
                json_str = json.dumps(value, sort_keys=True, default=str)
                return int(hashlib.md5(json_str.encode()).hexdigest()[:16], 16)
            except (TypeError, ValueError):
                # Fallback to pickle + hash for complex objects
                try:
                    pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    return int(hashlib.md5(pickled).hexdigest()[:16], 16)
                except (TypeError, ValueError, pickle.PicklingError):
                    # Last resort: use type and id (ensures no dedup for this value)
                    return hash((type(value).__name__, id(value)))

    def _deduplicate_value(self, value: Any, value_hash: int) -> ValueWrapper:
        """
        Find existing ValueWrapper or create new one.
        Uses WeakValueDictionary for automatic cleanup.

        The key insight: When we find an existing wrapper with the same content,
        we return that SAME wrapper (not a new one), which means all keys will
        point to the same ValueWrapper, which contains the same value object.

        NOTE: We only deduplicate hashable types. Unhashable types (lists, dicts)
        are stored as-is to preserve object identity.
        """
        # Skip deduplication for unhashable types to preserve object identity
        if not self._enable_dedup or not self._is_hashable(value):
            return ValueWrapper(value=value, content_hash=value_hash)

        # Try to get existing wrapper from dedup index
        existing = self._dedup_index.get(value_hash)
        if existing is not None:
            # Verify it's actually the same value content (handle hash collisions)
            try:
                if self._values_equal(existing.value, value):
                    self._stats["dedup_saves"] += 1
                    # Return the EXISTING wrapper (so all keys share same wrapper & value)
                    return existing
            except Exception:
                # If comparison fails, treat as different
                pass

        # Create new wrapper with the value
        wrapper = ValueWrapper(value=value, content_hash=value_hash)

        # Store in dedup index (only for hashable types)
        self._dedup_index[value_hash] = wrapper

        return wrapper

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Safe equality check for value comparison."""
        try:
            return v1 == v2
        except Exception:
            return False

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get value for key. O(1) operation.

        Args:
            key: The key to lookup
            default: Value to return if key not found

        Returns:
            The value associated with key, or default if not found
        """
        with self._lock:
            self._stats["gets"] += 1

            # Check cache first
            cached_value = self._cache.get(key, self._MISSING)
            if cached_value is not self._MISSING:
                self._stats["cache_hits"] += 1
                return cached_value

            # Lookup in main store
            wrapper = self._data.get(key, self._MISSING)
            if wrapper is self._MISSING:
                return default

            value = wrapper.value

            # Update cache
            self._cache[key] = value
            return value

    def set(self, key: Any, value: Any) -> None:
        """
        Set key to value. O(1) operation.

        Args:
            key: The key to set
            value: The value to associate with key
        """
        with self._lock:
            self._stats["sets"] += 1

            # Compute hash once
            value_hash = self._compute_hash(value)

            # Find or create deduplicated wrapper
            wrapper = self._deduplicate_value(value, value_hash)

            # Store the wrapper
            self._data[key] = wrapper

            # Update cache with the wrapper's value (not the input value!)
            # This ensures we cache the deduplicated value
            self._cache[key] = wrapper.value

    def delete(self, key: Any) -> bool:
        """
        Delete key from store. O(1) operation.

        Args:
            key: The key to delete

        Returns:
            True if key was deleted, False if key didn't exist
        """
        with self._lock:
            self._stats["deletes"] += 1

            if key not in self._data:
                return False

            # WeakValueDictionary handles cleanup automatically
            del self._data[key]

            # Remove from cache if present
            self._cache.pop(key, None)

            return True

    def has(self, key: Any) -> bool:
        """Check if key exists. O(1) operation."""
        with self._lock:
            return key in self._data

    def keys(self):
        """Return all keys."""
        with self._lock:
            return list(self._data.keys())

    def values(self):
        """Return all values."""
        with self._lock:
            return [wrapper.value for wrapper in self._data.values()]

    def items(self):
        """Return all (key, value) pairs."""
        with self._lock:
            return [(k, wrapper.value) for k, wrapper in self._data.items()]

    def clear(self) -> None:
        """Clear all data from store."""
        with self._lock:
            self._data.clear()
            self._dedup_index.clear()
            self._cache.clear()

    def size(self) -> int:
        """Return number of keys in store."""
        with self._lock:
            return len(self._data)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about store operations."""
        with self._lock:
            stats = self._stats.copy()
            stats["total_keys"] = len(self._data)
            stats["unique_values"] = len(self._dedup_index)
            stats["cache_size"] = len(self._cache)
            stats["cache_hit_rate"] = (
                stats["cache_hits"] / stats["gets"] if stats["gets"] > 0 else 0
            )
            return stats

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory usage information.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            # Count unique value references
            unique_wrappers = set(id(w) for w in self._data.values())

            return {
                "total_keys": len(self._data),
                "unique_values": len(unique_wrappers),
                "dedup_index_size": len(self._dedup_index),
                "cache_size": len(self._cache),
                "deduplication_ratio": (
                    len(self._data) / len(unique_wrappers) if unique_wrappers else 1
                ),
            }

    def __len__(self) -> int:
        """Return number of keys in store."""
        return self.size()

    def __contains__(self, key: Any) -> bool:
        """Check if key exists."""
        return self.has(key)

    def __getitem__(self, key: Any) -> Any:
        """Get value for key, raises KeyError if not found."""
        value = self.get(key, self._MISSING)
        if value is self._MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set key to value."""
        self.set(key, value)

    def __delitem__(self, key: Any) -> None:
        """Delete key, raises KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)

    def execute_async(
        self, func: Callable, *args, **kwargs
    ) -> concurrent.futures.Future:
        """
        Execute a function asynchronously using the thread pool.

        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Future object representing the async operation
        """
        if not self._async_operations or not self._executor:
            raise RuntimeError("Async operations not enabled")

        return self._executor.submit(func, *args, **kwargs)

    def close(self):
        """Clean up resources, especially the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def async_enabled(self) -> bool:
        """Check if async operations are enabled."""
        return self._async_operations and self._executor is not None


def create_store(
    cache_size: int = 10000,
    enable_dedup: bool = True,
    async_operations: bool = False,
    max_workers: int = 1,
) -> KeyValueStore:
    """
    Create a key-value store with specified settings.

    Args:
        cache_size: Size of the LRU cache
        enable_dedup: Whether to enable value deduplication
        async_operations: Whether to enable async operation support
        max_workers: Maximum number of worker threads for async operations (default: 1)

    Returns:
        Configured KeyValueStore instance
    """
    return KeyValueStore(
        enable_dedup=enable_dedup,
        cache_size=cache_size,
        async_operations=async_operations,
        max_workers=max_workers,
    )
