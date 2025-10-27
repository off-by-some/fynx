"""
Global Store - singleton store for backwards compatibility.

Provides global_observable() function creating observables without explicit
Store. Used for simple cases where Store-based organization not needed.

Implementation:
    - get_global_store(): lazy singleton pattern
    - GlobalStore: wrapper around DeltaKVStore with key generation
    - observable(): factory function using global store

Thread Safety:
    _global_store access synchronized via module-level global keyword.
"""

from typing import Any

from .delta_kv_store import DeltaKVStore

# Import here to avoid circular dependency
_OBSERVABLE_CLASS = None  # Will be imported lazily


class GlobalStore:
    """
    Singleton store wrapper around DeltaKVStore.

    Provides key generation and observable caching for global observable()
    function. Manages _key_counter for unique key generation and _stream_cache
    for StreamObservable instance reuse.

    Cache Key Strategy:
        Uses tuple of source identifiers as cache key, with id() fallback
        for virtual computeds. This enables StreamObservable reuse across
        multiple then() calls on same sources.
    """

    def __init__(self):
        self._kv = DeltaKVStore()
        self._observables = {}
        self._key_counter = 0
        self._stream_cache = {}  # Cache StreamObservable instances

    def _get_or_create_stream(self, sources: tuple) -> "StreamObservable":
        """
        Get cached StreamObservable or create new one.

        Cache key is tuple of source identifiers (keys or ids). Enables
        StreamObservable reuse when same sources used in multiple streams.
        """

        # Use tuple of source identifiers as cache key
        def get_source_id(src):
            if hasattr(src, "_key"):
                return src._key
            elif hasattr(src, "_materialized_key") and src._materialized_key:
                return src._materialized_key
            else:
                # For virtual ComputedObservable, use id-based key
                return f"virtual_{id(src)}"

        cache_key = tuple(get_source_id(src) for src in sources)
        if cache_key not in self._stream_cache:
            from .observable import StreamObservable

            # Register sources as dependents before creating stream
            for source in sources:
                try:
                    source._register_dependent()
                except AttributeError:
                    pass

            self._stream_cache[cache_key] = StreamObservable(self, list(sources))
        return self._stream_cache[cache_key]

    def observable(self, key: str, initial_value: Any = None) -> Any:
        """
        Create or get Observable with given key.

        Caches Observable instances to enable reuse. Lazy imports Observable
        class to avoid circular dependencies.
        """
        if key not in self._observables:
            global _OBSERVABLE_CLASS
            if _OBSERVABLE_CLASS is None:
                from .observable import Observable

                _OBSERVABLE_CLASS = Observable
            self._observables[key] = _OBSERVABLE_CLASS(self, key, initial_value)
        return self._observables[key]

    def batch(self):
        """Return DeltaKVStore batch context for transaction support."""
        return self._kv.batch()

    def _gen_key(self, prefix: str) -> str:
        """
        Generate unique key with prefix.

        Format: {prefix}${incrementing_number}. Used for Observable keys,
        StreamObservable keys, and ComputedObservable materialization keys.
        """
        self._key_counter += 1
        return f"{prefix}${self._key_counter}"


_global_store = None


def get_global_store() -> GlobalStore:
    """
    Get or create the global store instance.

    Lazy singleton pattern: creates on first access, reuses thereafter.
    Testing can reset via _reset_global_store().
    """
    global _global_store
    if _global_store is None:
        _global_store = GlobalStore()
    return _global_store


def _reset_global_store() -> None:
    """
    Reset the global store for testing purposes.

    Clears singleton to enable fresh state in tests. Not for production use.
    """
    global _global_store
    _global_store = None


def observable(initial_value: Any = None) -> Any:
    """
    Create Observable using global store.

    Factory function for simple reactive values not requiring Store organization.
    Generates unique key via _key_counter and returns cached Observable instance.

    Returns: Observable instance with generated key
    """
    store = get_global_store()
    store._key_counter += 1
    key = f"obs${store._key_counter}"
    return store.observable(key, initial_value)


def transaction():
    """
    Create transaction context for batch operations.

    Returns DeltaKVStore batch context enabling batched updates with
    single propagation cycle. Supports nested transactions.
    """
    return get_global_store().batch()
