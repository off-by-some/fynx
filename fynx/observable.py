"""
Optimal Fynx API Layer - Smart DeltaKVStore Usage
=================================================

Design Principles:
1. Use DeltaKVStore ONLY when it provides value:
   - Multiple dependents (fan-out)
   - Need for change tracking
   - Complex dependency graphs

2. Stay virtual when:
   - Single-use computations (no fan-out)
   - Simple linear chains
   - Temporary values

3. Smart materialization heuristics:
   - Materialize on first subscribe
   - Materialize when used as dependency by 2+ computeds
   - Keep linear chains virtual until branch point

Performance Goals:
- Observable creation: 200k+ ops/sec (current: 176k)
- Individual updates: 200k+ ops/sec (current: 216k)
- Chain propagation: 40k+ ops/sec (current: 8.8k) ← KEY TARGET
- Stream ops: 700k+ ops/sec (current: 718k) ✓
"""

import threading
from typing import Any, Callable, Generic, List, Optional, Set, TypeVar

from .delta_kv_store import ChangeType, Delta, DeltaKVStore

T = TypeVar("T")


# ============================================================================
# Smart Observable - Adaptive Materialization
# ============================================================================


class Observable:
    """
    Adaptive observable that materializes based on usage patterns.

    States:
    1. Virtual: In-memory value only (80 bytes)
    2. Tracked: DeltaKVStore entry for dependency tracking (400 bytes)

    Transitions:
    - Virtual → Tracked when:
      a) First subscription
      b) Used by 2+ dependents (fan-out detected)
      c) Part of complex dependency graph
    """

    __slots__ = (
        "_store",
        "_key",
        "_value",
        "_is_tracked",
        "_subscribers",
        "_dependents_count",
        "_notifying",
    )

    def __init__(self, store: "Store", key: str, initial_value: Any = None):
        self._store = store
        self._key = key
        self._value = initial_value
        self._is_tracked = False
        self._subscribers = None
        self._dependents_count = 0
        self._notifying = False  # Track if we're currently notifying subscribers

    @property
    def value(self) -> Any:
        if self._is_tracked:
            return self._store._kv.get(self._key)
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        # Check for circular dependency - don't allow setting while notifying subscribers
        if self._notifying:
            raise RuntimeError("Circular dependency detected")

        if self._is_tracked:
            self._notifying = True
            try:
                self._store._kv.set(self._key, new_value)
            finally:
                self._notifying = False
        else:
            old = self._value
            self._value = new_value
            if self._subscribers:
                self._notifying = True
                try:
                    for cb in self._subscribers:
                        cb(new_value)
                finally:
                    self._notifying = False

    def set(self, new_value: Any) -> None:
        """Alias for value setter."""
        self.value = new_value

    def _track_in_store(self):
        """Upgrade to tracked mode in DeltaKVStore."""
        if self._is_tracked:
            return

        self._is_tracked = True
        self._store._kv.set(self._key, self._value)

        if self._subscribers:
            for cb in self._subscribers:
                self._store._kv.subscribe(
                    self._key, lambda d: cb(d.new_value) if d.key == self._key else None
                )
            self._subscribers = None

        # Invalidate stream cache that might reference this observable
        # since it changed from virtual to tracked
        self._store._stream_cache.clear()

    def _register_dependent(self):
        """Called when a computed depends on this observable."""
        self._dependents_count += 1

        # Fan-out detected: materialize for efficient propagation
        if self._dependents_count >= 2 and not self._is_tracked:
            self._track_in_store()

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe - promotes to tracked for change propagation."""
        # First subscription: go tracked for proper change propagation
        if not self._is_tracked:
            self._track_in_store()

        def on_delta(delta: Delta):
            if delta.key == self._key:
                callback(delta.new_value)

        return self._store._kv.subscribe(self._key, on_delta)

    # ========================================================================
    # Operators
    # ========================================================================

    def then(self, transform: Callable[[T], Any]) -> "SmartComputed":
        """Transform: obs >> f"""
        self._register_dependent()
        return SmartComputed(self._store, [self], lambda v: transform(v))

    def alongside(self, *others: "Observable") -> "StreamMerge":
        """
        Combine: obs1 + obs2

        Returns cached StreamMerge for minimal allocation.
        """
        all_obs = (self,) + others  # Use tuple for hashability
        return self._store._get_or_create_stream(all_obs)

    def requiring(self, condition: "Observable") -> "SmartComputed":
        """Filter: obs & condition"""
        self._register_dependent()
        # Check if condition is actually an observable
        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()
        elif callable(condition):
            # It's a predicate function, not an observable
            # Create a computed that applies the predicate
            return SmartComputed(
                self._store, [self], lambda val: val if condition(val) else None
            )
        return SmartComputed(
            self._store,
            [self, condition],
            lambda val, cond: val if bool(cond) else None,
        )

    def either(self, other: "Observable") -> "SmartComputed":
        """OR: obs1 | obs2"""
        self._register_dependent()
        other._register_dependent()
        return SmartComputed(
            self._store, [self, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self) -> "SmartComputed":
        """NOT: ~obs"""
        self._register_dependent()
        return SmartComputed(self._store, [self], lambda x: not bool(x))

    __rshift__ = then
    __add__ = alongside
    __and__ = requiring
    __or__ = either
    __invert__ = negate

    def __repr__(self) -> str:
        state = "tracked" if self._is_tracked else "virtual"
        return f"Observable({self._key}={self.value}, {state}, deps={self._dependents_count})"


# ============================================================================
# StreamMerge - Ultra-Fast Stream Operations
# ============================================================================


class StreamMerge:
    """
    Event-driven stream merge with zero-allocation cached values.

    Performance characteristics:
    - O(1) value access (cached tuple)
    - O(1) per source update (single index update + tuple recreation)
    - Lazy subscription setup (only when needed)
    - Zero allocation on repeated .value access

    Memory: ~200 bytes + 8*n bytes for n sources (vs 80 bytes per Observable)
    """

    __slots__ = (
        "_sources",  # tuple[Observable]: Source observables
        "_cached_values",  # list: Individual cached values [fast indexed update]
        "_cached_tuple",  # tuple: Cached combined tuple [fast repeated access]
        "_subscriptions",  # list[Callable]: Unsubscribe functions
        "_callbacks",  # set[Callable]: Registered callbacks
        "_is_subscribed",  # bool: Track if we've set up source subscriptions
    )

    def __init__(self, store: "Store", sources: List[Observable]):
        self._sources = tuple(sources)

        # Initialize cached values - pay upfront cost once
        self._cached_values = [src.value for src in self._sources]
        self._cached_tuple = tuple(self._cached_values)

        # Lazy subscription state
        self._subscriptions = []
        self._callbacks = set()
        self._is_subscribed = False

    @property
    def value(self) -> tuple:
        """
        Return cached tuple - zero allocation.

        This is the fast path: just return the cached tuple.
        No tuple creation, no source access, just a single attribute lookup.
        """
        return self._cached_tuple

    def set(self, new_value: Any) -> None:
        """Stream merges are read-only."""
        pass  # No-op for API compatibility

    def subscribe(self, callback: Callable[[tuple], None]) -> Callable[[], None]:
        """
        Subscribe to stream updates.

        Lazy subscription model:
        - First subscriber triggers source subscriptions
        - Subsequent subscribers just add to callback set
        - Callbacks fire when ANY source updates
        """
        # Add callback to set
        self._callbacks.add(callback)

        # Lazy setup: only subscribe to sources on first subscriber
        if not self._is_subscribed:
            self._setup_source_subscriptions()
            self._is_subscribed = True

        # Immediately call with current value (standard reactive behavior)
        callback(self._cached_tuple)

        # Return unsubscribe function
        def unsubscribe():
            self._callbacks.discard(callback)
            # Optional: cleanup source subscriptions if no more callbacks
            if not self._callbacks and self._is_subscribed:
                self._teardown_source_subscriptions()

        return unsubscribe

    def _setup_source_subscriptions(self):
        """
        Subscribe to all sources for reactive updates.

        Creates one subscription per source with an update handler
        that updates the cache and notifies callbacks.
        """
        for idx, src in enumerate(self._sources):
            # Create closure that captures the index
            def make_update_handler(i):
                def update_handler(new_val):
                    # Update cached value at index
                    self._cached_values[i] = new_val

                    # Recreate cached tuple
                    # This is the only allocation cost per update
                    self._cached_tuple = tuple(self._cached_values)

                    # Notify all subscribers
                    self._notify_callbacks()

                return update_handler

            # Subscribe to source
            handler = make_update_handler(idx)
            unsub = src.subscribe(handler)
            self._subscriptions.append(unsub)

    def _teardown_source_subscriptions(self):
        """Clean up source subscriptions when no longer needed."""
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()
        self._is_subscribed = False

    def _notify_callbacks(self):
        """Notify all callbacks with current cached tuple."""
        # Make a copy of callbacks in case callback adds/removes subscribers
        for cb in list(self._callbacks):
            try:
                cb(self._cached_tuple)
            except Exception as e:
                # Log error but don't break other callbacks
                # In production, use proper logging
                print(f"StreamMerge callback error: {e}")

    # ========================================================================
    # Operators - Maintain fluent API
    # ========================================================================

    def then(self, transform: Callable) -> "SmartComputed":
        """
        Transform the merged stream.

        Creates a SmartComputed that depends on all sources.
        The transform receives the tuple of values.
        """
        store = self._sources[0]._store

        # Register all sources as having a dependent
        for src in self._sources:
            if hasattr(src, "_register_dependent"):
                src._register_dependent()

        return SmartComputed(
            store,
            list(self._sources),
            lambda *vals: transform(*vals),  # Unpack tuple as individual args
        )

    def alongside(self, *others: "Observable") -> "StreamMerge":
        """
        Extend the merge with additional observables.

        Returns a cached StreamMerge instance for the combined sources.
        """
        store = self._sources[0]._store
        all_obs = self._sources + others

        # Register new sources as having dependents
        for obs in others:
            if hasattr(obs, "_register_dependent"):
                obs._register_dependent()

        return store._get_or_create_stream(all_obs)

    def requiring(self, condition: "Observable") -> "SmartComputed":
        """Filter: only emit when condition is truthy."""
        store = self._sources[0]._store

        # Register all sources as having dependents
        for src in self._sources:
            if hasattr(src, "_register_dependent"):
                src._register_dependent()

        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()

        # Create computed that filters based on condition
        return SmartComputed(
            store,
            list(self._sources) + [condition],
            lambda *vals: vals[:-1] if bool(vals[-1]) else None,
        )

    # Operator overloads
    __rshift__ = then
    __add__ = alongside
    __and__ = requiring

    def __repr__(self) -> str:
        state = "subscribed" if self._is_subscribed else "unsubscribed"
        return (
            f"StreamMerge({len(self._sources)} sources, "
            f"{len(self._callbacks)} callbacks, {state})"
        )


# ============================================================================
# Smart Computed - Intelligent Chain Handling
# ============================================================================


class SmartComputed:
    """
    Computed observable with optimal DeltaKVStore usage.

    Key Optimization: Linear chains stay virtual and fused.
    Only materialize at branch points and subscriptions.

    Example:
        a >> f1 >> f2 >> f3  (linear, stays virtual)
        b = a >> f1 >> f2
        c = b >> f3          (branch point, b materializes)
        d = b >> f4          (branch point, b materializes)
    """

    __slots__ = (
        "_store",
        "_sources",
        "_fused_fn",
        "_materialized_key",
        "_dependents_count",
        "_is_subscribed",
    )

    def __init__(self, store: "Store", sources: List[Observable], compute_fn: Callable):
        self._store = store
        self._sources = sources
        self._fused_fn = compute_fn
        self._materialized_key = None
        self._dependents_count = 0
        self._is_subscribed = False

    @property
    def value(self) -> Any:
        """Fast path: compute directly if virtual."""
        if self._materialized_key:
            return self._store._kv.get(self._materialized_key)

        # Direct computation - fast for linear chains
        vals = [src.value for src in self._sources]
        if len(vals) == 1:
            return self._fused_fn(vals[0])
        return self._fused_fn(*vals)

    def set(self, new_value: Any) -> None:
        """Computed values are read-only."""
        pass  # No-op for API compatibility

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        """Subscription forces materialization."""
        self._is_subscribed = True
        return self._materialize_full_chain().subscribe(callback)

    def _register_dependent(self):
        """Called when another computed depends on this one."""
        self._dependents_count += 1

        # Branch point detected: materialize for fan-out efficiency
        if self._dependents_count >= 2 and not self._materialized_key:
            self._materialize_as_node()

    def _materialize_as_node(self) -> Observable:
        """
        Materialize THIS node into DeltaKVStore.
        Sources stay as-is (may be virtual or tracked).
        """
        if self._materialized_key:
            obs = self._store._observables.get(self._materialized_key)
            if obs:
                return obs

        # Create key
        self._materialized_key = self._store._gen_key("node")

        # Build compute function
        compute_fn = self._fused_fn
        sources = self._sources

        def compute():
            vals = [src.value for src in sources]
            if len(vals) == 1:
                return compute_fn(vals[0])
            return compute_fn(*vals)

        # Get dependency keys
        dep_keys = []
        for src in sources:
            if isinstance(src, SmartComputed):
                # Source is computed - ensure it's materialized for dependency tracking
                if not src._materialized_key:
                    src._materialize_as_node()
                dep_keys.append(src._materialized_key)
            else:
                # Source is observable - track it if not already
                if not src._is_tracked:
                    src._track_in_store()
                dep_keys.append(src._key)

        # Register with DeltaKVStore
        self._store._kv.computed(self._materialized_key, compute, deps=dep_keys)

        # Create observable wrapper
        obs = Observable(self._store, self._materialized_key)
        obs._is_tracked = True
        self._store._observables[self._materialized_key] = obs
        return obs

    def _materialize_full_chain(self) -> Observable:
        """
        Materialize the entire chain for subscriptions.

        Strategy: Materialize the LEAF (this node) which will recursively
        materialize all dependencies via _materialize_as_node.
        """
        return self._materialize_as_node()

    # ========================================================================
    # Operators - Chain building with fusion
    # ========================================================================

    def then(self, transform: Callable) -> "SmartComputed":
        """
        Extend the chain with operation fusion.

        Key insight: If this computed has no other dependents,
        fuse the operation instead of creating a new node.
        """
        self._register_dependent()

        # If we're already materialized OR have multiple dependents,
        # create a new computed that depends on us
        if self._materialized_key or self._dependents_count > 1:
            materialized = self._materialize_as_node()
            return SmartComputed(self._store, [materialized], transform)

        # Fuse operations: create new computed with combined function
        old_fn = self._fused_fn

        def fused(*vals):
            intermediate = old_fn(*vals) if len(vals) > 1 else old_fn(vals[0])
            # Handle case where old_fn returns multiple values (like StreamMerge)
            if isinstance(intermediate, tuple):
                return transform(*intermediate)
            else:
                return transform(intermediate)

        return SmartComputed(self._store, self._sources, fused)

    def alongside(self, *others: "Observable") -> "StreamMerge":
        """
        Combine with other observables.

        Returns cached StreamMerge for optimal stream performance.
        """
        all_obs = (self,) + others
        store = self._store
        return store._get_or_create_stream(all_obs)

    def requiring(self, condition: "Observable") -> "SmartComputed":
        """Add conditional filter."""
        self._register_dependent()

        # Check if condition is callable (predicate function)
        if callable(condition) and not hasattr(condition, "_register_dependent"):
            # It's a predicate function
            materialized = self._materialize_as_node()
            return SmartComputed(
                self._store, [materialized], lambda val: val if condition(val) else None
            )

        # It's an observable
        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()

        materialized = self._materialize_as_node()
        return SmartComputed(
            self._store,
            [materialized, condition],
            lambda val, cond: val if bool(cond) else None,
        )

    def either(self, other: "Observable") -> "SmartComputed":
        """Logical OR."""
        self._register_dependent()
        if hasattr(other, "_register_dependent"):
            other._register_dependent()

        materialized = self._materialize_as_node()
        return SmartComputed(
            self._store, [materialized, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self) -> "SmartComputed":
        """Boolean negation with fusion."""
        self._register_dependent()

        # Fuse if possible
        if not self._materialized_key and self._dependents_count == 1:
            old_fn = self._fused_fn

            def fused(*vals):
                result = old_fn(*vals) if len(vals) > 1 else old_fn(vals[0])
                return not bool(result)

            return SmartComputed(self._store, self._sources, fused)

        materialized = self._materialize_as_node()
        return SmartComputed(self._store, [materialized], lambda x: not bool(x))

    __rshift__ = then
    __add__ = alongside
    __and__ = requiring
    __or__ = either
    __invert__ = negate

    def __repr__(self) -> str:
        state = "materialized" if self._materialized_key else "virtual"
        return f"SmartComputed({state}, deps={self._dependents_count}, sources={len(self._sources)})"


# ============================================================================
# Transaction support
# ============================================================================
_global_store = None


class GlobalStore:
    """Legacy store implementation for global observables."""

    def __init__(self):
        self._kv = DeltaKVStore()
        self._observables = {}
        self._key_counter = 0
        self._stream_cache = {}  # Cache StreamMerge instances

    def _get_or_create_stream(self, sources: tuple) -> "StreamMerge":
        """Get cached StreamMerge or create new one."""

        # Use tuple of source identifiers as cache key
        def get_source_id(src):
            if hasattr(src, "_key"):
                return src._key
            elif hasattr(src, "_materialized_key") and src._materialized_key:
                return src._materialized_key
            else:
                # For virtual SmartComputed, use id-based key
                return f"virtual_{id(src)}"

        cache_key = tuple(get_source_id(src) for src in sources)
        if cache_key not in self._stream_cache:
            self._stream_cache[cache_key] = StreamMerge(self, list(sources))
        return self._stream_cache[cache_key]

    def observable(self, key: str, initial_value: Any = None) -> Observable:
        """Create or get observable."""
        if key not in self._observables:
            self._observables[key] = Observable(self, key, initial_value)
        return self._observables[key]

    def batch(self):
        return self._kv.batch()

    def _gen_key(self, prefix: str) -> str:
        self._key_counter += 1
        return f"{prefix}${self._key_counter}"


def get_global_store():
    global _global_store
    if _global_store is None:
        _global_store = GlobalStore()
    return _global_store


def _reset_global_store():
    """Reset the global store for testing purposes."""
    global _global_store
    _global_store = None


def observable(initial_value: Any = None) -> Observable:
    store = get_global_store()
    store._key_counter += 1
    key = f"obs${store._key_counter}"
    return store.observable(key, initial_value)


# ============================================================================
# SubscriptableDescriptor - Type-safe Observable Descriptors
# ============================================================================


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor for subscriptable observables used in Store classes.

    This descriptor wraps Observable instances and provides type-safe access
    for Store class attributes.
    """

    def __init__(
        self,
        initial_value: Optional[T] = None,
        original_observable: Optional[Any] = None,
    ):
        self._initial_value = initial_value
        self._original_observable = original_observable

    def __get__(self, instance, owner):
        """Get the observable value for class access."""
        if instance is None:
            # Class access - return the original observable or create a new one
            if self._original_observable is not None:
                return self._original_observable
            # Create a new observable if none exists
            return Observable("standalone", self._initial_value)
        return self

    def __set__(self, instance, value):
        """Setting on instances is not supported."""
        raise AttributeError("can't set attribute on instance")

    @property
    def value(self):
        """Get the current value."""
        if self._original_observable is not None:
            return self._original_observable.value
        return self._initial_value

    def set(self, value):
        """Set the value."""
        if self._original_observable is not None:
            self._original_observable.set(value)


# ============================================================================
# Transaction support
# ============================================================================


def transaction():
    """Create a transaction context for safe reentrant updates."""
    return get_global_store().batch()


def reactive(*dependencies):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for dep in dependencies:
                if hasattr(dep, "subscribe"):
                    dep.subscribe(lambda _: func(*args, **kwargs))
            return func

        return wrapper

    return decorator
