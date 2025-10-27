"""
Reactive Computation System

This module implements a reactive computation system based on delta-based change propagation
and adaptive materialization strategies.

Key Concepts:
- Observable: A value that can change over time and notify dependents

Materialization Strategy:
- Virtual: Computed values stay as functions until needed
- Tracked: Observable registered with DeltaKVStore for change propagation
- Materialized: Computed value stored in DeltaKVStore when fan-out detected

Change Propagation:
- Delta-based: Only changed values trigger updates (O(affected) complexity)
- Topological: Updates propagate in dependency order
- Lazy: Computations only run when values are accessed
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, TypeVar

from .base import TupleOperable, _get_dependency_keys
from .computed import (
    ComputedObservable,
    ConditionalObservable,
    ConditionNeverMet,
    StreamMerge,
)
from .delta_kv_store import Delta, DeltaKVStore

if TYPE_CHECKING:
    from .store import Store

T = TypeVar("T")


# Sentinel Types
# ============================================================================
# Import NULL_EVENT from computed to ensure singleton consistency
from .computed import NULL_EVENT

# ============================================================================
# Utility Functions
# ============================================================================


def _collect_source_values(sources: List["Observable"]) -> List[Any]:
    """Collect current values from sources, handling conditional exceptions."""
    vals = []
    for src in sources:
        try:
            val = src.value
            vals.append(val)
        except Exception as e:
            # For conditionals that have never met, treat as None
            if "never" in str(e).lower() and "active" in str(e).lower():
                vals.append(None)
            else:
                raise
    return vals


# ============================================================================
# Base Interfaces and Mixins
# ============================================================================


class ObservableInterface(ABC):
    """Abstract interface for all observable-like objects."""

    @property
    @abstractmethod
    def value(self) -> Any:
        """Get the current value."""
        pass

    @abstractmethod
    def set(self, new_value: Any) -> None:
        """Set a new value."""
        pass

    @abstractmethod
    def subscribe(
        self, callback: Callable[[Any], None], call_immediately: bool = False
    ) -> Callable[[], None]:
        """Subscribe to value changes."""
        pass


class Trackable:
    """Mixin for DeltaKVStore integration and change tracking."""

    def __init__(self):
        self._is_tracked = False

    def _track_in_store(self):
        """Override in subclasses to implement specific tracking logic."""
        raise NotImplementedError

    def _ensure_tracked(self):
        """Ensure this observable is tracked in the store."""
        if not self._is_tracked:
            self._track_in_store()


class Subscribable:
    """Mixin for subscription management."""

    def __init__(self):
        self._subscriptions = []

    def _add_subscription(self, unsubscribe_fn: Callable[[], None]) -> None:
        """Add a subscription to be managed."""
        self._subscriptions.append(unsubscribe_fn)

    def _clear_subscriptions(self) -> None:
        """Clear all managed subscriptions."""
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()


class Materializable:
    """Mixin for materialization logic and dependency tracking."""

    def __init__(self):
        self._materialized_key = None
        self._dependents_count = 0

    def _register_dependent(self):
        """Called when a computed depends on this observable."""
        self._dependents_count += 1

    def _should_materialize_for_fanout(self) -> bool:
        """Check if we should materialize due to fan-out."""
        return self._dependents_count >= 2 and not self._is_materialized()

    def _is_materialized(self) -> bool:
        """Check if this observable is materialized."""
        return self._materialized_key is not None


class ValueOperable:
    """Mixin providing operator implementations for value-based observables."""

    def then(self, transform: Callable):
        """Apply transformation: obs >> f → f(obs)"""
        self._register_dependent()
        return ComputedObservable(self._store, [self], lambda x: transform(x))

    def alongside(self, *others):
        """Combine streams: (obs₁, obs₂, ..., obsₙ)"""
        all_obs = (self,) + others
        return self._store._get_or_create_stream(all_obs)

    def requiring(self, condition):
        """Filter by condition: only emit when conditions are met"""
        self._register_dependent()
        # ConditionalObservable handles both callables and observables
        return ConditionalObservable(self._store, self, condition)

    def either(self, other):
        """Logical OR: bool(a) or bool(b)"""
        self._register_dependent()
        other._register_dependent()
        return ComputedObservable(
            self._store, [self, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self):
        """Logical NOT: not bool(obs)"""
        self._register_dependent()
        return ComputedObservable(self._store, [self], lambda x: not bool(x))

    # Operator overloads
    __rshift__ = then
    __add__ = alongside
    __and__ = requiring
    __or__ = either
    __invert__ = negate


# ============================================================================
# Base Observable Classes
# ============================================================================


class ObservableBase(ObservableInterface):
    """Base class for all observable implementations with common functionality."""

    def __init__(self, store):
        self._store = store
        self._key = None  # Will be set by subclasses

    def set(self, new_value: Any) -> None:
        """Set a new value - override in subclasses."""
        raise NotImplementedError


# ============================================================================
# Smart Observable - Adaptive Materialization
# ============================================================================


class Observable(ObservableBase, Trackable, ValueOperable, Materializable):
    """
    Observable value with optimal performance for untracked state.

    Theoretical optimal path for untracked:
    - Read:  O(1) - direct attribute access
    - Write: O(1) - direct assignment

    When tracked, delegates to node for store integration.
    """

    __slots__ = (
        "_store",
        "_key",
        "_value",
        "_node",
        "_subscribers",
        "_notifying",
        "_is_tracked",
    )

    def __init__(self, store: "Store", key: str, initial_value: Any = None):
        ObservableBase.__init__(self, store)
        Trackable.__init__(self)
        ValueOperable.__init__(self)
        Materializable.__init__(self)
        self._key = key
        self._value = initial_value  # Optimal: direct storage
        self._subscribers = []
        self._notifying = False
        self._is_tracked = False
        self._node = None  # Lazy: create only when tracked

    @property
    def value(self) -> Any:
        """Get the current value - optimized for untracked path."""
        if not self._is_tracked:
            return self._value
        return self._store._kv.get(self._key)

    @value.setter
    def value(self, new_value: Any) -> None:
        if not self._is_tracked:
            # Untracked: optimized for Window benchmark pattern
            # Check identity first for O(1) common case
            if self._value is new_value:
                # Same reference: if mutable type, allow in-place mutations to propagate
                if isinstance(self._value, (list, dict, set)):
                    # Update to trigger reactivity even though value unchanged
                    self._value = new_value
                    if self._subscribers:
                        self._notify_legacy_subscribers(new_value)
                # For immutable, skip update (no change possible)
            else:
                # Different reference: update
                self._value = new_value
                if self._subscribers:
                    self._notify_legacy_subscribers(new_value)
        else:
            # Tracked: use store for change propagation
            if self._notifying:
                raise RuntimeError("Circular dependency detected")
            self._notifying = True
            self._store._kv.set(self._key, new_value)
            self._notifying = False

    def _notify_legacy_subscribers(self, value: Any) -> None:
        """Notify legacy subscribers with proper notifying flag management."""
        self._notifying = True
        try:
            for cb in self._subscribers:
                cb(value)
        finally:
            self._notifying = False

    def set(self, new_value: Any) -> None:
        """Set the value - aliases value setter."""
        self.value = new_value

    def _track_in_store(self):
        """Transition to tracked mode for change propagation."""
        if self._is_tracked:
            return

        self._is_tracked = True
        # Register current value with store
        self._store._kv.set(self._key, self._value)

        # Create node for dependency tracking
        factory = self._store._kv._get_node_factory()
        self._node = factory.create_source(self._key, self._value)
        self._node.mark_tracked()

        # Invalidate stream cache
        self._store._stream_cache.clear() if self._store._stream_cache else None

        # Migrate legacy subscribers
        if self._subscribers:
            for cb in self._subscribers:
                self._store._kv.subscribe(
                    self._key, lambda d: cb(d.new_value) if d.key == self._key else None
                )
            self._subscribers = None

    def _register_dependent(self):
        """Called when a computed depends on this observable."""
        Materializable._register_dependent(self)
        # DO NOT track automatically - only track when subscription or propagation needed
        # This keeps untracked observables fast for common cases

    def subscribe(
        self, callback: Callable[[Any], None], call_immediately: bool = False
    ) -> Callable[[], None]:
        """Subscribe to value changes."""
        self._track_in_store()

        def on_delta(delta: Delta):
            if (
                delta.key == self._key
                and delta.new_value is not None
                and delta.new_value is not NULL_EVENT
            ):
                self._notifying = True
                try:
                    callback(delta.new_value)
                finally:
                    self._notifying = False

        unsubscribe = self._store._kv.subscribe(self._key, on_delta)

        # Call immediately if requested
        if call_immediately:
            try:
                value = self.value
                if value is not None and value is not NULL_EVENT:
                    self._notifying = True
                    try:
                        callback(value)
                    finally:
                        self._notifying = False
            except ConditionNeverMet:
                pass

        return unsubscribe

    def __repr__(self) -> str:
        state = "tracked" if self._is_tracked else "virtual"
        return f"Observable({self._key}={self.value}, {state})"


# ============================================================================
# Smart Computed - Intelligent Chain Handling
# ============================================================================


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
        # Initialize node factory
        self._node_factory = self._kv._get_node_factory()

    def _get_or_create_stream(self, sources: tuple) -> "StreamMerge":
        """Get cached StreamMerge or create new one."""

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
# ConditionNotMet - Exception for unmet conditions
# ============================================================================


class ConditionNotMet(Exception):
    """Raised when accessing a conditional observable with unmet conditions."""

    pass


# ============================================================================
# Transaction support
# ============================================================================


def transaction():
    """Create a transaction context for safe reentrant updates."""
    return get_global_store().batch()


def reactive(*dependencies, autorun: bool = True):
    def decorator(func: Callable) -> Callable:
        # Reactive functions call immediately by default, unless autorun=False or conditional not satisfied

        # Subscribe for future changes
        unsubscribers = []
        for dep in dependencies:
            if hasattr(dep, "subscribe"):
                # Handle SubscriptableDescriptor wrappers
                actual_dep = (
                    dep._original_observable
                    if hasattr(dep, "_original_observable")
                    else dep
                )
                # Call immediately for computed observables
                call_now = autorun

                # Special handling for conditional observables
                if actual_dep.__class__.__name__ == "ConditionalObservable":
                    # Conditionals should not call immediately by default
                    # They only fire on state transitions (active ↔ inactive)
                    call_now = False

                    # Check if already active on first access
                    try:
                        value = actual_dep.value
                        if value is not None and value is not NULL_EVENT:
                            # Already active - don't call, wait for transition
                            pass
                    except Exception as e:
                        if "Conditional" in str(type(e)) or "never" in str(e).lower():
                            # Never been active - call with False
                            func(False)
                        else:
                            raise

                unsubscribers.append(
                    dep.subscribe(lambda value: func(value), call_immediately=call_now)
                )

        # Return wrapper that prevents manual calls
        def unsubscribe():
            for unsub in unsubscribers:
                unsub()
            wrapper._unsubscribed = True

        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "_unsubscribed") or not wrapper._unsubscribed:
                raise RuntimeError(
                    "Reactive functions cannot be called manually. They are called automatically when dependencies change."
                )
            return func(*args, **kwargs)

        wrapper.unsubscribe = unsubscribe
        return wrapper

    return decorator
