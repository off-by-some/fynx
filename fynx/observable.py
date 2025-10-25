"""
Reactive Computation System

This module implements a reactive computation system based on delta-based change propagation
and adaptive materialization strategies.

Key Concepts:
- Observable: A value that can change over time and notify dependents
- StreamMerge: Combines multiple observable streams into a single tuple stream
- SmartComputed: Computed values with intelligent materialization

Materialization Strategy:
- Virtual: Computed values stay as functions until needed
- Tracked: Observable registered with DeltaKVStore for change propagation
- Materialized: Computed value stored in DeltaKVStore when fan-out detected

Change Propagation:
- Delta-based: Only changed values trigger updates (O(affected) complexity)
- Topological: Updates propagate in dependency order
- Lazy: Computations only run when values are accessed
"""

import threading
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Set, TypeVar

from .delta_kv_store import ChangeType, Delta, DeltaKVStore

if TYPE_CHECKING:
    from .store import Store

T = TypeVar("T")


# ============================================================================
# Smart Observable - Adaptive Materialization
# ============================================================================


class Observable:
    """
    Observable value with adaptive materialization.

    An observable represents a value that can change over time. It maintains a dependency
    graph and propagates changes to all dependent computations.

    Materialization States:
    - Virtual: Value stored in memory, no change tracking overhead
    - Tracked: Registered with DeltaKVStore for efficient change propagation

    Transition Rules:
    - Virtual → Tracked on first subscription (enables O(affected) updates)
    - Virtual → Tracked when dependency count ≥ 2 (fan-out optimization)
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

    def subscribe(
        self, callback: Callable[[Any], None], call_immediately: bool = False
    ) -> Callable[[], None]:
        """Subscribe - promotes to tracked for change propagation."""
        # First subscription: go tracked for proper change propagation
        if not self._is_tracked:
            self._track_in_store()

        def on_delta(delta: Delta):
            if delta.key == self._key and delta.new_value is not None:
                callback(delta.new_value)

        unsubscribe = self._store._kv.subscribe(self._key, on_delta)

        # Call immediately if requested and value is not None and can be computed
        if call_immediately:
            try:
                value = self.value
                if value is not None:
                    callback(value)
            except ConditionNeverMet:
                # Don't call immediately if conditions have never been met
                pass

        return unsubscribe

    # ========================================================================
    # Operators
    # ========================================================================

    def then(self, transform: Callable[[T], Any]) -> "SmartComputed":
        """Apply transformation: obs >> f → f(obs)"""
        self._register_dependent()
        return SmartComputed(self._store, [self], lambda v: transform(v))

    def alongside(self, *others: "Observable") -> "StreamMerge":
        """Combine streams: (obs₁, obs₂, ..., obsₙ)"""
        all_obs = (self,) + others  # Use tuple for hashability
        return self._store._get_or_create_stream(all_obs)

    def requiring(self, condition: "Observable") -> "ConditionalObservable":
        """Filter by condition: only emit when conditions are met"""
        self._register_dependent()
        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()
        elif callable(condition):
            # For callable conditions, create a wrapper observable that tracks the condition
            condition_obs = SmartComputed(
                self._store, [self], lambda val: condition(val)
            )
            return ConditionalObservable(self._store, self, condition_obs)
        return ConditionalObservable(self._store, self, condition)

    def either(self, other: "Observable") -> "SmartComputed":
        """Logical OR: bool(a) or bool(b)"""
        self._register_dependent()
        other._register_dependent()
        return SmartComputed(
            self._store, [self, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self) -> "SmartComputed":
        """Logical NOT: not bool(obs)"""
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
    Merges multiple observable streams into a single tuple stream.

    Given observables A₁, A₂, ..., Aₙ, produces stream (A₁, A₂, ..., Aₙ).

    Key Properties:
    - Cached tuple: O(1) access to current values
    - Incremental updates: Only changed indices updated on source changes
    - Lazy subscription: Sources only subscribed to when StreamMerge is subscribed
    - Memory efficient: Single tuple allocation per update cycle
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
        """Return current combined values as tuple."""
        return self._cached_tuple

    def set(self, new_value: Any) -> None:
        """Stream merges are read-only."""
        pass  # No-op for API compatibility

    def subscribe(
        self, callback: Callable[[tuple], None], call_immediately: bool = True
    ) -> Callable[[], None]:
        """Subscribe to tuple updates. Lazy: sources only subscribed on first subscriber."""
        # Add callback to set
        self._callbacks.add(callback)

        # Lazy setup: only subscribe to sources on first subscriber
        if not self._is_subscribed:
            self._setup_source_subscriptions()
            self._is_subscribed = True

        # Immediately call with current value if requested
        if call_immediately:
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

    def requiring(self, condition: "Observable") -> "ConditionalObservable":
        """Filter: only emit when condition is truthy."""
        store = self._sources[0]._store

        # Register all sources as having dependents
        for src in self._sources:
            if hasattr(src, "_register_dependent"):
                src._register_dependent()

        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()

        # Create a computed that evaluates the condition on the tuple
        def condition_fn(*vals):
            return bool(vals[-1])  # Last value is the condition

        # Create computed for the condition evaluation
        condition_computed = SmartComputed(
            store, list(self._sources) + [condition], condition_fn
        )

        # Create conditional observable that filters the tuple
        return ConditionalObservable(store, self, condition_computed)

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
    Computed value with intelligent materialization.

    Represents f(A₁, A₂, ..., Aₙ) where each Aᵢ is an observable.

    Materialization Strategy:
    - Linear chains: f₁(f₂(...fₖ(x)...)) stays as composed function
    - Branch points: Materialize intermediate results when multiple dependents
    - Subscriptions: Force materialization for change propagation

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
        "_cached_value",
        "_is_dirty",
    )

    def __init__(self, store: "Store", sources: List[Observable], compute_fn: Callable):
        self._store = store
        self._sources = sources
        self._fused_fn = compute_fn
        self._materialized_key = None
        self._dependents_count = 0
        self._is_subscribed = False
        self._cached_value = None
        self._is_dirty = True

    @property
    def value(self) -> Any:
        """Get computed value."""
        if self._materialized_key:
            # Materialized: use DeltaKVStore caching
            return self._store._kv.get(self._materialized_key)

        # Virtual: recompute every time (no caching for virtual SmartComputed)
        vals = [src.value for src in self._sources]
        if len(vals) == 1:
            return self._fused_fn(vals[0])
        return self._fused_fn(*vals)

    def set(self, new_value: Any) -> None:
        """Computed values are read-only."""
        pass  # No-op for API compatibility

    def subscribe(
        self, callback: Callable[[Any], None], call_immediately: bool = False
    ) -> Callable[[], None]:
        """Subscription forces materialization."""
        self._is_subscribed = True
        materialized = self._materialize_full_chain()

        # The materialized observable will handle calling immediately
        return materialized.subscribe(callback, call_immediately=call_immediately)

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
            vals = []
            for src in sources:
                try:
                    vals.append(src.value)
                except ConditionNeverMet:
                    # For conditionals that have never met, treat as False/None
                    vals.append(False)
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
            elif hasattr(src, "_is_tracked") and hasattr(src, "_track_in_store"):
                # Source is observable - track it if not already
                if not src._is_tracked:
                    src._track_in_store()
                dep_keys.append(src._key)
            elif hasattr(src, "_key"):
                # Source has a key but is not tracked (e.g., ConditionalObservable)
                dep_keys.append(src._key)
            else:
                # Source doesn't have tracking - this shouldn't happen
                raise ValueError(f"Cannot track dependency: {src}")

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
        """Apply transformation with function composition optimization."""
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
        """Combine with other observables into tuple stream."""
        all_obs = (self,) + others
        store = self._store
        return store._get_or_create_stream(all_obs)

    def requiring(self, condition: "Observable") -> "ConditionalObservable":
        """Filter by condition."""
        self._register_dependent()

        if callable(condition) and not hasattr(condition, "_register_dependent"):
            materialized = self._materialize_as_node()
            return ConditionalObservable(self._store, materialized, condition)

        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()

        materialized = self._materialize_as_node()
        return ConditionalObservable(self._store, materialized, condition)

    def either(self, other: "Observable") -> "SmartComputed":
        """Logical OR with other observable."""
        self._register_dependent()
        if hasattr(other, "_register_dependent"):
            other._register_dependent()

        materialized = self._materialize_as_node()
        return SmartComputed(
            self._store, [materialized, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self) -> "SmartComputed":
        """Logical NOT with function fusion when possible."""
        self._register_dependent()

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
# ConditionNotMet - Exception for unmet conditions
# ============================================================================


class ConditionNotMet(Exception):
    """Raised when accessing a conditional observable with unmet conditions."""

    pass


class ConditionNeverMet(Exception):
    """Raised when accessing a conditional observable that has never had conditions met."""

    pass


# ============================================================================
# ConditionalObservable - Only emits when conditions are met
# ============================================================================


class ConditionalObservable:
    """
    Conditional observable with proper pullback semantics.

    States:
    - NEVER_MET: Conditions have never been satisfied (no pullback exists yet)
    - MET: Conditions currently satisfied (pullback exists, has value)
    - UNMET: Conditions were met but are now unsatisfied (pullback exists, no current value)
    """

    __slots__ = (
        "_store",
        "_key",
        "_source",
        "_condition",
        "_has_ever_been_met",
        "_condition_currently_met",
        "_cached_value",
        "_subscribers",
        "_is_tracked",
    )

    def __init__(self, store, source, condition):
        self._store = store
        self._key = store._gen_key("conditional")
        self._source = source
        self._condition = condition
        self._has_ever_been_met = False
        self._condition_currently_met = False
        self._cached_value = None
        self._subscribers = []
        self._is_tracked = (
            False  # ConditionalObservables are not tracked in DeltaKVStore
        )

        # Start in NEVER_MET state - pullback doesn't exist until first change
        # Don't evaluate conditions initially

        # Subscribe to changes
        self._setup_subscriptions()

    def _register_dependent(self):
        """Called when another computed depends on this conditional."""
        # When a computed depends on us, we need to materialize in DeltaKVStore
        self._track_in_store()

    def _track_in_store(self):
        """Materialize this conditional in DeltaKVStore for dependency tracking."""
        if self._is_tracked:
            return

        self._is_tracked = True
        # Set initial value in store - None since conditions never met
        self._store._kv.set(self._key, None)

    def _setup_subscriptions(self):
        """Set up subscriptions to source and condition changes."""
        # Subscribe to source changes
        self._source.subscribe(self._on_source_change, call_immediately=False)

        # Subscribe to condition changes if condition is an observable
        # But only if it's not a computed that depends on the source
        if (
            hasattr(self._condition, "subscribe")
            and not self._is_condition_dependent_on_source()
        ):
            self._condition.subscribe(self._on_condition_change, call_immediately=False)

    def _is_condition_dependent_on_source(self):
        """Check if the condition depends on the source (to avoid double subscriptions)."""
        if hasattr(self._condition, "_sources"):
            return self._source in self._condition._sources
        return False

    def _on_source_change(self, new_value):
        """Handle source value changes."""
        # Evaluate condition with the NEW value
        condition_met = self._evaluate_condition_with_value(new_value)

        if condition_met:
            if not self._has_ever_been_met:
                # First time conditions are met - transition to MET state (create pullback)
                self._has_ever_been_met = True
                self._condition_currently_met = True
                self._cached_value = new_value
                self._notify_subscribers(new_value)
            elif not self._condition_currently_met:
                # Conditions just became met again - transition from UNMET to MET
                self._condition_currently_met = True
                self._cached_value = new_value
                self._notify_subscribers(new_value)
            elif new_value != self._cached_value:
                # Conditions still met, but value changed
                self._cached_value = new_value
                self._notify_subscribers(new_value)
        else:
            # Conditions not met - transition to UNMET state
            if self._condition_currently_met:
                self._condition_currently_met = False
            # Note: We don't emit anything when conditions become unmet

    def _on_condition_change(self, new_condition_value):
        """Handle condition changes."""
        # Evaluate condition with current source value
        condition_met = self._evaluate_condition()

        if condition_met:
            if not self._has_ever_been_met:
                # First time conditions are met - create pullback
                self._has_ever_been_met = True
                self._condition_currently_met = True
                self._cached_value = self._source.value
                self._notify_subscribers(self._source.value)
            elif not self._condition_currently_met:
                # Conditions just became met again - transition from UNMET to MET
                self._condition_currently_met = True
                self._cached_value = self._source.value
                self._notify_subscribers(self._source.value)
        else:
            # Conditions not met - transition to UNMET state
            if self._condition_currently_met:
                self._condition_currently_met = False
            # Note: We don't emit anything when conditions become unmet

    def _evaluate_condition(self):
        """Evaluate the current condition with current source value."""
        return self._evaluate_condition_with_value(self._source.value)

    def _evaluate_condition_with_value(self, value):
        """Evaluate the condition with a specific value."""
        if callable(self._condition):
            return bool(self._condition(value))
        elif hasattr(self._condition, "value"):
            return bool(self._condition.value)
        else:
            return bool(self._condition)

    def _notify_subscribers(self, value):
        """Notify all subscribers with the value."""
        # Update DeltaKVStore if we're tracked
        if self._is_tracked:
            self._store._kv.set(self._key, value)

        # Notify direct subscribers
        for subscriber in self._subscribers:
            subscriber(value)

    @property
    def value(self):
        """Get current value based on state."""
        if self._is_tracked:
            stored_value = self._store._kv.get(self._key)
            if stored_value is None and not self._has_ever_been_met:
                raise ConditionNeverMet(
                    "Conditional observable has never had conditions met"
                )
            return stored_value

        if not self._has_ever_been_met:
            raise ConditionNeverMet(
                "Conditional observable has never had conditions met"
            )
        elif self._condition_currently_met:
            return self._source.value
        else:
            # UNMET state - return cached last valid value
            return self._cached_value

    def subscribe(self, callback, call_immediately: bool = False):
        """Subscribe to conditional emissions."""
        self._subscribers.append(callback)

        # Call immediately ONLY if conditions are currently met AND requested
        if call_immediately and self._condition_currently_met:
            callback(self._cached_value)

    def unsubscribe(self, callback):
        """Unsubscribe from conditional emissions."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def then(self, transform):
        """Apply transformation to conditional observable."""
        # Register as dependent
        self._register_dependent()
        return SmartComputed(self._store, [self], lambda val: transform(val))

    def requiring(self, condition):
        """Chain conditional with another condition."""
        # For chaining conditionals, create a combined condition
        if hasattr(condition, "value"):
            # condition is an observable - combine with AND
            def combined(src_val=None):
                return self._evaluate_condition() and bool(condition.value)

            return ConditionalObservable(self._store, self._source, combined)
        else:
            # condition is a callable
            def combined(src_val):
                return self._evaluate_condition() and condition(src_val)

            return ConditionalObservable(self._store, self._source, combined)

    __and__ = requiring
    __rshift__ = then

    def set(self, value):
        """Conditional observables are read-only."""
        raise AttributeError("Conditional observables are read-only")


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
                # Call immediately unless autorun=False, or if conditional not satisfied
                call_now = autorun
                if isinstance(actual_dep, ConditionalObservable):
                    try:
                        actual_dep.value  # Check if conditional has value
                    except ConditionNeverMet:
                        call_now = False  # Don't call if conditional never met
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
