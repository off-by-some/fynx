"""
Fusion-optimized computed observables.

This module implements lazy-evaluated derived values with aggressive fusion
optimization to minimize intermediate materialization nodes.

Key Concepts:
    - Virtual: computed on-demand, no storage overhead
    - Materialized: stored in DeltaKVStore with automatic invalidation
    - Fusion: chaining transforms without creating intermediate nodes

Materialization Strategy:
    - Single dependent: stay virtual, use direct propagation
    - Multiple dependents (fan-out): materialize to enable store-based propagation
    - Subscriptions: force materialization for reactivity

Fusion Optimization:
    When chaining transforms (obs >> f1 >> f2 >> f3), fusion prevents creating
    three separate computed nodes. Instead, it composes the functions into a
    single computation over the original sources, reducing memory and recomputation
    overhead in linear chains.
"""

from typing import TYPE_CHECKING, Any, Callable, List

from .base import (
    BaseObservable,
    Materializable,
    OperatorMixin,
    Subscribable,
    Trackable,
    TupleOperable,
    _get_dependency_keys,
    _try_register_dependent,
)

if TYPE_CHECKING:
    from .observable import Observable
    from .store import Store


def _flatten_nested_values(*vals):
    """Flatten nested tuples - optimized."""
    if not vals:
        return []

    # Fast path: no tuples
    has_tuple = False
    for v in vals:
        if isinstance(v, tuple):
            has_tuple = True
            break

    if not has_tuple:
        return list(vals)

    # Slow path: recursive flatten
    result = []
    for v in vals:
        if isinstance(v, tuple):
            result.extend(_flatten_nested_values(*v))
        else:
            result.append(v)
    return result


class NullEvent:
    """
    Sentinel value representing an inactive conditional observable.

    Singleton used to distinguish between "never met" and "currently unmet but
    was active before" states. When a conditional's condition is not satisfied,
    it emits NULL_EVENT to downstream computeds.

    Implementation uses singleton pattern for memory efficiency.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "NullEvent"

    def __bool__(self):
        return False


NULL_EVENT = NullEvent()


class ConditionNotMet(Exception):
    """Raised when accessing a conditional observable with unmet conditions."""

    pass


class ConditionNeverMet(Exception):
    """Raised when accessing inactive conditional."""

    pass


# ============================================================================
# ComputedBase - Minimal Base
# ============================================================================


class ComputedBase(BaseObservable, Materializable, OperatorMixin):
    """
    Base class for computed observables with materialization support.

    Handles the virtual-to-materialized transition based on dependency count.
    Subclasses implement _compute_virtual_value() to define computation.

    State Management:
        - _sources: list of observable dependencies
        - _is_subscribed: whether any code is subscribed to changes
        - _materialized_key: DeltaKVStore key if materialized, None if virtual
        - _direct_dependents: list for O(1) direct propagation (virtual mode)

    Value Access Semantics:
        - Virtual: recompute on every access (no caching)
        - Materialized: value stored in DeltaKVStore with invalidation tracking
    """

    __slots__ = (
        "_store",
        "_sources",
        "_is_subscribed",
        "_direct_dependents",
    )

    def __init__(self, store: "Store", sources: List["Observable"]):
        BaseObservable.__init__(self, store)
        Materializable.__init__(self)
        self._sources = sources
        self._is_subscribed = False
        self._direct_dependents = []  # Initialize direct dependents

    @property
    def value(self) -> Any:
        """
        Get computed value, materialized or virtual.

        Materialized values are cached in DeltaKVStore with automatic
        invalidation. Virtual values use _cached_value if available.

        Raises ConditionNeverMet if dependency is an inactive conditional.
        """
        if self.is_materialized:
            val = self._store._kv.get(self._materialized_key)
        else:
            # Use cached value if available (for ComputedObservable optimization)
            if hasattr(self, "_cached_value") and self._cached_value is not None:
                val = self._cached_value
            else:
                val = self._compute_virtual_value()

        # NULL_EVENT indicates dependency on inactive conditionals
        if val is NULL_EVENT:
            raise ConditionNeverMet("Computed value depends on inactive conditional")
        return val

    @property
    def is_materialized(self) -> bool:
        """Check if materialized."""
        return self._materialized_key is not None

    def _compute_virtual_value(self) -> Any:
        """Override in subclasses."""
        raise NotImplementedError

    def set(self, new_value: Any) -> None:
        """Computed values are read-only."""
        pass

    def subscribe(
        self, callback: Callable[[Any], None], call_immediately: bool = False
    ) -> Callable[[], None]:
        """Subscribe - forces materialization."""
        self._is_subscribed = True
        materialized = self._materialize_as_node()

        # Filter out None/NullEvent values
        def filtered_callback(value):
            if value is not None and value is not NULL_EVENT:
                callback(value)

        return materialized.subscribe(
            filtered_callback, call_immediately=call_immediately
        )

    def _register_dependent(self, dependent=None):
        """
        Called when another computed observable depends on this one.

        Implements fan-out detection: if 2+ dependents or any dependent is
        materialized, materialize this node to enable efficient change propagation.

        Optimization: For virtual computeds, track direct dependents in a list
        to enable O(1) direct propagation without DeltaKVStore overhead.
        """
        Materializable._register_dependent(self)

        # Add to direct dependents if not materialized (for fast direct propagation)
        if (
            dependent is not None
            and not dependent.is_materialized
            and hasattr(dependent, "_on_source_changed")
        ):
            self._direct_dependents.append(dependent)

        # Materialize if:
        # 1. Fan-out (2+ dependents) OR
        # 2. Has a materialized dependent (need store tracking for reactivity)
        should_materialize = self._dependents_count >= 2 or (
            dependent is not None
            and hasattr(dependent, "is_materialized")
            and dependent.is_materialized
        )

        if should_materialize and not self.is_materialized:
            self._materialize_as_node()

    def _make_computed(self, sources: list, fn: Callable) -> "ComputedObservable":
        """Create computed - override for fusion."""
        return ComputedObservable(self._store, sources, fn)

    def _make_stream(self, sources: list) -> "StreamMerge":
        """Create stream observable."""
        return StreamMerge(self._store, sources)

    def _make_conditional(
        self, source: "BaseObservable", condition
    ) -> "ConditionalObservable":
        """Create conditional observable."""
        self._register_dependent()
        materialized = self._materialize_as_node()
        return ConditionalObservable(self._store, materialized, condition)

    def _materialize_as_node(self):
        """
        Materialize this computed observable as a node in DeltaKVStore.

        Creates a DeltaKVStore entry with:
            - Unique key generated by store
            - Computation function wrapping _compute_virtual_value()
            - Dependency keys extracted from sources

        Returns an Observable wrapping the materialized store entry.
        The Observable is marked as tracked and registered in the store.
        """
        if self.is_materialized:
            obs = self._store._observables.get(self._materialized_key)
            if obs:
                return obs

        self._materialized_key = self._store._gen_key("node")

        # Delegate to subclass implementation
        def compute():
            try:
                return self._compute_virtual_value()
            except ConditionNeverMet:
                # During reactive computation, return NULL_EVENT for inactive conditionals
                return NULL_EVENT

        dep_keys = _get_dependency_keys(self._sources, self._store)
        self._store._kv.computed(self._materialized_key, compute, deps=dep_keys)

        from .observable import Observable

        obs = Observable(self._store, self._materialized_key)
        obs._is_tracked = True
        self._store._observables[self._materialized_key] = obs
        return obs

    def __repr__(self) -> str:
        state = "materialized" if self.is_materialized else "virtual"
        return f"{self.__class__.__name__}({state}, deps={self._dependents_count})"


# ============================================================================
# StreamMerge - Ultra-Fast Stream Operations
# ============================================================================


class StreamMerge(ComputedBase, Trackable, TupleOperable, Subscribable):
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
        "_callbacks",  # set[Callable]: Registered callbacks
        "_is_subscribed",  # bool: Track if we've set up source subscriptions
        "_is_tracked",  # bool: Whether this StreamMerge is tracked in DeltaKVStore
    )

    def __init__(self, store: "Store", sources: List["Observable"]):
        ComputedBase.__init__(self, store, sources)
        Trackable.__init__(self)
        TupleOperable.__init__(self)
        Subscribable.__init__(self)
        self._sources = tuple(sources)
        self._key = store._gen_key("stream")  # For dependency tracking

        # Initialize cached values - pay upfront cost once
        self._cached_values = [src.value for src in self._sources]
        self._cached_tuple = tuple(self._cached_values)

        # Lazy subscription state
        self._callbacks = set()
        self._is_subscribed = False

    @property
    def value(self) -> tuple:
        """Return current combined values as tuple."""
        if self._is_tracked:
            return self._store._kv.get(self._key)
        return self._cached_tuple

    def set(self, new_value: Any) -> None:
        """Stream merges are read-only."""
        pass  # No-op for API compatibility

    def _track_in_store(self):
        """Track this StreamMerge in DeltaKVStore for change propagation."""
        if self._is_tracked:
            return

        self._is_tracked = True

        # Get dependency keys for all sources
        dep_keys = _get_dependency_keys(list(self._sources), self._store)

        # Register computed function that returns the tuple
        def compute_tuple():
            return tuple(src.value for src in self._sources)

        self._store._kv.computed(self._key, compute_tuple, deps=dep_keys)

        # Set up subscription to update our cached values when the store value changes
        def on_tuple_change(delta):
            if delta.key == self._key and delta.new_value is not None:
                self._cached_tuple = delta.new_value
                self._cached_values = list(delta.new_value)
                self._notify_callbacks()

        self._store._kv.subscribe(self._key, on_tuple_change)

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
            self._add_subscription(unsub)

    def _teardown_source_subscriptions(self):
        """Clean up source subscriptions when no longer needed."""
        self._clear_subscriptions()
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
    # Custom Operators for StreamMerge
    # ========================================================================

    def then(self, transform: Callable) -> "ComputedObservable":
        """
        Transform the merged stream.

        Creates a ComputedObservable that depends on all sources.
        The transform receives the tuple of values.
        """
        store = self._sources[0]._store

        # Register all sources as having a dependent
        for src in self._sources:
            if hasattr(src, "_register_dependent"):
                src._register_dependent()

        def compute_func(*vals):
            # Flatten nested tuples from nested StreamMerges
            def flatten_values(v):
                if isinstance(v, tuple):
                    result = []
                    for item in v:
                        result.extend(flatten_values(item))
                    return result
                else:
                    return [v]

            flattened = []
            for v in vals:
                flattened.extend(flatten_values(v))
            return transform(*flattened)

        return ComputedObservable(
            store,
            list(self._sources),
            compute_func,
        )

    def alongside(self, *others) -> "StreamMerge":
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

    def requiring(self, condition) -> "ConditionalObservable":
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
        condition_computed = ComputedObservable(
            store, list(self._sources) + [condition], condition_fn
        )

        # Create conditional observable that filters the tuple
        return ConditionalObservable(store, self, condition_computed)

    def __repr__(self) -> str:
        state = "subscribed" if self._is_subscribed else "unsubscribed"
        return (
            f"StreamMerge({len(self._sources)} sources, "
            f"{len(self._callbacks)} callbacks, {state})"
        )


# ============================================================================
# ComputedObservable - THE CORE - Aggressive Fusion
# ============================================================================


class ComputedObservable(ComputedBase):
    """
    Universal computed observable with fusion optimization.

    Fusion combines adjacent transform operations into single computations,
    avoiding intermediate materialization nodes in linear dependency chains.

    State:
        - _fused_fn: the computation function to execute
        - _direct_dependents: list for O(1) direct propagation (virtual mode)
        - _cached_value: cached result to avoid re-executing _compute_virtual_value()

    Materialization Rules:
        1. Single dependent: stay virtual with direct propagation
        2. Multiple dependents (fan-out): materialize for store-based propagation
        3. Subscriptions: force materialization for reactivity
        4. Fusion applies when chaining transforms (obs >> f1 >> f2)
    """

    __slots__ = ("_fused_fn", "_cached_value")

    def __init__(
        self,
        store: "Store",
        sources: List["Observable"],
        compute_fn: Callable,
        _skip_source_registration: bool = False,
    ):
        super().__init__(store, sources)
        self._fused_fn = compute_fn  # Just store the function directly
        self._cached_value = None  # Cache to avoid recomputation

        if not _skip_source_registration:
            for source in sources:
                if hasattr(source, "_register_dependent"):
                    # Pass self only if source can handle dependent tracking (computed observables)
                    # Regular observables just need to know they have dependents
                    if hasattr(source, "_direct_dependents"):  # ComputedBase has this
                        source._register_dependent(self)
                    else:
                        source._register_dependent()

        # Don't populate cache during init - use lazy evaluation on first access
        # This matches the expected behavior in tests and avoids unnecessary computation

    def _compute_virtual_value(self) -> Any:
        """
        Compute value by applying the fused function to source values.

        Executes computation inline without store overhead. For virtual computeds,
        this is called on every .value access.

        Handles both single and multi-argument functions correctly.
        """
        vals = [src.value for src in self._sources]
        return self._fused_fn(*vals) if len(vals) > 1 else self._fused_fn(vals[0])

    # ========================================================================
    # OperatorMixin Implementation - FUSION MAGIC HERE
    # ========================================================================

    def _make_computed(self, sources: list, fn: Callable) -> "ComputedObservable":
        """
        Create computed - WITH FUSION OPTIMIZATION.

        Key: If sources=[self] and we can fuse, extend chain instead of creating new node.
        """
        # FUSION PATH: Simple transform on self
        if len(sources) == 1 and sources[0] is self and self._should_fuse():
            return self._create_fused(fn)

        # NORMAL PATH: New computed
        return ComputedObservable(self._store, sources, fn)

    def _make_stream(self, sources: list) -> "StreamObservable":
        """Create stream - uses cached instances."""
        return self._store._get_or_create_stream(tuple(sources))

    def _make_conditional(
        self, source: "BaseObservable", condition
    ) -> "ConditionalObservable":
        """Create conditional - forces materialization of source."""
        self._register_dependent()
        materialized = self._materialize_as_node()
        return ConditionalObservable(self._store, materialized, condition)

    # ========================================================================
    # FUSION CORE - The Secret Sauce
    # ========================================================================

    def _should_fuse(self) -> bool:
        """
        Check if we should fuse instead of materializing.

        Fusion is possible when:
            - Not yet materialized
            - No fan-out detected (≤1 dependent)

        Fusion preserves single-chain benefits while maintaining direct propagation.
        """
        # Fuse if: not materialized AND no fan-out (≤1 dependent)
        return not self.is_materialized and self._dependents_count <= 1

    def _materialize_as_node(self):
        """
        Materialize with cleanup of direct propagation chains.

        When transitioning to materialized state, must clean up direct_dependents
        lists to prevent stale references. Direct propagation only works for
        virtual computeds; materialized ones use store-based propagation.
        """
        # Remove from source's direct dependents when materializing
        for source in self._sources:
            if hasattr(source, "_direct_dependents"):
                try:
                    source._direct_dependents.remove(self)
                except ValueError:
                    pass

        # Now materialize using parent implementation
        return super()._materialize_as_node()

    def _create_fused(self, new_transform: Callable) -> "ComputedObservable":
        """
        Create fused computed by composing functions.

        Instead of creating two ComputedObservables (old → new), creates one
        with a composed function (new_transform ∘ old_fn). This maintains
        virtual state while collapsing the chain.

        Critical: reuse _sources from parent to maintain single dependency path,
        and skip source registration to avoid double-counting dependents.
        """
        old_fn = self._fused_fn

        def fused_fn(*vals):
            intermediate = old_fn(*vals)  # old_fn handles unpacking correctly
            return new_transform(intermediate)

        # Store original sources to check if tracking changed
        original_sources = self._sources

        # Check if any source is tracked - if so, materialize this before fusion
        has_tracked_source = any(
            hasattr(src, "_is_tracked") and src._is_tracked for src in original_sources
        )

        if has_tracked_source and not self.is_materialized:
            # Source is tracked but we're virtual - need materialization for proper invalidation
            self._materialize_as_node()
            # Now create non-fused computed that depends on materialized self
            return ComputedObservable(self._store, [self], new_transform)

        return ComputedObservable(
            self._store,
            self._sources,  # CRITICAL: Reuse parent sources
            fused_fn,  # Single composed function
            _skip_source_registration=True,  # CRITICAL: No double-registration
        )

    def _on_source_changed(self):
        """
        Called directly by source during direct propagation (no store overhead).

        This is the fast path for virtual computeds: when a source changes,
        it notifies direct dependents via this method, which then invalidates
        the cache and propagates to its own dependents.

        This avoids DeltaKVStore overhead for linear chains staying in virtual mode.
        """
        # Invalidate cache so next value access recomputes
        self._cached_value = None

        # Propagate to our direct dependents
        for dependent in self._direct_dependents:
            if hasattr(dependent, "_on_source_changed"):
                dependent._on_source_changed()

    def negate(self) -> "ComputedObservable":
        """Logical NOT with fusion."""
        if self._should_fuse():
            return self._create_fused(lambda x: not bool(x))

        # Can't fuse - materialize and create new computed
        self._register_dependent()
        return super().negate()

    def __repr__(self) -> str:
        state = "materialized" if self.is_materialized else "virtual"
        return f"ComputedObservable({state}, deps={self._dependents_count}, sources={len(self._sources)})"


# ============================================================================
# ConditionalObservable - Built on ComputedObservable
# ============================================================================


class ConditionalObservable(ComputedObservable):
    """
    Conditional observable emitting values only when condition is satisfied.

    Wraps ComputedObservable with conditional logic, maintaining caching benefits.
    """

    __slots__ = (
        "_last_emitted_value",
        "_has_ever_been_active",
    )

    def __init__(self, store, source, condition):
        # If condition is already an observable, include it as a source
        if hasattr(condition, "_register_dependent"):
            # Condition is an observable - create conditional compute function
            compute_fn = lambda src_val, cond_val: src_val if cond_val else NULL_EVENT
            super().__init__(
                store, [source, condition], compute_fn, _skip_source_registration=True
            )
        else:
            # Callable condition - evaluate on source value
            compute_fn = lambda src_val: src_val if condition(src_val) else NULL_EVENT
            super().__init__(
                store, [source], compute_fn, _skip_source_registration=True
            )

        # Conditional state
        self._last_emitted_value = None
        self._has_ever_been_active = False

    def _ensure_registered(self):
        """Lazily register dependencies when first accessed."""
        if not hasattr(self, "_dependencies_registered"):
            for source in self._sources:
                if hasattr(source, "_register_dependent"):
                    if hasattr(source, "_direct_dependents"):
                        source._register_dependent(self)
                    else:
                        source._register_dependent()
            self._dependencies_registered = True

    @property
    def value(self):
        """Get value - handles NULL_EVENT with proper caching."""
        # Register dependencies lazily on first access
        self._ensure_registered()

        # Compute value using parent's implementation
        if self.is_materialized:
            val = self._store._kv.get(self._materialized_key)
        else:
            if hasattr(self, "_cached_value") and self._cached_value is not None:
                val = self._cached_value
            else:
                val = self._compute_virtual_value()

        # Handle NULL_EVENT - condition not met
        if val is NULL_EVENT:
            if not self._has_ever_been_active:
                raise ConditionNeverMet("Conditional never active")
            return self._last_emitted_value

        # Condition met - update cache and return
        self._has_ever_been_active = True
        self._last_emitted_value = val
        return val

    def then(self, transform):
        """Transform - creates a new ComputedObservable."""
        _try_register_dependent(self)
        return ComputedObservable(self._store, [self], lambda val: transform(val))

    def requiring(self, condition):
        """Chain conditional."""
        _try_register_dependent(self)
        return ConditionalObservable(self._store, self, condition)

    __and__ = requiring
    __rshift__ = then

    def subscribe(self, callback, call_immediately=False):
        """Subscribe - lazily registers dependencies."""
        self._ensure_registered()
        return super().subscribe(callback, call_immediately)

    def set(self, value):
        """Read-only."""
        raise AttributeError("Conditional observables are read-only")
