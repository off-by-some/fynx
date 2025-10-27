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

from .base import BaseObservable, Materializable, OperatorMixin, _try_register_dependent

if TYPE_CHECKING:
    from .observable import Observable, StreamObservable
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
        return materialized.subscribe(callback, call_immediately=call_immediately)

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

    def _make_stream(self, sources: list) -> "StreamObservable":
        """Create stream observable."""
        from .observable import StreamObservable

        return StreamObservable(self._store, sources)

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
        from .observable import Observable, _get_dependency_keys

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

        obs = Observable(self._store, self._materialized_key)
        obs._is_tracked = True
        self._store._observables[self._materialized_key] = obs
        return obs

    def __repr__(self) -> str:
        state = "materialized" if self.is_materialized else "virtual"
        return f"{self.__class__.__name__}({state}, deps={self._dependents_count})"


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
                    # This will handle adding to direct_dependents
                    source._register_dependent(self)

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

    Extends ComputedObservable to add conditional emission logic:
        - Activates when condition becomes true (or is initially true)
        - Returns cached value when condition becomes false
        - Raises ConditionNeverMet when accessed if never been active

    State Management:
        - _is_active: whether condition is currently satisfied
        - _has_ever_been_active: whether condition was ever true
        - _cached_value: last value emitted before condition became false

    Inherits fusion optimization from ComputedObservable.
    """

    __slots__ = (
        "_is_active",
        "_last_emitted_value",
        "_has_ever_been_active",
        "_condition_fn",
        "_last_source_val",
        "_last_computed_val",
    )

    def __init__(self, store, source, condition):
        # Optimize simple callable conditions by evaluating inline
        if callable(condition):
            # Store condition function directly - evaluate on-demand instead of creating ComputedObservable
            self._condition_fn = condition
            # Create a computed that evaluates the condition on source value
            compute_fn = lambda src_val: src_val if condition(src_val) else NULL_EVENT
            # Initialize with just the source
            super().__init__(store, [source], compute_fn)
        else:
            # Condition is already an observable
            condition_obs = condition
            # Conditional compute: source if condition else NULL_EVENT
            compute_fn = lambda src_val, cond_val: src_val if cond_val else NULL_EVENT
            # Initialize as ComputedObservable
            super().__init__(store, [source, condition_obs], compute_fn)
            self._condition_fn = None

        # Conditional state
        self._is_active = False
        self._last_emitted_value = None
        self._has_ever_been_active = False
        # Cache for fast path when source hasn't changed
        self._last_source_val = None
        self._last_computed_val = None

        # Initialize state - handle case where condition has never been met
        try:
            val = super().value
            if val is not NULL_EVENT:
                self._is_active = True
                self._has_ever_been_active = True
                self._last_emitted_value = val
        except ConditionNeverMet:
            # Condition has never been met - this is normal during initialization
            pass

    @property
    def value(self):
        """Get value - handles NULL_EVENT with proper caching."""
        # Fast path: Check if source value changed using identity
        if self._last_source_val is not None and len(self._sources) == 1:
            current_source_val = self._sources[0].value
            # If source value identity hasn't changed, return cached result
            if current_source_val is self._last_source_val:
                cached = self._last_computed_val
                if cached is NULL_EVENT:
                    if not self._has_ever_been_active:
                        raise ConditionNeverMet("Conditional never active")
                    return self._last_emitted_value
                return cached

        # Compute new value
        if self.is_materialized:
            val = self._store._kv.get(self._materialized_key)
        else:
            if self._cached_value is not None:
                val = self._cached_value
            else:
                val = self._compute_virtual_value()

        # Cache the source value and computed result for next time
        if len(self._sources) == 1:
            self._last_source_val = self._sources[0].value
            self._last_computed_val = val

        if val is NULL_EVENT:
            if not self._has_ever_been_active:
                raise ConditionNeverMet("Conditional never active")
            return self._last_emitted_value

        self._last_emitted_value = val
        return val

    def _on_source_changed(self):
        """Clear cache when source changes."""
        super()._on_source_changed()
        # Invalidate our cache
        self._last_source_val = None
        self._last_computed_val = None
        self._last_emitted_value = None

    def subscribe(self, callback, call_immediately: bool = False):
        """Subscribe with conditional emission logic."""
        self._is_subscribed = True
        materialized = self._materialize_as_node()

        # Wrap callback with conditional logic
        def conditional_callback(val):
            if val is not NULL_EVENT:
                # Condition met
                was_active = self._is_active
                self._is_active = True

                if not was_active:
                    # Transition to active
                    self._has_ever_been_active = True
                    self._last_emitted_value = val
                    callback(val)
                elif val != self._last_emitted_value:
                    # Value changed while active
                    self._last_emitted_value = val
                    callback(val)
                # Else: no change, no emission
            else:
                # Condition not met - update state but don't emit
                self._is_active = False

        # Subscribe to the materialized observable with our conditional callback
        # We need to bypass the NULL_EVENT filtering for state management
        def delta_callback(delta):
            if delta.key == materialized._key:
                conditional_callback(delta.new_value)

        return materialized._store._kv.subscribe(materialized._key, delta_callback)

    def then(self, transform):
        """Transform - registers dependency."""
        _try_register_dependent(self)
        return ComputedObservable(self._store, [self], lambda val: transform(val))

    def requiring(self, condition):
        """Chain conditional."""
        _try_register_dependent(self)
        return ConditionalObservable(self._store, self, condition)

    __and__ = requiring
    __rshift__ = then

    def set(self, value):
        """Read-only."""
        raise AttributeError("Conditional observables are read-only")
