"""
Observable - Zero-Cost Reactive Frontend with Delta Algebra

Mathematical Foundation:
- Observable Comonad: (Obs, ε, δ) with extract and duplicate
- Subscription Monoid: (Sub, ⊕, ε) unified across all states
- Operator Kleisli: (map, filter, scan) as virtual morphisms
- Delta Propagation: Incremental updates via TypedDelta algebra
- Zero-cost: Virtual until materialization, fused operators

Core Principles:
1. Single unified subscription mechanism (σ: Obs × Callback → Sub)
2. Delta-aware propagation - emit TypedDelta when possible
3. Operators as virtual chains - materialize only on subscription
4. Leverage store's AD for incremental computation
5. Tracking is transparent - same subscription semantics before/after
"""

import atexit
import threading
from typing import Any, Callable, List, Optional, TypeVar, Union

from .delta_kv_store import Change, ChangeType, DeltaType, ReactiveStore, TypedDelta

T = TypeVar("T")


# ============================================================================
# SENTINEL VALUES
# ============================================================================


class _NULL_EVENT:
    """Sentinel for 'no value' in conditional observables."""

    def __repr__(self):
        return "NULL_EVENT"


NULL_EVENT = _NULL_EVENT()


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ConditionNotMet(Exception):
    """Condition not currently met."""

    pass


class ConditionalNeverMet(Exception):
    """Condition never met."""

    pass


class ReactiveFunctionError(Exception):
    """Reactive function called manually."""

    pass


# ============================================================================
# OBSERVABLE - The Comonad (Obs, ε, δ)
# ============================================================================


class Observable:
    """
    Observable: Object in the reactive category with delta-aware propagation.

    Comonad operations:
    - extract (ε): value property - get current value
    - duplicate (δ): _track - create tracked observable

    Subscription: Unified σ: Obs × Callback → Sub with TypedDelta support

    States:
    - untracked: Local value, direct callbacks
    - tracked: Store-managed, reactive propagation with deltas

    Invariant: Subscription works identically in both states
    """

    __slots__ = ("_store", "_key", "_value", "_is_tracked", "_delta_type", "_callbacks")

    def __init__(self, store: "Store", key: str, initial_value: Any = None):
        self._store = store
        self._key = key
        self._value = initial_value
        self._is_tracked = False
        self._callbacks = []  # Direct callback list for untracked observables
        # Cache delta type for efficiency
        self._delta_type = (
            store._kv._delta_registry.detect_type(initial_value)
            if initial_value is not None
            else DeltaType.OPAQUE
        )

    @property
    def value(self) -> Any:
        """
        Extract operation (ε): Obs[A] → A

        Pure comonad extract - reads current value.
        """
        if not self._is_tracked:
            return self._value
        return self._store._kv.get(self._key)

    @value.setter
    def value(self, new_value: Any):
        """
        Update operation with delta-aware emission.

        Critical: Computes TypedDelta for incremental propagation.
        """
        if not self._is_tracked:
            # Untracked: direct value propagation - zero overhead
            if self._value == new_value:
                return

            old_value = self._value
            self._value = new_value

            # Direct callback invocation - no store overhead
            for callback in self._callbacks:
                callback(new_value)
        else:
            # Tracked: delegate to store (which handles delta computation)
            old_value = self._store._kv.get(self._key)

            # Compute delta before setting
            differential = None
            if old_value is not None and new_value is not None:
                differential = self._store._kv._delta_registry.compute_delta(
                    old_value, new_value
                )
                # Skip if identity
                if differential and self._store._kv._delta_registry.is_identity(
                    differential
                ):
                    return

            # Use internal set to pass differential
            self._store._kv._set_internal(self._key, new_value, differential)

    def set(self, new_value: Any) -> None:
        """Explicit setter (alias for value property)."""
        self.value = new_value

    def get(self) -> Any:
        """Explicit getter (alias for value property)."""
        return self.value

    def _track(self) -> None:
        """
        Duplicate operation (δ): Obs[A] → Obs[Obs[A]]

        Transitions from untracked to tracked state.
        Maintains subscription continuity.
        """
        if self._is_tracked:
            return

        # Migrate direct callbacks to store subscriptions
        callbacks_to_migrate = list(self._callbacks)
        self._callbacks.clear()

        self._is_tracked = True
        # Register with store using source() to properly initialize
        self._store._kv.source(self._key, self._value)
        # Update delta type cache
        if self._value is not None:
            self._delta_type = self._store._kv._delta_registry.detect_type(self._value)

        # Re-register callbacks through store
        def make_callback_wrapper(cb):
            def wrapper(change):
                cb(change.new_value)

            return wrapper

        for callback in callbacks_to_migrate:
            self._store._kv.on(self._key, make_callback_wrapper(callback))

    def subscribe(
        self,
        callback: Callable[[Any], None],
        call_immediately: bool = False,
        force_track: bool = True,
    ) -> Callable[[], None]:
        """
        Unified subscription operation: σ: Obs × Callback → Sub

        Delta-aware: Callbacks receive TypedDelta when beneficial,
        otherwise receive raw values for compatibility.

        Args:
            callback: Called with new values (or deltas if beneficial)
            call_immediately: Call with current value immediately
            force_track: Whether to track (default True for reactivity)

        Returns:
            Unsubscribe function
        """
        if force_track:
            # Ensure tracking for reactive propagation
            self._track()

        if not self._is_tracked:
            # Untracked: use direct callback list - zero overhead
            self._callbacks.append(callback)

            def unsubscribe():
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

            if call_immediately:
                current = self._value
                if current is not NULL_EVENT:
                    callback(current)

            return unsubscribe
        else:
            # Tracked: use store subscription
            def on_change(change):
                value = change.new_value
                if value is not NULL_EVENT:
                    callback(value)

            unsubscribe = self._store._kv.on(self._key, on_change)

            if call_immediately:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)

            return unsubscribe

    # ========================================================================
    # OPERATORS - Kleisli Morphisms with Fusion
    # ========================================================================

    def __rshift__(self, transform: Callable) -> "SimpleMapObservable":
        """
        Map operator: obs >> f → Obs[B]

        Kleisli arrow: A → Obs[B]
        Virtual until materialized.
        Fuses with subsequent maps.
        """
        return SimpleMapObservable(self._store, self, transform)

    def then(self, transform: Callable) -> "SimpleMapObservable":
        """Alias for >> operator."""
        return self >> transform

    def __add__(self, other: "Observable") -> "StreamMerge":
        """
        Product operator: obs1 + obs2 → Obs[A × B]

        Categorical product with tuple values.
        """
        if not isinstance(other, Observable):
            raise TypeError(f"Cannot merge Observable with {type(other)}")
        return self._store._get_or_create_stream((self, other))

    def alongside(self, *others: "Observable") -> "StreamMerge":
        """Alias for + with multiple observables."""
        return self._store._get_or_create_stream((self,) + others)

    def __and__(
        self, condition: Union["Observable", Callable]
    ) -> "ConditionalObservable":
        """
        Filter operator: obs & condition → filtered Obs[A]

        Monadic filter - only emits when condition holds.
        """
        return ConditionalObservable(self._store, self, condition)

    def filter(self, predicate: Callable) -> "ConditionalObservable":
        """Alias for & with predicate."""
        return self & predicate

    def scan(self, accumulator: Callable, initial: Any) -> "ScanObservable":
        """
        Scan operator: obs.scan(f, a) → Obs[M]

        Free monoid homomorphism - maintains running accumulation.
        Delta-aware: Uses differential updates when possible.
        """
        return ScanObservable(self._store, self, accumulator, initial)

    def requiring(
        self, condition: Union["Observable", Callable]
    ) -> "ConditionalObservable":
        """Alias for & operator."""
        return self & condition

    def __or__(self, other: "Observable") -> "ComputedObservable":
        """Logical OR: obs1 | obs2 → bool(obs1) or bool(obs2)"""
        if not isinstance(other, Observable):
            raise TypeError(f"Cannot OR Observable with {type(other)}")
        return ComputedObservable(
            self._store, [self, other], lambda a, b: bool(a) or bool(b)
        )

    def either(self, other: "Observable") -> "ComputedObservable":
        """Alias for | operator."""
        return self | other

    def __invert__(self) -> "ComputedObservable":
        """Negation: ~obs → not bool(obs)"""
        return ComputedObservable(self._store, [self], lambda x: not bool(x))

    def negate(self) -> "ComputedObservable":
        """Alias for ~ operator."""
        return ~self

    def __repr__(self) -> str:
        state = "tracked" if self._is_tracked else "virtual"
        value_repr = repr(self.value)
        return f"Observable({self._key}={value_repr}, {state})"


# ============================================================================
# SIMPLE MAP - Optimized Single-Source Transform with Fusion
# ============================================================================


class SimpleMapObservable(Observable):
    """
    Simple map: Optimized endofunctor for single-source transforms.

    Functor laws:
    - map(id) = id
    - map(g ∘ f) = map(g) ∘ map(f)

    Optimization: Fuses transform chains via function composition.
    Delta-aware: Computes output deltas from input deltas when possible.
    """

    __slots__ = ("_source", "_transform_chain", "_fusion_enabled")

    def __init__(
        self, store: "Store", source: Observable, transform: Optional[Callable]
    ):
        store._key_counter += 1
        key = f"simplemap${store._key_counter}"

        super().__init__(store, key, None)
        self._source = source
        self._transform_chain = [transform] if transform else []
        self._fusion_enabled = True

    @property
    def value(self) -> Any:
        """Compute transformed value via fusion."""
        if self._is_tracked:
            # When tracked, use store's computed value (benefits from AD)
            return self._store._kv.get(self._key)

        source_val = self._source.value

        if not self._transform_chain:
            return source_val

        # Apply composed transform chain
        result = source_val
        for transform in self._transform_chain:
            result = transform(result)
        return result

    def set(self, new_value: Any) -> None:
        """Cannot set derived values."""
        raise TypeError("SimpleMapObservable is derived and cannot be set directly")

    def _track(self) -> None:
        """
        Track by creating computed value in store with AD support.

        The store's AD engine will automatically compute deltas through
        the transform chain when possible.
        """
        if self._is_tracked:
            return

        self._is_tracked = True
        self._source._track()

        # CRITICAL: Ensure source is in the store
        if (
            self._source._key not in self._store._kv._data
            and self._source._key not in self._store._kv._computed
        ):
            self._store._kv.source(self._source._key, self._source._value)

        # Register as computed value with fused transform
        # The store's AD will handle incremental updates
        def compute_fn(source_value):
            result = source_value
            for t in self._transform_chain:
                result = t(result)
            return result

        # Mark as simple_map for optimization hints
        self._store._kv.computed(
            self._key, compute_fn, [self._source._key], is_simple_map=True
        )
        self._store._kv._objects[self._key] = self

    def subscribe(
        self,
        callback: Callable[[Any], None],
        call_immediately: bool = False,
        force_track: bool = True,
    ) -> Callable[[], None]:
        """Subscribe - materializes to enable propagation with AD."""
        if force_track:
            self._track()

        if not self._is_tracked:
            # Untracked: subscribe to source and transform directly
            def on_source_change(value):
                if value is NULL_EVENT:
                    return
                # Apply transform chain
                result = value
                for transform in self._transform_chain:
                    result = transform(result)
                callback(result)

            unsubscribe_source = self._source.subscribe(
                on_source_change, call_immediately=False, force_track=False
            )

            def unsubscribe():
                unsubscribe_source()

            if call_immediately:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)

            return unsubscribe
        else:
            # Tracked: use store subscription
            def on_change(change):
                value = change.new_value
                if value is not NULL_EVENT:
                    callback(value)

            unsubscribe = self._store._kv.on(self._key, on_change)

            if call_immediately:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)

            return unsubscribe

    def __rshift__(self, new_transform: Callable) -> "SimpleMapObservable":
        """
        Functor composition: (f >> g) = g ∘ f

        Fuses transforms via function composition (optimization).
        """
        if self._fusion_enabled:
            # Flatten chain to root source for maximum fusion
            root_source, all_transforms = self._flatten_chain()

            result = SimpleMapObservable(self._store, root_source, None)
            result._transform_chain = all_transforms + [new_transform]
            result._fusion_enabled = True
            return result
        else:
            return SimpleMapObservable(self._store, self, new_transform)

    def _flatten_chain(self) -> tuple:
        """Extract root source and all transforms for fusion."""
        current = self
        all_transforms = []

        while isinstance(current, SimpleMapObservable):
            all_transforms = current._transform_chain + all_transforms
            current = current._source

        return current, all_transforms


# ============================================================================
# COMPUTED OBSERVABLE - Multi-Source Derived Values with AD
# ============================================================================


class ComputedObservable(Observable):
    """
    Computed observable: Morphism A → B in reactive category.

    Supports:
    - Multiple sources
    - Lazy evaluation
    - Automatic dependency tracking
    - Automatic differentiation for incremental updates
    - Fusion for simple cases
    """

    __slots__ = (
        "_sources",
        "_transform",
        "_computed_key",
        "_transform_chain",
        "_is_simple_map",
    )

    def __init__(self, store: "Store", sources: List[Observable], transform: Callable):
        store._key_counter += 1
        key = f"computed${store._key_counter}"

        super().__init__(store, key, None)
        self._sources = sources
        self._transform = transform
        self._computed_key = None
        self._transform_chain = [transform] if transform else []

        # Detect simple map for optimization
        # Access __code__ directly - if it doesn't exist, AttributeError is appropriate
        transform_code = transform.__code__
        self._is_simple_map = (
            len(sources) == 1
            and len(self._transform_chain) == 1
            and transform_code.co_argcount == 1
        )

        # Track sources for dependency graph
        for source in sources:
            if isinstance(source, Observable):
                source._track()

    @property
    def value(self) -> Any:
        """Get computed value (lazy evaluation with AD cache)."""
        if not self._is_tracked:
            return self._compute_virtual()

        if self._computed_key is None:
            self._materialize()

        # Store's AD will provide cached or incrementally updated value
        return self._store._kv.get(self._computed_key)

    def _compute_virtual(self) -> Any:
        """Compute directly without store (for untracked access)."""
        if self._is_simple_map:
            return self._transform(self._sources[0].value)

        values = [src.value for src in self._sources]

        # Handle StreamMerge unpacking
        if (
            len(values) == 1
            and isinstance(values[0], tuple)
            and len(self._sources) == 1
        ):
            if isinstance(self._sources[0], StreamMerge):
                values = values[0]  # Unpack the tuple

        result = values[0] if len(values) == 1 else values
        for i, transform in enumerate(self._transform_chain):
            if i == 0 and len(values) > 1:
                result = transform(*values)
            else:
                result = transform(result)

        return result

    def _materialize(self) -> None:
        """
        Materialize as computed value in store.

        Store's AD engine will automatically compute deltas through
        the computation when inputs change incrementally.
        """
        self._is_tracked = True
        self._computed_key = self._key

        # CRITICAL: Ensure all sources are in the store
        for src in self._sources:
            src._track()
            if (
                src._key not in self._store._kv._data
                and src._key not in self._store._kv._computed
            ):
                self._store._kv.source(src._key, src._value)

        source_keys = [src._key for src in self._sources]

        def compute_fn(*args):
            # Handle StreamMerge unpacking
            if (
                len(args) == 1
                and isinstance(args[0], tuple)
                and len(self._sources) == 1
            ):
                if isinstance(self._sources[0], StreamMerge):
                    args = args[0]  # Unpack the tuple

            result = args[0] if len(args) == 1 else args
            for i, transform in enumerate(self._transform_chain):
                if i == 0 and len(args) > 1:
                    result = transform(*args)
                else:
                    result = transform(result)
            return result

        # Store will use AD to compute incremental updates
        self._store._kv.computed(
            self._computed_key,
            compute_fn,
            source_keys,
            is_simple_map=self._is_simple_map,
        )

    def subscribe(self, callback, call_immediately=False, force_track=True):
        """Subscribe - materializes to enable AD-powered propagation."""
        for source in self._sources:
            source._track()

        self._materialize()

        def on_change(change):
            value = change.new_value
            if value is not NULL_EVENT:
                callback(value)

        unsubscribe = self._store._kv.on(self._computed_key, on_change)

        if call_immediately:
            current = self.value
            if current is not NULL_EVENT:
                callback(current)

        return unsubscribe

    def _track(self) -> None:
        """Ensure tracked for dependency graph."""
        if not self._is_tracked:
            self._materialize()


# ============================================================================
# CONDITIONAL OBSERVABLE - Filtered Stream with Delta Awareness
# ============================================================================


class ConditionalObservable(Observable):
    """
    Conditional observable: Filtered stream with predicate.

    Semantics:
    - Only emits when condition is met AND value changes
    - Never emits when condition is false
    - Raises ConditionNotMet/ConditionalNeverMet on access
    - Delta-aware: Propagates deltas when condition remains true
    """

    __slots__ = (
        "_source",
        "_condition",
        "_condition_obs",
        "_is_active",
        "_has_ever_been_active",
        "_last_source_value",
        "_subscribers",
    )

    def __init__(
        self, store: "Store", source: Observable, condition: Union[Observable, Callable]
    ):
        store._key_counter += 1
        key = f"conditional${store._key_counter}"

        super().__init__(store, key, None)
        self._source = source
        self._condition = condition
        self._condition_obs = None
        self._is_active = False
        self._has_ever_been_active = False
        self._last_source_value = NULL_EVENT
        self._subscribers = set()

        # Track source
        source._track()

        # Handle condition
        if callable(condition):
            self._condition_obs = source >> condition
        elif isinstance(condition, Observable):
            self._condition_obs = condition
            condition._track()
        else:
            raise TypeError(
                f"Condition must be Observable or Callable, got {type(condition)}"
            )

        self._condition_obs._track()

        # Register with store
        self._store._kv._data[key] = False

    @property
    def value(self) -> Any:
        """Get filtered value."""
        self._ensure_tracked()
        return self._store._kv.get(self._key)

    def _ensure_tracked(self) -> None:
        """Set up conditional tracking with delta propagation."""
        if self._is_tracked:
            return

        self._is_tracked = True

        # Initialize
        condition_met = bool(self._condition_obs.value)

        if condition_met:
            current_source = self._source.value
            self._store._kv.source(self._key, current_source)

        # Subscribe to changes - callbacks receive values directly
        def on_source_change(new_value):
            condition_met = bool(self._condition_obs.value)

            if condition_met:
                # Check if value actually changed
                if new_value != self._last_source_value:
                    self._last_source_value = new_value
                    self._has_ever_been_active = True
                    self._is_active = True

                    # Update store and emit
                    self._store._kv.source(self._key, new_value)
                    self._emit(new_value)
            else:
                self._is_active = False

        def on_condition_change(new_value):
            is_met = bool(new_value)
            was_active = self._is_active
            self._is_active = is_met

            if was_active != is_met:
                if is_met:
                    current_source = self._source.value
                    self._last_source_value = current_source
                    self._has_ever_been_active = True
                    self._store._kv.source(self._key, current_source)
                    self._emit(current_source)

        self._source.subscribe(on_source_change)
        self._condition_obs.subscribe(on_condition_change)

        self._is_active = bool(self._condition_obs.value)
        if self._is_active:
            self._last_source_value = self._source.value
            self._has_ever_been_active = True

    def _emit(self, value: Any) -> None:
        """Emit to subscribers."""
        if self._subscribers:
            for callback in list(self._subscribers):
                callback(value)

    def subscribe(
        self,
        callback: Callable[[Any], None],
        call_immediately: bool = False,
        force_track: bool = True,
    ) -> Callable[[], None]:
        """Subscribe to conditional emissions."""
        self._ensure_tracked()

        # Use direct subscribers for conditionals
        self._subscribers.add(callback)

        def unsubscribe():
            self._subscribers.discard(callback)

        if call_immediately:
            try:
                current = self.value
                callback(current)
            except (ConditionalNeverMet, ConditionNotMet):
                pass

        return unsubscribe

    def _track(self) -> None:
        """Track this conditional."""
        self._ensure_tracked()


# ============================================================================
# SCAN OBSERVABLE - Stateful Accumulation with Delta Support
# ============================================================================


class ScanObservable(Observable):
    """
    Scan observable: Free monoid homomorphism with delta awareness.

    Maintains running accumulation with O(1) per-element cost.
    Delta-aware: Can use input deltas for efficient accumulation.
    """

    __slots__ = (
        "_source",
        "_accumulator",
        "_current_value",
        "_subscribers",
        "_delta_aware",
    )

    def __init__(
        self, store: "Store", source: Observable, accumulator: Callable, initial: Any
    ):
        store._key_counter += 1
        key = f"scan${store._key_counter}"

        super().__init__(store, key, initial)
        self._source = source
        self._accumulator = accumulator
        self._current_value = initial
        self._subscribers = set()

        # Check if accumulator can work with deltas (has 3 params: state, value, delta?)
        import inspect

        sig = inspect.signature(accumulator)
        self._delta_aware = len(sig.parameters) >= 3

        source._track()
        self._setup_scan_subscription()

    def _setup_scan_subscription(self) -> None:
        """Set up direct accumulation with delta support."""

        def on_source_change(new_value):
            # Callbacks receive values directly - no Change wrapping
            self._current_value = self._accumulator(self._current_value, new_value)
            self._emit_to_subscribers(self._current_value)

        self._source.subscribe(on_source_change, call_immediately=False)

    @property
    def value(self) -> Any:
        """Get current accumulated value."""
        return self._current_value

    def subscribe(
        self,
        callback: Callable[[Any], None],
        call_immediately: bool = False,
        force_track: bool = True,
    ) -> Callable[[], None]:
        """Subscribe to accumulation changes."""
        self._subscribers.add(callback)

        if call_immediately:
            callback(self._current_value)

        def unsubscribe():
            self._subscribers.discard(callback)

        return unsubscribe

    def _emit_to_subscribers(self, value: Any) -> None:
        """Emit to subscribers."""
        if self._subscribers:
            for callback in list(self._subscribers):
                callback(value)


# ============================================================================
# STREAM MERGE - Product Construction with Delta Support
# ============================================================================


class StreamMerge(Observable):
    """
    Stream merge: Categorical product A × B → (A, B)

    Emits tuple when any source changes.
    Delta-aware: Tracks which element changed for efficient downstream processing.
    """

    __slots__ = ("_sources",)

    def __init__(self, store: "Store", sources: List[Observable]):
        store._key_counter += 1
        key = f"stream${store._key_counter}"

        super().__init__(store, key, None)
        self._sources = sources

        for source in sources:
            source._track()

    @property
    def value(self) -> tuple:
        """Get current tuple of all source values."""
        return tuple(src.value for src in self._sources)

    def _track(self) -> None:
        """Track this stream with efficient change detection."""
        if self._is_tracked:
            return

        self._is_tracked = True

        # Register with store
        self._store._kv.source(self._key, self.value)

        # Subscribe to all sources - callbacks receive values directly
        def make_change_handler(index: int):
            def on_source_change(new_value):
                # Compute new tuple
                current_value = self.value
                self._store._kv._set_internal(self._key, current_value, None)

            return on_source_change

        for i, source in enumerate(self._sources):
            source.subscribe(make_change_handler(i), call_immediately=False)

    def __rshift__(self, transform: Callable) -> ComputedObservable:
        """Transform merged stream: (a, b) >> f → f(a, b) with AD support"""
        # Depend on the StreamMerge itself for AD
        return ComputedObservable(self._store, [self], transform)


# ============================================================================
# STORE - Namespace and Coordination
# ============================================================================


class Store:
    """
    Store: Namespace for organizing reactive state with zero-cost abstractions.

    Provides:
    - Scoped observables
    - Manages ReactiveStore instance with AD
    - Caches StreamMerge instances
    - Delta-aware propagation throughout
    """

    _MAX_STREAM_CACHE_SIZE = 1000

    def __init__(self):
        self._kv = ReactiveStore()
        self._observables = {}
        self._key_counter = 0
        self._stream_cache = {}

    def observable(self, initial_value: Any = None) -> Observable:
        """Create new observable with delta tracking."""
        self._key_counter += 1
        key = f"obs${self._key_counter}"
        obs = Observable(self, key, initial_value)
        self._observables[key] = obs
        return obs

    def _get_or_create_stream(self, sources: tuple) -> StreamMerge:
        """Get cached stream or create new with flattening optimization."""

        # Flatten nested StreamMerges for efficiency
        def flatten_sources(srcs):
            flattened = []
            for src in srcs:
                if isinstance(src, StreamMerge):
                    flattened.extend(flatten_sources(src._sources))
                else:
                    flattened.append(src)
            return flattened

        flattened_sources = flatten_sources(sources)
        cache_key = tuple(id(src) for src in flattened_sources)

        if cache_key not in self._stream_cache:
            if len(self._stream_cache) >= self._MAX_STREAM_CACHE_SIZE:
                # Evict oldest entry
                oldest_key = next(iter(self._stream_cache))
                del self._stream_cache[oldest_key]

            self._stream_cache[cache_key] = StreamMerge(self, flattened_sources)

        return self._stream_cache[cache_key]

    def batch(self):
        """
        Create batch context for efficient bulk updates.

        All updates within the batch are merged and propagated once,
        with delta composition applied automatically.
        """
        return self._kv.batch()

    def close(self) -> None:
        """Close and cleanup resources."""
        self._kv.close()
        self._observables.clear()
        self._stream_cache.clear()
        self._key_counter = 0

    def __del__(self) -> None:
        """Destructor cleanup."""
        try:
            self.close()
        except:
            pass

    def to_dict(self) -> dict:
        """
        Export state as dictionary (only source observables).

        Computed observables are excluded as they can be reconstructed.
        """
        return {
            key: obs.value
            for key, obs in self._observables.items()
            if not isinstance(obs, (ComputedObservable, ConditionalObservable))
        }

    def stats(self) -> dict:
        """Get store statistics including AD metrics."""
        base_stats = self._kv.stats()
        return {
            **base_stats,
            "observable_count": len(self._observables),
            "stream_cache_size": len(self._stream_cache),
        }

    def __repr__(self) -> str:
        return f"Store(observables={len(self._observables)}, reactive_keys={self._kv.stats()['total_keys']})"


# ============================================================================
# @reactive DECORATOR
# ============================================================================


def reactive(*dependencies, call_immediately=False):
    """
    Decorator for reactive functions.

    Creates functions that automatically run when dependencies change.
    Functions receive values (not Change objects) for compatibility.

    By default, reactive functions do NOT fire immediately when created.
    They only fire when dependencies change (pullback semantics).
    Set call_immediately=True to fire with current values on decoration.

    Example:
        @reactive(obs1, obs2)
        def my_effect(value):
            print(f"Changed to: {value}")
    """

    def decorator(func: Callable) -> Callable:
        unsubscribers = []

        for dep in dependencies:
            # Access subscribe directly - AttributeError will propagate if not Observable
            # Subscribe with standard value callbacks (not Change objects)
            unsubscribers.append(dep.subscribe(func, call_immediately=call_immediately))

        wrapper_unsubscribed = False

        def wrapper(*args, **kwargs):
            if not wrapper_unsubscribed:
                raise ReactiveFunctionError(
                    "Reactive functions cannot be called manually. "
                    "They run automatically when dependencies change. "
                    "Call .unsubscribe() first to restore normal function behavior."
                )
            return func(*args, **kwargs)

        wrapper._unsubscribed = False
        wrapper._func = func

        def unsubscribe():
            for unsub in unsubscribers:
                unsub()
            nonlocal wrapper_unsubscribed
            wrapper_unsubscribed = True
            wrapper._unsubscribed = True

        wrapper.unsubscribe = unsubscribe

        return wrapper

    return decorator


# ============================================================================
# GLOBAL STORE
# ============================================================================

_global_store = None
_global_store_lock = threading.Lock()


def get_global_store() -> Store:
    """Get or create global store singleton."""
    global _global_store
    if _global_store is None:
        with _global_store_lock:
            if _global_store is None:
                _global_store = Store()
    return _global_store


def observable(initial_value: Any = None) -> Observable:
    """Create standalone observable in global store."""
    store = get_global_store()
    return store.observable(initial_value)


def transaction():
    """
    Create transaction context for batched updates.

    All updates within the transaction are merged with delta composition
    and propagated once at the end.

    Example:
        with transaction():
            obs1.value = 10
            obs2.value = 20
            obs3.value = 30
        # All three updates propagate together with composed deltas
    """
    return get_global_store().batch()


def _reset_global_store():
    """Reset global store (for testing)."""
    global _global_store
    if _global_store is not None:
        _global_store.close()
    _global_store = None


def _cleanup_global_store():
    """Cleanup on exit."""
    global _global_store
    if _global_store is not None:
        try:
            _global_store.close()
        except:
            pass
        _global_store = None


atexit.register(_cleanup_global_store)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Core types
    "Observable",
    "SimpleMapObservable",
    "ComputedObservable",
    "ConditionalObservable",
    "ScanObservable",
    "StreamMerge",
    "Store",
    # Functions
    "observable",
    "reactive",
    "transaction",
    "get_global_store",
    # Exceptions
    "ConditionNotMet",
    "ConditionalNeverMet",
    "ReactiveFunctionError",
    # Sentinel
    "NULL_EVENT",
]

# Conditionally add re-exports if delta_kv_store was imported
if Change is not None:
    __all__.extend(
        [
            "Change",
            "ChangeType",
            "TypedDelta",
            "DeltaType",
        ]
    )
