"""
Observable - Category-Theoretic Reactive Frontend

Mathematical Foundation:
- Observable Comonad: (Obs, ε, δ) with extract and duplicate
- Subscription Monoid: (Sub, ⊕, ε) unified across all states
- Operator Kleisli: (map, filter, scan) as virtual morphisms
- Zero-cost: Virtual until materialization

Core Principles:
1. Single unified subscription mechanism (σ: Obs × Callback → Sub)
2. Always emit on set() - no significance testing at observable layer
3. Operators as virtual chains - materialize only on subscription
4. Tracking is transparent - same subscription semantics before/after
"""

import atexit
import threading
from typing import Any, Callable, List, Optional, TypeVar, Union

from .delta_kv_store import ChangeType, Delta, ReactiveStore

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
    Observable: Object in the reactive category.

    Comonad operations:
    - extract (ε): value property - get current value
    - duplicate (δ): _track - create tracked observable

    Subscription: Unified σ: Obs × Callback → Sub

    States:
    - untracked: Local value, direct callbacks
    - tracked: Store-managed, reactive propagation

    Invariant: Subscription works identically in both states
    """

    __slots__ = ("_store", "_key", "_value", "_is_tracked")

    def __init__(self, store: "Store", key: str, initial_value: Any = None):
        self._store = store
        self._key = key
        self._value = initial_value
        self._is_tracked = False

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
        Update operation with emission.

        Critical: Always emits on set() (event algebra).
        No significance testing here - that's propagation layer concern.
        """
        if not self._is_tracked:
            # Untracked: update local value and emit to store
            old_value = self._value
            self._value = new_value

            # Always propagate through store for consistency
            # This ensures batching works correctly and subscribers are notified
            delta = Delta(
                key=self._key,
                change_type=ChangeType.SOURCE_UPDATE,
                old_value=old_value,
                new_value=new_value,
                timestamp=None,
            )
            self._store._kv._propagate_change(delta)
        else:
            # Tracked: delegate to store
            self._store._kv.set(self._key, new_value)

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

        self._is_tracked = True
        # Register with store (using set which properly registers in _data)
        self._store._kv.set(self._key, self._value)

    def subscribe(
        self,
        callback: Callable[[Any], None],
        call_immediately: bool = False,
        force_track: bool = True,
    ) -> Callable[[], None]:
        """
        Unified subscription operation: σ: Obs × Callback → Sub

        Single subscription mechanism that works in both tracked/untracked states.
        Tracking is transparent to the subscription API.

        Args:
            callback: Called with new values
            call_immediately: Call with current value immediately
            force_track: Whether to track (default True for reactivity)

        Returns:
            Unsubscribe function
        """
        if force_track:
            # Ensure tracking for reactive propagation
            self._track()

        # Unified subscription through store
        def on_delta(delta: Delta):
            if delta.new_value is not NULL_EVENT:
                callback(delta.new_value)

        unsubscribe = self._store._kv.subscribe(self._key, on_delta)

        if call_immediately:
            try:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)
            except (ConditionalNeverMet, ConditionNotMet):
                pass

        return unsubscribe

    # ========================================================================
    # OPERATORS - Kleisli Morphisms
    # ========================================================================

    def __rshift__(self, transform: Callable) -> "SimpleMapObservable":
        """
        Map operator: obs >> f → Obs[B]

        Kleisli arrow: A → Obs[B]
        Virtual until materialized.
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
# SIMPLE MAP - Optimized Single-Source Transform
# ============================================================================


class SimpleMapObservable(Observable):
    """
    Simple map: Optimized endofunctor for single-source transforms.

    Functor laws:
    - map(id) = id
    - map(g ∘ f) = map(g) ∘ map(f)

    Optimization: Fuses transform chains via function composition.
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
        """Track by creating computed value in store."""
        if self._is_tracked:
            return

        self._is_tracked = True
        self._source._track()

        # CRITICAL: Ensure source is actually in the store before creating computed
        if (
            self._source._key not in self._store._kv._data
            and self._source._key not in self._store._kv._computed
        ):
            self._store._kv.set(self._source._key, self._source._value)

        # Register as computed value
        def compute_fn(source_value):
            result = source_value
            for t in self._transform_chain:
                result = t(result)
            return result

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
        """Subscribe - materializes to enable propagation."""
        if force_track:
            self._track()

        def on_delta(delta: Delta):
            if delta.new_value is not NULL_EVENT:
                callback(delta.new_value)

        unsubscribe = self._store._kv.subscribe(self._key, on_delta)

        if call_immediately:
            try:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)
            except (ConditionalNeverMet, ConditionNotMet):
                pass

        return unsubscribe

    def __rshift__(self, new_transform: Callable) -> "SimpleMapObservable":
        """
        Functor composition: (f >> g) = g ∘ f

        Fuses transforms via function composition (optimization).
        """
        if self._fusion_enabled:
            # Flatten chain to root source
            root_source, all_transforms = self._flatten_chain()

            result = SimpleMapObservable(self._store, root_source, None)
            result._transform_chain = all_transforms + [new_transform]
            result._fusion_enabled = True
            return result
        else:
            return SimpleMapObservable(self._store, self, new_transform)

    def _flatten_chain(self) -> tuple:
        """Extract root source and all transforms."""
        current = self
        all_transforms = []

        while isinstance(current, SimpleMapObservable):
            all_transforms = current._transform_chain + all_transforms
            current = current._source

        return current, all_transforms


# ============================================================================
# COMPUTED OBSERVABLE - Multi-Source Derived Values
# ============================================================================


class ComputedObservable(Observable):
    """
    Computed observable: Morphism A → B in reactive category.

    Supports:
    - Multiple sources
    - Lazy evaluation
    - Automatic dependency tracking
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
        self._is_simple_map = (
            len(sources) == 1
            and len(self._transform_chain) == 1
            and hasattr(transform, "__code__")
            and transform.__code__.co_argcount == 1
        )

        # Track sources
        for source in sources:
            if isinstance(source, Observable):
                source._track()

    @property
    def value(self) -> Any:
        """Get computed value (lazy evaluation)."""
        if not self._is_tracked:
            return self._compute_virtual()

        if self._computed_key is None:
            self._materialize()

        return self._store._kv.get(self._computed_key)

    def _compute_virtual(self) -> Any:
        """Compute directly without store."""
        if self._is_simple_map:
            return self._transform(self._sources[0].value)

        values = [src.value for src in self._sources]

        result = values[0] if len(values) == 1 else values
        for i, transform in enumerate(self._transform_chain):
            if i == 0 and len(values) > 1:
                result = transform(*values)
            else:
                result = transform(result)

        return result

    def _materialize(self) -> None:
        """Materialize as computed value in store."""
        self._is_tracked = True
        self._computed_key = self._key

        # CRITICAL: Ensure all sources are actually in the store
        for src in self._sources:
            src._track()
            if (
                src._key not in self._store._kv._data
                and src._key not in self._store._kv._computed
            ):
                self._store._kv.set(src._key, src._value)

        source_keys = [src._key for src in self._sources]

        def compute_fn(*args):
            result = args[0] if len(args) == 1 else args
            for i, transform in enumerate(self._transform_chain):
                if i == 0 and len(args) > 1:
                    result = transform(*args)
                else:
                    result = transform(result)
            return result

        self._store._kv.computed(
            self._computed_key,
            compute_fn,
            source_keys,
            is_simple_map=self._is_simple_map,
        )

    def subscribe(self, callback, call_immediately=False, force_track=True):
        """Subscribe - materializes to enable propagation."""
        for source in self._sources:
            try:
                source._track()
            except AttributeError:
                pass

        self._materialize()

        def on_delta(delta: Delta):
            if delta.new_value is not NULL_EVENT:
                callback(delta.new_value)

        unsubscribe = self._store._kv.subscribe(self._computed_key, on_delta)

        if call_immediately:
            try:
                current = self.value
                if current is not NULL_EVENT:
                    callback(current)
            except (ConditionalNeverMet, ConditionNotMet):
                pass

        return unsubscribe

    def _track(self) -> None:
        """Ensure tracked."""
        if not self._is_tracked:
            self._materialize()


# ============================================================================
# CONDITIONAL OBSERVABLE - Filtered Stream
# ============================================================================


class ConditionalObservable(Observable):
    """
    Conditional observable: Filtered stream with predicate.

    Semantics:
    - Only emits when condition is met AND value changes
    - Never emits when condition is false
    - Raises ConditionNotMet/ConditionalNeverMet on access
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
        """Set up conditional tracking."""
        if self._is_tracked:
            return

        self._is_tracked = True

        # Initialize
        condition_met = bool(self._condition_obs.value)

        if condition_met:
            current_source = self._source.value
            self._store._kv.set(self._key, current_source)

        # Subscribe to changes
        def on_source_change(new_value):
            condition_met = bool(self._condition_obs.value)

            if condition_met:
                if new_value != self._last_source_value:
                    self._last_source_value = new_value
                    self._has_ever_been_active = True
                    self._is_active = True
                    self._store._kv.set(self._key, new_value)
                    self._emit(new_value)
            else:
                self._is_active = False

        def on_condition_change(is_met):
            is_met = bool(is_met)
            was_active = self._is_active
            self._is_active = is_met

            if was_active != is_met:
                if is_met:
                    current_source = self._source.value
                    self._last_source_value = current_source
                    self._has_ever_been_active = True
                    self._store._kv.set(self._key, current_source)
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

        # Use legacy subscribers for conditionals
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
# SCAN OBSERVABLE - Stateful Accumulation
# ============================================================================


class ScanObservable(Observable):
    """
    Scan observable: Free monoid homomorphism.

    Maintains running accumulation with O(1) per-element cost.
    Bypasses reactive system for pure accumulation.
    """

    __slots__ = ("_source", "_accumulator", "_current_value", "_subscribers")

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

        source._track()
        self._setup_scan_subscription()

    def _setup_scan_subscription(self) -> None:
        """Set up direct accumulation."""

        def on_source_change(new_value):
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
# STREAM MERGE - Product Construction
# ============================================================================


class StreamMerge(Observable):
    """
    Stream merge: Categorical product A × B → (A, B)

    Emits tuple when any source changes.
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
        """Track this stream."""
        if self._is_tracked:
            return

        self._is_tracked = True

        # Register with store
        self._store._kv.source(self._key, self.value)

        # Subscribe to all sources
        def on_any_change(_):
            current_value = self.value
            self._store._kv.source(self._key, current_value)

        for source in self._sources:
            source.subscribe(on_any_change)

    def __rshift__(self, transform: Callable) -> ComputedObservable:
        """Transform merged stream: (a, b) >> f → f(a, b)"""

        def flatten_sources(sources):
            flattened = []
            for src in sources:
                if isinstance(src, StreamMerge):
                    flattened.extend(flatten_sources(src._sources))
                else:
                    flattened.append(src)
            return flattened

        flattened_sources = flatten_sources(self._sources)
        return ComputedObservable(self._store, flattened_sources, transform)


# ============================================================================
# STORE - Namespace and Coordination
# ============================================================================


class Store:
    """
    Store: Namespace for organizing reactive state.

    Provides:
    - Scoped observables
    - Manages ReactiveStore instance
    - Caches StreamMerge instances
    """

    _MAX_STREAM_CACHE_SIZE = 1000

    def __init__(self):
        self._kv = ReactiveStore()
        self._observables = {}
        self._key_counter = 0
        self._stream_cache = {}

    def observable(self, initial_value: Any = None) -> Observable:
        """Create new observable."""
        self._key_counter += 1
        key = f"obs${self._key_counter}"
        obs = Observable(self, key, initial_value)
        self._observables[key] = obs
        return obs

    def _get_or_create_stream(self, sources: tuple) -> StreamMerge:
        """Get cached stream or create new."""
        cache_key = tuple(id(src) for src in sources)

        if cache_key not in self._stream_cache:
            if len(self._stream_cache) >= self._MAX_STREAM_CACHE_SIZE:
                oldest_key = next(iter(self._stream_cache))
                del self._stream_cache[oldest_key]

            self._stream_cache[cache_key] = StreamMerge(self, list(sources))

        return self._stream_cache[cache_key]

    def batch(self):
        """Create batch context."""
        return self._kv.batch()

    def close(self) -> None:
        """Close and cleanup."""
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
        """Export state as dictionary."""
        return {
            key: obs.value
            for key, obs in self._observables.items()
            if not isinstance(obs, (ComputedObservable, ConditionalObservable))
        }

    def __repr__(self) -> str:
        return f"Store(observables={len(self._observables)})"


# ============================================================================
# @reactive DECORATOR
# ============================================================================


def reactive(*dependencies, autorun=None):
    """
    Decorator for reactive functions.

    Creates functions that automatically run when dependencies change.
    """

    def decorator(func: Callable) -> Callable:
        unsubscribers = []

        for dep in dependencies:
            if not hasattr(dep, "subscribe"):
                raise TypeError(f"Dependency must be Observable, got {type(dep)}")

            call_now = (
                True if autorun is True else (False if autorun is False else True)
            )

            unsubscribers.append(dep.subscribe(func, call_immediately=call_now))

        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "_unsubscribed") or not wrapper._unsubscribed:
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
    """Get or create global store."""
    global _global_store
    if _global_store is None:
        with _global_store_lock:
            if _global_store is None:
                _global_store = Store()
    return _global_store


def observable(initial_value: Any = None) -> Observable:
    """Create standalone observable."""
    store = get_global_store()
    return store.observable(initial_value)


def transaction():
    """Create transaction context."""
    return get_global_store().batch()


def _reset_global_store():
    """Reset global store (for testing)."""
    global _global_store
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
