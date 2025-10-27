"""
Base classes and mixins for the observable system.

This module provides the foundational classes and interfaces for all observable types.
The design uses protocol-based typing for documentation while using EAFP (try/except)
at runtime to avoid isinstance() overhead in hot paths.

Architecture:
    ObservableInterface: abstract base defining value/set/subscribe contract
    BaseObservable: concrete base with store integration
    Materializable: mixin for dependency counting and materialization tracking
    Trackable: mixin for DeltaKVStore integration
    OperatorMixin: mixin providing operator overloading (>>, +, &, |, ~)

Protocols (TYPE_CHECKING only):
    DependencyTrackable: objects that can register dependents
    TrackableProtocol: objects that can be tracked in the store

Performance Optimization:
    Protocols are documented but not checked at runtime. Instead, code uses:
        - try/except AttributeError to detect capabilities (fast in happy path)
        - direct method calls without isinstance() checks
    This avoids the overhead of protocol checking while maintaining type safety.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Protocol

if TYPE_CHECKING:
    from .computed import ComputedObservable, ConditionalObservable
    from .store import Store


def _get_dependency_keys(sources: List["Observable"], store) -> List[str]:
    """Extract dependency keys from sources, ensuring they're tracked/materialized."""
    dep_keys = []
    for src in sources:
        # Check by class name to avoid forward reference issues
        class_name = src.__class__.__name__
        if class_name in ("ComputedObservable", "ConditionalObservable"):
            # Source is computed - ensure it's materialized for dependency tracking
            if not src._materialized_key:
                src._materialize_as_node()
            dep_keys.append(src._materialized_key)
        elif class_name == "StreamMerge":
            # StreamMerge - ensure it's tracked for dependency tracking
            if not src._is_tracked:
                src._track_in_store()
            dep_keys.append(src._key)
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
    return dep_keys


# ============================================================================
# Protocols (Documentation Only - NOT for runtime checks)
# ============================================================================


def _try_register_dependent(obj, dependent=None):
    """Helper to register dependent if supported."""
    try:
        if hasattr(obj, "_register_dependent"):
            obj._register_dependent(dependent)
    except AttributeError:
        pass


class DependencyTrackable(Protocol):
    """Protocol for objects that can track dependents.

    NOTE: This is for TYPE CHECKING ONLY. Do not use isinstance() at runtime.
    Use try/except instead for performance.
    """

    def _register_dependent(self) -> None:
        """Register a dependent on this observable."""
        ...


class TrackableProtocol(Protocol):
    """Protocol for objects that can be tracked in the store.

    NOTE: This is for TYPE CHECKING ONLY. Do not use isinstance() at runtime.
    """

    _is_tracked: bool

    def _track_in_store(self) -> None:
        """Track this observable in the store."""
        ...


# ============================================================================
# Base Observable Interface
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


# ============================================================================
# Base Observable Class
# ============================================================================


class BaseObservable(ObservableInterface):
    """Base class for all observable types with common functionality."""

    def __init__(self, store):
        self._store = store
        self._key = None

    def _is_stored(self) -> bool:
        """Check if this observable's value is stored in the KV store."""
        return (
            getattr(self, "_is_tracked", False)
            or getattr(self, "_materialized_key", None) is not None
        )

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


# ============================================================================
# Mixins
# ============================================================================


class Materializable:
    """
    Mixin for materialization logic and dependency tracking.

    Materialization is the process of moving from virtual computation (function
    calls) to tracked storage (DeltaKVStore entries). This mixin tracks:
        - _materialized_key: key in DeltaKVStore if materialized (None otherwise)
        - _dependents_count: number of observables depending on this one

    Materialization rules:
        - Virtual: dependencies computed on-demand via function calls
        - Materialized: value stored in DeltaKVStore with automatic invalidation
        - Trigger: materialize when fan-out detected (2+ dependents) or on subscription
    """

    def __init__(self):
        self._materialized_key = None
        self._dependents_count = 0

    def _register_dependent(self):
        """
        Called when a computed observable depends on this observable.

        Increments dependency counter. Subclasses may use this to trigger
        materialization when fan-out detected.
        """
        self._dependents_count += 1

    @property
    def is_materialized(self) -> bool:
        """
        Check if this observable is materialized in DeltaKVStore.

        Returns True if _materialized_key is set (non-None).
        """
        return self._materialized_key is not None


class Trackable:
    """
    Mixin for DeltaKVStore integration and change tracking.

    Tracks whether observable is in tracked state (registered with DeltaKVStore).
    _track_in_store() must be implemented by subclasses to transition from
    virtual to tracked mode (see Observable and StreamObservable for examples).
    """

    def __init__(self):
        self._is_tracked = False

    def _track_in_store(self):
        """
        Override in subclasses to implement specific tracking logic.

        Should:
            - Set _is_tracked = True
            - Register observable with DeltaKVStore
            - Clear direct propagation lists if applicable
            - Migrate subscribers to store subscription system
        """
        raise NotImplementedError


class Subscribable:
    """
    Mixin for managing subscription lifecycle.

    Maintains list of unsubscribe functions and provides cleanup methods.
    Used by StreamObservable to track source subscriptions and enable
    automatic cleanup when object is no longer referenced.

    Implementation:
        _add_subscription: add unsubscribe function to list
        _clear_subscriptions: call all unsubscribe functions and clear list
    """

    def __init__(self):
        self._subscriptions = []

    def _add_subscription(self, unsubscribe_fn: Callable[[], None]) -> None:
        """
        Add an unsubscribe function to be managed.

        The unsubscribe function will be called during cleanup to ensure
        proper resource release.
        """
        self._subscriptions.append(unsubscribe_fn)

    def _clear_subscriptions(self) -> None:
        """
        Clear all managed subscriptions.

        Calls each unsubscribe function then clears the list. Used during
        teardown to ensure no leaked subscriptions.
        """
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()


class OperatorMixin:
    """
    Mixin providing operator overloading for observables.

    Operators:
        >> (__rshift__): compose transforms (obs >> f → f(obs.value))
        +  (__add__): merge streams (obs1 + obs2 → (obs1, obs2))
        &  (__and__): filter with condition (obs & cond)
        |  (__or__): logical OR (obs1 | obs2)
        ~  (__invert__): logical NOT (~obs)

    All operators delegate to _make_* methods that subclasses implement.
    This keeps operator logic pure while allowing per-type customization.
    """

    def _make_computed(self, sources: list, fn: Callable) -> "BaseObservable":
        """Make a computed observable. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _make_computed")

    def _make_stream(self, sources: list) -> "BaseObservable":
        """Make a stream observable. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _make_stream")

    def _make_conditional(
        self, source: "BaseObservable", condition
    ) -> "BaseObservable":
        """Make a conditional observable. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _make_conditional")

    def then(self, transform: Callable) -> "BaseObservable":
        """Apply transformation: obs >> f → f(obs)"""
        return self._make_computed([self], transform)

    def alongside(self, *others) -> "BaseObservable":
        """Combine streams: (obs₁, obs₂, ..., obsₙ)"""
        all_sources = [self] + list(others)
        return self._make_stream(all_sources)

    def requiring(self, condition) -> "BaseObservable":
        """Filter by condition: only emit when conditions are met"""
        return self._make_conditional(self, condition)

    def either(self, other) -> "BaseObservable":
        """Logical OR: bool(a) or bool(b)"""
        return self._make_computed([self, other], lambda a, b: bool(a) or bool(b))

    def negate(self) -> "BaseObservable":
        """Logical NOT: not bool(obs)"""
        return self._make_computed([self], lambda x: not bool(x))

    # Operator overloads - defined as methods to respect MRO
    def __rshift__(self, other):
        return self.then(other)

    def __add__(self, other):
        return self.alongside(other)

    def __and__(self, other):
        return self.requiring(other)

    def __or__(self, other):
        return self.either(other)

    def __invert__(self):
        return self.negate()


class TupleOperable:
    """Mixin providing operator implementations for tuple-based observables."""

    def then(self, transform: Callable):
        """Apply transformation: obs >> f → f(obs)"""
        self._register_dependent()

        def compute_func(*vals):
            # Flatten nested tuples from StreamMerge sources
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

        # Lazy import to avoid circular dependency
        from .computed import ComputedObservable

        return ComputedObservable(self._store, [self], compute_func)

    def alongside(self, *others):
        """Combine streams: (obs₁, obs₂, ..., obsₙ)"""
        all_obs = (self,) + others
        return self._store._get_or_create_stream(all_obs)

    def requiring(self, condition):
        """Filter by condition: only emit when conditions are met"""
        self._register_dependent()
        if hasattr(condition, "_register_dependent"):
            condition._register_dependent()
        # Pass callable conditions directly - ConditionalObservable handles them efficiently
        from .computed import ConditionalObservable

        return ConditionalObservable(self._store, self, condition)

    def either(self, other):
        """Logical OR: bool(a) or bool(b)"""
        self._register_dependent()
        other._register_dependent()

        from .computed import ComputedObservable

        return ComputedObservable(
            self._store, [self, other], lambda a, b: bool(a) or bool(b)
        )

    def negate(self):
        """Logical NOT: not bool(obs)"""
        self._register_dependent()

        from .computed import ComputedObservable

        return ComputedObservable(self._store, [self], lambda x: not bool(x))

    # Operator overloads
    __rshift__ = then
    __add__ = alongside
    __and__ = requiring
    __or__ = either
    __invert__ = negate
