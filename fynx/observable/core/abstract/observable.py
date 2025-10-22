"""
FynX BaseObservable - Abstract Base Class for All Observables
=============================================================

This module provides the BaseObservable abstract base class that contains
the core reactive infrastructure shared by all observables in FynX.

BaseObservable provides:
- Observer management and subscription system
- Notification system with propagation context
- Value storage and access with dependency tracking
- Reactive context integration
- Type checking utilities
- Magic methods for transparent behavior
- Key/identity management
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Set, Type, TypeVar

from fynx.observable.core.abstract.operations import OperatorMixin
from fynx.observable.core.context import (
    PropagationContext,
    ReactiveContext,
    TransactionContext,
)
from fynx.observable.core.value.value import ObservableValue
from fynx.observable.types.protocols.observable_protocol import (
    Observable as ObservableInterface,
)

T = TypeVar("T")


class BaseObservable(ABC, ObservableInterface[T], OperatorMixin):
    """
    Abstract base class for all observables in FynX.

    This class provides the core reactive infrastructure that all observables share:
    - Observer management and subscription system
    - Notification system with propagation context
    - Value storage and access with dependency tracking
    - Reactive context integration
    - Type checking utilities
    - Magic methods for transparent behavior
    - Key/identity management

    Subclasses must implement:
    - `_should_notify_observers()` - Hook for conditional notification
    - `set()` - Value setting behavior (varies by observable type)
    """

    _current_context: Optional["ReactiveContext"] = None

    def __init__(
        self, key: Optional[str] = None, initial_value: Optional[T] = None
    ) -> None:
        self._key = key or "<unnamed>"
        self._observers: Set[Callable] = set()
        self._lock = threading.RLock()
        self._value_wrapper = ObservableValue(
            initial_value, on_change=lambda old, new: self._notify_observers(new)
        )

    @property
    def key(self) -> str:
        """Get the key/identifier for this observable."""
        return self._key

    @property
    def value(self) -> Optional[T]:
        """
        Get the current value with dependency tracking.

        This property automatically tracks dependencies when accessed within
        a reactive context (e.g., during computed observable evaluation).
        """
        if BaseObservable._current_context is not None:
            BaseObservable._current_context.add_dependency(self)

        # Track dependencies for computed observables during evaluation
        current_computed = getattr(BaseObservable, "_current_computed_observable", None)
        if current_computed is not None:
            from .context import ReactiveContextImpl

            cycle_detector = ReactiveContextImpl._get_cycle_detector()
            try:
                cycle_detector.add_edge(self, current_computed)
            except ValueError:
                raise RuntimeError(
                    f"Circular dependency detected: accessing {self._key} during computation of {current_computed._key}"
                )

        return self._value_wrapper.unwrap()

    @abstractmethod
    def set(self, value: Optional[T]) -> "BaseObservable[T]":
        """
        Set the value of this observable.

        This method must be implemented by subclasses to provide
        appropriate value setting behavior for each observable type.
        """
        pass

    def add_observer(self, observer: Callable) -> None:
        """Add an observer for change notifications."""
        with self._lock:
            self._observers.add(observer)

    def remove_observer(self, observer: Callable) -> None:
        """Remove an observer from change notifications."""
        with self._lock:
            self._observers.discard(observer)

    def has_observer(self, observer: Callable) -> bool:
        """Check if an observer is registered."""
        with self._lock:
            return observer in self._observers

    def subscribe(self, func: Callable) -> "BaseObservable[T]":
        """Subscribe to value changes."""
        self.add_observer(func)
        return self

    def unsubscribe(self, func: Callable) -> None:
        """Unsubscribe from value changes."""
        self.remove_observer(func)

    def _notify_observers(self, value: Optional[T]) -> None:
        """
        Notify all observers of a value change.

        This method implements the core notification system with:
        - Circular dependency protection
        - Propagation context for breadth-first updates
        - Transaction support
        - Conditional notification hook
        """
        # Check for circular dependency: cannot notify while already notifying
        if hasattr(self, "_is_notifying") and self._is_notifying:
            raise RuntimeError(
                f"Circular dependency detected: cannot notify '{self._key}' while it is already notifying observers"
            )

        # Mark that we're notifying (for other checks)
        self._is_notifying = True
        try:
            # Fast snapshot of observers
            with self._lock:
                observers_snapshot = tuple(self._observers)

            # Check transaction
            active_transactions = TransactionContext._get_active()

            # Always use propagation context to prevent stack overflow
            state = PropagationContext._get_state()

            # Enqueue notifications
            for observer in observers_snapshot:
                PropagationContext._enqueue_notification(observer, self, value)

            # Process if not already propagating
            if not state["is_propagating"]:
                PropagationContext._process_notifications()
        finally:
            self._is_notifying = False

    def _should_notify_observers(self) -> bool:
        """
        Hook for subclasses to control notification behavior.

        By default, all observables notify their observers. Subclasses
        can override this to implement conditional notification (e.g.,
        ConditionalObservable only notifies when conditions are met).

        Returns:
            True if observers should be notified, False otherwise.
        """
        return True

    def transaction(self):
        """Create a transaction context for batching updates."""
        return TransactionContext(self)

    @classmethod
    def _reset_notification_state(cls) -> None:
        """Reset notification state for testing."""
        PropagationContext._reset_state()
        TransactionContext._reset_state()

    # Type checking utilities
    @staticmethod
    def _is_observable(obj) -> bool:
        """Check if an object is an observable without triggering property access."""
        return isinstance(obj, BaseObservable)

    @staticmethod
    def _is_derived(obj) -> bool:
        """Check if an object is a derived observable."""
        from .derived import DerivedValue

        return isinstance(obj, DerivedValue)

    @staticmethod
    def _is_conditional(obj) -> bool:
        """Check if an object is a conditional observable."""
        from fynx.observable.computed.types import is_conditional_observable

        return is_conditional_observable(obj)

    # Magic methods for transparent behavior
    def __bool__(self) -> bool:
        return bool(self._value_wrapper)

    def __str__(self) -> str:
        return str(self._value_wrapper)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._key!r}, {self._value_wrapper.unwrap()!r})"

    def __eq__(self, other: object) -> bool:
        return self._value_wrapper == other

    def __hash__(self) -> int:
        return id(self)

    def __len__(self) -> int:
        return len(self._value_wrapper)

    def __iter__(self):
        return iter(self._value_wrapper)

    def __getitem__(self, key: Any) -> Any:
        return self._value_wrapper[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._value_wrapper[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._value_wrapper[key]

    def __contains__(self, item: Any) -> bool:
        return item in self._value_wrapper

    def __lt__(self, other: Any) -> bool:
        return self._value_wrapper < other

    def __le__(self, other: Any) -> bool:
        return self._value_wrapper <= other

    def __gt__(self, other: Any) -> bool:
        return self._value_wrapper > other

    def __ge__(self, other: Any) -> bool:
        return self._value_wrapper >= other

    def __neg__(self) -> Any:
        return -self._value_wrapper

    def __pos__(self) -> Any:
        return +self._value_wrapper

    def __abs__(self) -> Any:
        return abs(self._value_wrapper)

    def __sub__(self, other: Any) -> Any:
        return self._value_wrapper - other

    def __mul__(self, other: Any) -> Any:
        return self._value_wrapper * other

    def __truediv__(self, other: Any) -> Any:
        return self._value_wrapper / other

    def __floordiv__(self, other: Any) -> bool:
        return self._value_wrapper // other

    def __mod__(self, other: Any) -> Any:
        return self._value_wrapper % other

    def __pow__(self, other: Any) -> Any:
        return self._value_wrapper**other

    def __iadd__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper += other
        return self

    def __isub__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper -= other
        return self

    def __imul__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper *= other
        return self

    def __itruediv__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper /= other
        return self

    def __ifloordiv__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper //= other
        return self

    def __imod__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper %= other
        return self

    def __ipow__(self, other: Any) -> "BaseObservable[T]":
        self._value_wrapper **= other
        return self

    def __set_name__(self, owner: Type, name: str) -> None:
        if self._key == "<unnamed>":
            if getattr(self, "_is_computed", False):
                self._key = f"<computed:{name}>"
            else:
                self._key = name

        if getattr(self, "_is_computed", False):
            return

        try:
            from ...store import Store

            if issubclass(owner, Store):
                return
        except ImportError:
            pass

        from fynx.observable.core.value.descriptors import SubscriptableDescriptor

        descriptor = SubscriptableDescriptor(self._value_wrapper.unwrap())
        descriptor.attr_name = name
        descriptor._owner_class = owner
        setattr(owner, name, descriptor)
