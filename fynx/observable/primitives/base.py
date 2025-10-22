"""
FynX Observable - Core Reactive Value Implementation
====================================================

This module provides the fundamental building blocks for reactive programming in FynX.
"""

import asyncio
import logging
import threading
from collections import deque
from typing import Any, Callable, Optional, Set, Type, TypeVar

from ..operations import OperatorMixin
from ..types.observable_protocols import Observable as ObservableInterface
from ..value.value import ObservableValue
from .context import ReactiveContext

T = TypeVar("T")


class PropagationContext:
    """Manages breadth-first change propagation to prevent stack overflow."""

    _local = threading.local()

    @classmethod
    def _get_state(cls) -> dict:
        if not hasattr(cls._local, "state"):
            cls._local.state = {"is_propagating": False, "pending": deque()}
        return cls._local.state

    @classmethod
    def _enqueue_notification(
        cls, observer: Callable, observable: Any, value: Any
    ) -> None:
        cls._get_state()["pending"].append((observer, observable, value))

    @classmethod
    def _process_notifications(cls) -> None:
        state = cls._get_state()
        if state["is_propagating"]:
            return

        state["is_propagating"] = True
        try:
            while state["pending"]:
                observer, observable, value = state["pending"].popleft()
                # Check if observable should notify before calling observer
                if (
                    hasattr(observable, "_should_notify_observers")
                    and not observable._should_notify_observers()
                ):
                    continue
                observer(value)
        finally:
            state["is_propagating"] = False

    @classmethod
    async def _process_notifications_async(cls) -> None:
        state = cls._get_state()
        if state["is_propagating"]:
            return

        state["is_propagating"] = True
        try:
            while state["pending"]:
                observer, observable, value = state["pending"].popleft()
                # Check if observable should notify before calling observer
                if (
                    hasattr(observable, "_should_notify_observers")
                    and not observable._should_notify_observers()
                ):
                    continue
                try:
                    if asyncio.iscoroutinefunction(observer):
                        await observer(value)
                    else:
                        observer(value)
                except Exception as e:
                    logging.error(f"Error in async observer notification: {e}")
        finally:
            state["is_propagating"] = False


class TransactionContext:
    """Batches observable updates and emits single notification on commit."""

    _local = threading.local()

    @classmethod
    def _get_active(cls) -> list:
        if not hasattr(cls._local, "active"):
            cls._local.active = []
        return cls._local.active

    def __init__(self, observable: "Observable"):
        self.observable = observable
        self._is_outermost = False

    def __enter__(self):
        active = self._get_active()
        self._is_outermost = not active
        active.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        active = self._get_active()
        active.pop()
        if self._is_outermost and not active:
            PropagationContext._process_notifications()


class Observable(ObservableInterface[T], OperatorMixin):
    """
    A reactive value that automatically notifies dependents when it changes.
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
        return self._key

    @property
    def value(self) -> Optional[T]:
        if Observable._current_context is not None:
            Observable._current_context.add_dependency(self)

        # Track dependencies for computed observables during evaluation
        current_computed = getattr(Observable, "_current_computed_observable", None)
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

    def set(self, value: Optional[T]) -> "Observable[T]":
        # Check for circular dependency: cannot modify observable being computed from
        if hasattr(self, "_computed_from"):
            raise RuntimeError(
                f"Circular dependency detected: cannot modify '{self._key}' during computation that depends on it"
            )

        # Check for circular dependency: cannot modify observable that is currently notifying
        # But allow updating the value if we're not going to notify (e.g., internal state updates)
        if hasattr(self, "_is_notifying") and self._is_notifying:
            # Allow internal value updates but prevent notification cycles
            # We'll check this in the _notify_observers method instead
            pass

        self._value_wrapper.value = value
        return self

    def add_observer(self, observer: Callable) -> None:
        with self._lock:
            self._observers.add(observer)

    def remove_observer(self, observer: Callable) -> None:
        with self._lock:
            self._observers.discard(observer)

    def has_observer(self, observer: Callable) -> bool:
        with self._lock:
            return observer in self._observers

    def subscribe(self, func: Callable) -> "Observable[T]":
        self.add_observer(func)
        return self

    def unsubscribe(self, func: Callable) -> None:
        self.remove_observer(func)

    def _notify_observers(self, value: Optional[T]) -> None:
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
            # The fast path was causing recursion issues with long chains
            state = PropagationContext._get_state()

            # Enqueue notifications
            for observer in observers_snapshot:
                PropagationContext._enqueue_notification(observer, self, value)

            # Process if not already propagating
            if not state["is_propagating"]:
                PropagationContext._process_notifications()
        finally:
            self._is_notifying = False

    def transaction(self):
        return TransactionContext(self)

    @classmethod
    def _reset_notification_state(cls) -> None:
        PropagationContext._local.__dict__.clear()
        TransactionContext._local.__dict__.clear()

    # Magic methods for transparent behavior
    def __bool__(self) -> bool:
        return bool(self._value_wrapper)

    def __str__(self) -> str:
        return str(self._value_wrapper)

    def __repr__(self) -> str:
        return f"Observable({self._key!r}, {self._value_wrapper.unwrap()!r})"

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

    def __floordiv__(self, other: Any) -> Any:
        return self._value_wrapper // other

    def __mod__(self, other: Any) -> Any:
        return self._value_wrapper % other

    def __pow__(self, other: Any) -> Any:
        return self._value_wrapper**other

    def __iadd__(self, other: Any) -> "Observable[T]":
        self._value_wrapper += other
        return self

    def __isub__(self, other: Any) -> "Observable[T]":
        self._value_wrapper -= other
        return self

    def __imul__(self, other: Any) -> "Observable[T]":
        self._value_wrapper *= other
        return self

    def __itruediv__(self, other: Any) -> "Observable[T]":
        self._value_wrapper /= other
        return self

    def __ifloordiv__(self, other: Any) -> "Observable[T]":
        self._value_wrapper //= other
        return self

    def __imod__(self, other: Any) -> "Observable[T]":
        self._value_wrapper %= other
        return self

    def __ipow__(self, other: Any) -> "Observable[T]":
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

        from ..value.descriptors import SubscriptableDescriptor

        descriptor = SubscriptableDescriptor(self._value_wrapper.unwrap())
        descriptor.attr_name = name
        descriptor._owner_class = owner
        setattr(owner, name, descriptor)
