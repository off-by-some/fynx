"""
FynX Observable - Concrete Mutable Observable Implementation
===========================================================

This module provides the concrete Observable class, a mutable observable
that extends BaseObservable with write access and transaction support.
"""

from typing import Any, Callable, Optional, Type, TypeVar

from fynx.observable.core.abstract.observable import BaseObservable

T = TypeVar("T")


class Observable(BaseObservable[T]):
    """
    A reactive value that automatically notifies dependents when it changes.

    This is the concrete mutable observable that extends BaseObservable
    with write access and transaction support.
    """

    def set(self, value: Optional[T]) -> "Observable[T]":
        """
        Set the value of this observable and notify all observers.

        This method provides write access for mutable observables,
        including circular dependency protection.
        """
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
