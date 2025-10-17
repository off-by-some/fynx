"""
FynX Observable Computed - Computed Observable Implementation
===========================================================

This module provides the ComputedObservable class, a read-only observable that
derives its value from other observables.

Computed observables enable:
- **Derived State**: Create values that depend on other values
- **Read-only Protection**: Prevent accidental direct modification
- **Type Safety**: Type-based distinction from regular observables

The ComputedObservable class is a subclass of Observable that represents computed/derived
values. Unlike regular observables, computed observables are read-only and cannot
be set directly - their values are automatically calculated from their dependencies.

Example:
    ```python
    from fynx.observable.computed import ComputedObservable

    # Regular observable
    counter = observable(0)

    # Computed observable (read-only)
    doubled = ComputedObservable("doubled", lambda: counter.value * 2)
    doubled.set(10)  # Raises ValueError: Computed observables are read-only
    ```
"""

from typing import Optional, TypeVar

from .base import Observable

T = TypeVar("T")


class ComputedObservable(Observable[T]):
    """
    A read-only observable that derives its value from other observables.

    ComputedObservable is a subclass of Observable that represents computed/derived
    values. Unlike regular observables, computed observables are read-only and cannot
    be set directly - their values are automatically calculated from their dependencies.

    This provides type-based distinction from regular observables, eliminating the need
    for magic strings or runtime checks. Computed observables maintain the same interface
    as regular observables for reading values and subscribing to changes, but enforce
    immutability at runtime.

    Example:
        ```python
        # Regular observable
        counter = observable(0)

        # Computed observable (read-only)
        doubled = ComputedObservable("doubled", lambda: counter.value * 2)
        doubled.set(10)  # Raises ValueError: Computed observables are read-only
        ```
    """

    def __init__(
        self, key: Optional[str] = None, initial_value: Optional[T] = None
    ) -> None:
        super().__init__(key, initial_value)

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating computed observable values.

        This method is called by the computed() function when dependencies change.
        It bypasses the read-only protection to allow legitimate internal updates.

        Args:
            value: The new computed value
        """
        super().set(value)

    def set(self, value: Optional[T]) -> None:
        """
        Attempting to set a computed observable directly is not allowed.

        Computed observables are read-only and their values are automatically
        calculated from their dependencies. Use the computed() function to create
        derived observables instead.

        Args:
            value: The value to set (ignored)

        Raises:
            ValueError: Always raised since computed observables cannot be set directly
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )
