"""
FynX DerivedValue - Abstract Base Class for Read-Only Derived Observables
==========================================================================

This module provides the DerivedValue abstract base class that represents
read-only observables that derive their values from other observables.

DerivedValue provides:
- Read-only enforcement (set() raises ValueError)
- Source tracking and dependency management
- Lazy evaluation infrastructure
- Computation hooks (abstract methods)
- Internal value updates for framework use
- Dependency subscription setup
- Chain building for .then() operator
"""

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Set, TypeVar

from .observable import BaseObservable

T = TypeVar("T")


class DerivedValue(BaseObservable[T]):
    """
    Abstract base class for read-only derived observables.

    This class represents observables that derive their values from other
    observables and cannot be set directly. It provides the infrastructure
    for lazy evaluation, dependency tracking, and computation management.

    Subclasses must implement:
    - `_compute_value()` - The computation logic
    - `_should_recompute()` - When recomputation is needed
    - `_on_source_change()` - Handle source changes
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        source_observable: Optional["BaseObservable"] = None,
        source_observables: Optional[List["BaseObservable"]] = None,
    ) -> None:
        super().__init__(key, initial_value)

        # Source tracking
        self._source_observable = source_observable
        self._source_observables = source_observables or []

        # Lazy evaluation infrastructure
        self._is_dirty = True
        self._is_updating = False

        # Set up dependency tracking and observers
        if source_observable is not None:
            self._setup_source_observers()

    def set(self, value: Optional[T]) -> None:
        """
        Prevent direct modification of derived observable values.

        Derived observables are read-only by design because their values are
        automatically calculated from other observables. Attempting to set them
        directly would break the reactive relationship.

        Raises:
            ValueError: Always raised to prevent direct modification.
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )

    def _find_ultimate_source(self) -> "BaseObservable":
        """
        Walk the chain to find the root non-derived observable.

        Returns:
            The root source observable.
        """
        current = self._source_observable
        while (
            isinstance(current, DerivedValue) and current._source_observable is not None
        ):
            current = current._source_observable
        return current if current is not None else self

    def _mark_dirty(self) -> None:
        """Mark that recomputation is needed."""
        self._is_dirty = True

    def _evaluate_if_dirty(self) -> None:
        """Check and recompute if needed."""
        if self._should_recompute():
            self._compute_and_update()

    def _compute_and_update(self) -> None:
        """Compute the value and update if changed."""
        if self._is_updating:
            return  # Prevent recursion

        self._is_updating = True
        try:
            old_value = self._value_wrapper.unwrap()
            new_value = self._compute_value()

            if old_value != new_value:
                self._set_computed_value(new_value)
        finally:
            self._is_updating = False

    @abstractmethod
    def _compute_value(self) -> T:
        """
        Compute the derived value.

        Subclasses must implement this method to define how the
        derived value is calculated from its sources.

        Returns:
            The computed value.
        """
        pass

    @abstractmethod
    def _should_recompute(self) -> bool:
        """
        Determine if recomputation is needed.

        Subclasses must implement this method to define when
        the derived value should be recomputed.

        Returns:
            True if recomputation is needed, False otherwise.
        """
        pass

    def _set_computed_value(self, value: T) -> None:
        """
        Internal method for framework to update derived values.

        This method bypasses the read-only restriction and is used
        internally by the framework to update computed values.

        Args:
            value: The new computed value.
        """
        self._is_dirty = False
        self._value_wrapper._value = value
        self._notify_observers(value)

    def _setup_source_observers(self) -> None:
        """
        Subscribe to source changes - template method pattern.

        Default implementation handles single source observables.
        Subclasses can override for multiple sources or special handling.
        """
        if self._source_observable is not None:
            self._source_observable.subscribe(self._on_source_change)

    @abstractmethod
    def _on_source_change(self, value: Any) -> None:
        """
        Handle source changes.

        Subclasses must implement this method to define how
        source changes are handled.

        Args:
            value: The new value from the source.
        """
        pass

    def _extend_chain(self, func: Callable) -> "LazyChainBuilder":
        """
        Common chain extension logic for .then() operator.

        Returns:
            A LazyChainBuilder for building transformation chains.
        """
        from ...util import LazyChainBuilder

        # Find the ultimate source observable
        source = self._find_ultimate_source()

        # Build the chain of functions
        functions = []

        # If we have a composed function, add it to the chain
        if hasattr(self, "_composed_func") and self._composed_func is not None:
            functions.append(self._composed_func)

        # Add the new function
        functions.append(func)

        # Return a lazy chain builder
        return LazyChainBuilder(source, functions)

    @property
    def value(self) -> T:
        """
        Get the current value with lazy evaluation.

        For derived observables, this triggers recomputation if needed.
        """
        # Track dependency if we're in a reactive context
        if BaseObservable._current_context is not None:
            BaseObservable._current_context.add_dependency(self)

        # Only recompute if dirty
        if self._is_dirty and self._should_recompute():
            self._compute_and_update()

        # Return the current value (either cached or newly computed)
        return super().value
