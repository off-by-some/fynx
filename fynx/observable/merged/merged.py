"""
FynX MergedObservable - Tuple-Based Reactive Values
==================================================

This module provides a MergedObservable that combines multiple observables
into a single reactive tuple. It provides semantic clarity for merging operations
without complex special-cased implementations.

The key insight: merging is just tuple construction with multiple dependencies.
No need for complex caching or special chain handling.
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from ..computed import ComputedObservable
from ..merged.protocol import Mergeable
from ..operations import OperatorMixin, TupleMixin

if TYPE_CHECKING:
    from ..primitives.protocol import Observable

# Global registry for function-to-context mappings (for cleanup testing)
_func_to_contexts = {}

T = TypeVar("T")


class MergedObservable(ComputedObservable[T], Mergeable[T], OperatorMixin, TupleMixin):
    """
    A wrapper that combines multiple observables into a single reactive tuple.

    This is essentially a ComputedObservable with multiple dependencies.
    The merging is tuple construction with reactive updates.

    Key characteristics:
    - Combines observables into tuples: `x + y` creates `(x.value, y.value)`
    - Updates when any source changes
    - Works with all existing operators
    - Simple implementation with good performance

    Example:
        ```python
        from fynx import observable

        # Individual observables
        x = observable(10)
        y = observable(20)

        # Merge them into a single reactive unit
        point = x + y
        print(point.value)  # (10, 20)

        # Works with all operators
        distance = point >> (lambda px, py: (px**2 + py**2)**0.5)
        print(distance.value)  # 22.360679774997898

        # Changes to either coordinate update everything
        x.set(15)
        print(point.value)                  # (15, 20)
        print(distance.value)               # 25.0
        ```
    """

    def __init__(self, *observables: "Observable") -> None:
        """
        Create a merged observable from multiple source observables.

        Uses efficient updates and handles nested MergedObservables properly.

        Args:
            *observables: Variable number of Observable instances to combine.
                         At least one observable must be provided.

        Raises:
            ValueError: If no observables are provided
        """
        if not observables:
            raise ValueError("At least one observable must be provided for merging")

        # Flatten nested MergedObservables for associativity: (a + b) + c = a + b + c
        flattened_sources = []
        for obs in observables:
            if isinstance(obs, MergedObservable):
                flattened_sources.extend(obs._source_observables)
            else:
                flattened_sources.append(obs)

        # Store flattened source observables
        self._source_observables = flattened_sources
        self._n_sources = len(flattened_sources)

        # Pre-allocate tuple storage for efficient updates
        self._current_values = [None] * self._n_sources

        # Initialize current values
        for i, obs in enumerate(flattened_sources):
            self._current_values[i] = obs.value

        # Initialize as a ComputedObservable
        initial_tuple = tuple(self._current_values)
        super().__init__("merged", initial_tuple, lambda _: tuple(self._current_values))

        # Add dependency edges to cycle detector for all sources
        from ..primitives.context import ReactiveContextImpl

        cycle_detector = ReactiveContextImpl._get_cycle_detector()
        for obs in flattened_sources:
            cycle_detector.add_edge(obs, self)

        # Subscribe to sources with index tracking
        self._update_handlers = []
        for i, obs in enumerate(flattened_sources):
            handler = self._create_update_handler(i)
            self._update_handlers.append(handler)
            obs.subscribe(handler)

    def _create_update_handler(self, index):
        """Create an update handler for a specific source index."""

        def update_merged(_=None):
            # Update only the specific index
            self._current_values[index] = self._source_observables[index].value
            # Rebuild tuple efficiently
            self._set_computed_value(tuple(self._current_values))

        return update_merged

    def __add__(self, other: "Observable") -> "MergedObservable":
        """
        Chain merging with another observable using the + operator.

        Args:
            other: Another Observable to merge with this merged observable

        Returns:
            A new MergedObservable containing all source observables from this
            merged observable plus the additional observable.
        """
        return MergedObservable(*self._source_observables, other)  # type: ignore

    def __enter__(self) -> "MergedObservable[T]":
        """Support context manager protocol for convenient unpacking."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager protocol."""
        return False

    def __call__(self, func: Callable) -> None:
        """Make MergedObservable callable for reactive callbacks."""
        # Call immediately with current values
        func(*self.value)

        # Subscribe to changes for reactive behavior
        def reactive_handler(values):
            func(*values)

        self.subscribe(reactive_handler)

    def __iter__(self):
        """Support tuple unpacking: a, b = merged_observable."""
        return iter(self.value)

    def __setitem__(self, key: int, value: Any) -> None:
        """Support index assignment: merged[0] = new_value."""
        if isinstance(key, int) and 0 <= key < len(self._source_observables):
            self._source_observables[key].set(value)
        else:
            raise IndexError(
                f"Index {key} out of range for merged observable with {len(self._source_observables)} elements"
            )

    def subscribe(self, func: Callable) -> "MergedObservable[T]":
        """
        Subscribe a function to react to changes in any of the merged observables.

        Args:
            func: A callable that will receive the tuple of values from all merged
                  observables as a single argument.

        Returns:
            This merged observable instance for method chaining.
        """
        # Direct subscription - func will receive the tuple value directly
        self.add_observer(func)
        return self

    def unsubscribe(self, func: Callable) -> None:
        """
        Unsubscribe a function from this merged observable.

        Args:
            func: The function to unsubscribe from this merged observable
        """
        self.remove_observer(func)
