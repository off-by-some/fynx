"""
Fynx MergedObservable - Combined Reactive Values
================================================

This module provides the MergedObservable class, which combines multiple individual
observables into a single reactive unit. This enables treating related observables
as a cohesive group that updates atomically when any component changes.

Merged observables are useful for:
- **Coordinated Updates**: When multiple values need to change together
- **Computed Relationships**: When derived values depend on multiple inputs
- **Tuple Operations**: When you need to pass multiple reactive values as a unit
- **State Composition**: Building complex state from simpler reactive components

The merge operation is created using the `|` operator between observables:

```python
from fynx import observable

width = observable(10)
height = observable(20)
dimensions = width | height  # Creates MergedObservable
print(dimensions.value)  # (10, 20)

width.set(15)
print(dimensions.value)  # (15, 20)
```
"""

from typing import Callable, Iterator, Optional, Tuple, TypeVar

from ..registry import _all_reactive_contexts, _func_to_contexts
from .base import Observable, ReactiveContext

T = TypeVar("T")


class MergedObservable(Observable[T]):
    """
    An observable that combines multiple observables into a single reactive tuple.

    MergedObservable creates a composite observable whose value is a tuple containing
    the current values of all source observables. When any source observable changes,
    the merged observable updates its tuple value and notifies all subscribers.

    This enables treating multiple related reactive values as a single atomic unit,
    which is particularly useful for:

    - Functions that need multiple related parameters
    - Computed values that depend on several inputs
    - Coordinated state updates across multiple variables
    - Maintaining referential consistency between related values

    Example:
        ```python
        from fynx import observable, computed

        # Individual observables
        x = observable(10)
        y = observable(20)

        # Merge them into a single reactive unit
        point = x | y
        print(point.value)  # (10, 20)

        # Computed values can work with the tuple
        distance_from_origin = computed(
            lambda px, py: (px**2 + py**2)**0.5,
            point
        )
        print(distance_from_origin.value)  # 22.360679774997898

        # Changes to either coordinate update everything
        x.set(15)
        print(point.value)                  # (15, 20)
        print(distance_from_origin.value)   # 25.0
        ```

    Note:
        The merged observable's value is always a tuple, even when merging just
        two observables. This provides a consistent interface for computed functions.

    See Also:
        Observable: Base observable class
        computed: For creating derived values from merged observables
    """

    def __init__(self, *observables: Observable) -> None:
        """
        Create a merged observable from multiple source observables.

        Args:
            *observables: Variable number of Observable instances to combine.
                         At least one observable must be provided.

        Raises:
            ValueError: If no observables are provided
        """
        # Call parent constructor with a key and initial tuple value
        initial_tuple = tuple(obs.value for obs in observables)

        # NOTE: MyPy's generics can't perfectly model this complex inheritance pattern
        # where T represents a tuple type in the subclass but a single value in the parent
        super().__init__("merged", initial_tuple)  # type: ignore
        self._source_observables = list(observables)
        self._cached_tuple = None  # Cache for tuple value

        # Set up observers on all source observables to update our tuple
        def update_merged():
            # Invalidate cache and update value
            self._cached_tuple = None
            new_value = tuple(obs.value for obs in self._source_observables)
            # Use the parent set method to trigger our own observers
            super(MergedObservable, self).set(new_value)

        for obs in self._source_observables:
            obs.add_observer(update_merged)

    @property
    def value(self):
        """Get the current tuple value, using cache when possible."""
        if self._cached_tuple is None:
            self._cached_tuple = tuple(obs.value for obs in self._source_observables)

        return self._cached_tuple

    def set(self, value):
        """Override set to invalidate cache."""
        self._cached_tuple = None
        super().set(value)

    def __enter__(self):
        """Context manager entry - returns reactive context ."""

        class ReactiveWithContext:
            def __init__(self, merged_obs):
                self.merged_obs = merged_obs

            def __iter__(self):
                """Allow unpacking as tuple."""
                return iter(self.merged_obs._value)

            def __call__(self, block):
                """Set up reactive execution of the block function."""

                def run():
                    values = tuple(
                        obs.value for obs in self.merged_obs._source_observables
                    )
                    block(*values)

                # Bind to all source observables
                for obs in self.merged_obs._source_observables:
                    obs.add_observer(run)

                # Execute once immediately
                run()

        return ReactiveWithContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def __or__(self, other: Observable) -> "MergedObservable":
        """
        Chain merging with another observable.

        This supports syntax like: obs1 | obs2 | obs3

        Args:
            other: Another Observable to merge with

        Returns:
            A new MergedObservable containing all source observables
        """
        return MergedObservable(*self._source_observables, other)

    def __iter__(self):
        """Allow unpacking the tuple value."""
        return iter(self._value)

    def __len__(self) -> int:
        """Return the number of combined observables."""
        return len(self._source_observables)

    def __getitem__(self, index: int) -> T:
        """Allow indexing into the merged observable like a tuple."""
        if self._value is None:
            raise IndexError("MergedObservable has no value")
        return self._value[index]  # type: ignore

    def __setitem__(self, index: int, value: T) -> None:
        """Allow setting values by index (updates the corresponding source observable)."""
        if 0 <= index < len(self._source_observables):
            self._source_observables[index].set(value)
        else:
            raise IndexError("Index out of range")

    def subscribe(self, func: Callable) -> "MergedObservable[T]":
        """
        Subscribe a function to react to changes in any of the merged observables.

        The function will be called with the current values of all merged observables
        whenever any of them changes.

        Args:
            func: The function to call when observables change.
                  It will receive the current values as arguments in the order
                  the observables were merged.

        Returns:
            This merged observable instance for method chaining.
        """

        def multi_observable_reaction():
            # Disable automatic dependency tracking for merged observables
            # since we don't want to add observers to source observables
            old_context = Observable._current_context
            Observable._current_context = None
            try:
                # Get values from all observables in the order they were merged
                values = [obs.value for obs in self._source_observables]
                func(*values)
            finally:
                Observable._current_context = old_context

        context = ReactiveContext(multi_observable_reaction, func, self)

        # Register context globally for unsubscribe functionality
        _all_reactive_contexts.add(context)

        # Add to function mapping for O(1) unsubscribe
        _func_to_contexts.setdefault(func, []).append(context)

        # Track this merged observable as the dependency (not the source observables)
        # since the observer is added to this merged observable
        context.dependencies.add(self)
        self.add_observer(context.run)

        return self

    def unsubscribe(self, func: Callable) -> None:
        """
        Unsubscribe a function from this merged observable.

        This will dispose of any ReactiveContext instances that were created
        for the given function and are subscribed to this merged observable.

        Args:
            func: The function to unsubscribe from this merged observable
        """
        if func in _func_to_contexts:
            # Filter contexts that are subscribed to this observable
            contexts_to_remove = [
                ctx
                for ctx in _func_to_contexts[func]
                if ctx.subscribed_observable is self
            ]

            for context in contexts_to_remove:
                context.dispose()
                _all_reactive_contexts.discard(context)
                _func_to_contexts[func].remove(context)

            # Clean up empty lists
            if not _func_to_contexts[func]:
                del _func_to_contexts[func]
