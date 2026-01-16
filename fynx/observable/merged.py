"""
FynX MergedObservable - Combined Reactive Values
================================================

MergedObservable combines multiple observables into a single reactive tuple that
updates when any component changes. Like coordinate pairs where both x and y must
be known together, merged observables treat multiple related values as a single
atomic unit. Change any component, and the entire tuple updates automatically.

MergedObservable creates a read-only computed observable whose value is a tuple containing
the current values of all source observables. When any source observable changes, the merged
observable automatically recalculates its tuple value and notifies all subscribers. That
automatic coordination eliminates manual synchronization—you declare the relationship, and
the framework maintains it.

The merge operation uses the `+` operator between observables. That syntax creates a new MergedObservable containing both values as a tuple:

```python
from fynx import observable

width = observable(10)
height = observable(20)
dimensions = width + height  # Creates MergedObservable
print(dimensions.value)  # (10, 20)

width.set(15)
print(dimensions.value)  # (15, 20)

# Merged observables are read-only
dimensions.set((5, 5))  # Raises ValueError: Computed observables are read-only and cannot be set directly
```

Result: multiple reactive values that behave as a single atomic unit, updating automatically when any component changes. That pattern enables functions that need multiple related parameters, computed values that depend on several inputs, and coordinated state updates across multiple variables.
"""

from typing import Callable, Iterable, TypeVar

from ..registry import _all_reactive_contexts, _func_to_contexts
from .base import Observable, ReactiveContext
from .computed import ComputedObservable
from .interfaces import Mergeable
from .operators import OperatorMixin, TupleMixin

T = TypeVar("T")


class MergedObservable(ComputedObservable[T], Mergeable[T], OperatorMixin, TupleMixin):
    """
    A computed observable that combines multiple observables into a single reactive tuple.

    MergedObservable combines multiple observables into a single reactive tuple that
    updates when any component changes. Change any source value, and the entire tuple
    recalculates automatically. This enables treating multiple related values as a
    single atomic unit.

    The merged observable creates a read-only computed observable whose value is a tuple containing the current values of all source observables. When any source observable changes, the merged observable automatically recalculates its tuple value and notifies all subscribers. That automatic coordination eliminates manual synchronization—you declare the relationship, and the framework maintains it.

    As a computed observable, MergedObservable is read-only and cannot be set directly. Its value always derives from its source observables, ensuring consistency. That constraint prevents breaking reactive relationships—if you could set a merged value, it would diverge from its sources.

    This enables treating multiple related reactive values as a single atomic unit. Functions that need multiple related parameters can receive the tuple directly. Computed values that depend on several inputs can derive from the merged observable. Coordinated state updates across multiple variables happen automatically when any component changes.

    Example:
        ```python
        from fynx import observable

        # Individual observables
        x = observable(10)
        y = observable(20)

        # Merge them into a single reactive unit
        point = x + y
        print(point.value)  # (10, 20)

        # Computed values can work with the tuple
        distance_from_origin = point.then(
            lambda px, py: (px**2 + py**2)**0.5
        )
        print(distance_from_origin.value)  # 22.360679774997898

        # Changes to either coordinate update everything
        x.set(15)
        print(point.value)                  # (15, 20)
        print(distance_from_origin.value)   # 25.0
        ```

    The merged observable's value is always a tuple, even when merging just two observables. That consistency provides a uniform interface for computed functions—they always receive a tuple, regardless of how many observables were merged.

    See Also:
        ComputedObservable: Base computed observable class
        >> operator: For creating derived values from merged observables
    """

    def __init__(self, *observables: "Observable") -> None:
        """
        Create a merged observable from multiple source observables.

        Args:
            *observables: Variable number of Observable instances to combine.
                         At least one observable must be provided.

        Raises:
            ValueError: If no observables are provided
        """
        if not observables:
            raise ValueError("At least one observable must be provided for merging")

        # Call ComputedObservable constructor with appropriate parameters
        initial_tuple = tuple(obs.value for obs in observables)

        # Create a computation function that combines the source observables
        def compute_merged_value():
            return tuple(obs.value for obs in observables)

        # NOTE: MyPy's generics can't perfectly model this complex inheritance pattern
        # where T represents a tuple type in the subclass but a single value in the parent
        super().__init__("merged", initial_tuple, compute_merged_value)  # type: ignore
        self._source_observables = list(observables)
        self._cached_tuple = None  # Cache for tuple value

        # Set up observers on all source observables to update our tuple
        def update_merged():
            # Invalidate cache and update value
            self._cached_tuple = None
            new_value = tuple(obs.value for obs in self._source_observables)
            # Use the computed observable's internal method to update value
            self._set_computed_value(new_value)

        # Set up dependency tracking for each source observable
        for obs in self._source_observables:
            obs.add_observer(update_merged)

    @property
    def value(self):
        """
        Get the current tuple value, using cache when possible.

        Returns the current values of all source observables as a tuple.
        Uses caching to avoid recomputing the tuple on every access.

        Returns:
            A tuple containing the current values of all source observables,
            in the order they were provided to the constructor.

        Example:
            ```python
            from fynx import observable

            x = observable(10)
            y = observable(20)
            merged = x + y

            print(merged.value)  # (10, 20)
            x.set(15)
            print(merged.value)  # (15, 20) - cache invalidated and recomputed
            ```
        """
        if self._cached_tuple is None:
            self._cached_tuple = tuple(obs.value for obs in self._source_observables)

        return self._cached_tuple

    def __enter__(self):
        """
        Context manager entry for reactive blocks.

        Enables experimental syntax for defining reactive blocks that execute
        whenever any of the merged observables change.

        Returns:
            A context object that can be called with a function to create reactive behavior.

        Example:
            ```python
            # Experimental context manager syntax
            with merged_obs as ctx:
                ctx(lambda x, y: print(f"Values changed: {x}, {y}"))
            ```

        Note:
            This is an experimental feature. The more common approach is to use
            subscribe() or the @reactive decorator.
        """

        class ReactiveWithContext:
            def __init__(self, merged_obs):
                self.merged_obs = merged_obs

            def __iter__(self):
                """Allow unpacking the current tuple value."""
                return iter(self.merged_obs._value)

            def __call__(self, block):
                """
                Set up reactive execution of the block function.

                The block function will be called with the current values of all
                merged observables whenever any of them change.

                Args:
                    block: Function to call reactively. Should accept as many
                          arguments as there are merged observables.
                """

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
        """
        Context manager exit.

        Currently does nothing, but allows the context manager to work properly.
        """
        pass

    def __add__(self, other: "Observable") -> "MergedObservable":
        """
        Chain merging with another observable using the + operator.

        Enables fluent syntax for building up merged observables incrementally.
        Each + operation creates a new MergedObservable containing all previous
        observables plus the new one.

        Args:
            other: Another Observable to merge with this merged observable

        Returns:
            A new MergedObservable containing all source observables from this
            merged observable plus the additional observable.
        """
        return MergedObservable(*self._source_observables, other)  # type: ignore

    def subscribe(self, func: Callable) -> "MergedObservable[T]":
        """
        Subscribe a function to react to changes in any of the merged observables.

        The subscribed function will be called whenever any source observable changes.
        This provides a way to react to coordinated changes across multiple observables.

        Args:
            func: A callable that will receive the current values of all merged
                  observables as separate arguments, in the order they were merged.
                  The function signature should match the number of merged observables.

        Returns:
            This merged observable instance for method chaining.

        Examples:
            ```python
            from fynx import observable

            x = observable(1)
            y = observable(2)
            coords = x + y

            def on_coords_change(x_val, y_val):
                print(f"Coordinates: ({x_val}, {y_val})")

            coords.subscribe(on_coords_change)

            x.set(10)  # Prints: "Coordinates: (10, 2)"
            y.set(20)  # Prints: "Coordinates: (10, 20)"
            ```

        The function is called only when observables change. It is not called immediately upon subscription—that behavior differs from some reactive frameworks, but ensures that subscriptions don't trigger unnecessary initial executions.

        See Also:
            unsubscribe: Remove a subscription
            reactive: Decorator-based reactive functions
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

        Removes the subscription for the specified function, preventing it from
        being called when the merged observable changes. This properly cleans up
        the reactive context and removes all observers.

        Args:
            func: The function that was previously subscribed to this merged observable.
                  Must be the same function object that was passed to subscribe().

        Examples:
            ```python
            from fynx import observable

            x = observable(1)
            y = observable(2)

            def handler(x, y):
                print(f"Changed: {x}, {y}")

            coords = x + y
            coords.subscribe(handler)

            # Later, unsubscribe
            coords.unsubscribe(handler)  # No longer called when coords change
            ```

        This only removes subscriptions to this specific merged observable. If the same function is subscribed to other observables, those subscriptions remain active. That selective removal enables fine-grained control over reactive behavior—you can unsubscribe from specific merged observables without affecting other subscriptions.

        See Also:
            subscribe: Add a subscription
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
