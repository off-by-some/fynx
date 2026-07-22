"""
FynX MergedObservable - combined reactive values
==================================================

MergedObservable combines multiple observables into a single reactive tuple -
like a coordinate pair where x and y need to be read together.

It's a read-only computed observable whose value is a tuple of the current
values of all its sources. Reading it is lazy and version-validated: the
tuple only refreshes if a source's version changed since the last read.
Subscribing creates the eager boundary needed to deliver notifications when
any source changes.

The merge operation uses the `+` operator between observables, producing a
new MergedObservable containing both values as a tuple:

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

That gives you multiple reactive values that behave as a single atomic unit -
useful for functions that need several related parameters, computed values
that depend on more than one input, or state updates coordinated across
several variables.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Optional,
    TypeVar,
    overload,
)
from weakref import WeakValueDictionary

from typing_extensions import TypeVarTuple, Unpack

from ..registry import _all_reactive_contexts, _func_to_contexts
from ..types import ObservableOperand
from . import base as _base
from .base import (
    Observable,
    ReactiveContext,
    TransformPurityError,
    _transform_purity_message,
)
from .computed import ComputedObservable
from .interfaces import Mergeable
from .operands import is_observable_operand, unwrap_observable
from .operators import OperatorMixin, TupleMixin

T = TypeVar("T")
U = TypeVar("U")
Ts = TypeVarTuple("Ts")


class MergedObservable(
    ComputedObservable[tuple[Unpack[Ts]]],
    Mergeable[tuple[Unpack[Ts]]],
    OperatorMixin[tuple[Unpack[Ts]]],
    TupleMixin,
    Generic[Unpack[Ts]],
):
    """
    A computed observable that combines multiple observables into a single reactive tuple.

    Change any source value and the next read sees a fresh tuple; if the
    product is observed, source changes also notify subscribers directly with
    the updated tuple. Products are canonical for a given ordered source
    list while live, so repeated `a + b` expressions reuse the same product
    node rather than creating a new one each time.

    Like any computed observable, a MergedObservable is read-only - its value
    always derives from its sources, so setting it directly would let it
    diverge from them.

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

    The value is always a tuple, even when merging just two observables, so
    computed functions get a uniform interface regardless of how many
    observables went in.

    See Also:
        ComputedObservable: Base computed observable class
        >> operator: For creating derived values from merged observables
    """

    _product_cache: ClassVar[
        WeakValueDictionary[tuple[int, ...], "MergedObservable"]
    ] = WeakValueDictionary()

    @classmethod
    def _flatten_sources(
        cls, observables: tuple["Observable[Any]", ...]
    ) -> tuple["Observable[Any]", ...]:
        sources: list[Observable[Any]] = []
        for observable in observables:
            if isinstance(observable, MergedObservable):
                sources.extend(observable._source_observables)
            else:
                sources.append(observable)
        return tuple(sources)

    @classmethod
    def from_sources(cls, *observables: "Observable[Any]") -> "MergedObservable":
        sources = cls._flatten_sources(tuple(observables))
        if not sources:
            raise ValueError("At least one observable must be provided for merging")

        cache_key = tuple(id(observable) for observable in sources)
        cached = cls._product_cache.get(cache_key)
        if cached is not None:
            return cached

        merged = cls(*sources)
        cls._product_cache[cache_key] = merged
        return merged

    def __init__(self, *observables: "Observable[Any]") -> None:
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

        self._source_observables: list[Observable[Any]] = list(observables)

        # Call ComputedObservable constructor with appropriate parameters
        initial_tuple: tuple[Any, ...] = tuple(obs.value for obs in observables)

        # Create a computation function that combines the source observables
        def compute_merged_value():
            return tuple(obs.value for obs in observables)

        # NOTE: MyPy's generics can't perfectly model this complex inheritance pattern
        # where T represents a tuple type in the subclass but a single value in the parent
        super().__init__("merged", initial_tuple, compute_merged_value)  # type: ignore
        self._cached_tuple: Optional[tuple[Any, ...]] = initial_tuple
        self._source_signature = self._current_source_signature()
        self._is_dirty = False
        self._source_observer: Callable[[], Any] = self._source_changed_for_product

    def _source_values(self) -> tuple[Any, ...]:
        captured_dependencies: set[Observable[Any]] = set()
        Observable._dependency_capture_stack.append(captured_dependencies)
        try:
            return tuple(obs.value for obs in self._source_observables)
        finally:
            Observable._dependency_capture_stack.pop()

    def _current_source_signature(self) -> Optional[tuple[tuple[int, int], ...]]:
        if not self._source_observables:
            return None
        return tuple(
            (id(observable), observable._version)
            for observable in self._source_observables
        )

    def _source_changed_for_product(self) -> None:
        self._cached_tuple = None
        self._is_dirty = True
        if self._observers:
            Observable._schedule_notification(self)

    def _source_changed(self) -> bool:
        return self._current_source_signature() != self._source_signature or any(
            self._observable_is_stale(observable)
            for observable in self._source_observables
        )

    def _refresh_tuple(self) -> bool:
        new_value = self._source_values()
        self._cached_tuple = new_value
        self._source_signature = self._current_source_signature()
        self._is_dirty = False
        if _base.value_changed(self._value, new_value):
            self._value = new_value
            self._version += 1
            return True
        return False

    def _notify_observers(self) -> None:
        """Refresh the product once per stabilization pass before notifying."""
        if self._is_dirty or self._cached_tuple is None or self._source_changed():
            if not self._refresh_tuple():
                return
        super()._notify_observers()

    @property
    def value(self) -> tuple[Unpack[Ts]]:
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
        if _base._TRANSFORM_EVALUATION_STATE[0]:
            raise TransformPurityError(_transform_purity_message("read", self._key))

        if Observable._dependency_capture_stack:
            Observable._dependency_capture_stack[-1].add(self)

        if self._is_dirty or self._cached_tuple is None or self._source_changed():
            self._refresh_tuple()

        return self._cached_tuple  # type: ignore[return-value]

    def __enter__(self) -> Any:
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

    @overload  # type: ignore[override]
    def __add__(
        self, other: ObservableOperand[U]
    ) -> "MergedObservable[Unpack[Ts], U]": ...

    @overload
    def __add__(self, other: Any) -> Any: ...

    def __add__(self, other: Any) -> Any:
        """
        Chain merging with another observable using the + operator.

        Enables fluent syntax for building up merged observables incrementally.
        The result is the canonical ordered product containing all previous
        observables plus the new one while that product is live.

        Args:
            other: Another Observable to merge with this merged observable

        Returns:
            A MergedObservable containing all source observables from this
            merged observable plus the additional observable.
        """
        if is_observable_operand(other):
            return MergedObservable.from_sources(  # type: ignore[return-value]
                *self._source_observables, unwrap_observable(other)
            )
        return self.value + other

    def __rshift__(  # type: ignore[override]
        self, func: Callable[[Unpack[Ts]], U]
    ) -> "Observable[U]":
        """Transform a merged observable with an unpacked product callback."""
        return self.then(func)

    def then(  # type: ignore[override]
        self, func: Callable[[Unpack[Ts]], U]
    ) -> "Observable[U]":
        """Transform a merged observable by unpacking its tuple value."""
        return super().then(func)  # type: ignore[arg-type]

    def _activate_dependencies(self) -> None:
        if self._dependencies_active:
            return
        self._dependencies_active = True
        for observable in self._source_observables:
            observable.add_observer(self._source_observer)

    def _deactivate_dependencies(self) -> None:
        if not self._dependencies_active:
            return
        for observable in self._source_observables:
            observable.remove_observer(self._source_observer)
        self._dependencies_active = False

    def subscribe(  # type: ignore[override]
        self, func: Callable[[Unpack[Ts]], object]
    ) -> "MergedObservable[Unpack[Ts]]":
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

        The function is called only when a source changes - not immediately
        on subscription, so subscribing doesn't trigger an unnecessary initial call.

        See Also:
            unsubscribe: Remove a subscription
            reactive: Decorator-based reactive functions
        """

        def direct_reaction():
            values = self.value
            func(*values)

        self._subscribe_direct_callback(func, direct_reaction)

        return self

    def unsubscribe(  # type: ignore[override]
        self, func: Callable[[Unpack[Ts]], object]
    ) -> None:
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

        This only removes the subscription on this merged observable - if the
        same function is also subscribed elsewhere, those subscriptions stay active.

        See Also:
            subscribe: Add a subscription
        """
        self._unsubscribe_direct_callback(func)

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
