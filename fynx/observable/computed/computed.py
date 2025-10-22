"""
FynX Observable Computed - Computed Observable Implementation
===========================================================

This module provides the ComputedObservable class, a read-only observable that
derives its value from other observables through automatic computation.

What are Computed Observables?
------------------------------

Computed observables are read-only reactive values that automatically calculate
their value based on other observables. They provide derived state without manual
synchronization, ensuring that computed values always stay in sync with their inputs.

Key characteristics:
- **Read-only**: Cannot be set directly (prevents accidental mutation)
- **Automatic Updates**: Recalculates when dependencies change
- **Lazy Evaluation**: Only computes when accessed
- **Dependency Tracking**: Framework tracks what observables are used
- **Type Safety**: Compile-time distinction from regular observables

When to Use Computed Observables
---------------------------------

Use computed observables when you need:
- **Derived State**: Values that depend on other reactive values
- **Calculated Properties**: Mathematical or logical transformations
- **Data Formatting**: Converting raw data to display formats
- **Validation Results**: Computed validation states
- **Aggregations**: Summing, counting, or combining multiple values

Creating Computed Observables
-----------------------------

While you can create ComputedObservable instances directly, it's more common to use
the `>>` operator or `.then()` method which handles the reactive setup automatically:

```python
from fynx import observable

# Base observables
price = observable(10.0)
quantity = observable(5)

# Computed observable using the >> operator (modern approach)
total = (price + quantity) >> (lambda p, q: p * q)
print(total.value)  # 50.0

# Alternative using .then() method
total_alt = (price + quantity).then(lambda p, q: p * q)
print(total_alt.value)  # 50.0
```

Read-Only Protection
--------------------

Computed observables prevent accidental direct modification:

```python
total = (price + quantity) >> (lambda p, q: p * q)

# This works - updates automatically
price.set(15)
print(total.value)  # 75.0

# This raises ValueError
total.set(100)  # ValueError: Computed observables are read-only
```

Internal Updates
----------------

The framework can update computed values through the internal `_set_computed_value()` method:

```python
# This is used internally by the >> operator and .then() method
computed_obs._set_computed_value(new_value)  # Allowed
computed_obs.set(new_value)                  # Not allowed
```

Performance Considerations
--------------------------

- **Lazy Evaluation**: Values only recalculate when accessed after dependencies change
- **Caching**: Results are cached until dependencies actually change
- **Dependency Tracking**: Only tracks observables actually accessed during computation
- **Memory Efficient**: Minimal overhead beyond regular observables

Common Patterns
---------------

**Mathematical Computations**:
```python
width = observable(10)
height = observable(20)
area = (width + height) >> (lambda w, h: w * h)
perimeter = (width + height) >> (lambda w, h: 2 * (w + h))
```

**String Formatting**:
```python
first_name = observable("John")
last_name = observable("Doe")
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")
```

**Validation States**:
```python
email = observable("")
is_valid_email = email >> (lambda e: "@" in e and len(e) > 5)
```

**Conditional Computations**:
```python
count = observable(0)
is_even = count >> (lambda c: c % 2 == 0)
```

Limitations
-----------

- Cannot be set directly (by design)
- Dependencies must be accessed synchronously during computation
- Cannot depend on external state that changes independently
- Computation functions should be pure (no side effects)

Error Handling
--------------

Computed observables handle errors gracefully:
- Computation errors are logged but don't break the reactive system
- Failed computations may result in stale values
- Dependencies continue to work normally even if one computation fails

See Also
--------

- `fynx.observable`: The >> operator and .then() method for creating computed observables
- `fynx.observable`: Core observable classes
- `fynx.store`: For organizing observables in reactive containers
"""

import weakref
from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar

# Import at runtime for actual usage
from ..primitives.base import Observable


# Lazy imports to avoid circular dependencies
def _get_lazy_chain_builder():
    from ...util import LazyChainBuilder

    return LazyChainBuilder


def _get_find_ultimate_source():
    from ...util import find_ultimate_source

    return find_ultimate_source


T = TypeVar("T")

# Global intern pool for structural sharing
_intern_pool = None


class ComputedObservable(Observable[T]):
    """
    A read-only observable that derives its value from other observables.

    ComputedObservable is a subclass of Observable that represents computed/derived
    values. Unlike regular observables, computed observables are read-only and cannot
    be set directly - their values are automatically calculated from their dependencies.

    This implementation includes function composition optimization that combines
    chains of .then() operations into a single composed function, reducing
    overhead for transformation chains.

    Performance characteristics:
        - Chain composition reduces intermediate observables
        - Memory usage scales with chain complexity
        - Updates propagate through composed functions
        - Single subscription per composed chain

    Example:
        ```python
        # Regular observable
        counter = observable(0)

        # Computed observable (read-only)
        doubled = ComputedObservable("doubled", lambda: counter.value * 2)
        doubled.set(10)  # Raises ValueError: Computed observables are read-only

        # Function composition applies to chains:
        # source.then(f).then(g).then(h) → source.then(h ∘ g ∘ f)
        ```
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        computation_func: Optional[Callable] = None,
        source_observable: Optional["Observable"] = None,
    ) -> None:
        super().__init__(key, initial_value)

        # Original fields
        self._computation_func = computation_func
        self._source_observable = source_observable

        # FUNCTION COMPOSITION: Track transformation chain for optimization
        # This enables automatic collapsing of .then().then().then() chains
        self._transformation_chain: Optional[Any] = None

        # If we have a computation function, start a chain
        if computation_func is not None:
            self._transformation_chain = (computation_func,)

        # Track if this is part of an optimized chain
        self._is_composition_optimized = False

        # If we have a source observable, subscribe to it for updates
        if source_observable is not None:
            # Add dependency edge to cycle detector
            from ..primitives.context import ReactiveContextImpl

            cycle_detector = ReactiveContextImpl._get_cycle_detector()
            cycle_detector.add_edge(source_observable, self)

            def on_source_change(value):
                # Evaluate immediately and notify if value changed
                if self._computation_func is not None:
                    # Get current value and compute new value
                    old_value = self._value_wrapper.unwrap()
                    source_value = self._source_observable.value

                    # Mark source as being computed from to detect circular dependencies
                    was_computed_from = getattr(
                        self._source_observable, "_computed_from", None
                    )
                    self._source_observable._computed_from = self

                    try:
                        computed_value = self._computation_func(source_value)
                    finally:
                        # Restore previous state
                        if was_computed_from is not None:
                            self._source_observable._computed_from = was_computed_from
                        else:
                            delattr(self._source_observable, "_computed_from")

                    # Update stored value
                    self._value_wrapper._value = computed_value
                    self._is_dirty = False

                    # Notify observers if the value changed
                    if old_value != computed_value:
                        self._notify_observers(computed_value)
                else:
                    # Lazy evaluation: just mark as dirty
                    self._is_dirty = True

            source_observable.subscribe(on_source_change)

        # Cache the composed function for performance
        self._composed_func: Optional[Callable] = computation_func

        # Cache the ultimate source to avoid O(N) chain walking
        self._cached_ultimate_source: Optional["Observable"] = None

        # Flag to prevent recursion during updates
        self._is_updating = False

        # Lazy evaluation: track if value needs recomputation
        self._is_dirty = True

    def _evaluate_and_notify(self):
        """Evaluate the computation and notify observers if value changed."""
        if (
            not self._is_dirty
            or self._computation_func is None
            or self._source_observable is None
        ):
            return

        # Get current value and compute new value
        old_value = self._value_wrapper.unwrap()
        source_value = self._get_source_value_iteratively()

        # Mark source as being computed from to detect circular dependencies
        was_computed_from = getattr(self._source_observable, "_computed_from", None)
        self._source_observable._computed_from = self

        try:
            computed_value = self._computation_func(source_value)
        finally:
            # Restore previous state
            if was_computed_from is not None:
                self._source_observable._computed_from = was_computed_from
            else:
                delattr(self._source_observable, "_computed_from")

        # Update stored value
        self._value_wrapper._value = computed_value
        self._is_dirty = False

        # Notify observers if the value changed
        if old_value != computed_value:
            self._notify_observers(computed_value)

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating computed observable values.

        If this is part of an optimized chain, updates propagate through
        the composed function.

        Warning:
            This method should only be called by the FynX framework internals.
            Direct use may break reactive relationships and is not supported.

        Args:
            value: The new computed value calculated from dependencies.
                  Can be any type that the computed function returns.
        """
        # Mark as clean to prevent re-evaluation
        self._is_dirty = False
        # Always notify observers when computed value changes
        super().set(value)

    def _extend_chain(self, func: Callable) -> "LazyChainBuilder":
        """
        Extend chain with lazy building approach.

        Returns a LazyChainBuilder that builds the entire chain as pure data,
        then compiles it ONCE when materialized.
        """

        # Find the ultimate source observable
        source = _get_find_ultimate_source()(self)

        # Build the chain of functions
        functions = []

        # If we have a composed function, add it to the chain
        if self._composed_func is not None:
            functions.append(self._composed_func)

        # Add the new function
        functions.append(func)

        # Return a lazy chain builder
        return _get_lazy_chain_builder()(source, functions)

    @property
    def value(self):
        """
        Get the current value with lazy evaluation.

        For lazy computed observables, materializes the chain on first access.
        For regular computed observables, only recomputes when dirty.
        """
        # Track dependency if we're in a reactive context
        if Observable._current_context is not None:
            Observable._current_context.add_dependency(self)

        # If this is a lazy computed observable, materialize on first access
        if getattr(self, "_is_lazy", False):
            materialized = self._lazy_builder.materialize()
            # Transfer the materialized observable's properties to self
            self._value_wrapper = materialized._value_wrapper
            self._computation_func = materialized._computation_func
            self._source_observable = materialized._source_observable
            self._is_updating = materialized._is_updating
            self._is_dirty = materialized._is_dirty
            # Remove lazy flags
            delattr(self, "_is_lazy")
            delattr(self, "_lazy_builder")

        # Only recompute if dirty
        elif (
            self._is_dirty
            and self._computation_func is not None
            and self._source_observable is not None
        ):
            self._evaluate_and_notify()

        # Return the current value (either cached or newly computed)
        return super().value

    def _get_source_value_iteratively(self) -> any:
        """Get source value iteratively to avoid deep recursion in long chains."""
        # Build the chain of computed observables that need evaluation
        chain = []
        current = self._source_observable

        # Traverse up the chain collecting computed observables that need evaluation
        while (
            hasattr(current, "_is_dirty")
            and current._is_dirty
            and hasattr(current, "_computation_func")
            and current._computation_func is not None
            and hasattr(current, "_source_observable")
            and current._source_observable is not None
        ):
            chain.append(current)
            current = current._source_observable

        # If chain is empty, just return the base value
        if not chain:
            return self._source_observable.value

        # Evaluate from bottom to top (base to derived)
        for computed_obs in reversed(chain):
            # Mark source as being computed from
            was_computed_from = getattr(
                computed_obs._source_observable, "_computed_from", None
            )
            computed_obs._source_observable._computed_from = computed_obs

            try:
                source_value = computed_obs._source_observable.value
                old_value = computed_obs._value_wrapper.unwrap()
                computed_value = computed_obs._computation_func(source_value)

                # Update stored value and mark as clean
                computed_obs._value_wrapper._value = computed_value
                computed_obs._is_dirty = False

                # Don't notify here - we'll notify at the top level
            finally:
                # Restore previous state
                if was_computed_from is not None:
                    computed_obs._source_observable._computed_from = was_computed_from
                else:
                    delattr(computed_obs._source_observable, "_computed_from")

        # Return the value of our direct source (which should now be evaluated)
        return self._source_observable.value

    def subscribe(self, callback):
        """Subscribe to changes (no lazy initialization needed)."""
        return super().subscribe(callback)

    def then(self, func: Callable) -> "ComputedObservable":
        """
        Override then() to use LazyChainBuilder for chaining optimization.

        CRITICAL FIX: Return LazyChainBuilder directly to avoid eager materialization.
        Only materialize when actually needed (value access, set operations, etc.)
        """
        from ...util import LazyChainBuilder

        # Create a wrapper function that handles tuple unpacking for merged observables
        def tuple_aware_func(value):
            # Check if this observable produces tuples (i.e., it's mergeable)
            from ..protocols.merged_protocol import Mergeable

            if isinstance(self, Mergeable) and isinstance(value, tuple):
                # This is a merged observable producing a tuple - unpack it
                return func(*value)
            else:
                # Single value observable
                return func(value)

        # Create a regular computed observable (disable lazy optimization for now)
        from ..operations import OperationsMixin

        return OperationsMixin.then(self, func)

    def _find_ultimate_source(self) -> "Observable":
        """
        Find the ultimate non-computed source observable with O(1) caching.

        Uses cached result to avoid O(N) chain walking on repeated calls.
        Only walks the chain once and caches the result.

        Returns:
            The root source observable
        """
        # Return cached result if available
        if self._cached_ultimate_source is not None:
            return self._cached_ultimate_source

        # Walk the chain once and cache the result
        current = self._source_observable

        # Keep unwrapping computed observables
        while (
            isinstance(current, ComputedObservable)
            and current._source_observable is not None
        ):
            current = current._source_observable

        # Cache the result for future calls
        self._cached_ultimate_source = current if current is not None else self

        return self._cached_ultimate_source

    @staticmethod
    def _compose_chain(funcs: List[Callable]) -> Callable:
        """
        Compose a list of functions into a single function.

        Implements function composition: (f ∘ g)(x) = f(g(x))
        For a chain [f, g, h], creates: x → h(g(f(x)))

        Args:
            funcs: List of functions to compose (applied left-to-right)

        Returns:
            Single composed function
        """

        # Create composed function for pure composition
        def composed_func(x):
            result = x
            for f in funcs:
                result = f(result)
            return result

        # Cache function name for debugging
        composed_func.__name__ = f"composed_{'_'.join(f.__name__ if hasattr(f, '__name__') else 'lambda' for f in funcs)}"

        return composed_func

    def get_optimization_stats(self) -> dict:
        """
        Get statistics about function composition optimizations applied to this observable.

        Returns:
            Dictionary with optimization metrics:
            - is_optimized: Whether function composition optimization is active
            - chain_length: Number of composed transformations
            - functions_collapsed: Original functions replaced by single composition
            - memory_saved_estimate: Estimated observables eliminated
            - subscription_reduction: Number of subscriptions saved
        """
        if not self._is_composition_optimized:
            chain_len = (
                len(self._transformation_chain.transformations)
                if self._transformation_chain
                else 0
            )
            return {
                "is_optimized": False,
                "chain_length": chain_len,
                "functions_collapsed": 0,
                "memory_saved_estimate": 0,
                "subscription_reduction": 0,
            }

        chain_len = (
            len(self._transformation_chain.transformations)
            if self._transformation_chain
            else 0
        )

        return {
            "is_optimized": True,
            "chain_length": chain_len,
            "functions_collapsed": chain_len,
            # Each intermediate observable in the chain is eliminated
            "memory_saved_estimate": max(0, chain_len - 1),
            # Each intermediate subscription is eliminated
            "subscription_reduction": max(0, chain_len - 1),
            # Show the composed function signature
            "composed_function": (
                self._composed_func.__name__ if self._composed_func else None
            ),
        }

    def set(self, value: Optional[T]) -> None:
        """
        Prevent direct modification of computed observable values.

        Computed observables are read-only by design because their values are
        automatically calculated from other observables. Attempting to set them
        directly would break the reactive relationship and defeat the purpose
        of computed values.

        To create a computed observable, use the >> operator or .then() method instead:

        ```python
        from fynx import observable

        base = observable(5)
        # Correct: Create computed value
        doubled = base >> (lambda x: x * 2)

        # Incorrect: Try to set computed value directly
        doubled.set(10)  # Raises ValueError
        ```

        Args:
            value: The value that would be set (ignored).
                  This parameter exists for API compatibility but is not used.

        Raises:
            ValueError: Always raised to prevent direct modification of computed values.
                       Use the >> operator or .then() method to create derived observables instead.

        See Also:
            >> operator: Modern syntax for creating computed observables
            _set_computed_value: Internal method used by the framework
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )
