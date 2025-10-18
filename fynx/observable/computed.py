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
the `computed()` function which handles the reactive setup automatically:

```python
from fynx import observable, computed

# Base observables
price = observable(10.0)
quantity = observable(5)

# Computed observable using the computed() function
total = computed(lambda p, q: p * q, price | quantity)
print(total.value)  # 50.0

# Direct creation (less common)
from fynx.observable.computed import ComputedObservable
manual = ComputedObservable("manual", 42)
```

Read-Only Protection
--------------------

Computed observables prevent accidental direct modification:

```python
total = computed(lambda p, q: p * q, price | quantity)

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
# This is used internally by the computed() function
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
area = computed(lambda w, h: w * h, width | height)
perimeter = computed(lambda w, h: 2 * (w + h), width | height)
```

**String Formatting**:
```python
first_name = observable("John")
last_name = observable("Doe")
full_name = computed(
    lambda f, l: f"{f} {l}",
    first_name | last_name
)
```

**Validation States**:
```python
email = observable("")
is_valid_email = computed(
    lambda e: "@" in e and len(e) > 5,
    email
)
```

**Conditional Computations**:
```python
count = observable(0)
is_even = computed(lambda c: c % 2 == 0, count)
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

- `fynx.computed`: The computed() function for creating computed observables
- `fynx.observable`: Core observable classes
- `fynx.store`: For organizing observables in reactive containers
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

        This method is called internally by the computed() function when dependencies
        change. It bypasses the read-only protection enforced by the public set() method
        to allow legitimate framework-driven updates of computed values.

        Warning:
            This method should only be called by the FynX framework internals.
            Direct use may break reactive relationships and is not supported.

        Args:
            value: The new computed value calculated from dependencies.
                  Can be any type that the computed function returns.
        """
        super().set(value)

    def set(self, value: Optional[T]) -> None:
        """
        Prevent direct modification of computed observable values.

        Computed observables are read-only by design because their values are
        automatically calculated from other observables. Attempting to set them
        directly would break the reactive relationship and defeat the purpose
        of computed values.

        To create a computed observable, use the `computed()` function instead:

        ```python
        from fynx import observable, computed

        base = observable(5)
        # Correct: Create computed value
        doubled = computed(lambda x: x * 2, base)

        # Incorrect: Try to set computed value directly
        doubled.set(10)  # Raises ValueError
        ```

        Args:
            value: The value that would be set (ignored).
                  This parameter exists for API compatibility but is not used.

        Raises:
            ValueError: Always raised to prevent direct modification of computed values.
                       Use the `computed()` function to create derived observables instead.

        See Also:
            computed: Function for creating computed observables
            _set_computed_value: Internal method used by the framework
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )
