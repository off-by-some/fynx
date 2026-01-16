"""
FynX Observable Computed - Computed Observable Implementation
===========================================================

This module provides the ComputedObservable class, a read-only observable that
derives its value from other observables through automatic computation.

Computed observables represent derived state—values calculated from other observables
rather than set directly. When source observables change, computed values recalculate
automatically. That automatic recalculation eliminates manual synchronization: you
declare the relationship, and the framework maintains it.

Computed observables are read-only by design. You cannot set them directly because
their values flow from dependencies. This constraint prevents breaking reactive
relationships—if you could set a computed value, it would diverge from its source.
The framework updates computed values internally when dependencies change, but the
public interface enforces immutability.

Creating Computed Observables
-----------------------------

You create computed observables using the `>>` operator or the `.then()` method.
Both approaches transform source observables through pure functions, creating
derived values that update automatically:

```python
from fynx import observable

# Base observables
price = observable(10.0)
quantity = observable(5)

# Computed observable using the >> operator
total = (price + quantity) >> (lambda p, q: p * q)
print(total.value)  # 50.0

# Alternative using .then() method
total_alt = (price + quantity).then(lambda p, q: p * q)
print(total_alt.value)  # 50.0
```

The `>>` operator applies the function to the observable's value, creating a new
computed observable. For merged observables (created with `+`), the function receives
multiple arguments corresponding to the tuple values. That pattern enables reactive
calculations across multiple inputs without manual coordination.

Read-Only Protection
--------------------

Computed observables prevent direct modification to maintain reactive integrity:

```python
total = (price + quantity) >> (lambda p, q: p * q)

# This works—updates propagate automatically
price.set(15)
print(total.value)  # 75.0

# This raises ValueError—computed values are read-only
total.set(100)  # ValueError: Computed observables are read-only
```

Attempting to set a computed observable raises `ValueError`. That error signals
that you're trying to break the reactive relationship—computed values derive from
dependencies, not direct assignment. To change a computed value, modify its source
observables instead.

Internal Updates
----------------

The framework updates computed values through the internal `_set_computed_value()`
method. This method bypasses the read-only protection to allow legitimate framework-driven
updates when dependencies change:

```python
# This is used internally by the >> operator and .then() method
computed_obs._set_computed_value(new_value)  # Allowed internally
computed_obs.set(new_value)                  # Not allowed—raises ValueError
```

External code should not call `_set_computed_value()` directly. That method exists
for framework internals—the `>>` operator and `.then()` method use it to update
computed values when dependencies change. Direct use may break reactive relationships.

Performance Considerations
--------------------------

Computed observables use lazy evaluation: they recalculate only when accessed after
dependencies change. That strategy avoids unnecessary work—if nothing reads the computed
value, it doesn't recompute. Results are cached until dependencies actually change,
so repeated reads return the cached value without recomputation.

Dependency tracking captures only observables actually accessed during computation.
If your function reads `obs1.value` but not `obs2.value`, only `obs1` becomes a
dependency. That precision enables efficient updates—changes to untracked observables
don't trigger recomputation.

Memory overhead is minimal beyond regular observables. Computed observables store
the computation function and source observable reference, but they reuse the same
observer infrastructure as regular observables.

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

Computed observables cannot be set directly—that constraint is by design, not a
missing feature. Dependencies must be accessed synchronously during computation;
asynchronous access won't be tracked. They cannot depend on external state that
changes independently of observables—only observable values trigger recomputation.

Computation functions should be pure: no side effects, no external state access.
Pure functions ensure that computed values depend only on their observable inputs,
making the reactive system predictable and debuggable.

Error Handling
--------------

When computation functions raise exceptions, those errors propagate normally. The
reactive system doesn't swallow exceptions—if a computation fails, the error
surfaces to the caller. Failed computations may leave computed observables with
stale values until dependencies change and recomputation succeeds.

Dependencies continue working normally even if one computation fails. That isolation
prevents cascading failures—one broken computation doesn't break the entire reactive
graph. You can handle errors in computation functions using try-except blocks if
you need graceful degradation.

See Also
--------

- `fynx.observable`: The >> operator and .then() method for creating computed observables
- `fynx.observable`: Core observable classes
- `fynx.store`: For organizing observables in reactive containers
"""

from typing import Callable, Optional, TypeVar

from .base import Observable

T = TypeVar("T")


class ComputedObservable(Observable[T]):
    """
    A read-only observable that derives its value from other observables.

    ComputedObservable extends Observable to represent computed values—values that
    derive from other observables rather than direct assignment. Unlike regular
    observables, computed observables are read-only: you cannot set them directly
    because their values flow from dependencies.

    This class provides type-based distinction from regular observables. That distinction
    enables compile-time type checking and runtime behavior differences—computed
    observables maintain the same interface as regular observables for reading values
    and subscribing to changes, but they enforce immutability at runtime.

    You typically create computed observables using the `>>` operator or `.then()` method,
    not by direct instantiation. The framework creates ComputedObservable instances
    internally when you use those operators:

    Example:
        ```python
        from fynx import observable

        # Regular observable
        counter = observable(0)

        # Computed observable using >> operator (typical approach)
        doubled = counter >> (lambda x: x * 2)
        print(doubled.value)  # 0

        # Attempting to set a computed observable raises ValueError
        doubled.set(10)  # Raises ValueError: Computed observables are read-only
        ```

    Direct instantiation of ComputedObservable is supported but rarely needed. The
    framework handles creation automatically when you use reactive operators.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        computation_func: Optional[Callable] = None,
        source_observable: Optional["Observable"] = None,
    ) -> None:
        super().__init__(key, initial_value)
        # Store computation function for chain fusion optimization
        self._computation_func = computation_func
        # Store source observable for fusion
        self._source_observable = source_observable

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating computed observable values.

        This method bypasses the read-only protection enforced by the public `set()`
        method. The framework calls it internally when dependencies change—the `>>`
        operator and `.then()` method use it to update computed values after
        recomputation.

        External code should not call this method directly. That restriction prevents
        breaking reactive relationships—only the framework should update computed
        values, and only when dependencies actually change. Direct use may create
        inconsistent state where computed values don't match their dependencies.

        Args:
            value: The new computed value calculated from dependencies.
                  Can be any type that the computed function returns.
        """
        super().set(value)

    def set(self, value: Optional[T]) -> None:
        """
        Prevent direct modification of computed observable values.

        This method always raises `ValueError` to enforce read-only behavior.
        Computed observables derive their values from dependencies—setting them
        directly would break that reactive relationship. The framework updates
        computed values internally when dependencies change, but external code
        cannot modify them.

        To change a computed value, modify its source observables instead. The
        computed value updates automatically when dependencies change:

        ```python
        from fynx import observable

        base = observable(5)
        doubled = base >> (lambda x: x * 2)
        print(doubled.value)  # 10

        # Correct: Modify the source observable
        base.set(6)
        print(doubled.value)  # 12 (updated automatically)

        # Incorrect: Try to set computed value directly
        doubled.set(20)  # Raises ValueError
        ```

        Args:
            value: The value that would be set (ignored).
                  This parameter exists for API compatibility but is not used.

        Raises:
            ValueError: Always raised to prevent direct modification of computed values.
                       Modify source observables instead to update computed values.

        See Also:
            >> operator: Modern syntax for creating computed observables
            _set_computed_value: Internal method used by the framework for updates
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )
