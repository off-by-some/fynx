"""
FynX Operators - Observable Operator Implementations
===================================================

This module provides the core operator implementations that enable FynX's fluent
reactive programming syntax. These operators allow observables to be composed
using intuitive Python operators, creating complex reactive behaviors from simple
building blocks.

Why Operators?
--------------

FynX uses Python's operator overloading to provide a natural, readable syntax
for reactive programming. Instead of verbose method calls, you can express
reactive relationships using familiar operators:

- `observable >> function` - Transform values reactively
- `observable & condition` - Filter values conditionally
- `obs1 | obs2 | obs3` - Combine observables

This approach makes reactive code more declarative and easier to understand.

Operator Overview
-----------------

**Transform (`>>`)**: Apply functions to create derived values
```python
doubled = counter >> (lambda x: x * 2)
```

**Filter (`&`)**: Only emit values when conditions are met
```python
valid_data = data & is_valid
```

**Combine (`|`)**: Merge multiple observables into tuples
```python
coordinates = x | y | z
```

These operators work together to create complex reactive pipelines:
```python
result = (x | y) >> (lambda a, b: a + b) & (total >> (lambda t: t > 10))
```

Implementation Details
----------------------

The operators are implemented as standalone functions rather than methods
to avoid circular import issues and enable lazy loading. They are called
automatically when you use the corresponding operators on Observable instances.

The functions handle different observable types (regular, merged, conditional)
appropriately, ensuring consistent behavior across the reactive system.

Performance Characteristics
---------------------------

- **Lazy Evaluation**: Operators create computed/conditional observables that
  only evaluate when needed
- **Efficient Composition**: Multiple operators can be chained without
  creating intermediate objects
- **Memory Conscious**: Operators reuse existing infrastructure rather than
  creating new classes

Common Patterns
---------------

**Data Processing Pipeline**:
```python
raw_data = observable([])
processed = (raw_data
    >> (lambda d: [x for x in d if x > 0])  # Filter positive values
    >> (lambda d: sorted(d))                # Sort results
    >> (lambda d: sum(d) / len(d) if d else 0))  # Calculate average
```

**Conditional UI Updates**:
```python
user_input = observable("")
is_valid = user_input >> (lambda s: len(s) >= 3)
show_error = user_input & ~is_valid  # Show error when input is invalid but not empty
```

**Reactive Calculations**:
```python
price = observable(10.0)
quantity = observable(1)
tax_rate = observable(0.08)

subtotal = (price | quantity) >> (lambda p, q: p * q)
tax = subtotal >> (lambda s: s * tax_rate.value)
total = (subtotal | tax) >> (lambda s, t: s + t)
```

Error Handling
--------------

Operators handle errors gracefully:
- Transformation function errors are propagated but don't break the reactive system
- Invalid operator usage provides clear error messages
- Circular dependencies are detected and prevented

Best Practices
--------------

- **Keep Functions Pure**: Transformation functions should not have side effects
- **Use Meaningful Lambdas**: Complex operations deserve named functions
- **Chain Thoughtfully**: Break complex chains into intermediate variables for clarity
- **Handle Edge Cases**: Consider what happens with None, empty collections, etc.

Migration from Method Calls
---------------------------

If you're familiar with other reactive libraries, here's how FynX operators compare:

```python
# Other libraries (method-based)
result = obs.map(lambda x: x * 2).filter(lambda x: x > 10)

# FynX (operator-based)
result = obs >> (lambda x: x * 2) & (obs >> (lambda x: x > 10))
```

The operator syntax is more concise and readable for simple transformations.

See Also
--------

- `fynx.observable`: Core observable classes that use these operators
- `fynx.computed`: Computed observables created by the `>>` operator
- `fynx.watch`: Conditional reactive functions (alternative to `&`)
"""

from typing import TYPE_CHECKING, Callable, TypeVar

from .base import Observable

if TYPE_CHECKING:
    from .conditional import ConditionalObservable

T = TypeVar("T")
U = TypeVar("U")


def rshift_operator(obs: Observable[T], func: Callable[..., U]) -> Observable[U]:
    """
    Implement the `>>` operator for creating computed observables.

    This operator enables the functorial map operation over observables, transforming
    observable values through pure functions while preserving reactivity. The resulting
    computed observable automatically updates whenever the source observable changes.

    For merged observables (created with `|`), the function receives multiple arguments
    corresponding to the tuple values. For single observables, it receives one argument.

    Args:
        obs: The source observable(s) to transform. Can be a single Observable or
             a MergedObservable (from `|` operator).
        func: A pure function that transforms the observable value(s). For merged
              observables, receives unpacked tuple values as separate arguments.

    Returns:
        A new computed observable containing the transformed values. Updates automatically
        when source observables change.

    Examples:
        ```python
        from fynx.observable import Observable

        # Single observable transformation
        counter = Observable("counter", 5)
        doubled = counter >> (lambda x: x * 2)  # ComputedObservable with value 10

        # Merged observable transformation
        width = Observable("width", 10)
        height = Observable("height", 20)
        area = (width | height) >> (lambda w, h: w * h)  # ComputedObservable with value 200

        # Function chaining
        result = counter >> (lambda x: x + 1) >> str >> (lambda s: f"Count: {s}")
        # Result: "Count: 6"
        ```

    Note:
        The transformation function should be pure (no side effects) and relatively
        fast, as it may be called frequently when dependencies change.

    See Also:
        computed: The underlying function that creates computed observables
        MergedObservable: For combining multiple observables with `|`
    """
    # Import here to avoid circular import
    from ..computed import computed

    return computed(func, obs)


def and_operator(
    obs: Observable[T], condition: Observable[bool]
) -> "ConditionalObservable[T]":
    """
    Implement the `&` operator for creating conditional observables.

    This operator creates conditional observables that only emit values when boolean
    conditions are satisfied. The resulting observable filters the reactive stream,
    preventing unnecessary updates and computations when conditions aren't met.

    Args:
        obs: The source observable whose values will be conditionally emitted.
        condition: A boolean observable that acts as a gate. Values from `obs`
                  are only emitted when this condition is True.

    Returns:
        A new ConditionalObservable that only emits values when the condition is met.
        The observable starts with None if the condition is initially False.

    Examples:
        ```python
        from fynx.observable import Observable

        # Basic conditional filtering
        data = Observable("data", "hello")
        is_ready = Observable("ready", False)

        filtered = data & is_ready  # Only emits when is_ready is True

        filtered.subscribe(lambda x: print(f"Received: {x}"))
        data.set("world")      # No output (is_ready is False)
        is_ready.set(True)     # Prints: "Received: world"

        # Multiple conditions (chained)
        user_present = Observable("present", True)
        smart_data = data & is_ready & user_present  # All must be True

        # Practical example: temperature monitoring
        temperature = Observable("temp", 20)
        alarm_enabled = Observable("alarm", True)
        is_critical = Observable("critical", False)

        alarm_trigger = temperature & alarm_enabled & is_critical
        alarm_trigger.subscribe(lambda t: print(f"🚨 Alarm: {t}°C"))
        ```

    Note:
        Multiple conditions can be chained: `obs & cond1 & cond2 & cond3`.
        All conditions must be True for values to be emitted.

    See Also:
        ConditionalObservable: The class that implements conditional behavior
        Observable.__and__: The magic method that calls this operator
    """
    # Import here to avoid circular import
    from .conditional import ConditionalObservable

    return ConditionalObservable(obs, condition)
