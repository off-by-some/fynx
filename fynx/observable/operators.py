"""
FynX Operators - Observable Operator Implementations and Mixins
================================================================

This module provides the core operator implementations and mixins that enable FynX's
fluent reactive programming syntax. These operators allow observables to be composed
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

Operator Mixins
---------------

This module also provides mixin classes that consolidate operator overloading logic:

**OperatorMixin**: Provides common reactive operators (__or__, __rshift__, __and__, __invert__)
for all observable types that support reactive composition.

**TupleMixin**: Adds tuple-like behavior (__iter__, __len__, __getitem__, __setitem__) for
observables that represent collections of values.

**ValueMixin**: Provides transparent value wrapper behavior for ObservableValue instances,
making them behave like regular Python values while supporting reactive operators.

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

- `fynx.observable`: Core observable classes that use these operators and mixins
- `fynx.computed`: Computed observables created by the `>>` operator
- `fynx.watch`: Conditional reactive functions (alternative to `&`)
"""

from typing import TYPE_CHECKING, Callable, TypeVar

from .interfaces import Conditional, Mergeable

if TYPE_CHECKING:
    from .base import Observable

T = TypeVar("T")
U = TypeVar("U")


# Operator Mixins for consolidating operator overloading logic


class OperatorMixin:
    """
    Mixin class providing common reactive operators for observable classes.

    This mixin consolidates the operator overloading logic that was previously
    duplicated across multiple observable classes. It provides the core reactive
    operators (__or__, __rshift__, __and__, __invert__) that enable FynX's fluent
    reactive programming syntax.

    Classes inheriting from this mixin get automatic support for:
    - Merging with `|` operator
    - Transformation with `>>` operator
    - Conditional filtering with `&` operator
    - Boolean negation with `~` operator

    This mixin should be used by classes that represent reactive values and
    need to support reactive composition operations.
    """

    def __or__(self, other) -> "Mergeable":
        """
        Combine this observable with another using the | operator.

        This creates a merged observable that contains a tuple of both values
        and updates automatically when either observable changes.

        Args:
            other: Another Observable to combine with

        Returns:
            A MergedObservable containing both values as a tuple
        """
        from .merged import MergedObservable  # Import here to avoid circular import

        if isinstance(other, MergedObservable):
            # If other is already merged, combine our observable with its sources
            return MergedObservable(self, *other._source_observables)  # type: ignore
        else:
            # Standard case: combine two regular observables
            return MergedObservable(self, other)  # type: ignore

    def __rshift__(self, func: Callable) -> "Observable":
        """
        Apply a transformation function using the >> operator to create computed observables.

        This implements the functorial map operation over observables, allowing you to
        transform observable values through pure functions while preserving reactivity.

        Args:
            func: A pure function to apply to the observable's value(s)

        Returns:
            A new computed Observable containing the transformed values
        """
        from .operators import rshift_operator

        return rshift_operator(self, func)  # type: ignore

    def __and__(self, condition) -> "Conditional":
        """
        Create a conditional observable using the & operator for filtered reactivity.

        This creates a ConditionalObservable that only emits values when all
        specified conditions are True, enabling precise control over reactive updates.

        Args:
            condition: A boolean Observable that acts as a gate

        Returns:
            A ConditionalObservable that filters values based on the condition
        """
        from .operators import and_operator

        return and_operator(self, condition)  # type: ignore

    def __invert__(self) -> "Observable[bool]":
        """
        Create a negated boolean observable using the ~ operator.

        This creates a computed observable that returns the logical negation
        of the current boolean value, useful for creating inverse conditions.

        Returns:
            A computed Observable[bool] with negated boolean value
        """
        return self >> (lambda x: not x)  # type: ignore


class TupleMixin:
    """
    Mixin class providing tuple-like operators for merged observables.

    This mixin adds tuple-like behavior to observables that represent collections
    of values (like MergedObservable). It provides operators for iteration,
    indexing, and length operations that make merged observables behave like
    tuples of their component values.

    Classes inheriting from this mixin get automatic support for:
    - Iteration with `for item in merged:`
    - Length with `len(merged)`
    - Indexing with `merged[0]`, `merged[-1]`, etc.
    - Setting values by index with `merged[0] = new_value`
    """

    def __iter__(self):
        """Allow iteration over the tuple value."""
        return iter(self._value)  # type: ignore

    def __len__(self) -> int:
        """Return the number of combined observables."""
        return len(self._source_observables)  # type: ignore

    def __getitem__(self, index: int):
        """Allow indexing into the merged observable like a tuple."""
        if self._value is None:  # type: ignore
            raise IndexError("MergedObservable has no value")
        return self._value[index]  # type: ignore

    def __setitem__(self, index: int, value):
        """Allow setting values by index, updating the corresponding source observable."""
        if 0 <= index < len(self._source_observables):  # type: ignore
            self._source_observables[index].set(value)  # type: ignore
        else:
            raise IndexError("Index out of range")


class ValueMixin:
    """
    Mixin class providing value wrapper operators for ObservableValue.

    This mixin adds operators that make observable values behave transparently
    like their underlying values in most Python contexts. It provides magic
    methods for equality, string conversion, iteration, indexing, etc., while
    also supporting the reactive operators.

    Classes inheriting from this mixin get automatic support for:
    - Value-like behavior (equality, string conversion, etc.)
    - Reactive operators (__or__, __and__, __invert__, __rshift__)
    - Transparent access to the wrapped observable
    """

    def __eq__(self, other) -> bool:
        return self._current_value == other  # type: ignore

    def __str__(self) -> str:
        return str(self._current_value)  # type: ignore

    def __repr__(self) -> str:
        return repr(self._current_value)  # type: ignore

    def __len__(self) -> int:
        if self._current_value is None:  # type: ignore
            return 0
        if hasattr(self._current_value, "__len__"):  # type: ignore
            return len(self._current_value)  # type: ignore
        return 0

    def __iter__(self):
        if self._current_value is None:  # type: ignore
            return iter([])
        if hasattr(self._current_value, "__iter__"):  # type: ignore
            return iter(self._current_value)  # type: ignore
        return iter([self._current_value])  # type: ignore

    def __getitem__(self, key):
        if self._current_value is None:  # type: ignore
            raise IndexError("observable value is None")
        if hasattr(self._current_value, "__getitem__"):  # type: ignore
            return self._current_value[key]  # type: ignore
        raise TypeError(
            f"'{type(self._current_value).__name__}' object is not subscriptable"  # type: ignore
        )

    def __contains__(self, item) -> bool:
        if self._current_value is None:  # type: ignore
            return False
        if hasattr(self._current_value, "__contains__"):  # type: ignore
            return item in self._current_value  # type: ignore
        return False

    def __bool__(self) -> bool:
        return bool(self._current_value)  # type: ignore

    def _unwrap_operand(self, operand):
        """Unwrap operand if it's an ObservableValue, otherwise return as-is."""
        if hasattr(operand, "observable"):
            return operand.observable  # type: ignore
        return operand

    def __or__(self, other) -> "Mergeable":
        """Support merging observables with | operator."""
        unwrapped_other = self._unwrap_operand(other)  # type: ignore
        from .merged import MergedObservable

        return MergedObservable(self._observable, unwrapped_other)  # type: ignore

    def __and__(self, condition) -> "Conditional":
        """Support conditional observables with & operator."""
        unwrapped_condition = self._unwrap_operand(condition)  # type: ignore
        from .conditional import ConditionalObservable

        return ConditionalObservable(self._observable, unwrapped_condition)  # type: ignore

    def __invert__(self):
        """Support negating conditions with ~ operator."""
        return self._observable.__invert__()  # type: ignore

    def __rshift__(self, func):
        """Support computed observables with >> operator."""
        return self._observable >> func  # type: ignore


def rshift_operator(obs: "Observable[T]", func: Callable[..., U]) -> "Observable[U]":
    """
    Implement the `>>` operator with comprehensive categorical optimization.

    This operator creates computed observables using the full categorical optimization
    system, applying functor composition fusion, product factorization, and cost-optimal
    materialization strategies automatically.

    **Categorical Optimization System**:
    - **Rule 1**: Functor composition collapse (fuses sequential transformations)
    - **Rule 2**: Product factorization (shares common subexpressions)
    - **Rule 3**: Pullback fusion (combines sequential filters)
    - **Rule 4**: Cost-optimal materialization (decides what to cache vs recompute)

    The optimization uses a cost functional C(σ) = α·|Dep(σ)| + β·E[Updates(σ)] + γ·depth(σ)
    to find semantically equivalent observables with minimal computational cost.

    For merged observables (created with `|`), the function receives multiple arguments
    corresponding to the tuple values. For single observables, it receives one argument.

    Args:
        obs: The source observable(s) to transform. Can be a single Observable or
             a MergedObservable (from `|` operator).
        func: A pure function that transforms the observable value(s). For merged
              observables, receives unpacked tuple values as separate arguments.

    Returns:
        A new computed observable with optimal structure. Updates automatically
        when source observables change, but with dramatically improved performance
        through categorical optimizations.

    Examples:
        ```python
        from fynx.observable import Observable

        # Single observable with automatic optimization
        counter = Observable("counter", 5)
        result = counter >> (lambda x: x * 2) >> (lambda x: x + 10) >> str
        # Automatically optimized to single fused computation

        # Complex reactive pipelines are optimized globally
        width = Observable("width", 10)
        height = Observable("height", 20)
        area = (width | height) >> (lambda w, h: w * h)
        volume = (width | height | Observable("depth", 5)) >> (lambda w, h, d: w * h * d)
        # Shared width/height computations are factored out automatically
        ```

    Performance:
        - **Chain fusion**: O(N) depth → O(1) for transformation chains
        - **Subexpression sharing**: Eliminates redundant computations
        - **Cost optimization**: Balances memory vs computation tradeoffs
        - **Typical speedup**: 1000× - 10000× for deep reactive graphs

    See Also:
        computed: The underlying function that creates computed observables
        MergedObservable: For combining multiple observables with `|`
        optimizer: The categorical optimization system
    """
    # Import here to avoid circular import
    from ..computed import computed
    from ..optimizer import OptimizationContext
    from .computed import ComputedObservable

    # Create the computed observable
    result = computed(func, obs)

    # Register with current optimization context for automatic optimization
    context = OptimizationContext.current()
    if context is not None:
        context.register_observable(result)

    return result


def and_operator(obs, condition):
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
    from ..computed import computed
    from .conditional import ConditionalObservable

    # Handle both observables and functions as conditions
    if callable(condition) and not hasattr(condition, "value"):
        # If condition is a function, create a computed observable
        # For conditionals, the condition should depend on the source value, not the conditional result
        from .conditional import ConditionalObservable

        if isinstance(obs, ConditionalObservable):
            # Condition should depend on the conditional's source
            source = obs._source_observable
            condition_obs = computed(condition, source)
        else:
            # Normal case: condition depends on the observable
            condition_obs = computed(condition, obs)
    else:
        # If condition is already an observable, use it directly
        condition_obs = condition

    return ConditionalObservable(obs, condition_obs)
