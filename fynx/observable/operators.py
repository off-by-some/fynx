"""
FynX Operators - observable operator implementations and mixins
===================================================================

FynX overloads Python operators so reactive code reads like an expression
rather than a method chain. The operators handle dependency tracking and
updates behind the scenes.

Three mixins provide this:

- `OperatorMixin` gives every observable type the core operators (`+`, `>>`, `&`, `@`, `~`, `|`).
- `TupleMixin` adds tuple-like behavior to merged observables - iteration, indexing, length.
- `ValueMixin` gives ObservableValue transparent value access, so reactive
  attributes behave like regular values while keeping reactive capabilities.

Operator Semantics
------------------

**Transform (`>>`)**: Apply functions to create derived values
```python
from fynx.observable import Observable

counter = Observable("counter", 5)
doubled = counter >> (lambda x: x * 2)
print(doubled.value)  # 10
```

**Boolean AND (`&`)**: Combine boolean conditions
```python
authenticated = observable(True)
connected = observable(True)
loading = observable(False)
ready = authenticated & connected & ~loading
```

**Gate (`@`)**: Only emit values when conditions are met
```python
data = Observable("data", "hello")
is_ready = Observable("ready", False)
filtered = data @ is_ready  # Only emits when is_ready is True
```

**Combine (`+`)**: Merge multiple observables into tuples
```python
x = Observable("x", 1)
y = Observable("y", 2)
z = Observable("z", 3)
coordinates = x + y + z
print(coordinates.value)  # (1, 2, 3)
```

These operators compose to create complex reactive pipelines:
```python
result = ((x + y) >> (lambda a, b: a + b)) @ (total >> (lambda t: t > 10))
```

Implementation Architecture
----------------------------

Operators delegate to OperationsMixin rather than implementing logic
directly, which keeps imports lazy and avoids circular-import issues.
`obs >> func` calls `__rshift__`, which delegates to `obs.then(func)`; `then`
creates a computed observable through `_create_computed`.

Transformation functions receive unpacked tuple values as separate arguments
for merged observables, and a single argument for regular ones.

Performance Characteristics
---------------------------

Operators create computed or conditional observables that evaluate lazily,
recalculating only when accessed after a dependency changes. Chained
operators fuse into a single composed function rather than creating
intermediate objects, and they reuse existing infrastructure instead of new
classes, keeping memory overhead low.

Common Patterns
---------------

**Data Processing Pipeline**:
```python
from fynx import observable

raw_data = observable([1, -2, 3, -4, 5])
processed = (raw_data
    >> (lambda d: [x for x in d if x > 0])  # Filter positive values
    >> (lambda d: sorted(d))                # Sort results
    >> (lambda d: sum(d) / len(d) if d else 0))  # Calculate average
print(processed.value)  # 3.0
```

**Conditional UI Updates**:
```python
user_input = observable("")
is_valid = user_input >> (lambda s: len(s) >= 3)
has_input = user_input >> bool
show_error = user_input @ (has_input & ~is_valid)  # Show error once invalid input exists
```

**Reactive Calculations**:
```python
price = observable(10.0)
quantity = observable(1)
tax_rate = observable(0.08)

subtotal = (price + quantity) >> (lambda p, q: p * q)
tax = (subtotal + tax_rate) >> (lambda s, rate: s * rate)
total = (subtotal + tax) >> (lambda s, t: s + t)
print(total.value)  # 10.8
```

Error Handling
--------------

Errors from transformation functions propagate normally rather than being
swallowed. Invalid operator usage raises TypeError with a descriptive
message. A transform that reads `.value` or calls `.set()` on an observable
raises `TransformPurityError`, with a hint to combine inputs explicitly or
move the effect to a subscription. Circular dependencies are caught during
`.set()` and raise RuntimeError before they can loop.

Best Practices
--------------

Keep transformation functions pure: use only the values passed in as
arguments, with no side effects and no hidden observable reads - combine
every reactive input first with `+` / `.alongside()`. Prefer named functions
over long lambdas for complex operations, and break long chains into
intermediate variables for clarity.

See Also
--------

- `fynx.observable`: Core observable classes that use these operators and mixins
- `fynx.observable.computed`: Computed observables created by the `>>` operator
- `fynx.observable.conditional`: Conditional observables created by the `@` operator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar, overload

from ..equality import values_equal
from ..types import BooleanOperand, ConditionOperand, ObservableOperand
from .operands import is_observable_operand, unwrap_condition, unwrap_observable
from .operations import OperationsMixin

if TYPE_CHECKING:
    from .base import Observable
    from .conditional import ConditionalObservable
    from .descriptors import ObservableValue
    from .interfaces import Conditional, Mergeable
    from .merged import MergedObservable

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Operator Mixins for consolidating operator overloading logic


class OperatorMixin(OperationsMixin[T], Generic[T]):
    """
    Mixin class providing common reactive operators for observable classes.

    Consolidates the operator overloading logic (`__add__`, `__rshift__`,
    `__and__`, `__matmul__`, `__or__`, `__invert__`) that would otherwise be
    duplicated across every observable class.

    Classes inheriting from this mixin get automatic support for:
    - Merging with `+` operator
    - Transformation with `>>` operator
    - Boolean AND with `&` operator
    - Conditional gating with `@` operator
    - Boolean negation with `~` operator

    This mixin should be used by classes that represent reactive values and
    need to support reactive composition operations.
    """

    @overload
    def __add__(self, other: ObservableOperand[U]) -> "MergedObservable[T, U]": ...

    @overload
    def __add__(self, other: Any) -> Any: ...

    def __add__(self, other: Any) -> Any:
        """
        Combine observables, or delegate plain additions to the wrapped value.

        Observable-like operands create a merged observable containing an
        ordered tuple of both values. Plain Python values use the underlying
        value's own `+`, so `items + [new_item]` works for immutable updates.

        Args:
            other: Another Observable to combine with, or a plain value to add

        Returns:
            A MergedObservable for observable operands; otherwise the plain
            Python addition result.
        """
        if is_observable_operand(other):
            return self.alongside(other)  # type: ignore
        return self.value + other  # type: ignore[attr-defined]

    @overload
    def __radd__(self, other: ObservableOperand[U]) -> "MergedObservable[U, T]": ...

    @overload
    def __radd__(self, other: Any) -> Any: ...

    def __radd__(self, other: Any) -> Any:
        """
        Support right-side addition for merging observables.

        This enables expressions like `other + self` to work correctly,
        ensuring that merged observables can be chained properly. Python calls
        this method when the left operand doesn't support `__add__`.

        Args:
            other: Another Observable to combine with

        Returns:
            A MergedObservable containing both values as a tuple
        """
        if is_observable_operand(other):
            return unwrap_observable(other).alongside(self)  # type: ignore
        return other + self.value  # type: ignore[attr-defined]

    def __rshift__(self, func: Callable[[T], U]) -> "Observable[U]":
        """
        Apply a transformation function using the >> operator to create computed observables.

        This implements the functorial map operation over observables, allowing you to
        transform observable values through pure functions while preserving reactivity.
        The operation satisfies the functor laws: identity and composition preservation.

        Args:
            func: A pure function to apply to the observable's value(s)

        Returns:
            A new computed Observable containing the transformed values
        """
        return self.then(func)

    def __and__(self, other: BooleanOperand) -> "Observable[bool]":
        """
        Create a total boolean AND observable using the & operator.

        This creates a computed boolean observable that is True when both
        operands are truthy and False otherwise.

        Args:
            other: Another observable-like boolean value

        Returns:
            A computed Observable[bool] containing the AND result.
        """
        return self.all(other)  # type: ignore

    def __matmul__(self, condition: ConditionOperand[T]) -> "ConditionalObservable[T]":
        """
        Gate this observable using the @ operator for conditional reactivity.

        Creates a ConditionalObservable that only emits this observable's
        values while every condition is True.

        Args:
            condition: A boolean Observable, callable, or compound condition

        Returns:
            A ConditionalObservable that gates values based on the condition.
        """
        return self.requiring(condition)  # type: ignore

    def __invert__(self) -> "Observable[bool]":
        """
        Create a negated boolean observable using the ~ operator.

        This creates a computed observable that returns the logical negation
        of the current boolean value, useful for creating inverse conditions.
        The negation updates automatically when the source changes.

        Returns:
            A computed Observable[bool] with negated boolean value
        """
        return self.negate()  # type: ignore

    def __or__(self, other: ObservableOperand[Any]) -> "Observable[bool]":
        """
        Create a logical OR condition using the | operator.

        This creates a computed boolean observable that is True when either operand is
        truthy and False otherwise. The operation combines boolean observables with
        logical disjunction.

        Args:
            other: Another boolean observable to OR with

        Returns:
            A computed Observable[bool] containing the OR result.
        """
        return self.either(other)  # type: ignore


class TupleMixin:
    """
    Mixin class providing tuple-like operators for merged observables.

    Adds iteration, indexing, and length operators so a MergedObservable
    behaves like a tuple of its component values.

    Classes inheriting from this mixin get automatic support for:
    - Iteration with `for item in merged:`
    - Length with `len(merged)`
    - Indexing with `merged[0]`, `merged[-1]`, etc.
    - Setting values by index with `merged[0] = new_value`
    """

    def __iter__(self):
        """Allow iteration over the tuple value."""
        self._raise_if_transform_reads()  # type: ignore[attr-defined]
        return iter(self._value)  # type: ignore

    def __len__(self) -> int:
        """Return the number of combined observables."""
        self._raise_if_transform_reads()  # type: ignore[attr-defined]
        return len(self._source_observables)  # type: ignore

    def __getitem__(self, index: int):
        """Allow indexing into the merged observable like a tuple."""
        self._raise_if_transform_reads()  # type: ignore[attr-defined]
        if self._value is None:  # type: ignore
            raise IndexError("MergedObservable has no value")
        return self._value[index]  # type: ignore

    def __setitem__(self, index: int, value):
        """Allow setting values by index, updating the corresponding source observable."""
        if 0 <= index < len(self._source_observables):  # type: ignore
            self._source_observables[index].set(value)  # type: ignore
        else:
            raise IndexError("Index out of range")


class ValueMixin(Generic[T]):
    """
    Mixin class providing value wrapper operators for ObservableValue.

    Adds the magic methods (equality, string conversion, iteration, indexing)
    that let an observable value behave like its underlying value in most
    Python contexts, alongside the reactive operators.

    Classes inheriting from this mixin get automatic support for:
    - Value-like behavior (equality, string conversion, etc.)
    - Reactive operators (__add__, __and__, __matmul__, __or__, __invert__, __rshift__)
    - Transparent access to the wrapped observable
    """

    if TYPE_CHECKING:
        _observable: Any

    def _raise_if_transform_reads(self) -> None:
        self._observable._raise_if_transform_reads()  # type: ignore[attr-defined]

    def __eq__(self, other) -> bool:
        self._raise_if_transform_reads()
        return values_equal(self._current_value, other)  # type: ignore

    def __str__(self) -> str:
        self._raise_if_transform_reads()
        return str(self._current_value)  # type: ignore

    def __format__(self, format_spec: str) -> str:
        self._raise_if_transform_reads()
        return format(self._current_value, format_spec)  # type: ignore

    def __repr__(self) -> str:
        self._raise_if_transform_reads()
        return repr(self._current_value)  # type: ignore

    def __len__(self) -> int:
        self._raise_if_transform_reads()
        if self._current_value is None:  # type: ignore
            return 0
        if hasattr(self._current_value, "__len__"):  # type: ignore
            return len(self._current_value)  # type: ignore
        return 0

    def __iter__(self):
        self._raise_if_transform_reads()
        if self._current_value is None:  # type: ignore
            return iter([])
        if hasattr(self._current_value, "__iter__"):  # type: ignore
            return iter(self._current_value)  # type: ignore
        return iter([self._current_value])  # type: ignore

    def __getitem__(self, key):
        self._raise_if_transform_reads()
        if self._current_value is None:  # type: ignore
            raise IndexError("observable value is None")
        if hasattr(self._current_value, "__getitem__"):  # type: ignore
            return self._current_value[key]  # type: ignore
        raise TypeError(
            f"'{type(self._current_value).__name__}' object is not subscriptable"  # type: ignore
        )

    def __contains__(self, item) -> bool:
        self._raise_if_transform_reads()
        if self._current_value is None:  # type: ignore
            return False
        if hasattr(self._current_value, "__contains__"):  # type: ignore
            return item in self._current_value  # type: ignore
        return False

    def __bool__(self) -> bool:
        self._raise_if_transform_reads()
        return bool(self._current_value)  # type: ignore

    def _unwrap_operand(self, operand):
        """Unwrap operand if it's an ObservableValue, otherwise return as-is."""
        from .descriptors import ObservableValue

        if isinstance(operand, ObservableValue):
            return operand.observable  # type: ignore
        return operand

    @overload
    def __add__(self, other: ObservableOperand[U]) -> "MergedObservable[T, U]": ...

    @overload
    def __add__(self: "ValueMixin[list[V]]", other: list[V]) -> list[V]: ...

    @overload
    def __add__(self, other: Any) -> Any: ...

    def __add__(self, other: Any) -> Any:
        """Support merging observables with + operator."""
        from .merged import MergedObservable

        if not is_observable_operand(other):
            return self._current_value + other  # type: ignore

        return MergedObservable.from_sources(  # type: ignore[arg-type, return-value]
            self._observable, unwrap_observable(other)
        )

    @overload
    def __radd__(self, other: ObservableOperand[U]) -> "MergedObservable[U, T]": ...

    @overload
    def __radd__(self: "ValueMixin[list[V]]", other: list[V]) -> list[V]: ...

    @overload
    def __radd__(self, other: Any) -> Any: ...

    def __radd__(self, other: Any) -> Any:
        """Support right-side addition for merging observables."""
        from .merged import MergedObservable

        if not is_observable_operand(other):
            return other + self._current_value  # type: ignore

        return MergedObservable.from_sources(  # type: ignore[arg-type, return-value]
            unwrap_observable(other), self._observable
        )

    def __and__(self, other: BooleanOperand) -> "Observable[bool]":
        """Support boolean AND with & operator."""
        return self._observable.all(other)

    def __matmul__(self, condition: ConditionOperand[T]) -> "ConditionalObservable[T]":
        """Support conditional gating with @ operator."""
        return self._observable.requiring(condition)

    def __invert__(self) -> "Observable[bool]":
        """Support negating conditions with ~ operator."""
        return self._observable.__invert__()

    def __or__(self, other: ObservableOperand[Any]) -> "Observable[bool]":
        """Support logical OR conditions with | operator."""
        unwrapped_other = unwrap_observable(other)
        return self._observable.either(unwrapped_other)

    def __rshift__(self, func: Callable[[T], U]) -> "Observable[U]":
        """Support computed observables with >> operator."""
        return self._observable >> func

    def then(self, func: Callable[[T], U]) -> "Observable[U]":
        """Transform this observable value with a typed natural-language method."""
        return self._observable.then(func)

    def alongside(self, other: ObservableOperand[U]) -> "MergedObservable[T, U]":
        """Combine this observable value with another observable-like value."""
        return self + other

    def requiring(self, *conditions: ConditionOperand[T]) -> "ConditionalObservable[T]":
        """Gate this observable value behind one or more conditions."""
        return self._observable.requiring(*conditions)

    def negate(self) -> "Observable[bool]":
        """Negate this observable value as a boolean condition."""
        return ~self

    def either(self, other: ObservableOperand[Any]) -> "Observable[bool]":
        """Combine this observable value with another condition using OR."""
        return self | other

    def all(self, *others: BooleanOperand) -> "Observable[bool]":
        """Combine this observable value with conditions using AND."""
        return self._observable.all(*others)


def rshift_operator(obs: "Observable[T]", func: Callable[[T], U]) -> "Observable[U]":
    """
    Implement the `>>` operator for pure reactive transforms.

    Sequential pure transforms compose over the original source when no
    observed boundary requires an intermediate notification: `obs >> f >> g`
    behaves like mapping through `f` then `g`, while the runtime represents
    it as one composed transform.

    For merged observables (created with `+`), the function receives multiple arguments
    corresponding to the tuple values. For single observables, it receives one argument.

    Args:
        obs: The source observable(s) to transform. Can be a single Observable or
             a MergedObservable (from `+` operator).
        func: A pure function that transforms the observable value(s). For merged
              observables, receives unpacked tuple values as separate arguments.

    Returns:
        A new computed observable. Unobserved computed values are cached and
        version-validated lazily; observed values are maintained eagerly enough
        to deliver subscriber notifications.

    Examples:
        ```python
        from fynx.observable import Observable

        # Single observable with automatic optimization
        counter = Observable("counter", 5)
        result = counter >> (lambda x: x * 2) >> (lambda x: x + 10) >> str
        # Automatically optimized to single fused computation

        # Repeated products reuse the same ordered product while live
        width = Observable("width", 10)
        height = Observable("height", 20)
        area = (width + height) >> (lambda w, h: w * h)
        volume = (width + height + Observable("depth", 5)) >> (lambda w, h, d: w * h * d)
        ```

    Performance:
        - **Transform fusion**: reduces intermediate reactive-node overhead
        - **Canonical products**: reuses repeated ordered products while live
        - **Version invalidation**: refreshes lazy values only when inputs changed
        - **Demand frontier**: subscribers create eager maintenance where needed

    See Also:
        Observable.then: The method that creates computed observables
        MergedObservable: For combining multiple observables with `+`
    """
    # Delegate to the observable's optimized _create_computed method
    return obs._create_computed(func, obs)


def and_operator(obs: "Observable[Any]", other: BooleanOperand) -> "Observable[bool]":
    """
    Implement the `&` operator for total boolean AND observables.

    Args:
        obs: The first boolean-like observable.
        other: Another boolean-like observable.

    Returns:
        A computed Observable[bool] that updates whenever either input changes.
    """
    return obs.all(other)


def matmul_operator(
    obs: "Observable[T]", condition: ConditionOperand[T]
) -> "ConditionalObservable[T]":
    """Implement the `@` operator for gated conditional observables."""
    from .conditional import ConditionalObservable

    condition = unwrap_condition(condition)

    # Handle both observables and functions as conditions
    condition_obs: ConditionOperand[T]
    if callable(condition):
        # If condition is a function, create a computed observable
        # For conditionals, the condition should depend on the source value, not the conditional result

        if isinstance(obs, ConditionalObservable):
            # Condition should depend on the conditional's source
            source = obs._source_observable
            condition_obs = source._create_computed(condition, source)
        else:
            # Normal case: condition depends on the observable
            condition_obs = obs._create_computed(condition, obs)
    else:
        # If condition is already an observable, use it directly
        condition_obs = condition

    return ConditionalObservable(obs, condition_obs)
