"""
FynX Operations Protocols - Reactive Operations and Tuple Behavior Interface Definitions
======================================================================================

This module defines Protocol-based interfaces for reactive operations and tuple behavior in FynX,
providing both natural language methods and operator overloading protocols with
automatic unwrapping of nested Observable values.

The protocols provide a fluent, readable API for reactive programming:

Natural Language Methods:
- `then(func)` - Transform values (equivalent to `>>` operator)
- `alongside(other)` - Merge observables (equivalent to `+` operator)
- `requiring(condition)` - Compose boolean conditions with AND (equivalent to `&` operator)
- `negate()` - Boolean negation (equivalent to `~` operator)
- `either(other)` - OR logic for boolean conditions

Operator Protocols:
- `__add__`, `__radd__` - Merge observables with `+`
- `__rshift__` - Transform values with `>>`
- `__and__` - Filter values with `&`
- `__invert__` - Negate boolean values with `~`
- `__or__` - Create OR conditions with `|`

Tuple Behavior:
- `__iter__` - Iteration over tuple values
- `__len__` - Length of merged observables
- `__getitem__`, `__setitem__` - Indexing and assignment

Auto-Unwrapping:
All operations automatically unwrap nested Observable values:
- `Observable[Observable[T]]` → `Observable[T]`
- `Observable[Tuple[Observable[T], Observable[U]]]` → `Observable[Tuple[T, U]]`
- Works transparently in all operations (+, >>, &, |, ~)

Key Benefits:
- No circular imports (protocols don't import concrete implementations)
- Better type safety than ABCs
- Runtime isinstance() support with @runtime_checkable
- Structural subtyping (duck typing with type safety)
- Clean separation of interface from implementation
- Automatic value unwrapping for ergonomic API
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

# Import common types
from ..types.common_types import (
    Condition,
    ConditionFunction,
    T,
    TransformFunction,
    U,
    V,
)

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from ..types.observable_protocols import Computed, Conditional, Mergeable
    from .primitives_protocol import Observable

# ============================================================================
# REACTIVE OPERATIONS PROTOCOL
# ============================================================================


@runtime_checkable
class ReactiveOperations(Protocol[T]):
    """
    Protocol defining reactive operations for observables.

    This protocol provides both natural language methods and operator overloading
    for reactive operations. It enables observables to be composed using either
    a fluent method-based API or concise operator syntax, with automatic
    unwrapping of nested Observable values.

    All operations automatically unwrap nested Observable values, allowing users
    to pass either raw values or Observable values interchangeably while
    maintaining full type safety.

    Key Features:
    - Natural language methods (then, alongside, requiring, negate, either)
    - Operator overloading (+, >>, &, ~, |)
    - Automatic type unwrapping for all operations
    - Fluent chaining support
    - Self-documenting code options
    - Type-safe overloads for IntelliSense support

    Example:
        ```python
        def process_data(data: ReactiveOperations[str]) -> None:
            # Natural language approach (more readable)
            processed = data.then(lambda x: x.strip().upper())
            merged = data.alongside(other_data)
            filtered = data.requiring(lambda x: len(x) > 0)
            negated = is_loading.negate()

            # Operator approach (more concise)
            processed = data >> (lambda x: x.strip().upper())
            merged = data + other_data
            filtered = data & (lambda x: len(x) > 0)
            negated = ~is_loading

            # Auto-unwrapping works everywhere:
            wrapped = observable(observable(5))  # Observable[int] not Observable[Observable[int]]
            merged = observable(1) + observable(2)  # Observable[Tuple[int, int]]
            tuple_obs = observable((observable(1), observable(2)))  # Observable[Tuple[int, int]]
        ```
    """

    # ========================================================================
    # TRANSFORMATION OPERATIONS (Natural Language + Operators)
    # ========================================================================

    @overload
    def then(self, func: TransformFunction[T, U]) -> "Computed[U]": ...

    @overload
    def then(self, func: TransformFunction[T, "Observable[U]"]) -> "Computed[U]": ...

    def then(
        self, func: TransformFunction[T, Union[U, "Observable[U]"]]
    ) -> "Computed[U]":
        """
        Transform this observable's value with the given function.

        This is the natural language equivalent of the `>>` operator.
        It creates a computed observable that applies the transformation
        function to the current value. The function can return either a raw
        value or an Observable, which will be automatically unwrapped.

        Args:
            func: A pure function to apply to the observable's value.
                  Can return either T or Observable[T].

        Returns:
            A computed observable containing the transformed value.

        Example:
            ```python
            # Regular transformation
            doubled = counter.then(lambda x: x * 2)
            uppercased = name.then(lambda x: x.upper())

            # Transformation returning Observable (auto-unwrapped)
            chained = counter.then(lambda x: observable(x * 2))

            # Chaining transformations
            result = data.then(str.strip).then(str.upper).then(lambda x: x[:10])
            ```
        """
        ...

    @overload
    def __rshift__(self, func: TransformFunction[T, U]) -> "Computed[U]": ...

    @overload
    def __rshift__(
        self, func: TransformFunction[T, "Observable[U]"]
    ) -> "Computed[U]": ...

    def __rshift__(
        self, func: TransformFunction[T, Union[U, "Observable[U]"]]
    ) -> "Computed[U]":
        """
        Apply a transformation function using the >> operator.

        This implements the functorial map operation over observables, allowing
        you to transform observable values through pure functions while preserving
        reactivity. The >> operator is chosen because it visually represents
        data flow from left to right.

        Automatically unwraps Observable return values, enabling fluent
        chaining without manual unwrapping.

        Args:
            func: A pure function to apply to the observable's value(s).
                  Can return either T or Observable[T].

        Returns:
            A new computed Observable containing the transformed values.

        Example:
            ```python
            doubled = counter >> (lambda x: x * 2)
            uppercased = name >> (lambda x: x.upper())

            # Returns Observable are auto-unwrapped
            chained = counter >> (lambda x: observable(x * 2))

            # Chaining is seamless
            result = data >> str.strip >> str.upper >> (lambda x: x[:10])
            ```
        """
        ...

    # ========================================================================
    # MERGE OPERATIONS (Natural Language + Operators)
    # ========================================================================

    @overload
    def alongside(self, other: U) -> "Mergeable[Tuple[T, U]]": ...

    @overload
    def alongside(self, other: "Observable[U]") -> "Mergeable[Tuple[T, U]]": ...

    @overload
    def alongside(
        self, other: Tuple["Observable[U]", ...]
    ) -> "Mergeable[Tuple[T, ...]]": ...

    def alongside(
        self, other: Union[U, "Observable[U]", Tuple["Observable", ...]]
    ) -> "Mergeable":
        """
        Merge this observable with another into a tuple.

        This is the natural language equivalent of the `+` operator.
        It creates a merged observable that combines the values of both
        observables into a tuple, updating when either source changes.

        Automatically unwraps Observable values, so you can pass either
        raw values or Observable instances.

        Args:
            other: Another value, observable, or tuple of observables to merge with.

        Returns:
            A merged observable containing both values as a tuple.

        Example:
            ```python
            # Basic merge
            coordinates = x.alongside(y)  # Creates (x_value, y_value)

            # Chaining creates nested tuples
            point3d = x.alongside(y).alongside(z)  # (x, y, z)

            # Mix raw values and observables (auto-unwrapping)
            coords = x.alongside(5)  # Observable[Tuple[int, int]]
            coords2 = x.alongside(observable(5))  # Same type!

            # Merge tuple of observables
            point = x.alongside((y, z))  # Observable[Tuple[int, int, int]]
            ```
        """
        ...

    @overload
    def __add__(self, other: U) -> "Mergeable[Tuple[T, U]]": ...

    @overload
    def __add__(self, other: "Observable[U]") -> "Mergeable[Tuple[T, U]]": ...

    @overload
    def __add__(
        self, other: "Mergeable[Tuple[U, ...]]"
    ) -> "Mergeable[Tuple[T, U, ...]]": ...

    def __add__(self, other: Union[U, "Observable[U]", "Mergeable"]) -> "Mergeable":
        """
        Combine observables with the + operator.

        This creates a merged observable that contains a tuple of both values
        and updates automatically when either observable changes.

        Automatically unwraps Observable values and flattens nested merges
        for intuitive tuple construction.

        Args:
            other: Another value, Observable, or Mergeable to combine with.

        Returns:
            A MergedObservable containing both values as a tuple.

        Example:
            ```python
            # Basic merge
            coordinates = x + y  # Creates (x.value, y.value)

            # Chaining extends the tuple
            point3d = x + y + z  # Creates (x.value, y.value, z.value)

            # Auto-unwrapping works seamlessly
            coords2 = x + observable(5)  # Observable[Tuple[int, int]]

            # Mix different types
            data = name + age + is_active  # Observable[Tuple[str, int, bool]]
            ```
        """
        ...

    @overload
    def __radd__(self, other: U) -> "Mergeable[Tuple[U, T]]": ...

    @overload
    def __radd__(self, other: "Observable[U]") -> "Mergeable[Tuple[U, T]]": ...

    def __radd__(self, other: Union[U, "Observable[U]"]) -> "Mergeable":
        """
        Support right-side addition for merging observables.

        This enables expressions like `other + self` to work correctly,
        ensuring that merged observables can be chained properly.

        Automatically unwraps Observable values for seamless chaining.

        Args:
            other: Another value or Observable to combine with.

        Returns:
            A MergedObservable containing both values as a tuple.

        Example:
            ```python
            # Both work with auto-unwrapping:
            result1 = x + y
            result2 = y + x  # Uses __radd__

            # Works with raw values too
            result3 = 5 + x  # Uses __radd__
            ```
        """
        ...

    # ========================================================================
    # CONDITIONAL OPERATIONS (Natural Language + Operators)
    # ========================================================================

    @overload
    def requiring(self, condition: ConditionFunction[T]) -> "Conditional[T]": ...

    @overload
    def requiring(self, condition: "Observable[bool]") -> "Conditional[T]": ...

    @overload
    def requiring(self, condition: bool) -> "Conditional[T]": ...

    @overload
    def requiring(self, condition: "Conditional") -> "Conditional[T]": ...

    @overload
    def requiring(
        self,
        *conditions: Union[
            ConditionFunction[T], "Observable[bool]", bool, "Conditional"
        ]
    ) -> "Conditional[T]": ...

    def requiring(self, *conditions: Any) -> "Conditional[T]":
        """
        Compose this observable with conditions using AND logic.

        This is the natural language equivalent of the `&` operator.
        It creates a ConditionalObservable that combines this observable with
        additional conditions. Supports the same condition types as the & operator.

        Automatically unwraps Observable[bool] values, allowing conditions
        to be specified as raw bools, callables, or Observable instances.

        Args:
            *conditions: Variable number of conditions (raw bools, observables,
                        callables, or other Conditional observables).

        Returns:
            A ConditionalObservable representing the AND of all conditions.

        Example:
            ```python
            # Single condition (callable)
            positive = data.requiring(lambda x: x > 0)

            # Multiple conditions (AND logic)
            result = data.requiring(
                lambda x: x > 0,
                is_ready,
                other_condition
            )

            # Mix condition types with auto-unwrapping
            valid = data.requiring(
                lambda x: x != "",           # Callable
                observable(True),             # Observable (unwrapped)
                True,                         # Raw bool
                is_enabled                    # Another observable
            )

            # Compose with existing conditionals
            filtered = data.requiring(other_conditional)
            ```
        """
        ...

    @overload
    def __and__(self, condition: ConditionFunction[T]) -> "Conditional[T]": ...

    @overload
    def __and__(self, condition: "Observable[bool]") -> "Conditional[T]": ...

    @overload
    def __and__(self, condition: bool) -> "Conditional[T]": ...

    @overload
    def __and__(self, condition: "Conditional") -> "Conditional[T]": ...

    def __and__(self, condition: Condition) -> "Conditional[T]":
        """
        Create a conditional observable using the & operator.

        This creates a ConditionalObservable that only emits values when all
        specified conditions are True, enabling precise control over reactive
        updates. The & operator represents logical AND.

        Automatically unwraps Observable[bool] values for seamless
        condition composition.

        Args:
            condition: A boolean value, Observable, callable, or another Conditional.

        Returns:
            A ConditionalObservable that filters values based on the condition.

        Example:
            ```python
            # Callable condition
            valid_data = data & (lambda x: len(x) > 0)

            # Observable condition (auto-unwrapped)
            active_data = data & is_active

            # Raw bool
            simple = data & True

            # Chaining conditions (AND logic)
            filtered = data & (lambda x: x > 0) & is_enabled & is_ready

            # Compose with other conditionals
            combined = data & other_conditional
            ```
        """
        ...

    # ========================================================================
    # BOOLEAN OPERATIONS (Natural Language + Operators)
    # ========================================================================

    def negate(self) -> "Computed[bool]":
        """
        Create a negated boolean version of this observable.

        This is the natural language equivalent of the `~` operator.
        It creates a computed observable that returns the logical negation
        of the current boolean value.

        Automatically unwraps any nested Observable values.

        Returns:
            A computed observable with negated boolean values.

        Example:
            ```python
            # Basic negation
            is_disabled = is_enabled.negate()
            is_not_ready = is_ready.negate()

            # Use in conditions:
            data_when_disabled = data.requiring(is_enabled.negate())

            # Works even with nested observables (auto-unwrapped)
            nested = observable(observable(True))
            inverted = nested.negate()  # Observable[bool] = False

            # Chaining
            complex = is_loading.negate().requiring(is_ready)
            ```
        """
        ...

    def __invert__(self) -> "Computed[bool]":
        """
        Create a negated boolean observable using the ~ operator.

        This creates a computed observable that returns the logical negation
        of the current boolean value, useful for creating inverse conditions.
        The ~ operator represents logical NOT.

        Automatically unwraps any nested Observable values.

        Returns:
            A computed Observable[bool] with negated boolean value.

        Example:
            ```python
            # Basic negation
            is_disabled = ~is_enabled
            is_not_loading = ~is_loading

            # Use in conditions
            show_when_disabled = content & ~is_enabled

            # Double negation
            original = ~~is_enabled  # Back to original

            # Complex expressions
            complex = ~(is_loading | is_disabled)
            ```
        """
        ...

    @overload
    def either(self, other: bool) -> "Conditional[bool]": ...

    @overload
    def either(self, other: "Observable[bool]") -> "Conditional[bool]": ...

    @overload
    def either(self, other: "Conditional[bool]") -> "Conditional[bool]": ...

    def either(
        self, other: Union[bool, "Observable[bool]", "Conditional[bool]"]
    ) -> "Conditional[bool]":
        """
        Create an OR condition between this observable and another.

        This is the natural language equivalent of the `|` operator.
        It creates a conditional observable that only emits when the OR result
        is truthy. If the initial OR result is falsy, raises ConditionalNeverMet.

        Automatically unwraps Observable[bool] values.

        Args:
            other: Another boolean value, observable, or conditional to OR with.

        Returns:
            A conditional observable that only emits when OR is truthy.

        Raises:
            ConditionalNeverMet: If initial OR result is falsy.

        Example:
            ```python
            # Must start with at least one being True
            needs_attention = is_error.either(is_warning)
            can_proceed = has_permission.either(is_admin)

            # Auto-unwrapping
            can_edit = is_owner.either(observable(True))

            # Mix with raw bools
            always_show = some_condition.either(True)

            # Chaining OR conditions
            complex = cond1.either(cond2).either(cond3)
            ```
        """
        ...

    @overload
    def __or__(self, other: bool) -> "Conditional[bool]": ...

    @overload
    def __or__(self, other: "Observable[bool]") -> "Conditional[bool]": ...

    @overload
    def __or__(self, other: "Conditional[bool]") -> "Conditional[bool]": ...

    def __or__(
        self, other: Union[bool, "Observable[bool]", "Conditional[bool]"]
    ) -> "Conditional[bool]":
        """
        Create a logical OR condition using the | operator.

        This creates a conditional observable that only emits when the OR result
        is truthy. If the initial OR result is falsy, raises ConditionalNeverMet.
        The | operator represents logical OR.

        Automatically unwraps Observable[bool] values.

        Args:
            other: Another boolean value, observable, or conditional to OR with.

        Returns:
            A conditional observable that only emits when OR is truthy.

        Raises:
            ConditionalNeverMet: If initial OR result is falsy.

        Example:
            ```python
            # Basic OR
            needs_attention = is_error | is_warning
            can_proceed = has_permission | is_admin

            # Auto-unwrapping
            show_badge = has_notification | observable(is_important)

            # Complex boolean logic
            can_edit = (is_owner | is_admin) & is_logged_in
            ```
        """
        ...


# ============================================================================
# TUPLE BEHAVIOR PROTOCOL
# ============================================================================


@runtime_checkable
class TupleBehavior(Protocol[T]):
    """
    Protocol for observables that behave like tuples.

    This protocol enables tuple-like behavior for observables that represent
    collections of values (like MergedObservable). It provides iteration,
    indexing, and length operations that make merged observables behave like
    tuples of their component values.

    All values are automatically unwrapped when accessed, so you always get
    the raw values, not Observable wrappers.

    Key Features:
    - Iteration support (auto-unwrapped values)
    - Indexing support (auto-unwrapped values)
    - Length operations
    - Tuple-like access patterns
    - Assignment updates source observables

    Example:
        ```python
        def process_coordinates(coords: TupleBehavior[tuple[int, int]]) -> None:
            # Tuple-like access (values are unwrapped)
            x, y = coords[0], coords[1]
            length = len(coords)

            # Iteration (values are unwrapped)
            for coord in coords:
                print(coord)  # Prints unwrapped int values

            # Assignment updates source observables
            coords[0] = 10  # Updates the underlying x observable
        ```
    """

    def __iter__(self) -> Iterator[Any]:
        """
        Allow iteration over the tuple value.

        This enables the observable to be used in for loops and other
        iteration contexts, making it behave like a tuple.

        Returns unwrapped values during iteration.

        Returns:
            An iterator over the unwrapped tuple values.

        Example:
            ```python
            coords = x + y  # Creates (x.value, y.value)
            for coord in coords:
                print(coord)  # Prints x.value, then y.value (unwrapped)

            # Unpacking works
            a, b, c = point3d  # Gets unwrapped values
            ```
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of combined observables.

        This provides the length of the tuple, which corresponds to the
        number of source observables that were merged together.

        Returns:
            The number of source observables in the merged observable.

        Example:
            ```python
            coords = x + y  # Creates (x.value, y.value)
            print(len(coords))  # Prints 2

            point3d = x + y + z
            print(len(point3d))  # Prints 3
            ```
        """
        ...

    def __getitem__(self, index: int) -> Any:
        """
        Allow indexing into the merged observable like a tuple.

        This enables tuple-like indexing access to individual values
        in the merged observable.

        Returns the unwrapped value at the specified index.

        Args:
            index: The index to access (0-based).

        Returns:
            The unwrapped value at the specified index.

        Raises:
            IndexError: If the index is out of range.

        Example:
            ```python
            coords = x + y  # Creates (x.value, y.value)
            x_val = coords[0]  # Gets x.value (unwrapped)
            y_val = coords[1]  # Gets y.value (unwrapped)

            # Negative indexing works
            last = coords[-1]  # Gets y.value
            ```
        """
        ...

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Allow setting values by index, updating the corresponding source observable.

        This enables tuple-like assignment to individual values in the
        merged observable, which updates the corresponding source observable.

        Automatically unwraps Observable values before setting.

        Args:
            index: The index to set (0-based).
            value: The new value to set (can be raw value or Observable).

        Raises:
            IndexError: If the index is out of range.

        Example:
            ```python
            coords = x + y  # Creates (x.value, y.value)

            # Update source observables via index
            coords[0] = 10  # Sets x.value = 10 (updates x observable)
            coords[1] = 20  # Sets y.value = 20 (updates y observable)

            # Auto-unwrapping works
            coords[0] = observable(30)  # Unwraps and sets x.value = 30
            ```
        """
        ...
