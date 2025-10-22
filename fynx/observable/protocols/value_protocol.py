"""
FynX Transparent Value Protocol - Value-Like Behavior for ObservableValue Instances
=================================================================================

This module defines the TransparentValue protocol that enables ObservableValue instances
to behave like regular Python values while maintaining reactive capabilities.

The protocol provides magic methods for equality, string conversion, iteration, indexing,
etc., while also supporting reactive operators. All magic methods automatically unwrap
ObservableValue instances for seamless integration with existing Python code.

Key Features:
- Value-like behavior (equality, string conversion, etc.)
- Transparent access to the wrapped observable
- Seamless integration with existing Python code
- Reactive capabilities preserved
- Auto-unwrapping in all operations

Example:
    ```python
    def process_value_wrapper(value: TransparentValue[str]) -> None:
        # Behaves like a regular string (auto-unwrapped)
        print(f"Value: {value}")     # String conversion
        if value == "test":          # Equality comparison
            print("Match!")

        # But also supports reactive operations
        doubled = value >> (lambda x: x * 2)
    ```
"""

from typing import Any, Iterator, Protocol, TypeVar, runtime_checkable

# Import common types
from ..types.common_types import T

# ============================================================================
# TRANSPARENT VALUE PROTOCOL
# ============================================================================


@runtime_checkable
class TransparentValue(Protocol[T]):
    """
    Protocol for observable values that behave transparently like their underlying values.

    This protocol enables ObservableValue instances to behave like regular
    Python values while maintaining reactive capabilities. It provides comprehensive
    magic methods for all common Python operations including equality, string conversion,
    iteration, indexing, comparison, arithmetic, and more, while also supporting
    the reactive operators.

    All magic methods automatically unwrap ObservableValue instances for seamless integration
    with existing Python code.

    Key Features:
    - Raw value access via .value property (no unwrapping)
    - Explicit unwrapping via .unwrap() method
    - Change notification via on_change callback
    - Value-like behavior (equality, string conversion, etc.)
    - Transparent access to the wrapped observable
    - Seamless integration with existing Python code
    - Reactive capabilities preserved
    - Auto-unwrapping in all magic methods
    - Complete operator support (arithmetic, comparison, etc.)
    - Hashing support for use as dictionary keys
    - Sorting and ordering support

    Core Methods:
    - .value: Returns the raw value as set (no unwrapping)
    - .unwrap(): Recursively unwraps nested ObservableValue instances
    - on_change: Optional callback for value change notifications

    Supported Operations:
    - Basic: __eq__, __str__, __repr__, __bool__, __len__, __iter__, __getitem__, __contains__
    - Hashing: __hash__ (for dictionary keys and sets)
    - Comparison: __lt__, __le__, __gt__, __ge__ (for sorting and ordering)
    - Arithmetic: __add__, __sub__, __mul__, __truediv__, __floordiv__, __mod__, __pow__
    - Unary: __neg__, __pos__, __abs__

    Example:
        ```python
        def on_value_change(old_value, new_value):
            print(f"Value changed from {old_value} to {new_value}")

        def process_value_wrapper(value: TransparentValue[int]) -> None:
            # Access raw value (no unwrapping)
            raw_value = value.value  # Returns exactly what was set

            # Explicit unwrapping when needed
            unwrapped = value.unwrap()  # Recursively unwraps nested ObservableValues

            # Behaves like a regular integer in operations (auto-unwrapped)
            print(f"Value: {value}")     # String conversion
            if value == 42:              # Equality comparison
                print("Match!")

            # Arithmetic operations work transparently
            result = value + 10          # Addition
            doubled = value * 2          # Multiplication

            # But also supports reactive operations
            reactive_doubled = value >> (lambda x: x * 2)
        ```
    """

    @property
    def value(self) -> T:
        """
        Get the raw value as it was set (no unwrapping).

        This property returns the exact value that was set, without any
        automatic unwrapping of nested Observable values. For unwrapping,
        use the .unwrap() method explicitly.

        Returns:
            The raw value as it was set.

        Example:
            ```python
            obs_value = ObservableValue(observable(observable(42)))
            raw = obs_value.value  # Returns Observable(Observable(42))
            unwrapped = obs_value.unwrap()  # Returns 42
            ```
        """
        ...

    @value.setter
    def value(self, new_value: T) -> None:
        """
        Set a new value and trigger change notification.

        When a new value is set, the cache is invalidated and the on_change
        callback (if provided) is called with the old and new values.

        Args:
            new_value: The new value to set (can be Observable or regular value).

        Example:
            ```python
            def on_change(old_val, new_val):
                print(f"Changed from {old_val} to {new_val}")

            obs_value = ObservableValue(42, on_change=on_change)
            obs_value.value = 100  # Calls on_change(42, 100)
            ```
        """
        ...

    def unwrap(self) -> Any:
        """
        Recursively unwrap nested Observable values.

        This method performs the same recursive unwrapping that was previously
        done automatically by the .value property. It unwraps all nested
        Observable values to return the underlying raw value.

        Returns:
            The unwrapped value with all nested Observables resolved.

        Example:
            ```python
            obs_value = ObservableValue(observable(observable(42)))
            raw = obs_value.value  # Observable(Observable(42))
            unwrapped = obs_value.unwrap()  # 42

            # Works with complex nested structures
            nested = ObservableValue(observable([observable(1), observable(2)]))
            result = nested.unwrap()  # [1, 2]
            ```
        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison with another value or observable.

        This enables the observable value to be compared with regular
        Python values using the == operator.

        Automatically unwraps Observable values on both sides.

        Args:
            other: Value or Observable to compare with.

        Returns:
            True if the values are equal, False otherwise.

        Example:
            ```python
            obs_value = ObservableValue(observable("test"))
            print(obs_value == "test")  # True
            print(obs_value == "other")  # False

            # Works with other observables too
            obs2 = ObservableValue(observable("test"))
            print(obs_value == obs2)  # True (both unwrapped to "test")
            ```
        """
        ...

    def __str__(self) -> str:
        """
        String representation of the wrapped value.

        This enables the observable value to be used in string contexts
        like f-strings, print statements, etc.

        Automatically unwraps the Observable value.

        Returns:
            String representation of the wrapped value.

        Example:
            ```python
            obs_value = ObservableValue(observable("Hello"))
            print(f"Message: {obs_value}")  # Prints: "Message: Hello"

            # Works in all string contexts
            result = "Value is: " + str(obs_value)
            ```
        """
        ...

    def __repr__(self) -> str:
        """
        Developer representation of the wrapped value.

        This provides a string representation useful for debugging
        and development.

        Automatically unwraps the Observable value.

        Returns:
            Developer representation of the wrapped value.

        Example:
            ```python
            obs_value = ObservableValue(observable("test"))
            print(repr(obs_value))  # Shows debug representation

            # Useful in REPL
            >>> obs_value
            ObservableValue("test")
            ```
        """
        ...

    def __len__(self) -> int:
        """
        Length of the wrapped value if it's a collection.

        This enables the observable value to be used with len() if
        the underlying value supports it.

        Automatically unwraps the Observable value.

        Returns:
            Length of the wrapped value, or 0 if not applicable.

        Example:
            ```python
            obs_list = ObservableValue(observable([1, 2, 3]))
            print(len(obs_list))  # Prints: 3

            obs_str = ObservableValue(observable("hello"))
            print(len(obs_str))  # Prints: 5
            ```
        """
        ...

    def __iter__(self) -> Iterator[Any]:
        """
        Iteration over the wrapped value if it's iterable.

        This enables the observable value to be used in for loops
        if the underlying value is iterable.

        Automatically unwraps the Observable value.

        Returns:
            Iterator over the wrapped value, or empty iterator if not applicable.

        Example:
            ```python
            obs_list = ObservableValue(observable([1, 2, 3]))
            for item in obs_list:
                print(item)  # Prints: 1, 2, 3

            # Unpacking works
            a, b, c = obs_list
            ```
        """
        ...

    def __getitem__(self, key: Any) -> Any:
        """
        Indexing into the wrapped value if it's subscriptable.

        This enables the observable value to be used with indexing
        if the underlying value supports it.

        Automatically unwraps the Observable value.

        Args:
            key: The key or index to access.

        Returns:
            The value at the specified key/index.

        Raises:
            TypeError: If the wrapped value is not subscriptable.
            KeyError/IndexError: If the key/index is not found.

        Example:
            ```python
            obs_dict = ObservableValue(observable({"key": "value"}))
            print(obs_dict["key"])  # Prints: "value"

            obs_list = ObservableValue(observable([1, 2, 3]))
            print(obs_list[0])  # Prints: 1
            print(obs_list[-1])  # Prints: 3
            ```
        """
        ...

    def __contains__(self, item: Any) -> bool:
        """
        Check if the wrapped value contains an item.

        This enables the observable value to be used with the `in` operator
        if the underlying value supports it.

        Automatically unwraps the Observable value.

        Args:
            item: The item to check for.

        Returns:
            True if the item is found, False otherwise.

        Example:
            ```python
            obs_list = ObservableValue(observable([1, 2, 3]))
            print(2 in obs_list)  # True
            print(5 in obs_list)  # False

            obs_str = ObservableValue(observable("hello"))
            print("e" in obs_str)  # True
            ```
        """
        ...

    def __bool__(self) -> bool:
        """
        Boolean conversion of the wrapped value.

        This enables the observable value to be used in boolean contexts
        like if statements, while loops, etc.

        Automatically unwraps the Observable value.

        Returns:
            Boolean value of the wrapped value.

        Example:
            ```python
            obs_value = ObservableValue(observable("test"))
            if obs_value:  # Uses __bool__
                print("Value is truthy")

            obs_empty = ObservableValue(observable(""))
            if not obs_empty:  # Uses __bool__
                print("Value is falsy")
            ```
        """
        ...

    def __hash__(self) -> int:
        """
        Hash of the wrapped value for use as dictionary keys or in sets.

        This enables the observable value to be used as a dictionary key
        or included in sets if the underlying value is hashable.

        Automatically unwraps the Observable value.

        Returns:
            Hash value of the wrapped value.

        Raises:
            TypeError: If the wrapped value is not hashable.

        Example:
            ```python
            obs_value = ObservableValue(observable("test"))
            my_dict = {obs_value: "data"}  # Uses __hash__

            obs_set = {obs_value, "other"}  # Uses __hash__
            ```
        """
        ...

    def __lt__(self, other: Any) -> bool:
        """
        Less than comparison with another value.

        This enables the observable value to be compared with other values
        using the < operator for sorting and ordering operations.

        Automatically unwraps the Observable value.

        Args:
            other: Value to compare with.

        Returns:
            True if this value is less than other, False otherwise.

        Raises:
            TypeError: If comparison is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            print(obs_value < 10)  # True
            print(obs_value < 3)   # False

            # Works with sorting
            values = [obs_value, ObservableValue(observable(3))]
            sorted_values = sorted(values)  # Uses __lt__
            ```
        """
        ...

    def __le__(self, other: Any) -> bool:
        """
        Less than or equal comparison with another value.

        This enables the observable value to be compared with other values
        using the <= operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to compare with.

        Returns:
            True if this value is less than or equal to other, False otherwise.

        Raises:
            TypeError: If comparison is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            print(obs_value <= 5)  # True
            print(obs_value <= 3)  # False
            ```
        """
        ...

    def __gt__(self, other: Any) -> bool:
        """
        Greater than comparison with another value.

        This enables the observable value to be compared with other values
        using the > operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to compare with.

        Returns:
            True if this value is greater than other, False otherwise.

        Raises:
            TypeError: If comparison is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            print(obs_value > 3)   # True
            print(obs_value > 10)  # False
            ```
        """
        ...

    def __ge__(self, other: Any) -> bool:
        """
        Greater than or equal comparison with another value.

        This enables the observable value to be compared with other values
        using the >= operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to compare with.

        Returns:
            True if this value is greater than or equal to other, False otherwise.

        Raises:
            TypeError: If comparison is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            print(obs_value >= 5)  # True
            print(obs_value >= 3)  # True
            print(obs_value >= 10) # False
            ```
        """
        ...

    def __add__(self, other: Any) -> Any:
        """
        Addition operation with another value.

        This enables the observable value to be used in addition operations
        with other values using the + operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to add to this value.

        Returns:
            Result of the addition operation.

        Raises:
            TypeError: If addition is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            result = obs_value + 3  # 8
            result = obs_value + ObservableValue(observable(2))  # 7
            ```
        """
        ...

    def __sub__(self, other: Any) -> Any:
        """
        Subtraction operation with another value.

        This enables the observable value to be used in subtraction operations
        with other values using the - operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to subtract from this value.

        Returns:
            Result of the subtraction operation.

        Raises:
            TypeError: If subtraction is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(10))
            result = obs_value - 3  # 7
            result = obs_value - ObservableValue(observable(2))  # 8
            ```
        """
        ...

    def __mul__(self, other: Any) -> Any:
        """
        Multiplication operation with another value.

        This enables the observable value to be used in multiplication operations
        with other values using the * operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to multiply with this value.

        Returns:
            Result of the multiplication operation.

        Raises:
            TypeError: If multiplication is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            result = obs_value * 3  # 15
            result = obs_value * ObservableValue(observable(2))  # 10
            ```
        """
        ...

    def __truediv__(self, other: Any) -> Any:
        """
        True division operation with another value.

        This enables the observable value to be used in division operations
        with other values using the / operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to divide this value by.

        Returns:
            Result of the division operation.

        Raises:
            TypeError: If division is not supported between the types.
            ZeroDivisionError: If dividing by zero.

        Example:
            ```python
            obs_value = ObservableValue(observable(10))
            result = obs_value / 2  # 5.0
            result = obs_value / ObservableValue(observable(3))  # 3.333...
            ```
        """
        ...

    def __floordiv__(self, other: Any) -> Any:
        """
        Floor division operation with another value.

        This enables the observable value to be used in floor division operations
        with other values using the // operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to floor divide this value by.

        Returns:
            Result of the floor division operation.

        Raises:
            TypeError: If floor division is not supported between the types.
            ZeroDivisionError: If dividing by zero.

        Example:
            ```python
            obs_value = ObservableValue(observable(10))
            result = obs_value // 3  # 3
            result = obs_value // ObservableValue(observable(4))  # 2
            ```
        """
        ...

    def __mod__(self, other: Any) -> Any:
        """
        Modulo operation with another value.

        This enables the observable value to be used in modulo operations
        with other values using the % operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to use as the modulus.

        Returns:
            Result of the modulo operation.

        Raises:
            TypeError: If modulo is not supported between the types.
            ZeroDivisionError: If modulus is zero.

        Example:
            ```python
            obs_value = ObservableValue(observable(10))
            result = obs_value % 3  # 1
            result = obs_value % ObservableValue(observable(4))  # 2
            ```
        """
        ...

    def __pow__(self, other: Any) -> Any:
        """
        Power operation with another value.

        This enables the observable value to be used in power operations
        with other values using the ** operator.

        Automatically unwraps the Observable value.

        Args:
            other: Value to raise this value to the power of.

        Returns:
            Result of the power operation.

        Raises:
            TypeError: If power operation is not supported between the types.

        Example:
            ```python
            obs_value = ObservableValue(observable(2))
            result = obs_value ** 3  # 8
            result = obs_value ** ObservableValue(observable(4))  # 16
            ```
        """
        ...

    def __neg__(self) -> Any:
        """
        Negation operation (unary minus).

        This enables the observable value to be negated using the - operator.

        Automatically unwraps the Observable value.

        Returns:
            Negated value.

        Raises:
            TypeError: If negation is not supported for the wrapped value type.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            result = -obs_value  # -5

            obs_float = ObservableValue(observable(3.14))
            result = -obs_float  # -3.14
            ```
        """
        ...

    def __pos__(self) -> Any:
        """
        Positive operation (unary plus).

        This enables the observable value to be used with the unary + operator.

        Automatically unwraps the Observable value.

        Returns:
            Positive value (usually the same as the original value).

        Raises:
            TypeError: If positive operation is not supported for the wrapped value type.

        Example:
            ```python
            obs_value = ObservableValue(observable(5))
            result = +obs_value  # 5

            obs_float = ObservableValue(observable(-3.14))
            result = +obs_float  # -3.14 (unchanged)
            ```
        """
        ...

    def __abs__(self) -> Any:
        """
        Absolute value operation.

        This enables the observable value to be used with the abs() function.

        Automatically unwraps the Observable value.

        Returns:
            Absolute value of the wrapped value.

        Raises:
            TypeError: If absolute value is not supported for the wrapped value type.

        Example:
            ```python
            obs_value = ObservableValue(observable(-5))
            result = abs(obs_value)  # 5

            obs_float = ObservableValue(observable(-3.14))
            result = abs(obs_float)  # 3.14
            ```
        """
        ...
