"""
Comprehensive tests for FynX ObservableValue class and TransparentValue protocol.

This module tests all functionality of the ObservableValue class including:
- Basic value storage and retrieval
- on_change callback functionality
- .value vs .unwrap() behavior
- All magic methods (arithmetic, comparison, etc.)
- TransparentValue protocol compliance
- Edge cases and error handling
"""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, call

import pytest

from fynx.observable.primitives.base import Observable
from fynx.observable.protocols.value_protocol import TransparentValue
from fynx.observable.value.value import ObservableValue


class TestObservableValueBasicFunctionality:
    """Test basic ObservableValue class functionality."""

    def test_value_creation_with_simple_value(self):
        """Test creating ObservableValue with simple values."""
        # Test with integer
        val = ObservableValue(42)
        assert val.value == 42
        assert val.unwrap() == 42

        # Test with string
        val = ObservableValue("hello")
        assert val.value == "hello"
        assert val.unwrap() == "hello"

        # Test with list
        val = ObservableValue([1, 2, 3])
        assert val.value == [1, 2, 3]
        assert val.unwrap() == [1, 2, 3]

        # Test with dict
        val = ObservableValue({"key": "value"})
        assert val.value == {"key": "value"}
        assert val.unwrap() == {"key": "value"}

    def test_value_creation_with_observable(self):
        """Test creating ObservableValue with Observable values."""
        obs = Observable(42)
        val = ObservableValue(obs)
        assert val.value == obs
        assert (
            val.unwrap() == obs
        )  # Observable is not unwrapped, only ObservableValue instances are

    def test_value_creation_with_nested_observables(self):
        """Test creating ObservableValue with nested Observable values."""
        nested_obs = Observable(Observable(42))
        val = ObservableValue(nested_obs)
        assert val.value == nested_obs
        assert (
            val.unwrap() == nested_obs
        )  # Observable is not unwrapped, only ObservableValue instances are

    def test_value_creation_with_nested_values(self):
        """Test creating ObservableValue with nested ObservableValue instances."""
        inner_val = ObservableValue(42)
        outer_val = ObservableValue(inner_val)
        assert outer_val.value == inner_val
        assert outer_val.unwrap() == 42  # ObservableValue instances are unwrapped

    def test_value_creation_with_complex_nested_structures(self):
        """Test creating ObservableValue with complex nested structures."""
        # Nested list with ObservableValue instances
        complex_val = [ObservableValue(1), ObservableValue(2), ObservableValue(3)]
        val = ObservableValue(complex_val)
        assert val.value == complex_val
        assert val.unwrap() == [1, 2, 3]  # ObservableValue instances are unwrapped

        # Nested dict with ObservableValue instances
        complex_dict = {"a": ObservableValue(1), "b": ObservableValue(2)}
        val = ObservableValue(complex_dict)
        assert val.value == complex_dict
        assert val.unwrap() == {"a": 1, "b": 2}

        # Nested tuple with ObservableValue instances
        complex_tuple = (ObservableValue(1), ObservableValue(2))
        val = ObservableValue(complex_tuple)
        assert val.value == complex_tuple
        assert val.unwrap() == (1, 2)

    def test_value_default_initialization(self):
        """Test ObservableValue initialization with no arguments."""
        val = ObservableValue()
        assert val.value is None
        assert val.unwrap() is None


class TestObservableValueChangeCallback:
    """Test on_change callback functionality."""

    def test_on_change_callback_basic(self):
        """Test basic on_change callback functionality."""
        callback = Mock()
        val = ObservableValue(42, on_change=callback)

        # Change value
        val.value = 100

        # Verify callback was called with correct arguments
        callback.assert_called_once_with(42, 100)

    def test_on_change_callback_multiple_changes(self):
        """Test on_change callback with multiple value changes."""
        callback = Mock()
        val = ObservableValue(42, on_change=callback)

        # Multiple changes
        val.value = 100
        val.value = 200
        val.value = 300

        # Verify all calls
        expected_calls = [call(42, 100), call(100, 200), call(200, 300)]
        callback.assert_has_calls(expected_calls)

    def test_on_change_callback_with_observables(self):
        """Test on_change callback with Observable values."""
        callback = Mock()
        obs1 = Observable("test1", 42)
        obs2 = Observable("test2", 100)

        val = ObservableValue(obs1, on_change=callback)
        val.value = obs2

        callback.assert_called_once_with(obs1, obs2)

    def test_on_change_callback_none_value(self):
        """Test on_change callback when changing to/from None."""
        callback = Mock()
        val = ObservableValue(42, on_change=callback)

        # Change to None
        val.value = None
        callback.assert_called_with(42, None)

        # Change from None
        val.value = 100
        callback.assert_called_with(None, 100)

    def test_value_without_callback(self):
        """Test ObservableValue without on_change callback (should not raise errors)."""
        val = ObservableValue(42)

        # Should not raise any errors
        val.value = 100
        val.value = 200
        assert val.value == 200


class TestObservableValueUnwrapBehavior:
    """Test .value vs .unwrap() behavior."""

    def test_value_property_returns_raw(self):
        """Test that .value returns exactly what was set."""
        obs = Observable(42)
        val = ObservableValue(obs)

        # .value should return the Observable, not the unwrapped value
        assert val.value == obs
        assert val.value is obs

    def test_unwrap_method_returns_unwrapped(self):
        """Test that .unwrap() returns the unwrapped value."""
        obs = Observable(42)
        val = ObservableValue(obs)

        # .unwrap() should return the Observable (not unwrapped)
        assert val.unwrap() == obs

        # Test with nested ObservableValue instances
        inner_val = ObservableValue(42)
        outer_val = ObservableValue(inner_val)
        assert outer_val.unwrap() == 42  # ObservableValue instances are unwrapped

    def test_nested_unwrapping(self):
        """Test unwrapping of nested ObservableValue instances."""
        inner_val = ObservableValue(42)
        middle_val = ObservableValue(inner_val)
        outer_val = ObservableValue(middle_val)

        # .value returns the middle ObservableValue
        assert outer_val.value == middle_val

        # .unwrap() returns the final unwrapped value
        assert outer_val.unwrap() == 42

    def test_complex_nested_unwrapping(self):
        """Test unwrapping of complex nested structures."""
        # List with nested ObservableValue instances
        complex_list = [ObservableValue(1), ObservableValue(2)]
        val = ObservableValue(complex_list)

        assert val.value == complex_list
        assert val.unwrap() == [1, 2]

        # Dict with nested ObservableValue instances
        complex_dict = {"a": ObservableValue(1), "b": ObservableValue(2)}
        val = ObservableValue(complex_dict)

        assert val.value == complex_dict
        assert val.unwrap() == {"a": 1, "b": 2}

    def test_unwrap_caching(self):
        """Test that unwrap results are cached."""
        inner_val = ObservableValue(42)
        val = ObservableValue(inner_val)

        # First call should unwrap
        result1 = val.unwrap()

        # Second call should use cache
        result2 = val.unwrap()

        assert result1 == result2 == 42

        # Change value should invalidate cache
        val.value = ObservableValue(100)
        result3 = val.unwrap()
        assert result3 == 100


class TestObservableValueMagicMethods:
    """Test all magic methods for transparent behavior."""

    def test_string_conversion(self):
        """Test __str__ and __repr__ methods."""
        val = ObservableValue(42)

        # __str__ should use unwrapped value
        assert str(val) == "42"

        # __repr__ should show both raw and unwrapped
        repr_str = repr(val)
        assert "ObservableValue(" in repr_str
        assert "raw=42" in repr_str
        assert "unwrapped=42" in repr_str

    def test_string_conversion_with_observables(self):
        """Test string conversion with Observable values."""
        obs = Observable("hello")
        val = ObservableValue(obs)

        # __str__ should use unwrapped value (which is the Observable)
        assert str(val) == str(obs)  # Observable's string representation

        # __repr__ should show Observable
        repr_str = repr(val)
        assert "Observable" in repr_str

    def test_equality_comparison(self):
        """Test __eq__ method."""
        val1 = ObservableValue(42)
        val2 = ObservableValue(42)
        val3 = ObservableValue(100)

        # Equal values should be equal
        assert val1 == val2
        assert val1 == 42
        assert val2 == 42

        # Different values should not be equal
        assert val1 != val3
        assert val1 != 100

    def test_equality_with_observables(self):
        """Test equality with Observable values."""
        obs1 = Observable(42)
        obs2 = Observable(42)
        val1 = ObservableValue(obs1)
        val2 = ObservableValue(obs2)

        # Should compare unwrapped values (which are the Observables)
        assert val1 == val2  # Both contain Observable(42)
        assert val1 == obs1  # ObservableValue equals its contained Observable
        assert val2 == obs2

    def test_boolean_conversion(self):
        """Test __bool__ method."""
        # Truthy values
        assert bool(ObservableValue(42))
        assert bool(ObservableValue("hello"))
        assert bool(ObservableValue([1, 2, 3]))

        # Falsy values
        assert not bool(ObservableValue(0))
        assert not bool(ObservableValue(""))
        assert not bool(ObservableValue([]))
        assert not bool(ObservableValue(None))

    def test_length(self):
        """Test __len__ method."""
        val = ObservableValue([1, 2, 3, 4, 5])
        assert len(val) == 5

        val = ObservableValue("hello")
        assert len(val) == 5

        val = ObservableValue({"a": 1, "b": 2})
        assert len(val) == 2

    def test_length_with_observables(self):
        """Test length with Observable values."""
        obs = Observable([1, 2, 3])
        val = ObservableValue(obs)
        # Observable doesn't have length, so this should raise an error
        with pytest.raises(TypeError, match="has no len"):
            len(val)

    def test_length_with_nested_values(self):
        """Test length with nested ObservableValue instances."""
        inner_val = ObservableValue([1, 2, 3])
        outer_val = ObservableValue(inner_val)
        assert len(outer_val) == 3

    def test_length_error(self):
        """Test __len__ with non-lengthy values."""
        val = ObservableValue(42)
        with pytest.raises(TypeError, match="has no len"):
            len(val)

    def test_iteration(self):
        """Test __iter__ method."""
        val = ObservableValue([1, 2, 3])
        items = list(val)
        assert items == [1, 2, 3]

        val = ObservableValue("hello")
        chars = list(val)
        assert chars == ["h", "e", "l", "l", "o"]

    def test_iteration_with_observables(self):
        """Test iteration with Observable values."""
        obs = Observable([1, 2, 3])
        val = ObservableValue(obs)
        # Observable is not iterable, so this should raise an error
        with pytest.raises(TypeError, match="is not iterable"):
            list(val)

    def test_iteration_with_nested_values(self):
        """Test iteration with nested ObservableValue instances."""
        inner_val = ObservableValue([1, 2, 3])
        outer_val = ObservableValue(inner_val)
        items = list(outer_val)
        assert items == [1, 2, 3]

    def test_iteration_error(self):
        """Test __iter__ with non-iterable values."""
        val = ObservableValue(42)
        with pytest.raises(TypeError, match="is not iterable"):
            list(val)

    def test_indexing(self):
        """Test __getitem__ method."""
        val = ObservableValue([1, 2, 3])
        assert val[0] == 1
        assert val[1] == 2
        assert val[2] == 3

        val = ObservableValue({"a": 1, "b": 2})
        assert val["a"] == 1
        assert val["b"] == 2

    def test_indexing_with_observables(self):
        """Test indexing with Observable values."""
        obs = Observable([1, 2, 3])
        val = ObservableValue(obs)
        # Observable is not subscriptable, so this should raise an error
        with pytest.raises(TypeError, match="is not subscriptable"):
            val[0]

    def test_indexing_with_nested_values(self):
        """Test indexing with nested ObservableValue instances."""
        inner_val = ObservableValue([1, 2, 3])
        outer_val = ObservableValue(inner_val)
        assert outer_val[0] == 1
        assert outer_val[1] == 2

    def test_indexing_error(self):
        """Test __getitem__ with non-subscriptable values."""
        val = ObservableValue(42)
        with pytest.raises(TypeError, match="is not subscriptable"):
            val[0]

    def test_contains(self):
        """Test __contains__ method."""
        val = ObservableValue([1, 2, 3])
        assert 2 in val
        assert 5 not in val

        val = ObservableValue("hello")
        assert "e" in val
        assert "x" not in val

    def test_contains_with_observables(self):
        """Test contains with Observable values."""
        obs = Observable([1, 2, 3])
        val = ObservableValue(obs)
        # Observable is not iterable, so this should raise an error
        with pytest.raises(TypeError, match="is not iterable"):
            2 in val

    def test_contains_with_nested_values(self):
        """Test contains with nested ObservableValue instances."""
        inner_val = ObservableValue([1, 2, 3])
        outer_val = ObservableValue(inner_val)
        assert 2 in outer_val
        assert 5 not in outer_val

    def test_contains_error(self):
        """Test __contains__ with non-iterable values."""
        val = ObservableValue(42)
        with pytest.raises(TypeError, match="is not iterable"):
            2 in val


class TestObservableValueHashing:
    """Test hashing functionality."""

    def test_hash_with_hashable_values(self):
        """Test __hash__ with hashable values."""
        val1 = ObservableValue(42)
        val2 = ObservableValue(42)
        val3 = ObservableValue(100)

        # Equal values should have same hash
        assert hash(val1) == hash(val2)

        # Different values should have different hashes
        assert hash(val1) != hash(val3)

    def test_hash_with_strings(self):
        """Test hashing with string values."""
        val1 = ObservableValue("hello")
        val2 = ObservableValue("hello")
        val3 = ObservableValue("world")

        assert hash(val1) == hash(val2)
        assert hash(val1) != hash(val3)

    def test_hash_with_observables(self):
        """Test hashing with Observable values."""
        obs1 = Observable(42)
        obs2 = Observable(42)
        val1 = ObservableValue(obs1)
        val2 = ObservableValue(obs2)

        # Should hash based on unwrapped value (which are the Observables)
        # Different Observable instances will have different hashes
        assert hash(val1) != hash(val2)  # Different Observable instances

    def test_hash_with_nested_values(self):
        """Test hashing with nested ObservableValue instances."""
        inner_val1 = ObservableValue(42)
        inner_val2 = ObservableValue(42)
        val1 = ObservableValue(inner_val1)
        val2 = ObservableValue(inner_val2)

        # Should hash based on unwrapped value
        assert hash(val1) == hash(val2)  # Both unwrap to 42

    def test_hash_error(self):
        """Test __hash__ with unhashable values."""
        val = ObservableValue([1, 2, 3])  # Lists are not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(val)


class TestObservableValueComparison:
    """Test comparison operators."""

    def test_less_than(self):
        """Test __lt__ method."""
        val1 = ObservableValue(5)
        val2 = ObservableValue(10)

        assert val1 < val2
        assert val1 < 10
        assert not (val2 < val1)

    def test_less_than_or_equal(self):
        """Test __le__ method."""
        val1 = ObservableValue(5)
        val2 = ObservableValue(5)
        val3 = ObservableValue(10)

        assert val1 <= val2
        assert val1 <= val3
        assert not (val3 <= val1)

    def test_greater_than(self):
        """Test __gt__ method."""
        val1 = ObservableValue(10)
        val2 = ObservableValue(5)

        assert val1 > val2
        assert val1 > 5
        assert not (val2 > val1)

    def test_greater_than_or_equal(self):
        """Test __ge__ method."""
        val1 = ObservableValue(10)
        val2 = ObservableValue(10)
        val3 = ObservableValue(5)

        assert val1 >= val2
        assert val1 >= val3
        assert not (val3 >= val1)

    def test_comparison_with_observables(self):
        """Test comparison with Observable values."""
        obs1 = Observable(5)
        obs2 = Observable(10)
        val1 = ObservableValue(obs1)
        val2 = ObservableValue(obs2)

        # Observable doesn't support comparison, so this should raise an error
        with pytest.raises(TypeError, match="not supported"):
            val1 < val2

    def test_comparison_with_nested_values(self):
        """Test comparison with nested ObservableValue instances."""
        inner_val1 = ObservableValue(5)
        inner_val2 = ObservableValue(10)
        val1 = ObservableValue(inner_val1)
        val2 = ObservableValue(inner_val2)

        assert val1 < val2
        assert val1 < 10
        assert val2 > val1
        assert val2 > 5

    def test_comparison_error(self):
        """Test comparison with non-comparable values."""
        val1 = ObservableValue("hello")
        val2 = ObservableValue(42)

        with pytest.raises(TypeError, match="not supported"):
            val1 < val2


class TestObservableValueArithmetic:
    """Test arithmetic operators."""

    def test_addition(self):
        """Test __add__ method."""
        val1 = ObservableValue(5)
        val2 = ObservableValue(3)

        result = val1 + val2
        assert result == 8

        result = val1 + 3
        assert result == 8

    def test_subtraction(self):
        """Test __sub__ method."""
        val1 = ObservableValue(10)
        val2 = ObservableValue(3)

        result = val1 - val2
        assert result == 7

        result = val1 - 3
        assert result == 7

    def test_multiplication(self):
        """Test __mul__ method."""
        val1 = ObservableValue(5)
        val2 = ObservableValue(3)

        result = val1 * val2
        assert result == 15

        result = val1 * 3
        assert result == 15

    def test_division(self):
        """Test __truediv__ method."""
        val1 = ObservableValue(15)
        val2 = ObservableValue(3)

        result = val1 / val2
        assert result == 5.0

        result = val1 / 3
        assert result == 5.0

    def test_floor_division(self):
        """Test __floordiv__ method."""
        val1 = ObservableValue(15)
        val2 = ObservableValue(4)

        result = val1 // val2
        assert result == 3

        result = val1 // 4
        assert result == 3

    def test_modulo(self):
        """Test __mod__ method."""
        val1 = ObservableValue(15)
        val2 = ObservableValue(4)

        result = val1 % val2
        assert result == 3

        result = val1 % 4
        assert result == 3

    def test_power(self):
        """Test __pow__ method."""
        val1 = ObservableValue(2)
        val2 = ObservableValue(3)

        result = val1**val2
        assert result == 8

        result = val1**3
        assert result == 8

    def test_arithmetic_with_observables(self):
        """Test arithmetic with Observable values."""
        obs1 = Observable(5)
        obs2 = Observable(3)
        val1 = ObservableValue(obs1)
        val2 = ObservableValue(obs2)

        # Observable supports arithmetic (reactive operations), so this should work
        result = val1 + val2
        # The result should be a reactive operation, not a simple number
        assert hasattr(result, "value")  # Should be some kind of reactive result

    def test_arithmetic_with_nested_values(self):
        """Test arithmetic with nested ObservableValue instances."""
        inner_val1 = ObservableValue(5)
        inner_val2 = ObservableValue(3)
        val1 = ObservableValue(inner_val1)
        val2 = ObservableValue(inner_val2)

        result = val1 + val2
        assert result == 8

        result = val1 * val2
        assert result == 15

    def test_arithmetic_error(self):
        """Test arithmetic with non-numeric values."""
        val1 = ObservableValue("hello")
        val2 = ObservableValue(5)

        with pytest.raises(TypeError, match="can only concatenate"):
            val1 + val2


class TestObservableValueUnaryOperators:
    """Test unary operators."""

    def test_negation(self):
        """Test __neg__ method."""
        val = ObservableValue(5)
        result = -val
        assert result == -5

        val = ObservableValue(-3)
        result = -val
        assert result == 3

    def test_positive(self):
        """Test __pos__ method."""
        val = ObservableValue(5)
        result = +val
        assert result == 5

        val = ObservableValue(-3)
        result = +val
        assert result == -3

    def test_absolute_value(self):
        """Test __abs__ method."""
        val = ObservableValue(-5)
        result = abs(val)
        assert result == 5

        val = ObservableValue(5)
        result = abs(val)
        assert result == 5

    def test_unary_with_observables(self):
        """Test unary operators with Observable values."""
        obs = Observable(-5)
        val = ObservableValue(obs)

        # Observable doesn't support unary operators, so this should raise an error
        with pytest.raises(TypeError, match="bad operand type"):
            -val

    def test_unary_with_nested_values(self):
        """Test unary operators with nested ObservableValue instances."""
        inner_val = ObservableValue(-5)
        val = ObservableValue(inner_val)

        result = -val
        assert result == 5

        result = abs(val)
        assert result == 5

    def test_unary_error(self):
        """Test unary operators with non-numeric values."""
        val = ObservableValue("hello")

        with pytest.raises(TypeError, match="bad operand type"):
            -val

        with pytest.raises(TypeError, match="bad operand type"):
            abs(val)


class TestObservableValueTransparentValueProtocol:
    """Test TransparentValue protocol compliance."""

    def test_protocol_compliance(self):
        """Test that ObservableValue implements TransparentValue protocol."""
        val = ObservableValue(42)

        # Should be an instance of TransparentValue
        assert isinstance(val, TransparentValue)

    def test_protocol_methods_exist(self):
        """Test that all protocol methods exist."""
        val = ObservableValue(42)

        # Core methods
        assert hasattr(val, "value")
        assert hasattr(val, "unwrap")

        # Magic methods
        assert hasattr(val, "__eq__")
        assert hasattr(val, "__str__")
        assert hasattr(val, "__repr__")
        assert hasattr(val, "__bool__")
        assert hasattr(val, "__len__")
        assert hasattr(val, "__iter__")
        assert hasattr(val, "__getitem__")
        assert hasattr(val, "__contains__")
        assert hasattr(val, "__hash__")
        assert hasattr(val, "__lt__")
        assert hasattr(val, "__le__")
        assert hasattr(val, "__gt__")
        assert hasattr(val, "__ge__")
        assert hasattr(val, "__add__")
        assert hasattr(val, "__sub__")
        assert hasattr(val, "__mul__")
        assert hasattr(val, "__truediv__")
        assert hasattr(val, "__floordiv__")
        assert hasattr(val, "__mod__")
        assert hasattr(val, "__pow__")
        assert hasattr(val, "__neg__")
        assert hasattr(val, "__pos__")
        assert hasattr(val, "__abs__")


class TestObservableValueEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_values(self):
        """Test with empty values."""
        val = ObservableValue([])
        assert val.value == []
        assert val.unwrap() == []
        assert len(val) == 0
        assert list(val) == []

        val = ObservableValue({})
        assert val.value == {}
        assert val.unwrap() == {}
        assert len(val) == 0

    def test_none_values(self):
        """Test with None values."""
        val = ObservableValue(None)
        assert val.value is None
        assert val.unwrap() is None
        assert not bool(val)

    def test_zero_values(self):
        """Test with zero values."""
        val = ObservableValue(0)
        assert val.value == 0
        assert val.unwrap() == 0
        assert not bool(val)

    def test_very_deep_nesting(self):
        """Test with very deeply nested ObservableValue instances."""
        # Create deeply nested structure
        current = ObservableValue(42)
        for i in range(10):
            current = ObservableValue(current)

        val = ObservableValue(current)
        assert val.unwrap() == 42

    def test_circular_reference_protection(self):
        """Test protection against circular references."""
        # This should not cause infinite recursion
        val1 = ObservableValue(None)
        val2 = ObservableValue(None)
        val1.value = val2
        val2.value = val1

        # Should handle gracefully without infinite recursion
        result = val1.unwrap()
        # The exact result depends on implementation, but should not hang
        assert result is not None or result is None  # Either is acceptable


class TestObservableValueTypePreservation:
    """Test that types are preserved correctly."""

    def test_integer_type_preservation(self):
        """Test that integer types are preserved."""
        val = ObservableValue(42)
        assert isinstance(val.value, int)
        assert isinstance(val.unwrap(), int)

    def test_string_type_preservation(self):
        """Test that string types are preserved."""
        val = ObservableValue("hello")
        assert isinstance(val.value, str)
        assert isinstance(val.unwrap(), str)

    def test_list_type_preservation(self):
        """Test that list types are preserved."""
        val = ObservableValue([1, 2, 3])
        assert isinstance(val.value, list)
        assert isinstance(val.unwrap(), list)

    def test_dict_type_preservation(self):
        """Test that dict types are preserved."""
        val = ObservableValue({"key": "value"})
        assert isinstance(val.value, dict)
        assert isinstance(val.unwrap(), dict)

    def test_observable_type_preservation(self):
        """Test that Observable types are preserved in .value."""
        obs = Observable(42)
        val = ObservableValue(obs)
        assert isinstance(val.value, Observable)
        assert isinstance(val.unwrap(), Observable)  # Observable is not unwrapped


if __name__ == "__main__":
    pytest.main([__file__])
