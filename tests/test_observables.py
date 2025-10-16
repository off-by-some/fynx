"""Tests for basic observable functionality."""

import pytest
from fynx import Observable, observable, computed


def test_observable_creation_with_initial_value():
    """Test that an observable can be created with an initial value."""
    obs = Observable("test_key", "initial_value")
    assert obs.value == "initial_value"


def test_observable_creation_without_initial_value():
    """Test that an observable can be created without an initial value."""
    obs = Observable("test_key")
    assert obs.value is None


def test_observable_key_storage():
    """Test that observable key is stored correctly."""
    obs = Observable("my_key", "value")
    assert obs.key == "my_key"


def test_observable_value_update():
    """Test that observable value can be updated."""
    obs = Observable("test", "initial")
    obs.set("updated")
    assert obs.value == "updated"


def test_observable_string_representation():
    """Test string representation of observable."""
    obs = Observable("test", "value")
    assert str(obs) == "value"


def test_observable_repr():
    """Test repr representation of observable."""
    obs = Observable("test", "value")
    assert repr(obs) == "Observable('test', 'value')"


def test_observable_equality_with_same_value():
    """Test observable equality when values are the same."""
    obs1 = Observable("key1", "value")
    obs2 = Observable("key2", "value")
    assert obs1 == obs2


def test_observable_equality_with_different_values():
    """Test observable equality when values are different."""
    obs1 = Observable("key", "value1")
    obs2 = Observable("key", "value2")
    assert obs1 != obs2


def test_observable_equality_with_non_observable():
    """Test observable equality with non-observable object."""
    obs = Observable("key", "value")
    assert obs == "value"
    assert "value" == obs


def test_observable_no_value_change_when_same():
    """Test that setting same value doesn't trigger unnecessary updates."""
    obs = Observable("test", "value")
    # This should not cause any issues
    obs.set("value")
    assert obs.value == "value"


def test_observable_truthiness():
    """Test observable boolean conversion."""
    truthy_obs = Observable("test", "non_empty")
    falsy_obs = Observable("test", "")

    assert bool(truthy_obs) is True
    assert bool(falsy_obs) is False


def test_observable_hash_is_object_id():
    """Test that observable hash is based on object identity."""
    obs1 = Observable("key", "value")
    obs2 = Observable("key", "value")

    # Different objects should have different hashes
    assert hash(obs1) != hash(obs2)
    # Same object should have consistent hash
    assert hash(obs1) == hash(obs1)


def test_merged_observable_creation():
    """Test that merged observable can be created from multiple observables."""
    obs1 = Observable("key1", "value1")
    obs2 = Observable("key2", "value2")

    merged = obs1 | obs2
    assert len(merged) == 2
    assert merged[0] == "value1"
    assert merged[1] == "value2"


def test_merged_observable_tuple_access():
    """Test tuple-like access to merged observable values."""
    obs1 = Observable("key1", "a")
    obs2 = Observable("key2", "b")

    merged = obs1 | obs2
    assert merged.value == ("a", "b")


def test_merged_observable_iteration():
    """Test that merged observable can be iterated over."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)

    merged = obs1 | obs2
    values = list(merged)
    assert values == [1, 2]


def test_merged_observable_index_access():
    """Test index access to merged observable."""
    obs1 = Observable("key1", "first")
    obs2 = Observable("key2", "second")

    merged = obs1 | obs2
    assert merged[0] == "first"
    assert merged[1] == "second"


def test_merged_observable_index_update():
    """Test updating merged observable by index."""
    obs1 = Observable("key1", "first")
    obs2 = Observable("key2", "second")

    merged = obs1 | obs2
    merged[0] = "updated_first"

    assert obs1.value == "updated_first"
    assert obs2.value == "second"


def test_chained_merged_observables():
    """Test chaining multiple merged observables."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    obs3 = Observable("key3", 3)

    merged = obs1 | obs2 | obs3
    assert len(merged) == 3
    assert merged.value == (1, 2, 3)


def test_context_manager_with_merged_observables():
    """Test context manager usage with merged observables."""
    obs1 = Observable("key1", "hello")
    obs2 = Observable("key2", "world")

    merged = obs1 | obs2

    with merged as context:
        # Test that context allows unpacking
        name, greeting = context
        assert name == "hello"
        assert greeting == "world"

        # Test that context is callable for reactive behavior
        assert callable(context)


def test_context_manager_reactive_execution():
    """Test that context manager enables reactive execution."""
    obs1 = Observable("key1", 10)
    obs2 = Observable("key2", 20)

    merged = obs1 | obs2
    execution_count = 0

    def reactive_callback(a, b):
        nonlocal execution_count
        execution_count += 1
        # Note: callback receives current values
        if execution_count == 1:
            assert a == 10 and b == 20
        elif execution_count == 2:
            assert a == 15 and b == 20

    with merged as context:
        context(reactive_callback)

    # Should execute once immediately
    assert execution_count == 1

    # Changing values should trigger reactive execution
    obs1.set(15)
    assert execution_count == 2

    obs2.set(25)
    assert execution_count == 3


def test_context_manager_no_initial_execution():
    """Test context manager when not using callable interface."""
    obs1 = Observable("key1", "test")
    obs2 = Observable("key2", "value")

    merged = obs1 | obs2
    execution_count = 0

    with merged as context:
        # Just access values without calling
        val1, val2 = context
        assert val1 == "test"
        assert val2 == "value"

    # No reactive execution should happen
    obs1.set("changed")
    # execution_count should remain 0 since we didn't set up reactive callback


def test_observable_function_creates_observable():
    """Test that observable() function creates an Observable instance."""
    obs = observable("initial")
    assert isinstance(obs, Observable)
    assert obs.value == "initial"


def test_observable_function_without_initial_value():
    """Test observable() function without initial value."""
    obs = observable()
    assert isinstance(obs, Observable)
    assert obs.value is None


def test_observable_with_none_value():
    """Test observables with None values work correctly."""
    obs = Observable("test", None)
    assert obs.value is None

    obs.set("not_none")
    assert obs.value == "not_none"

    obs.set(None)
    assert obs.value is None


def test_observable_key_uniqueness():
    """Test that observables can have same keys (identity-based, not key-based)."""
    obs1 = Observable("same_key", "value1")
    obs2 = Observable("same_key", "value2")

    assert obs1.key == obs2.key
    assert obs1.value != obs2.value
    assert obs1 is not obs2


def test_observable_repr_with_special_characters():
    """Test repr of observables with special characters in values."""
    obs = Observable("test", "value\nwith\ttabs")
    repr_str = repr(obs)
    assert "test" in repr_str
    assert "value\\nwith\\ttabs" in repr_str


def test_empty_string_observable():
    """Test observables with empty string values."""
    obs = Observable("empty", "")
    assert obs.value == ""
    assert bool(obs) is False  # Empty string is falsy

    obs.set("non_empty")
    assert obs.value == "non_empty"
    assert bool(obs) is True


def test_zero_value_observable():
    """Test observables with zero values."""
    obs = Observable("zero", 0)
    assert obs.value == 0
    assert bool(obs) is False  # Zero is falsy

    obs.set(1)
    assert obs.value == 1
    assert bool(obs) is True


def test_extreme_values_in_observables():
    """Test observables with extreme values."""
    # Very large numbers
    obs = Observable("large", float('inf'))
    assert obs.value == float('inf')

    # Very small numbers
    obs.set(float('-inf'))
    assert obs.value == float('-inf')

    # NaN
    obs.set(float('nan'))
    assert obs.value != obs.value  # NaN != NaN


def test_observable_set_same_value_no_trigger():
    """Test that setting the same value doesn't trigger unnecessary updates."""
    obs = Observable("test", "value")
    trigger_count = 0

    def callback(value):
        nonlocal trigger_count
        trigger_count += 1

    obs.subscribe(callback)

    # Setting same value should not trigger
    obs.set("value")
    assert trigger_count == 0

    # Setting different value should trigger
    obs.set("different")
    assert trigger_count == 1


def test_unicode_values_in_observables():
    """Test observables with unicode string values."""
    unicode_value = "Hello 世界 🌍"
    obs = Observable("unicode", unicode_value)
    assert obs.value == unicode_value

    computed_obs = computed(lambda x: x.upper(), obs)
    assert computed_obs.value == unicode_value.upper()
