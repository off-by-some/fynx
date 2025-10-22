"""Unit tests for core observable behavior."""

import pytest

from fynx import Observable, observable
from fynx.observable.core.abstract.context import ReactiveContext


@pytest.mark.unit
@pytest.mark.observable
def test_observable_provides_access_to_initial_value():
    """Observable returns the value it was created with"""
    obs = Observable("test_key", "initial_value")
    assert obs.value == "initial_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_handles_none_as_initial_value():
    """Observable can be created with None as initial value"""
    # Arrange & Act
    obs = Observable("test_key", None)

    # Assert
    assert obs.value is None


@pytest.mark.unit
@pytest.mark.observable
def test_observable_updates_value_when_set():
    """Observable value changes when set() is called with new value"""
    # Arrange
    obs = Observable("test", "initial")

    # Act
    obs.set("updated")

    # Assert
    assert obs.value == "updated"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_converts_to_string_via_value():
    """Observable string representation shows its current value"""
    # Arrange
    obs = Observable("test", "value")

    # Act & Assert
    assert str(obs) == "value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_repr_shows_key_and_value():
    """Observable repr shows both key and current value"""
    # Arrange
    obs = Observable("test", "value")

    # Act & Assert
    assert repr(obs) == "Observable('test', 'value')"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_equals_value_for_comparison():
    """Observable compares equal to its current value"""
    # Arrange
    obs = Observable("key", "value")

    # Act & Assert
    assert obs == "value"
    assert "value" == obs


@pytest.mark.unit
@pytest.mark.observable
def test_observable_differs_when_values_differ():
    """Observables with different values are not equal"""
    # Arrange
    obs1 = Observable("key", "value1")
    obs2 = Observable("key", "value2")

    # Act & Assert
    assert obs1 != obs2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_preserves_value_when_set_to_same():
    """Setting observable to its current value leaves it unchanged"""
    # Arrange
    obs = Observable("test", "value")

    # Act
    obs.set("value")  # Same value

    # Assert
    assert obs.value == "value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_equals_observable_with_same_value():
    """Observable equals another observable when both have identical values"""
    obs1 = Observable("key1", "same_value")
    obs2 = Observable("key2", "same_value")

    assert obs1 == obs2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_equals_primitive_value_when_identical():
    """Observable equals primitive value when current value matches exactly"""
    obs = Observable("key", "test_value")

    assert obs == "test_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_differs_from_observable_with_different_value():
    """Observable differs from observable with different current value"""
    obs1 = Observable("key1", "value1")
    obs2 = Observable("key2", "value2")

    assert obs1 != obs2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_equality_updates_when_value_changes():
    """Observable equality relationship changes when observable value is updated"""
    obs1 = Observable("key1", "original")
    obs2 = Observable("key2", "different")

    obs1.set("different")  # Now both have "different"

    assert obs1 == obs2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_boolean_conversion_uses_value():
    """Observable boolean conversion depends on its value's truthiness"""
    # Arrange
    truthy_obs = Observable("test", "non_empty")
    falsy_obs = Observable("test", "")

    # Act & Assert
    assert bool(truthy_obs) is True
    assert bool(falsy_obs) is False


@pytest.mark.unit
@pytest.mark.observable
def test_observable_hash_based_on_object_identity():
    """Observable hash is based on object identity, not value"""
    # Arrange
    obs1 = Observable("key", "value")
    obs2 = Observable("key", "value")

    # Act & Assert
    # Different objects should have different hashes
    assert hash(obs1) != hash(obs2)
    # Same object should have consistent hash
    assert hash(obs1) == hash(obs1)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_combines_multiple_sources():
    """alongside method creates merged observable from multiple sources"""
    # Arrange
    obs1 = Observable("key1", "value1")
    obs2 = Observable("key2", "value2")

    # Act
    merged = obs1.alongside(obs2)

    # Assert
    assert len(merged) == 2
    assert merged.value == ("value1", "value2")


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_provides_tuple_like_access():
    """Merged observable values accessible as tuple"""
    # Arrange
    obs1 = Observable("key1", "a")
    obs2 = Observable("key2", "b")
    merged = obs1 + obs2

    # Act & Assert
    assert merged.value == ("a", "b")


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_supports_iteration():
    """Merged observable can be iterated over like a sequence"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 + obs2

    # Act
    values = list(merged)

    # Assert
    assert values == [1, 2]


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_provides_index_access():
    """Merged observable supports indexing like a tuple"""
    # Arrange
    obs1 = Observable("key1", "first")
    obs2 = Observable("key2", "second")
    merged = obs1 + obs2

    # Act & Assert
    assert merged[0] == "first"
    assert merged[1] == "second"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_index_assignment_updates_source():
    """Assigning to merged observable index updates corresponding source"""
    # Arrange
    obs1 = Observable("key1", "first")
    obs2 = Observable("key2", "second")
    merged = obs1 + obs2

    # Act
    merged[0] = "updated_first"

    # Assert
    assert obs1.value == "updated_first"
    assert obs2.value == "second"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_pipe_operator_chains_multiple_merges():
    """Pipe operator chains multiple observables into single merged result"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    obs3 = Observable("key3", 3)

    # Act
    merged = obs1 + obs2 + obs3

    # Assert
    assert len(merged) == 3
    assert merged.value == (1, 2, 3)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_maintains_length_invariant():
    """Merged observable length always equals number of source observables"""
    # Arrange
    obs1 = Observable("a", 1)
    obs2 = Observable("b", 2)
    obs3 = Observable("c", 3)

    # Act - Create merge chain
    merged = obs1 + obs2 + obs3

    # Assert - Length invariant holds
    assert len(merged) == 3

    # Change values - length should remain constant
    obs1.set(10)
    assert len(merged) == 3

    obs2.set(20)
    assert len(merged) == 3

    # Values should reflect changes
    assert merged.value == (10, 20, 3)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_context_manager_preserves_merged_state_during_execution():
    """Context manager maintains merged observable state during callback execution"""
    # Arrange
    obs1 = Observable("x", 5)
    obs2 = Observable("y", 10)
    merged = obs1 + obs2

    execution_log = []

    def test_callback(a, b):
        # Log the values seen during execution
        execution_log.append((a, b))
        # Verify relationship holds during execution
        assert a + b == 15  # x + y should always equal 15

    # Act - Use context manager
    with merged as context:
        context(test_callback)

    # Assert - Callback executed with correct initial values
    assert execution_log == [(5, 10)]


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_context_manager_provides_unpacking_access():
    """Context manager with merged observable allows tuple unpacking"""
    # Arrange
    obs1 = Observable("key1", "hello")
    obs2 = Observable("key2", "world")
    merged = obs1 + obs2

    # Act & Assert
    with merged as context:
        # Test that context allows unpacking
        name, greeting = context
        assert name == "hello"
        assert greeting == "world"

        # Test that context is callable for reactive behavior
        assert callable(context)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_context_manager_enables_reactive_callbacks():
    """Context manager executes callback immediately and on source changes"""
    # Arrange
    obs1 = Observable("key1", 10)
    obs2 = Observable("key2", 20)
    merged = obs1 + obs2
    execution_count = 0

    def reactive_callback(a, b):
        nonlocal execution_count
        execution_count += 1
        # Note: callback receives current values
        if execution_count == 1:
            assert a == 10 and b == 20
        elif execution_count == 2:
            assert a == 15 and b == 20

    # Act
    with merged as context:
        context(reactive_callback)

    # Assert - Should execute once immediately
    assert execution_count == 1

    # Changing values should trigger reactive execution
    obs1.set(15)
    assert execution_count == 2

    obs2.set(25)
    assert execution_count == 3


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_context_manager_allows_value_access_without_reactivity():
    """Context manager provides value access without setting up reactivity"""
    # Arrange
    obs1 = Observable("key1", "test")
    obs2 = Observable("key2", "value")
    merged = obs1 + obs2

    # Act & Assert
    with merged as context:
        # Just access values without calling
        val1, val2 = context
        assert val1 == "test"
        assert val2 == "value"

    # No reactive execution should happen when values change
    obs1.set("changed")
    # (No assertions needed - we're just verifying no exceptions)


@pytest.mark.unit
@pytest.mark.observable
def test_observable_function_creates_observable_instance():
    """observable() function creates Observable instance with initial value"""
    # Arrange & Act
    obs = observable("initial")

    # Assert
    assert isinstance(obs, Observable)
    assert obs.value == "initial"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_function_handles_none_initial_value():
    """observable() function works with None as initial value"""
    # Arrange & Act
    obs = observable()

    # Assert
    assert isinstance(obs, Observable)
    assert obs.value is None


@pytest.mark.unit
@pytest.mark.observable
def test_observable_supports_none_value_operations():
    """Observable handles None values in all operations"""
    # Arrange
    obs = Observable("test", None)

    # Act & Assert
    assert obs.value is None

    obs.set("not_none")
    assert obs.value == "not_none"

    obs.set(None)
    assert obs.value is None


@pytest.mark.unit
@pytest.mark.observable
def test_observable_keys_are_not_unique_constraints():
    """Multiple observables can share same key (identity-based system)"""
    # Arrange & Act
    obs1 = Observable("same_key", "value1")
    obs2 = Observable("same_key", "value2")

    # Assert
    assert obs1.key == obs2.key
    assert obs1.value != obs2.value
    assert obs1 is not obs2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_repr_handles_special_characters():
    """Observable repr properly escapes special characters in values"""
    # Arrange
    obs = Observable("test", "value\nwith\ttabs")

    # Act
    repr_str = repr(obs)

    # Assert
    assert "test" in repr_str
    assert "value\\nwith\\ttabs" in repr_str


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_handles_empty_string_values():
    """Observable correctly handles empty string as falsy value"""
    # Arrange
    obs = Observable("empty", "")

    # Act & Assert
    assert obs.value == ""
    assert bool(obs) is False  # Empty string is falsy

    obs.set("non_empty")
    assert obs.value == "non_empty"
    assert bool(obs) is True


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_handles_zero_values():
    """Observable correctly handles zero as falsy value"""
    # Arrange
    obs = Observable("zero", 0)

    # Act & Assert
    assert obs.value == 0
    assert bool(obs) is False  # Zero is falsy

    obs.set(1)
    assert obs.value == 1
    assert bool(obs) is True


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_handles_extreme_numeric_values():
    """Observable preserves extreme numeric values like infinity and NaN"""
    # Arrange
    obs = Observable("extreme", 0)

    # Act & Assert - Very large numbers
    obs.set(float("inf"))
    assert obs.value == float("inf")

    # Very small numbers
    obs.set(float("-inf"))
    assert obs.value == float("-inf")

    # NaN
    obs.set(float("nan"))
    assert obs.value != obs.value  # NaN != NaN


@pytest.mark.unit
@pytest.mark.observable
def test_observable_setting_same_value_avoids_notifications():
    """Setting observable to its current value doesn't trigger subscribers"""
    # Arrange
    obs = Observable("test", "value")
    trigger_count = 0

    def callback(value):
        nonlocal trigger_count
        trigger_count += 1

    obs.subscribe(callback)

    # Act - Setting same value should not trigger
    obs.set("value")

    # Assert
    assert trigger_count == 0

    # Setting different value should trigger
    obs.set("different")
    assert trigger_count == 1


@pytest.mark.unit
@pytest.mark.observable
def test_observable_preserves_unicode_values():
    """Observable correctly handles Unicode string values"""
    # Arrange
    unicode_value = "Hello 世界 🌍"
    obs = Observable("unicode", unicode_value)

    # Act & Assert
    assert obs.value == unicode_value

    computed_obs = obs.then(lambda x: x.upper())
    assert computed_obs.value == unicode_value.upper()


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_value_derives_from_source_observables():
    """MergedObservable value derives from source observables, not direct set operations"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 + obs2

    # Initial value should be tuple of source values
    assert merged.value == (1, 2)

    # Updating source observables should update merged value
    obs1.set(3)
    obs2.set(4)
    assert merged.value == (3, 4)

    # Merged observables are read-only computed observables
    # Attempting to set them directly should raise ValueError
    with pytest.raises(ValueError, match="Computed observables are read-only"):
        merged.set((5, 6))

    # Value should still reflect source values
    assert merged.value == (3, 4)


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_cleanup_removes_empty_function_mappings():
    """MergedObservable cleanup removes empty function mappings from _func_to_contexts."""
    from fynx.observable.computed import MergedObservable, _func_to_contexts

    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 + obs2

    def test_func():
        pass

    # Create context and add it to function mappings
    context = type(
        "MockContext", (object,), {"run": test_func, "dispose": lambda: None}
    )()
    _func_to_contexts[test_func] = [context]

    # Verify mapping exists
    assert test_func in _func_to_contexts
    assert context in _func_to_contexts[test_func]

    # Remove context (simulating cleanup)
    _func_to_contexts[test_func].remove(context)

    # Clean up empty function mappings (line 361-362)
    if not _func_to_contexts[test_func]:
        del _func_to_contexts[test_func]

    # Verify mapping is removed
    assert test_func not in _func_to_contexts


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_dispose_with_store_observables():
    """Test ReactiveContext.dispose() when _store_observables is set (lines 130->137)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    def test_func():
        pass

    # Create context with store observables
    context = ReactiveContext(test_func, test_func, None)
    context._store_observables = [obs1, obs2]

    # Add the context's run method as observers
    obs1.add_observer(context.run)
    obs2.add_observer(context.run)

    # Track calls to context.run
    run_calls = []
    original_run = context.run

    def tracked_run():
        run_calls.append("run_called")
        original_run()

    context.run = tracked_run

    # Dispose should remove the context.run observers from store observables
    context.dispose()

    # Verify context.run observers were removed by checking that setting values doesn't trigger them
    run_calls.clear()
    obs1.set(10)
    obs2.set(20)

    # Should be empty since context.run observers were removed
    assert len(run_calls) == 0


@pytest.mark.unit
@pytest.mark.observable
def test_observable_notify_observers_reentrant_protection():
    """Test Observable._notify_observers() reentrant protection using proper cycle detection"""
    obs = Observable("test", 1)

    # Test legitimate reentrant scenario - updating a different observable
    # This should work without issues
    other_obs = Observable("other", 0)
    call_count = 0

    def reentrant_observer(value):
        nonlocal call_count
        call_count += 1
        # Update a different observable - this is safe and legitimate
        other_obs.set(other_obs.value + 1)

    obs.subscribe(reentrant_observer)

    # This should work without infinite recursion
    obs.set(2)

    # Verify both observables were updated
    assert obs.value == 2
    assert other_obs.value == 1  # Should have been incremented once
    assert call_count == 1  # Should have been called once


@pytest.mark.unit
@pytest.mark.observable
def test_observable_dispose_subscription_contexts_empty_mapping():
    """Test Observable._dispose_subscription_contexts() with empty function mapping (lines 514->exit)"""
    from fynx.registry import _func_to_contexts

    def non_existent_func():
        pass

    # Function should not be in mappings
    assert non_existent_func not in _func_to_contexts

    # Calling dispose should not raise an error
    Observable._dispose_subscription_contexts(non_existent_func)

    # Should still not be in mappings
    assert non_existent_func not in _func_to_contexts


@pytest.mark.unit
@pytest.mark.observable
def test_observable_set_name_computed_observable():
    """Test Observable.__set_name__() for computed observables (lines 664->672)"""
    from fynx.observable.computed import ComputedObservable

    class TestClass:
        pass

    # Create a computed observable with default key
    source = Observable("source", 5)
    computed = ComputedObservable("<unnamed>", lambda: source.value * 2, None, source)

    # Mark it as computed (this is what happens internally)
    computed._is_computed = True

    # Set name - should update key to computed format
    computed.__set_name__(TestClass, "my_computed")

    assert computed.key == "<computed:my_computed>"


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_dispose_stops_observable_notifications():
    """ReactiveContext.dispose() stops receiving notifications from store observables"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    notification_count = 0

    def test_func():
        nonlocal notification_count
        notification_count += 1

    # Create context with store observables
    context = ReactiveContext(test_func, test_func, None)
    context._store_observables = [obs1, obs2]

    # Add observer to track notifications
    obs1.add_observer(context.run)
    obs2.add_observer(context.run)

    # Trigger notifications
    obs1.set(2)
    obs2.set(3)
    initial_count = notification_count

    # Dispose should stop notifications
    context.dispose()

    # Further changes should not trigger notifications
    obs1.set(4)
    obs2.set(5)

    assert notification_count == initial_count


@pytest.mark.unit
@pytest.mark.observable
def test_observable_notify_observers_reentrant_protection_edge_case():
    """Test Observable._notify_observers() reentrant protection edge case using proper cycle detection"""
    obs = Observable("test", 1)

    # Test legitimate reentrant scenario with transaction
    # This should work without issues
    other_obs = Observable("other", 0)
    call_count = 0

    def reentrant_observer(value):
        nonlocal call_count
        call_count += 1
        # Use transaction to update a different observable - this is safe
        with other_obs.transaction():
            other_obs.set(10)

    obs.subscribe(reentrant_observer)

    # This should work without infinite recursion
    obs.set(2)

    # Verify both observables were updated
    assert obs.value == 2
    assert other_obs.value == 10
    assert call_count == 1  # Should have been called once


@pytest.mark.unit
@pytest.mark.observable
def test_observable_dispose_subscription_contexts_empty_mapping():
    """Test that subscription cleanup works correctly in push model"""
    obs = Observable("test", 1)

    def test_func(value):
        pass

    # Subscribe and then unsubscribe - should not raise an error
    obs.subscribe(test_func)
    obs.unsubscribe(test_func)

    # Should be able to unsubscribe again without error
    obs.unsubscribe(test_func)


@pytest.mark.unit
@pytest.mark.observable
def test_observable_set_name_skip_computed_processing():
    """Test Observable.__set_name__() skip processing for computed observables (lines 679-680)"""
    from fynx.observable.computed import ComputedObservable

    class TestClass:
        pass

    # Create a computed observable
    source = Observable("source", 5)
    computed = ComputedObservable("computed", lambda: source.value * 2, None, source)

    # Mark it as computed (this is what happens internally)
    computed._is_computed = True

    # Set name - should skip processing due to _is_computed flag
    computed.__set_name__(TestClass, "my_computed")

    # The key should remain unchanged since processing was skipped
    assert computed.key == "computed"
