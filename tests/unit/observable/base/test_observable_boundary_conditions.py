"""Edge case tests for observable base functionality."""

import gc
import weakref

import pytest

from fynx import Observable


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_handles_rapid_successive_updates():
    """Observable correctly handles rapid successive value changes"""
    # Arrange
    obs = Observable("rapid", 0)
    received_values = []

    def record_value(value):
        received_values.append(value)

    obs.subscribe(record_value)

    # Act - Rapid updates (starts at 0, sets to 0-99, but setting to same value doesn't notify)
    for i in range(100):
        obs.set(i)

    # Assert - Final value is correct, and notifications were sent for actual changes
    assert obs.value == 99
    assert (
        len(received_values) == 99
    )  # 99 notifications (1-99, since 0 was already the value)
    assert received_values[-1] == 99


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_with_none_initial_value_operations():
    """Observable initialized with None handles all operations correctly"""
    # Arrange
    obs = Observable("none_test", None)

    # Act & Assert
    assert obs.value is None

    obs.set("not_none")
    assert obs.value == "not_none"

    obs.set(None)
    assert obs.value is None

    # Test with complex objects
    obs.set({"key": "value"})
    assert obs.value == {"key": "value"}


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_preserves_extreme_numeric_values():
    """Observable correctly handles extreme numeric values without precision loss"""
    # Arrange
    obs = Observable("extreme", 0)

    # Test very large integers
    large_int = 10**18
    obs.set(large_int)
    assert obs.value == large_int

    # Test very small decimals
    small_decimal = 10**-10
    obs.set(small_decimal)
    assert obs.value == small_decimal

    # Test infinity
    obs.set(float("inf"))
    assert obs.value == float("inf")

    # Test negative infinity
    obs.set(float("-inf"))
    assert obs.value == float("-inf")


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_with_mutable_containers():
    """Observable correctly handles mutable container values"""
    # Arrange
    obs = Observable("container", [1, 2, 3])
    notifications = []

    def track_changes(value):
        notifications.append(value.copy() if hasattr(value, "copy") else value)

    obs.subscribe(track_changes)

    # Act - Modify the list in place (not through set())
    current_list = obs.value
    current_list.append(4)

    # Assert - Observable doesn't auto-detect in-place mutations
    # This tests the expected behavior: observables track explicit set() calls only
    assert obs.value == [1, 2, 3, 4]
    assert len(notifications) == 0  # No notification for in-place change

    # Act - Proper update via set()
    obs.set([1, 2, 3, 4, 5])

    # Assert - Now notification is sent
    assert len(notifications) == 1
    assert notifications[0] == [1, 2, 3, 4, 5]


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.memory
def test_observable_subscription_cleanup_with_weak_references(no_leaks):
    """Observable subscriptions can be properly cleaned up using weak references"""
    # Arrange
    obs = Observable("cleanup_test", "initial")
    callback_refs = []

    def create_callbacks():
        callbacks_created = []
        for i in range(10):

            def callback(value, idx=i):
                pass

            callbacks_created.append(callback)
            obs.subscribe(callback)
            # Keep weak reference to callback
            callback_refs.append(weakref.ref(callback))

        return callbacks_created

    # Act - Create callbacks, unsubscribe them, then delete references
    callbacks = create_callbacks()

    # Unsubscribe all callbacks
    for callback in callbacks:
        obs.unsubscribe(callback)

    # Delete callback references
    del callbacks

    # Force garbage collection
    gc.collect()

    # Assert - Callbacks should be garbage collected after unsubscribing
    collected_count = sum(1 for ref in callback_refs if ref() is None)
    assert collected_count > 0  # At least some callbacks were collected


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_key_edge_cases():
    """Observable handles various key types and edge cases"""
    # Test different key types
    obs_int = Observable(42, "int_key")
    obs_str = Observable("string_key", "str_key")
    obs_tuple = Observable(("compound", "key"), "tuple_key")

    assert obs_int.key == 42
    assert obs_str.key == "string_key"
    assert obs_tuple.key == ("compound", "key")

    # Test special characters in keys
    obs_special = Observable("key\nwith\ttabs", "special")
    assert obs_special.key == "key\nwith\ttabs"

    # Test empty key (gets converted to default name)
    obs_empty = Observable("", "empty_key")
    assert obs_empty.key == "<unnamed>"
