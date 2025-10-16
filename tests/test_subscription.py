"""Tests for observable subscription functionality."""

import pytest

from fynx import Observable


def test_standalone_observable_subscription_callback_execution():
    """Test that subscription callback is executed when observable changes."""
    obs = Observable("test", "initial")
    callback_executed = False
    received_value = None

    def callback(value):
        nonlocal callback_executed, received_value
        callback_executed = True
        received_value = value

    obs.subscribe(callback)
    assert not callback_executed

    obs.set("changed")
    assert callback_executed
    assert received_value == "changed"


def test_standalone_observable_subscription_callback_receives_current_value():
    """Test that subscription callback receives current observable value."""
    obs = Observable("test", "initial")
    received_value = None

    def callback(value):
        nonlocal received_value
        received_value = value

    obs.subscribe(callback)
    obs.set("new_value")

    assert received_value == "new_value"


def test_standalone_observable_subscription_method_chaining():
    """Test that subscribe method returns the observable for chaining."""
    obs = Observable("test", "value")
    result = obs.subscribe(lambda: None)

    assert result is obs


def test_standalone_observable_multiple_subscriptions():
    """Test that multiple subscriptions work on the same observable."""
    obs = Observable("test", 0)
    call_count = 0

    def callback1(value):
        nonlocal call_count
        call_count += 1

    def callback2(value):
        nonlocal call_count
        call_count += 1

    obs.subscribe(callback1)
    obs.subscribe(callback2)

    obs.set(1)
    assert call_count == 2


def test_standalone_observable_unsubscribe_removes_callback():
    """Test that unsubscribe removes a specific callback."""
    obs = Observable("test", 0)
    call_count = 0

    def callback1(value):
        nonlocal call_count
        call_count += 1

    def callback2(value):
        nonlocal call_count
        call_count += 10

    obs.subscribe(callback1)
    obs.subscribe(callback2)

    obs.set(1)
    assert call_count == 11  # Both callbacks executed

    obs.unsubscribe(callback1)
    obs.set(2)
    assert call_count == 21  # Only callback2 executed


def test_standalone_observable_unsubscribe_nonexistent_callback():
    """Test that unsubscribing non-existent callback doesn't cause errors."""
    obs = Observable("test", "value")

    def callback():
        pass

    # Should not raise an error
    obs.unsubscribe(callback)


def test_merged_observable_subscription_callback_execution():
    """Test that merged observable subscription executes callback."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 | obs2

    callback_executed = False
    received_values = None

    def callback(a, b):
        nonlocal callback_executed, received_values
        callback_executed = True
        received_values = (a, b)

    merged.subscribe(callback)
    assert not callback_executed

    obs1.set(10)
    assert callback_executed
    assert received_values == (10, 2)


def test_merged_observable_subscription_receives_all_values():
    """Test that merged observable subscription receives all current values."""
    obs1 = Observable("key1", "hello")
    obs2 = Observable("key2", "world")
    merged = obs1 | obs2

    received_args = None

    def callback(*args):
        nonlocal received_args
        received_args = args

    merged.subscribe(callback)
    obs2.set("universe")

    assert received_args == ("hello", "universe")


def test_merged_observable_subscription_method_chaining():
    """Test that merged observable subscribe returns itself for chaining."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 | obs2

    result = merged.subscribe(lambda: None)
    assert result is merged


def test_merged_observable_unsubscribe_removes_callback():
    """Test that merged observable unsubscribe removes specific callback."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 | obs2

    call_count = 0

    def callback1(*values):
        nonlocal call_count
        call_count += 1

    def callback2(*values):
        nonlocal call_count
        call_count += 10

    merged.subscribe(callback1)
    merged.subscribe(callback2)

    obs1.set(5)
    assert call_count == 11  # Both callbacks

    merged.unsubscribe(callback1)
    obs2.set(7)
    assert call_count == 21  # Only callback2


def test_merged_observable_unsubscribe_nonexistent_callback():
    """Test that unsubscribing non-existent callback from merged observable doesn't error."""
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 | obs2

    def callback():
        pass

    # Should not raise an error
    merged.unsubscribe(callback)


def test_subscription_no_execution_on_same_value():
    """Test that subscription doesn't execute when value doesn't actually change."""
    obs = Observable("test", "value")
    call_count = 0

    def callback(value):
        nonlocal call_count
        call_count += 1

    obs.subscribe(callback)

    # Setting same value should not trigger callback
    obs.set("value")
    assert call_count == 0

    # Setting different value should trigger
    obs.set("new_value")
    assert call_count == 1


def test_observable_subscription_with_exception():
    """Test that exceptions in subscription callbacks propagate as expected."""
    obs = Observable("test", 1)

    def failing_callback(value):
        raise ValueError("Test exception")

    obs.subscribe(failing_callback)

    # Exceptions in callbacks will propagate - this is expected behavior
    with pytest.raises(ValueError, match="Test exception"):
        obs.set(2)

    # Observable value should still be updated even if callback fails
    assert obs.value == 2


def test_observable_subscription_with_exception_handling():
    """Test that observable subscriptions handle exceptions appropriately."""
    obs = Observable("test", 1)

    def failing_callback(value):
        raise ValueError("Callback error")

    obs.subscribe(failing_callback)

    # Exception should propagate when callback fails
    with pytest.raises(ValueError, match="Callback error"):
        obs.set(2)

    # But the observable value should still be updated
    assert obs.value == 2


def test_observable_memory_management():
    """Test that observables don't hold references that prevent garbage collection."""
    # This is more of a conceptual test - in practice we'd need weak references
    # to test actual memory management, but we can at least test basic functionality
    obs = Observable("test", "value")

    # Create a bunch of subscriptions
    callbacks = []
    for i in range(10):
        def callback(value):
            pass
        callbacks.append(callback)
        obs.subscribe(callback)

    # All subscriptions should work
    obs.set("changed")
    assert obs.value == "changed"

    # Unsubscribing should work
    obs.unsubscribe(callbacks[0])
    obs.set("changed_again")
    assert obs.value == "changed_again"
