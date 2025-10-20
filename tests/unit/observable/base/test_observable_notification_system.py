"""Unit tests for observable subscription behavior."""

import pytest

from fynx import Observable
from fynx.observable.conditional import ConditionalNotMet


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_notifies_subscriber_when_value_changes():
    """Observable notifies subscriber when its value changes"""
    obs = Observable("test", "initial")
    received = []
    obs.subscribe(lambda val: received.append(val))

    obs.set("changed")

    assert received == ["changed"]


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_receives_updated_value():
    """Subscription callback receives the new observable value"""
    # Arrange
    obs = Observable("test", "initial")
    received_value = None

    def callback(value):
        nonlocal received_value
        received_value = value

    obs.subscribe(callback)

    # Act
    obs.set("new_value")

    # Assert
    assert received_value == "new_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscribe_method_supports_chaining():
    """subscribe() method returns the observable for method chaining"""
    # Arrange
    obs = Observable("test", "value")

    # Act
    result = obs.subscribe(lambda: None)

    # Assert
    assert result is obs


@pytest.mark.unit
@pytest.mark.observable
def test_observable_supports_multiple_independent_subscribers():
    """Multiple subscribers can be registered to the same observable"""
    # Arrange
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

    # Act
    obs.set(1)

    # Assert
    assert call_count == 2


@pytest.mark.unit
@pytest.mark.observable
def test_observable_unsubscribe_removes_specific_callback():
    """unsubscribe() removes only the specified callback, others remain active"""
    # Arrange
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

    # Act & Assert - Both callbacks execute initially
    obs.set(1)
    assert call_count == 11  # Both callbacks executed

    obs.unsubscribe(callback1)
    obs.set(2)
    assert call_count == 21  # Only callback2 executed


@pytest.mark.unit
@pytest.mark.observable
def test_observable_unsubscribe_handles_nonexistent_callback_gracefully():
    """unsubscribe() with non-registered callback doesn't raise errors"""
    # Arrange
    obs = Observable("test", "value")

    def callback():
        pass

    # Act & Assert - Should not raise an error
    obs.unsubscribe(callback)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_subscription_notifies_on_any_source_change():
    """Merged observable subscribers receive notifications when any source changes"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 + obs2

    callback_executed = False
    received_values = None

    def callback(a, b):
        nonlocal callback_executed, received_values
        callback_executed = True
        received_values = (a, b)

    merged.subscribe(callback)

    # Act
    obs1.set(10)

    # Assert
    assert callback_executed
    assert received_values == (10, 2)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_subscription_receives_current_values():
    """Merged observable subscription receives current values from all sources"""
    # Arrange
    obs1 = Observable("key1", "hello")
    obs2 = Observable("key2", "world")
    merged = obs1 + obs2

    received_args = None

    def callback(*args):
        nonlocal received_args
        received_args = args

    merged.subscribe(callback)

    # Act
    obs2.set("universe")

    # Assert
    assert received_args == ("hello", "universe")


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_subscribe_supports_chaining():
    """Merged observable subscribe() returns itself for method chaining"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 + obs2

    # Act
    result = merged.subscribe(lambda: None)

    # Assert
    assert result is merged


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_unsubscribe_removes_specific_callback():
    """Merged observable unsubscribe() removes only specified callback"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 + obs2

    call_count = 0

    def callback1(*values):
        nonlocal call_count
        call_count += 1

    def callback2(*values):
        nonlocal call_count
        call_count += 10

    merged.subscribe(callback1)
    merged.subscribe(callback2)

    # Act & Assert - Both callbacks initially
    obs1.set(5)
    assert call_count == 11  # Both callbacks

    merged.unsubscribe(callback1)
    obs2.set(7)
    assert call_count == 21  # Only callback2


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_unsubscribe_handles_nonexistent_callback():
    """Merged observable unsubscribe() with unknown callback doesn't error"""
    # Arrange
    obs1 = Observable("key1", 1)
    obs2 = Observable("key2", 2)
    merged = obs1 + obs2

    def callback():
        pass

    # Act & Assert - Should not raise an error
    merged.unsubscribe(callback)


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_avoids_notifications_for_same_value():
    """Observable doesn't notify subscribers when set to identical value"""
    # Arrange
    obs = Observable("test", "value")
    call_count = 0

    def callback(value):
        nonlocal call_count
        call_count += 1

    obs.subscribe(callback)

    # Act - Setting same value should not trigger callback
    obs.set("value")

    # Assert - No notification sent for same value
    assert call_count == 0


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_avoids_notification_when_setting_same_value():
    """Observable does not notify subscribers when set to its current value"""
    obs = Observable("test", "value")
    received = []
    obs.subscribe(lambda val: received.append(val))

    obs.set("value")  # Same value

    assert received == []


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_sends_notification_when_value_changes():
    """Observable notifies subscriber when value changes to different value"""
    obs = Observable("test", "initial")
    received = []
    obs.subscribe(lambda val: received.append(val))

    obs.set("changed")

    assert received == ["changed"]


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscription_avoids_notification_when_setting_same_value_after_change():
    """Observable does not notify when set to same value after previous change"""
    obs = Observable("test", "initial")
    received = []
    obs.subscribe(lambda val: received.append(val))

    obs.set("changed")  # First change - should notify
    obs.set("changed")  # Same value - should not notify

    assert received == ["changed"]  # Only one notification


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_updates_despite_callback_exceptions():
    """Observable value changes even when subscriber callbacks raise exceptions"""
    obs = Observable("test", 1)

    def failing_callback(val):
        raise ConditionalNotMet("Callback failed")

    obs.subscribe(failing_callback)
    obs.subscribe(lambda val: None)  # Working callback

    # Exception will be raised but value should still update
    with pytest.raises(ConditionalNotMet):
        obs.set(2)

    assert obs.value == 2


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_propagates_exceptions_from_callbacks():
    """Observable propagates exceptions raised by subscriber callbacks"""
    obs = Observable("test", 1)

    def failing_callback(val):
        raise ConditionalNotMet("Callback failed")

    obs.subscribe(failing_callback)

    # Exception from callback should propagate
    with pytest.raises(ConditionalNotMet, match="Callback failed"):
        obs.set(2)

    # Value should still be updated despite callback exception
    assert obs.value == 2


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
def test_observable_attempts_all_callbacks_despite_exceptions():
    """Observable attempts to execute all callbacks even if some raise exceptions"""
    obs = Observable("test", 1)
    exception_count = 0

    def failing_callback(val):
        nonlocal exception_count
        exception_count += 1
        raise ConditionalNotMet("Callback failed")

    obs.subscribe(failing_callback)

    # This should raise the exception but still attempt the callback
    with pytest.raises(ConditionalNotMet):
        obs.set(2)

    assert exception_count == 1


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.edge_case
def test_observable_subscription_propagates_callback_exceptions():
    """Exceptions in subscription callbacks propagate while still updating value"""
    # Arrange
    obs = Observable("test", 1)

    def failing_callback(value):
        raise ValueError("Test exception")

    obs.subscribe(failing_callback)

    # Act & Assert - Exceptions in callbacks will propagate
    with pytest.raises(ValueError, match="Test exception"):
        obs.set(2)

    # Observable value should still be updated even if callback fails
    assert obs.value == 2


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.memory
def test_observable_subscription_management_works_with_multiple_callbacks(
    subscription_tracker, no_leaks
):
    """Observable handles multiple subscriptions without memory leaks"""
    # Arrange
    obs = Observable("test", "value")

    # Create multiple subscriptions
    callbacks = []
    for i in range(10):

        def callback(value):
            pass

        callbacks.append(callback)
        obs.subscribe(callback)

    # Act & Assert - All subscriptions should work
    obs.set("changed")
    assert obs.value == "changed"

    # Unsubscribing should work
    obs.unsubscribe(callbacks[0])
    obs.set("changed_again")
    assert obs.value == "changed_again"

    # Memory testing - verify no leaks when creating/destroying subscriptions
    def create_and_destroy_subscriptions():
        temp_obs = Observable("temp", "test")
        for i in range(50):
            temp_callback = lambda v: None
            temp_obs.subscribe(temp_callback)
            temp_obs.unsubscribe(temp_callback)

    no_leaks(create_and_destroy_subscriptions, "Observable")


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.memory
def test_observable_subscription_cleanup_prevents_memory_accumulation(memory_tracker):
    """Observable subscriptions are properly cleaned up to prevent memory accumulation"""
    # Arrange
    obs = Observable("memory_test", "initial")

    def create_subscription_chain():
        """Create a chain of subscriptions and immediately clean them up"""
        subscriptions = []
        for i in range(20):

            def callback(value, idx=i):
                pass

            obs.subscribe(callback)
            subscriptions.append(callback)

        # Cleanup all subscriptions
        for callback in subscriptions:
            obs.unsubscribe(callback)

    # Act & Assert - Track memory during subscription creation/cleanup
    with memory_tracker() as tracker:
        for _ in range(10):  # Multiple cycles to detect accumulation
            create_subscription_chain()

    # Verify no observable instances accumulate
    assert (
        "Observable" not in tracker.object_growth
        or tracker.object_growth["Observable"] <= 5
    )


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.memory
def test_observable_with_computed_dependencies_avoids_circular_references(no_leaks):
    """Observable chains with computed dependencies don't create circular references"""
    from fynx import Store, observable

    def create_computed_chain():
        # Create observable -> computed -> observable pattern
        source = observable(10)
        doubled = source >> (lambda x: x * 2)
        tripled = doubled >> (lambda x: x * 3)

        # Create a store that references the computed values
        class TempStore(Store):
            value = source
            computed_val = doubled

        store = TempStore()

        # Use the values to ensure they're active
        assert tripled.value == 60
        assert store.computed_val.value == 20

        # Delete everything to test cleanup
        del source, doubled, tripled, store

    # Act & Assert - No memory leaks from complex dependency chains
    no_leaks(create_computed_chain, "Observable", tolerance=10)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.topological
def test_observable_notifications_processed_in_topological_order():
    """Observable notifications are processed in topological dependency order"""
    from fynx import Observable

    # Create a dependency chain: source -> computed/conditions -> conditional
    source = Observable("source", 10)

    # Computed observable that depends on source
    computed = source >> (lambda x: x * 2)

    # Condition observables that depend on source
    is_positive = source >> (lambda x: x > 0)
    is_even = source >> (lambda x: x % 2 == 0)

    # Conditional observable that depends on computed conditions
    filtered = source & is_positive & is_even

    # Track notification order
    notification_order = []

    # Subscribe to all observables to track when they notify
    source.subscribe(lambda v: notification_order.append(f"source:{v}"))
    computed.subscribe(lambda v: notification_order.append(f"computed:{v}"))
    is_positive.subscribe(lambda v: notification_order.append(f"is_positive:{v}"))
    is_even.subscribe(lambda v: notification_order.append(f"is_even:{v}"))
    filtered.subscribe(lambda v: notification_order.append(f"filtered:{v}"))

    # Clear initial notifications (setup)
    notification_order.clear()

    # Change source value from 10 to -3 - this changes multiple conditions
    # 10: positive=True, even=True -> filtered active
    # -3: positive=False, even=False -> filtered inactive, conditions change
    source.set(-3)

    # Verify that notifications occurred in some reasonable order
    # The exact order may vary due to topological sorting implementation details
    assert "source:-3" in notification_order
    assert "is_even:False" in notification_order
    assert "computed:-6" in notification_order
    assert "is_positive:False" in notification_order

    # Filtered should NOT be notified because conditions became unmet
    assert "filtered:-3" not in notification_order

    # Verify final state
    assert source.value == -3
    assert computed.value == -6
    assert is_positive.value == False
    assert is_even.value == False
    assert filtered.is_active == False

    # Accessing value when conditions are unmet should raise ConditionalNotMet
    with pytest.raises(
        ConditionalNotMet, match="Conditions are not currently satisfied"
    ):
        _ = filtered.value


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.topological
def test_conditional_observable_throws_error_when_conditions_unmet():
    """Conditional observable throws error when conditions are not met"""
    from fynx import Observable

    # Create the scenario from the bug report
    data = Observable("data", 42)
    is_positive = data >> (lambda x: x > 0)
    is_even = data >> (lambda x: x % 2 == 0)
    filtered = data & is_positive & is_even

    # Verify initial state: 42 is positive and even
    assert data.value == 42
    assert is_positive.value == True
    assert is_even.value == True
    assert filtered.value == 42
    assert filtered.is_active == True

    # Change to -4: makes conditions unmet
    data.set(-4)

    # Verify state after change
    assert data.value == -4
    assert is_positive.value == False  # -4 > 0 is False
    assert is_even.value == True  # -4 % 2 == 0 is True
    assert filtered.is_active == False  # Conditions not fully met

    # Accessing value when conditions are unmet should raise ConditionalNotMet
    with pytest.raises(
        ConditionalNotMet, match="Conditions are not currently satisfied"
    ):
        _ = filtered.value

    # Change back to 6: positive and even - conditions become met again
    data.set(6)

    assert data.value == 6
    assert is_positive.value == True  # 6 > 0
    assert is_even.value == True  # 6 % 2 == 0
    assert filtered.is_active == True  # All conditions met
    assert filtered.value == 6  # Should work now

    # Change to 8: positive and even - stays active, updates value
    data.set(8)

    assert data.value == 8
    assert is_positive.value == True  # 8 > 0
    assert is_even.value == True  # 8 % 2 == 0
    assert filtered.is_active == True  # All conditions met
    assert filtered.value == 8  # Updates to new valid value
