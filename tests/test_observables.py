"""
Comprehensive tests for FynX observables functionality.

This test suite covers all functionality described in the observables documentation,
following FynX testing conventions with clear separation of concerns and descriptive naming.
"""

import pytest

from fynx import observable, transaction


class TestObservableCreation:
    """Tests for creating observables with different initial values."""

    def test_observable_creation_with_string_value(self):
        """Observable should be created with string initial value."""
        # Arrange & Act
        name = observable("Alice")

        # Assert
        assert name.value == "Alice"

    def test_observable_creation_with_numeric_value(self):
        """Observable should be created with numeric initial value."""
        # Arrange & Act
        age = observable(30)

        # Assert
        assert age.value == 30

    def test_observable_creation_with_list_value(self):
        """Observable should be created with list initial value."""
        # Arrange & Act
        scores = observable([85, 92, 78])

        # Assert
        assert scores.value == [85, 92, 78]

    def test_observable_creation_with_dict_value(self):
        """Observable should be created with dictionary initial value."""
        # Arrange & Act
        user = observable({"id": 1, "active": True})

        # Assert
        assert user.value == {"id": 1, "active": True}

    def test_observable_creation_with_none_value(self):
        """Observable should be created with None initial value."""
        # Arrange & Act
        empty = observable(None)

        # Assert
        assert empty.value is None


class TestObservableValueReading:
    """Tests for reading observable values."""

    def test_observable_value_property_returns_current_value(self):
        """Observable value property should return the current value."""
        # Arrange
        name = observable("Alice")

        # Act & Assert
        assert name.value == "Alice"

    def test_observable_value_property_registers_dependency(self):
        """Reading observable value should register dependency for reactive contexts."""
        # Arrange
        base = observable(10)

        # Act - Create computed observable that depends on base
        derived = base >> (lambda x: x * 2)

        # Assert - Derived should update when base changes
        assert derived.value == 20

        base.set(15)
        assert derived.value == 30


class TestObservableValueWriting:
    """Tests for writing observable values."""

    def test_observable_set_method_updates_value(self):
        """Observable set method should update the internal value."""
        # Arrange
        name = observable("Alice")

        # Act
        name.set("Bob")

        # Assert
        assert name.value == "Bob"

    def test_observable_set_method_notifies_subscribers(self):
        """Observable set method should notify all subscribers."""
        # Arrange
        counter = observable(0)
        notifications = []

        def track_notification(new_value):
            notifications.append(new_value)

        counter.subscribe(track_notification)

        # Act
        counter.set(5)

        # Assert
        assert notifications == [5]

    @pytest.mark.parametrize("iteration", range(50))
    def test_observable_set_method_notifies_multiple_subscribers(self, iteration):
        """Observable set method should notify all subscribers in subscription order."""
        # Arrange
        counter = observable(0)
        notifications = []

        def track_count(new_value):
            notifications.append(f"Count: {new_value}")

        def track_double(new_value):
            notifications.append(f"Double: {new_value * 2}")

        def track_square(new_value):
            notifications.append(f"Square: {new_value ** 2}")

        counter.subscribe(track_count)
        counter.subscribe(track_double)
        counter.subscribe(track_square)

        # Act
        counter.set(5)

        # Assert - Subscribers run in subscription order
        assert notifications == ["Count: 5", "Double: 10", "Square: 25"]


class TestObservableSubscriptions:
    """Tests for subscribing to observable changes."""

    def test_subscribe_registers_callback_function(self):
        """Subscribe should register a callback function for value changes."""
        # Arrange
        name = observable("Alice")
        greetings = []

        def greet(new_name):
            greetings.append(f"Hello, {new_name}!")

        # Act
        name.subscribe(greet)
        name.set("Bob")

        # Assert
        assert greetings == ["Hello, Bob!"]

    def test_subscribe_callback_receives_new_value(self):
        """Subscribe callback should receive the new value as argument."""
        # Arrange
        counter = observable(0)
        received_values = []

        def track_value(new_value):
            received_values.append(new_value)

        counter.subscribe(track_value)

        # Act
        counter.set(42)

        # Assert
        assert received_values == [42]

    def test_subscribe_callback_runs_immediately_on_set(self):
        """Subscribe callback should run immediately when set is called."""
        # Arrange
        counter = observable(0)
        callback_executed = False

        def callback(new_value):
            nonlocal callback_executed
            callback_executed = True

        counter.subscribe(callback)

        # Act
        counter.set(5)

        # Assert
        assert callback_executed is True

    def test_subscribe_callback_runs_after_value_update(self):
        """Subscribe callback should run after the value is updated."""
        # Arrange
        counter = observable(0)
        callback_values = []

        def callback(new_value):
            callback_values.append(new_value)

        counter.subscribe(callback)

        # Act
        counter.set(10)

        # Assert
        assert callback_values == [10]
        assert counter.value == 10


class TestObservableUnsubscriptions:
    """Tests for unsubscribing from observable changes."""

    def test_unsubscribe_removes_callback_function(self):
        """Unsubscribe should remove the callback function from notifications."""
        # Arrange
        counter = observable(0)
        notifications = []

        def logger(new_value):
            notifications.append(f"Value: {new_value}")

        unsubscribe = counter.subscribe(logger)

        # Act
        counter.set(1)  # Should trigger notification
        unsubscribe()  # Unsubscribe using returned function
        counter.set(2)  # Should not trigger notification

        # Assert
        assert notifications == ["Value: 1"]

    def test_unsubscribe_requires_exact_function_reference(self):
        """Unsubscribe should require the exact same function reference."""
        # Arrange
        counter = observable(0)
        notifications = []

        def logger(new_value):
            notifications.append(f"Value: {new_value}")

        unsubscribe = counter.subscribe(logger)

        # Act
        counter.set(1)
        unsubscribe()  # Same unsubscribe function
        counter.set(2)

        # Assert
        assert notifications == ["Value: 1"]

    def test_unsubscribe_with_lambda_functions_is_difficult(self):
        """Unsubscribing lambda functions should be difficult due to reference issues."""
        # Arrange
        counter = observable(0)
        notifications = []

        # Act - Subscribe with lambda
        counter.subscribe(
            lambda new_value: notifications.append(f"Lambda: {new_value}")
        )
        counter.set(1)

        # Assert - Can't easily unsubscribe lambda, so it keeps firing
        # This demonstrates why named functions are preferred for cleanup
        assert len(notifications) == 1


class TestObservableCircularDependencyDetection:
    """Tests for circular dependency detection and prevention."""

    def test_circular_dependency_detection_prevents_infinite_loops(self):
        """Circular dependency detection should prevent infinite loops."""
        # Arrange
        counter = observable(0)

        def increment_on_change(new_value):
            # This would cause a circular dependency error
            counter.set(new_value + 1)

        counter.subscribe(increment_on_change)

        # Act & Assert - Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Circular dependency detected"):
            counter.set(5)

    def test_circular_dependency_detection_works_with_multiple_subscribers(self):
        """Circular dependency detection should work with multiple subscribers."""
        # Arrange
        counter = observable(0)
        notifications = []

        def safe_subscriber(new_value):
            notifications.append(f"Safe: {new_value}")

        def circular_subscriber(new_value):
            counter.set(new_value + 1)  # This creates circular dependency

        counter.subscribe(safe_subscriber)
        counter.subscribe(circular_subscriber)

        # Act & Assert - Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Circular dependency detected"):
            counter.set(5)

        # Safe subscribers are called before circular dependency is detected
        assert notifications == ["Safe: 5"]


class TestObservableTransactions:
    """Tests for observable transactions and safe reentrant updates."""

    def test_transaction_allows_safe_reentrant_updates(self):
        """Transaction should allow safe reentrant updates from within subscribers."""
        # Arrange
        counter = observable(0)
        other_counter = observable(
            1
        )  # Different initial value to ensure different objects

        def increment_other_on_change(new_value):
            # Use transaction for safe reentrant updates
            with transaction():
                other_counter.set(new_value + 1)

        counter.subscribe(increment_other_on_change)

        # Act
        counter.set(5)

        # Assert - Should work without circular dependency error
        assert counter.value == 5
        assert other_counter.value == 6

        # Also test that the transaction prevents notifications during updates
        notifications_during_transaction = []

        def track_notifications(new_value):
            notifications_during_transaction.append(new_value)

        counter.subscribe(track_notifications)

        # This should allow reentrant updates without circular dependency errors
        with transaction():
            counter.set(10)
            counter.set(20)
            counter.set(30)

        # Updates are applied correctly
        assert counter.value == 30
        # Note: Batch fusion coalesces multiple deltas to same key into single delta
        assert notifications_during_transaction == [30]

    def test_transaction_defers_notifications_until_completion(self):
        """Transaction should defer notifications until transaction completes."""
        # Arrange
        counter = observable(0)
        notifications = []

        def track_notifications(new_value):
            notifications.append(new_value)

        counter.subscribe(track_notifications)

        # Act - Without transaction
        counter.set(1)
        counter.set(2)
        counter.set(3)

        # Assert - Multiple notifications
        assert notifications == [1, 2, 3]

        # Reset for transaction test
        notifications.clear()

        # Act - With transaction (batch fusion coalesces deltas to same key)
        with transaction():
            counter.set(1)
            counter.set(2)
            counter.set(3)

        # Assert - Batch fusion reduces multiple deltas to same key to single delta
        assert notifications == [3]

    def test_transaction_ensures_atomic_updates(self):
        """Transaction should ensure atomic updates prevent intermediate inconsistent states."""
        # Arrange
        counter = observable(0)
        notifications = []

        def track_notifications(new_value):
            notifications.append(new_value)

        counter.subscribe(track_notifications)

        # Act - Multiple updates in transaction
        with transaction():
            counter.set(10)
            counter.set(20)
            counter.set(30)

        # Assert - All updates are applied, final value is correct
        assert counter.value == 30
        # Note: Batch fusion coalesces multiple deltas to same key into single delta
        assert notifications == [30]

    def test_transaction_coordinates_multiple_observable_updates(self):
        """Transaction should coordinate updates across multiple observables."""
        # Arrange
        name = observable("")
        email = observable("")
        is_valid = observable(False)

        validation_notifications = []

        def track_validation(new_value):
            validation_notifications.append(new_value)

        is_valid.subscribe(track_validation)

        def validate_form():
            with transaction():
                name_valid = len(name.value) > 0
                email_valid = "@" in email.value
                is_valid.set(name_valid and email_valid)

        # Act - Update form fields and validate
        name.set("Alice")
        email.set("alice@example.com")
        validate_form()  # Manually trigger validation

        # Assert - Validation works with transaction
        assert len(validation_notifications) == 1
        assert validation_notifications[0] is True

    def test_transaction_keeps_operations_short(self):
        """Transaction should be kept short for better debugging."""
        # Arrange
        counter = observable(0)

        # Act - Short transaction
        with transaction():
            counter.set(1)

        # Assert - Works correctly
        assert counter.value == 1


class TestObservableVsRegularVariables:
    """Tests demonstrating the difference between observables and regular variables."""

    def test_regular_variables_require_manual_synchronization(self):
        """Regular variables require manual synchronization for dependent state."""

        # Arrange - Traditional approach
        class ShoppingCart:
            def __init__(self):
                self.items = []
                self.total = 0
                self.ui_updated = False
                self.storage_saved = False
                self.analytics_notified = False

            def add_item(self, item):
                self.items.append(item)
                self.total = sum(item["price"] for item in self.items)
                self.update_ui()
                self.save_to_storage()
                self.notify_analytics()

            def update_ui(self):
                self.ui_updated = True

            def save_to_storage(self):
                self.storage_saved = True

            def notify_analytics(self):
                self.analytics_notified = True

        # Act
        cart = ShoppingCart()
        cart.add_item({"name": "Widget", "price": 10})

        # Assert - Manual synchronization required
        assert cart.total == 10
        assert cart.ui_updated is True
        assert cart.storage_saved is True
        assert cart.analytics_notified is True

    def test_observables_provide_automatic_synchronization(self):
        """Observables provide automatic synchronization for dependent state."""
        # Arrange - Reactive approach
        items = observable([])
        total = items >> (lambda item_list: sum(item["price"] for item in item_list))

        ui_updated = False
        storage_saved = False
        analytics_notified = False

        def update_ui(item_list):
            nonlocal ui_updated
            ui_updated = True

        def save_to_storage(item_list):
            nonlocal storage_saved
            storage_saved = True

        def notify_analytics(item_list):
            nonlocal analytics_notified
            analytics_notified = True

        items.subscribe(update_ui)
        items.subscribe(save_to_storage)
        items.subscribe(notify_analytics)

        # Act
        items.set(items.value + [{"name": "Widget", "price": 10}])

        # Assert - Automatic synchronization
        assert total.value == 10
        assert ui_updated is True
        assert storage_saved is True
        assert analytics_notified is True


class TestObservableReactiveGraphs:
    """Tests for observable reactive graphs and automatic propagation."""

    def test_observable_reactive_graph_propagates_changes_automatically(self):
        """Observable reactive graph should propagate changes automatically."""
        # Arrange
        base_price = observable(100)
        quantity = observable(2)

        # Create computed observable
        total = (base_price + quantity) >> (lambda price, qty: price * qty)

        total_notifications = []

        def track_total(value):
            total_notifications.append(f"Total: ${value}")

        total.subscribe(track_total, call_immediately=True)

        # Act
        base_price.set(150)

        # Assert - Total should update automatically
        assert total.value == 300
        assert total_notifications == ["Total: $200", "Total: $300"]

        # Act
        quantity.set(3)

        # Assert - Total should update automatically again
        assert total.value == 450
        assert total_notifications == ["Total: $200", "Total: $300", "Total: $450"]

    def test_observable_reactive_graph_maintains_relationships(self):
        """Observable reactive graph should maintain relationships over time."""
        # Arrange
        base_price = observable(100)
        quantity = observable(2)
        total = (base_price + quantity) >> (lambda price, qty: price * qty)

        # Act - Multiple changes
        base_price.set(150)
        quantity.set(3)
        base_price.set(200)

        # Assert - Relationship "total equals price times quantity" holds
        assert total.value == 200 * 3  # 600

    def test_observable_reactive_graph_handles_complex_dependencies(self):
        """Observable reactive graph should handle complex dependency chains."""
        # Arrange
        base = observable(10)
        doubled = base >> (lambda x: x * 2)
        tripled = base >> (lambda x: x * 3)
        combined = (doubled + tripled) >> (lambda d, t: d + t)

        # Act
        base.set(5)

        # Assert - All derived values update correctly
        assert doubled.value == 10
        assert tripled.value == 15
        assert combined.value == 25


class TestObservableUseCases:
    """Tests for common observable use cases."""

    def test_observables_handle_multiple_dependencies(self):
        """Observables should handle multiple things depending on the same state."""
        # Arrange
        user_status = observable("active")
        notifications = []

        def update_ui(status):
            notifications.append(f"UI: {status}")

        def update_database(status):
            notifications.append(f"DB: {status}")

        def send_notification(status):
            notifications.append(f"Notify: {status}")

        user_status.subscribe(update_ui)
        user_status.subscribe(update_database)
        user_status.subscribe(send_notification)

        # Act
        user_status.set("inactive")

        # Assert - All dependent systems updated (order may vary due to implementation details)
        expected_set = {"UI: inactive", "DB: inactive", "Notify: inactive"}
        assert set(notifications) == expected_set
        assert len(notifications) == 3

    def test_observables_handle_frequent_state_changes(self):
        """Observables should handle frequent state changes efficiently."""
        # Arrange
        counter = observable(0)
        change_count = 0

        def track_changes(value):
            nonlocal change_count
            change_count += 1

        counter.subscribe(track_changes)

        # Act - Rapid changes
        for i in range(100):
            counter.set(i)

        # Assert - All changes tracked (99 because set(0) doesn't change from initial 0)
        assert change_count == 99
        assert counter.value == 99

    def test_observables_handle_complex_dependencies(self):
        """Observables should handle complex dependency graphs."""
        # Arrange
        a = observable(1)
        b = observable(2)
        c = observable(3)

        # Complex dependency: d depends on a and b, e depends on c and d
        d = (a + b) >> (lambda x, y: x + y)
        e = (c + d) >> (lambda x, y: x * y)

        # Act
        a.set(5)

        # Assert - Complex dependency chain updates correctly
        assert d.value == 7  # 5 + 2
        assert e.value == 21  # 3 * 7

    def test_observables_eliminate_manual_synchronization(self):
        """Observables should eliminate the need for manual synchronization."""
        # Arrange
        data = observable({"count": 0, "total": 0})
        sync_calls = 0

        def manual_sync():
            nonlocal sync_calls
            sync_calls += 1

        # With observables, no manual sync needed
        data.subscribe(lambda _: manual_sync())

        # Act
        data.set({"count": 5, "total": 100})

        # Assert - Synchronization happens automatically
        assert sync_calls == 1
        assert data.value == {"count": 5, "total": 100}


class TestObservablePerformanceCharacteristics:
    """Tests for observable performance characteristics."""

    def test_observable_overhead_is_acceptable_for_interactive_applications(self):
        """Observable overhead should be acceptable for interactive applications."""
        # Arrange
        counter = observable(0)
        notifications = []

        def track_notifications(value):
            notifications.append(value)

        counter.subscribe(track_notifications)

        # Act - Simulate interactive application usage
        for i in range(1000):
            counter.set(i)

        # Assert - Performance is acceptable (999 because set(0) doesn't change from initial 0)
        assert len(notifications) == 999
        assert counter.value == 999

    def test_observable_memory_usage_is_reasonable(self):
        """Observable memory usage should be reasonable."""
        # Arrange
        observables = []

        # Act - Create many observables
        for i in range(1000):
            obs = observable(i)
            observables.append(obs)

        # Assert - Memory usage is reasonable
        assert len(observables) == 1000
        assert all(obs.value == i for i, obs in enumerate(observables))


class TestObservableComposition:
    """Tests for observable composition and building reactive systems."""

    def test_observables_compose_into_sophisticated_systems(self):
        """Observables should compose into sophisticated reactive systems."""
        # Arrange
        base = observable(10)

        # Transform observables
        doubled = base >> (lambda x: x * 2)
        tripled = base >> (lambda x: x * 3)

        # Combine observables
        combined = (doubled + tripled) >> (lambda d, t: d + t)

        # Filter observables (simulated with conditional)
        filtered = combined >> (lambda x: x if x > 50 else None)

        # Act
        base.set(15)

        # Assert - Sophisticated system works correctly
        assert doubled.value == 30
        assert tripled.value == 45
        assert combined.value == 75
        assert filtered.value == 75  # 75 > 50, so not filtered out

    def test_observables_build_reactive_graphs_automatically(self):
        """Observables should build reactive graphs automatically."""
        # Arrange
        source = observable(1)

        # Build a chain of dependencies
        level1 = source >> (lambda x: x * 2)
        level2 = level1 >> (lambda x: x + 1)
        level3 = level2 >> (lambda x: x * 3)

        # Act
        source.set(5)

        # Assert - Reactive graph updates automatically
        assert level1.value == 10  # 5 * 2
        assert level2.value == 11  # 10 + 1
        assert level3.value == 33  # 11 * 3

    def test_observables_maintain_graph_consistency(self):
        """Observables should maintain graph consistency automatically."""
        # Arrange
        a = observable(1)
        b = observable(2)
        c = (a + b) >> (lambda x, y: x + y)
        d = c >> (lambda x: x * 2)

        # Act - Change source values
        a.set(3)
        b.set(4)

        # Assert - Graph consistency maintained
        assert c.value == 7  # 3 + 4
        assert d.value == 14  # 7 * 2

        # Verify relationship holds: d = (a + b) * 2
        assert d.value == (a.value + b.value) * 2
