"""
Comprehensive tests for FynX @reactive decorator functionality.

This test suite validates the @reactive decorator for automatic reactions to observable changes,
ensuring it works exactly as described in the using-reactive.md documentation.
"""

import pytest

from fynx import Store, observable, reactive


class TestBasicReactiveBehavior:
    """Tests for basic @reactive decorator functionality."""

    def test_reactive_decorator_subscribes_automatically(self):
        """@reactive automatically subscribes to observables."""
        count = observable(0)
        notifications = []

        @reactive(count, autorun=False)
        def log_count(value):
            notifications.append(f"count_{value}")

        # Initially no notifications
        assert notifications == []

        # Change triggers reaction
        count.set(5)
        assert notifications == ["count_5"]

        count.set(10)
        assert notifications == ["count_5", "count_10"]

    def test_reactive_functions_dont_fire_immediately(self):
        """Reactive functions don't execute immediately with current values."""
        ready = observable(True)
        notifications = []

        @reactive(ready, autorun=False)
        def on_ready(value):
            notifications.append(f"ready_{value}")

        # Nothing happens initially, even though ready is True
        assert notifications == []

        # Change triggers reaction
        ready.set(False)
        assert notifications == ["ready_False"]

        ready.set(True)
        assert notifications == ["ready_False", "ready_True"]

    def test_reactive_functions_only_fire_on_actual_changes(self):
        """Reactive functions only fire when values actually change."""
        count = observable(0)
        notifications = []

        @reactive(count, autorun=False)
        def log_count(value):
            notifications.append(f"count_{value}")

        # Initial state - no execution
        assert notifications == []

        # Change triggers reaction
        count.set(5)
        assert notifications == ["count_5"]

        # Same value doesn't trigger
        count.set(5)
        assert notifications == ["count_5"]  # No additional notification

        # Different value triggers
        count.set(10)
        assert notifications == ["count_5", "count_10"]

    def test_reactive_functions_execute_synchronously(self):
        """Reactive functions execute synchronously before .set() returns."""
        count = observable(0)
        execution_order = []

        @reactive(count, autorun=False)
        def log_count(value):
            execution_order.append(f"reactive_{value}")

        def setter():
            execution_order.append("set_start")
            count.set(5)
            execution_order.append("set_end")

        setter()

        # Reactive function executed during set() call
        assert execution_order == ["set_start", "reactive_5", "set_end"]


class TestConditionalReactions:
    """Tests for conditional reactions using boolean operators."""

    def test_conditional_reactions_with_and_operator(self):
        """@reactive works with & operator for conditional reactions."""
        is_logged_in = observable(False)
        has_data = observable(False)
        notifications = []

        @reactive(is_logged_in & has_data)
        def sync_data(sync_value):
            notifications.append(f"sync_{sync_value}")

        # Initial state - conditional starts unmet, calls immediately with False
        assert notifications == ["sync_False"]

        # Set logged in but no data - still unmet
        is_logged_in.set(True)
        assert notifications == ["sync_False"]  # No additional emission

        # Set data - now condition becomes met
        has_data.set(True)
        assert notifications == [
            "sync_False",
            "sync_True",
        ]  # Receives True when condition becomes met

        # Remove data - no emission when condition becomes unmet
        has_data.set(False)
        assert notifications == ["sync_False", "sync_True"]  # No additional emission

    def test_conditional_reactions_with_or_operator(self):
        """@reactive works with | operator for OR conditions."""
        is_error = observable(False)
        should_sync = observable(False)
        notifications = []

        @reactive(is_error | should_sync, autorun=False)
        def handle_condition(condition):
            notifications.append(f"handle_{condition}")

        # Initial state - no immediate calls for reactive functions
        assert notifications == []

        # Set should_sync - triggers
        should_sync.set(True)
        assert notifications == ["handle_True"]

        # Set error too - still True, no change
        is_error.set(True)
        assert notifications == ["handle_True"]  # No change

        # Clear both - becomes False
        should_sync.set(False)
        is_error.set(False)
        assert notifications == ["handle_True", "handle_False"]

    def test_conditional_reactions_with_negation(self):
        """@reactive works with ~ operator for negation."""
        is_loading = observable(True)
        notifications = []

        @reactive(~is_loading, autorun=False)
        def process_data(can_process):
            notifications.append(f"process_{can_process}")

        # Initially loading - no processing
        assert notifications == []

        # Still loading - no change
        is_loading.set(True)
        assert notifications == []

        # Stop loading - can process
        is_loading.set(False)
        assert notifications == ["process_True"]

    def test_complex_boolean_expressions(self):
        """@reactive handles complex boolean expressions."""
        logged_in = observable(True)
        verified = observable(True)
        notifications = []

        @reactive(logged_in & verified)
        def enable_premium_features(both_true):
            notifications.append(f"premium_{both_true}")

        # Initial state - conditions already met, calls immediately
        assert notifications == ["premium_True"]

        # Change triggers reaction
        logged_in.set(False)
        assert notifications == ["premium_True", "premium_False"]

        verified.set(False)
        assert notifications == ["premium_True", "premium_False"]  # Still False

        logged_in.set(True)
        assert notifications == ["premium_True", "premium_False"]  # Still one False

        verified.set(True)
        assert notifications == [
            "premium_True",
            "premium_False",
            "premium_True",
        ]  # Now both True


class TestMultipleObservableReactions:
    """Tests for reacting to multiple observables."""

    def test_react_to_combined_observables(self):
        """React to derived observables from multiple sources."""
        first_name = observable("Alice")
        last_name = observable("Smith")
        notifications = []

        # Derive combined observable first
        full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")

        # Then react to the derivation
        @reactive(full_name, autorun=False)
        def update_display(display_name):
            notifications.append(f"display_{display_name}")

        # No immediate execution
        assert notifications == []

        # Changes trigger reactions
        first_name.set("Bob")
        assert notifications == ["display_Bob Smith"]

        last_name.set("Jones")
        assert notifications == ["display_Bob Smith", "display_Bob Jones"]


class TestStoreReactions:
    """Tests for reacting to entire Store changes."""

    def test_react_to_store_changes(self):
        """@reactive works with entire Store instances."""

        class UserStore(Store):
            name = observable("Alice")
            age = observable(30)
            email = observable("alice@example.com")

        notifications = []

        @reactive(UserStore, autorun=False)
        def sync_to_server(store_snapshot):
            notifications.append(f"sync_{store_snapshot.name}_{store_snapshot.email}")

        # Store reactions call immediately with initial snapshot
        assert notifications == []

        # Any change triggers reaction with full snapshot
        UserStore.name = "Bob"
        assert notifications == ["sync_Bob_alice@example.com"]

        UserStore.age = 31  # Still triggers
        assert notifications == [
            "sync_Bob_alice@example.com",
            "sync_Bob_alice@example.com",
        ]

        UserStore.email = "bob@example.com"
        assert notifications == [
            "sync_Bob_alice@example.com",
            "sync_Bob_alice@example.com",
            "sync_Bob_bob@example.com",
        ]

    def test_store_reactions_receive_snapshots(self):
        """Store reactions receive snapshot objects, not observables."""

        class TestStore(Store):
            value = observable(42)

        snapshots = []

        @reactive(TestStore, autorun=False)
        def capture_snapshot(store):
            snapshots.append(store)

        TestStore.value = 100

        # Should have received a snapshot after change
        assert len(snapshots) == 1
        snapshot = snapshots[0]

        # Snapshot has current values
        assert snapshot.value == 100

        # But snapshot properties are not observables
        assert not hasattr(snapshot.value, "set")


class TestComputedObservableReactions:
    """Tests for reacting to computed observables."""

    def test_react_to_computed_observables(self):
        """@reactive works with computed observables."""
        items = observable([])
        notifications = []

        # Computed observable
        item_count = items >> (lambda i: len(i))

        @reactive(item_count)
        def update_badge(count):
            notifications.append(f"badge_{count}")

        # Computed observables call immediately with current value (0)
        assert notifications == ["badge_0"]

        # Add items - count changes
        items.set([{"name": "Widget", "price": 10}])
        assert notifications == ["badge_0", "badge_1"]

        items.set([{"name": "Widget", "price": 10}, {"name": "Gadget", "price": 15}])
        assert notifications == ["badge_0", "badge_1", "badge_2"]

    def test_computed_reactions_only_fire_on_actual_changes(self):
        """Computed reactions only fire when computed values actually change."""
        items = observable([1, 2, 3])
        notifications = []

        length = items >> (lambda i: len(i))

        @reactive(length)
        def log_length(l):
            notifications.append(f"length_{l}")

        # Computed observables call immediately with current value (3)
        assert notifications == ["length_3"]

        # Same length - no reaction
        items.set([4, 5, 6])
        assert notifications == ["length_3"]  # Length still 3

        # Different length - reaction fires
        items.set([7, 8, 9, 10])
        assert notifications == ["length_3", "length_4"]


class TestUnsubscribeMechanism:
    """Tests for unsubscribing reactive functions."""

    def test_unsubscribe_stops_reactive_behavior(self):
        """unsubscribe() stops reactive behavior and allows manual calls."""
        count = observable(0)
        notifications = []

        @reactive(count, autorun=False)
        def log_count(value):
            notifications.append(f"count_{value}")

        # Initial state
        assert notifications == []

        # Trigger reaction
        count.set(5)
        assert notifications == ["count_5"]

        # Unsubscribe
        log_count.unsubscribe()

        # No more reactive behavior
        count.set(10)
        assert notifications == ["count_5"]

        # Can now call manually
        log_count(15)
        assert notifications == ["count_5", "count_15"]


class TestManualCallPrevention:
    """Tests that reactive functions can't be called manually."""

    def test_reactive_functions_prevent_manual_calls(self):
        """Reactive functions raise exception when called manually."""
        count = observable(0)

        @reactive(count, autorun=False)
        def reactive_func(value):
            pass

        # Can't call manually while reactive
        with pytest.raises(Exception):  # Should be ReactiveFunctionWasCalled
            reactive_func(42)

    def test_manual_calls_allowed_after_unsubscribe(self):
        """Manual calls work after unsubscribing."""
        count = observable(0)
        call_log = []

        @reactive(count, autorun=False)
        def reactive_func(value):
            call_log.append(f"reactive_{value}")

        # Can't call manually initially
        with pytest.raises(Exception):
            reactive_func(42)

        # Unsubscribe
        reactive_func.unsubscribe()

        # Now can call manually
        reactive_func(100)
        assert call_log == ["reactive_100"]


class TestRealWorldExamples:
    """Tests for real-world examples from documentation."""

    def test_form_validation_example(self):
        """Complete form validation example with reactive functions."""

        class FormStore(Store):
            email = observable("")
            password = observable("")
            confirm_password = observable("")

        # Computed validations
        email_valid = FormStore.email >> (
            lambda e: "@" in e and "." in e.split("@")[-1]
        )

        password_valid = FormStore.password >> (lambda p: len(p) >= 8)

        passwords_match = (FormStore.password + FormStore.confirm_password) >> (
            lambda pwd, confirm: pwd == confirm and pwd != ""
        )

        form_valid = email_valid & password_valid & passwords_match

        # Reactive UI updates
        email_notifications = []
        password_notifications = []
        match_notifications = []
        form_notifications = []

        @reactive(email_valid)
        def update_email_indicator(is_valid):
            email_notifications.append(f"email_{'✓' if is_valid else '✗'}")

        @reactive(password_valid)
        def update_password_indicator(is_valid):
            password_notifications.append(f"password_{'✓' if is_valid else '✗'}")

        @reactive(passwords_match)
        def update_match_indicator(match):
            match_notifications.append(f"match_{'✓' if match else '✗'}")

        @reactive(form_valid)
        def update_submit_button(is_valid):
            form_notifications.append(f"submit_{'enabled' if is_valid else 'disabled'}")

        # Initial state - computed observables call immediately with current values
        assert email_notifications == ["email_✗"]  # empty email is invalid
        assert password_notifications == ["password_✗"]  # empty password is invalid
        assert match_notifications == ["match_✗"]  # empty passwords don't match
        assert form_notifications == ["submit_disabled"]  # form is invalid

        # Set valid email
        FormStore.email = "alice@example.com"
        assert email_notifications == ["email_✗", "email_✓"]

        # Set password (too short)
        FormStore.password = "pass"
        assert password_notifications == ["password_✗"]  # Still invalid, no change
        assert match_notifications == ["match_✗"]  # No change

        # Set strong password
        FormStore.password = "secure123"
        assert password_notifications == ["password_✗", "password_✓"]
        assert match_notifications == ["match_✗"]  # Still don't match, no change

        # Set matching confirm password
        FormStore.confirm_password = "secure123"
        assert match_notifications == ["match_✗", "match_✓"]
        assert form_notifications == [
            "submit_disabled",
            "submit_enabled",
        ]  # Form becomes valid

    def test_cart_total_reactions(self):
        """Cart total calculation with reactive UI updates."""

        class CartStore(Store):
            items = observable([])

        # Computed total
        total = CartStore.items >> (
            lambda items: sum(item["price"] * item["qty"] for item in items)
        )

        notifications = []

        @reactive(total)
        def update_total_display(total_amount):
            notifications.append(f"total_${total_amount:.2f}")

        # Computed observables call immediately with current value (0)
        assert notifications == ["total_$0.00"]

        # Add items
        CartStore.items = [{"name": "Widget", "price": 10, "qty": 2}]
        assert notifications == ["total_$0.00", "total_$20.00"]

        CartStore.items = [
            {"name": "Widget", "price": 10, "qty": 2},
            {"name": "Gadget", "price": 15, "qty": 1},
        ]
        assert notifications == ["total_$0.00", "total_$20.00", "total_$35.00"]


class TestAntiPatternsAndEdgeCases:
    """Tests for anti-patterns and edge cases."""

    def test_infinite_loops_prevented_by_design(self):
        """Reactive functions shouldn't modify what they watch."""
        count = observable(0)
        notifications = []

        @reactive(count, autorun=False)
        def increment_forever(value):
            notifications.append(f"increment_{value}")
            # This would cause infinite loop if allowed
            # count.set(value + 1)

        count.set(5)
        assert notifications == ["increment_5"]

        # Manually verify no infinite loop occurred
        assert len(notifications) == 1

    def test_reactive_functions_dont_track_hidden_dependencies(self):
        """Reactive functions don't automatically track .get() or .value reads."""
        count = observable(0)
        other_count = observable(10)
        notifications = []

        @reactive(count, autorun=False)
        def show_sum(value):
            total = value + other_count.value  # Hidden dependency
            notifications.append(f"sum_{total}")

        count.set(5)
        assert notifications == ["sum_15"]

        # Changing hidden dependency doesn't trigger reaction
        other_count.set(20)
        assert notifications == ["sum_15"]  # No new notification

    def test_reactive_functions_receive_values_not_observables(self):
        """Reactive functions receive primitive values, not observables."""
        count = observable(42)
        received_types = []

        @reactive(count, autorun=False)
        def check_type(value):
            received_types.append(type(value).__name__)

        count.set(100)
        assert received_types == ["int"]

    def test_conditional_guards_with_expensive_operations(self):
        """Conditional guards prevent expensive operations."""
        user_active = observable(False)
        is_loading = observable(True)
        notifications = []

        @reactive(user_active & ~is_loading)
        def auto_save(should_save):
            if should_save:
                notifications.append("saved")

        # Initial state - no save
        assert notifications == []

        # Make user active but still loading
        user_active.set(True)
        assert notifications == []

        # Stop loading - should save
        is_loading.set(False)
        assert notifications == ["saved"]

        # Make inactive - should stop saving (no emission when condition becomes unmet)
        user_active.set(False)
        assert notifications == ["saved"]

    def test_unsubscribe_in_component_cleanup(self):
        """Unsubscribe works for component lifecycle management."""
        data = observable("initial")
        notifications = []

        @reactive(data, autorun=False)
        def process_data(value):
            notifications.append(f"processed_{value}")

        # Trigger reaction
        data.set("updated")
        assert notifications == ["processed_updated"]

        # Unsubscribe (like component cleanup)
        process_data.unsubscribe()

        # No more reactions
        data.set("final")
        assert notifications == ["processed_updated"]

        # Can call manually after unsubscribe
        process_data("manual")
        assert notifications == ["processed_updated", "processed_manual"]


class TestPerformanceConsiderations:
    """Tests for performance aspects of reactive functions."""

    def test_react_to_filtered_derived_state(self):
        """React to derived state that filters unnecessary changes."""
        search_query = observable("")
        notifications = []

        # Only process queries with meaningful length
        filtered_results = search_query >> (
            lambda q: f"results_for_{q}" if len(q) >= 3 else None
        )

        @reactive(filtered_results)
        def update_ui(results):
            if results is not None:
                notifications.append(f"ui_{results}")

        # Short queries don't trigger
        search_query.set("a")
        search_query.set("ab")
        assert notifications == []

        # Long enough query triggers
        search_query.set("abc")
        assert notifications == ["ui_results_for_abc"]

    def test_conditional_observables_limit_reaction_frequency(self):
        """Conditional observables prevent reactions when conditions aren't met."""
        data = observable("initial")
        should_process = observable(False)
        notifications = []

        @reactive(data & should_process)
        def process_data(should):
            if should:
                notifications.append(f"processed_{data.value}")

        # Data changes but shouldn't process
        data.set("updated")
        assert notifications == []

        # Enable processing
        should_process.set(True)
        assert notifications == ["processed_updated"]

        # More data changes trigger since value is different
        data.set("final")
        assert notifications == ["processed_updated", "processed_final"]


class TestStorePatterns:
    """Tests for Store-specific reactive patterns."""

    def test_functional_core_reactive_shell_pattern(self):
        """Demonstrate functional core, reactive shell separation."""

        class OrderStore(Store):
            items = observable([])
            shipping_address = observable(None)
            payment_method = observable(None)
            is_processing = observable(False)

            # Functional core - pure derivations
            subtotal = items >> (lambda i: sum(x["price"] * x["qty"] for x in i))
            has_items = items >> (lambda i: len(i) > 0)
            has_address = shipping_address >> (lambda a: a is not None)
            has_payment = payment_method >> (lambda p: p is not None)
            can_checkout = has_address & has_payment
            tax = subtotal >> (lambda s: s * 0.08)
            total = (subtotal + tax) >> (lambda s, t: s + t)

        checkout_notifications = []
        total_notifications = []

        # Reactive shell - side effects
        @reactive(OrderStore.can_checkout)
        def update_checkout_button(can_checkout):
            checkout_notifications.append(
                f"checkout_{'enabled' if can_checkout else 'disabled'}"
            )

        @reactive(OrderStore.total)
        def update_display(total_amount):
            total_notifications.append(f"display_${total_amount:.2f}")

        # Initial state - computed observables call immediately with current values
        assert checkout_notifications == [
            "checkout_disabled"
        ]  # can_checkout has never been active
        assert total_notifications == ["display_$0.00"]  # total is 0

        # Add items - total updates
        OrderStore.items = [{"price": 10, "qty": 2}]
        assert total_notifications == [
            "display_$0.00",
            "display_$21.60",
        ]  # 20 + 1.6 tax

        # Add address and payment - checkout enables
        OrderStore.shipping_address = "123 Main St"
        OrderStore.payment_method = "credit_card"
        assert checkout_notifications == ["checkout_disabled", "checkout_enabled"]
