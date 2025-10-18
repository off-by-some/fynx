#!/usr/bin/env python3
"""Test script for the @reactive documentation examples."""

import sys

sys.path.insert(0, "/home/fox/Workspace/fynx")

from fynx import Store, observable, reactive


def test_basic_reactive():
    """Test the basic @reactive example."""
    print("Testing basic @reactive example...")

    count = observable(0)
    results = []

    @reactive(count)
    def log_count(value):
        results.append(f"Count: {value}")

    # Should have run once already with initial value
    assert len(results) == 1 and results[0] == "Count: 0"

    count.set(5)  # Should trigger again
    assert len(results) == 2 and results[1] == "Count: 5"

    count.set(10)  # Should trigger again
    assert len(results) == 3 and results[2] == "Count: 10"

    print("✓ Basic @reactive example works!")


def test_scattered_subscriptions():
    """Test the scattered subscriptions example."""
    print("Testing scattered subscriptions example...")

    count = observable(0)
    name = observable("Alice")
    email = observable("alice@example.com")

    results = []

    # Mock functions that would normally update UI/database/etc
    def update_ui(value):
        results.append(f"UI: {value}")

    def save_to_database(value):
        results.append(f"DB: {value}")

    def notify_analytics(value):
        results.append(f"Analytics: {value}")

    def update_greeting(value):
        results.append(f"Greeting: {value}")

    def validate_email(value):
        results.append(f"Email valid: {value}")

    def update_display_name(first, last):
        results.append(f"Display: {first} {last}")

    # Subscribe manually (these do NOT run immediately with initial values)
    count.subscribe(update_ui)
    count.subscribe(save_to_database)
    count.subscribe(notify_analytics)
    name.subscribe(update_greeting)
    email.subscribe(validate_email)

    # This would be first | last, but let's simplify
    first_name = observable("John")
    last_name = observable("Doe")
    (first_name | last_name).subscribe(lambda f, l: update_display_name(f, l))

    # Manual subscriptions do NOT run immediately - only on changes
    # So initially, results should be empty
    assert results == []

    # Change count
    count.set(42)
    # Should have 3 more results for count changes
    assert results.count("UI: 42") == 1
    assert results.count("DB: 42") == 1
    assert results.count("Analytics: 42") == 1

    print("✓ Scattered subscriptions example works!")


def test_reactive_decorator():
    """Test the @reactive decorator example."""
    print("Testing @reactive decorator example...")

    count = observable(0)
    results = []

    @reactive(count)
    def log_count(value):
        results.append(f"Count: {value}")

    # @reactive runs immediately with initial value
    assert results == ["Count: 0"]

    count.set(5)
    assert results == ["Count: 0", "Count: 5"]

    count.set(10)
    assert results == ["Count: 0", "Count: 5", "Count: 10"]

    print("✓ @reactive decorator example works!")


def test_execution_model():
    """Test the execution model example."""
    print("Testing execution model example...")

    count = observable(0)
    results = []

    @reactive(count)
    def log_count(value):
        results.append(f"Count: {value}")

    # Initial execution with current value
    assert results == ["Count: 0"]

    count.set(5)
    assert results == ["Count: 0", "Count: 5"]

    # Same value does NOT trigger
    count.set(5)
    assert results == ["Count: 0", "Count: 5"]

    print("✓ Execution model example works!")


def test_declarative_side_effects():
    """Test the declarative side effects example."""
    print("Testing declarative side effects example...")

    count = observable(0)
    results = []

    @reactive(count)
    def update_ui(value):
        results.append(f"UI: {value}")

    @reactive(count)
    def save_to_database(value):
        results.append(f"Saving: {value}")

    @reactive(count)
    def log_change(value):
        results.append(f"Log: {value}")

    # Initial executions with current value (0)
    expected_initial = ["UI: 0", "Saving: 0", "Log: 0"]
    assert set(results) == set(expected_initial)

    count.set(42)
    expected_after = expected_initial + ["UI: 42", "Saving: 42", "Log: 42"]
    assert set(results) == set(expected_after)

    print("✓ Declarative side effects example works!")


def test_multiple_observables():
    """Test the multiple observables example."""
    print("Testing multiple observables example...")

    first_name = observable("Alice")
    last_name = observable("Smith")
    results = []

    @reactive(first_name, last_name)
    def greet(first, last):
        results.append(f"Hello, {first} {last}!")

    # Initial execution with current values
    assert results == ["Hello, Alice Smith!"]

    first_name.set("Bob")
    assert results == ["Hello, Alice Smith!", "Hello, Bob Smith!"]

    last_name.set("Jones")
    assert results == ["Hello, Alice Smith!", "Hello, Bob Smith!", "Hello, Bob Jones!"]

    print("✓ Multiple observables example works!")


def test_cart_store_example():
    """Test the CartStore example."""
    print("Testing CartStore example...")

    class CartStore(Store):
        items = observable([])
        tax_rate = observable(0.08)

    results = []

    @reactive(CartStore.items, CartStore.tax_rate)
    def update_total_display(items, rate):
        subtotal = sum(item["price"] * item["quantity"] for item in items)
        tax = subtotal * rate
        total = subtotal + tax
        results.append(f"Total: ${total:.2f}")

    # Initial execution with empty cart
    assert results == ["Total: $0.00"]

    # Add items
    CartStore.items = [{"name": "Widget", "price": 10, "quantity": 2}]
    assert results == ["Total: $0.00", "Total: $21.60"]  # 20 + (20 * 0.08)

    print("✓ CartStore example works!")


def test_store_level_reaction():
    """Test the store-level reaction example."""
    print("Testing store-level reaction example...")

    class UserStore(Store):
        name = observable("Alice")
        age = observable(30)
        email = observable("alice@example.com")

    results = []

    @reactive(UserStore)
    def sync_to_server(store_snapshot):
        results.append(f"Syncing: {store_snapshot.name}, {store_snapshot.email}")

    # Initial execution with current store state
    assert len(results) == 1
    assert "Syncing: Alice, alice@example.com" in results[0]

    # Change name
    UserStore.name = "Bob"
    assert len(results) == 2
    assert "Syncing: Bob, alice@example.com" in results[1]

    # Change age (should still trigger)
    UserStore.age = 31
    assert len(results) == 3

    # Change email
    UserStore.email = "bob@example.com"
    assert len(results) == 4
    assert any("Syncing: Bob, bob@example.com" in r for r in results)

    print("✓ Store-level reaction example works!")


def test_computed_reactive():
    """Test the computed observable with @reactive example."""
    print("Testing computed observable with @reactive example...")

    class CartStore(Store):
        items = observable([])

    # Computed observable
    item_count = CartStore.items >> (lambda items: len(items))

    results = []

    @reactive(item_count)
    def update_badge(count):
        results.append(f"Cart badge: {count}")

    # Initial execution with computed value (0)
    assert results == ["Cart badge: 0"]

    # Add item
    CartStore.items = [{"name": "Widget", "price": 10}]
    assert results == ["Cart badge: 0", "Cart badge: 1"]

    # Add another item
    current_items = CartStore.items[:]  # Get current value
    CartStore.items = current_items + [{"name": "Gadget", "price": 15}]
    assert results == ["Cart badge: 0", "Cart badge: 1", "Cart badge: 2"]

    print("✓ Computed observable with @reactive example works!")


if __name__ == "__main__":
    test_basic_reactive()
    test_scattered_subscriptions()
    test_reactive_decorator()
    test_execution_model()
    test_declarative_side_effects()
    test_multiple_observables()
    test_cart_store_example()
    test_store_level_reaction()
    test_computed_reactive()
    print("\n🎉 All @reactive examples work correctly!")
