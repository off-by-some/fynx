#!/usr/bin/env python3
"""
Examples demonstrating the @watch decorator for conditional reactive programming.

This script showcases various patterns for using @watch to create event-driven
reactive functions that trigger when specific conditions become true.

Run this script to see @watch in action:

    python examples/using_watch.py

Each example demonstrates a different aspect of conditional reactive programming.
"""

import sys

from fynx import Store, observable, reactive


def basic_watch_example():
    """Demonstrate basic @watch functionality with an age threshold.

    This example shows how @watch triggers only when a condition transitions
    from false to true, not on every change.
    """
    print("=== Basic @watch Example ===")
    print("Watching for when a user becomes an adult (age >= 18)")
    print()

    age = observable(16)

    @watch(lambda: age.value >= 18)
    def on_becomes_adult():
        print(f"User became an adult at age {age.value}")

    # Demonstrate the transition behavior
    print(f"Initial age: {age.value}")
    age.set(17)
    print(f"Age set to: {age.value} (condition still false, no trigger)")

    age.set(18)
    print(f"Age set to: {age.value} (condition became true, triggered!)")

    age.set(19)
    print(f"Age set to: {age.value} (condition stays true, no additional trigger)")
    print()


def multiple_conditions_and_example():
    """Demonstrate multiple conditions with AND logic.

    When multiple condition functions are passed to @watch, all must be true
    for the decorated function to trigger. This implements logical AND behavior.
    """
    print("=== Multiple Conditions AND Logic ===")
    print("Both conditions must be true: has items AND is logged in")
    print()

    has_items = observable(False)
    is_logged_in = observable(False)

    @watch(lambda: has_items.value, lambda: is_logged_in.value)
    def on_ready_to_checkout():
        print("Ready to checkout - all conditions met!")

    # Demonstrate AND logic
    print("Initial state: has_items=False, is_logged_in=False")

    has_items.set(True)
    print("Set has_items=True (only one condition true, no trigger)")

    is_logged_in.set(True)
    print("Set is_logged_in=True (both conditions now true, triggers!)")

    print()


def conditional_observable_and_example():
    """Demonstrate ConditionalObservable with the & operator.

    The & operator creates ConditionalObservable objects that combine multiple
    boolean observables with AND logic. This provides a more concise syntax
    for complex conditions.
    """
    print("=== ConditionalObservable with & Operator ===")
    print(
        "Using & operator for AND conditions: has_items & is_logged_in & payment_valid"
    )
    print()

    has_items = observable(False)
    is_logged_in = observable(False)
    payment_valid = observable(False)

    @watch(has_items & is_logged_in & payment_valid)
    def on_ready_to_checkout():
        print("Ready to checkout - all conditions met using & operator!")

    # Demonstrate the & operator behavior
    print("Initial state: has_items=False, is_logged_in=False, payment_valid=False")

    has_items.set(True)
    print("Set has_items=True (1/3 conditions met)")

    is_logged_in.set(True)
    print("Set is_logged_in=True (2/3 conditions met)")

    payment_valid.set(True)
    print("Set payment_valid=True (3/3 conditions met, triggers!)")

    print()


def form_submission_flow_example():
    """Demonstrate a multi-step form submission workflow.

    This example shows how @watch can be used to manage complex state transitions
    in a form submission process, from validation to completion.
    """
    print("=== Form Submission Flow ===")
    print("Multi-step workflow: validation -> submission -> completion")
    print()

    class FormStore(Store):
        email = observable("")
        password = observable("")
        terms_accepted = observable(False)
        is_submitting = observable(False)
        submission_complete = observable(False)

    # Computed validations using the >> operator
    email_valid = FormStore.email >> (lambda e: "@" in e and len(e) > 3)

    password_valid = FormStore.password >> (lambda p: len(p) >= 8)

    # Combine all validations
    all_valid = (email_valid | password_valid | FormStore.terms_accepted) >> (
        lambda e, p, t: e and p and t
    )

    @watch(lambda: all_valid.value)
    def on_form_valid():
        print("Form validation passed - submit button enabled")

    @watch(lambda: FormStore.is_submitting.value)
    def on_submit_start():
        print("Form submission started")

    @watch(lambda: FormStore.submission_complete.value)
    def on_submit_complete():
        print("Form submission completed successfully")

    # Simulate user filling out the form
    print("User fills out form:")
    FormStore.email = "user@example.com"
    print("Email set")

    FormStore.password = "secure123"
    print("Password set")

    FormStore.terms_accepted = True
    print("Terms accepted (all validations now pass, triggers form valid)")

    # Simulate submission process
    print("\nUser submits form:")
    FormStore.is_submitting = True
    print("Submission initiated")

    # Simulate server response
    print("\nServer processes submission:")
    FormStore.submission_complete = True
    print("Submission completed")

    print()


def complex_conditions_example():
    """Demonstrate complex boolean conditions with logical operators.

    This example shows how @watch can handle complex conditions involving
    AND, OR, and other logical operators within lambda functions.
    """
    print("=== Complex Boolean Conditions ===")
    print("Condition: temperature > 25 AND humidity < 60")
    print()

    temperature = observable(20)
    humidity = observable(50)

    @watch(lambda: (temperature.value > 25 and humidity.value < 60))
    def on_comfortable():
        print(
            f"Climate became comfortable (temp: {temperature.value}, humidity: {humidity.value})"
        )

    # Demonstrate the complex condition logic
    print("Initial state: temperature=20, humidity=50")
    print("Condition: 20 > 25 AND 50 < 60 = False AND True = False")

    temperature.set(30)
    print("Set temperature=30")
    print("Condition: 30 > 25 AND 50 < 60 = True AND True = True (triggers)")

    # Reset and test alternative path
    temperature.set(20)
    humidity.set(70)
    print("\nReset to temperature=20, humidity=70")
    print("Condition: 20 > 25 AND 70 < 60 = False AND False = False")

    humidity.set(50)
    print("Set humidity=50")
    print("Condition: 20 > 25 AND 50 < 60 = False AND True = False (no trigger)")

    temperature.set(30)
    print("Set temperature=30")
    print("Condition: 30 > 25 AND 50 < 60 = True AND True = True (triggers again)")

    print()


def one_time_vs_repeating_events_example():
    """Demonstrate one-time events versus repeating milestone events.

    This example shows how @watch can handle both events that occur only once
    and events that repeat at regular intervals.
    """
    print("=== One-time vs Repeating Events ===")
    print(
        "Tracking login milestones: first login (one-time) vs every 10 logins (repeating)"
    )
    print()

    login_count = observable(0)

    @watch(lambda: login_count.value == 1)
    def on_first_login():
        print(f"First login milestone reached at login #{login_count.value}")

    last_milestone = 0

    @watch(lambda: login_count.value >= last_milestone + 10)
    def on_login_milestone():
        nonlocal last_milestone
        last_milestone = login_count.value
        print(f"Login milestone reached: {login_count.value} total logins")

    # Simulate user logins
    print("Simulating user logins:")
    for i in range(1, 26):
        login_count.set(i)
        if i in [1, 10, 20]:  # Show key milestones
            print(
                f"  Login #{i}: {'(first login triggered)' if i == 1 else '(milestone triggered)' if i in [10, 20] else ''}"
            )

    print()


def computed_observables_combination_example():
    """Demonstrate combining @watch with computed observables.

    This example shows how to use computed observables to derive complex state
    from simple observables, then watch for transitions on the computed values.
    """
    print("=== Computed Observables Combination ===")
    print("Using computed observables to determine checkout readiness")
    print()

    class ShoppingCartStore(Store):
        items = observable([])
        shipping_address = observable(None)
        payment_method = observable(None)

    # Computed boolean values
    has_items = ShoppingCartStore.items >> (lambda items: len(items) > 0)
    has_shipping = ShoppingCartStore.shipping_address >> (lambda addr: addr is not None)
    has_payment = ShoppingCartStore.payment_method >> (lambda pm: pm is not None)

    # Combine all conditions into a single computed observable
    can_checkout = (has_items | has_shipping | has_payment) >> (
        lambda items, shipping, payment: items and shipping and payment
    )

    @watch(lambda: can_checkout.value)
    def on_checkout_ready():
        print("Checkout became available - all requirements met")

    # Demonstrate the checkout flow
    print("Building checkout state:")
    ShoppingCartStore.items = [{"name": "Widget", "price": 10}]
    print("Added items to cart")

    ShoppingCartStore.shipping_address = "123 Main St"
    print("Added shipping address")

    ShoppingCartStore.payment_method = "credit_card"
    print("Added payment method (all conditions now met, triggers checkout ready)")

    print()


def conditional_observable_with_computed_example():
    """Demonstrate ConditionalObservable combined with computed values.

    This example shows how to mix the & operator (for AND conditions) with
    computed observables to create sophisticated conditional logic.
    """
    print("=== ConditionalObservable with Computed Values ===")
    print("Combining & operator with computed observables")
    print()

    user_logged_in = observable(False)
    data_loaded = observable(False)
    cart_total = observable(0)

    # Computed observable: cart meets minimum total
    has_minimum_total = cart_total >> (lambda total: total >= 50)

    @watch(user_logged_in & data_loaded & has_minimum_total)
    def on_ready_with_minimum():
        print(f"Ready for premium checkout (total: ${cart_total.value})")

    # Demonstrate the combined logic
    print("Setting up conditions:")
    user_logged_in.set(True)
    print("User logged in")

    data_loaded.set(True)
    print("Data loaded")

    cart_total.set(30)
    print("Cart total set to $30 (below minimum $50)")

    cart_total.set(60)
    print("Cart total set to $60 (meets minimum, all conditions now true, triggers)")

    print()


def user_engagement_system_example():
    """Demonstrate a comprehensive user engagement tracking system.

    This example shows how @watch can be used to create sophisticated engagement
    tracking with multiple threshold levels and different types of events.
    """
    print("=== User Engagement System ===")
    print("Multi-level engagement tracking with different event types")
    print()

    class UserActivityStore(Store):
        page_views = observable(0)
        actions_taken = observable(0)
        time_on_site = observable(0)  # seconds
        has_account = observable(False)
        is_premium = observable(False)

    # Computed engagement score: min(100, (views * 5) + (actions * 10) + (time / 6))
    engagement_score = (
        UserActivityStore.page_views
        | UserActivityStore.actions_taken
        | UserActivityStore.time_on_site
    ) >> (
        lambda views, actions, time: min(100, (views * 5) + (actions * 10) + (time / 6))
    )

    @watch(lambda: engagement_score.value >= 25)
    def on_low_engagement():
        print(f"Low engagement reached (score: {engagement_score.value:.1f})")

    @watch(lambda: engagement_score.value >= 50)
    def on_medium_engagement():
        print(f"Medium engagement reached (score: {engagement_score.value:.1f})")

    @watch(lambda: engagement_score.value >= 75)
    def on_high_engagement():
        print(f"High engagement reached (score: {engagement_score.value:.1f})")

    @watch(lambda: UserActivityStore.has_account.value)
    def on_account_created():
        print("Account created - user registration completed")

    # Simulate user engagement progression
    print("User engagement progression:")

    # Stage 1: Initial browsing (score: 22.5)
    UserActivityStore.page_views = 3
    UserActivityStore.time_on_site = 45
    print("User browses 3 pages for 45 seconds (score: 22.5)")

    # Stage 2: Low engagement reached (score: 42.5)
    UserActivityStore.actions_taken = 2
    print("User takes 2 actions (score: 42.5, triggers low engagement)")

    # Stage 3: Medium engagement reached (score: 57.5)
    UserActivityStore.page_views = 6
    print("User views 3 more pages (score: 57.5, triggers medium engagement)")

    # Stage 4: Account creation (separate event)
    UserActivityStore.has_account = True
    print("User creates account (triggers account creation event)")

    # Stage 5: Extended engagement (score: 70, still medium)
    UserActivityStore.time_on_site = 120
    print("User spends more time (score: 70, still medium engagement)")

    # Stage 6: High engagement reached (score: 80)
    UserActivityStore.time_on_site = 180
    print("User spends even more time (score: 80, triggers high engagement)")

    print()


def order_processing_pipeline_example():
    """Demonstrate an order processing pipeline using @watch.

    This example shows how @watch can orchestrate complex multi-step workflows
    where each stage depends on the completion of previous stages.
    """
    print("=== Order Processing Pipeline ===")
    print("Multi-step workflow orchestration with @watch")
    print()

    class OrderStore(Store):
        items = observable([])
        payment_verified = observable(False)
        inventory_reserved = observable(False)
        shipping_label_created = observable(False)
        shipped = observable(False)
        delivered = observable(False)

    # Pipeline stages - each @watch represents a transition to the next stage
    @watch(lambda: len(OrderStore.items.value) > 0)
    def on_order_created():
        print("Order created - items added to cart")

    @watch(
        lambda: OrderStore.payment_verified.value,
        lambda: len(OrderStore.items.value) > 0,
    )
    def on_payment_verified():
        print("Payment verified - proceeding to inventory check")
        # Simulate automatic inventory reservation
        OrderStore.inventory_reserved = True

    @watch(lambda: OrderStore.inventory_reserved.value)
    def on_inventory_reserved():
        print("Inventory reserved - generating shipping label")
        # Simulate automatic label creation
        OrderStore.shipping_label_created = True

    @watch(lambda: OrderStore.shipping_label_created.value)
    def on_label_created():
        print("Shipping label created - order ready for shipment")

    @watch(lambda: OrderStore.shipped.value)
    def on_shipped():
        print("Order shipped - tracking number generated")

    @watch(lambda: OrderStore.delivered.value)
    def on_delivered():
        print("Order delivered - customer notified")

    # Execute the order processing pipeline
    print("Processing customer order:")

    # Stage 1: Customer adds items
    OrderStore.items = [
        {"name": "Widget", "price": 29.99},
        {"name": "Gadget", "price": 49.99},
    ]
    print("Items added to cart")

    # Stage 2: Customer completes payment
    OrderStore.payment_verified = True
    print("Payment processed")

    # Subsequent stages are triggered automatically by the @watch decorators
    # Each stage advances the order through the pipeline

    # Stage 5: Warehouse ships the order
    OrderStore.shipped = True
    print("Order shipped from warehouse")

    # Stage 6: Order is delivered
    OrderStore.delivered = True
    print("Order delivered to customer")

    print()


def main():
    """Run all examples demonstrating @watch functionality."""
    print("Running @watch examples demonstrating conditional reactive programming...")
    print("=" * 70)
    print()

    basic_watch_example()
    multiple_conditions_and_example()
    conditional_observable_and_example()
    form_submission_flow_example()
    complex_conditions_example()
    one_time_vs_repeating_events_example()
    computed_observables_combination_example()
    conditional_observable_with_computed_example()
    user_engagement_system_example()
    order_processing_pipeline_example()

    print("=" * 70)
    print("All examples completed successfully.")
    print(
        "These examples demonstrate the various ways to use @watch for conditional reactive programming."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
