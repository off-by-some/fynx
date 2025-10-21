#!/usr/bin/env python3
"""
Examples demonstrating the @reactive decorator for conditional reactive programming.

This script showcases various patterns for using @reactive to create event-driven
reactive functions that trigger when specific conditions become true.

Run this script to see @reactive in action:

    python examples/using_reactive_conditionals.py

Each example demonstrates a different aspect of conditional reactive programming.
"""

import sys

from fynx import Store, observable, reactive


def basic_watch_example():
    """Demonstrate basic @reactive functionality with an age threshold.

    This example shows how @reactive triggers only when a condition transitions
    from False to True, not on every change.
    """
    print("=== Basic @reactive Example ===")
    print("Watching for when a user becomes an adult (age >= 18)")
    print()

    age = observable(16)

    # Create a conditional observable for the age threshold
    is_adult = age >> (lambda a: a >= 18)

    @reactive(is_adult)
    def on_becomes_adult_basic(is_adult_value):
        if is_adult_value:
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

    When multiple condition functions are passed to @reactive, all must be true
    for the decorated function to trigger. This implements logical AND behavior.
    """
    print("=== Multiple Conditions AND Logic ===")
    print("Both conditions must be true: has items AND is logged in")
    print()

    has_items = observable(False)
    is_logged_in = observable(False)

    # Create conditional observables for AND logic
    ready_to_checkout = (has_items + is_logged_in) >> (lambda h, l: h and l)

    @reactive(ready_to_checkout)
    def on_ready_to_checkout_and(ready_value):
        if ready_value:
            print("Ready to checkout - all conditions met!")

    # Demonstrate AND logic
    print("Initial state: has_items=False, is_logged_in=False")

    has_items.set(True)
    print("Set has_items=True (only one condition true, no trigger)")

    is_logged_in.set(True)
    print("Set is_logged_in=True (both conditions now true, triggers!)")

    # Clean up
    on_ready_to_checkout_and.unsubscribe()
    print()


def conditional_observable_or_example():
    """Demonstrate ConditionalObservable with the | operator.

    The | operator creates ConditionalObservable objects that combine multiple
    boolean observables with OR logic, emitting when ANY condition is true.
    """
    print("=== ConditionalObservable with | Operator ===")
    print("Using | operator for OR conditions: is_error | is_warning | is_critical")
    print()

    is_error = observable(False)
    is_warning = observable(True)  # Start with True to avoid ConditionalNeverMet
    is_critical = observable(False)

    # Create OR condition using | operator
    needs_attention = is_error | is_warning | is_critical

    @reactive(needs_attention)
    def on_attention_needed_or(needs_attention_state):
        if needs_attention_state:
            print("⚠️ System needs attention! (OR condition met)")

    # Demonstrate the | operator behavior
    print("Initial state: is_warning=True, others False")
    print("OR condition: False | True | False = True")

    is_error.set(True)
    print("Set is_error=True (2/3 conditions met, still triggers)")

    is_warning.set(False)
    print("Set is_warning=False (1/3 conditions met, still triggers)")

    is_error.set(False)
    print("Set is_error=False (0/3 conditions met, no longer triggers)")

    # Clean up
    on_attention_needed_or.unsubscribe()
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

    @reactive(has_items & is_logged_in & payment_valid)
    def on_ready_to_checkout_and_op(condition_value):
        print("Ready to checkout - all conditions met using & operator!")

    # Demonstrate the & operator behavior
    print("Initial state: has_items=False, is_logged_in=False, payment_valid=False")

    has_items.set(True)
    print("Set has_items=True (1/3 conditions met)")

    is_logged_in.set(True)
    print("Set is_logged_in=True (2/3 conditions met)")

    payment_valid.set(True)
    print("Set payment_valid=True (3/3 conditions met, triggers!)")

    # Clean up
    on_ready_to_checkout_and_op.unsubscribe()
    print()


def form_submission_flow_example():
    """Demonstrate a multi-step form submission workflow.

    This example shows how @reactive can be used to manage complex state transitions
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
    all_valid = (email_valid + password_valid + FormStore.terms_accepted) >> (
        lambda e, p, t: e and p and t
    )

    @reactive(all_valid)
    def on_form_valid_submit(is_valid):
        if is_valid:
            print("Form validation passed - submit button enabled")

    @reactive(FormStore.is_submitting)
    def on_submit_start_submit(is_submitting):
        if is_submitting:
            print("Form submission started")

    @reactive(FormStore.submission_complete)
    def on_submit_complete_submit(is_complete):
        if is_complete:
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

    # Clean up
    on_form_valid_submit.unsubscribe()
    on_submit_start_submit.unsubscribe()
    on_submit_complete_submit.unsubscribe()
    print()


def complex_conditions_example():
    """Demonstrate complex boolean conditions with logical operators.

    This example shows how @reactive can handle complex conditions involving
    AND, OR, and other logical operators within lambda functions.
    """
    print("=== Complex Boolean Conditions ===")
    print("Condition: temperature > 25 AND humidity < 60")
    print()

    temperature = observable(20)
    humidity = observable(50)

    # Create conditional observable for complex conditions
    is_comfortable = (temperature + humidity) >> (lambda t, h: t > 25 and h < 60)

    @reactive(is_comfortable)
    def on_comfortable_complex(comfortable_value):
        if comfortable_value:
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

    # Clean up
    on_comfortable_complex.unsubscribe()
    print()


def one_time_vs_repeating_events_example():
    """Demonstrate one-time events versus repeating milestone events.

    This example shows how @reactive can handle both events that occur only once
    and events that repeat at regular intervals.
    """
    print("=== One-time vs Repeating Events ===")
    print(
        "Tracking login milestones: first login (one-time) vs every 10 logins (repeating)"
    )
    print()

    login_count = observable(0)

    # Create conditional observables for milestone tracking
    is_first_login = login_count >> (lambda count: count == 1)

    @reactive(is_first_login)
    def on_first_login_milestone(is_first_value):
        if is_first_value:
            print(f"First login milestone reached at login #{login_count.value}")

    last_milestone = observable(0)
    is_milestone_login = (login_count + last_milestone) >> (
        lambda count, last: count >= last + 10
    )

    @reactive(is_milestone_login)
    def on_login_milestone_repeat(is_milestone_value):
        if is_milestone_value:
            last_milestone.set(login_count.value)
            print(f"Login milestone reached: {login_count.value} total logins")

    # Simulate user logins
    print("Simulating user logins:")
    for i in range(1, 26):
        login_count.set(i)
        if i in [1, 10, 20]:  # Show key milestones
            print(
                f"  Login #{i}: {'(first login triggered)' if i == 1 else '(milestone triggered)' if i in [10, 20] else ''}"
            )

    # Clean up
    on_first_login_milestone.unsubscribe()
    on_login_milestone_repeat.unsubscribe()
    print()


def computed_observables_combination_example():
    """Demonstrate combining @reactive with computed observables.

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
    can_checkout = (has_items + has_shipping + has_payment) >> (
        lambda items, shipping, payment: items and shipping and payment
    )

    @reactive(can_checkout)
    def on_checkout_ready_computed(can_checkout_value):
        if can_checkout_value:
            print("Checkout became available - all requirements met")

    # Demonstrate the checkout flow
    print("Building checkout state:")
    ShoppingCartStore.items = [{"name": "Widget", "price": 10}]
    print("Added items to cart")

    ShoppingCartStore.shipping_address = "123 Main St"
    print("Added shipping address")

    ShoppingCartStore.payment_method = "credit_card"
    print("Added payment method (all conditions now met, triggers checkout ready)")

    # Clean up
    on_checkout_ready_computed.unsubscribe()
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

    @reactive(user_logged_in & data_loaded & has_minimum_total)
    def on_ready_with_minimum_computed(condition_value):
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

    # Clean up
    on_ready_with_minimum_computed.unsubscribe()
    print()


def user_engagement_system_example():
    """Demonstrate a comprehensive user engagement tracking system.

    This example shows how @reactive can be used to create sophisticated engagement
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
        + UserActivityStore.actions_taken
        + UserActivityStore.time_on_site
    ) >> (
        lambda views, actions, time: min(100, (views * 5) + (actions * 10) + (time / 6))
    )

    # Create conditional observables for engagement levels
    low_engagement = engagement_score >> (lambda score: score >= 25)
    medium_engagement = engagement_score >> (lambda score: score >= 50)
    high_engagement = engagement_score >> (lambda score: score >= 75)

    @reactive(low_engagement)
    def on_low_engagement_user(has_low):
        if has_low:
            print(f"Low engagement reached (score: {engagement_score.value:.1f})")

    @reactive(medium_engagement)
    def on_medium_engagement_user(has_medium):
        if has_medium:
            print(f"Medium engagement reached (score: {engagement_score.value:.1f})")

    @reactive(high_engagement)
    def on_high_engagement_user(has_high):
        if has_high:
            print(f"High engagement reached (score: {engagement_score.value:.1f})")

    @reactive(UserActivityStore.has_account)
    def on_account_created_user(has_account):
        if has_account:
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

    # Clean up
    on_low_engagement_user.unsubscribe()
    on_medium_engagement_user.unsubscribe()
    on_high_engagement_user.unsubscribe()
    on_account_created_user.unsubscribe()
    print()


def order_processing_pipeline_example():
    """Demonstrate an order processing pipeline using @reactive.

    This example shows how @reactive can orchestrate complex multi-step workflows
    where each stage depends on the completion of previous stages.
    """
    print("=== Order Processing Pipeline ===")
    print("Multi-step workflow orchestration with @reactive")
    print()

    class OrderStore(Store):
        items = observable([])
        payment_verified = observable(False)
        inventory_reserved = observable(False)
        shipping_label_created = observable(False)
        shipped = observable(False)
        delivered = observable(False)

    # Pipeline stages - each @reactive represents a transition to the next stage
    has_items = OrderStore.items >> (lambda items: len(items) > 0)

    @reactive(has_items)
    def on_order_created_pipeline(has_items_value):
        if has_items_value:
            print("Order created - items added to cart")

    payment_and_items = (OrderStore.payment_verified + OrderStore.items) >> (
        lambda payment, items: payment and len(items) > 0
    )

    @reactive(payment_and_items)
    def on_payment_verified_pipeline(payment_and_items_value):
        if payment_and_items_value:
            print("Payment verified - proceeding to inventory check")
            # Simulate automatic inventory reservation
            OrderStore.inventory_reserved = True

    @reactive(OrderStore.inventory_reserved)
    def on_inventory_reserved_pipeline(inventory_reserved):
        if inventory_reserved:
            print("Inventory reserved - generating shipping label")
            # Simulate automatic label creation
            OrderStore.shipping_label_created = True

    @reactive(OrderStore.shipping_label_created)
    def on_label_created_pipeline(shipping_label_created):
        if shipping_label_created:
            print("Shipping label created - order ready for shipment")

    @reactive(OrderStore.shipped)
    def on_shipped_pipeline(shipped):
        if shipped:
            print("Order shipped - tracking number generated")

    @reactive(OrderStore.delivered)
    def on_delivered_pipeline(delivered):
        if delivered:
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

    # Subsequent stages are triggered automatically by the @reactive decorators
    # Each stage advances the order through the pipeline

    # Stage 5: Warehouse ships the order
    OrderStore.shipped = True
    print("Order shipped from warehouse")

    # Stage 6: Order is delivered
    OrderStore.delivered = True
    print("Order delivered to customer")

    # Clean up
    on_order_created_pipeline.unsubscribe()
    on_payment_verified_pipeline.unsubscribe()
    on_inventory_reserved_pipeline.unsubscribe()
    on_label_created_pipeline.unsubscribe()
    on_shipped_pipeline.unsubscribe()
    on_delivered_pipeline.unsubscribe()
    print()


def main():
    """Run all examples demonstrating @reactive functionality."""
    print(
        "Running @reactive examples demonstrating conditional reactive programming..."
    )
    print("=" * 70)
    print()

    basic_watch_example()
    multiple_conditions_and_example()
    conditional_observable_or_example()
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
        "These examples demonstrate the various ways to use @reactive for conditional reactive programming."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
