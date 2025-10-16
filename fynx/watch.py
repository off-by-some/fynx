"""
Fynx Watch - Conditional Reactive Utilities
===========================================

This module provides the `watch` decorator for creating conditional reactive
computations that only execute when specific conditions are met.

Conditional reactivity enables:
- **Guarded Reactions**: Functions that only run when prerequisites are satisfied
- **State Machines**: React differently based on application state
- **Event Filtering**: Only respond to changes when conditions allow
- **Resource Optimization**: Avoid unnecessary computations when not needed

The `watch` decorator automatically discovers observable dependencies in condition
functions and creates subscriptions that trigger only when ALL conditions become
true (after previously being false).

Key Concepts:
- **Condition Functions**: Pure functions that return boolean values
- **Dependency Discovery**: Automatic detection of accessed observables
- **Transition Detection**: Only triggers on true→false→true transitions
- **Error Resilience**: Handles condition evaluation failures gracefully

Example:
    ```python
    from fynx import observable, watch

    user_status = observable("offline")
    message_count = observable(0)

    @watch(
        lambda: user_status.value == "online",
        lambda: message_count.value > 0
    )
    def notify_user():
        print(f"📬 Notifying user: {message_count.value} new messages!")

    # Only triggers when user is online AND has messages
    user_status.set("online")  # No notification yet (no messages)
    message_count.set(3)       # Triggers notification!

    user_status.set("away")    # User goes away
    message_count.set(5)       # No notification (user away)
    ```
"""

from .observable import Observable


def watch(*conditions) -> callable:
    """
    Decorator for conditional reactive functions that run only when conditions are met.

    The `watch` decorator creates a reactive function that only executes when ALL
    specified conditions become true, after previously being false. This enables
    guarded reactions that wait for specific state combinations before triggering.

    The decorator automatically discovers which observables are accessed within the
    condition functions and sets up the appropriate subscriptions. When any of these
    observables change, the conditions are re-evaluated, and the decorated function
    runs only if this represents a transition from "not all conditions met" to
    "all conditions met".

    Args:
        *conditions: Variable number of condition functions. Each condition should be
                    a callable that returns a boolean value. Condition functions can
                    access observable values via `.value` attribute. All conditions
                    must return `True` for the decorated function to execute.

    Returns:
        A decorator function that can be applied to reactive functions.

    Examples:
        ```python
        from fynx import observable, watch

        # Basic conditional reaction
        user_logged_in = observable(False)
        data_loaded = observable(False)

        @watch(
            lambda: user_logged_in.value,
            lambda: data_loaded.value
        )
        def show_dashboard():
            print("Welcome to your dashboard!")

        # Only shows when both conditions are true
        user_logged_in.set(True)  # Not yet (data not loaded)
        data_loaded.set(True)     # Now shows dashboard!

        # State-based reactions
        app_state = observable("loading")
        error_count = observable(0)

        @watch(
            lambda: app_state.value == "error",
            lambda: error_count.value >= 3
        )
        def show_error_recovery():
            print("Too many errors - showing recovery options")

        # Advanced conditions with computations
        temperature = observable(20)
        humidity = observable(50)

        @watch(
            lambda: temperature.value > 30,
            lambda: humidity.value < 30
        )
        def activate_cooling():
            print("Hot and dry - activating cooling system!")

        # Conditions can be complex expressions
        @watch(lambda: temperature.value < 0 or temperature.value > 40)
        def extreme_temperature_alert():
            print("Extreme temperature detected!")
        ```

    Note:
        - Condition functions should be pure and relatively fast
        - The decorated function only runs on the transition from conditions not being
          met to conditions being met (not on every change while conditions remain true)
        - If condition evaluation fails during discovery or runtime, it's treated as False
        - Observables accessed in conditions are automatically tracked as dependencies

    See Also:
        reactive: For unconditional reactive functions
        computed: For derived reactive values
    """
    def decorator(func):
        # Track which observables are accessed during condition evaluation
        accessed_observables = set()

        class TrackingContext:
            """Context manager to track observable access during condition evaluation."""
            def __init__(self):
                self.subscribed_observable = None  # No observable being computed
                
            def __enter__(self):
                self._old_context = Observable._current_context
                self._accessed = accessed_observables
                Observable._current_context = self
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                Observable._current_context = self._old_context

            def add_dependency(self, observable):
                """Track that this observable was accessed."""
                self._accessed.add(observable)

        def evaluate_conditions():
            """Evaluate all conditions and return True if all pass."""
            try:
                return all(condition() for condition in conditions)
            except Exception:
                # If condition evaluation fails at runtime, treat as False
                return False

        previous_conditions_met = False

        def wrapped_reaction():
            """Check conditions and call func if all are met and this is a transition."""
            nonlocal previous_conditions_met
            current_conditions_met = evaluate_conditions()
            if current_conditions_met and not previous_conditions_met:
                func()
                previous_conditions_met = True
            elif not current_conditions_met:
                previous_conditions_met = False

        # Discover observables by evaluating conditions in tracking context
        # We need to evaluate each condition individually to discover all accessed observables,
        # since all() short-circuits and might not evaluate all conditions
        with TrackingContext():
            for condition in conditions:
                try:
                    condition()  # Evaluate each condition to discover accessed observables
                except Exception as e:
                    # If evaluation fails during discovery (e.g., uninitialized values),
                    # we'll still track the accessed observables
                    print(f"Warning: condition evaluation failed during discovery: {e}")
                    pass


        # Subscribe to all discovered observables
        for obs in accessed_observables:
            obs.add_observer(wrapped_reaction)

        # Run immediately if conditions are currently met
        wrapped_reaction()

        return func

    return decorator
