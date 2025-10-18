"""
FynX Watch - Conditional Reactive Utilities
===========================================

This module provides the `watch` decorator for creating conditional reactive
computations that only execute when specific conditions are met. Unlike `@reactive`
decorators that run on every change, `@watch` decorators only trigger when ALL
specified conditions transition from unmet to met.

When to Use Watch vs Reactive
------------------------------

**Use `@watch` when you need:**
- Functions that should only run when specific prerequisites are satisfied
- State machines that react differently based on application state
- Event filtering to avoid unnecessary operations when conditions aren't right
- Resource optimization by avoiding computations when not needed

**Use `@reactive` when you need:**
- Functions that should run on every change to their dependencies
- Unconditional side effects like logging or UI updates
- Immediate responses to any state change

Key Characteristics
-------------------

- **Conditional Execution**: Only runs when ALL conditions become true after being false
- **Transition Detection**: Triggers on false->true transitions, not ongoing true states
- **Automatic Discovery**: Framework finds observables accessed in condition functions
- **Multiple Conditions**: Supports AND logic across multiple conditions
- **Error Resilience**: Gracefully handles condition evaluation failures

Basic Usage
-----------

```python
from fynx import observable, watch

# Setup state
user_online = observable(False)
has_messages = observable(0)
notification_enabled = observable(True)

@watch(
    lambda: user_online.value,           # User must be online
    lambda: has_messages.value > 0,      # Must have messages
    lambda: notification_enabled.value   # Notifications must be enabled
)
def send_notification():
    print(f"📬 Sending {has_messages.value} messages to user!")

# Only triggers when ALL conditions become true
user_online.set(True)        # Not yet (no messages)
has_messages.set(3)          # Not yet (notifications disabled)
notification_enabled.set(True)  # Now triggers: "📬 Sending 3 messages to user!"

has_messages.set(5)          # Triggers again: "📬 Sending 5 messages to user!"
user_online.set(False)       # Stop triggering
has_messages.set(10)         # No trigger (user offline)
```

Advanced Patterns
-----------------

### Complex Conditions

Conditions can be arbitrarily complex expressions:

```python
temperature = observable(20)
humidity = observable(50)
ac_enabled = observable(True)

@watch(
    lambda: temperature.value > 25,              # Hot enough
    lambda: humidity.value < 60,                 # Not too humid
    lambda: ac_enabled.value,                    # AC is enabled
    lambda: temperature.value < 30 or humidity.value < 40  # Either very hot OR very dry
)
def activate_cooling():
    print("🌡️ Activating air conditioning!")
```

### State Machines

Use watch decorators to implement state machine transitions:

```python
app_state = observable("loading")
user_authenticated = observable(False)
data_loaded = observable(False)

@watch(lambda: app_state.value == "loading")
def show_loading_screen():
    print("⏳ Showing loading screen...")

@watch(
    lambda: user_authenticated.value,
    lambda: data_loaded.value,
    lambda: app_state.value == "ready"
)
def show_main_app():
    print("✅ Showing main application!")

# State transitions
app_state.set("authenticating")
user_authenticated.set(True)
app_state.set("loading_data")
data_loaded.set(True)
app_state.set("ready")  # Now triggers show_main_app()
```

### Resource Management

Prevent unnecessary operations when resources aren't available:

```python
network_available = observable(True)
battery_level = observable(100)
data_fresh = observable(False)

@watch(
    lambda: network_available.value,
    lambda: battery_level.value > 20,    # Don't sync on low battery
    lambda: not data_fresh.value         # Only sync when data is stale
)
def sync_data():
    print("🔄 Syncing data...")
    # Perform expensive network operation
    data_fresh.set(True)

network_available.set(False)
battery_level.set(15)
data_fresh.set(False)  # No sync (battery too low)

battery_level.set(80)
network_available.set(True)  # Triggers sync
```

Real-World Examples
-------------------

### User Authentication Flow

```python
login_attempted = observable(False)
credentials_valid = observable(False)
two_factor_complete = observable(False)

@watch(
    lambda: login_attempted.value,
    lambda: credentials_valid.value,
    lambda: two_factor_complete.value
)
def grant_access():
    print("🔐 Access granted!")
    # Redirect to dashboard, set session, etc.

# Login flow
login_attempted.set(True)
credentials_valid.set(True)
two_factor_complete.set(True)  # Access granted
```

### Shopping Cart Checkout

```python
cart_items = observable([])
payment_method = observable(None)
terms_accepted = observable(False)

@watch(
    lambda: len(cart_items.value) > 0,
    lambda: payment_method.value is not None,
    lambda: terms_accepted.value
)
def enable_checkout():
    print("💳 Checkout button enabled!")

cart_items.set([{"name": "Widget", "price": 10}])
payment_method.set("credit_card")
terms_accepted.set(True)  # Checkout now enabled
```

### Background Task Management

```python
task_queue = observable([])
is_online = observable(True)
battery_saving = observable(False)

@watch(
    lambda: len(task_queue.value) > 0,
    lambda: is_online.value,
    lambda: not battery_saving.value  # Don't run background tasks in battery saving mode
)
def process_background_tasks():
    print(f"🔄 Processing {len(task_queue.value)} background tasks...")

task_queue.set(["sync", "backup", "cleanup"])
battery_saving.set(True)     # No processing (battery saving)
battery_saving.set(False)    # Now triggers processing
```

Performance Considerations
--------------------------

**Efficient Evaluation**: Conditions are only re-evaluated when their dependencies change

**Dependency Tracking**: Only tracks observables actually accessed in conditions

**Transition Optimization**: Only triggers on condition state changes, not every update

**Memory Management**: Automatic cleanup when watch decorators are no longer needed

Best Practices
--------------

### Keep Conditions Simple

Break complex conditions into simpler, more focused ones:

```python
# Good: Simple, focused conditions
@watch(lambda: user.is_authenticated)
def load_user_data():
    pass

@watch(lambda: user.has_premium_plan)
def enable_premium_features():
    pass

# Avoid: Complex condition logic
@watch(lambda: user.is_authenticated and user.has_premium_plan and not user.is_banned)
def handle_premium_user():
    pass
```

### Use Descriptive Condition Names

Make conditions self-documenting:

```python
def user_is_eligible():
    return user.age >= 18 and user.country in ALLOWED_COUNTRIES

def payment_is_complete():
    return payment.status == "completed" and payment.amount > 0

@watch(user_is_eligible, payment_is_complete)
def process_purchase():
    pass
```

### Handle Errors Gracefully

Conditions that fail during evaluation are treated as False:

```python
@watch(
    lambda: user.preferences is not None,  # Safe null check
    lambda: user.preferences.get("notifications", False)  # Safe dict access
)
def send_notification():
    pass
```

Common Patterns
---------------

**Feature Flags**:
```python
feature_enabled = observable(False)
user_has_access = observable(False)

@watch(lambda: feature_enabled.value and user_has_access.value)
def enable_new_feature():
    print("✨ New feature enabled!")
```

**Data Validation**:
```python
email = observable("")
password = observable("")

@watch(
    lambda: "@" in email.value and "." in email.value.split("@")[1],
    lambda: len(password.value) >= 8
)
def enable_submit_button():
    print("✅ Submit button enabled")
```

**Resource Availability**:
```python
network_online = observable(True)
disk_space = observable(100)

@watch(
    lambda: network_online.value,
    lambda: disk_space.value > 10  # GB
)
def start_download():
    print("📥 Starting download...")
```

Limitations
-----------

- Conditions must be synchronous (no async/await)
- Only triggers on false->true transitions (not during sustained true states)
- Cannot create circular dependencies with watched observables
- Condition evaluation failures are treated as False

Troubleshooting
---------------

**Watch function not triggering**: Ensure all observables accessed in conditions are properly tracked

```python
# Wrong: External observable not tracked
external_flag = observable(True)
@watch(lambda: some_obs.value > 0)
def wrong_func():
    if external_flag.value:  # Not tracked!
        pass

# Right: Include all dependencies in conditions
@watch(
    lambda: some_obs.value > 0,
    lambda: external_flag.value
)
def right_func():
    pass
```

**Unexpected triggering**: Remember watch only triggers on transitions, not sustained states

**Performance issues**: Simplify complex conditions or break them into multiple watches

Comparison with Reactive
------------------------

| Feature | @reactive | @watch |
|---------|-----------|--------|
| Execution | Every change | Condition transitions only |
| Use case | Side effects | Conditional logic |
| Granularity | Fine-grained | Coarse-grained |
| Performance | Higher overhead | Lower overhead |
| Complexity | Simple | More complex |

See Also
--------

- `fynx.reactive`: Unconditional reactive functions
- `fynx.computed`: Derived reactive values
- `fynx.store`: Reactive state containers
- `fynx.observable`: Core observable classes
"""

from typing import Callable

from .observable import Observable


def watch(*conditions) -> Callable:
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
