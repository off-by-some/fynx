"""
FynX ConditionalObservable - Conditional Reactive Computations
=============================================================

This module provides the ConditionalObservable class, a specialized computed observable
that enables filtering reactive streams based on boolean conditions. Conditional
observables only emit values from a source observable when all specified conditions
are satisfied, making them read-only computed observables.

When to Use Conditional Observables
-----------------------------------

Conditional observables are perfect for scenarios where you need to react to data
changes only under specific circumstances:

- **Guarded Updates**: Only reacting to changes when prerequisites are met
- **State Machines**: Different behavior based on application state
- **Resource Optimization**: Avoiding unnecessary computations when conditions aren't met
- **Event Filtering**: Processing events only under specific circumstances
- **Permission Checks**: Only updating UI when user has appropriate permissions

How It Works
------------

Conditional observables work by maintaining an internal cache of condition states.
They only emit values when:

1. **All conditions become True** (after previously being False)
2. **Source value changes** while all conditions remain True

This prevents unnecessary updates when conditions aren't met and provides precise
control over when reactive effects occur.

Creating Conditional Observables
---------------------------------

Conditional behavior is created using the `&` operator:

```python
from fynx.observable import Observable

# Source data
temperature = Observable("temp", 20)
humidity = Observable("humidity", 60)

# Conditions
is_heating_on = Observable("heating", False)
is_dry = Observable("dry", False)

# Conditional observable - only emits when heating is on AND it's dry
smart_humidity = humidity & is_heating_on & is_dry

smart_humidity.subscribe(lambda h: print(f"Adjusting humidity: {h}%"))
# Only prints when ALL conditions are True
```

Key Concepts
------------

- **Condition Chaining**: Multiple conditions combine with logical AND
- **Transition-Based**: Only triggers on false→true condition transitions
- **Value Filtering**: Source values only pass through when conditions are met
- **Immutable Composition**: Each `&` creates a new conditional observable

Performance Benefits
---------------------

- **Lazy Evaluation**: Conditions only re-evaluated when dependencies change
- **Efficient Filtering**: Prevents unnecessary reactive updates
- **Memory Conscious**: Minimal overhead for condition tracking
- **Composable**: Build complex conditions from simple boolean observables

Common Patterns
---------------

**Feature Flags**:
```python
data = Observable("data", [])
feature_enabled = Observable("feature", False)
user_premium = Observable("premium", True)

premium_data = data & feature_enabled & user_premium
```

**Form Validation**:
```python
email = Observable("email", "")
is_valid_email = Observable("valid", False)
user_consented = Observable("consent", False)

submittable_data = email & is_valid_email & user_consented
```

**Resource Availability**:
```python
updates = Observable("updates", [])
network_available = Observable("network", True)
battery_ok = Observable("battery", True)

sync_updates = updates & network_available & battery_ok
```

See Also
--------

- `fynx.watch`: For conditional reactive functions (alternative approach)
- `fynx.observable`: Core observable classes and operators
- `fynx.computed`: For derived reactive values
"""

from typing import List, TypeVar

from .computed import ComputedObservable
from .interfaces import Conditional, Observable
from .operators import OperatorMixin

T = TypeVar("T")


class ConditionalObservable(ComputedObservable[T], Conditional[T], OperatorMixin):
    """
    A computed observable that filters values from a source observable based on boolean conditions.

    ConditionalObservable is a specialized computed observable that creates a reactive stream
    that only emits values when ALL specified conditions are True. This enables precise
    control over when reactive updates occur, preventing unnecessary computations and side effects.

    As a computed observable, ConditionalObservable is read-only and derives its value from
    a source observable and boolean condition observables. It maintains an internal cache
    of the current condition state and only notifies subscribers when conditions transition
    from unmet to met, or when the source value changes while conditions remain met.

    Key Features:
    - **Condition Filtering**: Only emits when all conditions are satisfied
    - **Read-only**: Cannot be set directly (inherits from ComputedObservable)
    - **State Transitions**: Triggers on condition state changes
    - **Composable**: Can chain multiple conditions with additional `&` operators
    - **Memory Efficient**: Internal caching prevents redundant evaluations

    Example:
        ```python
        from fynx.observable import Observable

        # Source observable
        temperature = Observable("temp", 20)

        # Condition observables
        is_heating_enabled = Observable("heating", False)
        is_cold = Observable("cold_check", False)

        # Conditional observable - only emits when heating is enabled AND it's cold
        heating_trigger = temperature & is_heating_enabled & is_cold

        # Subscribe to conditional updates
        def activate_heating(temp):
            print(f"Activating heating at {temp}°C")

        heating_trigger.subscribe(activate_heating)

        # Only triggers when ALL conditions become true
        temperature.set(15)        # No trigger (heating not enabled)
        is_heating_enabled.set(True)  # No trigger (not marked as cold yet)
        is_cold.set(True)          # Triggers: "Activating heating at 15°C"
        temperature.set(10)        # Triggers: "Activating heating at 10°C"
        is_heating_enabled.set(False)  # Stops triggering
        ```

    Note:
        The conditional observable starts with `None` as its value if conditions
        are not initially met. It only takes on the source observable's value when
        all conditions become True.

    See Also:
        ComputedObservable: Parent class providing read-only behavior
        Observable: Base observable class
        MergedObservable: For combining multiple observables
        fynx.watch: For conditional reactive functions
    """

    def __init__(
        self,
        source_observable: "Observable[T]",
        *condition_observables: "Observable[bool]"
    ) -> None:
        """
        Create a conditional observable that filters values based on boolean conditions.

        The conditional observable will only emit values from the source observable
        when ALL condition observables are True. If no conditions are provided,
        the conditional observable behaves identically to the source observable.

        Args:
            source_observable: The observable whose values will be conditionally emitted.
                              This is the primary data source for the conditional stream.
            *condition_observables: Variable number of boolean observables that act as
                                   conditions. ALL conditions must be True for the source
                                   value to be emitted. Can be empty for unconditional behavior.

        Example:
            ```python
            from fynx.observable import Observable

            data = Observable("data", "hello")
            is_ready = Observable("ready", False)
            user_enabled = Observable("enabled", True)

            # Only emits when is_ready AND user_enabled are both True
            conditional_data = ConditionalObservable(data, is_ready, user_enabled)

            conditional_data.subscribe(lambda x: print(f"Received: {x}"))
            # Initially: no output (is_ready is False)

            is_ready.set(True)  # Now prints: "Received: hello"
            data.set("world")   # Prints: "Received: world"
            ```
        """
        # Call parent constructor with initial value (only if all conditions are met)
        initial_conditions_met = all(cond.value for cond in condition_observables)
        initial_value = source_observable.value if initial_conditions_met else None

        super().__init__("conditional", initial_value)
        self._source_observable = source_observable  # type: ignore
        self._condition_observables = list(condition_observables)
        self._conditions_met = initial_conditions_met

        # Set up observers
        def update_from_source():
            """Called when source observable changes."""
            # If conditions are currently met, emit the new source value
            if self._conditions_met:
                new_value = self._source_observable.value
                self._set_computed_value(new_value)

        def update_from_conditions():
            """Called when condition observables change."""
            # Update our cached condition state
            old_conditions_met = self._conditions_met
            self._conditions_met = all(
                cond.value for cond in self._condition_observables
            )

            # If conditions just became met, emit current source value
            if self._conditions_met and not old_conditions_met:
                current_value = self._source_observable.value
                self._set_computed_value(current_value)
            # If conditions became unmet, emit None to notify observers
            elif not self._conditions_met and old_conditions_met:
                self._set_computed_value(None)

        # Subscribe to all observables
        source_observable.add_observer(update_from_source)
        for cond_obs in condition_observables:
            cond_obs.add_observer(update_from_conditions)
