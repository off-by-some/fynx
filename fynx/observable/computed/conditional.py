"""
Conditional observables for filtering and conditional logic.

This module provides ConditionalObservable, which creates observables that only
emit values when certain conditions are satisfied. This is useful for filtering
data streams, implementing conditional logic, and creating reactive pipelines
that respond to specific states.

## How Conditional Observables Work

Conditional observables act as filters that only let values through when all
specified conditions are satisfied. Think of them as gates that only open when
all conditions return True.

### Basic Usage

When you write:
```python
data & condition1 & condition2
```

This creates a filtered observable that only emits values when:
- `condition1(value) == True` AND
- `condition2(value) == True` AND
- ... (all conditions must be True)

### Key Properties

- **Filtering**: Only notifies when conditions are satisfied AND value changes
- **Commutative**: `data & cond1 & cond2` ≡ `data & cond2 & cond1`
- **Associative**: `(data & cond1) & cond2` ≡ `data & (cond1 & cond2)`

## Practical Usage

Example:
    ```python
    from fynx import observable

    # Create a conditional observable that only emits when value > 10
    data = observable(5)
    filtered = data & (lambda x: x > 10)

    # The filtered observable will only emit when data > 10
    data.set(15)  # filtered will emit 15
    data.set(8)   # filtered will not emit
    ```
"""

from typing import Any, Callable, List, Optional, Set, TypeVar, Union

from fynx.observable.computed.computed import ComputedObservable
from fynx.observable.core.abstract.derived import DerivedValue
from fynx.observable.core.abstract.operations import OperatorMixin
from fynx.observable.types.protocols.conditional_protocol import Conditional
from fynx.observable.types.protocols.observable_protocol import Observable

T = TypeVar("T")
Condition = Union[Observable[bool], Callable[[T], bool], "ConditionalObservable"]


class ConditionalNeverMet(Exception):
    """
    Raised when attempting to access the value of a ConditionalObservable
    whose conditions have never been satisfied.

    This exception indicates that the conditional observable has not yet
    received any values that meet its filtering criteria. Check `is_active`
    before accessing the value to avoid this error.

    Example:
        ```python
        filtered = data & (lambda x: x > 0)
        if filtered.is_active:
            value = filtered.value  # Safe to access
        else:
            # Handle case where conditions are not met
            pass
        ```
    """


class ConditionalNotMet(Exception):
    """
    Raised when attempting to access the value of a ConditionalObservable
    whose conditions are not currently satisfied.

    This exception indicates that the conditional observable's conditions
    were previously met but are not currently satisfied. The observable
    may have a cached value, but access is restricted. Check `is_active`
    before accessing the value to avoid this error.

    Example:
        ```python
        filtered = data & (lambda x: x > 0)
        try:
            value = filtered.value
        except ConditionalNotMet:
            # Handle case where conditions are not currently met
            # Could fall back to a default value or cached value if needed
            pass
        ```
    """


class ConditionalObservable(DerivedValue[T], Conditional[T], OperatorMixin):
    """
    A conditional observable that filters values based on one or more conditions.

    This observable only emits values when all specified conditions are satisfied.
    It acts as a filter that blocks values that don't meet the criteria.

    ## How It Works

    ```
    Source Data → [Condition 1] → [Condition 2] → [Condition 3] → Filtered Output
                     ↓              ↓              ↓
                   Check A        Check B        Check C
    ```

    Only values where ALL conditions return `True` make it through to the output.

    ## Key Properties

    - **Filtering**: Only notifies when conditions are satisfied AND value changes
    - **Commutative**: `data & cond1 & cond2` ≡ `data & cond2 & cond1` (order doesn't matter)
    - **Associative**: `(data & cond1) & cond2` ≡ `data & (cond1 & cond2)` (grouping doesn't matter)

    ## Behavior States

    - **Active State**: All conditions are satisfied → values flow through
    - **Inactive State**: Any condition fails → no values pass
    - **Value Access**:
      - Active: Returns the current filtered value
      - Inactive: Raises ConditionalNotMet or ConditionalNeverMet
    - **Notifications**: Only emits when transitioning to active state or value changes while active

    ## Empty Conditions (Special Case)

    When no conditions are provided, the filter is always open:
    ```python
    always_open = data & ()  # Equivalent to just `data`
    ```

    This is useful for optimizer fusion where conditions might be empty during intermediate steps.

    ## Examples

    Basic filtering:
        ```python
        from fynx import observable

        data = observable(42)
        filtered = data & (lambda x: x > 10)  # Only values > 10 get through

        data.set(15)  # filtered emits 15 (passes the check)
        data.set(5)   # filtered becomes inactive (fails the check)
        ```

    Multiple conditions:
        ```python
        is_positive = observable(True)
        is_even = lambda x: x % 2 == 0

        filtered = data & is_positive & is_even
        # Only emits when data > 0 AND data is even
        ```

    Chaining conditionals:
        ```python
        step1 = data & (lambda x: x > 0)      # First filter
        step2 = step1 & (lambda x: x < 100)   # Second filter
        # Equivalent to: data & (lambda x: x > 0) & (lambda x: x < 100)
        ```
    """

    def __init__(
        self, source_observable: "Observable[T]", *conditions: Condition
    ) -> None:
        """
        Create a conditional observable that filters values based on conditions.

        Args:
            source_observable: The observable whose values will be conditionally emitted.
            *conditions: Variable number of conditions that form the filtering criteria.
                        Each condition can be:
                        - Observable[bool]: A boolean observable (external condition)
                        - Callable: A predicate function that takes the source value and returns bool
                        - ConditionalObservable: A compound condition (nested filter)

        Raises:
            ValueError: If source_observable is None
            TypeError: If conditions contain invalid types

        Empty Conditions Behavior:
            When no conditions are provided (`*conditions` is empty), the filter is always open:
            ```python
            always_open = ConditionalObservable(data)  # Equivalent to just `data`
            ```

            This is useful for:
            - Optimizer fusion during intermediate steps
            - Creating "pass-through" observables
            - Testing and debugging scenarios

        Examples:
            ```python
            from fynx import observable

            # Single predicate condition
            data = observable(42)
            positive = data & (lambda x: x > 0)

            # Multiple conditions
            filtered = data & (lambda x: x > 0) & (lambda x: x < 100)

            # Mixed condition types
            is_ready = observable(True)
            valid_data = data & is_ready & (lambda x: x % 2 == 0)

            # Nested conditionals
            step1 = data & (lambda x: x > 0)
            step2 = step1 & (lambda x: x < 100)

            # Empty conditions (always open filter)
            always_open = ConditionalObservable(data)  # Just passes through
            ```
        """
        # Validate inputs
        self._validate_inputs(source_observable, conditions)

        # Store the original source and conditions for reference
        self._source_observable = source_observable
        self._conditions = conditions  # Keep original name for test compatibility

        # Process conditions and create optimized observables
        self._processed_conditions = self._process_conditions(
            source_observable, conditions
        )

        # Determine initial state - only check local conditions, not the entire chain
        # This avoids double-evaluation of conditions
        self._conditions_met = (
            self._check_local_conditions_satisfied()
        )  # Keep original name for test compatibility
        self._has_ever_had_valid_value = self._conditions_met

        # Get initial value
        initial_value = self._get_initial_value()

        # Initialize the base observable WITHOUT the source to prevent automatic subscription
        # We'll handle the subscription manually with proper conditional logic
        super().__init__("conditional", initial_value, source_observable)

        # Set up dependency tracking and observers
        self._all_dependencies = self._find_all_dependencies()
        self._setup_observers()

        # If conditions are initially met, update the value
        if self._conditions_met:
            self._has_ever_had_valid_value = True
            self._value_wrapper.value = self._get_root_source_value()

    def _compute_value(self) -> T:
        """Pass through source value (no transformation)."""
        return self._source_observable.value

    def _should_recompute(self) -> bool:
        """Check if recomputation is needed."""
        return self._is_dirty and self._conditions_met

    def _on_source_change(self, value: Any) -> None:
        """Handle source changes."""
        self._update_conditional_state()

    def _update_conditional_state(self) -> None:
        """
        Update the conditional state when dependencies change.

        This method is called whenever any dependency changes value.
        """
        previous_conditions_satisfied = self._conditions_met
        previous_value = self._value_wrapper.value

        # Check current condition state
        self._conditions_met = self._check_if_conditions_are_satisfied()

        # Only update value and notify if conditions are satisfied
        if self._conditions_met:
            # For nested conditionals, source might be inactive - get root source value
            current_source_value = self._get_root_source_value()

            # Update value if it changed or if conditions just became met for the first time
            should_update = (
                self._value_wrapper.value != current_source_value
                or not self._has_ever_had_valid_value
            )

            if should_update:
                self._has_ever_had_valid_value = True
                # Setting value triggers _on_value_change which calls _notify_observers
                self._value_wrapper.value = current_source_value
        else:
            # Conditions are not met - update internal state but don't notify
            # This handles the case where we're notified by source but conditions are unmet
            if self._has_ever_had_valid_value:
                # We had a valid value before, so we're transitioning from active to inactive
                # Don't notify observers - this maintains pullback semantics
                pass

    def _should_notify_observers(self) -> bool:
        """
        Override hook from BaseObservable.

        Returns False when conditions are not met, preventing the
        epoch notification system from calling observers.
        """
        return self._conditions_met

    def _validate_inputs(
        self, source_observable: "Observable[T]", conditions: tuple
    ) -> None:
        """
        Validate the inputs to the constructor.

        Raises appropriate exceptions for invalid inputs.
        """
        if source_observable is None:
            raise ValueError("source_observable cannot be None")

        # Allow empty conditions for optimizer fusion - represents "always active" conditional
        # if not conditions:
        #     raise ValueError("At least one condition must be provided")

        # Validate each condition
        for i, condition in enumerate(conditions):
            if condition is None:
                raise ValueError(f"Condition {i} cannot be None")

            # Check if condition is a valid type (check class hierarchy to avoid triggering property access)
            from fynx.observable.core.abstract.observable import BaseObservable
            from fynx.observable.core.observable import Observable

            is_observable = isinstance(condition, Observable)
            is_base_observable = isinstance(condition, BaseObservable)
            is_observable_value = hasattr(condition, "observable") and hasattr(
                condition, "value"
            )
            is_callable = callable(condition)
            from .conditional import ConditionalObservable

            is_conditional = isinstance(condition, ConditionalObservable)

            if not (
                is_observable
                or is_base_observable
                or is_observable_value
                or is_callable
                or is_conditional
            ):
                raise TypeError(
                    f"Condition {i} must be an Observable, ObservableValue, callable, or ConditionalObservable, "
                    f"got {type(condition).__name__}"
                )

    def _process_conditions(
        self, source: "Observable[T]", conditions: tuple
    ) -> List[Any]:
        """
        Process raw conditions into optimized observables.

        For callable conditions, we keep them as-is since they will be
        evaluated dynamically against the source value. This avoids
        creating unnecessary computed observables.

        For ObservableValue conditions, we unwrap them to get the underlying observable.
        """
        processed = []
        for condition in conditions:
            if hasattr(condition, "observable") and hasattr(condition, "value"):
                # Unwrap ObservableValue-like objects to get the underlying observable
                processed.append(condition.observable)
            else:
                # Keep conditions as provided; evaluation is handled dynamically
                processed.append(condition)
        return processed

    def _check_if_conditions_are_satisfied(self) -> bool:
        """
        Check if all conditions are currently satisfied.

        Returns False if the source is inactive or any condition fails.
        If no conditions are provided, always returns True (always active).
        """
        # If no conditions, always active (for optimizer fusion)
        if not self._processed_conditions:
            return True

        # Get the root source value for condition evaluation
        root_value = self._get_root_source_value()

        # Collect all conditions from the entire chain
        all_conditions = self._collect_all_conditions()

        # Evaluate all conditions against the root source value
        return self._evaluate_all_conditions(root_value, all_conditions)

    def _evaluate_and_notify(self):
        """Override to only notify when conditions are met."""
        # Update conditions met status
        self._conditions_met = self._check_if_conditions_are_satisfied()
        self._has_ever_had_valid_value = (
            self._has_ever_had_valid_value or self._conditions_met
        )

        # For conditional observables, only notify when active
        if not self._conditions_met:
            # Mark as evaluated but don't notify
            self._is_dirty = False
            return

        # Otherwise, use the parent's evaluation and notification
        super()._evaluate_and_notify()

    def _check_local_conditions_satisfied(self) -> bool:
        """
        Check if only the local conditions (not the entire chain) are satisfied.

        Used during construction to avoid double-evaluation of conditions.
        """
        # If source is inactive, we're inactive
        is_conditional = self._is_conditional_observable(self._source_observable)

        if is_conditional and not self._source_observable.is_active:
            return False

        # If no conditions, always active
        if not self._processed_conditions:
            return True

        # Evaluate only local conditions
        try:
            source_value = self._source_observable.value
        except (ConditionalNeverMet, ConditionalNotMet):
            # If source is not available, conditions are not satisfied
            return False
        return self._evaluate_all_conditions(source_value, self._processed_conditions)

    def _is_conditional_observable(self, obj: Any) -> bool:
        """Check if an object is a ConditionalObservable without triggering value access."""
        try:
            return (
                hasattr(obj, "_conditions_met")
                and hasattr(obj, "is_active")
                and hasattr(obj, "_processed_conditions")
            )
        except Exception:
            return False

    def _collect_all_conditions(self) -> List[Any]:
        """Collect all conditions from the entire conditional chain."""
        conditions = []

        # Add conditions from the current level
        conditions.extend(self._processed_conditions)

        # Add conditions from source conditionals recursively
        # Only collect if the source hasn't already been evaluated (avoid duplicate evaluations)
        if self._is_conditional_observable(self._source_observable):
            conditions.extend(self._source_observable._collect_all_conditions())

        return conditions

    def _get_root_source_value(self) -> T:
        """Get the value from the root source observable without triggering condition evaluation."""
        current_source = self._source_observable
        while self._is_conditional_observable(current_source):
            current_source = current_source._source_observable
        # For regular observables, just get the value
        # For conditionals, we've already navigated to the root
        return current_source.value

    def _is_source_inactive(self) -> bool:
        """Check if the source observable is inactive (for conditional sources)."""
        return (
            isinstance(self._source_observable, Conditional)
            and not self._source_observable.is_active
        )

    def _get_initial_value(self) -> T:
        """Get the initial value for the conditional observable."""
        # If conditions are not met, return None as initial value
        if not self._conditions_met:
            return None

        # Avoid accessing private attributes of other objects; fall back to None when inactive
        if isinstance(self._source_observable, Conditional):
            try:
                return self._source_observable.value if self._source_observable.is_active else None  # type: ignore
            except (ConditionalNeverMet, ConditionalNotMet):
                return None
        try:
            return self._source_observable.value
        except (ConditionalNeverMet, ConditionalNotMet):
            return None

    def _evaluate_all_conditions(
        self, source_value: T, conditions: List[Any] = None
    ) -> bool:
        """
        Evaluate all conditions against the source value.

        Returns True only if all conditions are satisfied.
        """
        if conditions is None:
            conditions = self._processed_conditions

        for condition in conditions:
            if not self._evaluate_single_condition(condition, source_value):
                return False
        return True

    def _evaluate_single_condition(self, condition: Any, source_value: T) -> bool:
        """
        Evaluate a single condition against the source value.

        Handles different types of conditions appropriately.
        """
        from fynx.observable.core.abstract.observable import BaseObservable

        if isinstance(condition, ConditionalObservable):
            # Compound/public conditional interface - use its public state
            return condition.is_active
        if isinstance(condition, BaseObservable):
            # Regular observable - use its current value
            return bool(condition.value)
        if callable(condition):
            # Callable - evaluate against source value
            return self._evaluate_callable_condition(condition, source_value)
        # Unknown condition type - treat as falsy
        return False

    def _evaluate_callable_condition(
        self, condition: Callable, source_value: T
    ) -> bool:
        """
        Evaluate a callable condition against the source value.
        """
        if isinstance(source_value, tuple):
            # For merged observables, unpack tuple
            return bool(condition(*source_value))
        # Single value
        return bool(condition(source_value))

    def _find_all_dependencies(self) -> Set[Observable]:
        """
        Find all observable dependencies for this conditional.

        Includes the source observable and all condition observables.
        For nested conditionals, recursively finds dependencies.
        """
        dependencies = set()

        # Only add the source observable if it's not a conditional
        # For conditionals, we depend on their dependencies instead
        if self._is_conditional_observable(self._source_observable):
            # For conditional sources, depend on their dependencies
            dependencies.update(self._source_observable._all_dependencies)
        else:
            # For non-conditional sources, depend on them directly
            dependencies.add(self._source_observable)

        # Add dependencies from each condition
        for condition in self._processed_conditions:
            if isinstance(condition, Observable):
                dependencies.add(condition)
            elif isinstance(condition, Conditional):
                # For nested conditionals, add their dependencies
                dependencies.update(condition._all_dependencies)

        # Filter out None values
        return {dep for dep in dependencies if dep is not None}

    def _extract_condition_dependencies(self, condition: Any) -> Set[Observable]:
        # Deprecated: dependencies are gathered via public Observable interface only
        return set()

    def _setup_observers(self) -> None:
        """
        Set up observers for all dependencies.

        Safely handles None dependencies and missing subscribe methods.
        """

        def handle_value_change(value):
            """Handle changes to source or condition values."""
            self._update_conditional_state()

        # Subscribe to all dependencies using public interface
        for dependency in self._all_dependencies:
            dependency.subscribe(handle_value_change)

        # Add cycle detection for the source dependency
        if self._source_observable is not None:
            from fynx.observable.core.context import ReactiveContextImpl

            cycle_detector = ReactiveContextImpl._get_cycle_detector()
            cycle_detector.add_edge(self._source_observable, self)

    def _should_notify_observers(self) -> bool:
        """
        Override hook from BaseObservable.

        Returns False when conditions are not met, preventing the
        epoch notification system from calling observers.
        """
        return self._conditions_met

    @property
    def value(self) -> T:
        """
        Current value of the conditional observable.

        Returns the source observable's value when conditions are satisfied.
        Raises ConditionalNeverMet if conditions have never been satisfied.
        Raises ConditionalNotMet if conditions were previously satisfied but are not currently satisfied.

        This provides the expected behavior where conditional observables act as filters
        that pass through the source value when conditions are met.

        **Gate Analogy:**
            This is like asking "What's the current value at the filter?"
            - If filter is open: Returns the value that passed through
            - If filter is closed: Raises an exception (no value to return)

        When active: Returns values that satisfy all conditions
        When inactive: Filter is closed, no value available (raises exception)

        Example:
            ```python
            data = observable(5)
            filtered = data & (lambda x: x > 10)  # Filter checks: "Is value > 10?"

            try:
                value = filtered.value  # Raises ConditionalNeverMet (filter never opened)
            except ConditionalNeverMet:
                print("Filter has never been open")

            data.set(15)
            value = filtered.value  # Returns 15 (filter is open, value passed through)
            ```
        """
        if self.is_active:
            # Conditions are satisfied - return the current value
            return self._value_wrapper.value
        elif self._has_ever_had_valid_value:
            # Conditions were previously satisfied but are not now
            raise ConditionalNotMet("Conditions are not currently satisfied")
        else:
            # Conditions have never been satisfied
            raise ConditionalNeverMet("Conditions have never been satisfied")

    @property
    def is_active(self) -> bool:
        """
        True if conditions are currently satisfied (filter is open).

        This property indicates whether the conditional observable is currently
        in an active state where it can emit values. Think of it as checking
        if the filter is currently open.

        **Filter States:**

        When active (filter open):
        - ✅ Can emit values through notifications
        - ✅ Allows safe access to `.value` property
        - ✅ Represents a non-empty filtered subset
        - 🚪 Filter is open, values flow through

        When inactive (filter closed):
        - ❌ Does not emit notifications
        - ❌ Raises exceptions when accessing `.value`
        - ❌ Represents an empty filtered subset
        - 🚪 Filter is closed, no values pass

        Example:
            ```python
            data = observable(5)
            filtered = data & (lambda x: x > 10)  # Filter checks: "Is value > 10?"

            print(filtered.is_active)  # False (5 <= 10, filter is closed)

            data.set(15)
            print(filtered.is_active)  # True (15 > 10, filter is open)
            ```
        """
        return self._conditions_met

    def get_debug_info(self) -> dict:
        """
        Get debugging information about the conditional observable.

        Returns a dictionary with useful debugging information including
        condition states, dependencies, and current values.
        """
        # Collect all conditions across the entire chain for comprehensive reporting
        all_conditions = self._collect_all_conditions()

        debug_info = {
            "is_active": self.is_active,
            "has_ever_had_valid_value": self._has_ever_had_valid_value,
            "current_value": self._value_wrapper.value,
            "source_value": self._source_observable.value,
            "conditions_count": len(all_conditions),
            "dependencies_count": len(self._all_dependencies),
        }

        # Add condition-specific debug info
        condition_states = []
        for i, condition in enumerate(all_conditions):
            if isinstance(condition, Conditional):
                condition_states.append(
                    {
                        "index": i,
                        "type": "Conditional",
                        "is_active": condition.is_active,
                    }
                )
            elif isinstance(condition, Observable):
                condition_states.append(
                    {
                        "index": i,
                        "type": "Observable",
                        "value": condition.value,
                        "is_truthy": bool(condition.value),
                    }
                )
            elif callable(condition):
                source_value = self._source_observable.value
                if isinstance(source_value, tuple):
                    result = condition(*source_value)
                else:
                    result = condition(source_value)
                condition_states.append(
                    {
                        "index": i,
                        "type": "Callable",
                        "result": result,
                        "is_truthy": bool(result),
                    }
                )

        debug_info["condition_states"] = condition_states
        return debug_info
