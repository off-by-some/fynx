"""
Conditional observables for filtering and conditional logic.

This module provides ConditionalObservable, which creates observables that only
emit values when certain conditions are satisfied. This is useful for filtering
data streams, implementing conditional logic, and creating reactive pipelines
that respond to specific states.

## Mathematical Foundation: Pullbacks as Smart Gates

Conditional observables are built on the mathematical concept of pullbacks, but you can think of them as **smart gates** that only let values through when all conditions agree.

### The Intuitive Analogy

Think of a pullback like a security checkpoint:
- **Source data** = People trying to get through
- **Conditions** = Security guards checking different criteria
- **Pullback** = The gate that only opens when ALL guards say "yes"

```
People → [Guard1: Age>18] → Gate → [Guard2: HasTicket] → Gate → [Guard3: NoWeapons] → Gate → Allowed Through
```

Only people who pass ALL checks get through. This is exactly what `data & condition1 & condition2` does!

### What We're Actually Building

When you write:
```python
data & condition1 & condition2
```

We're creating a chain of gates:
```python
((data & condition1) & condition2)
```

Each `&` adds another security check. The final result only lets values through when:
- `condition1(value) == True` AND
- `condition2(value) == True` AND
- ... (all conditions must be True)

### The Mathematical Reality

Under the hood, this creates a filtered set:
```python
ConditionalObservable(source, c1, c2, ..., cn) ≅
    {s ∈ source | c1(s) ∧ c2(s) ∧ ... ∧ cn(s)}
```

Translation: "All values from source where every condition says True"

### Why Order Doesn't Matter

The `&` operator is commutative (`data & cond1 & cond2` ≡ `data & cond2 & cond1`) because:
- All conditions check the same thing: "Is this True?"
- Multiple "True" checks can be done in any order
- It's like having multiple security guards - the order doesn't matter, they all need to say "yes"

This gives us the intuitive behavior you'd expect from logical AND operations.

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

## Key Properties

- **Pullback Semantics**: Only notifies when conditions are satisfied AND value changes
- **Commutative**: `data & cond1 & cond2` ≡ `data & cond2 & cond1`
- **Associative**: `(data & cond1) & cond2` ≡ `data & (cond1 & cond2)`
- **Universal**: Any compatible reactive function factors through the conditional uniquely
"""

from typing import Any, Callable, List, Set, TypeVar, Union

from .computed import ComputedObservable
from .interfaces import Conditional, Observable
from .operators import OperatorMixin

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


class ConditionalObservable(ComputedObservable[T], Conditional[T], OperatorMixin):
    """
    A conditional observable that filters values based on one or more conditions.

    Think of this as a **smart gate** that only lets values through when all conditions
    are satisfied. It's like having multiple security guards - only values that pass
    ALL checks get through.

    ## How It Works (The Gate Analogy)

    ```
    Source Data → [Condition 1] → [Condition 2] → [Condition 3] → Filtered Output
                     ↓              ↓              ↓
                   Check A        Check B        Check C
    ```

    Only values where ALL conditions return `True` make it through to the output.

    ## Mathematical Foundation

    A ConditionalObservable represents the filtered subset:
    ```
    ConditionalObservable(source, c1, c2, ..., cn) ≅
        {s ∈ source | c1(s) ∧ c2(s) ∧ ... ∧ cn(s)}
    ```

    Translation: "All values from source where every condition says True"

    ## Key Properties

    - **Smart Gate Behavior**: Only notifies when conditions are satisfied AND value changes
    - **Commutative**: `data & cond1 & cond2` ≡ `data & cond2 & cond1` (order doesn't matter)
    - **Associative**: `(data & cond1) & cond2` ≡ `data & (cond1 & cond2)` (grouping doesn't matter)
    - **Universal**: Any compatible reactive function factors through the conditional uniquely

    ## Behavior States

    - **Active State**: All conditions are satisfied → gate is open, values flow through
    - **Inactive State**: Any condition fails → gate is closed, no values pass
    - **Value Access**:
      - Active: Returns the current filtered value
      - Inactive: Raises ConditionalNotMet or ConditionalNeverMet
    - **Notifications**: Only emits when transitioning to active state or value changes while active

    ## Empty Conditions (Special Case)

    When no conditions are provided, the gate is always open:
    ```python
    always_open = data & ()  # Equivalent to just `data`
    ```

    This is useful for optimizer fusion where conditions might be empty during intermediate steps.

    ## Examples

    Basic filtering (single guard):
        ```python
        from fynx import observable

        data = observable(42)
        filtered = data & (lambda x: x > 10)  # Only values > 10 get through

        data.set(15)  # filtered emits 15 (passes the check)
        data.set(5)   # filtered becomes inactive (fails the check)
        ```

    Multiple conditions (multiple guards):
        ```python
        is_positive = observable(True)
        is_even = lambda x: x % 2 == 0

        filtered = data & is_positive & is_even
        # Only emits when data > 0 AND data is even
        ```

    Chaining conditionals (checkpoint stations):
        ```python
        step1 = data & (lambda x: x > 0)      # First checkpoint
        step2 = step1 & (lambda x: x < 100)   # Second checkpoint
        # Equivalent to: data & (lambda x: x > 0) & (lambda x: x < 100)
        ```
    """

    def __init__(
        self, source_observable: "Observable[T]", *conditions: Condition
    ) -> None:
        """
        Create a conditional observable that filters values based on conditions.

        This constructor builds a smart gate that represents the filtered subset:
        ```
        ConditionalObservable(source, c1, c2, ..., cn) ≅
            {s ∈ source | c1(s) ∧ c2(s) ∧ ... ∧ cn(s)}
        ```

        Args:
            source_observable: The observable whose values will be conditionally emitted.
                              This is the "source" in the gate analogy.
            *conditions: Variable number of conditions that form the filtering criteria.
                        Each condition can be:
                        - Observable[bool]: A boolean observable (external condition)
                        - Callable: A predicate function that takes the source value and returns bool
                        - ConditionalObservable: A compound condition (nested gate)

        Raises:
            ValueError: If source_observable is None
            TypeError: If conditions contain invalid types

        Empty Conditions Behavior:
            When no conditions are provided (`*conditions` is empty), the gate is always open:
            ```python
            always_open = ConditionalObservable(data)  # Equivalent to just `data`
            ```

            This is useful for:
            - Optimizer fusion during intermediate steps
            - Creating "pass-through" observables
            - Testing and debugging scenarios

        Mathematical Interpretation:
            This creates the gate P in the diagram:
            ```
            Source → [Condition 1] → [Condition 2] → ... → Gate P → Filtered Output
            ```

            Where P = {s ∈ source | ∀c ∈ conditions: c(s) = True}

            If no conditions: P = source (gate always open)

        Examples:
            ```python
            from fynx import observable

            # Single predicate condition (one guard)
            data = observable(42)
            positive = data & (lambda x: x > 0)

            # Multiple conditions (multiple guards)
            filtered = data & (lambda x: x > 0) & (lambda x: x < 100)

            # Mixed condition types
            is_ready = observable(True)
            valid_data = data & is_ready & (lambda x: x % 2 == 0)

            # Nested conditionals (checkpoint stations)
            step1 = data & (lambda x: x > 0)
            step2 = step1 & (lambda x: x < 100)

            # Empty conditions (always open gate)
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

        # Initialize the base observable, passing the source for ComputedObservable
        super().__init__("conditional", initial_value, None, source_observable)

        # Set up dependency tracking and observers
        self._all_dependencies = self._find_all_dependencies()
        self._setup_observers()

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

            # Check if condition is a valid type
            is_observable = isinstance(condition, Observable)
            is_observable_value = hasattr(condition, "observable") and hasattr(
                condition, "value"
            )
            is_callable = callable(condition)
            is_conditional = isinstance(condition, Conditional)

            if not (
                is_observable or is_observable_value or is_callable or is_conditional
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

    def _check_local_conditions_satisfied(self) -> bool:
        """
        Check if only the local conditions (not the entire chain) are satisfied.

        Used during construction to avoid double-evaluation of conditions.
        """
        # If source is inactive, we're inactive
        if (
            isinstance(self._source_observable, Conditional)
            and not self._source_observable.is_active
        ):
            return False

        # If no conditions, always active
        if not self._processed_conditions:
            return True

        # Evaluate only local conditions
        source_value = (
            self._source_observable.value
            if isinstance(self._source_observable, Conditional)
            else self._source_observable.value
        )
        return self._evaluate_all_conditions(source_value, self._processed_conditions)

    def _collect_all_conditions(self) -> List[Any]:
        """Collect all conditions from the entire conditional chain."""
        conditions = []

        # Add conditions from the current level
        conditions.extend(self._processed_conditions)

        # Add conditions from source conditionals recursively
        # Only collect if the source hasn't already been evaluated (avoid duplicate evaluations)
        if isinstance(self._source_observable, Conditional):
            conditions.extend(self._source_observable._collect_all_conditions())

        return conditions

    def _get_root_source_value(self) -> T:
        """Get the value from the root source observable without triggering condition evaluation."""
        current_source = self._source_observable
        while isinstance(current_source, Conditional):
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
        # Avoid accessing private attributes of other objects; fall back to None when inactive
        if isinstance(self._source_observable, Conditional):
            return self._source_observable.value if self._source_observable.is_active else None  # type: ignore
        return self._source_observable.value

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
        if isinstance(condition, Conditional):
            # Compound/public conditional interface - use its public state
            return condition.is_active
        if isinstance(condition, Observable):
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
        if isinstance(self._source_observable, Conditional):
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

        Safely handles None dependencies and missing add_observer methods.
        """

        def handle_value_change():
            """Handle changes to source or condition values."""
            self._update_conditional_state()

        # Subscribe to all dependencies using public interface
        for dependency in self._all_dependencies:
            dependency.add_observer(handle_value_change)

    def _update_conditional_state(self) -> None:
        """
        Update the conditional state when dependencies change.

        This method is called whenever any dependency changes value.
        """
        previous_conditions_satisfied = self._conditions_met
        previous_value = self._value

        # Check current condition state
        self._conditions_met = self._check_if_conditions_are_satisfied()

        # Only update value and notify if conditions are satisfied AND value changes
        if self._conditions_met:
            # For nested conditionals, source might be inactive - get root source value
            current_source_value = self._get_root_source_value()
            if self._value != current_source_value:
                self._value = current_source_value
                self._has_ever_had_valid_value = True
                # Schedule notification to respect global topological ordering
                from .base import Observable as _Obs

                _Obs._pending_notifications.add(self)
            elif not self._has_ever_had_valid_value:
                # Conditions just became met for the first time - notify even if value didn't change
                self._value = current_source_value
                self._has_ever_had_valid_value = True
                from .base import Observable as _Obs

                _Obs._pending_notifications.add(self)
        else:
            # Conditions are not met - update internal state but don't notify
            # This handles the case where we're notified by source but conditions are unmet
            if self._has_ever_had_valid_value:
                # We had a valid value before, so we're transitioning from active to inactive
                # Don't notify observers - this maintains pullback semantics
                pass

    def _notify_observers(self) -> None:
        """Notify observers only when conditions are satisfied."""
        if self._conditions_met:
            super()._notify_observers()

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
            This is like asking "What's the current value at the gate?"
            - If gate is open: Returns the value that passed through
            - If gate is closed: Raises an exception (no value to return)

        Mathematical Interpretation:
            This property implements the projection from the gate:
            ```
            Source → [Condition 1] → [Condition 2] → Gate → Filtered Output
            ```

            When active: Returns s where s ∈ {x ∈ source | ∀c ∈ conditions: c(x) = True}
            When inactive: Gate is closed, no value available (raises exception)

        Example:
            ```python
            data = observable(5)
            filtered = data & (lambda x: x > 10)  # Gate checks: "Is value > 10?"

            try:
                value = filtered.value  # Raises ConditionalNeverMet (gate never opened)
            except ConditionalNeverMet:
                print("Gate has never been open")

            data.set(15)
            value = filtered.value  # Returns 15 (gate is open, value passed through)
            ```
        """
        if self.is_active:
            # Conditions are satisfied - return the current value
            return self._value
        elif self._has_ever_had_valid_value:
            # Conditions were previously satisfied but are not now
            raise ConditionalNotMet("Conditions are not currently satisfied")
        else:
            # Conditions have never been satisfied
            raise ConditionalNeverMet("Conditions have never been satisfied")

    @property
    def is_active(self) -> bool:
        """
        True if conditions are currently satisfied (gate is open).

        This property indicates whether the conditional observable is currently
        in an active state where it can emit values. Think of it as checking
        if the security gate is currently open.

        In mathematical terms, this represents whether the gate is non-empty:
        ```
        is_active = ∃s ∈ source: ∀c ∈ conditions: c(s) = True
        ```

        **Gate States:**

        When active (gate open):
        - ✅ Can emit values through notifications
        - ✅ Allows safe access to `.value` property
        - ✅ Represents a non-empty filtered subset
        - 🚪 Gate is open, values flow through

        When inactive (gate closed):
        - ❌ Does not emit notifications
        - ❌ Raises exceptions when accessing `.value`
        - ❌ Represents an empty filtered subset
        - 🚪 Gate is closed, no values pass

        Example:
            ```python
            data = observable(5)
            filtered = data & (lambda x: x > 10)  # Gate checks: "Is value > 10?"

            print(filtered.is_active)  # False (5 <= 10, gate is closed)

            data.set(15)
            print(filtered.is_active)  # True (15 > 10, gate is open)
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
            "current_value": self._value,
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
