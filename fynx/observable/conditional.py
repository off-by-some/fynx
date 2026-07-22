r"""
Conditional observables for filtering and conditional logic.

This module provides ConditionalObservable, which only emits values when a
set of conditions are all satisfied. Use it for filtering data streams,
conditional logic, and reactive pipelines that should only react in certain
states.

## The gate analogy

A ConditionalObservable is a gate: the source value passes through to
observers only when every condition evaluates to True against it. If any
condition is False, the gate stays closed and nothing is emitted.

Formally: ConditionalObservable(source, c1, c2, ..., cn) represents the set $\{s \in \text{source} \mid c_1(s) \land c_2(s) \land \ldots \land c_n(s)\}$ - all values from source where every condition evaluates to True. With no conditions, the gate is always open and the filtered set equals the source.

## How conditions chain

When you write `data @ condition1 @ condition2`, we build a chain:
`((data @ condition1) @ condition2)`. Each `@` adds another guard. The final
gate opens only when condition1(value) is True and condition2(value) is True
and so on.

The `@` operator is asymmetric: the left side supplies the value and the right
side supplies a guard. `data @ is_ready` exposes `data` while `is_ready` is
true. Use `&` for ordinary boolean AND when you want a reusable condition:
`ready = authenticated & connected & ~loading`.

## Practical Usage

Example:
    ```python
    from fynx import observable

    # Create a conditional observable that only emits when value > 10
    data = observable(5)
    filtered = data @ (lambda x: x > 10)

    # The filtered observable will only emit when data > 10
    data.set(15)  # filtered will emit 15
    data.set(8)   # filtered will not emit
    ```

## Key Properties

- **Pullback Semantics**: Only notifies when conditions are satisfied AND value changes
- **Asymmetric Gate**: `source @ condition` preserves the source value type
- **Guard-Order Commutative**: `data @ c1 @ c2` ≡ `data @ c2 @ c1` for pure guards over the same source
- **Guard-Chain Associative**: `(data @ c1) @ c2` ≡ `data @ (c1 & c2)`
"""

from __future__ import annotations

from typing import Any, Callable, Generic, List, Optional, Set, TypeVar, cast

from ..types import ConditionOperand
from . import base as _base
from .base import Observable as BaseObservable
from .computed import ComputedObservable
from .interfaces import Conditional
from .operands import unwrap_condition
from .operators import OperatorMixin

T = TypeVar("T")
U = TypeVar("U")
Observable = BaseObservable


class ConditionalNeverMet(Exception):
    """
    Raised when accessing `.value` on a ConditionalObservable whose
    conditions have never been satisfied - the gate has never opened, so no
    value has passed through. Check `is_active` first to avoid this.

    Example:
        ```python
        filtered = data @ (lambda x: x > 0)
        if filtered.is_active:
            value = filtered.value  # Safe to access
        else:
            # Handle case where conditions are not met
            pass
        ```
    """


class ConditionalNotMet(Exception):
    """
    Raised when accessing `.value` on a ConditionalObservable whose
    conditions were previously satisfied but aren't now - the gate was open
    and has since closed. Check `is_active` first to avoid this.

    Example:
        ```python
        filtered = data @ (lambda x: x > 0)
        try:
            value = filtered.value
        except ConditionalNotMet:
            # Handle case where conditions are not currently met
            # Could fall back to a default value or cached value if needed
            pass
        ```
    """


class ConditionalObservable(ComputedObservable[T], Conditional[T], OperatorMixin[T]):
    r"""
    A conditional observable that filters values based on one or more conditions.

    Source data flows through a chain of condition checks; the value only
    reaches observers when every condition returns True.

    ```
    Source Data → [Condition 1] → [Condition 2] → [Condition 3] → Filtered Output
                     ↓              ↓              ↓
                   Check A        Check B        Check C
    ```

    Formally, a ConditionalObservable represents the filtered subset
    $\{s \in \text{source} \mid c_1(s) \land c_2(s) \land \ldots \land c_n(s)\}$
    - all values from source where every condition evaluates to True.

    ## Key Properties

    - **Gate Behavior**: Only notifies when conditions are satisfied AND value changes
    - **Asymmetric Gate**: `source @ condition` preserves the source value type
    - **Guard-Order Commutative**: `data @ c1 @ c2` ≡ `data @ c2 @ c1` for pure guards over the same source
    - **Guard-Chain Associative**: `(data @ c1) @ c2` ≡ `data @ (c1 & c2)`

    ## Behavior States

    The gate has two states: open and closed.

    When active (gate open):
    - All conditions are satisfied
    - Values flow through to observers
    - Accessing `.value` returns the current filtered value
    - Notifications emit when the value changes

    When inactive (gate closed):
    - At least one condition fails
    - No values pass through
    - Accessing `.value` raises ConditionalNotMet or ConditionalNeverMet
    - No notifications are emitted

    ## Empty Conditions

    When no conditions are provided, the gate is always open:
    ```python
    always_open = ConditionalObservable(data)  # Equivalent to just `data`
    ```

    Useful as a pass-through gate in tests and internal composition.

    ## Examples

    Basic filtering (single guard):
        ```python
        from fynx import observable

        data = observable(42)
        filtered = data @ (lambda x: x > 10)  # Only values > 10 get through

        data.set(15)  # filtered emits 15 (passes the check)
        data.set(5)   # filtered becomes inactive (fails the check)
        ```

    Multiple conditions (multiple guards):
        ```python
        is_positive = observable(True)
        is_even = lambda x: x % 2 == 0

        filtered = data @ (is_positive & is_even)
        # Only emits when data > 0 AND data is even
        ```

    Chaining conditionals:
        ```python
        step1 = data @ (lambda x: x > 0)      # First gate
        step2 = step1 @ (lambda x: x < 100)   # Second gate
        # Equivalent to: data @ (lambda x: x > 0) @ (lambda x: x < 100)
        ```
    """

    def __init__(
        self, source_observable: "Observable[T]", *conditions: ConditionOperand[T]
    ) -> None:
        r"""
        Create a conditional observable that filters values based on conditions.

        Formally: ConditionalObservable(source, c1, c2, ..., cn) represents
        $\{s \in \text{source} \mid c_1(s) \land c_2(s) \land \ldots \land c_n(s)\}$.
        With no conditions, the gate is always open: $P = \text{source}$.

        Args:
            source_observable: The observable whose values will be conditionally emitted.
                              This is the source in the gate analogy.
            *conditions: Variable number of conditions that form the filtering criteria.
                        Each condition can be:
                        - Observable[bool]: A boolean observable (external condition)
                        - Callable: A predicate function that takes the source value and returns bool
                        - ConditionalObservable: A compound condition (nested gate)

        Raises:
            ValueError: If source_observable is None
            TypeError: If conditions contain invalid types

        Examples:
            ```python
            from fynx import observable

            # Single predicate condition (one guard)
            data = observable(42)
            positive = data @ (lambda x: x > 0)

            # Multiple conditions (multiple guards)
            filtered = data @ (lambda x: x > 0) @ (lambda x: x < 100)

            # Mixed condition types
            is_ready = observable(True)
            valid_data = data @ is_ready @ (lambda x: x % 2 == 0)

            # Nested conditionals
            step1 = data @ (lambda x: x > 0)
            step2 = step1 @ (lambda x: x < 100)

            # Empty conditions (always open gate)
            always_open = ConditionalObservable(data)  # Just passes through
            ```
        """
        # Validate inputs
        self._validate_inputs(source_observable, conditions)

        # Store the original source and conditions for reference
        self._source_observable: BaseObservable[T] = source_observable
        self._conditions: tuple[ConditionOperand[T], ...] = conditions
        self._dynamic_condition_dependencies: Set[BaseObservable[Any]] = set()
        self._observed_dependencies: Set[BaseObservable[Any]] = set()
        self._dependency_observer: Optional[Callable[[], Any]] = None

        # Process conditions and create optimized observables
        self._processed_conditions: List[Any] = self._process_conditions(
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
        self._all_dependencies: Set[BaseObservable[Any]] = self._find_all_dependencies()
        self._setup_observers()

    def _validate_inputs(
        self, source_observable: BaseObservable[T], conditions: tuple[Any, ...]
    ) -> None:
        """
        Validate the inputs to the constructor.

        Raises appropriate exceptions for invalid inputs.
        """
        if source_observable is None:
            raise ValueError("source_observable cannot be None")

        # Allow empty conditions to represent an always-active conditional.
        # if not conditions:
        #     raise ValueError("At least one condition must be provided")

        # Validate each condition
        from .descriptors import ObservableValue

        for i, condition in enumerate(conditions):
            if condition is None:
                raise ValueError(f"Condition {i} cannot be None")

            # Check if condition is a valid type
            is_observable = isinstance(condition, BaseObservable)
            is_observable_value = isinstance(condition, ObservableValue)
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
        self, source: BaseObservable[T], conditions: tuple[Any, ...]
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
            processed.append(unwrap_condition(condition))
        return processed

    def _check_if_conditions_are_satisfied(self) -> bool:
        """
        Check if all conditions are currently satisfied.

        Returns False if the source is inactive or any condition fails.
        If no conditions are provided, always returns True (always active).
        """
        # If no conditions, always active.
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
            isinstance(self._source_observable, ConditionalObservable)
            and not self._source_observable.is_active
        ):
            return False

        # If no conditions, always active
        if not self._processed_conditions:
            return True

        # Evaluate only local conditions
        source_value = cast(
            T,
            (
                self._source_observable.value
                if isinstance(self._source_observable, ConditionalObservable)
                else self._source_observable.value
            ),
        )
        return self._evaluate_all_conditions(source_value, self._processed_conditions)

    def _collect_all_conditions(self) -> List[Any]:
        """Collect all conditions from the entire conditional chain."""
        conditions = []

        # Add conditions from the current level
        conditions.extend(self._processed_conditions)

        # Add conditions from source conditionals recursively
        # Only collect if the source hasn't already been evaluated (avoid duplicate evaluations)
        if isinstance(self._source_observable, ConditionalObservable):
            conditions.extend(self._source_observable._collect_all_conditions())

        return conditions

    def _get_root_source_value(self) -> T:
        """Get the value from the root source observable without triggering condition evaluation."""
        current_source = self._source_observable
        while isinstance(current_source, ConditionalObservable):
            current_source = current_source._source_observable
        # For regular observables, just get the value
        # For conditionals, we've already navigated to the root
        return cast(T, current_source.value)

    def _is_source_inactive(self) -> bool:
        """Check if the source observable is inactive (for conditional sources)."""
        return (
            isinstance(self._source_observable, ConditionalObservable)
            and not self._source_observable.is_active
        )

    def _get_initial_value(self) -> Optional[T]:
        """Get the initial value for the conditional observable."""
        # Avoid accessing private attributes of other objects; fall back to None when inactive
        if isinstance(self._source_observable, ConditionalObservable):
            return self._source_observable.value if self._source_observable.is_active else None  # type: ignore
        return self._source_observable.value

    def _evaluate_all_conditions(
        self, source_value: T, conditions: Optional[List[Any]] = None
    ) -> bool:
        """
        Evaluate all conditions against the source value.

        Returns True only if all conditions are satisfied.
        """
        if conditions is None:
            conditions = self._processed_conditions

        captured_dependencies: Set[BaseObservable[Any]] = set()
        try:
            for condition in conditions:
                if callable(condition) and not isinstance(
                    condition, (BaseObservable, ConditionalObservable)
                ):
                    BaseObservable._dependency_capture_stack.append(
                        captured_dependencies
                    )
                    try:
                        condition_satisfied = self._evaluate_single_condition(
                            condition, source_value
                        )
                    finally:
                        BaseObservable._dependency_capture_stack.pop()
                else:
                    condition_satisfied = self._evaluate_single_condition(
                        condition, source_value
                    )

                if not condition_satisfied:
                    return False
            return True
        finally:
            captured_dependencies.difference_update(self._find_static_dependencies())
            captured_dependencies.discard(self)
            self._dynamic_condition_dependencies = captured_dependencies

    def _evaluate_single_condition(self, condition: Any, source_value: T) -> bool:
        """
        Evaluate a single condition against the source value.

        Handles different types of conditions appropriately.
        """
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

    def _find_all_dependencies(self) -> Set[BaseObservable[Any]]:
        """
        Find all observable dependencies for this conditional.

        Includes the source observable and all condition observables.
        For nested conditionals, recursively finds dependencies.
        """
        dependencies = self._find_static_dependencies()
        dependencies.update(getattr(self, "_dynamic_condition_dependencies", set()))

        # Filter out None values
        return {dep for dep in dependencies if dep is not None}

    def _find_static_dependencies(self) -> Set[BaseObservable[Any]]:
        """Find dependencies declared directly by source and observable conditions."""
        dependencies: Set[BaseObservable[Any]] = set()

        # Only add the source observable if it's not a conditional
        # For conditionals, we depend on their dependencies instead
        if isinstance(self._source_observable, ConditionalObservable):
            # For conditional sources, depend on their dependencies
            dependencies.update(self._source_observable._all_dependencies)
        else:
            # For non-conditional sources, depend on them directly
            dependencies.add(self._source_observable)

        # Add dependencies from each condition
        for condition in self._processed_conditions:
            if isinstance(condition, BaseObservable):
                dependencies.add(condition)
            elif isinstance(condition, ConditionalObservable):
                # For nested conditionals, add their dependencies
                dependencies.update(condition._all_dependencies)

        return dependencies

    def _extract_condition_dependencies(
        self, condition: Any
    ) -> Set[BaseObservable[Any]]:
        # Deprecated: dependencies are gathered via public Observable interface only
        return set()

    def _setup_observers(self) -> None:
        """
        Set up observers for all dependencies.

        Safely handles None dependencies and missing add_observer methods.
        """

        def handle_value_change():
            """Handle changes to source or condition values."""
            self._is_dirty = True
            BaseObservable._schedule_notification(self)

        self._dependency_observer = handle_value_change
        self._sync_dependency_observers()

    def _sync_dependency_observers(self) -> None:
        """Keep subscriptions aligned with static and captured predicate inputs."""
        if self._dependency_observer is None:
            return

        target_dependencies = self._find_all_dependencies()

        for dependency in self._observed_dependencies - target_dependencies:
            dependency.remove_observer(self._dependency_observer)

        for dependency in target_dependencies - self._observed_dependencies:
            dependency.add_observer(self._dependency_observer)

        self._observed_dependencies = target_dependencies
        self._all_dependencies = target_dependencies

    def _update_conditional_state(self, schedule_notifications: bool = True) -> bool:
        """
        Update the conditional state when dependencies change.

        This method is called whenever any dependency changes value.
        """
        previous_conditions_satisfied = self._conditions_met
        previous_value = self._value

        # Check current condition state
        self._conditions_met = self._check_if_conditions_are_satisfied()
        self._sync_dependency_observers()

        # Only update value and notify if conditions are satisfied AND value changes
        if self._conditions_met:
            # For nested conditionals, source might be inactive - get root source value
            current_source_value = self._get_root_source_value()
            if _base.value_changed(self._value, current_source_value):
                self._value = current_source_value
                self._version = getattr(self, "_version", 0) + 1
                self._has_ever_had_valid_value = True
                if schedule_notifications and self._observers:
                    BaseObservable._schedule_notification(self)
                return True
            elif not self._has_ever_had_valid_value:
                # Conditions just became met for the first time - notify even if value didn't change
                self._value = current_source_value
                self._version = getattr(self, "_version", 0) + 1
                self._has_ever_had_valid_value = True
                if schedule_notifications and self._observers:
                    BaseObservable._schedule_notification(self)
                return True
        else:
            # Conditions are not met - update internal state but don't notify
            # This handles the case where we're notified by source but conditions are unmet
            if self._has_ever_had_valid_value:
                # We had a valid value before, so we're transitioning from active to inactive
                # Don't notify observers - this maintains pullback semantics
                pass
        return False

    def _notify_observers(self) -> None:
        """Notify observers only when conditions are satisfied."""
        should_notify = False
        if self._is_dirty:
            self._is_dirty = False
            should_notify = self._update_conditional_state(schedule_notifications=False)

        if self._conditions_met and should_notify:
            super()._notify_observers()

    def then(self, func: Callable[[T], U]) -> "ConditionalObservable[U]":
        """
        Transform values that pass through this gate without forcing it open.

        Mapping over an inactive conditional produces another inactive
        conditional. The transform runs only when this source gate is active,
        so constructing `gate >> f` is safe even when `gate.value` would raise
        ConditionalNeverMet or ConditionalNotMet.
        """
        return MappedConditionalObservable(self, func)

    def __rshift__(self, func: Callable[[T], U]) -> "ConditionalObservable[U]":
        """Transform values that pass through this gate."""
        return self.then(func)

    @property
    def value(self) -> T:
        r"""
        Current value of the conditional observable.

        Returns the source observable's value when conditions are satisfied.
        Raises ConditionalNeverMet if conditions have never been satisfied, or
        ConditionalNotMet if they were previously satisfied but aren't now.

        Formally: when active, returns $s$ where $s \in \{x \in \text{source} \mid \forall c \in \text{conditions}: c(x) = \text{True}\}$.

        Example:
            ```python
            data = observable(5)
            filtered = data @ (lambda x: x > 10)  # Gate checks: "Is value > 10?"

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
            return cast(T, self._value)
        elif self._has_ever_had_valid_value:
            # Conditions were previously satisfied but are not now
            raise ConditionalNotMet("Conditions are not currently satisfied")
        else:
            # Conditions have never been satisfied
            raise ConditionalNeverMet("Conditions have never been satisfied")

    @property
    def is_active(self) -> bool:
        r"""
        True if conditions are currently satisfied (gate is open).

        Formally: $is\_active = \exists s \in \text{source}: \forall c \in \text{conditions}: c(s) = \text{True}$
        - the gate is open when there exists a source value satisfying every condition.

        When active (gate open):
        - Can emit values through notifications
        - Allows safe access to `.value` property
        - Represents a non-empty filtered subset
        - Gate is open, values flow through

        When inactive (gate closed):
        - Does not emit notifications
        - Raises exceptions when accessing `.value`
        - Represents an empty filtered subset
        - Gate is closed, no values pass

        Example:
            ```python
            data = observable(5)
            filtered = data @ (lambda x: x > 10)  # Gate checks: "Is value > 10?"

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

        debug_info: dict[str, Any] = {
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


class MappedConditionalObservable(ConditionalObservable[U], Generic[T, U]):
    """A conditional transform that preserves the source gate's active state."""

    def __init__(
        self,
        source_observable: ConditionalObservable[T],
        transform_func: Callable[[T], U],
    ) -> None:
        self._mapped_source_observable = source_observable
        self._source_observable = cast(BaseObservable[U], source_observable)
        self._conditions: tuple[ConditionOperand[U], ...] = ()
        self._processed_conditions: List[Any] = []
        self._dynamic_condition_dependencies: Set[BaseObservable[Any]] = set()
        self._observed_dependencies: Set[BaseObservable[Any]] = set()
        self._dependency_observer: Optional[Callable[[], Any]] = None
        self._conditional_transform_func: Callable[[T], U] = transform_func
        self._conditions_met = source_observable.is_active
        self._has_ever_had_valid_value = source_observable.is_active

        initial_value: Optional[U] = None
        if source_observable.is_active:
            initial_value = self._apply_conditional_transform(source_observable.value)

        ComputedObservable.__init__(
            self,
            "conditional",
            initial_value,
            None,
            source_observable,
        )

        self._all_dependencies: Set[BaseObservable[Any]] = {source_observable}
        self._setup_observers()

    def _apply_conditional_transform(self, value: T) -> U:
        transform_state = _base._TRANSFORM_EVALUATION_STATE
        previous_transform_state = transform_state[0]
        transform_state[0] = True
        try:
            return self._conditional_transform_func(value)
        finally:
            transform_state[0] = previous_transform_state

    def _setup_observers(self) -> None:
        def handle_value_change() -> None:
            self._is_dirty = True
            BaseObservable._schedule_notification(self)

        self._dependency_observer = handle_value_change
        self._sync_dependency_observers()

    def _sync_dependency_observers(self) -> None:
        if self._dependency_observer is None:
            return

        target_dependencies: Set[BaseObservable[Any]] = {self._mapped_source_observable}
        for dependency in self._observed_dependencies - target_dependencies:
            dependency.remove_observer(self._dependency_observer)

        for dependency in target_dependencies - self._observed_dependencies:
            dependency.add_observer(self._dependency_observer)

        self._observed_dependencies = target_dependencies
        self._all_dependencies = target_dependencies

    def _update_conditional_state(self, schedule_notifications: bool = True) -> bool:
        source_active = self._mapped_source_observable.is_active
        self._conditions_met = source_active

        if not source_active:
            return False

        new_value = self._apply_conditional_transform(
            self._mapped_source_observable.value
        )
        if (
            _base.value_changed(self._value, new_value)
            or not self._has_ever_had_valid_value
        ):
            self._value = new_value
            self._version += 1
            self._has_ever_had_valid_value = True
            if schedule_notifications and self._observers:
                BaseObservable._schedule_notification(self)
            return True

        return False

    @property
    def is_active(self) -> bool:
        """Mapped gates are active exactly when their source gate is active."""
        return self._mapped_source_observable.is_active
