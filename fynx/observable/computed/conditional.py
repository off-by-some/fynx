"""
Simplified ConditionalObservable - Cleaner with Base Class Support
================================================================

Key improvements:
1. Uses new DerivedValue base class with template method pattern
2. Much cleaner conditional logic
3. Consistent with Liskov Substitution Principle
4. All complexity handled by base class
"""

from typing import Any, Callable, List, Optional, TypeVar, Union

from fynx.observable.core.abstract.derived import ComputationState, DerivedValue
from fynx.observable.core.abstract.operations import OperatorMixin
from fynx.observable.types.protocols.conditional_protocol import Conditional
from fynx.observable.types.protocols.observable_protocol import Observable

T = TypeVar("T")
Condition = Union[Observable[bool], Callable[[T], bool], "ConditionalObservable"]


class ConditionalNeverMet(Exception):
    """
    Raised when attempting to access the value of a ConditionalObservable
    whose conditions have never been satisfied.
    """


class ConditionalNotMet(Exception):
    """
    Raised when attempting to access the value of a ConditionalObservable
    whose conditions are not currently satisfied.
    """


class ConditionalObservable(DerivedValue[T], Conditional[T], OperatorMixin):
    """
    Conditional observable - cleaner with base class support.

    Only emits values when all conditions are satisfied.
    Uses the new DerivedValue base class for all state management.
    """

    def __init__(self, source: "Observable[T]", *conditions):
        if source is None:
            raise ValueError("source_observable cannot be None")

        # Validate conditions
        for i, condition in enumerate(conditions):
            if condition is None:
                raise ValueError(f"Condition {i} cannot be None")
            # Check if condition is valid type
            if not (
                callable(condition)
                or hasattr(condition, "value")
                or hasattr(condition, "subscribe")
            ):
                raise TypeError(
                    f"Condition {i} must be an Observable, ObservableValue, callable, or ConditionalObservable"
                )

        self._conditions = conditions
        self._conditions_met = False
        self._has_ever_been_active = False  # Track if conditions were ever met
        # Add _processed_conditions for compatibility with operations.py
        self._processed_conditions = conditions
        super().__init__("conditional", None, source)

        # Subscribe to condition observables
        for condition in conditions:
            if hasattr(condition, "subscribe"):
                condition.subscribe(self._on_source_change)

        # Evaluate initial state
        self._conditions_met = self._evaluate_conditions()
        if self._conditions_met:
            self._has_ever_been_active = True

    def _compute_value(self) -> T:
        """Pass through source value."""
        return self._source_observable.value

    def _should_compute(self) -> bool:
        """Only compute when conditions are met."""
        self._conditions_met = self._evaluate_conditions()
        if self._conditions_met:
            self._has_ever_been_active = True
        return self._conditions_met and self._state == ComputationState.DIRTY

    def _should_notify(self, new_value: T) -> bool:
        """Only notify when active and value changed."""
        # Don't notify if conditions are not met
        if not self._conditions_met:
            return False

        if self._conditions_met:
            self._has_ever_been_active = True
        return super()._should_notify(new_value)

    def _can_access_value(self) -> bool:
        """Only allow access when conditions are met."""
        # Always evaluate conditions when checking access
        self._conditions_met = self._evaluate_conditions()
        if self._conditions_met:
            self._has_ever_been_active = True
        return self._conditions_met

    def _get_fallback_value(self) -> T:
        """Raise appropriate exception based on whether conditions were ever met."""
        if not self._has_ever_been_active:
            raise ConditionalNeverMet("Conditions have never been satisfied")
        else:
            raise ConditionalNotMet("Conditions not satisfied")

    def _evaluate_conditions(self) -> bool:
        """Evaluate all conditions."""
        try:
            source_value = self._source_observable.value
        except (ConditionalNeverMet, ConditionalNotMet):
            # If source is inactive, conditions cannot be met
            return False

        for condition in self._conditions:
            if callable(condition):
                if not condition(source_value):
                    return False
            elif hasattr(condition, "value"):
                if not bool(condition.value):
                    return False
        return True

    @property
    def is_active(self) -> bool:
        """Check if conditions are currently met."""
        # Always evaluate conditions when checking active state
        self._conditions_met = self._evaluate_conditions()
        return self._conditions_met
