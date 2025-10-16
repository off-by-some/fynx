"""
Fynx ConditionalObservable - Conditional reactive computations.

This module provides the ConditionalObservable class for filtering reactive
streams based on boolean conditions.
"""

from typing import TypeVar

from .base import Observable

T = TypeVar("T")


class ConditionalObservable(Observable[T]):
    """
    An observable that only emits values from a source observable when all conditions are met.

    This allows filtering reactive streams based on boolean conditions, creating
    conditional reactive computations.
    """

    def __init__(
        self, source_observable: Observable[T], *condition_observables: Observable[bool]
    ) -> None:
        """
        Create a conditional observable.

        Args:
            source_observable: The observable whose values to conditionally emit
            *condition_observables: Boolean observables that must all be True for emission
        """
        # Call parent constructor with initial value (only if all conditions are met)
        initial_conditions_met = all(cond.value for cond in condition_observables)
        initial_value = source_observable.value if initial_conditions_met else None

        super().__init__("conditional", initial_value)
        self._source_observable = source_observable
        self._condition_observables = list(condition_observables)
        self._conditions_met = initial_conditions_met

        # Set up observers
        def update_from_source():
            """Called when source observable changes."""
            # If conditions are currently met, emit the new source value
            if self._conditions_met:
                new_value = self._source_observable.value
                self.set(new_value)

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
                self.set(current_value)
            # If conditions became unmet, update internal state but don't notify
            elif not self._conditions_met and old_conditions_met:
                self._value = None  # Update internal value without notifying

        # Subscribe to all observables
        source_observable.add_observer(update_from_source)
        for cond_obs in condition_observables:
            cond_obs.add_observer(update_from_conditions)

    def __and__(self, condition: Observable[bool]) -> "ConditionalObservable[T]":
        """
        Add another condition to this conditional observable.

        Args:
            condition: Additional boolean Observable condition

        Returns:
            A new ConditionalObservable with the additional condition
        """
        return ConditionalObservable(
            self._source_observable, *self._condition_observables, condition
        )
