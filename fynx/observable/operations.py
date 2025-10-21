"""
FynX Operations - Natural Language Reactive Operations
======================================================

This module provides natural language methods for reactive operations in FynX,
serving as the core implementation layer that operators.py delegates to.

The operations provide a fluent, readable API for reactive programming:

- `then(func)` - Transform values (equivalent to `>>` operator)
- `alongside(other)` - Merge observables (equivalent to `+` operator)
- `requiring(condition)` - Compose boolean conditions with AND (equivalent to `&` operator)
- `negate()` - Boolean negation (equivalent to `~` operator)
- `either(other)` - OR logic for boolean conditions
"""

from typing import TYPE_CHECKING, Callable, TypeVar, Union

if TYPE_CHECKING:
    from .base import Observable, ReactiveContext

T = TypeVar("T")
U = TypeVar("U")


# Lazy imports to avoid circular dependencies
def _MergedObservable():
    from .merged import MergedObservable

    return MergedObservable


def _ConditionalObservable():
    from .conditional import ConditionalObservable

    return ConditionalObservable


def _ComputedObservable():
    from .computed import ComputedObservable

    return ComputedObservable


def _OptimizationContext():
    from ..optimizer import OptimizationContext

    return OptimizationContext


class OperationsMixin:
    """
    Mixin providing natural language reactive operations.

    This mixin provides the core reactive operations that can be used by any
    observable class. It serves as the foundation for both the operator syntax
    (in operators.py) and direct method calls.
    """

    def _create_computed(self, func: Callable, observable) -> "Observable":
        """
        Create a computed observable that derives its value from other observables.

        This implements the core computed logic that was previously in computed.py,
        but now integrated directly into the operations mixin for cleaner architecture.
        """
        MergedObservable = _MergedObservable()
        ComputedObservable = _ComputedObservable()
        OptimizationContext = _OptimizationContext()

        if isinstance(observable, MergedObservable):
            # For merged observables, apply func to the tuple values
            merged_computed_obs: "Observable" = ComputedObservable(
                None, None, func, observable
            )

            def update_merged_computed():
                values = tuple(obs.value for obs in observable._source_observables)
                result = func(*values)
                merged_computed_obs._set_computed_value(result)

            # Initial computation
            update_merged_computed()

            # Subscribe to changes in the source observable
            observable.subscribe(lambda *args: update_merged_computed())

            # Register with current optimization context for automatic optimization
            context = OptimizationContext.current()
            if context is not None:
                context.register_observable(merged_computed_obs)

            return merged_computed_obs
        else:
            # For single observables
            single_computed_obs: "Observable" = ComputedObservable(
                None, None, func, observable
            )

            def update_single_computed():
                result = func(observable.value)
                single_computed_obs._set_computed_value(result)

            # Initial computation
            update_single_computed()

            # Subscribe to changes
            observable.subscribe(lambda val: update_single_computed())

            # Register with current optimization context for automatic optimization
            context = OptimizationContext.current()
            if context is not None:
                context.register_observable(single_computed_obs)

            return single_computed_obs

    def then(self, func: Callable[[T], U]) -> "Observable[U]":
        """
        Transform this observable's value using a function.

        This creates a computed observable that applies the given function to
        this observable's value, updating automatically when the source changes.

        Args:
            func: A function to apply to the observable's value

        Returns:
            A new computed observable with the transformed value

        Example:
            ```python
            doubled = counter.then(lambda x: x * 2)
            uppercase = name.then(lambda s: s.upper())
            ```
        """
        return self._create_computed(func, self)

    def alongside(self, other: "Observable") -> "Observable":
        """
        Merge this observable with another into a tuple.

        This creates a merged observable that combines the values of both
        observables into a tuple, updating when either source changes.

        Args:
            other: Another observable to merge with

        Returns:
            A merged observable containing both values as a tuple

        Example:
            ```python
            coordinates = x.alongside(y)  # Creates (x_value, y_value)
            point3d = x.alongside(y).alongside(z)  # (x, y, z)
            ```
        """
        MergedObservable = _MergedObservable()

        if hasattr(other, "_source_observables"):
            # If other is already merged, combine with its sources
            return MergedObservable(self, *other._source_observables)  # type: ignore
        else:
            # Standard case: combine two observables
            return MergedObservable(self, other)  # type: ignore

    def requiring(self, *conditions) -> "Observable":
        """
        Compose this observable with conditions using AND logic.

        This creates a ConditionalObservable that combines this observable with
        additional conditions. Supports the same condition types as the & operator.

        Args:
            *conditions: Variable number of conditions (observables, callables, etc.)

        Returns:
            A ConditionalObservable representing the AND of all conditions

        Example:
            ```python
            # Compose multiple conditions
            result = data.requiring(lambda x: x > 0, is_ready, other_condition)
            ```
        """
        from .conditional import ConditionalObservable

        # If this is already a ConditionalObservable, create nested conditional
        if isinstance(self, ConditionalObservable):
            # Create a new conditional with this conditional as source and new conditions
            return ConditionalObservable(self, *conditions)  # type: ignore
        else:
            return ConditionalObservable(self, *conditions)  # type: ignore

    def negate(self) -> "Observable[bool]":
        """
        Create a negated boolean version of this observable.

        This creates a computed observable that returns the logical negation
        of the current boolean value.

        Returns:
            A computed observable with negated boolean values

        Example:
            ```python
            is_disabled = is_enabled.negate()
            is_not_ready = is_ready.negate()

            # Use in conditions:
            data_when_disabled = data.when(is_enabled.negate())
            ```
        """
        return self.then(lambda x: not x)  # type: ignore

    def either(self, other: "Observable") -> "Observable":
        """
        Create an OR condition between this observable and another.

        This creates a conditional observable that only emits when the OR result is truthy.
        If the initial OR result is falsy, raises ConditionNeverMet.

        Args:
            other: Another boolean observable to OR with

        Returns:
            A conditional observable that only emits when OR is truthy

        Raises:
            RuntimeError: If initial OR result is falsy (ConditionNeverMet)

        Example:
            ```python
            # Must start with at least one being True
            needs_attention = is_error.either(is_warning)  # If initially both False, raises error
            can_proceed = has_permission.either(is_admin)
            ```
        """
        # Create a computed observable for the OR result
        or_result = self.alongside(other).then(lambda a, b: a or b)

        # Return conditional observable that filters based on truthiness
        # Use a callable condition to avoid timing issues with computed observables
        return or_result & (lambda x: bool(x))
