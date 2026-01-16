"""
FynX Operations - Natural Language Reactive Operations
======================================================

This module provides natural language methods for transforming, merging, and filtering
reactive values. These operations read like English sentences while maintaining the
precision of reactive programming—operations like `then()`, `alongside()`, and `requiring()`
create new observables that update automatically when dependencies change.

The operations serve as the core implementation layer that operators.py delegates to.
Each method creates new observables that derive from their sources, updating automatically
when dependencies change. That automatic propagation eliminates manual synchronization—you
declare relationships, and the framework maintains them.

We provide five core operations:

- `then(func)` transforms values through functions (equivalent to `>>` operator)
- `alongside(other)` merges observables into tuples (equivalent to `+` operator)
- `requiring(*conditions)` composes boolean conditions with AND logic (equivalent to `&` operator)
- `negate()` inverts boolean values (equivalent to `~` operator)
- `either(other)` creates OR conditions between boolean observables

Result: a fluent API that reads like natural language while maintaining the precision of reactive programming. The methods compose naturally—chain transformations, nest conditions, build complex reactive pipelines from simple operations.
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

    Reactive operations extend the standard function pattern: instead of running once and
    returning, they create new observables that update automatically when sources change.
    Pass an observable, apply a transformation, and get a new observable that maintains
    the relationship reactively.

    This mixin provides the core reactive operations that any observable class can use. It serves as the foundation for both operator syntax (in operators.py) and direct method calls. The methods create computed observables internally—each operation builds a new reactive value that derives from its sources.

    The mixin integrates with FynX's optimization system. When operations create computed observables, they register with the current OptimizationContext for automatic dependency tracking and performance optimization. That registration happens transparently—you call the method, and the framework handles the reactive infrastructure.
    """

    def _create_computed(self, func: Callable, observable) -> "Observable":
        """
        Create a computed observable that derives its value from other observables.

        This method creates reactive computations that run automatically when inputs change,
        producing new computed values. The function receives observable values as arguments
        and returns a result that becomes the computed observable's value.

        The method handles two cases: single observables and merged observables. For single observables, it applies the function to the observable's value directly. For merged observables, it unpacks the tuple and passes values as separate arguments. That distinction enables functions that work with both single values and coordinate pairs.

        The computed observable subscribes to its source and updates automatically when dependencies change. It also registers with the current OptimizationContext for automatic optimization. That registration enables the framework to track dependencies, detect optimization opportunities, and manage reactive graph structure.
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

        This method creates a reactive transformation pipeline. The function runs automatically
        whenever the source value changes, producing a new computed observable with the
        transformed result. The transformation maintains the reactive relationship—changes
        to the source propagate through the function to the computed value.

        The method creates a ComputedObservable that applies the given function to this observable's value. When the source changes, the computed observable recalculates automatically. That automatic recalculation eliminates manual updates—you declare the transformation, and the framework maintains it.

        Args:
            func: A function to apply to the observable's value. For merged observables,
                  the function receives unpacked tuple values as separate arguments.

        Returns:
            A new computed observable with the transformed value. The computed observable
            updates automatically when this observable changes.

        Example:
            ```python
            from fynx import observable

            counter = observable(5)
            doubled = counter.then(lambda x: x * 2)
            print(doubled.value)  # 10

            counter.set(7)
            print(doubled.value)  # 14 (automatically recalculated)

            name = observable("hello")
            uppercase = name.then(lambda s: s.upper())
            print(uppercase.value)  # "HELLO"
            ```
        """
        return self._create_computed(func, self)

    def alongside(self, other: "Observable") -> "Observable":
        """
        Merge this observable with another into a tuple.

        This method combines two observables into a single reactive tuple that updates when
        either component changes. The merged observable treats both values as a single atomic
        unit, enabling functions that need multiple related parameters to receive them
        together in a coordinated update.

        The method creates a MergedObservable that combines the values of both observables into a tuple. When either source changes, the merged observable recalculates automatically. That automatic coordination eliminates manual synchronization—you declare the relationship, and the framework maintains it.

        If the other observable is already merged, the method combines with its sources directly, creating a flat merged observable. If this observable is already merged, chaining creates nested tuples—the merged observable becomes one component of a new tuple.

        Args:
            other: Another observable to merge with. If other is a MergedObservable,
                   its source observables are combined directly.

        Returns:
            A merged observable containing both values as a tuple. The tuple updates
            automatically when either source changes.

        Example:
            ```python
            from fynx import observable

            x = observable(10)
            y = observable(20)
            coordinates = x.alongside(y)
            print(coordinates.value)  # (10, 20)

            x.set(15)
            print(coordinates.value)  # (15, 20)

            z = observable(30)
            point3d = x.alongside(y).alongside(z)
            print(point3d.value)  # ((10, 20), 30) - nested tuple
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

        This method creates a conditional observable that only emits values when every
        condition evaluates to True. Combine this observable with multiple conditions,
        and the result represents the logical AND of all conditions—the gate opens only
        when all conditions are satisfied.

        The method creates a ConditionalObservable that combines this observable with additional conditions. Each condition can be a boolean observable, a callable that takes the source value and returns a boolean, or another ConditionalObservable. The result represents the logical AND of all conditions—the gate opens only when every condition approves.

        If this observable is already a ConditionalObservable, the method creates a nested conditional. That nesting enables building complex condition chains incrementally—each call adds another guard to the checkpoint.

        Args:
            *conditions: Variable number of conditions. Each condition can be:
                - A boolean Observable
                - A callable that takes the source value and returns a boolean
                - Another ConditionalObservable

        Returns:
            A ConditionalObservable representing the AND of all conditions. The observable
            only emits values when every condition evaluates to True.

        Example:
            ```python
            from fynx import observable

            data = observable(5)
            is_ready = observable(True)
            other_condition = observable(True)

            # Compose multiple conditions
            result = data.requiring(lambda x: x > 0, is_ready, other_condition)
            print(result.value)  # 5 (all conditions met)

            is_ready.set(False)
            # Accessing result.value now raises ConditionalNotMet
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

        This method applies logical negation to a boolean observable, producing a new
        observable with inverted values. When the source changes, the negated observable
        updates automatically with the opposite boolean value. This enables expressing
        negative conditions directly without separate "is_not_X" observables.

        The method creates a computed observable that returns the logical negation of the current boolean value. When the source changes, the negated observable updates automatically. That automatic inversion enables expressing negative conditions directly—you don't need separate "is_not_X" observables.

        Returns:
            A computed observable with negated boolean values. The observable updates
            automatically when this observable changes, always returning the opposite
            boolean value.

        Example:
            ```python
            from fynx import observable

            is_enabled = observable(True)
            is_disabled = is_enabled.negate()
            print(is_disabled.value)  # False

            is_enabled.set(False)
            print(is_disabled.value)  # True (automatically updated)

            is_ready = observable(False)
            is_not_ready = is_ready.negate()
            print(is_not_ready.value)  # True
            ```
        """
        return self.then(lambda x: not x)  # type: ignore

    def either(self, other: "Observable") -> "Observable":
        """
        Create an OR condition between this observable and another.

        This method combines two boolean observables with OR logic, creating a conditional
        observable that only emits when at least one is True. The result represents the
        logical OR of both values—the system works if either source is active. If both
        are False, accessing the value raises ConditionalNeverMet.

        The method creates a computed observable that calculates the OR of both boolean values, then wraps it in a ConditionalObservable that filters based on truthiness. That filtering ensures the result only emits when the OR condition is satisfied. If the initial OR result is falsy, accessing the value raises ConditionalNeverMet—the gate has never opened.

        Args:
            other: Another boolean observable to OR with. Both this observable and other
                   should contain boolean values.

        Returns:
            A conditional observable that only emits when the OR result is truthy. The
            observable updates automatically when either source changes.

        Raises:
            ConditionalNeverMet: If the initial OR result is falsy (both observables
                                are False). Accessing `.value` before any condition is
                                satisfied raises this exception.

        Example:
            ```python
            from fynx import observable
            from fynx.observable.conditional import ConditionalNeverMet

            is_error = observable(True)
            is_warning = observable(False)
            needs_attention = is_error.either(is_warning)
            print(needs_attention.value)  # True (at least one is True)

            has_permission = observable(True)
            is_admin = observable(False)
            can_proceed = has_permission.either(is_admin)
            print(can_proceed.value)  # True

            # If both are False initially, accessing value raises ConditionalNeverMet
            both_false_1 = observable(False)
            both_false_2 = observable(False)
            try:
                result = both_false_1.either(both_false_2)
                _ = result.value  # Raises ConditionalNeverMet
            except ConditionalNeverMet:
                print("Both conditions are False—gate never opens")
            ```
        """
        # Create a computed observable for the OR result
        or_result = self.alongside(other).then(lambda a, b: a or b)

        # Return conditional observable that filters based on truthiness
        # Use a callable condition to avoid timing issues with computed observables
        return or_result & (lambda x: bool(x))
