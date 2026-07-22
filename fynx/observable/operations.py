"""
FynX Operations - natural language reactive operations
==========================================================

This module provides readable methods for transforming, merging, and
filtering reactive values - the implementation layer that operators.py
delegates to. Each method creates a new observable that derives from its
sources and updates automatically when they change.

Five core operations:

- `then(func)` transforms values through functions (equivalent to `>>` operator)
- `alongside(other)` merges observables into tuples (equivalent to `+` operator)
- `requiring(*conditions)` composes boolean conditions with AND logic (equivalent to `&` operator)
- `negate()` inverts boolean values (equivalent to `~` operator)
- `either(other)` creates boolean OR observables

They compose naturally - chain transformations, nest conditions, and build
larger reactive pipelines out of these primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Set, TypeVar, cast

from ..types import ConditionOperand, ObservableOperand
from .operands import unwrap_observable

if TYPE_CHECKING:
    from .base import Observable, ReactiveContext
    from .conditional import ConditionalObservable
    from .merged import MergedObservable

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


class OperationsMixin(Generic[T]):
    """
    Mixin providing natural language reactive operations.

    Unlike a plain function that runs once and returns, these operations
    build a new observable that keeps updating as its sources change. This
    mixin is the foundation for both the operator syntax (`+`, `>>`, `&`, in
    operators.py) and the direct method calls (`.then()`, `.alongside()`, etc).

    The mixin keeps the runtime representation aligned with FynX's algebra:
    pure transform chains compose, products are explicit, and observed values
    create the notification boundary.
    """

    def _create_computed(
        self, func: Callable[..., U], observable: "Observable[Any]"
    ) -> "Observable[U]":
        """
        Create a computed observable that derives its value from other observables.

        For a single observable, applies the function to its value directly.
        For a merged observable, unpacks the tuple and passes its elements as
        separate arguments. Transform functions may only use those explicit
        argument values; reading or mutating observables inside the transform
        raises TransformPurityError with a fix-it hint.

        Transform chains fuse onto their original source when possible, so
        `obs >> f >> g` is represented as one composed transform over `obs`
        rather than as a line of separately maintained runtime nodes. The
        derived value is version-validated and recomputed lazily when read; if
        it has subscribers, those subscribers create the eager boundary needed
        to deliver notifications.
        """
        MergedObservable = _MergedObservable()
        ComputedObservable = _ComputedObservable()

        source_observable = observable
        computation_func = func
        unpack_source = isinstance(observable, MergedObservable)

        def evaluate_transform(callback):
            from . import base as observable_base

            transform_state = observable_base._TRANSFORM_EVALUATION_STATE
            previous_transform_state = transform_state[0]
            transform_state[0] = True
            try:
                return callback()
            finally:
                transform_state[0] = previous_transform_state

        def declared_source_dependencies(source):
            return {source}

        if (
            isinstance(observable, ComputedObservable)
            and not isinstance(observable, MergedObservable)
            and observable._source_observable is not None
            and observable._computation_func is not None
        ):
            previous = observable
            source_observable = previous._source_observable
            previous_unpack = previous._fusion_unpack_source
            fusion_funcs = tuple(previous._get_fusion_funcs())

            def computation_func(source_value):
                if not fusion_funcs:
                    intermediate = source_value
                elif previous_unpack and isinstance(source_value, tuple):
                    intermediate = fusion_funcs[0](*source_value)
                    for fusion_func in fusion_funcs[1:]:
                        intermediate = fusion_func(intermediate)
                else:
                    intermediate = source_value
                    for fusion_func in fusion_funcs:
                        intermediate = fusion_func(intermediate)
                return func(intermediate)

            unpack_source = False
            source_version_before = source_observable._version
            source_value = source_observable.value
            initial_value = evaluate_transform(lambda: computation_func(source_value))
            dynamic_dependencies: Set["Observable[Any]"] = set()
            activate_immediately = source_observable._version != source_version_before
        elif unpack_source:
            source_version_before = source_observable._version
            source_value = source_observable.value
            source_values = cast(tuple[Any, ...], source_value)
            initial_value = evaluate_transform(lambda: computation_func(*source_values))
            dynamic_dependencies = set()
            activate_immediately = source_observable._version != source_version_before
        else:
            source_version_before = source_observable._version
            source_value = source_observable.value
            initial_value = evaluate_transform(lambda: computation_func(source_value))
            dynamic_dependencies = set()
            activate_immediately = source_observable._version != source_version_before

        computed_obs: "Observable[U]" = ComputedObservable(
            None,
            initial_value,
            computation_func,
            source_observable,
            unpack_source=unpack_source,
        )

        if isinstance(computed_obs, ComputedObservable):
            dynamic_dependencies.discard(computed_obs)
            dynamic_dependencies.difference_update(
                declared_source_dependencies(source_observable)
            )
            computed_obs._dynamic_dependencies = dynamic_dependencies
            computed_obs._source_signature = computed_obs._current_source_signature()
            computed_obs._fusion_func = func
            computed_obs._fusion_unpack_source = (
                previous_unpack if "previous" in locals() else unpack_source
            )
            if "previous" in locals():
                computed_obs._fusion_parent = previous
                computed_obs._fusion_funcs = []
                computed_obs._captures_dynamic_dependencies = False
            else:
                computed_obs._fusion_parent = None
                computed_obs._fusion_funcs = [func]
                computed_obs._captures_dynamic_dependencies = False

            if activate_immediately:
                computed_obs._force_eager = True
                computed_obs._activate_dependencies()

        return computed_obs

    def then(self, func: Callable[[T], U]) -> "Observable[U]":
        """
        Transform this observable's value using a function.

        Creates a ComputedObservable that applies `func` to this observable's
        value and re-runs whenever the source changes. Chained pure
        transforms (`obs.then(f).then(g)`) compose internally rather than
        maintaining a separate reactive node per step. The result is cached
        and version-checked, and gains subscribers eagerly enough to deliver
        notifications.

        Args:
            func: A pure function to apply to the observable's value. For merged
                  observables, the function receives unpacked tuple values as
                  separate arguments. If the function needs another observable,
                  combine that observable first with `+` / `.alongside()`.

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
        return self._create_computed(func, cast("Observable[Any]", self))

    def alongside(self, other: ObservableOperand[U]) -> "MergedObservable[T, U]":
        """
        Merge this observable with another into a tuple.

        Creates a MergedObservable holding both values as a tuple, updated
        whenever either changes. Products are canonical for a given ordered
        source list, so repeated expressions like `a + b` reuse the same
        product node while it's live, and the product stays lazy until read
        or observed.

        If either side is already merged, FynX combines with its sources directly, creating a flat ordered product.

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
            print(point3d.value)  # (10, 20, 30)
            ```
        """
        MergedObservable = _MergedObservable()

        return MergedObservable.from_sources(  # type: ignore[return-value]
            cast("Observable[Any]", self), unwrap_observable(other)
        )

    def requiring(self, *conditions: ConditionOperand[T]) -> "ConditionalObservable[T]":
        """
        Compose this observable with conditions using AND logic.

        Creates a ConditionalObservable that only emits when every condition
        evaluates to True. Each condition can be a boolean observable, a
        callable taking the source value and returning a boolean, or another
        ConditionalObservable.

        If this observable is already a ConditionalObservable, this nests a
        new conditional on top, so condition chains can be built up
        incrementally.

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

        Produces a computed observable holding the logical negation of the
        current boolean value, updating automatically as the source changes -
        useful for expressing a negative condition without a separate
        "is_not_X" observable.

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

    def either(self, other: ObservableOperand[Any]) -> "Observable[bool]":
        """
        Create a boolean OR observable between this observable and another.

        Produces a computed observable that's True when either source is
        truthy, False otherwise. It's a total boolean observable - both True
        and False are valid values, nothing is gated or suppressed - so it
        works as a reusable condition for `&` / `.requiring()` and further
        boolean composition.

        Args:
            other: Another boolean observable to OR with. Both this observable and other
                   should contain boolean-like values.

        Returns:
            A computed Observable[bool] that updates automatically when either source
            changes.

        Example:
            ```python
            from fynx import observable

            is_error = observable(True)
            is_warning = observable(False)
            needs_attention = is_error.either(is_warning)
            print(needs_attention.value)  # True (at least one is True)

            has_permission = observable(True)
            is_admin = observable(False)
            can_proceed = has_permission.either(is_admin)
            print(can_proceed.value)  # True
            ```
        """
        return self.alongside(other).then(lambda a, b: bool(a) or bool(b))
