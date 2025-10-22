"""
FynX Operations - Natural Language Reactive Operations and Operator Mixins
==========================================================================

This module provides natural language methods and operator mixins for reactive operations in FynX,
serving as the core implementation layer for both natural language and operator-based reactive programming.

The operations provide a fluent, readable API for reactive programming:

Natural Language Methods:
- `then(func)` - Transform values (equivalent to `>>` operator)
- `alongside(other)` - Merge observables (equivalent to `+` operator)
- `requiring(condition)` - Compose boolean conditions with AND (equivalent to `&` operator)
- `negate()` - Boolean negation (equivalent to `~` operator)
- `either(other)` - OR logic for boolean conditions

Operator Mixins:
- `OperatorMixin` - Provides operator overloading (+, >>, &, ~, |)
- `TupleMixin` - Provides tuple-like behavior for merged observables
- `ValueMixin` - Provides transparent value wrapper behavior
"""

import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from fynx.observable.generic import GenericObservable
from fynx.util import LazyChainBuilder, find_ultimate_source

if TYPE_CHECKING:
    from fynx.observable.types.protocols.conditional_protocol import Conditional
    from fynx.observable.types.protocols.merged_protocol import Mergeable
    from fynx.observable.types.protocols.operations_protocol import Observable

T = TypeVar("T")
U = TypeVar("U")


class OperationsMixin:
    """
    Mixin providing natural language reactive operations.

    This mixin provides the core reactive operations that can be used by any
    observable class. It serves as the foundation for both the operator syntax
    and direct method calls.
    """

    def then(self, func: Callable[[T], U]) -> "Observable[U]":
        """
        Transform this observable's value with the given function.

        Creates a ComputedObservable that automatically recalculates when
        the source observable changes.
        """
        from fynx.observable.computed import ComputedObservable

        # Create a wrapper function that handles tuple unpacking for merged observables
        def tuple_aware_func(value):
            # Always pass the value as-is, no automatic unpacking
            return func(value)

        # Simply create a ComputedObservable - all complex logic is handled there
        return ComputedObservable(
            key=f"computed_from_{getattr(self, 'key', 'unknown')}",
            initial_value=None,
            computation_func=tuple_aware_func,
            source_observable=self,
        )

    def zip(self, *others: "Observable") -> "Observable":
        """
        Zip this observable with others using efficient pairwise emission.

        Emits pairs as soon as both sources have values,
        implementing the zip operation from functional programming.

        Args:
            *others: Other observables to zip with

        Returns:
            Observable that emits tuples of paired values
        """
        from fynx.observable.computed import ComputedObservable

        all_sources = (self,) + others
        n_sources = len(all_sources)

        # Simple state: track if each source has emitted
        has_emitted = [False] * n_sources
        current_values = [None] * n_sources

        def zip_computation(_):
            # Emit when all sources have values (simplified zip behavior)
            if all(has_emitted):
                return tuple(current_values)
            return None

        # Create zipped observable
        zipped = ComputedObservable(
            key=f"zipped_{n_sources}",
            initial_value=None,
            computation_func=zip_computation,
        )

        # Efficient subscription handlers
        def make_handler(index):
            def handler(_=None):
                has_emitted[index] = True
                current_values[index] = all_sources[index].value
                # Emit immediately when all sources have values
                if all(has_emitted):
                    zipped._set_computed_value(tuple(current_values))

            return handler

        # Subscribe to all sources
        for i, obs in enumerate(all_sources):
            obs.subscribe(make_handler(i))

        return zipped

    def debounce(self, milliseconds: float) -> "Observable":
        """
        Debounce emissions using temporal stabilization.

        Only emits the most recent value after a stabilization period with no new values.

        Args:
            milliseconds: Stabilization timeout in milliseconds

        Returns:
            Observable that emits temporally stabilized values
        """
        import time

        from fynx.observable.computed import ComputedObservable

        # Efficient state: track last value and deadline
        last_value = [self.value]
        deadline = [0]
        timer_active = [False]

        def debounce_computation(_):
            current_time = time.time() * 1000
            if current_time >= deadline[0] and timer_active[0]:
                timer_active[0] = False
                return last_value[0]
            return None

        # Create debounced observable
        debounced = ComputedObservable(
            key=f"debounced_{milliseconds}ms",
            initial_value=self.value,
            computation_func=debounce_computation,
        )

        # Single timer approach for efficiency
        def on_source_change(new_value):
            last_value[0] = new_value
            deadline[0] = time.time() * 1000 + milliseconds
            timer_active[0] = True

            # Single timer approach - much more efficient
            if (
                not hasattr(on_source_change, "_timer")
                or not on_source_change._timer.is_alive()
            ):

                def emit_last_value():
                    current_time = time.time() * 1000
                    if current_time >= deadline[0] and last_value[0] == new_value:
                        debounced._set_computed_value(last_value[0])
                        timer_active[0] = False

                import threading

                on_source_change._timer = threading.Timer(
                    milliseconds / 1000, emit_last_value
                )
                on_source_change._timer.start()

        self.subscribe(on_source_change)
        return debounced

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
        if GenericObservable.is_merged_observable(other):
            # If other is already merged, combine with its sources
            return GenericObservable.create_merged_observable(self, *other._source_observables)  # type: ignore
        else:
            # Standard case: combine two observables
            return GenericObservable.create_merged_observable(self, other)  # type: ignore

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
        # If this is already a ConditionalObservable, flatten nested conditionals
        if GenericObservable.is_conditional_observable(self):
            # Get existing conditions and combine with new ones
            existing_conditions = getattr(self, "_processed_conditions", [])
            all_conditions = list(existing_conditions) + list(conditions)

            # Find the ultimate source (skip all conditionals)
            original_source = GenericObservable.get_ultimate_source(self)

            # Create flattened conditional pointing to root source
            return GenericObservable.create_conditional_observable(original_source, *all_conditions)  # type: ignore
        else:
            return GenericObservable.create_conditional_observable(self, *conditions)  # type: ignore

    @staticmethod
    def lift(func: Callable, *observables: "Observable") -> "Observable":
        """
        Lift a pure function into the Observable context.

        Args:
            func: Pure function to lift
            *observables: Observables to apply the function to

        Returns:
            Computed observable that applies func to all observable values

        Example:
            ```python
            result = Observable.lift(lambda x, y, z: x + y + z, obs_a, obs_b, obs_c)
            ```
        """

        def computation():
            values = [obs.value for obs in observables]
            return func(*values)

        # Create computed observable
        lifted = GenericObservable.create_computed_observable(
            f"lift_{func.__name__}", computation(), computation, None
        )

        # Subscribe to all sources
        def update(_):
            lifted._set_computed_value(computation())

        for obs in observables:
            obs.subscribe(update)

        return lifted

    @staticmethod
    def add(*observables: "Observable[float]") -> "Observable[float]":
        """Reactive addition: a + b + c + ..."""
        return Observable.lift(sum, *observables)

    @staticmethod
    def all_true(*observables: "Observable[bool]") -> "Observable[bool]":
        """Reactive AND: a && b && c && ..."""
        return Observable.lift(lambda *vals: all(vals), *observables)

    @staticmethod
    def any_true(*observables: "Observable[bool]") -> "Observable[bool]":
        """Reactive OR: a || b || c || ..."""
        return Observable.lift(lambda *vals: any(vals), *observables)

    @staticmethod
    def race(*observables: "Observable[T]") -> "Observable[T]":
        """
        Create an observable that emits from whichever source emits first.

        Args:
            *observables: Observables to race

        Returns:
            Observable that emits from the first source to emit
        """
        result = GenericObservable.create_computed_observable("race", None, None, None)

        def forward_first_emission(value):
            """Forward the first emission and unsubscribe from others."""
            result.set(value)
            # Unsubscribe from all sources after first emission
            for obs in observables:
                obs.unsubscribe(forward_first_emission)

        # Subscribe to all sources
        for obs in observables:
            obs.subscribe(forward_first_emission)

        return result

    @staticmethod
    def either(left: "Observable[T]", right: "Observable[T]") -> "Observable[T]":
        """
        Emit from either observable, preferring left when both emit.

        Args:
            left: Left observable (preferred)
            right: Right observable

        Returns:
            Observable that emits from either source
        """
        result = GenericObservable.create_computed_observable(
            "either", left.value if left.value is not None else right.value, None, None
        )

        left.subscribe(lambda v: result.set(v))
        right.subscribe(lambda v: result.set(v) if left.value is None else None)

        return result

    @staticmethod
    def pure(value: T) -> "Observable[T]":
        """
        Lift a pure value into Observable context.

        Args:
            value: Pure value to lift

        Returns:
            Observable containing the value
        """
        return GenericObservable.create_computed_observable(
            f"pure_{value}", value, None, None
        )

    def extract(self) -> T:
        """
        Extract value from Observable context.

        Returns:
            Current value of the observable
        """
        return self.value  # type: ignore

    def flatten(self) -> "Observable[T]":
        """
        Flatten Observable[Observable[T]] → Observable[T]

        Returns:
            Flattened observable
        """
        if not GenericObservable.is_computed_observable(self.value):  # type: ignore
            return self  # Already flat

        # Create flattened observable
        flat = GenericObservable.create_computed_observable(f"{self.key}.flat", None, None, None)  # type: ignore

        current_inner_obs = None

        def update_from_inner(value):
            """Forward inner observable's value to flat."""
            flat.set(value)

        def switch_inner_subscription(outer_value):
            """Switch subscription when outer observable changes."""
            nonlocal current_inner_obs

            # Unsubscribe from previous inner observable
            if current_inner_obs is not None:
                current_inner_obs.unsubscribe(update_from_inner)

            # Subscribe to new inner observable
            if GenericObservable.is_computed_observable(outer_value):
                current_inner_obs = outer_value
                current_inner_obs.subscribe(update_from_inner)
                # Update with current inner value
                flat.set(current_inner_obs.value)

        # Subscribe to outer observable
        self.subscribe(switch_inner_subscription)  # type: ignore

        # Initial subscription
        if GenericObservable.is_computed_observable(self.value):  # type: ignore
            switch_inner_subscription(self.value)  # type: ignore

        return flat

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

    def dimap(self, pre: Callable, post: Callable) -> "Observable":
        """
        Transform input and output of this observable.

        Args:
            pre: Transform input before processing
            post: Transform output after processing

        Returns:
            Observable with transformed input and output

        Example:
            ```python
            # Transform input and output
            processed = data.dimap(
                pre=lambda x: x.strip().lower(),  # Input transformation
                post=lambda x: x.title()          # Output transformation
            )
            ```
        """
        # Apply pre-transformation to this observable
        pre_transformed = self.then(pre)

        # Apply post-transformation to the result
        return pre_transformed.then(post)


# ============================================================================
# OPERATOR MIXINS FOR CONSOLIDATING OPERATOR OVERLOADING LOGIC
# ============================================================================


class OperatorMixin(OperationsMixin):
    """
    Mixin class providing common reactive operators for observable classes.

    This mixin consolidates the operator overloading logic that was previously
    duplicated across multiple observable classes. It provides the core reactive
    operators (__add__, __rshift__, __and__, __invert__) that enable FynX's fluent
    reactive programming syntax.

    Classes inheriting from this mixin get automatic support for:
    - Merging with `+` operator
    - Transformation with `>>` operator
    - Conditional filtering with `&` operator
    - Boolean negation with `~` operator

    This mixin should be used by classes that represent reactive values and
    need to support reactive composition operations.
    """

    def __add__(self, other) -> "Mergeable":
        """
        Combine this observable with another using the + operator.

        This creates a merged observable that contains a tuple of both values
        and updates automatically when either observable changes.

        Args:
            other: Another Observable to combine with

        Returns:
            A MergedObservable containing both values as a tuple
        """
        return self.alongside(other)  # type: ignore

    def __radd__(self, other) -> "Mergeable":
        """
        Support right-side addition for merging observables.

        This enables expressions like `other + self` to work correctly,
        ensuring that merged observables can be chained properly.

        Args:
            other: Another Observable to combine with

        Returns:
            A MergedObservable containing both values as a tuple
        """
        return other.alongside(self)  # type: ignore

    def __rshift__(self, func: Callable) -> "Observable":
        """
        Apply a transformation function using the >> operator to create computed observables.

        This implements the functorial map operation over observables, allowing you to
        transform observable values through pure functions while preserving reactivity.

        Args:
            func: A pure function to apply to the observable's value(s)

        Returns:
            A new computed Observable containing the transformed values
        """
        return self.then(func)

    def __and__(self, condition) -> "Conditional":
        """
        Create a conditional observable using the & operator for filtered reactivity.

        This creates a ConditionalObservable that only emits values when all
        specified conditions are True, enabling precise control over reactive updates.

        Args:
            condition: A boolean Observable, callable, or compound condition

        Returns:
            A ConditionalObservable that filters values based on the condition
        """
        return self.requiring(condition)  # type: ignore

    def __invert__(self) -> "Observable[bool]":
        """
        Create a negated boolean observable using the ~ operator.

        This creates a computed observable that returns the logical negation
        of the current boolean value, useful for creating inverse conditions.

        Returns:
            A computed Observable[bool] with negated boolean value
        """
        return self.negate()  # type: ignore

    def __or__(self, other) -> "Observable":
        """
        Create a logical OR condition using the | operator.

        This creates a conditional observable that only emits when the OR result
        is truthy. If the initial OR result is falsy, raises ConditionalNeverMet.

        Args:
            other: Another boolean observable to OR with

        Returns:
            A conditional observable that only emits when OR is truthy

        Raises:
            ConditionalNeverMet: If initial OR result is falsy
        """
        return self.either(other)  # type: ignore


class TupleMixin:
    """
    Mixin class providing tuple-like operators for merged observables.

    This mixin adds tuple-like behavior to observables that represent collections
    of values (like MergedObservable). It provides operators for iteration,
    indexing, and length operations that make merged observables behave like
    tuples of their component values.

    Classes inheriting from this mixin get automatic support for:
    - Iteration with `for item in merged:`
    - Length with `len(merged)`
    - Indexing with `merged[0]`, `merged[-1]`, etc.
    - Setting values by index with `merged[0] = new_value`
    """

    def __iter__(self):
        """Allow iteration over the tuple value."""
        return iter(self._value)  # type: ignore

    def __len__(self) -> int:
        """Return the number of combined observables."""
        return len(self._source_observables)  # type: ignore

    def __getitem__(self, index: int):
        """Allow indexing into the merged observable like a tuple."""
        if self._value is None:  # type: ignore
            raise IndexError("MergedObservable has no value")
        return self._value[index]  # type: ignore

    def __setitem__(self, index: int, value):
        """Allow setting values by index, updating the corresponding source observable."""
        if 0 <= index < len(self._source_observables):  # type: ignore
            self._source_observables[index].set(value)  # type: ignore
        else:
            raise IndexError("Index out of range")


class ValueMixin:
    """
    Mixin class providing value wrapper operators for ObservableValue.

    This mixin adds operators that make observable values behave transparently
    like their underlying values in most Python contexts. It provides magic
    methods for equality, string conversion, iteration, indexing, etc., while
    also supporting the reactive operators.

    Classes inheriting from this mixin get automatic support for:
    - Value-like behavior (equality, string conversion, etc.)
    - Reactive operators (__add__, __and__, __invert__, __rshift__)
    - Transparent access to the wrapped observable
    """

    def __eq__(self, other) -> bool:
        return self._current_value == other  # type: ignore

    def __str__(self) -> str:
        return str(self._current_value)  # type: ignore

    def __repr__(self) -> str:
        return repr(self._current_value)  # type: ignore

    def __len__(self) -> int:
        if self._current_value is None:  # type: ignore
            return 0
        if hasattr(self._current_value, "__len__"):  # type: ignore
            return len(self._current_value)  # type: ignore
        return 0

    def __iter__(self):
        if self._current_value is None:  # type: ignore
            return iter([])
        if hasattr(self._current_value, "__iter__"):  # type: ignore
            return iter(self._current_value)  # type: ignore
        return iter([self._current_value])  # type: ignore

    def __getitem__(self, key):
        if self._current_value is None:  # type: ignore
            raise IndexError("observable value is None")
        if hasattr(self._current_value, "__getitem__"):  # type: ignore
            return self._current_value[key]  # type: ignore
        raise TypeError(
            f"'{type(self._current_value).__name__}' object is not subscriptable"  # type: ignore
        )

    def __contains__(self, item) -> bool:
        if self._current_value is None:  # type: ignore
            return False
        if hasattr(self._current_value, "__contains__"):  # type: ignore
            return item in self._current_value  # type: ignore
        return False

    def __bool__(self) -> bool:
        return bool(self._current_value)  # type: ignore

    def _unwrap_operand(self, operand):
        """Unwrap operand if it's an ObservableValue, otherwise return as-is."""
        if hasattr(operand, "observable"):
            return operand.observable  # type: ignore
        return operand

    def __add__(self, other) -> "Mergeable":
        """Support merging observables with + operator."""
        unwrapped_other = self._unwrap_operand(other)  # type: ignore
        from ..computed import MergedObservable

        return MergedObservable(self._observable, unwrapped_other)  # type: ignore[attr-defined]

    def __radd__(self, other) -> "Mergeable":
        """Support right-side addition for merging observables."""
        unwrapped_other = self._unwrap_operand(other)  # type: ignore
        from ..computed import MergedObservable

        return MergedObservable(unwrapped_other, self._observable)  # type: ignore[attr-defined]

    def __and__(self, condition) -> "Conditional":
        """Support conditional observables with & operator."""
        unwrapped_condition = self._unwrap_operand(condition)  # type: ignore

        # Handle callable conditions by creating computed observables
        if callable(unwrapped_condition) and not hasattr(unwrapped_condition, "value"):
            # Create a computed observable that evaluates the condition
            bool_condition = self._observable.then(
                lambda x: x is not None and bool(unwrapped_condition(x))
            )
            from ..computed import ConditionalObservable

            return ConditionalObservable(self._observable, bool_condition)  # type: ignore[attr-defined]
        else:
            # Boolean observable
            from ..computed import ConditionalObservable

            return ConditionalObservable(self._observable, unwrapped_condition)  # type: ignore[attr-defined]

    def __invert__(self):
        """Support negating conditions with ~ operator."""
        return self._observable.__invert__()  # type: ignore[attr-defined]

    def __rshift__(self, func):
        """Support computed observables with >> operator."""
        return self._observable >> func  # type: ignore[attr-defined]
