"""
FynX Observable Computed - computed observable implementation
===============================================================

This module provides `ComputedObservable`, a read-only observable whose value
is derived from other observables rather than set directly. When a source
changes, the computed value recalculates - you declare the relationship once
and FynX keeps it in sync.

Because the value flows from its dependencies, a computed observable can't be
set directly; doing so would let it diverge from its source. FynX updates it
internally through `_set_computed_value()` when a dependency changes, but the
public interface only allows reads.

Creating computed observables
------------------------------

Use the `>>` operator or the `.then()` method. Both transform one or more
source observables through a pure function into a derived value that updates
automatically:

```python
from fynx import observable

# Base observables
price = observable(10.0)
quantity = observable(5)

# Computed observable using the >> operator
total = (price + quantity) >> (lambda p, q: p * q)
print(total.value)  # 50.0

# Alternative using .then() method
total_alt = (price + quantity).then(lambda p, q: p * q)
print(total_alt.value)  # 50.0
```

The `>>` operator applies the function to the observable's value, creating a
new computed observable. For merged observables (created with `+`), the
function receives one argument per source, so a single transform can react to
several inputs at once.

Read-only protection
---------------------

Setting a computed observable directly raises an error instead of silently
breaking the reactive relationship:

```python
total = (price + quantity) >> (lambda p, q: p * q)

# This works - updates propagate automatically
price.set(15)
print(total.value)  # 75.0

# This raises ValueError - computed values are read-only
total.set(100)  # ValueError: Computed observables are read-only
```

To change a computed value, modify its source observables instead - the
computed value follows automatically.

Internal updates
------------------

FynX updates computed values through the internal `_set_computed_value()`
method, which bypasses the read-only check:

```python
# This is used internally by the >> operator and .then() method
computed_obs._set_computed_value(new_value)  # Allowed internally
computed_obs.set(new_value)                  # Not allowed - raises ValueError
```

Don't call `_set_computed_value()` from outside the framework; it exists so
`>>` and `.then()` can update computed values when dependencies change, and
calling it elsewhere can leave a computed value out of sync with its source.

Performance
-----------

Evaluation is lazy: a computed value only recalculates when read after a
dependency has changed, and the result is cached in between. A transform may
only use the values passed to it as arguments; to depend on more than one
observable, combine them first with `+` or `.alongside()`. That keeps the
dependency graph explicit and the composition rules predictable. Beyond the
computation function and a source reference, a computed observable carries no
more overhead than a regular one.

Common patterns
-----------------

**Mathematical computations**:
```python
width = observable(10)
height = observable(20)
area = (width + height) >> (lambda w, h: w * h)
perimeter = (width + height) >> (lambda w, h: 2 * (w + h))
```

**String formatting**:
```python
first_name = observable("John")
last_name = observable("Doe")
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")
```

**Validation states**:
```python
email = observable("")
is_valid_email = email >> (lambda e: "@" in e and len(e) > 5)
```

**Conditional computations**:
```python
count = observable(0)
is_even = count >> (lambda c: c % 2 == 0)
```

Limitations
-----------

Dependencies must be read synchronously during the computation; anything
accessed later (e.g. from a callback) won't be tracked. Only observable
values trigger recomputation - a computed value has no way to notice external
state changing on its own.

Computation functions must be pure: no side effects, no observable reads or
writes beyond the arguments passed in. Reading `.value` or calling `.set()`
on any observable inside a transform raises `TransformPurityError`, with a
hint to pass that dependency explicitly or move the effect to `.subscribe()`
/ `@reactive`.

Error handling
--------------

Exceptions from a computation function propagate to the caller rather than
being swallowed, and a failed computation can leave a computed observable
holding a stale value until the next successful recomputation. A failure in
one computation doesn't affect others in the graph, so try/except around a
transform is enough if you need to degrade gracefully.

See Also
--------

- `fynx.observable`: The >> operator and .then() method for creating computed observables
- `fynx.observable`: Core observable classes
- `fynx.store`: For organizing observables in reactive containers
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Set, TypeVar, cast

from . import base as _base
from .base import Observable, TransformPurityError

T = TypeVar("T")


class ComputedObservable(Observable[T]):
    """
    A read-only observable that derives its value from other observables.

    ComputedObservable extends Observable but disallows direct writes, since
    its value flows from dependencies rather than assignment. Reading it and
    subscribing to it work exactly like a regular observable; the difference
    is enforced only at write time.

    You'll normally get one from the `>>` operator or `.then()` method rather
    than instantiating it directly - both create a ComputedObservable
    internally:

    Example:
        ```python
        from fynx import observable

        # Regular observable
        counter = observable(0)

        # Computed observable using >> operator (typical approach)
        doubled = counter >> (lambda x: x * 2)
        print(doubled.value)  # 0

        # Attempting to set a computed observable raises ValueError
        doubled.set(10)  # Raises ValueError: Computed observables are read-only
        ```

    Direct instantiation is supported but rarely needed in practice.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        computation_func: Optional[Callable[..., T]] = None,
        source_observable: Optional["Observable[Any]"] = None,
        unpack_source: bool = False,
    ) -> None:
        super().__init__(key, initial_value)
        # Store computation function for chain fusion optimization
        self._computation_func = computation_func
        # Store source observable for fusion
        self._source_observable = source_observable
        self._unpack_source = unpack_source
        self._dynamic_dependencies: Set[Observable[Any]] = set()
        self._source_signature = self._current_source_signature()
        self._dependency_observers: Set[Observable[Any]] = set()
        self._dependencies_active = False
        self._is_dirty = False
        self._force_eager = False
        self._fusion_funcs: list[Callable[..., Any]] = (
            [computation_func] if computation_func is not None else []
        )
        self._fusion_parent: Optional[ComputedObservable[Any]] = None
        self._fusion_func: Optional[Callable[..., Any]] = computation_func
        self._fusion_unpack_source = unpack_source
        self._captures_dynamic_dependencies = False
        self._dependency_callback: Callable[..., Any] = self._dependency_changed
        self._dependency_uses_fast_observer = False

    def _get_fusion_funcs(self):
        funcs = []
        current = self

        while isinstance(current, ComputedObservable):
            fusion_func = current._fusion_func
            if fusion_func is not None:
                funcs.append(fusion_func)
            current = current._fusion_parent

        funcs.reverse()
        return funcs

    def _current_source_signature(self):
        """Return a cheap version signature for the observable's current inputs."""
        source = self._source_observable

        if source is None and not self._dynamic_dependencies:
            return None

        if not self._dynamic_dependencies and source is not None:
            return (id(source), source._version)

        dependencies = self._runtime_dependencies()
        return tuple(
            sorted((id(dependency), dependency._version) for dependency in dependencies)
        )

    def _source_changed(self) -> bool:
        return (
            self._current_source_signature() != self._source_signature
            or self._source_is_stale()
        )

    @staticmethod
    def _observable_is_stale(observable: "Observable[Any]") -> bool:
        return isinstance(observable, ComputedObservable) and (
            observable._is_dirty or observable._source_changed()
        )

    def _source_is_stale(self) -> bool:
        source = self._source_observable
        return source is not None and self._observable_is_stale(source)

    def _apply_computation_to_source_value(self, source_value):
        if self._computation_func is None:
            return source_value
        transform_state = _base._TRANSFORM_EVALUATION_STATE
        previous_transform_state = transform_state[0]
        transform_state[0] = True
        try:
            if self._unpack_source and isinstance(source_value, tuple):
                return self._computation_func(*source_value)
            return self._computation_func(source_value)
        finally:
            transform_state[0] = previous_transform_state

    def _source_current_value(self):
        source = self._source_observable
        if source is None:
            return None

        if self._source_is_stale():
            return source.value
        return source._value

    def _recompute_value(self) -> T:
        source = self._source_observable
        if source is None:
            return cast(T, self._value)

        value = self._apply_computation_to_source_value(self._source_current_value())
        self._source_signature = self._current_source_signature()
        self._is_dirty = False
        return value

    def _declared_source_dependencies(self):
        source = self._source_observable
        dependencies = set()

        if source is not None:
            dependencies.add(source)
        dependencies.discard(self)
        return dependencies

    def _runtime_dependencies(self):
        dependencies = self._declared_source_dependencies()
        dependencies.update(self._dynamic_dependencies)
        dependencies.discard(self)
        return dependencies

    def _guarded_recompute_value(self) -> T:
        if not self._dynamic_dependencies and self._source_observable is not None:
            dependencies = {self._source_observable}
        else:
            dependencies = self._runtime_dependencies()

        Observable._computation_dependency_stack.append(dependencies)
        try:
            return self._recompute_value()
        finally:
            Observable._computation_dependency_stack.pop()

    def _dependency_changed(self) -> None:
        """Handle an upstream change with lazy recomputation when unobserved."""
        self._is_dirty = True
        if self._observers or self._force_eager:
            Observable._schedule_notification(self)

    def _source_only_dependency_changed(self) -> None:
        """Handle an upstream change for pure single-source transforms."""
        self._is_dirty = True
        if self._observers or self._force_eager:
            self._notify_observers_source_only()

    def _source_only_dependency_changed_fast(self, source_value) -> None:
        """Fast-lane source observer for pure single-source transforms."""
        if not (self._observers or self._force_eager):
            self._is_dirty = True
            return

        if self._computation_func is None or self._source_observable is None:
            return

        new_value = self._apply_computation_to_source_value(source_value)
        self._source_signature = (
            id(self._source_observable),
            self._source_observable._version,
        )
        self._is_dirty = False
        if self._value == new_value:
            return

        self._value = new_value
        self._version += 1

        if not self._is_notifying:
            self._is_notifying = True
            try:
                if self._direct_observers and len(self._direct_observers) == len(
                    self._observers
                ):
                    callbacks = self._direct_callbacks_for_notification()
                    if len(callbacks) == 1:
                        callbacks[0](new_value)
                    else:
                        for callback in callbacks:
                            callback(new_value)
                else:
                    observers = self._observers_for_notification()
                    if len(observers) == 1:
                        observers[0]()
                    else:
                        for observer in observers:
                            observer()
            finally:
                self._is_notifying = False

    def _make_single_direct_source_observer(self):
        source = self._source_observable
        computation_func = self._computation_func
        callback = self._single_direct_callback
        if callback is None:
            callback = next(iter(self._direct_callbacks))
        unpack_source = self._unpack_source
        transform_state = _base._TRANSFORM_EVALUATION_STATE

        def single_direct_source_observer(source_value):
            previous_transform_state = transform_state[0]
            transform_state[0] = True
            try:
                if unpack_source and isinstance(source_value, tuple):
                    new_value = computation_func(*source_value)
                else:
                    new_value = computation_func(source_value)
            finally:
                transform_state[0] = previous_transform_state

            self._source_signature = (id(source), source._version)
            self._is_dirty = False
            if self._value == new_value:
                return

            self._value = new_value
            self._version += 1
            callback(new_value)

        return single_direct_source_observer

    def _choose_dependency_callback(self):
        use_fast_observer = (
            self._can_recompute_inline()
            and self._source_observable.__class__ is Observable
        )
        if use_fast_observer:
            self._dependency_uses_fast_observer = True
            if (
                self._direct_observers
                and len(self._direct_observers) == len(self._observers) == 1
            ):
                return self._make_single_direct_source_observer()
            return self._source_only_dependency_changed_fast

        self._dependency_uses_fast_observer = False
        if self._can_recompute_inline():
            return self._source_only_dependency_changed
        return self._dependency_changed

    def _refresh_dependency_callback(self) -> None:
        if not self._dependencies_active:
            return

        old_callback = self._dependency_callback
        old_uses_fast_observer = self._dependency_uses_fast_observer
        new_callback = self._choose_dependency_callback()
        new_uses_fast_observer = self._dependency_uses_fast_observer
        if (
            old_callback is new_callback
            and old_uses_fast_observer == new_uses_fast_observer
        ):
            return

        for observable in self._dependency_observers:
            if old_uses_fast_observer:
                observable.remove_fast_observer(old_callback)
            else:
                observable.remove_observer(old_callback)

        self._dependency_callback = new_callback
        for observable in self._dependency_observers:
            if new_uses_fast_observer:
                observable.add_fast_observer(new_callback)
            else:
                observable.add_observer(new_callback)

    def _can_recompute_inline(self) -> bool:
        """True when this node has a single runtime input and cannot glitch."""
        if self._computation_func is None:
            return False
        return self._source_observable is not None and not self._dynamic_dependencies

    def _notify_observers_source_only(self) -> None:
        """Fast path for pure transforms with exactly one runtime source."""
        source = self._source_observable
        if source is None or self._computation_func is None:
            return

        if self._source_is_stale():
            source_value = source.value
        else:
            source_value = source._value

        new_value = self._apply_computation_to_source_value(source_value)
        self._source_signature = (id(source), source._version)
        self._is_dirty = False
        if self._value == new_value:
            return

        self._value = new_value
        self._version += 1

        if not self._is_notifying:
            self._is_notifying = True
            try:
                if self._direct_observers and len(self._direct_observers) == len(
                    self._observers
                ):
                    callbacks = self._direct_callbacks_for_notification()
                    if len(callbacks) == 1:
                        callbacks[0](new_value)
                    else:
                        for callback in callbacks:
                            callback(new_value)
                else:
                    observers = self._observers_for_notification()
                    if len(observers) == 1:
                        observers[0]()
                    else:
                        for observer in observers:
                            observer()
            finally:
                self._is_notifying = False

    def _notify_observers(self) -> None:
        """Recompute once at stabilization time, then notify dependents if changed."""
        if self._computation_func is not None and (
            self._is_dirty or self._source_changed()
        ):
            new_value = self._guarded_recompute_value()
            if self._value != new_value:
                self._value = new_value
                self._version += 1
                super()._notify_observers()
            return

        super()._notify_observers()

    def _activate_dependencies(self) -> None:
        if (
            self._dependencies_active
            or self._source_observable is None
            or self._computation_func is None
        ):
            return

        self._dependencies_active = True
        self._dependency_callback = self._choose_dependency_callback()
        self._sync_dependency_observers()

    def _sync_dependency_observers(self) -> None:
        if not self._dependencies_active:
            return

        target_dependencies = self._runtime_dependencies()

        callback = self._dependency_callback

        for observable in self._dependency_observers - target_dependencies:
            if self._dependency_uses_fast_observer:
                observable.remove_fast_observer(callback)
            else:
                observable.remove_observer(callback)

        for observable in target_dependencies - self._dependency_observers:
            if self._dependency_uses_fast_observer:
                observable.add_fast_observer(callback)
            else:
                observable.add_observer(callback)

        self._dependency_observers = target_dependencies

    def _deactivate_dependencies(self) -> None:
        if not self._dependencies_active:
            return

        callback = self._dependency_callback
        for observable in self._dependency_observers:
            if self._dependency_uses_fast_observer:
                observable.remove_fast_observer(callback)
            else:
                observable.remove_observer(callback)
        self._dependency_observers = set()
        self._dependencies_active = False
        self._dependency_callback = self._dependency_changed
        self._dependency_uses_fast_observer = False

    @property
    def value(self) -> T:
        if _base._TRANSFORM_EVALUATION_STATE[0]:
            from .base import _transform_purity_message

            raise TransformPurityError(_transform_purity_message("read", self._key))

        if Observable._dependency_capture_stack:
            Observable._dependency_capture_stack[-1].add(self)

        if self._computation_func is not None and (
            self._is_dirty or self._source_changed()
        ):
            new_value = self._guarded_recompute_value()
            if self._value != new_value:
                if self._observers:
                    self._set_computed_value(new_value)
                else:
                    self._value = new_value
                    self._version += 1
        return cast(T, self._value)

    def add_observer(self, observer: Callable) -> None:
        was_active = self._dependencies_active
        super().add_observer(observer)
        if was_active:
            self._refresh_dependency_callback()
        else:
            self._activate_dependencies()

    def remove_observer(self, observer: Callable) -> None:
        super().remove_observer(observer)
        if not self._observers:
            self._deactivate_dependencies()
        else:
            self._refresh_dependency_callback()

    def _set_computed_value(self, value: T) -> None:
        """
        Internal method for updating computed observable values.

        Bypasses the read-only protection enforced by the public `set()`
        method. This is called internally when a dependency changes - the
        `>>` operator and `.then()` method use it to store a freshly
        recomputed value. Calling it from outside the framework can leave a
        computed observable holding a value inconsistent with its dependencies.

        Args:
            value: The new computed value calculated from dependencies.
                  Can be any type that the computed function returns.
        """
        self._source_signature = self._current_source_signature()
        self._is_dirty = False
        super().set(value)

    def set(self, value: T) -> None:
        """
        Prevent direct modification of computed observable values.

        Always raises `ValueError`: a computed observable's value flows from
        its dependencies, so setting it directly would break that
        relationship. FynX updates it internally when a dependency changes,
        but nothing else should.

        To change a computed value, modify its source observables instead -
        it updates automatically:

        ```python
        from fynx import observable

        base = observable(5)
        doubled = base >> (lambda x: x * 2)
        print(doubled.value)  # 10

        # Correct: Modify the source observable
        base.set(6)
        print(doubled.value)  # 12 (updated automatically)

        # Incorrect: Try to set computed value directly
        doubled.set(20)  # Raises ValueError
        ```

        Args:
            value: The value that would be set (ignored).
                  This parameter exists for API compatibility but is not used.

        Raises:
            ValueError: Always raised to prevent direct modification of computed values.
                       Modify source observables instead to update computed values.

        See Also:
            >> operator: Modern syntax for creating computed observables
            _set_computed_value: Internal method used by the framework for updates
        """
        raise ValueError(
            "Computed observables are read-only and cannot be set directly"
        )
