"""
FynX Observable Computed - Computed Observable Implementation
===========================================================

This module provides the ComputedObservable class, a read-only observable that
derives its value from other observables through automatic computation.

Computed observables represent derived state—values calculated from other observables
rather than set directly. When source observables change, computed values recalculate
automatically. That automatic recalculation eliminates manual synchronization: you
declare the relationship, and the framework maintains it.

Computed observables are read-only by design. You cannot set them directly because
their values flow from dependencies. This constraint prevents breaking reactive
relationships—if you could set a computed value, it would diverge from its source.
The framework updates computed values internally when dependencies change, but the
public interface enforces immutability.

Creating Computed Observables
-----------------------------

You create computed observables using the `>>` operator or the `.then()` method.
Both approaches transform source observables through pure functions, creating
derived values that update automatically:

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

The `>>` operator applies the function to the observable's value, creating a new
computed observable. For merged observables (created with `+`), the function receives
multiple arguments corresponding to the tuple values. That pattern enables reactive
calculations across multiple inputs without manual coordination.

Read-Only Protection
--------------------

Computed observables prevent direct modification to maintain reactive integrity:

```python
total = (price + quantity) >> (lambda p, q: p * q)

# This works—updates propagate automatically
price.set(15)
print(total.value)  # 75.0

# This raises ValueError—computed values are read-only
total.set(100)  # ValueError: Computed observables are read-only
```

Attempting to set a computed observable raises `ValueError`. That error signals
that you're trying to break the reactive relationship—computed values derive from
dependencies, not direct assignment. To change a computed value, modify its source
observables instead.

Internal Updates
----------------

The framework updates computed values through the internal `_set_computed_value()`
method. This method bypasses the read-only protection to allow legitimate framework-driven
updates when dependencies change:

```python
# This is used internally by the >> operator and .then() method
computed_obs._set_computed_value(new_value)  # Allowed internally
computed_obs.set(new_value)                  # Not allowed—raises ValueError
```

External code should not call `_set_computed_value()` directly. That method exists
for framework internals—the `>>` operator and `.then()` method use it to update
computed values when dependencies change. Direct use may break reactive relationships.

Performance Considerations
--------------------------

Computed observables use lazy evaluation: they recalculate only when accessed after
dependencies change. That strategy avoids unnecessary work—if nothing reads the computed
value, it doesn't recompute. Results are cached until dependencies actually change,
so repeated reads return the cached value without recomputation.

Dependencies are explicit. A transform may only use the values passed as arguments;
to use more than one observable, combine them first with `+` or `.alongside()`.
This keeps the reactive graph stable and lets composition laws apply predictably.

Memory overhead is minimal beyond regular observables. Computed observables store
the computation function and source observable reference, but they reuse the same
observer infrastructure as regular observables.

Common Patterns
---------------

**Mathematical Computations**:
```python
width = observable(10)
height = observable(20)
area = (width + height) >> (lambda w, h: w * h)
perimeter = (width + height) >> (lambda w, h: 2 * (w + h))
```

**String Formatting**:
```python
first_name = observable("John")
last_name = observable("Doe")
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")
```

**Validation States**:
```python
email = observable("")
is_valid_email = email >> (lambda e: "@" in e and len(e) > 5)
```

**Conditional Computations**:
```python
count = observable(0)
is_even = count >> (lambda c: c % 2 == 0)
```

Limitations
-----------

Computed observables cannot be set directly—that constraint is by design, not a
missing feature. Dependencies must be accessed synchronously during computation;
asynchronous access won't be tracked. They cannot depend on external state that
changes independently of observables—only observable values trigger recomputation.

Computation functions must be pure: no side effects, no observable reads, and no
observable writes. If a transform reads `.value` or calls `.set()` on any observable,
FynX raises `TransformPurityError` with a hint to pass that dependency explicitly
or move the effect to `.subscribe()` / `@reactive`.

Error Handling
--------------

When computation functions raise exceptions, those errors propagate normally. The
reactive system doesn't swallow exceptions—if a computation fails, the error
surfaces to the caller. Failed computations may leave computed observables with
stale values until dependencies change and recomputation succeeds.

Dependencies continue working normally even if one computation fails. That isolation
prevents cascading failures—one broken computation doesn't break the entire reactive
graph. You can handle errors in computation functions using try-except blocks if
you need graceful degradation.

See Also
--------

- `fynx.observable`: The >> operator and .then() method for creating computed observables
- `fynx.observable`: Core observable classes
- `fynx.store`: For organizing observables in reactive containers
"""

from typing import Callable, Optional, TypeVar

from . import base as _base
from .base import Observable, TransformPurityError

T = TypeVar("T")


class ComputedObservable(Observable[T]):
    """
    A read-only observable that derives its value from other observables.

    ComputedObservable extends Observable to represent computed values—values that
    derive from other observables rather than direct assignment. Unlike regular
    observables, computed observables are read-only: you cannot set them directly
    because their values flow from dependencies.

    This class provides type-based distinction from regular observables. That distinction
    enables compile-time type checking and runtime behavior differences—computed
    observables maintain the same interface as regular observables for reading values
    and subscribing to changes, but they enforce immutability at runtime.

    You typically create computed observables using the `>>` operator or `.then()` method,
    not by direct instantiation. The framework creates ComputedObservable instances
    internally when you use those operators:

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

    Direct instantiation of ComputedObservable is supported but rarely needed. The
    framework handles creation automatically when you use reactive operators.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        computation_func: Optional[Callable] = None,
        source_observable: Optional["Observable"] = None,
        unpack_source: bool = False,
    ) -> None:
        super().__init__(key, initial_value)
        # Store computation function for chain fusion optimization
        self._computation_func = computation_func
        # Store source observable for fusion
        self._source_observable = source_observable
        self._unpack_source = unpack_source
        self._dynamic_dependencies = set()
        self._source_signature = self._current_source_signature()
        self._dependency_observers = set()
        self._dependencies_active = False
        self._is_dirty = False
        self._force_eager = False
        self._fusion_funcs = [computation_func] if computation_func is not None else []
        self._fusion_parent = None
        self._fusion_func = computation_func
        self._fusion_unpack_source = unpack_source
        self._captures_dynamic_dependencies = False
        self._dependency_callback = self._dependency_changed
        self._dependency_uses_fast_observer = False

    def _get_fusion_funcs(self):
        funcs = []
        current = self

        while isinstance(current, ComputedObservable):
            fusion_func = getattr(current, "_fusion_func", None)
            if fusion_func is not None:
                funcs.append(fusion_func)
            current = getattr(current, "_fusion_parent", None)

        funcs.reverse()
        return funcs

    def _current_source_signature(self):
        """Return a cheap version signature for the observable's current inputs."""
        source = self._source_observable
        dynamic_dependencies = getattr(self, "_dynamic_dependencies", set())

        if source is None and not dynamic_dependencies:
            return None

        if not dynamic_dependencies and source is not None:
            return (id(source), getattr(source, "_version", 0))

        dependencies = self._runtime_dependencies()
        return tuple(
            sorted(
                (id(dependency), getattr(dependency, "_version", 0))
                for dependency in dependencies
            )
        )

    def _source_changed(self) -> bool:
        return self._current_source_signature() != self._source_signature

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
        if getattr(source, "_is_dirty", False):
            return source.value
        if hasattr(source, "_source_changed") and source._source_changed():
            return source.value
        if hasattr(source, "_value"):
            return source._value
        return source.value

    def _recompute_value(self):
        source = self._source_observable
        if source is None:
            return self._value

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
        dependencies.update(getattr(self, "_dynamic_dependencies", set()))
        dependencies.discard(self)
        return dependencies

    def _guarded_recompute_value(self):
        if (
            not getattr(self, "_dynamic_dependencies", set())
            and self._source_observable is not None
        ):
            dependencies = (self._source_observable,)
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
            getattr(self._source_observable, "_version", 0),
        )
        self._is_dirty = False
        if self._value == new_value:
            return

        self._value = new_value
        self._version = getattr(self, "_version", 0) + 1

        if not getattr(self, "_is_notifying", False):
            self._is_notifying = True
            try:
                direct_observers = getattr(self, "_direct_observers", None)
                if direct_observers and len(direct_observers) == len(self._observers):
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
        callback = getattr(self, "_single_direct_callback", None)
        if callback is None:
            callback = next(iter(getattr(self, "_direct_callbacks", ())))
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
            direct_observers = getattr(self, "_direct_observers", None)
            if direct_observers and len(direct_observers) == len(self._observers) == 1:
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
        dynamic_dependencies = getattr(self, "_dynamic_dependencies", set())
        return self._source_observable is not None and not dynamic_dependencies

    def _notify_observers_source_only(self) -> None:
        """Fast path for pure transforms with exactly one runtime source."""
        source = self._source_observable
        if source is None or self._computation_func is None:
            return

        if getattr(source, "_is_dirty", False) or (
            hasattr(source, "_source_changed") and source._source_changed()
        ):
            source_value = source.value
        else:
            source_value = source._value

        new_value = self._apply_computation_to_source_value(source_value)
        self._source_signature = (id(source), getattr(source, "_version", 0))
        self._is_dirty = False
        if self._value == new_value:
            return

        self._value = new_value
        self._version = getattr(self, "_version", 0) + 1

        if not getattr(self, "_is_notifying", False):
            self._is_notifying = True
            try:
                direct_observers = getattr(self, "_direct_observers", None)
                if direct_observers and len(direct_observers) == len(self._observers):
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
                self._version = getattr(self, "_version", 0) + 1
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
    def value(self) -> Optional[T]:
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
                    self._version = getattr(self, "_version", 0) + 1
        return self._value

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

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating computed observable values.

        This method bypasses the read-only protection enforced by the public `set()`
        method. The framework calls it internally when dependencies change—the `>>`
        operator and `.then()` method use it to update computed values after
        recomputation.

        External code should not call this method directly. That restriction prevents
        breaking reactive relationships—only the framework should update computed
        values, and only when dependencies actually change. Direct use may create
        inconsistent state where computed values don't match their dependencies.

        Args:
            value: The new computed value calculated from dependencies.
                  Can be any type that the computed function returns.
        """
        self._source_signature = self._current_source_signature()
        self._is_dirty = False
        super().set(value)

    def set(self, value: Optional[T]) -> None:
        """
        Prevent direct modification of computed observable values.

        This method always raises `ValueError` to enforce read-only behavior.
        Computed observables derive their values from dependencies—setting them
        directly would break that reactive relationship. The framework updates
        computed values internally when dependencies change, but external code
        cannot modify them.

        To change a computed value, modify its source observables instead. The
        computed value updates automatically when dependencies change:

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
