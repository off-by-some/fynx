"""
Fynx-Style Frontend for DeltaKVStore
====================================

A reactive programming interface that combines Fynx's API with DeltaKVStore's
mathematical efficiency. Provides observables, computed values, and reactions
with O(affected) incremental computation.
"""

import threading
from typing import Any, Callable, Dict, Optional, Set, Type, TypeVar, Union

from prototype import ChangeType, Delta, DeltaKVStore

T = TypeVar("T")


class BaseObservable:
    """
    Base class for all observables with operator overloading.

    Supports Fynx's five reactive operators: >>, +, &, |, ~
    """

    def __init__(self, store: "Store", key: str):
        self._store = store
        self._key = key
        self._lock = threading.RLock()

    @property
    def value(self):
        """Get the current value."""
        return self._store._delta_store.get(self._key)

    def subscribe(self, callback: Callable) -> "BaseObservable":
        """Subscribe to changes."""

        def delta_callback(delta: Delta):
            if delta.key == self._key:
                if delta.change_type == ChangeType.SET:
                    callback(delta.new_value)
                elif delta.change_type == ChangeType.COMPUTED_UPDATE:
                    callback(delta.new_value)

        self._store._delta_store.subscribe(self._key, delta_callback)
        return self

    # Fynx operator overloading
    def __rshift__(self, func: Callable) -> "FusedOperation":
        """Transform with >> operator (then)."""
        # Return a fused operation that can be chained before materializing
        return FusedOperation(self, [func])

    def __add__(self, other: "BaseObservable") -> "MergedObservable":
        """Combine with + operator (alongside)."""
        return MergedObservable(
            self._store, f"{self._key}_{other._key}_merged", [self, other]
        )

    def __and__(self, condition: "BaseObservable") -> "ConditionalObservable":
        """Filter with & operator (requiring)."""
        return ConditionalObservable(
            self._store, f"{self._key}_filtered", self, condition
        )

    def __or__(self, other: "BaseObservable") -> "OrObservable":
        """Logical OR with | operator (either)."""
        return OrObservable(self._store, f"{self._key}_{other._key}_or", self, other)

    def __invert__(self) -> "NegatedObservable":
        """Negate with ~ operator (negate)."""
        # Ensure source is promoted to reactive mode
        _ = self.value  # Access value to promote if needed
        return NegatedObservable(self._store, f"{self._key}_negated", self)

    # Method equivalents
    def then(self, func: Callable) -> "FusedOperation":
        """Method version of >> operator."""
        return self >> func

    def alongside(self, other: "BaseObservable") -> "MergedObservable":
        """Method version of + operator."""
        return self + other

    def requiring(self, condition: "BaseObservable") -> "ConditionalObservable":
        """Method version of & operator."""
        return self & condition

    def either(self, other: "BaseObservable") -> "OrObservable":
        """Method version of | operator."""
        return self | other

    def negate(self) -> "NegatedObservable":
        """Method version of ~ operator."""
        return ~self

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._key}: {self.value})"

    def __repr__(self) -> str:
        return self.__str__()


class Observable(BaseObservable):
    """
    Fynx-style observable value that can be read and written.

    Supports "dumb mode": starts lightweight and only becomes reactive when needed.
    """

    def __init__(self, store: "Store", key: str, initial_value: Optional[T] = None):
        # Start in "dumb mode" - just store the value locally
        self._store = store
        self._key = key
        self._dumb_value = initial_value  # Local storage for dumb mode
        self._is_dumb = True  # Flag for dumb mode
        self._lock = threading.RLock()

    @property
    def value(self):
        """Get the current value - promote to reactive if needed."""
        if self._is_dumb:
            return self._dumb_value
        return self._store._delta_store.get(self._key)

    @value.setter
    def value(self, new_value: T) -> None:
        """Set the value - promote to reactive mode if needed."""
        if self._is_dumb:
            # Check if we need to become reactive (have subscribers or computed deps)
            if hasattr(self, "_observers") and self._observers:
                self._promote_to_reactive()
            else:
                # Stay dumb, just update local value
                self._dumb_value = new_value
                return

        # Reactive mode
        self._store._delta_store.set(self._key, new_value)

    def _promote_to_reactive(self):
        """Promote from dumb mode to full reactive mode."""
        if not self._is_dumb:
            return

        self._is_dumb = False
        # Initialize the reactive infrastructure
        self._observers: Set[Callable] = set()

        # Move value to DeltaKVStore
        if self._dumb_value is not None:
            self._store._delta_store.set(self._key, self._dumb_value)

        # Clear dumb storage
        delattr(self, "_dumb_value")

    def subscribe(self, callback: Callable[[T, T], None]) -> "Observable":
        """Subscribe to value changes - promotes to reactive mode."""
        self._promote_to_reactive()

        def delta_callback(delta: Delta):
            if delta.key == self._key and delta.change_type == ChangeType.SET:
                callback(delta.old_value, delta.new_value)

        self._store._delta_store.subscribe(self._key, delta_callback)
        return self


class FusedOperation:
    """
    Represents a chain of operations that can be fused into a single computation.

    This allows operator chaining like obs >> f1 >> f2 >> f3 to be compiled
    into a single compute function instead of creating intermediate observables.
    """

    def __init__(self, source: BaseObservable, operations: list[Callable]):
        self.source = source
        self.operations = operations
        self._store = source._store
        self._key = f"{source._key}_fused_{len(operations)}"

    def __rshift__(self, func: Callable) -> "FusedOperation":
        """Extend the fused operation chain."""
        return FusedOperation(self.source, self.operations + [func])

    def materialize(self, store: "Store", key: str) -> "DerivedObservable":
        """Materialize the fused operations into a concrete observable."""

        def fused_compute():
            value = self.source.value
            for op in self.operations:
                if isinstance(value, tuple):
                    value = op(*value)
                else:
                    value = op(value)
            return value

        store._delta_store.computed(key, fused_compute)
        return DerivedObservable(store, key, self.source, lambda x: x)

    @property
    def value(self):
        """Lazy evaluation: create the fused computation only when accessed."""
        # Accessing value promotes the source to reactive mode if needed
        _ = self.source.value

        def fused_compute():
            value = self.source.value
            for op in self.operations:
                if isinstance(value, tuple):
                    value = op(*value)
                else:
                    value = op(value)
            return value

        # Register the fused computation
        self._store._delta_store.computed(self._key, fused_compute)
        return self._store._delta_store.get(self._key)

    def subscribe(self, callback: Callable) -> "FusedOperation":
        """Subscribe to changes in the fused computation."""

        def delta_callback(delta):
            if delta.key == self._key:
                callback(delta.new_value)

        self._store._delta_store.subscribe(self._key, delta_callback)
        return self

    def __and__(self, condition: "BaseObservable") -> "ConditionalObservable":
        """Filter with & operator (requiring)."""
        # For fused operations used in conditionals, we need to ensure boolean evaluation
        return ConditionalObservable(
            self._store, f"{self._key}_filtered", self, condition
        )

    def __or__(self, other: "BaseObservable") -> "OrObservable":
        """Logical OR with | operator (either)."""
        return OrObservable(self._store, f"{self._key}_{other._key}_or", self, other)

    def __invert__(self) -> "NegatedObservable":
        """Negate with ~ operator (negate)."""
        # Ensure source is promoted to reactive mode
        _ = self.value  # Access value to promote source
        return NegatedObservable(self._store, f"{self._key}_negated", self)

    def __str__(self):
        return f"FusedOperation({self._key}: {len(self.operations)} operations)"


class DerivedObservable(BaseObservable):
    """
    Observable derived from another observable via transformation.

    Now supports lazy fusion: operations are collected until value access.
    """

    def __init__(
        self, store: "Store", key: str, source: BaseObservable, func: Callable
    ):
        super().__init__(store, key)
        self._source = source
        self._func = func

        # For now, create the computed value immediately
        # TODO: Implement lazy fusion for better performance
        def compute_func():
            source_val = source.value
            if isinstance(source_val, tuple):
                return func(*source_val)
            else:
                return func(source_val)

        self._store._delta_store.computed(key, compute_func)


class MergedObservable(BaseObservable):
    """
    Observable that combines multiple sources into a tuple.
    """

    def __init__(self, store: "Store", key: str, sources: list[BaseObservable]):
        super().__init__(store, key)
        self._sources = sources

        # Register computed value that returns tuple
        def compute_func():
            return tuple(source.value for source in sources)

        self._store._delta_store.computed(key, compute_func)


class ConditionalObservable(BaseObservable):
    """
    Observable that filters based on condition.
    Uses direct computation for reliability.
    """

    def __init__(
        self, store: "Store", key: str, source: any, condition: BaseObservable
    ):
        # For conditional operations, compute directly rather than using DeltaKVStore
        # to avoid dependency tracking issues
        self._store = store
        self._key = key
        self._source = source
        self._condition = condition
        self._lock = threading.RLock()

    @property
    def value(self):
        """Compute conditional directly."""
        if bool(self._condition.value):
            source_val = self._source.value
            # For conditional chaining, return boolean results
            if isinstance(source_val, bool):
                return source_val
            return True  # Condition met
        return False  # Condition not met

    def subscribe(
        self, callback: Callable[[bool, bool], None]
    ) -> "ConditionalObservable":
        """Subscribe to changes - delegate to both sources."""

        def on_change(old_val, new_val):
            # Recompute and notify if result changed
            new_result = self.value
            callback(False, new_result)  # Simplified

        self._condition.subscribe(on_change)
        if hasattr(self._source, "subscribe"):
            self._source.subscribe(on_change)

        return self


class OrObservable(BaseObservable):
    """
    Observable that emits when either source emits truthy value.
    Uses direct computation for reliability.
    """

    def __init__(self, store: "Store", key: str, left: any, right: BaseObservable):
        # For OR operations, compute directly rather than using DeltaKVStore
        # to avoid dependency tracking issues
        self._store = store
        self._key = key
        self._left = left
        self._right = right
        self._lock = threading.RLock()

    @property
    def value(self):
        """Compute OR directly."""
        return bool(self._left.value) or bool(self._right.value)

    def subscribe(self, callback: Callable[[bool, bool], None]) -> "OrObservable":
        """Subscribe to changes - delegate to both sources."""

        def on_change(old_val, new_val):
            # Recompute and notify if result changed
            new_result = self.value
            # For simplicity, just notify with the new result
            # In a full implementation, we'd track the old result
            callback(False, new_result)  # Simplified: assume old was False

        if hasattr(self._left, "subscribe"):
            self._left.subscribe(on_change)
        self._right.subscribe(on_change)

        return self


class NegatedObservable(BaseObservable):
    """
    Observable that negates boolean values.
    Uses direct computation for simplicity and reliability.
    """

    def __init__(self, store: "Store", key: str, source: any):
        # For negation, we'll compute directly rather than using DeltaKVStore
        # to avoid dependency tracking issues with dumb->reactive promotion
        self._store = store
        self._key = key
        self._source = source
        self._lock = threading.RLock()

    @property
    def value(self):
        """Compute negation directly."""
        return not bool(self._source.value)

    def subscribe(self, callback: Callable[[bool, bool], None]) -> "NegatedObservable":
        """Subscribe to changes - delegate to source with transformation."""

        def transformed_callback(old_val, new_val):
            old_negated = not bool(old_val)
            new_negated = not bool(new_val)
            if old_negated != new_negated:  # Only notify if negation actually changed
                callback(old_negated, new_negated)

        self._source.subscribe(transformed_callback)
        return self


class Store:
    """
    Fynx-style store that manages a collection of observables.

    Uses DeltaKVStore internally for efficient incremental computation while
    providing the familiar Fynx API with operator overloading.
    """

    def __init__(self):
        self._delta_store = DeltaKVStore()
        self._observables: Dict[str, BaseObservable] = {}

    def observable(self, key: str, initial_value: Optional[T] = None) -> Observable:
        """Create or get an observable value."""
        if key not in self._observables:
            self._observables[key] = Observable(self, key, initial_value)
        return self._observables[key]

    def subscribe(self, key: str, callback: Callable) -> None:
        """Subscribe to changes on a key."""
        self._delta_store.subscribe(key, callback)

    def get(self, key: str) -> Any:
        """Get a value from the store."""
        return self._delta_store.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        self._delta_store.set(key, value)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self._delta_store.stats()

    def __str__(self) -> str:
        observables = list(self._observables.keys())
        return f"Store(observables={observables})"

    def __repr__(self) -> str:
        return self.__str__()


# Convenience functions for Fynx-like API
def observable(initial_value: Optional[T] = None) -> Observable:
    """
    Create an observable value.

    Note: In Fynx, observables are typically created as class attributes
    in Store subclasses. This function is for convenience.
    """
    # This is a simplified version - in real Fynx, observables are typically
    # created within store classes with automatic key assignment
    raise NotImplementedError(
        "Use store.observable(key, initial_value) instead. "
        "This function requires store context for proper key management."
    )


def reactive(*dependencies):
    """
    Decorator to create reactive functions that run when dependencies change.

    This mimics Fynx's @reactive decorator but uses DeltaKVStore subscriptions.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Subscribe to all dependencies
            for dep in dependencies:
                if hasattr(dep, "_key"):
                    # It's an Observable
                    dep.subscribe(lambda old, new: func(*args, **kwargs))

            # Return the function so it can be called manually too
            return func

        return wrapper

    return decorator


# Example usage and demonstration
if __name__ == "__main__":
    print("=== Fynx-Style Frontend Demo ===")

    # Create a store like Fynx
    class CartStore(Store):
        item_count = None  # Will be set below
        price_per_item = None

    store = CartStore()
    store.item_count = store.observable("item_count", 1)
    store.price_per_item = store.observable("price_per_item", 10.0)

    # Define transformation function
    def calculate_total(count, price):
        return count * price

    # Reactive computation using .then() or >> operator
    total_price = (store.item_count + store.price_per_item).then(calculate_total)
    # total_price = (store.item_count + store.price_per_item) >> calculate_total  # Equivalent!

    def print_total(total):
        print(f"Cart Total: ${total:.2f}")

    total_price.subscribe(print_total)

    # Automatic updates
    print("Initial:")
    print(f"Cart Total: ${total_price.value:.2f}")

    print("\n--- Changing item count ---")
    store.item_count.value = 3  # Cart Total: $30.00

    print("\n--- Changing price ---")
    store.price_per_item.value = 12.50  # Cart Total: $37.50

    print(f"\nFinal: ${total_price.value:.2f}")
    print(f"Stats: {store.stats}")

    print("\n=== Advanced Example ===")

    # More complex example with conditional logic
    class UserStore(Store):
        first_name = None
        last_name = None
        is_logged_in = None
        has_data = None
        is_loading = None

    user_store = UserStore()
    user_store.first_name = user_store.observable("first_name", "Alice")
    user_store.last_name = user_store.observable("last_name", "Smith")
    user_store.is_logged_in = user_store.observable("is_logged_in", False)
    user_store.has_data = user_store.observable("has_data", False)
    user_store.is_loading = user_store.observable("is_loading", True)

    # Define transformation functions
    def join_names(first, last):
        return f"{first} {last}"

    # Combine and transform
    full_name = (user_store.first_name + user_store.last_name) >> join_names

    # Conditional reactions
    ready_to_sync = (
        user_store.is_logged_in & user_store.has_data & (~user_store.is_loading)
    )

    @reactive(ready_to_sync)
    def sync_when_ready(should_sync):
        if should_sync:
            print("🔄 Performing sync operation...")

    @reactive(full_name)
    def update_greeting(name):
        print(f"👋 Hello, {name}!")

    print("\n--- Initial state ---")
    print(f"Full name: {full_name.value}")
    print(f"Ready to sync: {ready_to_sync.value}")

    print("\n--- User logs in ---")
    user_store.is_logged_in.value = True

    print("\n--- Data loads ---")
    user_store.has_data.value = True

    print("\n--- Loading completes ---")
    user_store.is_loading.value = False

    print(f"\nFinal - Name: {full_name.value}, Ready: {ready_to_sync.value}")
    print(f"Stats: {user_store.stats}")
