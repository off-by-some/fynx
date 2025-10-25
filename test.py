"""
Observable - FynX-Style Reactive API over DeltaKVStore
======================================================

A user-friendly reactive programming API that maintains O(affected) complexity
by building on top of the DeltaKVStore engine.

Design Philosophy:
- Hide key management completely (automatic internal key generation)
- Value semantics: observables feel like regular values
- Composable operators that create new observables
- Lazy evaluation: computations only happen when needed
- O(affected): only touched nodes recompute

All operators (>>, +, &, |, ~) maintain the O(affected) guarantee because they
delegate to DeltaKVStore's computed values with explicit dependency tracking.
"""

import weakref
from enum import Enum
from typing import Any, Callable, List, Optional, Set, Tuple

from fynx.delta_kv_store import DeltaKVStore  # Your store

# =============================================================================
# Observable - The User-Facing Reactive Value
# =============================================================================


class Observable:
    """
    A reactive value that automatically propagates changes.

    Observables are the user-facing API over DeltaKVStore. They hide complexity
    while maintaining O(affected) performance characteristics.

    Key Features:
    - Property-style access: obs.value = 5
    - Operator overloading: obs1 + obs2, obs >> func
    - Automatic dependency tracking
    - Lazy evaluation (pull-based)

    Design Invariants:
    - Each Observable has a unique internal key in the store
    - Computed observables are read-only (set() raises error)
    - All operations return new Observables (immutable chains)
    - Dependencies are tracked automatically or explicitly
    """

    # Class-level store instance (singleton pattern)
    _global_store = None
    _instance_counter = 0  # For generating unique keys

    def __init__(
        self,
        initial_value: Any = None,
        store=None,
        _internal_key: Optional[str] = None,
        _is_computed: bool = False,
    ):
        """
        Create a new Observable.

        Args:
            initial_value: The starting value
            store: Optional DeltaKVStore instance (uses global if None)
            _internal_key: Internal use only - the store key
            _is_computed: Internal use only - whether this is a computed value

        Users should create observables via:
            obs = Observable(42)           # Regular observable
            obs = price >> (lambda p: p*2) # Computed observable (via operator)
        """
        # Get or create global store
        if Observable._global_store is None:
            Observable._global_store = DeltaKVStore()

        self._store = store or Observable._global_store
        self._is_computed = _is_computed

        # Generate unique key if not provided
        if _internal_key is None:
            Observable._instance_counter += 1
            self._key = f"obs_{Observable._instance_counter}"
        else:
            self._key = _internal_key

        # Initialize value in store (if not computed)
        if not _is_computed:
            self._store.set(self._key, initial_value)

        # Track subscriptions for cleanup
        self._subscriptions: Set[Callable] = set()

    # =========================================================================
    # Value Access - Property-Style Interface
    # =========================================================================

    @property
    def value(self) -> Any:
        """
        Get the current value.

        For computed observables, this triggers recomputation if dirty.
        O(1) if clean, O(computation) if dirty.
        """
        return self._store.read(self._key)

    @value.setter
    def value(self, new_value: Any) -> None:
        """
        Set the value and trigger reactivity.

        Raises ValueError if this is a computed observable (read-only).
        O(affected) for propagation.
        """
        if self._is_computed:
            raise ValueError(
                "Cannot set value of computed observable. "
                "Computed observables are read-only and derive their value from dependencies."
            )
        self._store.set(self._key, new_value)

    def get(self) -> Any:
        """Alternative getter (functional style)."""
        return self.value

    def set(self, new_value: Any) -> None:
        """Alternative setter (functional style)."""
        self.value = new_value

    # =========================================================================
    # Operator Overloading - The Core Reactive Algebra
    # =========================================================================

    def __rshift__(self, func: Callable) -> "Observable":
        """
        Transform operator: obs >> func

        Creates a computed observable that applies func to this observable's value.

        Example:
            price = Observable(10.0)
            formatted = price >> (lambda p: f"${p:.2f}")
            # formatted.value == "$10.00"

            price.value = 20.0
            # formatted.value == "$20.00" (automatically updates)

        Special handling for tuples (from + operator):
            combined = (first + last) >> (lambda f, l: f"{f} {l}")
            # The tuple (first, last) is unpacked as arguments

        Complexity: O(1) to create, O(affected) when dependencies change
        """
        result = Observable(
            initial_value=None,
            store=self._store,
            _internal_key=f"{self._key}_then_{id(func)}",
            _is_computed=True,
        )

        # Define computation with explicit dependency
        def compute():
            value = self.get()
            # Auto-unpack tuples (from + operator) as function arguments
            if isinstance(value, tuple):
                return func(*value)
            return func(value)

        # Register as computed value with explicit deps (more efficient)
        self._store.computed(result._key, compute, deps=[self._key])

        return result

    def __add__(self, other: "Observable") -> "Observable":
        """
        Combine operator: obs1 + obs2

        Creates a computed observable containing a tuple of both values.
        The result is READ-ONLY - you cannot set merged observables.

        Example:
            first = Observable("John")
            last = Observable("Doe")
            full = (first + last) >> (lambda f, l: f"{f} {l}")
            # full.value == "John Doe"

            first.value = "Jane"
            # full.value == "Jane Doe" (automatically updates)

        Mathematical Note: This constructs a categorical product.

        Complexity: O(1) to create, O(affected) when either dependency changes
        """
        result = Observable(
            initial_value=None,
            store=self._store,
            _internal_key=f"{self._key}_plus_{other._key}",
            _is_computed=True,
        )

        # Define computation that produces tuple
        def compute():
            return (self.get(), other.get())

        # Register with both dependencies
        self._store.computed(result._key, compute, deps=[self._key, other._key])

        return result

    def __and__(self, condition: "Observable") -> "Observable":
        """
        Filter operator: obs & condition

        Creates a computed observable that only updates when condition is truthy.
        When condition is falsy, returns the last valid value (or None).

        Example:
            file = Observable(None)
            is_valid = file >> (lambda f: f is not None)
            valid_file = file & is_valid

            file.value = "data.csv"  # valid_file updates
            file.value = None        # valid_file keeps "data.csv"

        Stack multiple conditions:
            ready = file & is_valid & (~is_processing)

        Mathematical Note: This constructs a categorical pullback (fiber).

        Complexity: O(1) to create, O(affected) when dependencies change
        """
        result = Observable(
            initial_value=None,
            store=self._store,
            _internal_key=f"{self._key}_and_{condition._key}",
            _is_computed=True,
        )

        # Track last valid value (closure)
        last_valid = [None]

        def compute():
            cond_value = condition.get()
            if cond_value:
                current = self.get()
                last_valid[0] = current
                return current
            return last_valid[0]  # Return last valid when condition false

        # Register with both dependencies
        self._store.computed(result._key, compute, deps=[self._key, condition._key])

        return result

    def __or__(self, other: "Observable") -> "Observable":
        """
        Logical OR operator: obs1 | obs2

        Creates a computed observable that is True if either observable is truthy.

        Example:
            is_error = Observable(False)
            is_warning = Observable(True)
            needs_attention = is_error | is_warning
            # needs_attention.value == True

        Complexity: O(1) to create, O(affected) when either dependency changes
        """
        result = Observable(
            initial_value=None,
            store=self._store,
            _internal_key=f"{self._key}_or_{other._key}",
            _is_computed=True,
        )

        def compute():
            return self.get() or other.get()

        self._store.computed(result._key, compute, deps=[self._key, other._key])

        return result

    def __invert__(self) -> "Observable":
        """
        Negate operator: ~obs

        Creates a computed observable with inverted boolean value.

        Example:
            is_loading = Observable(True)
            is_ready = ~is_loading
            # is_ready.value == False

            is_loading.value = False
            # is_ready.value == True

        Complexity: O(1) to create, O(affected) when dependency changes
        """
        result = Observable(
            initial_value=None,
            store=self._store,
            _internal_key=f"not_{self._key}",
            _is_computed=True,
        )

        def compute():
            return not self.get()

        self._store.computed(result._key, compute, deps=[self._key])

        return result

    # =========================================================================
    # Method Aliases - Natural Language Interface
    # =========================================================================

    def then(self, func: Callable) -> "Observable":
        """
        Natural language alias for >> operator.

        Example:
            result = observable.then(lambda x: x * 2)
            # Equivalent to: result = observable >> (lambda x: x * 2)
        """
        return self >> func

    def alongside(self, other: "Observable") -> "Observable":
        """
        Natural language alias for + operator.

        Example:
            combined = first.alongside(last)
            # Equivalent to: combined = first + last
        """
        return self + other

    def requiring(self, condition: "Observable") -> "Observable":
        """
        Natural language alias for & operator.

        Example:
            filtered = data.requiring(is_valid)
            # Equivalent to: filtered = data & is_valid
        """
        return self & condition

    def either(self, other: "Observable") -> "Observable":
        """
        Natural language alias for | operator.

        Example:
            alert = error.either(warning)
            # Equivalent to: alert = error | warning
        """
        return self | other

    def negate(self) -> "Observable":
        """
        Natural language alias for ~ operator.

        Example:
            not_loading = is_loading.negate()
            # Equivalent to: not_loading = ~is_loading
        """
        return ~self

    # =========================================================================
    # Subscription System - React to Changes
    # =========================================================================

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        """
        Subscribe to value changes.

        Args:
            callback: Function called with new value when observable changes

        Returns:
            Unsubscribe function to stop receiving updates

        Example:
            counter = Observable(0)

            def on_change(value):
                print(f"Counter: {value}")

            unsubscribe = counter.subscribe(on_change)
            counter.value = 5  # Prints: "Counter: 5"

            unsubscribe()  # Stop receiving updates
            counter.value = 10  # Nothing printed

        Note: Callbacks receive only the NEW value, not the delta.
        For full delta info, use store.subscribe() directly.

        Complexity: O(1) to subscribe, O(1) per callback on changes
        """

        # Wrapper to extract just the value from delta
        def wrapper(delta):
            callback(delta.new_value)

        # Subscribe to store and track for cleanup
        unsubscribe = self._store.subscribe(self._key, wrapper)
        self._subscriptions.add(unsubscribe)

        # Return unsubscribe function
        def unsub():
            unsubscribe()
            self._subscriptions.discard(unsubscribe)

        return unsub

    def unsubscribe_all(self) -> None:
        """
        Remove all subscriptions on this observable.

        Useful for cleanup when destroying reactive components.
        """
        for unsubscribe in list(self._subscriptions):
            unsubscribe()
        self._subscriptions.clear()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        value_str = repr(self.value)
        computed_str = " (computed)" if self._is_computed else ""
        return f"Observable({value_str}{computed_str})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return str(self.value)

    def __bool__(self) -> bool:
        """
        Allow observables to be used in boolean contexts.

        Example:
            if obs:  # Checks truthiness of obs.value
                do_something()
        """
        return bool(self.value)

    def __eq__(self, other) -> bool:
        """
        Compare observable values.

        Note: Compares VALUES, not identity.
        """
        if isinstance(other, Observable):
            return self.value == other.value
        return self.value == other

    def __hash__(self) -> int:
        """Make observables hashable by their key."""
        return hash(self._key)


# =============================================================================
# Reactive Decorator - Side Effects
# =============================================================================


def reactive(*observables: Observable):
    """
    Decorator that runs a function when observables change.

    IMPORTANT: Use @reactive for SIDE EFFECTS ONLY:
    - UI updates
    - Logging
    - Network calls
    - File I/O

    For data transformations, use >> operator instead:
        derived = source >> transform  # GOOD: Pure transformation

    Example:
        counter = Observable(0)

        @reactive(counter)
        def update_ui(value):
            render(f"Count: {value}")  # Side effect: UI update

        counter.value = 5  # Triggers update_ui(5)

    Note: Reactive functions don't fire immediately when created - only
    when their dependencies CHANGE. If you need initialization, handle
    it separately before setting up the reaction.

    Lifecycle:
        func.unsubscribe()  # Stop reacting
        func()              # Can call manually after unsubscribing

    Complexity: O(1) per observer callback on changes
    """

    def decorator(func: Callable) -> Callable:
        # Track unsubscribe functions
        unsubscribers = []

        # Create callback that calls func with current values
        def on_change(*_args):
            values = [obs.get() for obs in observables]
            func(*values)

        # Subscribe to all observables
        for obs in observables:
            unsub = obs.subscribe(lambda _: on_change())
            unsubscribers.append(unsub)

        # Add unsubscribe method to function
        def unsubscribe():
            for unsub in unsubscribers:
                unsub()
            unsubscribers.clear()

        func.unsubscribe = unsubscribe

        return func

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================


def observable(initial_value: Any = None, store=None) -> Observable:
    """
    Factory function for creating observables.

    Equivalent to Observable(initial_value) but reads more naturally
    in some contexts.

    Example:
        count = observable(0)
        name = observable("Alice")
    """
    return Observable(initial_value, store=store)


def computed(func: Callable, *dependencies: Observable, store=None) -> Observable:
    """
    Create a computed observable from explicit dependencies.

    Example:
        x = observable(10)
        y = observable(20)

        # Using computed() function
        sum_xy = computed(lambda: x.get() + y.get(), x, y)

        # Using >> operator (preferred)
        sum_xy = (x + y) >> (lambda xy: xy[0] + xy[1])

    The >> operator is usually more readable, but computed() can be
    useful when you need explicit control over dependency tracking.
    """
    if store is None:
        store = Observable._global_store

    result = Observable(
        initial_value=None,
        store=store,
        _internal_key=f"computed_{id(func)}",
        _is_computed=True,
    )

    dep_keys = [dep._key for dep in dependencies]
    store.computed(result._key, func, deps=dep_keys)

    return result


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    print("=== Observable Demo ===\n")

    # Basic observable
    print("1. Basic Observable:")
    counter = Observable(0)
    print(f"   Initial: {counter.value}")
    counter.value = 5
    print(f"   After set: {counter.value}")

    # Transform with >>
    print("\n2. Transform with >>:")
    doubled = counter >> (lambda x: x * 2)
    print(f"   counter={counter.value}, doubled={doubled.value}")
    counter.value = 10
    print(f"   After update: counter={counter.value}, doubled={doubled.value}")

    # Combine with +
    print("\n3. Combine with +:")
    first = Observable("John")
    last = Observable("Doe")
    full_name = (first + last) >> (lambda f, l: f"{f} {l}")
    print(f"   full_name={full_name.value}")
    first.value = "Jane"
    print(f"   After update: full_name={full_name.value}")

    # Filter with &
    print("\n4. Filter with &:")
    value = Observable(10)
    is_positive = value >> (lambda x: x > 0)
    positive_value = value & is_positive
    print(f"   value={value.value}, positive_value={positive_value.value}")
    value.value = -5
    print(
        f"   After negative: value={value.value}, positive_value={positive_value.value}"
    )
    value.value = 20
    print(
        f"   After positive: value={value.value}, positive_value={positive_value.value}"
    )

    # Logical operators
    print("\n5. Logical operators:")
    is_error = Observable(False)
    is_warning = Observable(True)
    needs_attention = is_error | is_warning
    print(f"   needs_attention={needs_attention.value}")
    is_loading = Observable(True)
    is_ready = ~is_loading
    print(f"   is_ready={is_ready.value}")

    # Subscriptions
    print("\n6. Subscriptions:")
    count = Observable(0)

    def on_count_change(value):
        print(f"   Count changed to: {value}")

    unsub = count.subscribe(on_count_change)
    count.value = 1  # Triggers callback
    count.value = 2  # Triggers callback
    unsub()
    count.value = 3  # No callback (unsubscribed)

    # @reactive decorator
    print("\n7. @reactive decorator:")
    score = Observable(0)

    @reactive(score)
    def display_score(value):
        print(f"   Score: {value}")

    score.value = 100  # Triggers display_score
    score.value = 200  # Triggers display_score

    print("\n=== Demo Complete ===")
