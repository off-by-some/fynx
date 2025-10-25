"""
FynX Store - Reactive State Management Components
=================================================

This module provides the core components for reactive state management in FynX,
enabling you to create organized, reactive state containers that group related
observables together with convenient subscription and state management methods.

Why Use Stores?
---------------

Stores help you organize your application's reactive state into logical units. Instead
of having observables scattered throughout your codebase, Stores group related data
together and provide convenient methods for subscribing to changes, serializing state,
and managing the reactive lifecycle.

Stores are particularly useful for:
- **Application State**: Global app state like user preferences, theme settings
- **Feature State**: State for specific features like shopping cart, user profile
- **Component State**: Local state that needs to be shared across multiple components
- **Business Logic**: Computed values and derived state based on raw data

Core Components
---------------

**Store**: A base class for creating reactive state containers. Store classes can define
observable attributes using the `observable()` descriptor, and automatically provide
methods for subscribing to changes and managing state.

**observable**: A descriptor function that creates observable attributes on Store classes.
Use this to define reactive properties in your Store subclasses.

**StoreSnapshot**: An immutable snapshot of store state at a specific point in time,
useful for debugging, logging, and ensuring consistent state access.

**StoreMeta**: A metaclass that automatically converts observable attributes to descriptors
and provides type hint compatibility for mypy.

Key Features
------------

- **Automatic Observable Management**: Store metaclass handles observable creation
- **Convenient Subscriptions**: Subscribe to all changes or individual observables
- **State Serialization**: Save and restore store state with `to_dict()` and `load_state()`
- **Type Safety**: Full type hint support for better IDE experience
- **Memory Efficient**: Automatic cleanup and efficient change detection
- **Composable**: Easy to combine and nest multiple stores

Basic Usage
-----------

```python
from fynx import Store, observable

class CounterStore(Store):
    count = observable(0)
    name = observable("My Counter")

# Access values like regular attributes
print(CounterStore.count)  # 0
CounterStore.count = 5     # Updates the observable

# Subscribe to all changes in the store
@CounterStore.subscribe
def on_store_change(snapshot):
    print(f"Store changed: count={snapshot.count}, name={snapshot.name}")

CounterStore.count = 10  # Triggers: "Store changed: count=10, name=My Counter"
```

Advanced Patterns
-----------------

### Computed Properties in Stores

```python
from fynx import Store, observable

class UserStore(Store):
    first_name = observable("John")
    last_name = observable("Doe")
    age = observable(30)

    # Computed properties using the >> operator
    full_name = (first_name + last_name) >> (
        lambda fname, lname: f"{fname} {lname}"
    )

    is_adult = age >> (lambda a: a >= 18)

print(UserStore.full_name)  # "John Doe"
UserStore.first_name = "Jane"
print(UserStore.full_name)  # "Jane Doe" (automatically updated)
```

### State Persistence

```python
# Save store state
state = CounterStore.to_dict()
# state = {"count": 10, "name": "My Counter"}

# Restore state later
CounterStore.load_state(state)
print(CounterStore.count)  # 10
```

### Store Composition

```python
class AppStore(Store):
    theme = observable("light")
    language = observable("en")

class UserStore(Store):
    name = observable("Alice")
    preferences = observable({})

# Use both stores independently
AppStore.theme = "dark"
UserStore.name = "Bob"
```

Common Patterns
---------------

**Singleton Stores**: Use class-level access for global state:

```python
class GlobalStore(Store):
    is_loading = observable(False)
    current_user = observable(None)

# Access globally
GlobalStore.is_loading = True
```


```

See Also
--------

- `fynx.observable`: Core observable classes and operators
- `fynx.computed`: Creating computed properties
- `fynx.reactive`: Reactive decorators for side effects
- `fynx.watch`: Conditional reactive functions
"""

import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from .observable import BaseObservable, Observable

T = TypeVar("T")

# Create a simple SubscriptableDescriptor for type hints
from typing import Generic


class SubscriptableDescriptor(Generic[T]):
    """Descriptor for subscriptable observables."""

    pass


# Type alias for session state values (used for serialization)
SessionValue = Union[
    None, str, int, float, bool, Dict[str, "SessionValue"], List["SessionValue"]
]


class StoreSnapshot:
    """
    Immutable snapshot of store observable values at a specific point in time.
    """

    def __init__(self, store_class: Type, observable_attrs: List[str]):
        self._store_class = store_class
        self._observable_attrs = observable_attrs
        self._snapshot_values: Dict[str, SessionValue] = {}
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        """Capture current values of all observable attributes."""
        for attr_name in self._observable_attrs:
            if attr_name in self._store_class._observables:
                observable = self._store_class._observables[attr_name]
                self._snapshot_values[attr_name] = observable.value
            else:
                # For attributes that exist in the class but aren't observables,
                # get their value directly from the class
                try:
                    self._snapshot_values[attr_name] = getattr(
                        self._store_class, attr_name
                    )
                except AttributeError:
                    # If attribute doesn't exist at all, store None
                    self._snapshot_values[attr_name] = None

    def __getattr__(self, name: str) -> Any:
        """Access snapshot values or fall back to class attributes."""
        if name in self._snapshot_values:
            return self._snapshot_values[name]
        return getattr(self._store_class, name)

    def __repr__(self) -> str:
        if not self._snapshot_values:
            return "StoreSnapshot()"
        fields = [
            f"{name}={self._snapshot_values[name]!r}"
            for name in self._observable_attrs
            if name in self._snapshot_values
        ]
        return f"StoreSnapshot({', '.join(fields)})"


class Store:
    """
    Base class for reactive state containers.
    Each Store instance has its own DeltaKVStore to avoid lock contention.
    """

    def __init__(self):
        from fynx.observable.core.observable import DeltaKVStore

        self._store = DeltaKVStore()

    def observable(self, key: str, initial_value: Optional[T] = None):
        """Create an observable in this store's DeltaKVStore."""
        if initial_value is not None:
            self._store.set(key, initial_value)

        class StoreObservable:
            def __init__(self, store, key):
                self._store = store
                self._key = key

            def set(self, value):
                self._store.set(self._key, value)
                return self

            def get(self):
                return self._store.get(self._key)

            @property
            def value(self):
                return self.get()

            def subscribe(self, callback):
                def delta_callback(delta):
                    if delta.key == self._key:
                        callback(delta.new_value)

                return self._store.subscribe(self._key, delta_callback)

            def then(self, func):
                """Create a computed observable with dependency tracking."""
                computed_key = f"{self._key}_then_{id(func)}"

                def compute_func():
                    current_value = self._store.get(self._key)
                    return func(current_value)

                self._store.computed(computed_key, compute_func)
                _ = self._store.get(computed_key)  # Force evaluation

                return StoreObservable(self._store, computed_key)

            def __rshift__(self, func):
                return self.then(func)

            def __str__(self):
                return f"StoreObservable({self._key}: {self.value})"

            def __repr__(self):
                return self.__str__()

        return StoreObservable(self._store, key)


def observable(initial_value: Optional[T] = None) -> Any:
    """
    Create an observable with an initial value, used as a descriptor in Store classes.
    Uses direct DeltaKVStore integration for maximum performance.
    """
    # Import here to avoid circular imports
    from fynx.observable.core.observable import ComputedValue, _global_store

    # Create a direct observable that bypasses BaseObservable overhead
    key = f"obs_{id(initial_value) if initial_value is not None else time.time()}"

    # Set initial value if provided
    if initial_value is not None:
        _global_store.set(key, initial_value)

    # Return a lightweight wrapper that uses the global store directly
    class FastObservable:
        def __init__(self, key: str):
            self._key = key  # Cache the key to avoid regenerating
            self._store = _global_store

        def set(self, value):
            self._store.set(self._key, value)
            return self

        def get(self):
            return self._store.get(self._key)

        @property
        def value(self):
            return self.get()

        def subscribe(self, callback):
            def delta_callback(delta):
                if delta.key == self._key:
                    callback(delta.new_value)

            return self._store.subscribe(self._key, delta_callback)

        def then(self, func):
            """Create a computed observable with manual dependency tracking."""
            computed_key = f"computed_{id(self)}_{id(func)}"

            # Create a proper computed value that tracks dependencies
            def compute_func():
                # Access the source value to create dependency
                current_value = self._store.get(self._key)
                return func(current_value)

            # Use a simpler approach - create the computed value directly
            # and manually track the dependency
            self._store._computed[computed_key] = ComputedValue(
                computed_key, compute_func, self._store
            )
            self._store._dep_graph.add_dependency(computed_key, self._key)

            # Force evaluation to ensure dependencies are tracked
            _ = self._store.get(computed_key)

            # Return a wrapper for the computed value
            return FastObservable(computed_key)

        def __rshift__(self, func):
            """Transform with >> operator."""
            return self.then(func)

        def __add__(self, other):
            """Combine with + operator."""
            merged_key = f"merged_{id(self)}_{id(other)}"

            def compute_func():
                val1 = self._store.get(self._key)
                val2 = (
                    other._store.get(other._key)
                    if hasattr(other, "_store")
                    else other.value
                )
                return val1 + val2

            self._store.computed(merged_key, compute_func)
            return FastObservable(merged_key)

        def __and__(self, condition):
            """Filter with & operator (requiring)."""
            filtered_key = f"filtered_{id(self)}_{id(condition)}"

            def compute_func():
                val = self._store.get(self._key)
                cond_val = (
                    condition._store.get(condition._key)
                    if hasattr(condition, "_store")
                    else condition.value
                )
                return val if bool(cond_val) else False

            self._store.computed(filtered_key, compute_func)
            return FastObservable(filtered_key)

        def __or__(self, other):
            """Logical OR with | operator (either)."""
            or_key = f"or_{id(self)}_{id(other)}"

            def compute_func():
                val1 = self._store.get(self._key)
                val2 = (
                    other._store.get(other._key)
                    if hasattr(other, "_store")
                    else other.value
                )
                return bool(val1) or bool(val2)

            self._store.computed(or_key, compute_func)
            return FastObservable(or_key)

        def __invert__(self):
            """Negate with ~ operator (negate)."""
            negated_key = f"negated_{id(self)}"

            def compute_func():
                val = self._store.get(self._key)
                return not bool(val)

            self._store.computed(negated_key, compute_func)
            return FastObservable(negated_key)

        # Method equivalents
        def alongside(self, other):
            """Method version of + operator."""
            return self + other

        def requiring(self, condition):
            """Method version of & operator."""
            return self & condition

        def either(self, other):
            """Method version of | operator."""
            return self | other

        def negate(self):
            """Method version of ~ operator."""
            return ~self

        def __str__(self):
            return f"FastObservable({self._key}: {self.value})"

        def __repr__(self):
            return self.__str__()

    return FastObservable(key)


# Type alias for subscriptable observables (class variables)
Subscriptable = SubscriptableDescriptor[Optional[T]]


class StoreMeta(type):
    """
    Metaclass for Store to automatically convert observable attributes to descriptors
    and adjust type hints for mypy compatibility.
    """

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Type:
        # Process annotations and replace observable instances with descriptors
        annotations = namespace.get("__annotations__", {})
        new_namespace = namespace.copy()
        observable_attrs = []

        # First, collect inherited observable attributes that need descriptors
        inherited_observables = {}
        for base in bases:
            if hasattr(base, "_observable_attrs"):
                base_attrs = getattr(base, "_observable_attrs", [])
                for attr_name in base_attrs:
                    if (
                        attr_name not in namespace
                        and hasattr(base, "__dict__")
                        and attr_name in base.__dict__
                    ):
                        # This inherited attribute needs a descriptor in the child class
                        base_descriptor = base.__dict__[attr_name]
                        if isinstance(base_descriptor, SubscriptableDescriptor):
                            inherited_observables[attr_name] = base_descriptor

        # Create descriptors for inherited observables
        for attr_name, base_descriptor in inherited_observables.items():
            new_namespace[attr_name] = SubscriptableDescriptor(
                initial_value=base_descriptor._initial_value,
                original_observable=None,  # Don't share original observable
            )

        # Process directly defined observables
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, BaseObservable):
                observable_attrs.append(attr_name)
                # Wrap all observables (including computed ones) in descriptors
                initial_value = attr_value.value
                new_namespace[attr_name] = SubscriptableDescriptor(
                    initial_value=initial_value, original_observable=attr_value
                )

        # Add inherited observables to the list
        observable_attrs.extend(inherited_observables.keys())

        new_namespace["__annotations__"] = annotations
        cls = super().__new__(mcs, name, bases, new_namespace)

        # Cache observable attributes and their instances for efficient access
        cls._observable_attrs = list(observable_attrs)
        # Store the original observables from the namespace before they get replaced
        cls._observables = {
            attr: namespace[attr] for attr in observable_attrs if attr in namespace
        }

        return cls

    def __setattr__(cls, name: str, value: Any) -> None:
        """Intercept class attribute assignment for observables."""
        if hasattr(cls, "_observables") and name in getattr(cls, "_observables", {}):
            # It's a known observable, delegate to its set method
            getattr(cls, "_observables")[name].set(value)
        else:
            super().__setattr__(name, value)


class Store(metaclass=StoreMeta):
    """
    Base class for reactive state containers with observable attributes.

    Store provides a convenient way to group related observable values together
    and manage their lifecycle as a cohesive unit. Store subclasses can define
    observable attributes using the `observable()` descriptor, and Store provides
    methods for subscribing to changes, serializing state, and managing the
    reactive relationships.

    Key Features:
    - Automatic observable attribute detection and management
    - Convenient subscription methods for reacting to state changes
    - Serialization/deserialization support for persistence
    - Snapshot functionality for debugging and state inspection

    Example:
        ```python
        from fynx import Store, observable

        class CounterStore(Store):
            count = observable(0)
            name = observable("Counter")

        # Subscribe to all changes
        @CounterStore.subscribe
        def on_change(snapshot):
            print(f"Counter: {snapshot.count}, Name: {snapshot.name}")

        # Changes trigger reactions
        CounterStore.count = 5  # Prints: Counter: 5, Name: Counter
        CounterStore.name = "My Counter"  # Prints: Counter: 5, Name: My Counter
        ```

    Note:
        Store uses a metaclass to intercept attribute assignment, allowing
        `Store.attr = value` syntax to work seamlessly with observables.
    """

    # Class attributes set by metaclass
    _observable_attrs: List[str]
    _observables: Dict[str, Observable]

    @classmethod
    def _get_observable_attrs(cls) -> List[str]:
        """Get observable attribute names in definition order."""
        return list(cls._observable_attrs)

    @classmethod
    def _get_primitive_observable_attrs(cls) -> List[str]:
        """Get primitive (non-computed) observable attribute names for persistence."""
        return [
            attr
            for attr in cls._observable_attrs
            if not isinstance(cls._observables[attr], ComputedObservable)
        ]

    @classmethod
    def to_dict(cls) -> Dict[str, SessionValue]:
        """Serialize all observable values to a dictionary."""
        return {attr: observable.value for attr, observable in cls._observables.items()}

    @classmethod
    def load_state(cls, state_dict: Dict[str, SessionValue]) -> None:
        """Load state from a dictionary into the store's observables."""
        for attr_name, value in state_dict.items():
            if attr_name in cls._observables:
                cls._observables[attr_name].set(value)

    @classmethod
    def subscribe(cls, func: Callable[[StoreSnapshot], None]) -> None:
        """Subscribe a function to react to all observable changes in the store."""
        snapshot = StoreSnapshot(cls, cls._observable_attrs)

        def store_reaction(value=None):
            snapshot._take_snapshot()
            func(snapshot)

        # Subscribe to all observables directly
        subscriptions = []
        for observable in cls._observables.values():
            observable.add_observer(store_reaction)
            subscriptions.append(observable)

        # Store the subscriptions for later unsubscribe
        if not hasattr(cls, "_subscription_contexts"):
            cls._subscription_contexts = {}
        cls._subscription_contexts[func] = {
            "reaction": store_reaction,
            "subscriptions": subscriptions,
        }

    @classmethod
    def unsubscribe(cls, func: Callable) -> None:
        """Unsubscribe a function from all observables."""
        if (
            hasattr(cls, "_subscription_contexts")
            and func in cls._subscription_contexts
        ):
            context = cls._subscription_contexts[func]
            # Unsubscribe from all observables
            for observable in context["subscriptions"]:
                observable.remove_observer(context["reaction"])
            del cls._subscription_contexts[func]
