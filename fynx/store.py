"""
FynX Store - reactive state management components
====================================================

A Store groups related observables into a single container with unified
change notification, instead of scattering them through the codebase. It
works well for application state (preferences, theme), feature state
(shopping carts, user profiles), component state shared across several
components, or derived values computed from raw data.

Core Components
---------------

**Store**: Base class for reactive state containers. Store classes define observable
attributes using the `observable()` descriptor, and Store provides methods for subscribing
to changes and managing state. The metaclass intercepts attribute assignment, so
`Store.attr = value` works directly with observables.

**observable**: Descriptor function that creates observable attributes on Store classes.
Use this to define reactive properties in your Store subclasses. The returned
Observable is itself a typed descriptor, so class access returns an ObservableValue
with the right static type while preserving reactive operators.

**StoreSnapshot**: Immutable snapshot of store state at a specific point in time. Useful
for debugging, logging, and ensuring consistent state access during reactive callbacks.
Each snapshot captures all observable values at creation time.

**StoreMeta**: Metaclass that records observable attributes, provides class-level
assignment (`Store.attr = value`), and gives inherited Store fields owner-specific
backing observables so subclasses do not share mutable state accidentally.

Key Features
------------

- Store's metaclass records observable descriptors and resolves inherited
  ones to owner-specific backing observables, so subclasses never share
  mutable state by accident
- Subscribe to every change in a store with one callback that receives a
  StoreSnapshot, or subscribe to individual observables directly
- Save and restore state with `to_dict()` and `load_state()`. `to_dict()`
  serializes every observable including computed ones; use
  `_get_primitive_observable_attrs()` to filter those out for persistence
- Full type hints, so class access gets the right static type
- Stores operate independently, so you can organize state by domain without
  creating cross-store dependencies

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
def on_store_change(snapshot):
    print(f"Store changed: count={snapshot.count}, name={snapshot.name}")

CounterStore.subscribe(on_store_change)

CounterStore.count = 10  # Triggers: "Store changed: count=10, name=My Counter"

# Unsubscribe when done
CounterStore.unsubscribe(on_store_change)
```

Alternative: use the `@reactive` decorator for a more convenient syntax:

```python
from fynx import reactive

@reactive(CounterStore)
def on_store_change(snapshot):
    print(f"Store changed: count={snapshot.count}, name={snapshot.name}")

CounterStore.count = 10  # Automatically triggers the function
```

Advanced Patterns
-----------------

### Computed Properties in Stores

Stores support computed observables that derive values from other observables. These
update automatically when their dependencies change:

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

Computed observables participate in store subscriptions the same way -
subscribing to a store triggers updates for computed values just like
primitive ones.

### State Persistence

Stores can serialize their state to dictionaries and restore from them. This enables
persistence across sessions or state transfer between components:

```python
# Save store state
state = CounterStore.to_dict()
# state = {"count": 10, "name": "My Counter"}

# Restore state later
CounterStore.load_state(state)
print(CounterStore.count)  # 10
```

Note that `to_dict()` includes all observables, including computed ones. For persistence,
you typically want only primitive observables since computed values derive from them.
Use `_get_primitive_observable_attrs()` to filter if needed, though the current implementation
serializes everything for simplicity.

### Store Composition

Multiple stores operate independently, allowing you to organize state by domain:

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

That independence lets you compose complex applications out of focused
stores, each managing its own domain.

Common Patterns
---------------

**Singleton Stores**: Use class-level access for global state. Since Store attributes
are class-level, each Store class acts as a singleton:

```python
class GlobalStore(Store):
    is_loading = observable(False)
    current_user = observable(None)

# Access globally
GlobalStore.is_loading = True
```

**Store Inheritance**: Child classes inherit observable attributes from parent classes.
The metaclass handles descriptor creation for inherited observables:

```python
class BaseStore(Store):
    created_at = observable(None)

class UserStore(BaseStore):
    name = observable("")
    # UserStore automatically has created_at from BaseStore
```

See Also
--------

- `fynx.observable`: Core observable classes and operators
- `fynx.reactive`: Reactive decorators for side effects
- `fynx.observable.computed`: Creating computed properties
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    overload,
)

from .observable import Observable, SubscriptableDescriptor
from .observable.computed import ComputedObservable
from .types import SessionValue, StoreState, StoreStateMapping

T = TypeVar("T")


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


@overload
def observable() -> Observable[None]: ...


@overload
def observable(initial_value: T) -> Observable[T]: ...


def observable(initial_value: Any = None) -> Observable[Any]:
    """
    Create an observable with an initial value, used as a descriptor in Store classes.
    """
    return Observable("standalone", initial_value)


# Type alias for subscriptable observables (class variables)
Subscriptable = SubscriptableDescriptor[Optional[T]]


class StoreMeta(type):
    """
    Metaclass for Store observable registration and class-level assignment.

    Observable instances are already typed descriptors. StoreMeta records them,
    resolves inherited descriptors to owner-specific backing observables, and
    intercepts class assignment so `Store.attr = value` delegates to `.set()`.
    """

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Type:
        annotations = namespace.get("__annotations__", {})
        new_namespace = namespace.copy()
        observable_attrs: List[str] = []

        for base in bases:
            if hasattr(base, "_observable_attrs"):
                for attr_name in getattr(base, "_observable_attrs", []):
                    if attr_name not in observable_attrs:
                        observable_attrs.append(attr_name)

        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Observable):
                if attr_name not in observable_attrs:
                    observable_attrs.append(attr_name)

        new_namespace["__annotations__"] = annotations
        cls = super().__new__(mcs, name, bases, new_namespace)

        cls._observable_attrs = list(observable_attrs)
        cls._observables = {
            attr: descriptor._observable_for_owner(cls)
            for attr in observable_attrs
            if isinstance(
                descriptor := mcs._find_observable_descriptor(cls, attr), Observable
            )
        }

        return cls

    @staticmethod
    def _find_observable_descriptor(cls: Type, attr_name: str) -> Optional[Observable]:
        for owner in cls.__mro__:
            descriptor = owner.__dict__.get(attr_name)
            if isinstance(descriptor, Observable):
                return descriptor
        return None

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

    Store subclasses define observable attributes with the `observable()`
    descriptor; Store provides methods for subscribing to changes,
    serializing state, and managing reactive relationships.

    The metaclass intercepts attribute assignment so `Store.attr = value`
    delegates to the underlying observable's `set()` method, triggering
    reactive updates.

    Key Features:
    - Automatic observable attribute detection and management through metaclass
    - Unified subscription method that reacts to all observable changes in the store
    - Serialization/deserialization support via `to_dict()` and `load_state()`
    - Snapshot functionality through StoreSnapshot for consistent state access

    Example:
        ```python
        from fynx import Store, observable

        class CounterStore(Store):
            count = observable(0)
            name = observable("Counter")

        # Subscribe to all changes
        def on_change(snapshot):
            print(f"Counter: {snapshot.count}, Name: {snapshot.name}")

        CounterStore.subscribe(on_change)

        # Changes trigger reactions
        CounterStore.count = 5  # Prints: Counter: 5, Name: Counter
        CounterStore.name = "My Counter"  # Prints: Counter: 5, Name: My Counter

        # Unsubscribe when done
        CounterStore.unsubscribe(on_change)
        ```

    Note:
        Store uses a metaclass to intercept attribute assignment, so
        `Store.attr = value` works directly with observables. The
        `subscribe()` method is a classmethod that takes a function, not a decorator.
        Use `@reactive(Store)` from `fynx.reactive` for decorator syntax.
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
    def to_dict(cls) -> StoreState:
        """Serialize all observable values to a dictionary."""
        return {attr: observable.value for attr, observable in cls._observables.items()}

    @classmethod
    def load_state(cls, state_dict: StoreStateMapping) -> None:
        """Load state from a dictionary into the store's observables."""
        for attr_name, value in state_dict.items():
            if attr_name in cls._observables:
                cls._observables[attr_name].set(value)

    @classmethod
    def subscribe(cls, func: Callable[[StoreSnapshot], object]) -> None:
        """Subscribe a function to react to all observable changes in the store."""
        snapshot = StoreSnapshot(cls, cls._observable_attrs)

        def store_reaction():
            snapshot._take_snapshot()
            func(snapshot)

        context = Observable._create_subscription_context(store_reaction, func, None)
        # Subscribe to all observables (including computed ones)
        context._store_observables = list(cls._observables.values())
        for observable in context._store_observables:
            observable.add_observer(context.run)

    @classmethod
    def unsubscribe(cls, func: Callable[[StoreSnapshot], object]) -> None:
        """Unsubscribe a function from all observables."""
        Observable._dispose_subscription_contexts(func)
