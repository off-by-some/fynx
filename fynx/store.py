"""
FynX Store - Reactive State Management Components
=================================================

This module provides the core components for reactive state management in FynX:

- **Store**: A base class for creating reactive state containers that group related
  observables together and provide convenient subscription methods.

- **observable**: A descriptor that creates observable attributes on Store classes.

- **StoreSnapshot**: A utility class for capturing and accessing snapshots of
  store state at specific points in time.

The Store class enables you to create organized, reactive state containers that
automatically notify subscribers when any observable attribute changes.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from .observable import Observable, SubscriptableDescriptor
from .observable.computed import ComputedObservable

T = TypeVar("T")

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


def observable(initial_value: Optional[T] = None) -> Any:
    """
    Create an observable with an initial value, used as a descriptor in Store classes.
    """
    return Observable("standalone", initial_value)


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

        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Observable):
                observable_attrs.append(attr_name)
                # Wrap all observables (including computed ones) in descriptors
                initial_value = attr_value.value
                new_namespace[attr_name] = SubscriptableDescriptor(
                    initial_value=initial_value, original_observable=attr_value
                )

        new_namespace["__annotations__"] = annotations
        cls = super().__new__(mcs, name, bases, new_namespace)

        # Cache observable attributes and their instances for efficient access
        cls._observable_attrs = list(observable_attrs)
        # Store the original observables from the namespace before they get replaced
        cls._observables = {attr: namespace[attr] for attr in observable_attrs}

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
        """Subscribe a function to react to all observable changes."""
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
    def unsubscribe(cls, func: Callable) -> None:
        """Unsubscribe a function from all observables."""
        Observable._dispose_subscription_contexts(func)
