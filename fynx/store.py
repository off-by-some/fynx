"""
Fynx Store - Reactive State Management Components
=================================================

This module provides the core components for reactive state management in Fynx:

- **Store**: A base class for creating reactive state containers that group related
  observables together and provide convenient subscription methods.

- **observable**: A descriptor that creates observable attributes on Store classes.

- **StoreSnapshot**: A utility class for capturing and accessing snapshots of
  store state at specific points in time.

The Store class enables you to create organized, reactive state containers that
automatically notify subscribers when any observable attribute changes.
"""

from typing import (
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from .observable import Observable, SubscriptableDescriptor

T = TypeVar('T')

# Type alias for session state values (used for serialization)
SessionValue = Union[None, str, int, float, bool, Dict[str, 'SessionValue'], List['SessionValue']]


class StoreSnapshot:
    """
    Immutable snapshot of store observable values at a specific point in time.

    StoreSnapshot captures the current values of all observable attributes in a Store
    class and provides read-only access to those values. This is useful for:

    - Logging or debugging store state
    - Comparing store states over time
    - Serializing store state for persistence
    - Creating undo/redo functionality

    The snapshot behaves like a read-only version of the store, allowing attribute
    access to both observable values and non-observable class attributes.

    Example:
        ```python
        class MyStore(Store):
            count = observable(0)
            name = observable("test")

        # Take a snapshot
        snapshot = MyStore.snapshot()
        print(snapshot.count)  # 0
        print(snapshot.name)   # "test"

        # Changes to store don't affect snapshot
        MyStore.count = 5
        print(snapshot.count)  # Still 0
        ```
    """

    def __init__(self, store_class: type, observable_attrs: List[str]):
        """
        Initialize a store snapshot.

        Args:
            store_class: The Store class to snapshot
            observable_attrs: List of attribute names that are observables
        """
        self._store_class = store_class
        self._observable_attrs = observable_attrs
        self._snapshot_values: Dict[str, SessionValue] = {}
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        """
        Capture current values of all observable attributes.

        This method reads the current .value of each observable attribute
        and stores it in the snapshot for later retrieval. If an attribute
        doesn't exist or isn't an observable, it's skipped.
        """
        for attr_name in self._observable_attrs:
            try:
                attr_value = getattr(self._store_class, attr_name)
                if hasattr(attr_value, 'value'):  # It's an observable
                    self._snapshot_values[attr_name] = attr_value.value
            except AttributeError:
                # Attribute doesn't exist - skip it (useful for testing)
                continue

    def __getattr__(self, name: str):
        """
        Provide attribute access to snapshot values.

        Args:
            name: Attribute name to access

        Returns:
            The snapshot value for observable attributes, or the current
            class attribute value for non-observable attributes.

        Raises:
            AttributeError: If the attribute doesn't exist on the store class
        """
        if name in self._snapshot_values:
            return self._snapshot_values[name]

        # Fallback to store class attributes for non-observable attributes
        return getattr(self._store_class, name)

    def __repr__(self) -> str:
        """
        Return a string representation of the snapshot.

        Shows the snapshot values in a format similar to a named tuple,
        making it easy to inspect the captured state.
        """
        if not self._snapshot_values:
            return "StoreSnapshot()"

        # Format each field as name=value
        fields = []
        for name in self._observable_attrs:
            if name in self._snapshot_values:
                value = self._snapshot_values[name]
                fields.append(f"{name}={value!r}")

        return f"StoreSnapshot({', '.join(fields)})"

def observable(initial_value: Optional[T] = None) -> Observable[T]:
    """
    Create an observable with an initial value.

    This function creates observable instances that can be used standalone or as class attributes.

    Examples:
        ```python
        # Standalone observable
        name = observable("Alice")

        # Class attribute - when used in a class, it gets converted to a descriptor
        class MyStore(Store):
            age = observable(30)
        ```
    """
    return Observable("standalone", initial_value)


# Type alias for subscriptable observables (class variables)
Subscriptable = SubscriptableDescriptor[Optional[T]]


class StoreMeta(type):
    """Metaclass for Store to intercept class attribute assignment."""

    def __setattr__(cls, name: str, value: object) -> None:
        """Intercept class attribute assignment for observables."""
        # Check if this attribute has an observable
        obs_attr = f'_{name}_observable'
        if hasattr(cls, obs_attr):
            observable = getattr(cls, obs_attr)
            if isinstance(observable, Observable):
                # Set the value on the observable
                observable.set(value)
                return

        # Regular attribute assignment
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

    @classmethod
    def _get_observables(cls) -> Dict[str, Observable]:
        """
        Get all observable attributes defined on this store class.

        Returns:
            Dictionary mapping attribute names to their Observable instances
        """
        observables = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_'):
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, Observable):
                    observables[attr_name] = attr_value
        return observables

    @classmethod
    def to_dict(cls) -> Dict[str, SessionValue]:
        """
        Serialize all observable values in this store to a dictionary.

        This method creates a snapshot of all current observable values and returns
        them as a dictionary suitable for JSON serialization or persistence.

        Returns:
            Dictionary mapping observable attribute names to their current values

        Example:
            ```python
            class MyStore(Store):
                count = observable(5)
                name = observable("test")

            state = MyStore.to_dict()
            # {'count': 5, 'name': 'test'}
            ```
        """
        return {obs.key: obs.value for obs in cls._get_observables().values()}

    @classmethod
    def load_state(cls, state_dict: Dict[str, SessionValue]) -> None:
        """
        Load state from a dictionary into the store's observables.

        This method deserializes state that was previously saved using `to_dict()`,
        restoring the observable values to their previous state. Only observables
        that exist in both the store and the state dictionary will be updated.

        Args:
            state_dict: Dictionary mapping observable attribute names to values,
                       typically created by a previous call to `to_dict()`

        Example:
            ```python
            class MyStore(Store):
                count = observable(0)
                name = observable("")

            # Save state
            saved_state = MyStore.to_dict()

            # Modify state
            MyStore.count = 10
            MyStore.name = "modified"

            # Restore state
            MyStore.load_state(saved_state)
            # count is back to 0, name is back to ""
            ```
        """
        for obs in cls._get_observables().values():
            if obs.key in state_dict:
                obs.set(state_dict[obs.key])

    @classmethod
    def _get_observable_attrs(cls) -> List[str]:
        """Get observable attribute names in definition order."""
        return [
            attr_name
            for attr_name, attr_value in cls.__dict__.items()
            if not attr_name.startswith('_') and isinstance(attr_value, SubscriptableDescriptor)
        ]

    @classmethod
    def subscribe(cls, func: callable) -> None:
        """
        Subscribe a function to react to all observable changes in this store.

        The subscribed function will be called whenever any observable attribute in the
        store changes, receiving a StoreSnapshot object that provides read-only access
        to the current values of all observables. This provides a convenient way to
        react to any change in the store's state.

        Args:
            func: The function to call when observables change. It will receive a
                  StoreSnapshot object containing the current values of all observables
                  in the store. The function signature should be `func(snapshot)`.

        Note:
            The function is called immediately when subscription is created (to provide
            initial state), and then again whenever any observable in the store changes.

        Example:
            ```python
            class TodoStore(Store):
                items = observable([])
                filter = observable("all")

            @TodoStore.subscribe
            def on_store_change(snapshot):
                print(f"Store changed: {len(snapshot.items)} items, filter: {snapshot.filter}")
                # React to changes (e.g., update UI, save to disk, etc.)

            TodoStore.items = ["task1", "task2"]  # Triggers reaction
            TodoStore.filter = "completed"        # Triggers reaction
            ```

        See Also:
            unsubscribe(): Remove a subscription
            StoreSnapshot: The snapshot object passed to subscribers
        """
        from .observable import Observable  # Import here to avoid circular imports

        observable_attrs = cls._get_observable_attrs()
        snapshot = StoreSnapshot(cls, observable_attrs)

        def store_reaction():
            snapshot._take_snapshot()
            func(snapshot)

        context = Observable._create_subscription_context(store_reaction, func, None)
        context._store_observables = [getattr(cls, attr) for attr in observable_attrs]

        # Attach observers to all store observables (since the helper doesn't handle this for multi-observable cases)
        for observable in context._store_observables:
            observable.add_observer(context.run)

    @classmethod
    def unsubscribe(cls, func: callable) -> None:
        """
        Unsubscribe a reactive function from all observables in this store.

        This method removes all subscriptions for the given function from all
        observables in the store, stopping it from being called when observables
        change. This is the counterpart to the `subscribe()` method.

        Args:
            func: The function that was previously subscribed using `subscribe()`.
                  Must be the same function object that was passed to subscribe.

        Note:
            After unsubscribing, the function will no longer be called when
            observables in this store change. If the function was subscribed
            multiple times, all subscriptions are removed.

        Example:
            ```python
            def on_change(snapshot):
                print("Store changed!")

            # Subscribe
            TodoStore.subscribe(on_change)
            TodoStore.items = ["new item"]  # Prints: "Store changed!"

            # Unsubscribe
            TodoStore.unsubscribe(on_change)
            TodoStore.items = ["another item"]  # No output
            ```

        See Also:
            subscribe(): Subscribe to store changes
        """
        from .observable import Observable  # Import here to avoid circular imports
        Observable._dispose_subscription_contexts(func)
