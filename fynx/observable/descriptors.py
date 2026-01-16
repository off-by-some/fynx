"""
FynX Observable Descriptors - Reactive Attribute Descriptors
==========================================================

ObservableValue acts as a transparent wrapper that makes reactive values behave
like regular Python values. You access the value directly through familiar attribute
syntax, while the wrapper tracks changes and triggers updates automatically behind
the scenes. The value appears ordinary, but the system maintains reactive behavior.

This module provides descriptor classes that enable transparent reactive programming
in class attributes. These descriptors bridge regular Python attribute access with
reactive capabilities, allowing Store classes to provide familiar syntax alongside
automatic dependency tracking and change propagation.

Transparent Reactivity
----------------------

FynX's descriptors enable transparent reactivity—code that looks like regular
attribute access while maintaining automatic dependency tracking. You write
`store.counter = 5` and the system handles subscriptions, notifications, and
computed updates. That transparency means existing code works without modification.

Instead of explicit reactive patterns:
```python
# Traditional reactive approach
store.counter.subscribe(lambda v: print(v))
store.counter.set(5)

# Manual dependency tracking
def update_total():
    total = store.price.value * store.quantity.value
```

You write natural attribute access:
```python
# Transparent reactive approach
print(store.counter)  # Direct access
store.counter = 5     # Automatic updates

# Automatic dependency tracking
total = store.price * store.quantity  # Reactive computation
```

The descriptor system handles the reactive machinery behind the scenes. When you
access `store.counter`, the descriptor returns an ObservableValue that wraps the
actual Observable. That wrapper behaves like the value for most operations—equality,
string conversion, iteration—while also providing reactive methods like subscription
and operator overloading. We call this transparent reactivity—the value appears
ordinary, but the system tracks dependencies and propagates changes automatically.

How It Works
------------

The descriptor system operates through two components working together:

1. **SubscriptableDescriptor**: Attached to class attributes by StoreMeta, creates
   and manages the underlying Observable instances at the class level. The StoreMeta
   metaclass converts `observable()` instances (which return Observable objects) into
   SubscriptableDescriptor instances during class creation.

2. **ObservableValue**: Returned when accessing descriptor attributes, provides
   transparent value access through ValueMixin while maintaining reactive capabilities.
   This wrapper subscribes to the underlying Observable to keep its displayed value
   synchronized.

When you write `class UserStore(Store): name = observable("Alice")`, the StoreMeta
metaclass intercepts that Observable instance and replaces it with a SubscriptableDescriptor.
Accessing `UserStore.name` then triggers the descriptor's `__get__` method, which
creates or retrieves the class-level Observable and returns an ObservableValue wrapper.
That wrapper delegates value operations to the underlying value while preserving
reactive behavior. Result: attribute access looks normal, but the system maintains
dependency graphs and triggers updates automatically.

Common Patterns
---------------

**Store Attributes**:
```python
from fynx import Store, observable

class UserStore(Store):
    name = observable("Alice")
    age = observable(30)

# Access like regular attributes
print(UserStore.name)      # "Alice"
UserStore.age = 31         # Triggers reactive updates

# But also provides reactive methods
UserStore.name.subscribe(lambda n: print(f"Name: {n}"))
```

**Transparent Integration**:
```python
from fynx import Store, observable

class AppStore(Store):
    is_enabled = observable(True)
    items = observable([1, 2, 3])
    name = observable("Alice")
    age = observable(30)

# Works with existing Python constructs
if AppStore.is_enabled:
    print("Enabled")

for item in AppStore.items:
    print(item)

# String formatting
message = f"User: {AppStore.name}, Age: {AppStore.age}"
```

**Reactive Operators**:
```python
from fynx import Store, observable

class UserStore(Store):
    first_name = observable("John")
    last_name = observable("Doe")
    age = observable(20)
    name = observable("John")

# All operators work transparently
full_name = UserStore.first_name + UserStore.last_name >> (lambda f, l: f"{f} {l}")
is_adult = UserStore.age >> (lambda a: a >= 18)
valid_user = UserStore.name & is_adult
```

Implementation Details
----------------------

The descriptor protocol uses `__get__`, `__set__`, and `__set_name__` to integrate
with Python's attribute system. Observables are stored at the class level (as
`_{attr_name}_observable` attributes) to ensure shared state across all access.
ObservableValue instances are created on-demand when attributes are accessed, and
they subscribe to the underlying Observable to maintain synchronization.

The StoreMeta metaclass performs the conversion from Observable to SubscriptableDescriptor
during class creation. When you define `name = observable("Alice")` in a Store subclass,
StoreMeta detects the Observable instance and replaces it with a SubscriptableDescriptor
that wraps the original Observable. That descriptor then manages the class-level storage
and returns ObservableValue instances on access.

Performance Considerations
--------------------------

The system reuses Observable instances across attribute access, storing them as class
attributes. ObservableValue wrappers are created on-demand and subscribe to updates,
but the underlying Observable instances persist. This design minimizes memory overhead
while maintaining reactive behavior.

Limitations
-----------

Descriptor behavior requires class-level attribute assignment—instance-specific
reactive attributes are not supported. Some advanced Python features (like certain
metaclass interactions) may not work as expected with wrapped values, though common
operations (equality, iteration, string conversion) work transparently.

See Also
--------

- `fynx.store`: Store classes that use these descriptors
- `fynx.observable`: Core observable classes
- `fynx.computed`: Creating derived reactive values
"""

from typing import (
    Any,
    Generic,
    Optional,
    Type,
    TypeVar,
)

from .base import Observable as ObservableImpl
from .conditional import ConditionalObservable
from .interfaces import Conditional, Mergeable, Observable
from .merged import MergedObservable
from .operators import ValueMixin

T = TypeVar("T")


class ObservableValue(Generic[T], ValueMixin):
    """
    A transparent wrapper that combines direct value access with observable capabilities.

    ObservableValue acts as a bridge between regular Python value access and reactive
    programming. This wrapper behaves like the underlying value in most contexts—equality,
    string conversion, iteration, indexing—while also providing access to observable
    methods like subscription and operator overloading. The value appears ordinary, but
    the wrapper maintains reactive behavior behind the scenes.

    This class enables Store classes and other descriptor-based reactive systems to
    provide both familiar value access (`store.attr = value`) and reactive capabilities
    (`store.attr.subscribe(callback)`) through a single attribute. The wrapper subscribes
    to the underlying Observable during initialization, keeping its displayed value
    synchronized automatically.

    The ValueMixin provides transparent behavior: `__str__` delegates to the value,
    `__eq__` compares against the value (delegating to the underlying Observable's equality),
    `__iter__` iterates over collections or wraps scalars in a single-item list, `__len__`
    returns 0 for non-collections or the collection length otherwise, and `__contains__`
    works for collections but returns False for scalars. Reactive operators (`+`, `>>`, `&`)
    unwrap ObservableValue operands and delegate to the underlying Observable.

    Example:
        ```python
        from fynx import Store, observable

        class CounterStore(Store):
            count = observable(0)

        # ObservableValue provides both value access and reactive methods
        counter = CounterStore.count

        # Direct value access (like a regular attribute)
        print(counter)              # 0
        print(counter == 0)         # True
        print(len(counter))         # 0 (returns 0 for non-collections)

        # Iteration: scalars wrap in single-item list, collections iterate normally
        for x in counter:
            print(x)                 # 0 (scalar wrapped as [0])

        # Observable methods
        counter.set(5)              # Update the value
        counter.subscribe(lambda x: print(f"Count: {x}"))

        # Reactive operators
        doubled = counter >> (lambda x: x * 2)
        ```

    Note:
        ObservableValue instances are typically created automatically by
        SubscriptableDescriptor when accessing observable attributes on Store classes.
        You usually won't instantiate this class directly.

    See Also:
        SubscriptableDescriptor: Creates ObservableValue instances for class attributes
        Observable: The underlying reactive value class
        Store: Uses ObservableValue for transparent reactive attributes
    """

    def __init__(self, observable: "Observable[T]") -> None:
        self._observable = observable
        self._current_value = observable.value

        # Subscribe to updates to keep _current_value in sync
        def update_value(new_value):
            self._current_value = new_value

        observable.subscribe(update_value)

    # Observable methods
    @property
    def value(self) -> Optional[T]:
        return self._observable.value

    def set(self, value: Optional[T]) -> None:
        self._observable.set(value)
        self._current_value = value

    def subscribe(self, func) -> "Observable[T]":
        return self._observable.subscribe(func)

    @property
    def observable(self) -> "Observable[T]":
        """Get the underlying observable instance."""
        return self._observable

    def __getattr__(self, name: str):
        return getattr(self._observable, name)

    def __hash__(self) -> int:
        """Make ObservableValue hashable by delegating to the underlying observable."""
        return hash(self._observable)

    def __eq__(self, other) -> bool:
        """Equality comparison delegates to the underlying observable."""
        if isinstance(other, ObservableValue):
            return self._observable == other._observable
        return self._observable == other


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor that creates reactive class attributes with transparent observable behavior.

    SubscriptableDescriptor enables Store classes and other reactive containers to define
    attributes that behave like regular Python attributes while providing full reactive
    capabilities. When accessed, it returns an ObservableValue instance that combines
    direct value access with observable methods.

    This descriptor is the foundation for FynX's transparent reactive programming model.
    The StoreMeta metaclass converts `observable()` instances (which return Observable
    objects) into SubscriptableDescriptor instances during class creation. When you write
    `name = observable("Alice")` in a Store subclass, StoreMeta detects the Observable
    and replaces it with a SubscriptableDescriptor that wraps the original Observable.

    The descriptor stores observables at the class level (as `_{attr_name}_observable`
    attributes) to ensure shared state across all access. On first access via `__get__`,
    it creates or retrieves the class-level Observable and returns an ObservableValue wrapper.
    Subsequent accesses reuse the same observable instance. The `__set__` method delegates
    to the observable's `set()` method, triggering reactive updates.

    How It Works:
        1. StoreMeta converts `observable()` instances to SubscriptableDescriptor during
           class creation, storing the original Observable for later use
        2. On first attribute access, `__get__` creates a class-level Observable instance
           (or uses the original if provided) and stores it as `_{attr_name}_observable`
        3. Returns an ObservableValue wrapper for transparent reactive access
        4. Subsequent accesses reuse the same observable instance from the class

    Example:
        ```python
        from fynx import Store, observable

        class UserStore(Store):
            # StoreMeta converts this Observable to a SubscriptableDescriptor
            name = observable("Alice")
            age = observable(30)

        # Access returns ObservableValue instances
        user_name = UserStore.name    # ObservableValue wrapping Observable
        user_age = UserStore.age      # ObservableValue wrapping Observable

        # Behaves like regular attributes
        print(user_name)              # "Alice"
        UserStore.name = "Bob"        # Updates the observable
        print(user_name)              # "Bob"

        # But also provides reactive methods
        UserStore.name.subscribe(lambda n: print(f"Name changed to: {n}"))
        ```

    Note:
        This descriptor is typically used indirectly through the `observable()` function
        in Store classes, which returns an Observable that StoreMeta converts to a
        SubscriptableDescriptor. Direct instantiation is usually not needed.

    See Also:
        ObservableValue: The wrapper returned by this descriptor
        observable: Function that creates Observable instances (converted by StoreMeta)
        Store: Uses this descriptor for reactive class attributes
    """

    def __init__(
        self,
        initial_value: Optional[T] = None,
        original_observable: Optional["Observable[T]"] = None,
    ) -> None:
        self.attr_name: Optional[str] = None
        self._initial_value: Optional[T] = initial_value
        self._original_observable: Optional["Observable[T]"] = original_observable
        self._owner_class: Optional[Type] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """
        Called when the descriptor is assigned to a class attribute.

        This method is invoked automatically by Python when the descriptor is assigned
        to a class attribute. It stores the attribute name and owner class for later
        use in `__get__` and `__set__` methods.

        Args:
            owner: The class that owns this descriptor
            name: The name of the attribute this descriptor is assigned to
        """
        self.attr_name = name
        self._owner_class = owner

    def __get__(self, instance: Optional[object], owner: Optional[Type]) -> Any:
        """
        Get the observable value for this attribute.

        This method is called when the attribute is accessed. It creates or retrieves
        the class-level Observable instance and returns an ObservableValue wrapper
        that provides transparent value access while maintaining reactive capabilities.

        The observable is stored at the class level as `_{attr_name}_observable` to
        ensure shared state across all access. If the original Observable was provided
        during descriptor creation (by StoreMeta), it is reused; otherwise, a new
        Observable is created with the initial value.

        Args:
            instance: The instance that accessed the attribute (unused for class-level access)
            owner: The class that owns this descriptor

        Returns:
            An ObservableValue instance wrapping the class-level Observable

        Raises:
            AttributeError: If the descriptor is not properly initialized (owner is None)
        """
        # Always use the class being accessed (owner) as the target
        # This ensures each class gets its own observable instance
        target_class = owner
        if target_class is None:
            raise AttributeError("Descriptor not properly initialized")

        # Create class-level observable if it doesn't exist
        obs_key = f"_{self.attr_name}_observable"
        if obs_key not in target_class.__dict__:
            # Use the original observable if provided, otherwise create a new one
            if self._original_observable is not None:
                obs = self._original_observable
            else:
                obs = ObservableImpl(self.attr_name or "unknown", self._initial_value)
            setattr(target_class, obs_key, obs)

        retrieved_obs = getattr(target_class, obs_key)
        return ObservableValue(retrieved_obs)  # type: ignore

    def __set__(self, instance: Optional[object], value: Optional[T]) -> None:
        """
        Set the value on the observable.

        This method is called when the attribute is assigned a new value. It delegates
        to the underlying Observable's `set()` method, which triggers reactive updates
        and notifies all observers.

        The observable is created if it doesn't exist (using the same logic as `__get__`),
        ensuring that assignment works even before the attribute has been accessed.

        Args:
            instance: The instance that assigned the value (unused for class-level assignment)
            value: The new value to set on the observable

        Raises:
            AttributeError: If the descriptor is not properly initialized and no instance
                          is provided to determine the owner class
        """
        # Use the owner class (set in __set_name__) as the target
        # each descriptor's _owner_class will be the class that owns it
        target_class = self._owner_class
        if target_class is None:
            if instance is not None:
                target_class = type(instance)
            else:
                raise AttributeError("Cannot set value on uninitialized descriptor")

        # Create the observable if it doesn't exist (same logic as __get__)
        obs_key = f"_{self.attr_name}_observable"
        if obs_key not in target_class.__dict__:
            # Use the original observable if provided, otherwise create a new one
            if self._original_observable is not None:
                obs = self._original_observable
            else:
                obs = ObservableImpl(self.attr_name or "unknown", self._initial_value)
            setattr(target_class, obs_key, obs)

        observable = getattr(target_class, obs_key)
        observable.set(value)
