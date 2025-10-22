"""
FynX Observable Descriptors - Reactive Attribute Descriptors
==========================================================

This module provides descriptor classes that enable transparent reactive programming
in class attributes. These descriptors bridge the gap between regular Python attribute
access and reactive capabilities, allowing Store classes to provide both familiar
attribute syntax and full reactive behavior.

Transparent Reactivity
----------------------

FynX's descriptors enable "transparent reactivity" - the ability to write code that
looks like regular attribute access while maintaining automatic dependency tracking
and change propagation. This means you can use observables in existing code without
major syntax changes.

Instead of:
```python
# Traditional reactive approach
store.counter.subscribe(lambda v: print(v))
store.counter.set(5)

# Manual dependency tracking
def update_total():
    total = store.price.value * store.quantity.value
```

You can write:
```python
# Transparent reactive approach
print(store.counter)  # Direct access
store.counter = 5     # Automatic updates

# Automatic dependency tracking
total = store.price * store.quantity  # Reactive computation
```

How It Works
------------

The descriptor system works through two key components:

1. **SubscriptableDescriptor**: Attached to class attributes, creates and manages
   the underlying Observable instances at the class level.

2. **ObservableValue**: Returned when accessing descriptor attributes, provides
   transparent value access while maintaining reactive capabilities.

When you access `store.counter`, the descriptor returns an ObservableValue that
wraps the actual Observable, allowing natural value operations while preserving
reactive behavior.

Key Benefits
------------

- **Familiar Syntax**: Use regular attribute access (`obj.attr = value`)
- **Reactive Capabilities**: Full access to subscription and operators
- **Type Safety**: Maintains type information through generics
- **Performance**: Efficient caching and lazy evaluation
- **Compatibility**: Works with existing Python idioms (iteration, comparison, etc.)

Common Patterns
---------------

**Store Attributes**:
```python
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
# Works with existing Python constructs
if store.is_enabled:
    print("Enabled")

for item in store.items:
    print(item)

# String formatting
message = f"User: {store.name}, Age: {store.age}"
```

**Reactive Operators**:
```python
# All operators work transparently
full_name = store.first_name + store.last_name >> (lambda f, l: f"{f} {l}")
is_adult = store.age >> (lambda a: a >= 18)
valid_user = store.name & is_adult
```

Implementation Details
----------------------

**Descriptor Protocol**: Uses `__get__`, `__set__`, and `__set_name__` to integrate
with Python's attribute system.

**Class-Level Storage**: Observables are stored at the class level to ensure
shared state across instances.

**Lazy Initialization**: ObservableValue instances are created on-demand and
cached for performance.

**Type Preservation**: Generic types ensure compile-time type safety.

Performance Considerations
--------------------------

- **Memory Efficient**: Reuses Observable instances across attribute access
- **Lazy Creation**: ObservableValue wrappers created only when needed
- **Minimal Overhead**: Thin wrapper around actual Observable instances
- **Caching**: Internal caching prevents redundant operations

Best Practices
--------------

- **Use Store Classes**: Leverage descriptors through Store classes for better organization
- **Consistent Access**: Use either direct access or reactive methods, not both inconsistently
- **Type Hints**: Provide type annotations for better IDE support
- **Documentation**: Document store attributes and their purposes

Limitations
-----------

- Descriptor behavior requires class-level attribute assignment
- Not suitable for instance-specific reactive attributes
- Some advanced Python features may not work as expected with wrapped values

See Also
--------

- `fynx.store`: Store classes that use these descriptors
- `fynx.observable`: Core observable classes
- `fynx.computed`: Creating derived reactive values
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    from ..primitives.base import Observable as ObservableImpl
    from ..conditional import ConditionalObservable
    from ..conditional.protocol import Conditional
    from ..merged.protocol import Mergeable
    from ..primitives.protocol import Observable
    from ..merged import MergedObservable

from .value import ObservableValue


# Runtime imports to avoid circular dependencies
def _get_observable_impl():
    from ..primitives.base import Observable as ObservableImpl

    return ObservableImpl


def _get_conditional_observable():
    from ..conditional import ConditionalObservable

    return ConditionalObservable


def _get_merged_observable():
    from ..merged import MergedObservable

    return MergedObservable


T = TypeVar("T")


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor that creates reactive class attributes with transparent observable behavior.

    SubscriptableDescriptor enables Store classes and other reactive containers to define
    attributes that behave like regular Python attributes while providing full reactive
    capabilities. When accessed, it returns an ObservableValue instance that combines
    direct value access with observable methods.

    This descriptor is the foundation for FynX's transparent reactive programming model,
    allowing you to write code that looks like regular attribute access while maintaining
    full reactive capabilities.

    Key Features:
    - **Class-Level Observables**: Creates observables at the class level for shared state
    - **Transparent Access**: Attributes behave like regular values but are reactive
    - **Automatic Management**: Handles observable lifecycle and descriptor protocol
    - **Store Integration**: Designed to work seamlessly with Store classes
    - **Memory Efficient**: Reuses observable instances across class access

    How It Works:
        1. When assigned to a class attribute, stores initial value and owner class
        2. On first access, creates a class-level Observable instance
        3. Returns an ObservableValue wrapper for transparent reactive access
        4. Subsequent accesses reuse the same observable instance

    Example:
        ```python
        from fynx import Store, observable

        class UserStore(Store):
            # This creates a SubscriptableDescriptor
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
        in Store classes. Direct instantiation is usually not needed.

    See Also:
        ObservableValue: The wrapper returned by this descriptor
        observable: Convenience function that creates SubscriptableDescriptor instances
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
        """Called when the descriptor is assigned to a class attribute."""
        self.attr_name = name
        self._owner_class = owner

    def __get__(self, instance: Optional[object], owner: Optional[Type]) -> Any:
        """Get the observable value for this attribute."""
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
                obs = _get_observable_impl()(
                    self.attr_name or "unknown", self._initial_value
                )
            setattr(target_class, obs_key, obs)

        retrieved_obs = getattr(target_class, obs_key)
        return retrieved_obs  # Return the Observable directly, not wrapped in ObservableValue

    def __set__(self, instance: Optional[object], value: Optional[T]) -> None:
        """Set the value on the observable."""
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
                obs = _get_observable_impl()(
                    self.attr_name or "unknown", self._initial_value
                )
            setattr(target_class, obs_key, obs)

        observable = getattr(target_class, obs_key)
        observable.set(value)
