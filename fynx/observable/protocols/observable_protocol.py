"""
FynX Observable Protocol - Core Observable Interface Definitions
================================================================

This module defines Protocol-based interfaces for the core Observable in FynX,
providing the fundamental reactive behavior interface with automatic unwrapping
of nested Observable values.

The Observable protocol provides the essential reactive behavior:

Core Observable Protocol:
- `value` - Get the current value with automatic unwrapping
- `set(value)` - Update the value and notify observers
- `subscribe(func)` - Subscribe to value changes
- `unsubscribe(func)` - Unsubscribe from changes
- `add_observer(observer)` - Add observer for change notifications
- `remove_observer(observer)` - Remove observer

Auto-Unwrapping:
The value property automatically unwraps nested Observable values:
- `Observable[Observable[T]]` → `Observable[T]`
- `Observable[Tuple[Observable[T], Observable[U]]]` → `Observable[Tuple[T, U]]`

Key Benefits:
- No circular imports (protocols don't import concrete implementations)
- Better type safety than ABCs
- Runtime isinstance() support with @runtime_checkable
- Structural subtyping (duck typing with type safety)
- Clean separation of interface from implementation
- Automatic value unwrapping for ergonomic API
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

# Import common types
from ..types.common_types import (
    Condition,
    ConditionFunction,
    T,
    TransformFunction,
    U,
    V,
)

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from ..types.observable_protocols import Computed, Conditional, Mergeable

# ============================================================================
# CORE OBSERVABLE PROTOCOL
# ============================================================================


@runtime_checkable
class Observable(Protocol[T]):
    """
    Protocol defining the core observable interface.

    This is the fundamental interface that all reactive values must implement.
    It defines the essential reactive behavior: value access with dependency
    tracking, change notification, and basic lifecycle management.

    Key Features:
    - Value access with automatic dependency tracking
    - Automatic unwrapping of nested Observable values
    - Change notification to observers
    - Subscription management
    - Transparent behavior in most contexts

    Auto-Unwrapping:
    The value property automatically unwraps nested Observable values:
    - Observable(Observable(1)) → 1
    - (Observable(1), Observable(2)) → (1, 2)
    - Observable((Observable(1), Observable(2))) → (1, 2)

    Example:
        ```python
        def process_observable(obs: Observable[int]) -> None:
            print(f"Value: {obs.value}")  # Automatically unwrapped
            obs.subscribe(lambda v: print(f"Changed to: {v}"))

            # All observables support reactive operations
            doubled = obs.then(lambda x: x * 2)  # Natural language
            doubled = obs >> (lambda x: x * 2)   # Operator syntax

            # Auto-unwrapping works everywhere
            nested = observable(observable(5))
            print(nested.value)  # Prints: 5 (not Observable(5))
        ```
    """

    # Core properties
    key: str

    @property
    def value(self) -> T:
        """
        Get the current value of this observable.

        This property automatically unwraps nested Observable values,
        so you always get the raw value, not Observable wrappers.

        Returns:
            The current unwrapped value of this observable.

        Example:
            ```python
            obs = Observable('test', 5)
            print(obs.value)  # Prints: 5

            # Auto-unwrapping works
            nested = Observable('nested', Observable('inner', 10))
            print(nested.value)  # Prints: 10 (not Observable(10))
            ```
        """
        ...

    def set(self, value: Optional[T]) -> None:
        """
        Set the value of this observable and notify all observers.

        This method updates the observable's value and triggers change
        notifications to all subscribed observers and registered observers.

        Args:
            value: The new value to set. Can be None to clear the value.

        Example:
            ```python
            obs = Observable('test', 5)
            obs.set(10)  # Updates value to 10 and notifies observers
            obs.set(None)  # Clears the value
            ```
        """
        ...

    def subscribe(self, func: Callable[[T], None]) -> "Observable[T]":
        """
        Subscribe to value changes of this observable.

        This method registers a callback function that will be called
        whenever the observable's value changes. The callback receives
        the new unwrapped value.

        Args:
            func: A callback function that receives the new value.

        Returns:
            This observable for method chaining.

        Example:
            ```python
            obs = Observable('test', 5)
            obs.subscribe(lambda v: print(f"Value changed to: {v}"))
            obs.set(10)  # Triggers the callback with 10
            ```
        """
        ...

    def unsubscribe(self, func: Callable[[T], None]) -> None:
        """
        Unsubscribe from value changes of this observable.

        This method removes a previously registered callback function
        from the list of subscribers.

        Args:
            func: The callback function to remove.

        Example:
            ```python
            def callback(v): print(f"Value: {v}")

            obs = Observable('test', 5)
            obs.subscribe(callback)
            obs.set(10)  # Triggers callback

            obs.unsubscribe(callback)
            obs.set(20)  # No callback triggered
            ```
        """
        ...

    def add_observer(self, observer: Callable[[], None]) -> None:
        """
        Add an observer for change notifications.

        This method registers a callback function that will be called
        whenever the observable's value changes. Unlike subscribe(),
        this observer doesn't receive the new value, just a notification
        that something changed.

        Args:
            observer: A callback function that will be called on changes.

        Example:
            ```python
            obs = Observable('test', 5)
            obs.add_observer(lambda: print("Something changed"))
            obs.set(10)  # Triggers the observer
            ```
        """
        ...

    def remove_observer(self, observer: Callable[[], None]) -> None:
        """
        Remove an observer from change notifications.

        This method removes a previously registered observer callback
        from the list of observers.

        Args:
            observer: The observer callback to remove.

        Example:
            ```python
            def observer(): print("Changed")

            obs = Observable('test', 5)
            obs.add_observer(observer)
            obs.set(10)  # Triggers observer

            obs.remove_observer(observer)
            obs.set(20)  # No observer triggered
            ```
        """
        ...

    # Magic methods for transparent behavior
    def __bool__(self) -> bool:
        """
        Return the boolean value of this observable.

        This enables the observable to be used in boolean contexts
        like if statements and boolean operations.

        Returns:
            The boolean value of the current value.

        Example:
            ```python
            obs = Observable('test', 5)
            if obs:  # Uses __bool__
                print("Has value")

            bool_obs = Observable('bool', True)
            if bool_obs:  # Uses __bool__
                print("Is true")
            ```
        """
        ...

    def __str__(self) -> str:
        """
        Return the string representation of this observable.

        This enables the observable to be used in string contexts
        like print statements and string formatting.

        Returns:
            The string representation of the current value.

        Example:
            ```python
            obs = Observable('test', 5)
            print(obs)  # Prints: 5
            print(f"Value: {obs}")  # Prints: Value: 5
            ```
        """
        ...

    def __repr__(self) -> str:
        """
        Return the detailed string representation of this observable.

        This provides a detailed representation useful for debugging
        and development.

        Returns:
            A detailed string representation of this observable.

        Example:
            ```python
            obs = Observable('test', 5)
            print(repr(obs))  # Prints detailed representation
            ```
        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Compare this observable with another object for equality.

        This enables the observable to be used in equality comparisons
        and as dictionary keys (if hashable).

        Args:
            other: Another object to compare with.

        Returns:
            True if the values are equal, False otherwise.

        Example:
            ```python
            obs1 = Observable('test1', 5)
            obs2 = Observable('test2', 5)
            print(obs1 == obs2)  # True (values are equal)
            print(obs1 == 5)     # True (value equals 5)
            ```
        """
        ...

    def __hash__(self) -> int:
        """
        Return the hash value of this observable.

        This enables the observable to be used as dictionary keys
        and in sets.

        Returns:
            The hash value of the current value.

        Example:
            ```python
            obs = Observable('test', 5)
            my_set = {obs}  # Uses __hash__
            my_dict = {obs: "value"}  # Uses __hash__
            ```
        """
        ...
