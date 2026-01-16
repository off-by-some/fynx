"""
FynX Observable Interfaces - Abstract Base Classes for Reactive Programming
===========================================================================

This module defines the abstract interfaces that all reactive components must implement.
These ABCs establish contracts specifying required methods and behaviors without dictating
implementation details. That separation enables runtime type checking and polymorphism
while preventing circular dependencies.

The ABCs function as runtime type signatures: `isinstance(obj, Observable)` verifies
that an object implements the observable contract, regardless of its concrete class.
Classes depend on these ABCs rather than concrete implementations, eliminating circular
imports while maintaining type safety and enabling polymorphic behavior.

We apply that pattern across the reactive system. Observable defines the core contract:
value access with dependency tracking, change notification, and subscription management.
Mergeable extends it for observables that combine multiple sources into tuples.
Conditional extends it for observables that filter values through boolean gates.
ReactiveContext defines the execution environment contract that tracks dependencies
during reactive function execution.

Result: classes can reference interfaces without importing implementations, enabling
clean architecture while maintaining runtime type safety and isinstance checks.

Key Abstract Base Classes
-------------------------

**Observable**: Defines the core observable interface that all reactive values must implement.
Includes value access and change notification methods.

**Mergeable**: Extends Observable for observables that combine multiple source observables into tuples.

**Conditional**: Extends Observable for observables that filter values based on boolean conditions.

**ReactiveContext**: Defines the interface for execution contexts that track dependencies and manage
reactive function lifecycles.

Benefits
--------

- **Runtime Instance Checking**: Use isinstance(obj, Observable) at runtime
- **No Circular Imports**: Classes depend on ABCs, not concrete implementations
- **Type Safety**: Full generic type support with ABC-based typing
- **Clean Architecture**: Clear separation between interface contracts and implementations
- **IDE Support**: Better autocomplete and static analysis
- **Testability**: Easy to mock ABCs for unit testing
- **Polymorphism**: Runtime dispatch based on interface conformance

Usage
-----

Import these ABCs where you need to reference observable types:

```python
from fynx.observable.interfaces import Observable, Mergeable

# Runtime checking
if isinstance(some_obj, Observable):
    print(f"Value: {some_obj.value}")

# Type hints
def process_observable(obs: Observable[int]) -> None:
    pass
```

The ABCs use `abc.ABC` and `@abstractmethod` for proper abstract base class behavior.

See Also
--------

- `fynx.observable.operators`: Contains operator mixins and operator implementations
"""

import abc
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    TypeVar,
)

# Import operators locally in mixin methods to avoid circular imports

T = TypeVar("T")
U = TypeVar("U")


class ReactiveContext(abc.ABC):
    """
    Abstract Base Class defining the interface for reactive execution contexts.

    ReactiveContext tracks which observables get accessed during function execution
    and registers the function as an observer on each one. When any dependency changes,
    the context re-runs the function automatically. This ABC enables classes to depend
    on reactive contexts without importing concrete implementations.

    This ABC allows other classes to depend on reactive contexts without
    importing the concrete ReactiveContext implementation, while enabling
    runtime isinstance checks. That separation prevents circular imports—classes
    reference the interface, not the implementation.
    """

    @abc.abstractmethod
    def run(self) -> None:
        """
        Execute the reactive function and track its dependencies.

        This method runs the associated reactive function while automatically
        tracking which observables are accessed, setting up the necessary
        observers for future updates.
        """
        pass

    @abc.abstractmethod
    def dispose(self) -> None:
        """
        Clean up the reactive context and remove all observers.

        This method properly disposes of the context, removing all observers
        and cleaning up resources to prevent memory leaks.
        """
        pass


class Observable(abc.ABC, Generic[T]):
    """
    Abstract Base Class defining the core interface that all observable values must implement.

    This ABC captures the reactive contract: value access with dependency tracking,
    change notification, and subscription management. When an observable's value changes,
    it tracks what depends on it and notifies those dependents automatically. All
    observable implementations must conform to this interface to ensure consistent
    behavior across the reactive system.

    All observable implementations (regular, computed, merged, conditional) must
    conform to this ABC to ensure consistent behavior across the reactive system
    and enable runtime isinstance checks. That conformance enables polymorphism—
    code can work with any Observable without knowing its concrete type.
    """

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """
        Get a unique identifier for this observable.

        The key is used for debugging, serialization, and display purposes.
        It should be unique within a given context to allow observables to be
        distinguished from each other.

        Returns:
            A string identifier for this observable.
        """
        pass

    @property
    @abc.abstractmethod
    def value(self) -> Optional[T]:
        """
        Get the current value, automatically tracking dependencies in reactive contexts.

        Accessing this property registers the observable as a dependency if called
        within a reactive function execution context.

        Returns:
            The current value stored in the observable, or None if not set.
        """
        pass

    @abc.abstractmethod
    def set(self, value: Optional[T]) -> None:
        """
        Update the observable's value and notify all observers if the value changed.

        This method updates the internal value and triggers change notifications
        to all registered observers. Circular dependency detection is performed
        to prevent infinite loops.

        Args:
            value: The new value to store in the observable.
        """
        pass

    @abc.abstractmethod
    def subscribe(self, func: Callable) -> "Observable[T]":
        """
        Subscribe a function to react to value changes.

        The subscribed function will be called whenever the observable's value changes,
        receiving the new value as an argument.

        Args:
            func: A callable that accepts one argument (the new value).

        Returns:
            This observable instance for method chaining.
        """
        pass

    @abc.abstractmethod
    def add_observer(self, observer: Callable) -> None:
        """
        Add a low-level observer function that will be called when the value changes.

        Args:
            observer: A callable that takes no arguments and will be called
                     whenever the observable's value changes.
        """
        pass


class Mergeable(Observable[T], abc.ABC):
    """
    Abstract Base Class for observables that combine multiple source observables into tuples.

    Mergeable observables treat multiple related reactive values as a single atomic unit.
    They extend the base Observable ABC with tuple-specific operations, enabling functions
    that need multiple related parameters to receive them as a coordinated tuple that
    updates when any component changes.

    This ABC allows other classes to work with merged observables without
    importing the concrete MergedObservable implementation, while enabling
    runtime isinstance checks. That separation maintains clean architecture
    while preserving type safety.
    """

    _source_observables: List[Observable]


class Conditional(Observable[T], abc.ABC):
    """
    Abstract Base Class for observables that filter values based on boolean conditions.

    Conditional observables only emit values from a source observable when ALL specified
    conditions are True. They extend the base Observable ABC with condition-specific
    attributes like `is_active` to check whether the gate is currently open. This enables
    precise control over when reactive updates occur.

    This ABC allows other classes to work with conditional observables without
    importing the concrete ConditionalObservable implementation, while enabling
    runtime isinstance checks. That separation enables clean architecture
    while maintaining type safety across the reactive system.
    """

    _condition_observables: List[Observable[bool]]

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """Whether the conditional currently allows values through."""
        pass
