"""
FynX Observable Interfaces - Abstract Base Classes for Reactive Programming
===========================================================================

This module defines Abstract Base Classes (ABCs) for all observable classes to provide
runtime instance checking and eliminate circular imports. By defining contracts through
ABCs rather than concrete implementations, classes can depend on interfaces without
creating import cycles while enabling runtime isinstance checks.

These ABCs are used for both type hints and runtime checking, allowing the reactive
system to be statically typed while maintaining clean separation of concerns and
enabling runtime polymorphism checks.

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

    Reactive contexts track which observables are accessed during function execution
    and automatically set up observers to re-run the function when dependencies change.

    This ABC allows other classes to depend on reactive contexts without
    importing the concrete ReactiveContext implementation, while enabling
    runtime isinstance checks.
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

    This ABC captures the essential reactive behavior: value access with
    dependency tracking, change notification, and the operator overloading methods
    that enable FynX's fluent reactive syntax.

    All observable implementations (regular, computed, merged, conditional) must
    conform to this ABC to ensure consistent behavior across the reactive system
    and enable runtime isinstance checks.
    """

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

    Merged observables provide a unified interface for treating multiple related
    reactive values as a single atomic unit. They extend the base observable ABC
    with additional tuple-specific operations.

    This ABC allows other classes to work with merged observables without
    importing the concrete MergedObservable implementation, while enabling
    runtime isinstance checks.
    """

    _source_observables: List[Observable]


class Conditional(Observable[T], abc.ABC):
    """
    Abstract Base Class for observables that filter values based on boolean conditions.

    Conditional observables only emit values from a source observable when ALL
    specified conditions are True. They extend the base observable ABC
    with condition-specific attributes.

    This ABC allows other classes to work with conditional observables without
    importing the concrete ConditionalObservable implementation, while enabling
    runtime isinstance checks.
    """

    _condition_observables: List[Observable[bool]]
