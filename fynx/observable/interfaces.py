"""
FynX Observable Interfaces - abstract base classes for reactive programming
==============================================================================

This module defines the abstract interfaces that reactive components
implement. Classes depend on these ABCs rather than the concrete
implementations, which avoids circular imports and lets `isinstance(obj,
Observable)` work regardless of an object's concrete class.

- `Observable` is the core contract: value access with dependency tracking,
  change notification, and subscription management.
- `Mergeable` extends it for observables that combine multiple sources into tuples.
- `Conditional` extends it for observables that filter values through boolean gates.
- `ReactiveContext` is the execution-environment contract that tracks
  dependencies during reactive function execution.

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
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from ..types import Observer, Subscriber

# Import operators locally in mixin methods to avoid circular imports

T = TypeVar("T")
U = TypeVar("U")


class ReactiveContext(abc.ABC):
    """
    Abstract Base Class defining the interface for reactive execution contexts.

    A ReactiveContext tracks which observables get accessed during function
    execution and registers the function as an observer on each one, so it
    re-runs automatically when any dependency changes. Depending on this ABC
    rather than the concrete ReactiveContext avoids circular imports while
    still allowing runtime isinstance checks.
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

    Captures the reactive contract: value access with dependency tracking,
    change notification, and subscription management. Every observable
    implementation (regular, computed, merged, conditional) conforms to this
    ABC, so code can work with any Observable without knowing its concrete type.
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
    def value(self) -> T:
        """
        Get the current value, automatically tracking dependencies in reactive contexts.

        Accessing this property registers the observable as a dependency if called
        within a reactive function execution context.

        Returns:
            The current value stored in the observable.
        """
        pass

    @abc.abstractmethod
    def set(self, value: T) -> None:
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
    def subscribe(self, func: Subscriber[T]) -> "Observable[T]":
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
    def add_observer(self, observer: Observer) -> None:
        """
        Add a low-level observer function that will be called when the value changes.

        Args:
            observer: A callable that takes no arguments and will be called
                     whenever the observable's value changes.
        """
        pass

    def remove_observer(self, observer: Observer) -> None:
        """
        Remove a low-level observer function.

        Args:
            observer: A callable previously registered with add_observer.
        """
        pass


class Mergeable(Observable[T], abc.ABC):
    """
    Abstract Base Class for observables that combine multiple source observables into tuples.

    Mergeable observables treat several related reactive values as one atomic
    unit, so a function that needs multiple parameters can receive them as a
    coordinated tuple that updates when any component changes. This ABC lets
    other classes work with merged observables without importing the
    concrete MergedObservable implementation.
    """

    _source_observables: Sequence[Observable[Any]]


class Conditional(Observable[T], abc.ABC):
    """
    Abstract Base Class for observables that filter values based on boolean conditions.

    Conditional observables only emit values from a source observable when
    every specified condition is True, exposing `is_active` to check whether
    the gate is currently open. This ABC lets other classes work with
    conditional observables without importing the concrete
    ConditionalObservable implementation.
    """

    _condition_observables: List[Observable[bool]]

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """Whether the conditional currently allows values through."""
        pass
