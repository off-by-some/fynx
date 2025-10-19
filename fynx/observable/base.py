"""
FynX Observable - Core Reactive Value Implementation
====================================================

This module provides the fundamental building blocks for reactive programming in FynX:

- **Observable**: The core class representing a reactive value that can be observed
  for changes and automatically notifies dependents.

- **ReactiveContext**: Manages the execution context for reactive functions,
  tracking dependencies and coordinating updates.

- **MergedObservable**: Combines multiple observables into a single reactive unit
  that updates when any of its components change.

- **ConditionalObservable**: Creates observables that only trigger reactions under
  specific conditions.

The Observable class forms the foundation of FynX's reactivity system, providing
transparent dependency tracking and automatic change propagation.
"""

from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)

from ..registry import _all_reactive_contexts, _func_to_contexts
from .interfaces import Conditional, Mergeable
from .interfaces import Observable as ObservableInterface
from .interfaces import ReactiveContext as ReactiveContextInterface
from .operators import OperatorMixin

T = TypeVar("T")


class ReactiveContext(ReactiveContextInterface):
    """
    Execution context for reactive functions with automatic dependency tracking.

    ReactiveContext manages the lifecycle of reactive functions (computations and reactions).
    It automatically tracks which observables are accessed during execution and sets up
    the necessary observers to re-run the function when any dependency changes.

    Key Responsibilities:
    - Track observable dependencies during function execution
    - Coordinate re-execution when dependencies change
    - Manage observer registration and cleanup
    - Handle merged observables and complex dependency relationships

    The context uses a stack-based approach to handle nested reactive functions,
    ensuring that dependencies are tracked correctly even in complex scenarios.

    Attributes:
        func (Callable): The reactive function to execute
        original_func (Callable): The original user function (for unsubscribe)
        subscribed_observable (Observable): The observable this context is subscribed to
        dependencies (Set[Observable]): Set of observables accessed during execution
        is_running (bool): Whether the context is currently executing

    Note:
        This class is typically managed automatically by FynX's decorators and
        observable operations. Direct instantiation is usually not needed.

    Example:
        ```python
        # Usually created automatically by @reactive decorator
        context = ReactiveContext(my_function, my_function, some_observable)
        context.run()  # Executes function and tracks dependencies
        ```
    """

    def __init__(
        self,
        func: Callable,
        original_func: Optional[Callable] = None,
        subscribed_observable: Optional["Observable"] = None,
    ) -> None:
        self.func = func
        self.original_func = (
            original_func or func
        )  # Store the original user function for unsubscribe
        self.subscribed_observable = (
            subscribed_observable  # The observable this context is subscribed to
        )
        self.dependencies: Set["Observable"] = set()
        self.is_running = False
        # For merged observables, we need to remove the observer from the merged observable,
        # not from the automatically tracked source observables
        self._observer_to_remove_from = subscribed_observable
        # For store subscriptions, keep track of all store observables
        self._store_observables: Optional[List["Observable"]] = None

    def run(self) -> None:
        """Run the reactive function, tracking dependencies."""
        old_context = Observable._current_context
        Observable._current_context = self

        # Push this context onto the stack
        Observable._context_stack.append(self)

        try:
            self.is_running = True
            self.dependencies.clear()  # Clear old dependencies
            self.func()
        finally:
            self.is_running = False
            Observable._current_context = old_context
            # Pop this context from the stack
            Observable._context_stack.pop()

    def add_dependency(self, observable: "Observable") -> None:
        """Add an observable as a dependency of this context."""
        # Only add if not already a dependency to avoid redundant observer registration
        if observable not in self.dependencies:
            self.dependencies.add(observable)
            observable.add_observer(self.run)

    def dispose(self) -> None:
        """Stop  the reactive computation and remove all observers."""
        if self._observer_to_remove_from is not None:
            # For single observables or merged observables
            self._observer_to_remove_from.remove_observer(self.run)
        elif (
            hasattr(self, "_store_observables") and self._store_observables is not None
        ):
            # For store-level subscriptions, remove from all store observables
            for observable in self._store_observables:
                observable.remove_observer(self.run)

        self.dependencies.clear()


class Observable(ObservableInterface[T], OperatorMixin):
    """
    A reactive value that automatically notifies dependents when it changes.

    Observable is the core primitive of FynX's reactivity system. It wraps a value
    and provides transparent reactive behavior - when the value changes, all
    dependent computations and reactions are automatically notified and updated.

    Key Features:
    - **Transparent**: Behaves like a regular value but with reactive capabilities
    - **Dependency Tracking**: Automatically tracks which reactive contexts depend on it
    - **Change Notification**: Notifies all observers when the value changes
    - **Type Safety**: Generic type parameter ensures type-safe operations
    - **Lazy Evaluation**: Computations only re-run when actually needed
    - **Circular Dependency Detection**: Prevents infinite loops at runtime

    Observable implements various magic methods (`__eq__`, `__str__`, etc.) to
    behave like its underlying value in most contexts, making it easy to use
    in existing code without modification.

    Attributes:
        key (Optional[str]): Unique identifier for debugging and serialization
        _value (Optional[T]): The current wrapped value
        _observers (Set[Callable]): Set of observer functions

    Class Attributes:
        _current_context (Optional[ReactiveContext]): Current reactive execution context
        _context_stack (List[ReactiveContext]): Stack of nested reactive contexts

    Args:
        key: A unique identifier for this observable (used for debugging).
             If None, will be set to "<unnamed>" and updated in __set_name__.
        initial_value: The initial value to store. Can be any type.

    Raises:
        RuntimeError: If a circular dependency is detected during value updates.

    Example:
        ```python
        from fynx.observable import Observable

        # Create an observable
        counter = Observable("counter", 0)

        # Direct access (transparent behavior)
        print(counter.value)  # 0
        print(counter == 0)   # True
        print(str(counter))   # "0"

        # Subscribe to changes
        def on_change():
            print(f"Counter changed to: {counter.value}")

        counter.subscribe(on_change)
        counter.set(5)  # Prints: "Counter changed to: 5"
        ```

    Note:
        While you can create Observable instances directly, it's often more
        convenient to use the `observable()` descriptor in Store classes for
        better organization and automatic serialization support.

    See Also:
        Store: For organizing observables into reactive state containers
        computed: For creating derived values from observables
        reactive: For creating reactive functions that respond to changes
    """

    # Class variable to track the current reactive context
    _current_context: Optional["ReactiveContext"] = None

    # Stack of reactive contexts being computed (for proper cycle detection)
    _context_stack: List["ReactiveContext"] = []

    # Pending notifications to avoid recursion in long chains
    _pending_notifications: Set["Observable"] = set()
    _notification_scheduled: bool = False

    def __init__(
        self, key: Optional[str] = None, initial_value: Optional[T] = None
    ) -> None:
        """
        Initialize an observable value.

        Args:
            key: A unique identifier for this observable (used for serialization).
                 If None, will be set to "<unnamed>" and updated in __set_name__.
            initial_value: The initial value to store
        """
        self.key = key or "<unnamed>"
        self._value = initial_value
        self._observers: Set[Callable] = set()

    @property
    def value(self) -> Optional[T]:
        """
        Get the current value of this observable.

        Accessing the value property automatically registers this observable
        as a dependency if called within a reactive context (computation or reaction).

        Returns:
            The current value stored in this observable, or None if not set.

        Note:
            This property is tracked by the reactive system. Use it instead of
            accessing _value directly to ensure proper dependency tracking.

        Example:
            ```python
            obs = Observable("counter", 5)
            print(obs.value)  # 5

            # In a reactive context, this creates a dependency
            @reactive(obs)
            def print_value(val):
                print(f"Value: {val}")
            ```
        """
        # Track dependency if we're in a reactive context
        if Observable._current_context is not None:
            Observable._current_context.add_dependency(self)
        return self._value

    def set(self, value: Optional[T]) -> None:
        """
        Set the value and notify all observers if the value changed.

        This method updates the observable's value and triggers change notifications
        to all registered observers. The update only occurs if the new value is
        different from the current value (using != comparison).

        Circular dependency detection is performed to prevent infinite loops where
        a computation tries to modify one of its own dependencies.

        Args:
            value: The new value to set. Can be any type compatible with the
                   observable's generic type parameter.

        Raises:
            RuntimeError: If setting this value would create a circular dependency
                         (e.g., a computed value trying to modify its own input).

        Example:
            ```python
            obs = Observable("counter", 0)
            obs.set(5)  # Triggers observers if value changed

            # No change, no notification
            obs.set(5)  # Same value, observers not called
            ```

        Note:
            Equality is checked using `!=` operator, so custom objects should
            implement proper equality comparison if needed.
        """
        # Check for circular dependency: check if the current context
        # is computing a value that depends on this observable
        current_context = Observable._current_context
        if current_context and self in current_context.dependencies:
            error_msg = f"Circular dependency detected in reactive computation!\n"
            error_msg += f"Observable '{self.key}' is being modified while computing a value that depends on it.\n"
            error_msg += f"This creates a circular dependency."
            raise RuntimeError(error_msg)

        # Only update and notify if the value actually changed
        if self._value != value:
            self._value = value
            # Defer notifications to avoid recursion in long chains
            Observable._pending_notifications.add(self)
            # Schedule notification if not already scheduled
            if not Observable._notification_scheduled:
                Observable._notification_scheduled = True
                # Use a microtask-like mechanism to defer execution
                Observable._schedule_notification()
        else:
            # Even if the value didn't change, we still check for circular dependencies
            # in case the setter is being called from within its own computation
            pass

    def _notify_observers(self) -> None:
        """Notify all registered observers that this observable has changed."""
        # Create a copy of observers to avoid "Set changed size during iteration"
        for observer in list(self._observers):
            observer()

    @classmethod
    def _schedule_notification(cls) -> None:
        """Schedule deferred notification of all pending observables."""
        try:
            # Process all pending notifications in breadth-first order
            # to avoid deep recursion in long chains
            while cls._pending_notifications:
                pending = cls._pending_notifications.copy()
                cls._pending_notifications.clear()
                for observable in pending:
                    observable._notify_observers()
        finally:
            cls._notification_scheduled = False
            # Ensure clean state
            cls._pending_notifications.clear()

    def add_observer(self, observer: Callable) -> None:
        """
        Add an observer function that will be called when this observable changes.

        Args:
            observer: A callable that takes no arguments
        """
        self._observers.add(observer)

    def remove_observer(self, observer: Callable) -> None:
        """
        Remove an observer function.

        Args:
            observer: The observer function to remove
        """
        self._observers.discard(observer)

    def subscribe(self, func: Callable) -> "Observable[T]":
        """
        Subscribe a function to react to changes in this observable.

        The subscribed function will be called whenever the observable's value changes.

        Args:
            func: A callable that accepts one argument (the new value).
                  The function will be called whenever the observable's value changes.

        Returns:
            This observable instance for method chaining.

        Example:
            ```python
            def on_change(new_value):
                print(f"Observable changed to: {new_value}")

            obs = Observable("counter", 0)
            obs.subscribe(on_change)

            obs.set(5)  # Prints: "Observable changed to: 5"
            ```

        Note:
            The function is called only when the observable's value changes.
            It is not called immediately upon subscription.

        See Also:
            unsubscribe: Remove a subscription
            reactive: Decorator-based subscription with automatic dependency tracking
        """

        def single_reaction():
            func(self.value)

        self._create_subscription_context(single_reaction, func, self)
        return self

    def unsubscribe(self, func: Callable) -> None:
        """
        Unsubscribe a function from this observable.

        Args:
            func: The function to unsubscribe from this observable
        """
        self._dispose_subscription_contexts(
            func, lambda ctx: ctx.subscribed_observable is self
        )

    @staticmethod
    def _create_subscription_context(
        reaction_func: Callable,
        original_func: Callable,
        subscribed_observable: Optional["Observable"],
    ) -> ReactiveContext:
        """Create and register a subscription context."""
        context = ReactiveContext(reaction_func, original_func, subscribed_observable)

        # Register context globally for unsubscribe functionality
        _all_reactive_contexts.add(context)
        _func_to_contexts.setdefault(original_func, []).append(context)

        # If there's a single subscribed observable, track it for proper disposal
        if subscribed_observable is not None:
            context.dependencies.add(subscribed_observable)
            subscribed_observable.add_observer(context.run)

        return context

    @staticmethod
    def _dispose_subscription_contexts(
        func: Callable, filter_predicate: Optional[Callable] = None
    ) -> None:
        """
        Dispose of subscription contexts for a function with optional filtering.

        This internal method finds and cleans up ReactiveContext instances associated
        with a given function. It's used by unsubscribe() methods to properly clean up
        reactive subscriptions.

        Args:
            func: The function whose subscription contexts should be disposed
            filter_predicate: Optional predicate function to filter which contexts to dispose.
                            Should accept a ReactiveContext and return bool.

        Note:
            This is an internal method used by the reactive system.
            Direct use is not typically needed.
        """
        if func not in _func_to_contexts:
            return

        # Filter contexts based on predicate if provided
        contexts_to_remove = [
            ctx
            for ctx in _func_to_contexts[func]
            if filter_predicate is None or filter_predicate(ctx)
        ]

        for context in contexts_to_remove:
            context.dispose()
            _all_reactive_contexts.discard(context)
            _func_to_contexts[func].remove(context)

        # Clean up empty function mappings
        if not _func_to_contexts[func]:
            del _func_to_contexts[func]

    # Magic methods for transparent behavior
    def __bool__(self) -> bool:
        """
        Boolean conversion returns whether the value is truthy.

        This allows observables to be used directly in boolean contexts
        (if statements, boolean operations) just like regular values.

        Returns:
            True if the wrapped value is truthy, False otherwise.

        Example:
            ```python
            obs = Observable("flag", True)
            if obs:  # Works like if obs.value
                print("Observable is truthy")

            obs.set(0)  # False
            if not obs:  # Works like if not obs.value
                print("Observable is falsy")
            ```
        """
        return bool(self._value)

    def __str__(self) -> str:
        """
        String representation of the wrapped value.

        Returns the string representation of the current value,
        enabling observables to be used seamlessly in string contexts.

        Returns:
            String representation of the wrapped value.

        Example:
            ```python
            obs = Observable("name", "Alice")
            print(f"Hello {obs}")  # Prints: "Hello Alice"
            message = "User: " + obs  # Works like "User: " + obs.value
            ```
        """
        return str(self._value)

    def __repr__(self) -> str:
        """
        Developer representation showing the observable's key and current value.

        Returns:
            A string representation useful for debugging and development.

        Example:
            ```python
            obs = Observable("counter", 42)
            print(repr(obs))  # Observable('counter', 42)
            ```
        """
        return f"Observable({self.key!r}, {self._value!r})"

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison with another value or observable.

        Compares the wrapped values for equality. If comparing with another
        Observable, compares their wrapped values.

        Args:
            other: Value or Observable to compare with

        Returns:
            True if the values are equal, False otherwise.

        Example:
            ```python
            obs1 = Observable("a", 5)
            obs2 = Observable("b", 5)
            regular_val = 5

            obs1 == obs2      # True (both wrap 5)
            obs1 == regular_val  # True (observable equals regular value)
            obs1 == 10        # False (5 != 10)
            ```
        """
        if isinstance(other, Observable):
            return self._value == other._value
        return self._value == other

    def __hash__(self) -> int:
        """
        Hash based on object identity, not value.

        Since values may be unhashable (like dicts, lists), observables
        hash based on their object identity rather than their value.

        Returns:
            Hash of the observable's object identity.

        Note:
            This means observables with the same value will not be
            considered equal for hashing purposes, only identical objects.

        Example:
            ```python
            obs1 = Observable("a", [1, 2, 3])
            obs2 = Observable("b", [1, 2, 3])

            # These will have different hashes despite same value
            hash(obs1) != hash(obs2)  # True

            # But identical objects hash the same
            hash(obs1) == hash(obs1)  # True
            ```
        """
        return id(self)

    # Descriptor protocol for use as class attributes
    def __set_name__(self, owner: Type, name: str) -> None:
        """
        Called when this Observable is assigned to a class attribute.

        This method implements the descriptor protocol to enable automatic
        conversion of Observable instances to appropriate descriptors based
        on the owning class type.

        For Store classes, the conversion is handled by StoreMeta metaclass.
        For other classes, converts to SubscriptableDescriptor for class-level
        observable behavior.

        Args:
            owner: The class that owns this attribute
            name: The name of the attribute being assigned

        Note:
            This method is called automatically by Python when an Observable
            instance is assigned to a class attribute. It modifies the class
            to use the appropriate descriptor for reactive behavior.

        Example:
            ```python
            class MyClass:
                obs = Observable("counter", 0)  # __set_name__ called here

            # Gets converted to a descriptor automatically
            instance = MyClass()
            print(instance.obs)  # Uses descriptor
            ```
        """
        # Update key if it was defaulted to "<unnamed>"
        if self.key == "<unnamed>":
            # Check if this is a computed observable by checking for the _is_computed attribute
            if getattr(self, "_is_computed", False):
                self.key = f"<computed:{name}>"
            else:
                self.key = name

        # Skip processing for computed observables - they should remain as-is
        if getattr(self, "_is_computed", False):
            return

        # Check if owner is a Store class - if so, let StoreMeta handle the conversion
        try:
            from .store import Store

            if issubclass(owner, Store):
                return
        except ImportError:
            # If store module is not available, continue with normal processing
            pass

        # For non-Store classes, convert to a SubscriptableDescriptor
        # that will create class-level observables
        from .descriptors import SubscriptableDescriptor

        descriptor: SubscriptableDescriptor[T] = SubscriptableDescriptor(self._value)
        descriptor.attr_name = name
        descriptor._owner_class = owner

        # Replace this Observable instance with the descriptor on the class
        setattr(owner, name, descriptor)

        # Remove this instance since it's being replaced
        # The descriptor will create the actual Observable when accessed
