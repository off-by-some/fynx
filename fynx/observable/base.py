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
    TYPE_CHECKING,
    Callable,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    from .merged import MergedObservable
    from .conditional import ConditionalObservable

from ..registry import _all_reactive_contexts, _func_to_contexts

T = TypeVar("T")


class ReactiveContext:
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

    Note:
        This class is typically managed automatically by FynX's decorators and
        observable operations. Direct instantiation is usually not needed.
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
        # Simply add the dependency - cycle detection happens during set()
        self.dependencies.add(observable)
        observable.add_observer(self.run)

    def dispose(self) -> None:
        """Stop the reactive computation and remove all observers."""
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


class Observable(Generic[T]):
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

    Observable implements various magic methods (`__eq__`, `__str__`, etc.) to
    behave like its underlying value in most contexts, making it easy to use
    in existing code without modification.

    Example:
        ```python
        from fynx import observable

        # Create an observable
        counter = observable("counter", 0)

        # Direct access
        print(counter.value)  # 0

        # Subscribe to changes
        def on_change():
            print(f"Counter changed to: {counter.value}")

        counter.subscribe(on_change)
        counter.set(5)  # Prints: "Counter changed to: 5"
        ```

    Note:
        While you can create Observable instances directly, it's often more
        convenient to use the `observable()` descriptor in Store classes.
    """

    # Class variable to track the current reactive context
    _current_context: Optional["ReactiveContext"] = None

    # Stack of reactive contexts being computed (for proper cycle detection)
    _context_stack: List["ReactiveContext"] = []

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
        """Get the current value of this observable."""
        # Track dependency if we're in a reactive context
        if Observable._current_context is not None:
            Observable._current_context.add_dependency(self)
        return self._value

    def set(self, value: Optional[T]) -> None:
        """
        Set the value and notify all observers if the value changed.

        Args:
            value: The new value to set
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
            self._notify_observers()
        else:
            # Even if the value didn't change, we still check for circular dependencies
            # in case the setter is being called from within its own computation
            pass

    def _notify_observers(self) -> None:
        """Notify all registered observers that this observable has changed."""
        # Create a copy of observers to avoid "Set changed size during iteration"
        for observer in list(self._observers):
            observer()

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

        Args:
            func: The function to call when this observable changes.

        Returns:
            This observable instance for method chaining.
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
        """Dispose of subscription contexts for a function with optional filtering."""
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
        """Boolean conversion returns whether the value is truthy."""
        return bool(self._value)

    def __str__(self) -> str:
        """String representation of the value."""
        return str(self._value)

    def __repr__(self) -> str:
        """Developer representation of this observable."""
        return f"Observable({self.key!r}, {self._value!r})"

    def __eq__(self, other: object) -> bool:
        """Equality comparison with another value or observable."""
        if isinstance(other, Observable):
            return self._value == other._value
        return self._value == other

    def __hash__(self) -> int:
        """Hash based on object identity, not value (values may be unhashable)."""
        return id(self)

    # Descriptor protocol for use as class attributes
    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when assigned to a class attribute."""
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

    def __or__(self, other: "Observable") -> "MergedObservable[T]":
        """
        Combine this observable with another using the | operator.

        This creates a merged observable that contains a tuple of both values
        and updates automatically when either observable changes.

        Args:
            other: Another Observable to combine with

        Returns:
            A MergedObservable containing both values as a tuple

        Example:
            ```python
            combined = obs1 | obs2  # Creates MergedObservable((obs1.value, obs2.value))
            combined2 = combined | obs3  # Creates MergedObservable((obs1.value, obs2.value, obs3.value))
            ```
        """
        from .merged import MergedObservable  # Import here to avoid circular import

        if isinstance(other, MergedObservable):
            # If other is already merged, combine our observable with its sources
            return MergedObservable(self, *other._source_observables)
        else:
            # Standard case: combine two regular observables
            return MergedObservable(self, other)

    def __rshift__(self, func: Callable) -> "Observable":
        """
        Chain a transformation function to create a new computed observable using >>.

        This is the functorial map operation: it applies a function to this observable,
        creating a new observable with the transformed values.

        Args:
            func: Function to apply to the observable's value(s).
                  For merged observables, receives the tuple values as separate arguments.

        Returns:
            A new Observable with computed values

        Examples:
            ```python
            # Single observable
            doubled = obs >> (lambda x: x * 2)

            # Merged observable
            combined = obs1 | obs2
            result = combined >> (lambda a, b: a + b)

            # Chaining
            final = obs >> func1 >> func2 >> func3
            ```
        """
        from .operators import rshift_operator

        return rshift_operator(self, func)

    def __and__(self, condition: "Observable[bool]") -> "ConditionalObservable[T]":
        """
        Create a conditional observable using the & operator.

        This creates a conditional observable that only emits values when the condition
        (and any additional conditions) are all True.

        Args:
            condition: A boolean Observable that must be True for emission

        Returns:
            A ConditionalObservable that filters updates based on the condition

        Example:
            ```python
            # Only emit when image is uploaded AND valid
            valid_image = uploaded_image & is_valid_image

            # Chain multiple conditions
            ready_to_process = uploaded_image & is_valid_image & ~is_processing
            ```
        """
        from .operators import and_operator

        return and_operator(self, condition)

    def __invert__(self) -> "Observable[bool]":
        """
        Create a negated boolean observable using the ~ operator.

        This creates a computed observable that returns the logical negation
        of the current observable's boolean value.

        Returns:
            An Observable[bool] with the negated value

        Example:
            ```python
            is_not_processing = ~is_processing  # True when is_processing is False
            ready_to_process = is_valid & ~is_processing
            ```
        """
        return self >> (lambda x: not x)
