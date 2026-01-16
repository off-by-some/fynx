"""
FynX Observable - Core Reactive Value Implementation
====================================================

Reactive programming transforms static values into dynamic ones that automatically propagate changes. When you modify an observable value, every computation and function that depends on it updates automatically—no manual synchronization required. This module implements that foundation for Python.

Observable wraps a value and tracks which reactive functions access it during execution. When the value changes, those functions re-run automatically. We call this dependency tracking—the observable maintains a set of observers that get notified on change.

ReactiveContext manages the execution environment for reactive functions. Think of it as a scope that watches which observables get accessed during function execution. When an observable's value property is read, the context records that dependency. Later, when that observable changes, the context re-runs the function with fresh values.

The system uses a stack-based approach for nested reactive functions. Each context pushes itself onto a stack during execution, ensuring that dependencies are tracked correctly even when functions call other reactive functions. That stack also enables circular dependency detection—if a computation tries to modify one of its own inputs, the system raises an error.

Observable implements magic methods (`__eq__`, `__str__`, `__bool__`) to behave like its underlying value. You can use it in boolean contexts, string formatting, and equality comparisons without accessing `.value` explicitly. That transparency makes observables easy to integrate into existing code.

Result: values that look and feel like regular Python values, but propagate changes automatically through dependent computations and reactions.
"""

from typing import (
    Callable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)

from ..registry import _all_reactive_contexts, _func_to_contexts
from .interfaces import Observable as ObservableInterface
from .interfaces import ReactiveContext as ReactiveContextInterface
from .operators import OperatorMixin

T = TypeVar("T")


class ReactiveContext(ReactiveContextInterface):
    """
    Execution context for reactive functions with automatic dependency tracking.

    ReactiveContext manages the execution environment for reactive functions. During
    execution, it watches which observables get accessed and registers the function
    as an observer on each one. When any dependency changes, the context re-runs
    the function automatically.

    The context maintains a set of dependencies—observables that were accessed during the last execution. Each time the function runs, it clears old dependencies and builds a fresh set by tracking which observables' `.value` properties get accessed. That tracking happens transparently: reading `observable.value` within a reactive context automatically registers that observable as a dependency.

    The system uses a stack-based approach for nested reactive functions. When a context runs, it pushes itself onto Observable's context stack and sets itself as the current context. Nested reactive calls create new contexts that push onto the stack, ensuring that each level tracks its own dependencies correctly. When execution completes, the context pops itself from the stack and restores the previous context.

    For cleanup, the context tracks which observable it was originally subscribed to (if any). When disposed, it removes itself from that observable's observers and clears all dependency registrations. That prevents memory leaks when reactive functions are no longer needed.

    Attributes:
        func: The reactive function to execute
        original_func: The original user function (for unsubscribe operations)
        subscribed_observable: The observable this context is subscribed to, if any
        dependencies: Set of observables accessed during execution
        is_running: Whether the context is currently executing

    Note:
        This class is typically managed automatically by FynX's decorators and observable operations. Direct instantiation is rarely needed—use `@reactive` or `.subscribe()` instead.

    Example:
        ```python
        from fynx.observable.base import ReactiveContext, Observable

        def my_function():
            # This function accesses observables
            pass

        some_observable = Observable("test", 0)
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

            # Note: Instance dependency graphs are maintained separately
            # The reactive context handles the main dependency tracking

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

    Observable wraps any Python value and makes it reactive. When you modify the
    value, all computations and functions that depend on it recalculate automatically.
    Wrap any value in an Observable, and functions that read it during reactive
    execution will re-run when the value changes.

    The mechanism works through dependency tracking. When a reactive function executes, it runs within a ReactiveContext. That context watches which observables get accessed via their `.value` property. Each access registers the observable as a dependency and adds the context's re-run function as an observer. Later, when you call `.set()` on the observable, it notifies all observers, causing dependent functions to re-execute with fresh values.

    Observable implements magic methods to behave like its underlying value. You can use it in boolean contexts (`if observable:`), string formatting (`f"{observable}"`), and equality comparisons (`observable == 5`) without accessing `.value` explicitly. That transparency makes observables easy to integrate into existing code—they look and feel like regular values.

    The notification system uses batched processing with topological sorting. When multiple observables change, the system collects pending notifications and processes them in dependency order—source observables first, then computed observables, then conditional observables. That ordering ensures that when a conditional observable checks its condition values, those values have already been updated.

    Circular dependency detection prevents infinite loops. If a computation tries to modify one of its own dependencies (directly or indirectly), the system raises a RuntimeError. The detection works by checking whether the current reactive context depends on the observable being modified.

    Attributes:
        key: Unique identifier for debugging and serialization
        _value: The current wrapped value
        _observers: Set of observer functions to notify on change

    Class Attributes:
        _current_context: Current reactive execution context (None when not in reactive execution)
        _context_stack: Stack of nested reactive contexts for proper dependency tracking
        _pending_notifications: Set of observables waiting to notify observers
        _notification_scheduled: Whether notification processing is scheduled
        _currently_notifying: Set of observables currently notifying (prevents re-entrant notifications)

    Args:
        key: A unique identifier for this observable (used for debugging and serialization).
             If None, will be set to "<unnamed>" and updated in __set_name__ when used as a class attribute.
        initial_value: The initial value to store. Can be any type compatible with the generic type parameter.

    Raises:
        RuntimeError: If setting this value would create a circular dependency (e.g., a computed value trying to modify its own input).

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
        def on_change(new_value):
            print(f"Counter changed to: {new_value}")

        counter.subscribe(on_change)
        counter.set(5)  # Prints: "Counter changed to: 5"
        ```

    Note:
        While you can create Observable instances directly, it's often more convenient to use the `observable()` descriptor in Store classes for better organization and automatic serialization support.

    See Also:
        Store: For organizing observables into reactive state containers
        computed: For creating derived values from observables
        reactive: For creating reactive functions that respond to changes
    """

    # Class variable to track the current reactive context
    _current_context: Optional["ReactiveContext"] = None

    # Stack of reactive contexts being computed (for proper cycle detection)
    _context_stack: List["ReactiveContext"] = []

    # High-performance notification system with cycle detection
    _pending_notifications: Set["Observable"] = set()
    _notification_scheduled: bool = False
    _currently_notifying: Set["Observable"] = set()  # Prevent cycles

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
        self._key = key or "<unnamed>"
        self._value = initial_value
        self._observers: Set[Callable] = set()

    @property
    def key(self) -> str:
        """Get the unique identifier for this observable."""
        return self._key

    @property
    def value(self) -> Optional[T]:
        """
        Get the current value of this observable.

        Accessing this property reads the wrapped value. If called within a reactive context (during execution of a reactive function or computation), it also registers this observable as a dependency. That registration happens automatically—the current ReactiveContext (if any) adds this observable to its dependency set and registers itself as an observer.

        The dependency tracking enables automatic re-execution. When you later call `.set()` on this observable, all registered observers (including reactive contexts that depend on it) get notified and re-run their functions with fresh values.

        Returns:
            The current value stored in this observable, or None if not set.

        Note:
            This property is tracked by the reactive system. Use it instead of accessing `_value` directly to ensure proper dependency tracking. Outside reactive contexts, reading `.value` behaves like a regular property access.

        Example:
            ```python
            from fynx.observable import Observable
            from fynx import reactive

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

        This method updates the observable's wrapped value and triggers change notifications to all registered observers. The update only occurs if the new value differs from the current value (using `!=` comparison). If the value hasn't changed, observers are not notified—that avoids unnecessary re-computations.

        Before updating, the method checks for circular dependencies. If the current reactive context depends on this observable (directly or indirectly), setting the value would create a cycle. The system raises a RuntimeError in that case, preventing infinite loops.

        When the value does change, the observable adds itself to a pending notifications set. The notification system processes these in topological order—source observables first, then computed observables, then conditional observables. That ordering ensures that when a conditional observable checks its condition values, those values have already been updated.

        Args:
            value: The new value to set. Can be any type compatible with the observable's generic type parameter.

        Raises:
            RuntimeError: If setting this value would create a circular dependency (e.g., a computed value trying to modify its own input).

        Example:
            ```python
            from fynx.observable import Observable

            obs = Observable("counter", 0)
            obs.set(5)  # Triggers observers if value changed

            # No change, no notification
            obs.set(5)  # Same value, observers not called
            ```

        Note:
            Equality is checked using the `!=` operator, so custom objects should implement proper equality comparison if needed.
        """
        # Check for circular dependency: check if the current context
        # is computing a value that depends on this observable
        current_context = Observable._current_context
        if current_context and self in current_context.dependencies:
            error_msg = f"Circular dependency detected in reactive computation!\n"
            error_msg += f"Observable '{self._key}' is being modified while computing a value that depends on it.\n"
            error_msg += f"This creates a circular dependency."
            raise RuntimeError(error_msg)

        # Only update and notify if the value actually changed
        if self._value != value:
            self._value = value
            # Add to pending notifications for batched processing
            Observable._pending_notifications.add(self)
            # Schedule notification if not already scheduled and not currently notifying
            if (
                not Observable._notification_scheduled
                and self not in Observable._currently_notifying
            ):
                Observable._notification_scheduled = True
                # Process immediately for high performance (no deferral overhead)
                Observable._process_notifications()
        else:
            # Even if the value didn't change, we still check for circular dependencies
            # in case the setter is being called from within its own computation
            pass

    def _notify_observers(self) -> None:
        """Notify all registered observers that this observable has changed."""
        # Create a copy of observers to avoid "Set changed size during iteration"
        # Prevent re-entrant notifications on this observable
        if self not in Observable._currently_notifying:
            Observable._currently_notifying.add(self)
            try:
                for observer in list(self._observers):
                    observer()
            finally:
                Observable._currently_notifying.discard(self)

    @classmethod
    def _process_notifications(cls) -> None:
        """Process all pending notifications in topological order for correct dependency evaluation."""
        try:
            while cls._pending_notifications:
                pending = cls._pending_notifications.copy()
                cls._pending_notifications.clear()

                # Sort pending notifications in topological order (dependencies first)
                ordered_notifications = cls._topological_sort_notifications(pending)

                for observable in ordered_notifications:
                    observable._notify_observers()
        finally:
            cls._notification_scheduled = False

    @classmethod
    def _topological_sort_notifications(
        cls, observables: Set["Observable"]
    ) -> List["Observable"]:
        """
        Sort observables in topological order for correct notification processing.

        Dependencies must be notified before their dependents to ensure that when
        a conditional observable checks its condition values, they have been updated
        with the latest values.
        """
        # 1. Source observables (no computation) first
        # 2. Computed observables
        # 3. Conditional observables last (they depend on others)

        sources = []
        computed = []
        conditionals = []

        for obs in observables:
            from .computed import ComputedObservable
            from .conditional import ConditionalObservable

            if isinstance(obs, ConditionalObservable):
                conditionals.append(obs)
            elif isinstance(obs, ComputedObservable):
                computed.append(obs)
            else:
                sources.append(obs)

        # Return sources first, then computed, then conditionals
        return sources + computed + conditionals

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

        The subscribed function will be called whenever the observable's value changes. The function receives the new value as its single argument. This creates a ReactiveContext that wraps the function and registers it as an observer on this observable.

        When you call `.set()` with a new value, the observable notifies all observers. The subscription system ensures that your function runs with the updated value. The function is not called immediately upon subscription—only when the value actually changes.

        The subscription creates a reactive context internally. That context tracks dependencies if your function accesses other observables during execution, enabling automatic re-execution when those dependencies change as well.

        Args:
            func: A callable that accepts one argument (the new value). The function will be called whenever the observable's value changes.

        Returns:
            This observable instance for method chaining.

        Example:
            ```python
            from fynx.observable import Observable

            def on_change(new_value):
                print(f"Observable changed to: {new_value}")

            obs = Observable("counter", 0)
            obs.subscribe(on_change)

            obs.set(5)  # Prints: "Observable changed to: 5"
            ```

        Note:
            The function is called only when the observable's value changes. It is not called immediately upon subscription.

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
            # Note: String concatenation with + requires explicit .value access
            message = "User: " + str(obs)  # Works with str() conversion
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
        return f"Observable({self._key!r}, {self._value!r})"

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
        if self._key == "<unnamed>":
            # Check if this is a computed observable by checking for the _is_computed attribute
            if getattr(self, "_is_computed", False):
                self._key = f"<computed:{name}>"
            else:
                self._key = name

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
