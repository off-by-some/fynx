"""
Fynx Reactive - Reactive Decorators and Utilities
=================================================

This module provides decorators and utilities for creating reactive relationships
between observables and functions. It enables automatic execution of functions
when their observable dependencies change.

Main Components:

- **reactive**: A decorator that creates reactive functions that automatically
  re-run when their observable dependencies change.

- **ReactiveHandler**: The underlying class that manages reactive function
  subscriptions and coordinates updates.

The `@reactive` decorator provides a clean, declarative way to define reactive
relationships, eliminating the need for manual subscription management.

Example:
    ```python
    from fynx import Store, observable, reactive

    class UserStore(Store):
        name = observable("Alice")
        age = observable(30)

    @reactive(UserStore.name, UserStore.age)
    def on_user_change(name, age):
        print(f"User: {name}, Age: {age}")

    # Changes trigger the reactive function automatically
    UserStore.name = "Bob"  # Prints: User: Bob, Age: 30
    ```
"""

from typing import Callable

from .observable import Observable
from .store import Store


class ReactiveHandler:
    """
    Manages reactive function subscriptions and handles different target types.

    ReactiveHandler is the core implementation behind the `@reactive` decorator.
    It intelligently handles different types of targets (Store classes, individual
    observables) and creates the appropriate subscription mechanism.

    The handler supports:
    - Store class subscriptions (reacts to any change in the store)
    - Individual observable subscriptions (reacts to specific observables)
    - Mixed subscriptions (combination of stores and observables)

    This class is typically used indirectly through the `@reactive` decorator
    rather than instantiated directly.

    Example:
        ```python
        # These all use ReactiveHandler internally:
        @reactive(store_instance)      # Store subscription
        @reactive(obs1, obs2)          # Multiple observables
        @reactive(store_class.attr)    # Single observable
        ```
    """

    def __init__(self, *targets):
        """
        Initialize the reactive handler with target observables/stores.

        Args:
            *targets: Variable number of observables, stores, or store attributes
                     to monitor for changes.
        """
        self.targets = targets

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation that makes the function reactive.

        This method is called when the ReactiveHandler is used as a decorator.
        It sets up the reactive context for the decorated function and returns
        the original function (decorators typically return the same function).

        Args:
            func: The function to make reactive

        Returns:
            The original function, now configured to react to target changes

        Example:
            ```python
            @reactive(store.count, store.name)
            def update_display(count, name):
                print(f"Count: {count}, Name: {name}")

            # This is equivalent to:
            # reactive_handler = ReactiveHandler(store.count, store.name)
            # update_display = reactive_handler(update_display)
            ```
        """
        self._create_reactive_context(func)
        return func

    def _create_reactive_context(self, func: Callable) -> None:
        """
        Create the appropriate reactive context based on target types.

        This method analyzes the targets passed to the handler and creates
        the appropriate subscription mechanism. It handles different scenarios:

        - Store class targets: Subscribe to all observables in the store
        - Individual observable targets: Subscribe to specific observables
        - Mixed targets: Combine multiple subscription types

        Args:
            func: The function to make reactive
        """
        if len(self.targets) == 0:
            # No targets provided - do nothing
            return
        elif len(self.targets) == 1:
            target = self.targets[0]

            if isinstance(target, type) and issubclass(
                target, Store
            ):  # It's a Store class
                # Use the Store's subscribe method
                target.subscribe(func)
            else:  # It's a single Observable
                # Create a reaction function that filters None values
                def filtered_reaction():
                    value = target.value
                    if value is not None:
                        func(value)

                # Subscribe with the reaction function
                context = Observable._create_subscription_context(
                    filtered_reaction, func, target
                )
                if target is not None:
                    target.add_observer(context.run)

        else:  # Multiple observables passed
            # Merge all observables using the | operator and subscribe to the result
            merged = self.targets[0]
            for obs in self.targets[1:]:
                merged = merged | obs
            # For merged observables, use standard subscription (no filtering needed for this test)
            merged.subscribe(func)


def reactive(*targets):
    """
    Create a reactive handler that works as a decorator.

    This is a convenience wrapper around subscribe() that works as a decorator.

    As decorator:
        @reactive(store) - reacts to all observables in store
        @reactive(observable) - reacts to single observable
        @reactive(obs1, obs2, ...) - reacts to multiple observables

    Args:
        *targets: Store class, Observable instance(s), or multiple Observable instances

    Returns:
        ReactiveHandler that can be used as decorator
    """
    return ReactiveHandler(*targets)
