from typing import Callable

from .observable import Observable
from .store import Store, StoreSnapshot


class ReactiveFunctionWasCalled(Exception):
    """Raised when a reactive function is called manually instead of through reactive triggers.

    Reactive functions are designed to run automatically when their observable dependencies change.
    Manually calling them mixes reactive and imperative paradigms and should be avoided.

    Instead of calling reactive functions directly, modify the observable values that trigger them.
    """

    pass


class ReactiveWrapper:
    """
    Wraps a reactive function and manages its subscription lifecycle.

    This wrapper acts like the original function but prevents manual calls
    while subscribed. After unsubscribe(), it becomes a normal function again.
    """

    def __init__(self, func: Callable, targets: tuple):
        """
        Initialize the wrapper with the function and its reactive targets.

        Args:
            func: The original function to wrap
            targets: Tuple of observables/stores to react to
        """
        self._func = func
        self._targets = targets
        self._subscribed = False
        self._subscriptions = []  # Track what we subscribed to

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped function, raising an error if still subscribed.
        """
        if self._subscribed:
            raise ReactiveFunctionWasCalled(
                f"Reactive function {self.__name__} was called manually. "
                "Reactive functions should only not be invoked, but rather be called automatically when their dependencies change. "
                f"Modify the observable values instead or call {self._func.__qualname__}.unsubscribe() to unsubscribe."
            )
        return self._func(*args, **kwargs)

    def _invoke_reactive(self, *args, **kwargs):
        """
        Internal method to invoke the function reactively (bypasses the check).
        """
        return self._func(*args, **kwargs)

    def unsubscribe(self):
        """
        Unsubscribe from all reactive targets, making this a normal function again.
        """
        if not self._subscribed:
            return  # Already unsubscribed, idempotent

        # Unsubscribe from each target
        for target, handler in self._subscriptions:
            target.unsubscribe(handler)

        self._subscriptions.clear()
        self._subscribed = False

    def _setup_subscriptions(self):
        """
        Set up the reactive subscriptions based on targets.
        """
        self._subscribed = True

        if len(self._targets) == 0:
            return
        elif len(self._targets) == 1:
            target = self._targets[0]

            if isinstance(target, type) and issubclass(target, Store):
                # Store subscription
                def store_handler(snapshot):
                    self._invoke_reactive(snapshot)

                # Call immediately with current state
                snapshot = StoreSnapshot(target, target._observable_attrs)
                snapshot._take_snapshot()
                self._invoke_reactive(snapshot)

                # Subscribe
                target.subscribe(store_handler)
                self._subscriptions.append((target, store_handler))

            else:
                # Single observable subscription
                def observable_handler():
                    from .observable.conditional import ConditionalObservable

                    if (
                        isinstance(target, ConditionalObservable)
                        and not target.is_active
                    ):
                        # Don't call reactive function when conditional is not active
                        return
                    # For conditionals, we know they're active, so value access is safe
                    current_value = target.value
                    self._invoke_reactive(current_value)

                # Call immediately (if possible)
                from .observable.conditional import ConditionalObservable

                if isinstance(target, ConditionalObservable) and not target.is_active:
                    # Don't call reactive function when conditional is not active
                    pass
                else:
                    current_value = target.value
                    self._invoke_reactive(current_value)

                # Subscribe
                context = Observable._create_subscription_context(
                    observable_handler, self._func, target
                )
                if target is not None:
                    target.add_observer(context.run)
                    self._subscriptions.append((target, self._func))
        else:
            # Multiple observables - merge them
            merged = self._targets[0]
            for obs in self._targets[1:]:
                merged = merged + obs

            def merged_handler(*values):
                self._invoke_reactive(*values)

            # Call immediately with current values
            current_values = merged.value
            if current_values is not None:
                self._invoke_reactive(*current_values)

            # Subscribe
            merged.subscribe(merged_handler)
            self._subscriptions.append((merged, merged_handler))


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
        ReactiveWrapper instance that acts like the original function
    """

    def decorator(func: Callable) -> ReactiveWrapper:
        wrapper = ReactiveWrapper(func, targets)
        wrapper._setup_subscriptions()
        return wrapper

    return decorator
