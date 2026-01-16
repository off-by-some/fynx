from typing import Callable

from .observable import Observable
from .store import Store, StoreSnapshot


class ReactiveFunctionWasCalled(Exception):
    """Raised when a reactive function is called manually instead of through reactive triggers.

    Reactive functions run automatically when their observable dependencies change.
    Manual invocation mixes reactive and imperative paradigms—the framework prevents this
    to maintain clear separation. Modify the observable values that trigger the function instead,
    or call `.unsubscribe()` to convert the function back to normal, callable form.
    """

    pass


class ReactiveWrapper:
    """
    Wraps a reactive function and manages its subscription lifecycle.

    Consider a thermostat: it monitors temperature and triggers heating when needed.
    You don't manually flip the switch—the system responds automatically. This wrapper
    enforces that pattern: while subscribed, the function runs reactively, not manually.
    After `unsubscribe()`, it reverts to normal function behavior—you can call it directly.

    The wrapper preserves function metadata (name, docstring) and tracks subscriptions
    internally. When targets change, it invokes the function automatically. Manual calls
    raise `ReactiveFunctionWasCalled` to prevent mixing paradigms.
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

        While subscribed, manual calls raise `ReactiveFunctionWasCalled`. This enforces
        the reactive contract: functions run automatically when dependencies change, not
        when you call them. After `unsubscribe()`, this method delegates to the original
        function normally.
        """
        if self._subscribed:
            raise ReactiveFunctionWasCalled(
                f"Reactive function {self.__name__} was called manually. "
                "Reactive functions should not be invoked manually, but rather be called automatically when their dependencies change. "
                f"Modify the observable values instead or call {self._func.__qualname__}.unsubscribe() to unsubscribe."
            )
        return self._func(*args, **kwargs)

    def _invoke_reactive(self, *args, **kwargs):
        """
        Internal method to invoke the function reactively (bypasses the check).

        This bypasses the manual-call protection, allowing the subscription system
        to trigger the function when observables change. External code should not
        call this directly—use the observable's `set()` method or store assignments
        to trigger reactive updates.
        """
        return self._func(*args, **kwargs)

    def unsubscribe(self):
        """
        Unsubscribe from all reactive targets, making this a normal function again.

        This severs the reactive connection: the function stops responding to changes
        and becomes callable normally. The operation is idempotent—calling it multiple
        times is safe. After unsubscription, you can call the function manually without
        raising `ReactiveFunctionWasCalled`.
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

        This method handles three cases: empty targets (no subscription), single target
        (Store class or Observable instance), and multiple targets (merged observables).
        For stores, it creates a snapshot handler. For observables, it handles conditional
        observables that may be inactive. The function executes immediately with current
        values when possible, then subscribes to future changes.
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

    This decorator bridges declarative state management with imperative side effects.
    Instead of manually subscribing and unsubscribing, you declare what observables
    matter and the framework handles the lifecycle.

    The decorator accepts three patterns:

    1. **Store subscription**: `@reactive(StoreClass)` reacts to all observables
       in the store, passing a `StoreSnapshot` to the function.

    2. **Single observable**: `@reactive(observable)` reacts to one observable,
       passing its current value to the function.

    3. **Multiple observables**: `@reactive(obs1, obs2, ...)` merges observables
       and passes their values as separate arguments.

    The function executes immediately with current values when decorated, then
    runs automatically whenever dependencies change. While subscribed, manual
    calls raise `ReactiveFunctionWasCalled`. Call `.unsubscribe()` to restore
    normal function behavior.

    Examples:
        ```python
        from fynx import observable, reactive, Store

        # Single observable
        count = observable(0)
        @reactive(count)
        def log_count(value):
            print(f"Count: {value}")

        count.set(5)  # Prints: "Count: 5"

        # Store subscription
        class UserStore(Store):
            name = observable("Alice")
            age = observable(30)

        @reactive(UserStore)
        def on_user_change(snapshot):
            print(f"User: {snapshot.name}, Age: {snapshot.age}")

        UserStore.name = "Bob"  # Triggers on_user_change

        # Multiple observables
        @reactive(UserStore.name, UserStore.age)
        def on_name_or_age(name, age):
            print(f"Name: {name}, Age: {age}")

        UserStore.age = 31  # Triggers on_name_or_age
        ```

    Args:
        *targets: Store class, Observable instance(s), or multiple Observable instances

    Returns:
        ReactiveWrapper instance that acts like the original function but prevents
        manual calls while subscribed.
    """

    def decorator(func: Callable) -> ReactiveWrapper:
        wrapper = ReactiveWrapper(func, targets)
        wrapper._setup_subscriptions()
        return wrapper

    return decorator
