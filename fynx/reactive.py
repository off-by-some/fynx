from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, overload

from typing_extensions import ParamSpec

from .observable import Observable
from .observable.descriptors import ObservableValue
from .observable.operands import unwrap_observable
from .store import Store, StoreSnapshot
from .types import ObservableOperand

P = ParamSpec("P")
R = TypeVar("R")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")
I = TypeVar("I")
J = TypeVar("J")


class ReactiveFunctionWasCalled(Exception):
    """Raised when a reactive function is called manually instead of through reactive triggers.

    A reactive function is meant to run only when its observable dependencies
    change, not be called directly. Modify the observables that trigger it
    instead, or call `.unsubscribe()` to turn it back into a plain callable.
    """

    pass


class ReactiveWrapper(Generic[P, R]):
    """
    Wraps a reactive function and manages its subscription lifecycle.

    While subscribed, the function runs automatically when its targets
    change, and calling it directly raises `ReactiveFunctionWasCalled`. After
    `unsubscribe()`, it reverts to a plain callable. The wrapper preserves
    function metadata (name, docstring) and tracks subscriptions internally.
    """

    def __init__(self, func: Callable[P, R], targets: tuple[Any, ...]):
        """
        Initialize the wrapper with the function and its reactive targets.

        Args:
            func: The original function to wrap
            targets: Tuple of observables/stores to react to
        """
        self._func = func
        self._targets = targets
        self._subscribed = False
        self._subscriptions: list[tuple[Any, Callable[..., Any]]] = []

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Call the wrapped function, raising an error if still subscribed.

        While subscribed, manual calls raise `ReactiveFunctionWasCalled`,
        since the function is meant to run only when its dependencies change.
        After `unsubscribe()`, this delegates to the original function normally.
        """
        if self._subscribed:
            raise ReactiveFunctionWasCalled(
                f"Reactive function {self.__name__} was called manually. "
                "Reactive functions should not be invoked manually, but rather be called automatically when their dependencies change. "
                f"Modify the observable values instead or call {self._func.__qualname__}.unsubscribe() to unsubscribe."
            )
        return self._func(*args, **kwargs)

    def _invoke_reactive(self, *args: Any, **kwargs: Any) -> R:
        """
        Internal method to invoke the function reactively (bypasses the check).

        Lets the subscription system trigger the function when observables
        change. Don't call this from outside the framework - use the
        observable's `set()` method or a store assignment instead.
        """
        return self._func(*args, **kwargs)

    def unsubscribe(self) -> None:
        """
        Unsubscribe from all reactive targets, making this a normal function again.

        Idempotent - calling it multiple times is safe. Afterward, the
        function stops responding to changes and can be called manually
        without raising `ReactiveFunctionWasCalled`.
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
                target = unwrap_observable(target)

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
            merged = unwrap_observable(self._targets[0])
            for obs in self._targets[1:]:
                merged = merged + unwrap_observable(obs)

            def merged_handler(*values):
                self._invoke_reactive(*values)

            # Call immediately with current values
            current_values = merged.value
            if current_values is not None:
                self._invoke_reactive(*current_values)

            # Subscribe
            merged.subscribe(merged_handler)
            self._subscriptions.append((merged, merged_handler))


@overload
def reactive() -> Callable[[Callable[P, R]], ReactiveWrapper[P, R]]: ...


@overload
def reactive(
    target: type[Store], /
) -> Callable[[Callable[[StoreSnapshot], R]], ReactiveWrapper[[StoreSnapshot], R]]: ...


@overload
def reactive(
    target: Observable[A], /
) -> Callable[[Callable[[A], R]], ReactiveWrapper[[A], R]]: ...


@overload
def reactive(
    target: ObservableValue[A], /
) -> Callable[[Callable[[A], R]], ReactiveWrapper[[A], R]]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    /,
) -> Callable[[Callable[[A, B], R]], ReactiveWrapper[[A, B], R]]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    /,
) -> Callable[[Callable[[A, B, C], R]], ReactiveWrapper[[A, B, C], R]]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    /,
) -> Callable[[Callable[[A, B, C, D], R]], ReactiveWrapper[[A, B, C, D], R]]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    /,
) -> Callable[[Callable[[A, B, C, D, E], R]], ReactiveWrapper[[A, B, C, D, E], R]]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    sixth: ObservableOperand[F],
    /,
) -> Callable[
    [Callable[[A, B, C, D, E, F], R]], ReactiveWrapper[[A, B, C, D, E, F], R]
]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    sixth: ObservableOperand[F],
    seventh: ObservableOperand[G],
    /,
) -> Callable[
    [Callable[[A, B, C, D, E, F, G], R]],
    ReactiveWrapper[[A, B, C, D, E, F, G], R],
]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    sixth: ObservableOperand[F],
    seventh: ObservableOperand[G],
    eighth: ObservableOperand[H],
    /,
) -> Callable[
    [Callable[[A, B, C, D, E, F, G, H], R]],
    ReactiveWrapper[[A, B, C, D, E, F, G, H], R],
]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    sixth: ObservableOperand[F],
    seventh: ObservableOperand[G],
    eighth: ObservableOperand[H],
    ninth: ObservableOperand[I],
    /,
) -> Callable[
    [Callable[[A, B, C, D, E, F, G, H, I], R]],
    ReactiveWrapper[[A, B, C, D, E, F, G, H, I], R],
]: ...


@overload
def reactive(
    first: ObservableOperand[A],
    second: ObservableOperand[B],
    third: ObservableOperand[C],
    fourth: ObservableOperand[D],
    fifth: ObservableOperand[E],
    sixth: ObservableOperand[F],
    seventh: ObservableOperand[G],
    eighth: ObservableOperand[H],
    ninth: ObservableOperand[I],
    tenth: ObservableOperand[J],
    /,
) -> Callable[
    [Callable[[A, B, C, D, E, F, G, H, I, J], R]],
    ReactiveWrapper[[A, B, C, D, E, F, G, H, I, J], R],
]: ...


@overload
def reactive(*targets: Any) -> Callable[[Callable[P, R]], ReactiveWrapper[P, R]]: ...


def reactive(*targets: Any) -> Any:
    """
    Create a reactive handler that works as a decorator.

    Declare which observables the function cares about and FynX handles
    subscribing and unsubscribing for you.

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

    def decorator(func: Callable[P, R]) -> ReactiveWrapper[P, R]:
        wrapper = ReactiveWrapper(func, targets)
        wrapper._setup_subscriptions()
        return wrapper

    return decorator
