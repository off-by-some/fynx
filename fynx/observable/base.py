"""
FynX Observable - core reactive value implementation
=====================================================

Observable wraps a value and tracks which reactive functions read it during
execution, so it can re-run them automatically when the value changes.

ReactiveContext is the execution scope for a reactive function: while the
function runs, reading an observable's `.value` registers that observable as
a dependency of the context. When the observable later changes, the context
re-runs the function. Contexts push onto a stack while running so that nested
reactive calls track their own dependencies correctly, and that same stack is
what lets circular-dependency detection work - if a computation tries to
modify one of its own inputs, it raises an error instead of looping.

Observable also implements `__str__` and `__bool__` so it behaves like the
value it wraps in boolean contexts and string formatting without needing
`.value` everywhere. Equality is identity-based so mutable graph nodes remain
safe in sets and dictionaries.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
    overload,
)
from weakref import WeakKeyDictionary

from ..equality import value_changed, values_equal
from ..registry import _all_reactive_contexts, _func_to_contexts
from ..types import Observer, Subscriber, ValueObserver
from .interfaces import Observable as ObservableInterface
from .interfaces import ReactiveContext as ReactiveContextInterface
from .operators import OperatorMixin

if TYPE_CHECKING:
    from .descriptors import ObservableValue

T = TypeVar("T")


_TRANSFORM_EVALUATION_STATE = [False]


def _transform_purity_message(action: str, key: str) -> str:
    verb = "read" if action == "read" else "mutated"
    operation = "Reading" if action == "read" else "Mutating"
    return (
        f"Observable '{key}' was {verb} inside a transform (`>>` / `.then()`).\n"
        f"{operation} observables from inside a transform makes dependencies implicit.\n"
        "FynX transforms are pure: they may only use values passed as arguments.\n"
        "Hint: pass every reactive input explicitly with `+` / `.alongside()`, for example:\n"
        "    (price + discount) >> (lambda price, discount: price * (1 - discount))\n"
        "Move side effects and mutations to `.subscribe()` or `@reactive`."
    )


class TransformPurityError(RuntimeError):
    """Raised when a transform tries to read or mutate observables implicitly."""


class ReactiveContext(ReactiveContextInterface):
    """
    Execution context for a reactive function, with automatic dependency tracking.

    While the wrapped function runs, the context watches which observables get
    read via `.value` and registers itself as an observer on each one, so that
    any of them changing triggers a re-run.

    Each run rebuilds the dependency set from scratch: old dependencies that
    weren't touched this time are dropped, and newly-read ones are added.
    Contexts push onto Observable's context stack while running, so nested
    reactive calls (a reactive function that triggers another) track their
    own dependencies independently rather than merging into the outer one.

    On dispose, the context removes itself from every observable it's
    subscribed to, so a discarded reactive function doesn't keep observers
    alive.

    Attributes:
        func: The reactive function to execute
        original_func: The original user function (for unsubscribe operations)
        subscribed_observable: The observable this context is subscribed to, if any
        dependencies: Set of observables accessed during execution
        is_running: Whether the context is currently executing

    Note:
        This is normally created for you by `@reactive` or `.subscribe()` -
        direct instantiation is rarely needed.

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
        func: Observer,
        original_func: Optional[Callable[..., Any]] = None,
        subscribed_observable: Optional["Observable"] = None,
    ) -> None:
        self.func = func
        self.original_func = (
            original_func or func
        )  # Store the original user function for unsubscribe
        self.subscribed_observable = (
            subscribed_observable  # The observable this context is subscribed to
        )
        self.dependencies: Set["Observable[Any]"] = set()
        self.is_running = False
        # For merged observables, we need to remove the observer from the merged observable,
        # not from the automatically tracked source observables
        self._observer_to_remove_from = subscribed_observable
        # For store subscriptions, keep track of all store observables
        self._store_observables: Optional[List["Observable[Any]"]] = None

    def run(self) -> None:
        """Run the reactive function, tracking dependencies."""
        old_context = Observable._current_context
        Observable._current_context = self

        # Push this context onto the stack
        Observable._context_stack.append(self)
        previous_dependencies = set(self.dependencies)
        self.dependencies = set()
        if self._observer_to_remove_from is not None:
            self.dependencies.add(self._observer_to_remove_from)

        try:
            self.is_running = True
            self.func()
        finally:
            self.is_running = False
            Observable._current_context = old_context
            # Pop this context from the stack
            Observable._context_stack.pop()
            for observable in previous_dependencies - self.dependencies:
                observable.remove_observer(self.run)

    def add_dependency(self, observable: "Observable[Any]") -> None:
        """Add an observable as a dependency of this context."""
        # Only add if not already a dependency to avoid redundant observer registration
        if observable not in self.dependencies:
            self.dependencies.add(observable)
            observable.add_observer(self.run)

            # Note: Instance dependency graphs are maintained separately
            # The reactive context handles the main dependency tracking

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

        for observable in tuple(self.dependencies):
            observable.remove_observer(self.run)

        self.dependencies.clear()


class Observable(OperatorMixin[T], ObservableInterface[T]):
    """
    A reactive value that notifies dependents automatically when it changes.

    Wrap any Python value in an Observable, and functions that read it during
    reactive execution (via a ReactiveContext) will re-run when it changes.
    Reading `.value` inside a reactive context registers the observable as a
    dependency and adds the context's re-run function as an observer; calling
    `.set()` later notifies those observers with the fresh value.

    Observable also implements `__bool__` and `__str__`, so it can
    be used in boolean contexts (`if observable:`), string formatting
    (`f"{observable}"`), without reaching for `.value` explicitly. Equality
    is identity-based so observables remain safe in dependency sets and dicts;
    compare `.value` explicitly for value equality.

    Changes are batched and notified in topological order - source
    observables first, then computed, then conditional - so a conditional
    observable never checks a condition value before it's been updated.

    Circular dependencies raise a RuntimeError rather than looping forever:
    if a computation tries to modify one of its own dependencies (directly or
    indirectly), setting the value fails instead of recursing.

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
        print(counter.value == 0)  # True
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
    _current_context: Optional[ReactiveContext] = None

    # Stack of reactive contexts being computed (for proper cycle detection)
    _context_stack: List[ReactiveContext] = []
    _computation_dependency_stack: List[Set["Observable[Any]"]] = []
    _dependency_capture_stack: List[Set["Observable[Any]"]] = []

    # High-performance notification system with cycle detection
    _pending_notifications: Set["Observable[Any]"] = set()
    _notification_scheduled: bool = False
    _currently_notifying: Set["Observable[Any]"] = set()  # Prevent cycles

    def _raise_if_transform_reads(self) -> None:
        if _TRANSFORM_EVALUATION_STATE[0]:
            raise TransformPurityError(_transform_purity_message("read", self._key))

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
        self._observers: Set[Observer] = set()
        self._version = 0
        self._is_notifying = False
        self._observer_snapshot: tuple[Observer, ...] = ()
        self._fast_observers: Set[ValueObserver] = set()
        self._fast_observer_snapshot: tuple[ValueObserver, ...] = ()
        self._single_fast_observer: Optional[ValueObserver] = None
        self._direct_observers: Dict[Callable[..., Any], Observer] = {}
        self._direct_callbacks: Set[ValueObserver] = set()
        self._direct_callback_snapshot: tuple[ValueObserver, ...] = ()
        self._single_direct_callback: Optional[ValueObserver] = None
        self._descriptor_name: Optional[str] = None
        self._descriptor_owner: Optional[type] = None
        self._descriptor_observables: WeakKeyDictionary[type, "Observable[T]"] = (
            WeakKeyDictionary()
        )

    @property
    def key(self) -> str:
        """Get the unique identifier for this observable."""
        return self._key

    @property
    def value(self) -> T:
        """
        Get the current value of this observable.

        Outside a reactive context this is a plain read. Inside one, reading
        it also registers this observable as a dependency of the current
        ReactiveContext, so a later `.set()` will re-run whatever depends on
        it.

        Returns:
            The current value stored in this observable.

        Note:
            Always read through `.value` rather than `_value` directly - the
            dependency tracking depends on it.

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
        self._raise_if_transform_reads()

        if Observable._dependency_capture_stack:
            Observable._dependency_capture_stack[-1].add(self)

        # Track dependency if we're in a reactive context
        if Observable._current_context is not None:
            Observable._current_context.add_dependency(self)
        return cast(T, self._value)

    def set(self, value: T) -> None:
        """
        Set the value and notify all observers if the value changed.

        The update only happens if the new value differs from the current
        one (via `!=`); if it's unchanged, observers aren't notified and no
        recomputation happens.

        Before updating, this checks whether the current reactive context
        already depends on this observable - if so, the set would create a
        cycle, and a RuntimeError is raised instead.

        When the value does change, notifications are queued and processed
        in topological order (source, then computed, then conditional), so a
        conditional observable never sees a stale condition value.

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
            Equality is checked defensively. If a value's equality operator
            raises or returns a non-boolean comparison object that cannot be
            coerced safely, FynX treats the assignment as changed and notifies.
        """
        if _TRANSFORM_EVALUATION_STATE[0]:
            raise TransformPurityError(_transform_purity_message("mutate", self._key))

        # Check for circular dependency: check if the current context
        # is computing a value that depends on this observable
        current_context = Observable._current_context
        is_notifying = self._is_notifying
        direct_callbacks = self._direct_callbacks
        if is_notifying and direct_callbacks:
            error_msg = f"Circular dependency detected in reactive computation!\n"
            error_msg += f"Observable '{self._key}' is being modified while notifying observers.\n"
            error_msg += f"This creates a circular dependency."
            raise RuntimeError(error_msg)

        if current_context and self in current_context.dependencies:
            error_msg = f"Circular dependency detected in reactive computation!\n"
            error_msg += f"Observable '{self._key}' is being modified while computing a value that depends on it.\n"
            error_msg += f"This creates a circular dependency."
            raise RuntimeError(error_msg)

        if Observable._computation_dependency_stack:
            error_msg = f"Circular dependency detected in reactive computation!\n"
            error_msg += (
                f"Observable '{self._key}' is being modified while computing a value.\n"
            )
            error_msg += f"This creates a circular dependency."
            raise RuntimeError(error_msg)

        # Only update and notify if the value actually changed
        if value_changed(self._value, value):
            self._value = value
            self._version += 1
            observers = self._observers
            fast_observers = self._fast_observers

            if observers:
                single_direct_callback = self._single_direct_callback
                if (
                    single_direct_callback is not None
                    and len(observers) == 1
                    and not fast_observers
                    and not Observable._notification_scheduled
                    and not is_notifying
                ):
                    self._is_notifying = True
                    try:
                        single_direct_callback(value)
                    finally:
                        self._is_notifying = False

                    if (
                        Observable._pending_notifications
                        and not Observable._notification_scheduled
                    ):
                        Observable._notification_scheduled = True
                        Observable._process_notifications()
                    return

            single_fast_observer = self._single_fast_observer
            if (
                single_fast_observer is not None
                and not observers
                and not Observable._notification_scheduled
                and not is_notifying
            ):
                self._is_notifying = True
                try:
                    single_fast_observer(value)
                finally:
                    self._is_notifying = False

                if (
                    Observable._pending_notifications
                    and not Observable._notification_scheduled
                ):
                    Observable._notification_scheduled = True
                    Observable._process_notifications()
                return
            if (
                direct_callbacks
                and len(direct_callbacks) == len(observers)
                and not fast_observers
                and not Observable._notification_scheduled
                and not is_notifying
            ):
                Observable._notify_direct_callbacks_then_drain(self, value)
                return
            if not observers and not fast_observers:
                return
            if fast_observers and not observers:
                Observable._notify_fast_observers_then_drain(self, value)
                return
            Observable._notify_inline_then_drain(self)
        else:
            # Even if the value didn't change, we still check for circular dependencies
            # in case the setter is being called from within its own computation
            pass

    def _notify_observers(self) -> None:
        """Notify all registered observers that this observable has changed."""
        # Create a copy of observers to avoid "Set changed size during iteration"
        # Prevent re-entrant notifications on this observable
        if not self._is_notifying:
            self._is_notifying = True
            try:
                for observer in self._observers_for_notification():
                    observer()
            finally:
                self._is_notifying = False

    def _observers_for_notification(self) -> tuple[Observer, ...]:
        """Return a stable observer snapshot for notification dispatch."""
        snapshot = self._observer_snapshot
        if len(snapshot) != len(self._observers):
            snapshot = tuple(self._observers)
            self._observer_snapshot = snapshot
        return snapshot

    def _fast_observers_for_notification(self) -> tuple[ValueObserver, ...]:
        """Return a stable source-only observer snapshot."""
        if not self._fast_observers:
            return ()
        snapshot = self._fast_observer_snapshot
        if len(snapshot) != len(self._fast_observers):
            snapshot = tuple(self._fast_observers)
            self._fast_observer_snapshot = snapshot
        return snapshot

    def _direct_callbacks_for_notification(self) -> tuple[ValueObserver, ...]:
        """Return a stable direct value-callback snapshot."""
        if not self._direct_callbacks:
            return ()
        snapshot = self._direct_callback_snapshot
        if len(snapshot) != len(self._direct_callbacks):
            snapshot = tuple(self._direct_callbacks)
            self._direct_callback_snapshot = snapshot
        return snapshot

    def _refresh_single_direct_callback(self) -> None:
        self._single_direct_callback = (
            next(iter(self._direct_callbacks))
            if len(self._direct_callbacks) == 1
            else None
        )

    @classmethod
    def _schedule_notification(cls, observable: "Observable[Any]") -> None:
        """Queue an observable for the current stabilization pass."""
        cls._pending_notifications.add(observable)
        if not cls._notification_scheduled and not observable._is_notifying:
            cls._notification_scheduled = True
            cls._process_notifications()

    @classmethod
    def _notify_inline_then_drain(cls, observable: "Observable[Any]") -> None:
        """Notify a root mutation immediately, then drain queued dependents."""
        if cls._notification_scheduled or observable._is_notifying:
            cls._schedule_notification(observable)
            return

        cls._notification_scheduled = True
        try:
            observable._is_notifying = True
            try:
                fast_observers = observable._fast_observers_for_notification()
                if len(fast_observers) == 1:
                    fast_observers[0](observable._value)
                else:
                    for fast_observer in fast_observers:
                        fast_observer(observable._value)

                observers = observable._observers_for_notification()
                if len(observers) == 1:
                    observers[0]()
                else:
                    for plain_observer in observers:
                        plain_observer()
            finally:
                observable._is_notifying = False

            if cls._pending_notifications:
                cls._process_notifications()
            else:
                cls._notification_scheduled = False
        except Exception:
            cls._notification_scheduled = False
            raise

    @classmethod
    def _notify_fast_observers_then_drain(
        cls, observable: "Observable[Any]", value: Any
    ) -> None:
        """Dispatch source-only transform observers without generic observer wrapping."""
        if cls._notification_scheduled or observable._is_notifying:
            cls._schedule_notification(observable)
            return

        cls._notification_scheduled = True
        try:
            observable._is_notifying = True
            try:
                observers = observable._fast_observers_for_notification()
                if len(observers) == 1:
                    observers[0](value)
                else:
                    for observer in observers:
                        observer(value)
            finally:
                observable._is_notifying = False

            if cls._pending_notifications:
                cls._process_notifications()
            else:
                cls._notification_scheduled = False
        except Exception:
            cls._notification_scheduled = False
            raise

    @classmethod
    def _notify_direct_callbacks_then_drain(
        cls, observable: "Observable[Any]", value: Any
    ) -> None:
        """Dispatch plain subscriptions directly at the effect boundary."""
        observable._is_notifying = True
        try:
            callback = observable._single_direct_callback
            if callback is not None:
                callback(value)
            else:
                for callback in observable._direct_callbacks_for_notification():
                    callback(value)
        finally:
            observable._is_notifying = False

        if cls._pending_notifications and not cls._notification_scheduled:
            cls._notification_scheduled = True
            cls._process_notifications()

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
        cls, observables: Set["Observable[Any]"]
    ) -> List["Observable[Any]"]:
        """
        Sort observables in topological order for correct notification processing.

        Dependencies must be notified before their dependents to ensure that when
        a conditional observable checks its condition values, they have been updated
        with the latest values.
        """
        if len(observables) <= 1:
            return list(observables)

        pending = set(observables)
        ordered_seed = sorted(pending, key=cls._notification_sort_key)
        incoming_count: Dict[Observable[Any], int] = {obs: 0 for obs in ordered_seed}
        outgoing: Dict[Observable[Any], List[Observable[Any]]] = {
            obs: [] for obs in ordered_seed
        }

        for obs in ordered_seed:
            for dependency in cls._observable_dependencies(obs):
                if dependency in pending:
                    incoming_count[obs] += 1
                    outgoing[dependency].append(obs)

        ready = [obs for obs in ordered_seed if incoming_count[obs] == 0]
        ordered = []

        while ready:
            observable = ready.pop(0)
            ordered.append(observable)

            for dependent in sorted(
                outgoing[observable], key=cls._notification_sort_key
            ):
                incoming_count[dependent] -= 1
                if incoming_count[dependent] == 0:
                    ready.append(dependent)
                    ready.sort(key=cls._notification_sort_key)

        if len(ordered) == len(ordered_seed):
            return ordered

        # If a cycle or incomplete dependency view slips through, keep the old
        # source/computed/conditional ordering as a conservative fallback.
        remaining = [obs for obs in ordered_seed if obs not in set(ordered)]
        return ordered + remaining

    @classmethod
    def _notification_sort_key(cls, observable: "Observable[Any]") -> tuple[int, int]:
        from .computed import ComputedObservable
        from .conditional import ConditionalObservable

        if isinstance(observable, ConditionalObservable):
            rank = 2
        elif isinstance(observable, ComputedObservable):
            rank = 1
        else:
            rank = 0
        return (rank, id(observable))

    @classmethod
    def _observable_dependencies(
        cls, observable: "Observable[Any]"
    ) -> Set["Observable[Any]"]:
        from .computed import ComputedObservable
        from .conditional import ConditionalObservable
        from .merged import MergedObservable

        if isinstance(observable, ConditionalObservable):
            return set(observable._all_dependencies)
        if isinstance(observable, MergedObservable):
            return set(observable._source_observables)
        if isinstance(observable, ComputedObservable):
            return set(observable._runtime_dependencies())
        return set()

    def add_observer(self, observer: Observer) -> None:
        """
        Add an observer function that will be called when this observable changes.

        Args:
            observer: A callable that takes no arguments
        """
        self._observers.add(observer)
        self._observer_snapshot = ()

    def remove_observer(self, observer: Observer) -> None:
        """
        Remove an observer function.

        Args:
            observer: The observer function to remove
        """
        self._observers.discard(observer)
        self._observer_snapshot = ()

    def add_fast_observer(self, observer: ValueObserver) -> None:
        """Add an internal source-only observer that receives the new value."""
        self._fast_observers.add(observer)
        self._fast_observer_snapshot = ()
        self._single_fast_observer = (
            observer if len(self._fast_observers) == 1 else None
        )

    def remove_fast_observer(self, observer: ValueObserver) -> None:
        """Remove an internal source-only observer."""
        self._fast_observers.discard(observer)
        self._fast_observer_snapshot = ()
        self._single_fast_observer = (
            next(iter(self._fast_observers)) if len(self._fast_observers) == 1 else None
        )

    def subscribe(self, func: Subscriber[T]) -> "Observable[T]":
        """
        Subscribe a function to react to changes in this observable.

        The function is called with the new value whenever `.set()` changes
        it - not immediately on subscription. Internally this wraps `func` in
        a ReactiveContext, so if it also reads other observables, it'll
        re-run when those change too.

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

        def direct_reaction():
            func(self._value)

        self._subscribe_direct_callback(func, direct_reaction)
        return self

    def unsubscribe(self, func: Subscriber[T]) -> None:
        """
        Unsubscribe a function from this observable.

        Args:
            func: The function to unsubscribe from this observable
        """
        self._unsubscribe_direct_callback(func)
        self._dispose_subscription_contexts(
            func, lambda ctx: ctx.subscribed_observable is self
        )

    def _subscribe_direct_callback(
        self, func: Callable[..., Any], observer: Observer
    ) -> None:
        """Register or replace a direct subscription observer."""
        existing_observer = self._direct_observers.pop(func, None)
        if existing_observer is not None:
            self.remove_observer(existing_observer)
            self._direct_callbacks.discard(func)

        self._direct_observers[func] = observer
        self._direct_callbacks.add(func)
        self._direct_callback_snapshot = ()
        self._refresh_single_direct_callback()
        self.add_observer(observer)

    def _unsubscribe_direct_callback(self, func: Callable[..., Any]) -> None:
        """Remove a direct subscription observer if it is currently registered."""
        direct_observer = self._direct_observers.pop(func, None)
        if direct_observer is not None:
            self.remove_observer(direct_observer)
            self._direct_callbacks.discard(func)
            self._direct_callback_snapshot = ()
            self._refresh_single_direct_callback()

    @staticmethod
    def _create_subscription_context(
        reaction_func: Observer,
        original_func: Callable[..., Any],
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
        func: Callable[..., Any],
        filter_predicate: Optional[Callable[[ReactiveContext], bool]] = None,
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
        self._raise_if_transform_reads()
        return bool(self._value)

    def __str__(self) -> str:
        """
        String representation of the wrapped value.

        Lets an observable drop straight into string contexts (f-strings,
        `str()`) without unwrapping it first.

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
        self._raise_if_transform_reads()
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        """Format the wrapped value with Python's normal format protocol."""
        self._raise_if_transform_reads()
        return format(self._value, format_spec)

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
        self._raise_if_transform_reads()
        return f"Observable({self._key!r}, {self._value!r})"

    def __eq__(self, other: object) -> bool:
        """
        Identity comparison with another object.

        Observables are mutable reactive graph nodes, so equality is based on
        node identity rather than wrapped value. This preserves Python's
        equality/hash contract when observables are used in dependency sets and
        dictionaries. Compare `.value` explicitly for value equality.

        Args:
            other: Value or Observable to compare with

        Returns:
            True only when `other` is this same observable object.

        Example:
            ```python
            obs1 = Observable("a", 5)
            obs2 = Observable("b", 5)
            regular_val = 5

            obs1 == obs2          # False (different graph nodes)
            obs1.value == obs2.value  # True
            obs1.value == regular_val # True
            ```
        """
        self._raise_if_transform_reads()
        return self is other

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

            # Hashing follows graph-node identity, not wrapped value
            hash(obs1) == id(obs1)  # True

            # But identical objects hash the same
            hash(obs1) == hash(obs1)  # True
            ```
        """
        return id(self)

    # Descriptor protocol for use as class attributes
    def __set_name__(self, owner: Type, name: str) -> None:
        """
        Called when this Observable is assigned to a class attribute.

        This method records the descriptor name and defining owner. Observable
        implements the descriptor protocol directly: class access returns an
        ObservableValue[T] wrapper, and StoreMeta records the owner-specific
        backing Observable for class-level assignment.

        Args:
            owner: The class that owns this attribute
            name: The name of the attribute being assigned

        Note:
            This method is called automatically by Python when an Observable
            instance is assigned to a class attribute.

        Example:
            ```python
            class MyClass:
                obs = Observable("counter", 0)  # __set_name__ called here

            instance = MyClass()
            print(instance.obs)  # Uses descriptor
            ```
        """
        self._descriptor_name = name
        if self._descriptor_owner is None:
            self._descriptor_owner = owner

        if self._key == "<unnamed>":
            if getattr(self, "_is_computed", False):
                self._key = f"<computed:{name}>"
            else:
                self._key = name

    def _observable_for_owner(self, owner: Type) -> "Observable[T]":
        if self._descriptor_owner is None or owner is self._descriptor_owner:
            return self

        observable = self._descriptor_observables.get(owner)
        if observable is None:
            key = self._descriptor_name or self._key
            observable = Observable(key, self._value)
            self._descriptor_observables[owner] = observable
        return observable

    @overload
    def __get__(self, instance: None, owner: Type) -> "ObservableValue[T]": ...

    @overload
    def __get__(self, instance: object, owner: Type) -> "ObservableValue[T]": ...

    def __get__(
        self, instance: Optional[object], owner: Optional[Type] = None
    ) -> "ObservableValue[T]":
        """
        Return a typed observable value wrapper when used as a class descriptor.

        Standalone Observable instances keep their normal behavior. When an
        Observable is placed on a class, descriptor access returns
        ObservableValue[T], which gives static type checkers the same shape users
        interact with at runtime.
        """
        if owner is None:
            if instance is None:
                raise AttributeError("Descriptor access requires an owner class")
            owner = type(instance)

        from .descriptors import ObservableValue

        return ObservableValue(self._observable_for_owner(owner))

    def __set__(self, instance: object, value: T) -> None:
        owner = type(instance)
        self._observable_for_owner(owner).set(value)
