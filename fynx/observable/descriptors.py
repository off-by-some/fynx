"""
FynX Observable Descriptors - Descriptor classes for observable attributes.

This module provides descriptor classes for creating observable class attributes.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    from .base import Observable

T = TypeVar("T")


class ObservableValue(Generic[T]):
    """
    A value-like object that allows assignment and provides observable access.
    This represents the actual value that can be assigned to, while also providing
    access to observable methods.
    """

    def __init__(self, observable: "Observable[T]") -> None:
        self._observable = observable
        self._current_value = observable.value

        # Subscribe to updates to keep _current_value in sync
        def update_value(new_value):
            self._current_value = new_value

        observable.subscribe(update_value)

    def __eq__(self, other) -> bool:
        return self._current_value == other

    def __str__(self) -> str:
        return str(self._current_value)

    def __repr__(self) -> str:
        return repr(self._current_value)

    def __len__(self) -> int:
        if self._current_value is None:
            return 0
        if hasattr(self._current_value, "__len__"):
            return len(self._current_value)
        return 0

    def __iter__(self):
        if self._current_value is None:
            return iter([])
        if hasattr(self._current_value, "__iter__"):
            return iter(self._current_value)
        return iter([self._current_value])

    def __getitem__(self, key):
        if self._current_value is None:
            raise IndexError("observable value is None")
        if hasattr(self._current_value, "__getitem__"):
            return self._current_value[key]
        raise TypeError(
            f"'{type(self._current_value).__name__}' object is not subscriptable"
        )

    def __contains__(self, item) -> bool:
        if self._current_value is None:
            return False
        if hasattr(self._current_value, "__contains__"):
            return item in self._current_value
        return False

    def __bool__(self) -> bool:
        return bool(self._current_value)

    # Observable methods
    @property
    def value(self) -> Optional[T]:
        return self._observable.value

    def set(self, value: Optional[T]) -> None:
        self._observable.set(value)
        self._current_value = value

    def subscribe(self, func) -> "Observable[T]":
        return self._observable.subscribe(func)

    def __or__(self, other):
        """Support merging observables with | operator."""
        from .merged import MergedObservable

        if hasattr(other, "observable"):
            return MergedObservable(self._observable, other.observable)
        return MergedObservable(self._observable, other)

    def __rshift__(self, func):
        """Support computed observables with >> operator."""
        return self._observable >> func

    @property
    def observable(self) -> "Observable[T]":
        """Get the underlying observable instance."""
        return self._observable

    def __getattr__(self, name: str):
        return getattr(self._observable, name)


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor for creating observable class attributes.

    This descriptor allows Store subclasses to define observable attributes
    that behave like regular class attributes but provide reactive behavior.
    """

    def __init__(
        self,
        initial_value: Optional[T] = None,
        original_observable: Optional["Observable[T]"] = None,
    ) -> None:
        self.attr_name: Optional[str] = None
        self._initial_value: Optional[T] = initial_value
        self._original_observable: Optional["Observable[T]"] = original_observable
        self._owner_class: Optional[Type] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.attr_name = name
        self._owner_class = owner

    def __get__(self, instance: Optional[object], owner: Optional[Type]) -> Any:
        """Get the observable value for this attribute."""
        # Use the stored owner class if available, otherwise use the passed owner
        target_class = self._owner_class or owner
        if target_class is None:
            raise AttributeError("Descriptor not properly initialized")

        # Create class-level observable if it doesn't exist
        obs_key = f"_{self.attr_name}_observable"
        if not hasattr(target_class, obs_key):
            # Use the original observable if provided, otherwise create a new one
            if self._original_observable is not None:
                obs = self._original_observable
            else:
                # Import here to avoid circular import
                from .base import Observable

                obs = Observable(self.attr_name or "unknown", self._initial_value)
            setattr(target_class, obs_key, obs)

        retrieved_obs = getattr(target_class, obs_key)
        return ObservableValue(retrieved_obs)  # type: ignore

    def __set__(self, instance: Optional[object], value: Optional[T]) -> None:
        """Set the value on the observable."""
        # For both class and instance access, we need the class
        target_class = self._owner_class
        if target_class is None:
            if instance is not None:
                target_class = type(instance)
            else:
                raise AttributeError("Cannot set value on uninitialized descriptor")

        observable = getattr(target_class, f"_{self.attr_name}_observable")
        observable.set(value)
