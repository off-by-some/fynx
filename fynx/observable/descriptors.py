"""
Fynx Observable Descriptors - Descriptor classes for observable attributes.

This module provides descriptor classes for creating observable class attributes.
"""

from typing import TYPE_CHECKING, Any, Generic, Optional, Type, TypeVar, Union

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

    def __eq__(self, other) -> bool:
        return self._current_value == other

    def __str__(self) -> str:
        return str(self._current_value)

    def __repr__(self) -> str:
        return repr(self._current_value)

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

        if hasattr(other, "_observable"):
            return MergedObservable(self._observable, other._observable)
        return MergedObservable(self._observable, other)

    def __getattr__(self, name: str):
        return getattr(self._observable, name)


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor for creating observable class attributes.

    This descriptor allows Store subclasses to define observable attributes
    that behave like regular class attributes but provide reactive behavior.
    """

    def __init__(self, initial_value: Optional[T] = None) -> None:
        self.attr_name: Optional[str] = None
        self._initial_value: Optional[T] = initial_value
        self._owner_class: Optional[Type] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.attr_name = name
        self._owner_class = owner

    def __get__(
        self, instance: Optional[object], owner: Optional[Type]
    ) -> ObservableValue[T]:
        """Get the observable value for this attribute."""
        # Use the stored owner class if available, otherwise use the passed owner
        target_class = self._owner_class or owner
        if target_class is None:
            raise AttributeError("Descriptor not properly initialized")

        # Create class-level observable if it doesn't exist
        obs_key = f"_{self.attr_name}_observable"
        if not hasattr(target_class, obs_key):
            # Import here to avoid circular import
            from .base import Observable

            observable: Observable[T] = Observable(
                self.attr_name or "unknown", self._initial_value
            )
            setattr(target_class, obs_key, observable)

        obs = getattr(target_class, obs_key)
        return ObservableValue(obs)

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
