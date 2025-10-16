"""
Fynx Observable Descriptors - Descriptor classes for observable attributes.

This module provides descriptor classes for creating observable class attributes.
"""

from typing import TYPE_CHECKING, Generic, Optional, Type, TypeVar

if TYPE_CHECKING:
    from .base import Observable

T = TypeVar('T')


class SubscriptableDescriptor(Generic[T]):
    """
    Descriptor for creating observable class attributes.

    This descriptor allows Store subclasses to define observable attributes
    that behave like regular class attributes but provide reactive behavior.
    """

    def __init__(self) -> None:
        self.attr_name = None
        self._initial_value = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.attr_name = name

    def __get__(self, instance: Optional[object], owner: Type) -> 'Observable[T]':
        """Get the observable instance for this attribute."""
        # Create class-level observable if it doesn't exist
        obs_key = f'_{self.attr_name}_observable'
        if not hasattr(owner, obs_key):
            # Import here to avoid circular import
            from .base import Observable
            observable = Observable(self.attr_name, self._initial_value)
            setattr(owner, obs_key, observable)

        return getattr(owner, obs_key)

    def __set__(self, instance: object, value: Optional[T]) -> None:
        """Set the value on the observable."""
        observable = self.__get__(instance, type(instance))
        observable.set(value)
