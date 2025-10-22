"""
FynX Core Observable Type Helpers
=================================

This module provides type helper functions for core observable types.
These functions use dynamic imports to avoid circular dependency concerns.
"""

from typing import Any


def is_observable(obj: Any) -> bool:
    """
    Check if an object is an Observable instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is an Observable, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.core.types import is_observable

        base = observable(5)

        print(is_observable(base))    # True
        print(is_observable(5))       # False
        ```
    """
    from fynx.observable.core.observable import Observable

    return isinstance(obj, Observable)


def is_base_observable(obj: Any) -> bool:
    """
    Check if an object is a BaseObservable instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is a BaseObservable, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.core.types import is_base_observable

        base = observable(5)

        print(is_base_observable(base))    # True
        print(is_base_observable(5))       # False
        ```
    """
    from fynx.observable.core.abstract.observable import BaseObservable

    return isinstance(obj, BaseObservable)


def is_derived_observable(obj: Any) -> bool:
    """
    Check if an object is a DerivedValue instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is a DerivedValue, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.core.types import is_derived_observable

        base = observable(5)
        computed = base.then(lambda x: x * 2)

        print(is_derived_observable(base))     # False
        print(is_derived_observable(computed)) # True
        ```
    """
    from fynx.observable.core.abstract.derived import DerivedValue

    return isinstance(obj, DerivedValue)


def get_observable_type(obj: Any) -> str:
    """
    Get a string representation of the observable type.

    Args:
        obj: The object to check

    Returns:
        String describing the observable type

    Example:
        ```python
        from fynx import observable
        from fynx.observable.core.types import get_observable_type

        base = observable(5)
        computed = base.then(lambda x: x * 2)

        print(get_observable_type(base))    # "Observable"
        print(get_observable_type(computed))  # "ComputedObservable"
        ```
    """
    return obj.__class__.__name__


def create_observable(*args, **kwargs):
    """
    Create an Observable instance with the given arguments.

    This factory function forwards all arguments to the Observable constructor.

    Args:
        *args: Positional arguments passed to Observable constructor
        **kwargs: Keyword arguments passed to Observable constructor

    Returns:
        A new Observable instance

    Example:
        ```python
        from fynx.observable.core.types import create_observable

        # Create observable with key and initial value
        obs = create_observable("counter", 0)
        print(obs.value)  # 0
        ```
    """
    from fynx.observable.core.observable import Observable

    return Observable(*args, **kwargs)
