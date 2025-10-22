"""
Type helpers for computed observables.

This module provides utility functions to check if objects are instances
of specific computed observable types. Functions use dynamic imports to
avoid circular dependency issues.
"""

from typing import Any


def is_computed_observable(obj: Any) -> bool:
    """
    Check if an object is a ComputedObservable instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is a ComputedObservable, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.computed.types import is_computed_observable

        base = observable(5)
        computed = base.then(lambda x: x * 2)

        print(is_computed_observable(base))    # False
        print(is_computed_observable(computed))  # True
        ```
    """
    from fynx.observable.computed.computed import ComputedObservable

    return isinstance(obj, ComputedObservable)


def is_conditional_observable(obj: Any) -> bool:
    """
    Check if an object is a ConditionalObservable instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is a ConditionalObservable, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.computed.types import is_conditional_observable

        base = observable(5)
        conditional = base & (lambda x: x > 0)

        print(is_conditional_observable(base))      # False
        print(is_conditional_observable(conditional))  # True
        ```
    """
    from fynx.observable.computed.conditional import ConditionalObservable

    return isinstance(obj, ConditionalObservable)


def is_merged_observable(obj: Any) -> bool:
    """
    Check if an object is a MergedObservable instance.

    Args:
        obj: The object to check

    Returns:
        True if obj is a MergedObservable, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.computed.types import is_merged_observable

        x = observable(10)
        y = observable(20)
        merged = x + y

        print(is_merged_observable(x))      # False
        print(is_merged_observable(merged))  # True
        ```
    """
    from fynx.observable.computed.merged import MergedObservable

    return isinstance(obj, MergedObservable)


def is_derived_observable(obj: Any) -> bool:
    """
    Check if an object is any type of derived observable (computed, conditional, or merged).

    Args:
        obj: The object to check

    Returns:
        True if obj is any derived observable type, False otherwise

    Example:
        ```python
        from fynx import observable
        from fynx.observable.computed.types import is_derived_observable

        base = observable(5)
        computed = base.then(lambda x: x * 2)
        conditional = base & (lambda x: x > 0)

        print(is_derived_observable(base))      # False
        print(is_derived_observable(computed))  # True
        print(is_derived_observable(conditional))  # True
        ```
    """
    return (
        is_computed_observable(obj)
        or is_conditional_observable(obj)
        or is_merged_observable(obj)
    )


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
        from fynx.observable.computed.types import get_observable_type

        base = observable(5)
        computed = base.then(lambda x: x * 2)

        print(get_observable_type(base))    # "Observable"
        print(get_observable_type(computed))  # "ComputedObservable"
        ```
    """
    return obj.__class__.__name__
