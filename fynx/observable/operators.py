"""
Fynx Operators - Operator implementations for observables.

This module provides operator implementations. Imports are done lazily to avoid circular dependencies.
"""


def rshift_operator(obs, func):
    """Implementation of the >> operator for observables."""
    # Import here to avoid circular import
    from ..computed import computed

    return computed(func, obs)


def and_operator(obs, condition):
    """Implementation of the & operator for conditional observables."""
    # Import here to avoid circular import
    from .conditional import ConditionalObservable

    return ConditionalObservable(obs, condition)
