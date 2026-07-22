"""Shared typing and runtime helpers for observable-like operands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from typing_extensions import TypeGuard

from ..types import ConditionOperand, ObservableOperand, UnwrappedCondition

T = TypeVar("T")

if TYPE_CHECKING:
    from .base import Observable


def is_observable_operand(value: Any) -> TypeGuard[ObservableOperand[Any]]:
    """Return whether a value is a public observable-like operand."""
    from .base import Observable
    from .descriptors import ObservableValue

    return isinstance(value, (Observable, ObservableValue))


def unwrap_observable(operand: ObservableOperand[T]) -> "Observable[T]":
    """Return the underlying Observable for an observable-like public operand."""
    from .descriptors import ObservableValue

    if isinstance(operand, ObservableValue):
        return cast("Observable[T]", operand.observable)
    return cast("Observable[T]", operand)


def unwrap_condition(condition: ConditionOperand[T]) -> UnwrappedCondition[T]:
    """Return the runtime condition behind a public condition operand."""
    from .descriptors import ObservableValue

    if isinstance(condition, ObservableValue):
        return cast("Observable[bool]", condition.observable)
    return cast("UnwrappedCondition[T]", condition)
