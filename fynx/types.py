"""Shared public typing aliases for FynX."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    TypeVar,
    Union,
)

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from .observable.base import Observable
    from .observable.conditional import ConditionalObservable
    from .observable.descriptors import ObservableValue

T = TypeVar("T")

Observer: TypeAlias = Callable[[], Any]
ValueObserver: TypeAlias = Callable[[Any], object]
Subscriber: TypeAlias = Callable[[T], object]

SessionValue: TypeAlias = Union[
    None,
    str,
    int,
    float,
    bool,
    Dict[str, "SessionValue"],
    List["SessionValue"],
]
StoreState: TypeAlias = Dict[str, SessionValue]
StoreStateMapping: TypeAlias = Mapping[str, SessionValue]

ObservableOperand: TypeAlias = Union["Observable[T]", "ObservableValue[T]"]
ConditionOperand: TypeAlias = Union[
    "Observable[bool]",
    "ObservableValue[bool]",
    Callable[[T], bool],
    "ConditionalObservable[T]",
]
UnwrappedCondition: TypeAlias = Union[
    "Observable[bool]",
    Callable[[T], bool],
    "ConditionalObservable[T]",
]
