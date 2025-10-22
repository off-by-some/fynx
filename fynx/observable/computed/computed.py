"""
Simplified ComputedObservable - Much Cleaner Implementation
==========================================================

Key improvements:
1. Uses new DerivedValue base class with template method pattern
2. All complexity handled by base class
3. ComputedObservable is now just ~50 lines instead of 500+
4. Cleaner, more maintainable code
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from fynx.observable.core.abstract.derived import DerivedValue

if TYPE_CHECKING:
    from fynx.observable.core.abstract.observable import BaseObservable

T = TypeVar("T")


class ComputedObservable(DerivedValue[T]):
    """
    Computed observable - much simpler now!

    All complexity is handled by DerivedValue base class.
    This class just implements the computation logic.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        computation_func: Optional[Callable] = None,
        source_observable: Optional["BaseObservable"] = None,
    ):
        self._computation_func = computation_func
        super().__init__(key, initial_value, source_observable)

    def _compute_value(self) -> T:
        """Apply computation function to source value."""
        if self._computation_func is None:
            return self._value_wrapper._value

        source_value = self._source_observable.value
        return self._computation_func(source_value)
