"""
Derived Observable Protocol
==========================

This module defines the DerivedObservable protocol for observables that extend
the base observable with additional capabilities.
"""

from typing import TYPE_CHECKING, Optional, Protocol, TypeVar, runtime_checkable

# Import common types
from ...types.common_types import T

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from .observable_protocol import Observable
else:
    # Import at runtime to avoid circular imports
    from .observable_protocol import Observable


@runtime_checkable
class DerivedObservable(Observable[T], Protocol[T]):
    """
    Protocol for derived observables that extend the base observable with additional capabilities.

    Derived observables are read-only reactive values that automatically
    calculate their value based on other observables or conditions. They
    provide derived state without manual synchronization.

    Key Features:
    - Read-only (cannot be set directly via .set())
    - Automatic recalculation when dependencies change
    - Lazy evaluation (only computes when accessed)
    - Extends base Observable with derived-specific methods

    Example:
        ```python
        def process_derived_value(derived: DerivedObservable[float]) -> None:
            print(f"Derived value: {derived.value}")
            # derived.set(5) would raise ValueError - derived values are read-only
        ```
    """

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating derived observable values.

        Warning: This method should only be called by the FynX framework internals.
        Direct use may break reactive relationships and is not supported.

        Args:
            value: The new derived value calculated from dependencies.
        """
        ...
