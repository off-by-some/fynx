"""
FynX Computed Module Protocols - Computed Observable Interface Definitions
========================================================================

This module defines Protocol-based interfaces for computed observables in FynX,
providing the interface for read-only reactive values that derive their values
from other observables through automatic computation.

The Computed protocol provides the interface for computed observables that are
read-only reactive values automatically calculating their value based on other
observables.

Key Features:
- Read-only reactive values
- Automatic computation from dependencies
- Lazy evaluation
- Dependency tracking
- Type safety

Key Benefits:
- No circular imports (protocols don't import concrete implementations)
- Better type safety than ABCs
- Runtime isinstance() support with @runtime_checkable
- Structural subtyping (duck typing with type safety)
- Clean separation of interface from implementation
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

# Import common types
from ...types.common_types import Condition, ConditionFunction, T, TransformFunction, U

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from .derived_protocol import DerivedObservable
    from .observable_protocol import Observable
else:
    # Import at runtime to avoid circular imports
    from .derived_protocol import DerivedObservable
    from .observable_protocol import Observable

# ============================================================================
# COMPUTED OBSERVABLE PROTOCOL
# ============================================================================


@runtime_checkable
class Computed(DerivedObservable[T], Protocol[T]):
    """
    Protocol for computed observables that derive their values from other observables.

    Computed observables are read-only reactive values that automatically
    calculate their value based on other observables. They provide derived
    state without manual synchronization, ensuring that computed values
    always stay in sync with their inputs.

    Key Features:
    - Read-only (cannot be set directly)
    - Automatic updates when dependencies change
    - Lazy evaluation (only computes when accessed)
    - Dependency tracking
    - Type safety

    Example:
        ```python
        def process_computed_value(computed: Computed[float]) -> None:
            print(f"Computed value: {computed.value}")
            # computed.set(5) would raise ValueError - computed values are read-only
        ```
    """

    def then(self, func: Callable[[T], U]) -> "Computed[U]":
        """
        Transform this computed observable's value with the given function.

        Args:
            func: A pure function to apply to the computed value.

        Returns:
            A new computed Observable containing the transformed value.
        """
        ...
