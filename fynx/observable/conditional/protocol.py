"""
FynX Conditional Module Protocols - Conditional Observable Interface Definitions
=============================================================================

This module defines Protocol-based interfaces for conditional observables in FynX,
providing the interface for observables that filter values based on boolean conditions.

The Conditional protocol provides the interface for observables that only emit
values when ALL specified conditions are True, acting as smart gates that filter
reactive streams based on complex boolean logic.

Key Features:
- Filters values based on conditions
- Only emits when all conditions are satisfied
- Supports chaining with & operator
- Provides active/inactive state information
- Read-only reactive values

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
from ..types.common_types import Condition, ConditionFunction, T, TransformFunction, U

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from ..primitives.protocol import Observable
    from ..types.observable_protocols import DerivedObservable
else:
    # Import at runtime to avoid circular imports
    from ..primitives.protocol import Observable
    from ..types.observable_protocols import DerivedObservable

# ============================================================================
# CONDITIONAL OBSERVABLE PROTOCOL
# ============================================================================


class Conditional(DerivedObservable[T], Protocol[T]):
    """
    Protocol for observables that filter values based on boolean conditions.

    Conditional observables only emit values when ALL specified conditions
    are True. They act as smart gates that filter reactive streams based
    on complex boolean logic. This protocol composes DerivedObservableProtocol
    with ReactiveOperatorsProtocol for full functionality.

    Key Features:
    - Filters values based on conditions
    - Only emits when all conditions are satisfied
    - Supports chaining with & operator
    - Provides active/inactive state information
    - Read-only (inherits from DerivedObservable)
    - Inherits all DerivedObservable capabilities

    Example:
        ```python
        def process_valid_data(data: Conditional[str]) -> None:
            if data.is_active:
                print(f"Valid data: {data.value}")
            else:
                print("Data is not valid")
        ```
    """

    is_active: bool
