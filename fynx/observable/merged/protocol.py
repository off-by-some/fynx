"""
FynX Merged Module Protocols - Merged Observable Interface Definitions
=====================================================================

This module defines Protocol-based interfaces for merged observables in FynX,
providing the interface for observables that can be merged with others using
the + operator.

The Mergeable protocol provides the interface for observables that combine
multiple source observables into tuples, updating when any of their components
change. This enables coordinated reactive updates across related values.

Key Features:
- Combines multiple observables into tuples
- Updates when any source observable changes
- Supports chaining with + operator
- Tuple-like behavior for accessing combined values
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
    from ..operations.protocol import TupleBehavior
    from ..primitives.protocol import Observable
    from ..types.observable_protocols import DerivedObservable
else:
    # Import at runtime to avoid circular imports
    from ..operations.protocol import TupleBehavior
    from ..primitives.protocol import Observable
    from ..types.observable_protocols import DerivedObservable

# ============================================================================
# MERGEABLE OBSERVABLE PROTOCOL
# ============================================================================


@runtime_checkable
class Mergeable(DerivedObservable[T], TupleBehavior[T], Protocol[T]):
    """
    Protocol for observables that can be merged with others using the + operator.

    Merged observables combine multiple source observables into tuples,
    updating when any of their components change. This enables coordinated
    reactive updates across related values. This protocol composes
    DerivedObservable with TupleBehavior for tuple-specific functionality.

    Key Features:
    - Combines multiple observables into tuples
    - Updates when any source observable changes
    - Supports chaining with + operator
    - Tuple-like behavior for accessing combined values
    - Read-only (inherits from DerivedObservable)
    - Inherits all DerivedObservable capabilities

    Example:
        ```python
        def process_coordinates(coords: Mergeable[tuple[int, int]]) -> None:
            x, y = coords[0], coords[1]  # Tuple indexing
            for coord in coords:        # Tuple iteration
                print(coord)
        ```
    """

    _source_observables: List[Observable]
