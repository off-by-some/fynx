"""
FynX Generic Module Protocols - Generic Observable Interface Definitions
======================================================================

This module defines Protocol-based interfaces for generic observables in FynX,
providing the interface for generic observable utility functions and factory methods.

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
from ..types.common_types import T, U

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from ..primitives.protocol import Observable
