"""
FynX Common Types - Shared Type Definitions
==========================================

This module contains shared type definitions used across the FynX protocol system.
It helps avoid circular imports and provides a single source of truth for common types.

Key Benefits:
- Eliminates circular import issues
- Single source of truth for shared types
- Better type safety with forward references
- Cleaner protocol definitions
"""

from typing import TYPE_CHECKING, Any, Callable, Tuple, TypeVar, Union

# ============================================================================
# TYPE VARIABLES
# ============================================================================

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")  # Used in multi-step transformations

# ============================================================================
# FORWARD REFERENCES
# ============================================================================

if TYPE_CHECKING:
    from .protocols.observable_protocol import (
        Computed,
        Conditional,
        Mergeable,
        Observable,
    )

# ============================================================================
# CONDITION TYPES
# ============================================================================

# Union types for flexible condition handling
# Supports: Observable[bool], callables, Conditional instances, and raw bools
Condition = Union[
    "Observable[bool]",
    Callable[[Any], bool],
    "Conditional",
    bool,  # Support raw boolean values
]

# ============================================================================
# OPERATION FUNCTION TYPES
# ============================================================================

TransformFunction = Callable[[T], U]
ConditionFunction = Callable[[T], bool]
MergeFunction = Callable[[T, U], Any]

# ============================================================================
# TYPE CHECKING FUNCTIONS
# ============================================================================

TypeChecker = Callable[[Any], bool]

# ============================================================================
# FACTORY TYPES
# ============================================================================

# Factory functions for creating observables
ObservableFactory = Callable[[Any], "Observable[Any]"]
MergedFactory = Callable[[Any, ...], "Mergeable[Any]"]
ConditionalFactory = Callable[[Any, ...], "Conditional[Any]"]
ComputedFactory = Callable[[Callable[[], Any]], "Computed[Any]"]

# ============================================================================
# RESULT TYPES
# ============================================================================

# Helper for merge operation results
MergeResult = Union[Tuple[Any, ...], Any]
