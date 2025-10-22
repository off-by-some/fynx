"""
Function Flyweight Pattern
=========================

This module provides FunctionFlyweight for reusing common transformation functions
instead of creating new lambda functions each time.

Memory savings: ~100 bytes per lambda → ~8 bytes per reference

Common transformations include:
- Identity: x => x
- Arithmetic: x => x * n, x => x + n
- Type conversion: x => str(x)
"""

from typing import Any, Callable


class FunctionFlyweight:
    """
    Flyweight pattern for common transformation functions.

    Reuses function instances for common transformations instead of
    creating new lambda functions each time. This reduces memory usage
    and enables faster equality comparisons.

    Common transformations include:
    - Identity: x => x
    - Arithmetic: x => x * n, x => x + n
    - Type conversion: x => str(x)

    Memory savings: ~100 bytes per lambda → ~8 bytes per reference
    """

    # Global intern pool
    _pool: dict[tuple, Callable] = {}

    @staticmethod
    def get_identity() -> Callable:
        """Get identity function (x => x)."""
        return FunctionFlyweight._get_or_create("identity", lambda x: x)

    @staticmethod
    def get_multiply(n: int) -> Callable:
        """Get multiply function (x => x * n)."""
        key = ("multiply", n)
        return FunctionFlyweight._get_or_create(key, lambda x: x * n)

    @staticmethod
    def get_add(n: int) -> Callable:
        """Get add function (x => x + n)."""
        key = ("add", n)
        return FunctionFlyweight._get_or_create(key, lambda x: x + n)

    @staticmethod
    def get_to_string() -> Callable:
        """Get string conversion (x => str(x))."""
        return FunctionFlyweight._get_or_create("to_string", str)

    @staticmethod
    def _get_or_create(key: tuple, factory: Callable) -> Callable:
        """Retrieve function from pool or create and cache it."""
        if key not in FunctionFlyweight._pool:
            func = factory if callable(factory) else factory()
            FunctionFlyweight._pool[key] = func
        return FunctionFlyweight._pool[key]

    @staticmethod
    def clear_pool() -> None:
        """Clear the pool (for testing)."""
        FunctionFlyweight._pool.clear()
