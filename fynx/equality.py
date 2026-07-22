"""Value comparison helpers used for change detection."""

from __future__ import annotations

from typing import Any, Callable

ValueEquality = Callable[[Any, Any], bool]


def values_equal(left: Any, right: Any) -> bool:
    """Return whether two values are equal enough to skip notification.

    Python equality is not guaranteed to return a bool: NumPy arrays, Pandas
    objects, symbolic expressions, and user-defined classes may return rich
    comparison objects or raise when coerced. FynX treats non-boolean or
    failing equality as "changed" so updates are never silently dropped.
    """
    if left is right:
        return True

    try:
        result = left == right
    except Exception:
        return False

    if isinstance(result, bool):
        return result
    if result is NotImplemented:
        return False

    try:
        return bool(result)
    except Exception:
        return False


def value_changed(left: Any, right: Any) -> bool:
    """Return True when assigning ``right`` should notify dependents."""
    return not values_equal(left, right)
