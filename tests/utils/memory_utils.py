"""
Memory testing utilities for reactive systems.

These utilities help verify that reactive components properly clean up resources
and don't leak memory during operation.

Examples:
    Basic leak detection:

        >>> from tests.utils.memory_utils import assert_no_object_leak
        >>> def operation():
        ...     obs = observable(0)
        ...     temp = obs >> (lambda x: x * 2)
        ...     del temp
        >>> assert_no_object_leak(operation, 'Observable')

    Context manager for detailed tracking:

        >>> with MemoryTracker('Observable') as tracker:
        ...     obs = observable(0)
        ...     obs.set(100)
        >>> tracker.assert_no_growth(tolerance=5)

    Decorator for automatic tracking:

        >>> @with_memory_tracking('Observable')
        ... def my_test():
        ...     # ... test code that shouldn't leak observables
        ...     pass
"""

import gc
import sys
import weakref
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Optional


def assert_cleaned_up(
    obj: Any, description: str = "Object should be cleaned up"
) -> None:
    """Assert that an object gets garbage collected after deletion.

    Args:
        obj: The object to test for cleanup
        description: Custom description for the assertion failure
    """
    obj_ref = weakref.ref(obj)

    # Delete the object
    del obj
    gc.collect()

    assert obj_ref() is None, f"{description}: object was not cleaned up"


def count_types() -> Dict[str, int]:
    """Count instances of each object type currently in memory.

    Returns:
        Dictionary mapping type names to counts
    """
    gc.collect()
    counts = defaultdict(int)
    for obj in gc.get_objects():
        counts[type(obj).__name__] += 1
    return counts


def get_total_size() -> int:
    """Get shallow memory size of all objects with attributes.

    Note: This uses sys.getsizeof which only measures the object itself,
    not referenced data. Useful for detecting accumulation of object
    instances, not comprehensive memory profiling.

    Returns:
        Total size in bytes
    """
    gc.collect()
    return sum(
        sys.getsizeof(obj) for obj in gc.get_objects() if hasattr(obj, "__dict__")
    )


def assert_no_object_leak(
    operation: Callable[[], None],
    type_name: str,
    tolerance: int = 5,
    description: Optional[str] = None,
) -> None:
    """Assert that an operation doesn't leak objects of a specific type.

    Args:
        operation: Function to execute that should not create persistent objects
        type_name: Name of the object type to monitor (e.g., 'Observable')
        tolerance: Allowed variance in object count
        description: Custom description for assertion failures
    """
    if description is None:
        description = f"Operation should not leak {type_name} objects"

    initial_counts = count_types()
    initial_count = initial_counts.get(type_name, 0)

    operation()

    final_counts = count_types()
    final_count = final_counts.get(type_name, 0)

    assert (
        abs(final_count - initial_count) <= tolerance
    ), f"{description}: {type_name} count changed from {initial_count} to {final_count}"


def measure_memory_growth(operation: Callable[[], None]) -> int:
    """Measure memory growth caused by an operation.

    Args:
        operation: Function to execute and measure

    Returns:
        Memory growth in bytes
    """
    size_before = get_total_size()
    operation()
    size_after = get_total_size()
    return size_after - size_before


def test_sublinear_growth(
    base_operation: Callable[[int], None], batches: list = None
) -> None:
    """Test that memory growth is sublinear with operation count.

    Args:
        base_operation: Function that takes batch size and performs operations
        batches: List of batch sizes to test (default: [100, 200, 400])
    """
    if batches is None:
        batches = [100, 200, 400]

    measurements = []

    for batch in batches:

        def operation():
            base_operation(batch)

        growth = measure_memory_growth(operation)
        measurements.append((batch, growth))

    # Growth rate should decrease (or stay constant)
    growth_rates = [m[1] / m[0] for m in measurements]

    # Each doubling of operations shouldn't cause proportional memory growth
    assert (
        growth_rates[1] < growth_rates[0] * 1.5
    ), f"Growth rate increased from {growth_rates[0]} to {growth_rates[1]}"
    assert (
        growth_rates[2] < growth_rates[0] * 1.5
    ), f"Growth rate increased from {growth_rates[0]} to {growth_rates[2]}"


class MemoryTracker:
    """Context manager for tracking memory changes during operations."""

    def __init__(self, type_name: str = None):
        self.type_name = type_name
        self.initial_counts = None
        self.initial_size = None

    def __enter__(self):
        self.initial_counts = count_types()
        self.initial_size = get_total_size()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def object_growth(self) -> Dict[str, int]:
        """Get the change in object counts since entering the context."""
        final_counts = count_types()
        growth = {}
        for type_name, final_count in final_counts.items():
            initial_count = self.initial_counts.get(type_name, 0)
            if final_count != initial_count:
                growth[type_name] = final_count - initial_count
        return growth

    @property
    def memory_growth(self) -> int:
        """Get the memory growth since entering the context."""
        return get_total_size() - self.initial_size

    def assert_no_growth(self, type_name: str = None, tolerance: int = 0):
        """Assert no object growth occurred."""
        target_type = type_name or self.type_name
        if target_type:
            growth = self.object_growth.get(target_type, 0)
            assert (
                abs(growth) <= tolerance
            ), f"{target_type} count changed by {growth} (tolerance: {tolerance})"
        else:
            # Check that no types grew
            for type_name, growth in self.object_growth.items():
                assert growth <= tolerance, f"{type_name} count grew by {growth}"


def with_memory_tracking(type_name: Optional[str] = None):
    """Decorator to track memory changes during function execution.

    Args:
        type_name: Specific object type to monitor for leaks

    Example:
        @with_memory_tracking('Observable')
        def my_operation():
            # ... do something that might leak observables
            pass
    """

    def decorator(func):
        @wraps(func)  # Preserves func.__name__, __doc__, etc.
        def wrapper(*args, **kwargs):
            with MemoryTracker(type_name) as tracker:
                result = func(*args, **kwargs)

            if type_name:
                tracker.assert_no_growth(type_name)

            return result

        return wrapper

    return decorator


# Pytest fixture helpers (add to conftest.py)


def create_memory_tracker_fixture():
    """Factory for pytest fixture that provides MemoryTracker."""
    import pytest

    @pytest.fixture
    def memory_tracker():
        """Pytest fixture for memory tracking."""
        return MemoryTracker

    return memory_tracker


def create_no_leaks_fixture():
    """Factory for pytest fixture that provides assert_no_object_leak."""
    import pytest

    @pytest.fixture
    def no_leaks():
        """Pytest fixture for asserting no memory leaks."""

        def _assert_no_leaks(operation, type_name, tolerance=5):
            assert_no_object_leak(operation, type_name, tolerance)

        return _assert_no_leaks

    return no_leaks
