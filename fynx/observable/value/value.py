"""
FynX ObservableValue - Auto-Lifting Value Container
==================================================

This module provides an ObservableValue class that automatically unwraps nested ObservableValue instances
when accessed, making the API more ergonomic by ensuring users always get raw values rather
than ObservableValue wrappers.

The ObservableValue implements a caching mechanism that stores the unwrapped result until
the value is set again, providing efficient access to unwrapped values.

Key Features:
- Automatic unwrapping of nested ObservableValue instances
- Caching of unwrapped results for performance
- Property-based getter/setter interface
- Iterative unwrapping to avoid stack overflow
"""

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

if TYPE_CHECKING:
    from ..chain import LazyChainBuilder
    from ..types.observable_protocols import Observable

# Import common types
from ..types.common_types import T

# Import the TransparentValue protocol
from .protocol import TransparentValue

# Type variables for Value
V = TypeVar("V")
K = TypeVar("K")
U = TypeVar("U")

# ============================================================================
# VALUE UNWRAPPING UTILITIES
# ============================================================================


def _unwrap_observable_value(value: Any) -> Any:
    """
    Iteratively unwrap nested ObservableValue instances using a stack-based approach.
    This function handles the automatic unwrapping of nested ObservableValue instances,
    making the API more ergonomic by ensuring users always get the raw values
    rather than ObservableValue wrappers. Uses iteration instead of recursion to
    avoid stack overflow issues with deeply nested structures.

    Examples:
        - ObservableValue(ObservableValue(1)) → 1
        - (ObservableValue(1), ObservableValue(2)) → (1, 2)
        - ObservableValue((ObservableValue(1), ObservableValue(2))) → (1, 2)
        - ObservableValue([ObservableValue([ObservableValue(1)])]) → [[[1]]]
        - Regular values pass through unchanged

    Args:
        value: The value to unwrap (can be ObservableValue, tuple, list, dict, or regular value).

    Returns:
        The unwrapped value with all nested ObservableValue instances resolved.
    """
    # Use a work queue with explicit state tracking
    # Each item is (value, path) where path identifies where in the structure we are
    work_queue: List[Tuple[Any, Tuple[Any, ...]]] = [(value, ())]
    processed: Dict[Tuple[Any, ...], Any] = {}  # path -> unwrapped_value

    while work_queue:
        current, path = work_queue.pop(0)

        # Unwrap ObservableValue wrappers with cycle detection
        seen_values = set()
        while isinstance(current, ObservableValue):
            if id(current) in seen_values:
                # Circular reference detected, return the ObservableValue as-is
                break
            seen_values.add(id(current))
            current = current.value

        # Handle tuples
        if isinstance(current, tuple):
            # Add all children to work queue
            for i, item in enumerate(current):
                work_queue.append((item, path + ("tuple", i)))
            # Mark this as a tuple container (will be built later)
            processed[path] = ("TUPLE", len(current))

        # Handle lists
        elif isinstance(current, list):
            # Add all children to work queue
            for i, item in enumerate(current):
                work_queue.append((item, path + ("list", i)))
            # Mark this as a list container
            processed[path] = ("LIST", len(current))

        # Handle dictionaries
        elif isinstance(current, dict):
            # Add all keys and values to work queue
            for k, v in current.items():
                work_queue.append((k, path + ("dict_key", k)))
                work_queue.append((v, path + ("dict_value", k)))
            # Mark this as a dict container
            processed[path] = ("DICT", list(current.keys()))

        # Base case: regular value
        else:
            processed[path] = current

    # Rebuild the structure from the bottom up iteratively
    rebuild_stack: List[Tuple[Any, ...]] = [()]
    result_cache: Dict[Tuple[Any, ...], Any] = {}

    while rebuild_stack:
        path = rebuild_stack.pop()

        if path in result_cache:
            continue

        value = processed[path]

        if isinstance(value, tuple) and len(value) == 2:
            container_type, metadata = value

            if container_type == "TUPLE":
                # Check if all children are processed
                child_paths = [path + ("tuple", i) for i in range(metadata)]
                if all(child_path in result_cache for child_path in child_paths):
                    items = [result_cache[child_path] for child_path in child_paths]
                    result_cache[path] = tuple(items)
                else:
                    # Push this path back and add unprocessed children
                    rebuild_stack.append(path)
                    for child_path in child_paths:
                        if child_path not in result_cache:
                            rebuild_stack.append(child_path)

            elif container_type == "LIST":
                # Check if all children are processed
                child_paths = [path + ("list", i) for i in range(metadata)]
                if all(child_path in result_cache for child_path in child_paths):
                    items = [result_cache[child_path] for child_path in child_paths]
                    result_cache[path] = items
                else:
                    # Push this path back and add unprocessed children
                    rebuild_stack.append(path)
                    for child_path in child_paths:
                        if child_path not in result_cache:
                            rebuild_stack.append(child_path)

            elif container_type == "DICT":
                # Check if all children are processed
                child_key_paths = [path + ("dict_key", k) for k in metadata]
                child_value_paths = [path + ("dict_value", k) for k in metadata]
                all_child_paths = child_key_paths + child_value_paths

                if all(child_path in result_cache for child_path in all_child_paths):
                    result_dict = {}
                    for k in metadata:
                        unwrapped_key = result_cache[path + ("dict_key", k)]
                        unwrapped_value = result_cache[path + ("dict_value", k)]
                        result_dict[unwrapped_key] = unwrapped_value
                    result_cache[path] = result_dict
                else:
                    # Push this path back and add unprocessed children
                    rebuild_stack.append(path)
                    for child_path in all_child_paths:
                        if child_path not in result_cache:
                            rebuild_stack.append(child_path)
        else:
            # Base case: regular value
            result_cache[path] = value

    return result_cache[()]


# ============================================================================
# OBSERVABLE VALUE WRAPPER CLASS
# ============================================================================


class ObservableValue(Generic[V], TransparentValue[V]):
    """
    A pure value wrapper with no knowledge of observables.

    ObservableValue is a simple value container that stores a value and provides
    transparent behavior through magic methods. It has no reactive capabilities
    and no knowledge of observables - it's just a value wrapper.

    The Observable class uses ObservableValue internally and subscribes to its
    changes via the on_change callback to implement reactive behavior.

    Key Features:
    - Pure value storage with no reactive capabilities
    - Change notification via on_change callback
    - Transparent behavior through magic methods
    - No knowledge of observables or reactive systems

    Example:
        ```python
        def on_change(old_val, new_val):
            print(f"Value changed from {old_val} to {new_val}")

        wrapper = ObservableValue(42, on_change=on_change)

        # Access value
        print(wrapper.value)  # 42

        # Transparent behavior
        if wrapper == 42:  # Uses __eq__
            print("Match!")

        wrapper.value = 100  # Calls on_change(42, 100)
        ```
    """

    @overload
    def __init__(
        self: "ObservableValue[T]",
        initial_value: "ObservableValue[T]",
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[T]",
        initial_value: T,
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[T]",
        initial_value: "Observable[T]",
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[List[T]]",
        initial_value: List["Observable[T]"],
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[Tuple[T, ...]]",
        initial_value: Tuple["Observable[T]", ...],
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[Dict[K, T]]",
        initial_value: Dict[K, "Observable[T]"],
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[Dict[T, U]]",
        initial_value: Dict["Observable[T]", "Observable[U]"],
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "ObservableValue[Dict[T, U]]",
        initial_value: Dict["Observable[T]", U],
        *,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ) -> None: ...

    def __init__(
        self,
        initial_value: Any = None,
        on_change: Optional[Callable[[Any, Any], None]] = None,
    ):
        """
        Initialize the ObservableValue with an initial value and change callback.

        Args:
            initial_value: The initial value to store.
            on_change: Optional callback function called when value changes.
                      Receives (old_value, new_value) as arguments.
        """
        self._value = initial_value
        self._on_change: Optional[Callable[[Any, Any], None]] = on_change

    @property
    def value(self) -> V:
        """
        Get the stored value.

        Returns:
            The stored value.
        """
        return cast(V, self._value)

    @value.setter
    def value(self, new_value: V) -> None:
        """
        Set a new value and trigger change notification.

        When a new value is set, the cache is invalidated and the on_change
        callback (if provided) is called with the old and new values.

        Args:
            new_value: The new value to set (can be Observable or regular value).

        Example:
            ```python
            def on_change(old_val, new_val):
                print(f"Changed from {old_val} to {new_val}")

            wrapper = Value(42, on_change=on_change)
            wrapper.value = 100  # Calls on_change(42, 100)
            ```
        """
        old_value = self._value
        self._value = new_value

        # Call the change callback if provided and value actually changed
        if self._on_change is not None and old_value != new_value:
            self._on_change(old_value, new_value)

    def unwrap(self) -> Any:
        """
        Recursively unwrap nested Observable values.

        This method performs the same recursive unwrapping that was previously
        done automatically by the .value property. It unwraps all nested
        Observable values to return the underlying raw value.

        Returns:
            The unwrapped value with all nested Observables resolved.

        Example:
            ```python
            wrapper = Value(observable(observable(42)))
            raw = wrapper.value  # Observable(Observable(42))
            unwrapped = wrapper.unwrap()  # 42

            # Works with complex nested structures
            nested = Value(observable([observable(1), observable(2)]))
            result = nested.unwrap()  # [1, 2]
            ```
        """
        return _unwrap_observable_value(self._value)

    # ============================================================================
    # MAGIC METHODS - Transparent behavior
    # ============================================================================

    def __eq__(self, other: object) -> bool:
        """Equality comparison with auto-unwrapping."""
        return self.unwrap() == _unwrap_observable_value(other)

    def __str__(self) -> str:
        """String representation of unwrapped value."""
        unwrapped = self.unwrap()
        return str(unwrapped)

    def __repr__(self) -> str:
        """Developer representation."""
        raw = self._value
        unwrapped = self.unwrap()
        return f"ObservableValue(raw={raw!r}, unwrapped={unwrapped!r})"

    def __bool__(self) -> bool:
        """Boolean conversion of unwrapped value."""
        unwrapped = self.unwrap()
        return bool(unwrapped)

    def __len__(self) -> int:
        """Length of unwrapped value if applicable."""
        unwrapped = self.unwrap()
        return len(unwrapped)

    def __iter__(self):
        """Iteration over unwrapped value if applicable."""
        unwrapped = self.unwrap()
        return iter(unwrapped)

    def __getitem__(self, key):
        """Indexing into unwrapped value if applicable."""
        unwrapped = self.unwrap()
        return unwrapped[key]

    def __contains__(self, item) -> bool:
        """Check if unwrapped value contains item."""
        unwrapped = self.unwrap()
        return item in unwrapped

    def __hash__(self) -> int:
        """Hash of unwrapped value."""
        unwrapped = self.unwrap()
        return hash(unwrapped)

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self < unwrapped_other

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self <= unwrapped_other

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self > unwrapped_other

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self >= unwrapped_other

    def __add__(self, other):
        """Addition operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self + unwrapped_other

    def __sub__(self, other):
        """Subtraction operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self - unwrapped_other

    def __mul__(self, other):
        """Multiplication operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self * unwrapped_other

    def __truediv__(self, other):
        """True division operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self / unwrapped_other

    def __floordiv__(self, other):
        """Floor division operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self // unwrapped_other

    def __mod__(self, other):
        """Modulo operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self % unwrapped_other

    def __pow__(self, other):
        """Power operation."""
        unwrapped_self = self.unwrap()
        unwrapped_other = _unwrap_observable_value(other)
        return unwrapped_self**unwrapped_other

    def __neg__(self):
        """Negation operation."""
        unwrapped = self.unwrap()
        return -unwrapped

    def __pos__(self):
        """Positive operation."""
        unwrapped = self.unwrap()
        return +unwrapped

    def __abs__(self):
        """Absolute value operation."""
        unwrapped = self.unwrap()
        return abs(unwrapped)
