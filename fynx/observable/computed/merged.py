"""
Simplified MergedObservable - Elegant Tuple Composition
=======================================================

Key insight: Merging is just tuple construction with reactive updates.
No need for complex indexing or special handlers.

Key improvements:
1. Treats tuple as immutable value (functional style)
2. No pre-allocated arrays or index tracking
3. Clean subscription model
4. Flattening is explicit and simple
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from fynx.observable.core.abstract.derived import DerivedValue
from fynx.observable.core.abstract.operations import OperatorMixin, TupleMixin
from fynx.observable.types.protocols.merged_protocol import Mergeable

if TYPE_CHECKING:
    from fynx.observable.types.protocols.observable_protocol import Observable

T = TypeVar("T")

# Global registry for function-to-context mappings (for cleanup testing)
_func_to_contexts = {}


class MergedObservable(DerivedValue[tuple], Mergeable[T], OperatorMixin, TupleMixin):
    """
    Elegantly simple merged observable using tuple semantics.

    Before refactoring: 150+ lines with complex update handlers
    After refactoring: <50 lines with clear semantics

    Key improvements:
    1. Treats tuple as immutable value (functional style)
    2. No pre-allocated arrays or index tracking
    3. Clean subscription model
    4. Flattening is explicit and simple
    """

    def __init__(self, *observables: "Observable"):
        if not observables:
            raise ValueError("At least one observable required")

        # Flatten nested MergedObservables for associativity
        self._sources = self._flatten_sources(observables)

        # Initialize with tuple of current values
        initial_tuple = tuple(obs.value for obs in self._sources)

        # Use first source as primary, but track all
        super().__init__(
            key="merged",
            initial_value=initial_tuple,
            source_observable=self._sources[0],
            source_observables=self._sources,
        )

    def _flatten_sources(self, observables: tuple) -> list:
        """Flatten nested MergedObservables: (a+b)+c → a+b+c"""
        flattened = []
        for obs in observables:
            if isinstance(obs, MergedObservable):
                flattened.extend(obs._sources)
            else:
                flattened.append(obs)
        return flattened

    def _compute_value(self) -> tuple:
        """
        Recompute tuple from all sources.

        Simple and functional - just rebuild the tuple.
        """
        return tuple(obs.value for obs in self._sources)

    def _setup_source_observers(self) -> None:
        """
        Subscribe to all sources uniformly.

        No special indexing or handlers needed!
        """
        for source in self._sources:
            source.subscribe(self._on_source_change)

    # ============================================================
    # Operator Overloading - Clean API
    # ============================================================

    def __add__(self, other: "Observable") -> "MergedObservable":
        """Chain merging: (a + b) + c → MergedObservable(a, b, c)"""
        return MergedObservable(*self._sources, other)

    def __iter__(self):
        """Support tuple unpacking: a, b, c = merged"""
        return iter(self.value)

    def __getitem__(self, index: int) -> Any:
        """Support indexing: merged[0]"""
        return self.value[index]

    def __setitem__(self, index: int, value: Any) -> None:
        """Support assignment: merged[0] = new_value"""
        if 0 <= index < len(self._sources):
            self._sources[index].set(value)
        else:
            raise IndexError(f"Index {index} out of range")

    def __len__(self) -> int:
        """Support len(): len(merged)"""
        return len(self._sources)

    # ============================================================
    # Callable Pattern - Reactive Callbacks
    # ============================================================

    def __call__(self, func: Callable) -> None:
        """
        Make merged observable callable for reactive patterns.

        Example:
            merged = x + y + z
            merged(lambda x, y, z: print(f"({x}, {y}, {z})"))
        """
        # Call immediately with current values
        func(*self.value)

        # Subscribe for reactive updates
        self.subscribe(lambda values: func(*values))

    # ============================================================
    # Context Manager - Convenient Unpacking
    # ============================================================

    def __enter__(self) -> "MergedObservable":
        """Support context manager for unpacking."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager."""
        return False
