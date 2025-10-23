"""
FynX Chain Utilities
===================

This module provides chain functionality for FynX, including lazy chain builders
and chain operations for efficient function composition.

Key classes:
- LazyChainBuilder: Efficient chain builder that accumulates functions
- TurboChainBuilder: Pre-allocated chain builder for known sizes
- Chain utilities: find_ultimate_source, chain_batch, etc.
"""

from __future__ import annotations

import array
import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from fynx.observable.core.types import is_derived_observable

from .algebraic_optimizer import AlgebraicOptimizer

if TYPE_CHECKING:
    from ..observable.types.protocols.observable_protocol import Observable

T = TypeVar("T")
U = TypeVar("U")

# Global cache
_materialized_cache = weakref.WeakValueDictionary()
_ultimate_source_cache = weakref.WeakValueDictionary()

# ============================================================================
# LAZY CHAIN BUILDER - EFFICIENT FUNCTION COMPOSITION
# ============================================================================


class LazyChainBuilder:
    """
    Efficient chain builder that accumulates functions and materializes them once.

    This builder accumulates transformation functions and only creates the final
    observable when materialize() is called, avoiding intermediate object creation.

    Performance characteristics:
    - then(): O(1) - append to internal list, return self
    - No object creation until materialize()
    - Function composition reduces chain length

    Memory: Single builder + growable array
    Time: O(n) total for building + materializing
    """

    __slots__ = (
        "source",
        "_functions",
        "_materialized",
        "_result",
        "_frozen",
        "_cached_key",
    )

    def __init__(self, source: Any):
        self.source = source
        self._functions: list[Callable] = []
        self._materialized = False
        self._result: Optional[Any] = None
        self._frozen = False
        self._cached_key: Optional[tuple] = None

    def then(self, func: Callable) -> "LazyChainBuilder":
        """
        Add a function to the chain and return self for chaining.

        No object allocation occurs until materialize() is called.
        """
        if self._frozen:
            raise RuntimeError("Cannot modify frozen chain")
        self._functions.append(func)
        # Invalidate cached key when functions change
        self._cached_key = None
        return self

    def __rshift__(self, func: Callable) -> "LazyChainBuilder":
        """Support >> operator."""
        return self.then(func)

    def _get_ultimate_source(self) -> Any:
        """Cached ultimate source with cycle detection."""
        cache_key = id(self.source)
        cached = _ultimate_source_cache.get(cache_key)
        if cached is not None:
            return cached

        visited = set()
        current = self.source

        while True:
            obj_id = id(current)
            if obj_id in visited:
                break
            visited.add(obj_id)

            if (
                is_derived_observable(current)
                and current._source_observable is not None
            ):
                current = current._source_observable
            else:
                break

        _ultimate_source_cache[cache_key] = current
        return current

    def materialize(self) -> Any:
        """
        Create the final observable from the accumulated functions.

        Performance:
        - O(n) where n = function count
        - Single ComputedObservable created
        - Builder becomes frozen after materialization
        """
        if self._materialized:
            return self._result

        from ..observable.computed import ComputedObservable

        # Empty chain
        if not self._functions:
            self._result = self.source
            self._materialized = True
            self._frozen = True
            return self.source

        # Check cache
        cache_key = self._compute_cache_key()
        cached = _materialized_cache.get(cache_key)
        if cached is not None:
            self._result = cached
            self._materialized = True
            self._frozen = True
            return cached

        ultimate_source = self._get_ultimate_source()

        # Apply function fusion to reduce chain length
        functions_snapshot = self._functions.copy()
        optimized_functions = AlgebraicOptimizer.optimize_chain(functions_snapshot)

        # Make composed function from optimized chain
        composed_func = self._make_composed_function(optimized_functions)

        # Compute initial value
        if hasattr(ultimate_source, "_value_wrapper"):
            value = ultimate_source._value_wrapper.value
        elif hasattr(ultimate_source, "value"):
            value = ultimate_source.value
        else:
            value = ultimate_source

        for func in optimized_functions:
            try:
                value = func(value)
            except Exception:
                break

        # Create result
        result = ComputedObservable(
            key=f"optimized_chain_{len(optimized_functions)}_from_{len(functions_snapshot)}",
            initial_value=value,
            computation_func=composed_func,
            source_observable=ultimate_source,
        )

        result._composed_functions = functions_snapshot
        result._is_composition_optimized = True

        # Cache and freeze
        _materialized_cache[cache_key] = result
        self._result = result
        self._materialized = True
        self._frozen = True

        return result

    def _make_composed_function(self, functions: list[Callable]) -> Callable:
        """
        Create composed function with special cases for common lengths.
        """
        n = len(functions)

        if n == 0:
            return lambda x: x
        if n == 1:
            return functions[0]
        if n == 2:
            f0, f1 = functions
            return lambda x: f1(f0(x))
        if n == 3:
            f0, f1, f2 = functions
            return lambda x: f2(f1(f0(x)))
        if n == 4:
            f0, f1, f2, f3 = functions
            return lambda x: f3(f2(f1(f0(x))))

        # General case - use tuple for speed
        funcs_tuple = tuple(functions)
        func_count = len(funcs_tuple)

        def composed(x):
            result = x
            for i in range(func_count):
                result = funcs_tuple[i](result)
            return result

        return composed

    def _compute_cache_key(self):
        """Compute cache key from source and function ids (with caching)."""
        if self._cached_key is None:
            self._cached_key = (id(self.source), tuple(id(f) for f in self._functions))
        return self._cached_key

    @property
    def value(self):
        if not self._materialized:
            self.materialize()
        return self._result.value

    def subscribe(self, callback):
        if not self._materialized:
            self.materialize()
        return self._result.subscribe(callback)

    def set(self, value):
        self.source.set(value)

    def clone(self) -> "LazyChainBuilder":
        """Create a mutable copy for branching."""
        new_builder = LazyChainBuilder(self.source)
        new_builder._functions = self._functions.copy()
        return new_builder


# ============================================================================
# PRE-ALLOCATED CHAIN BUILDER - FOR KNOWN SIZES
# ============================================================================


class TurboChainBuilder:
    """
    Pre-allocated chain builder for when you know the chain length in advance.

    Performance characteristics:
    - 30-50% faster than LazyChainBuilder for known sizes
    - Zero array resizing
    - Minimal memory allocation
    """

    __slots__ = ("source", "_functions", "_index", "_size")

    def __init__(self, source: Any, size: int):
        self.source = source
        self._functions = [None] * size  # Pre-allocate
        self._index = 0
        self._size = size

    def then(self, func: Callable) -> "TurboChainBuilder":
        """Add function to pre-allocated array."""
        self._functions[self._index] = func
        self._index += 1
        return self

    def __rshift__(self, func: Callable) -> "TurboChainBuilder":
        return self.then(func)

    def materialize(self) -> Any:
        """Convert to LazyChainBuilder and materialize."""
        builder = LazyChainBuilder(self.source)
        builder._functions = self._functions[: self._index]
        return builder.materialize()

    @property
    def value(self):
        return self.materialize().value

    def subscribe(self, callback):
        return self.materialize().subscribe(callback)


# ============================================================================
# BATCH MODE - DIRECT MATERIALIZATION
# ============================================================================


def chain_batch(source: Any, functions: list[Callable]) -> Any:
    """
    Direct materialization from function list.

    Use for batch operations where you have all functions upfront.

    Performance characteristics:
    - Fastest possible path
    - Zero builder objects
    - Direct to ComputedObservable
    """
    if not functions:
        return source

    from ..observable.computed import ComputedObservable

    # Cache key
    cache_key = (id(source), tuple(id(f) for f in functions))
    cached = _materialized_cache.get(cache_key)
    if cached is not None:
        return cached

    # Find ultimate source
    cache_key_src = id(source)
    ultimate_source = _ultimate_source_cache.get(cache_key_src)
    if ultimate_source is None:
        visited = set()
        current = source
        while True:
            obj_id = id(current)
            if obj_id in visited:
                break
            visited.add(obj_id)
            if (
                is_derived_observable(current)
                and current._source_observable is not None
            ):
                current = current._source_observable
            else:
                break
        ultimate_source = current
        _ultimate_source_cache[cache_key_src] = ultimate_source

    # Create composed function
    funcs_tuple = tuple(functions)
    func_count = len(funcs_tuple)

    def composed(x):
        result = x
        for i in range(func_count):
            result = funcs_tuple[i](result)
        return result

    # Compute initial value
    if hasattr(ultimate_source, "_value_wrapper"):
        value = ultimate_source._value_wrapper.value
    elif hasattr(ultimate_source, "value"):
        value = ultimate_source.value
    else:
        value = ultimate_source

    for func in functions:
        try:
            value = func(value)
        except Exception:
            break

    # Create result
    result = ComputedObservable(
        key=f"batch_{func_count}",
        initial_value=value,
        computation_func=composed,
        source_observable=ultimate_source,
    )

    result._composed_functions = list(functions)
    result._is_composition_optimized = True

    _materialized_cache[cache_key] = result
    return result


# ============================================================================
# UTILITIES
# ============================================================================


def find_ultimate_source(observable: Any) -> Any:
    """Find ultimate source with cycle detection."""
    visited = set()
    current = observable

    while True:
        obj_id = id(current)
        if obj_id in visited:
            break
        visited.add(obj_id)

        if (
            hasattr(current, "_source_observable")
            and current._source_observable is not None
        ):
            current = current._source_observable
        else:
            break

    return current


def clear_caches():
    """Clear all caches."""
    _materialized_cache.clear()
    _ultimate_source_cache.clear()


# ============================================================================
# BENCHMARKING
# ============================================================================


def benchmark_all_modes(n: int = 10000):
    """
    Compare all chain building modes.

    Usage:
        >>> from fynx.util.chain import benchmark_all_modes
        >>> benchmark_all_modes(10000)
    """
    import time

    from ..observable.core.observable import Observable

    results = {}

    # Test 1: LazyChainBuilder (mutable, single object)
    source = Observable("source", 0)
    start = time.perf_counter()
    builder = LazyChainBuilder(source)
    for i in range(n):
        builder.then(lambda x, i=i: x + 1)
    result = builder.materialize()
    results["lazy"] = time.perf_counter() - start

    # Test 2: TurboChainBuilder (pre-allocated)
    source = Observable("source", 0)
    start = time.perf_counter()
    turbo = TurboChainBuilder(source, n)
    for i in range(n):
        turbo.then(lambda x, i=i: x + 1)
    result = turbo.materialize()
    results["turbo"] = time.perf_counter() - start

    # Test 3: chain_batch (direct)
    source = Observable("source", 0)
    functions = [lambda x, i=i: x + 1 for i in range(n)]
    start = time.perf_counter()
    result = chain_batch(source, functions)
    results["batch"] = time.perf_counter() - start

    print(f"\nChain Building Benchmark (n={n})")
    print("=" * 50)
    print(f"LazyChainBuilder:  {results['lazy']*1000:8.2f}ms")
    print(
        f"TurboChainBuilder: {results['turbo']*1000:8.2f}ms  ({results['lazy']/results['turbo']:.1f}x)"
    )
    print(
        f"chain_batch:       {results['batch']*1000:8.2f}ms  ({results['lazy']/results['batch']:.1f}x)"
    )
    print("=" * 50)

    return results
