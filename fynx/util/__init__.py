"""
FynX Observable Utils - Performance Optimization Classes
======================================================

This package contains performance optimization classes for the FynX reactive system.

Classes:
- SoAObserverSet: Struct-of-Arrays layout for cache-efficient data access
- CoWObserverSet: Copy-on-Write semantics for memory-efficient observer sets
- FunctionFlyweight: Flyweight pattern for function reuse
- AlgebraicOptimizer: Algebraic optimization for function composition
- AdaptiveObserverSet: Adaptive data structures that scale with usage patterns
- LazyChainBuilder: Efficient chain builder for function composition
- TurboChainBuilder: Pre-allocated chain builder for known sizes
"""

from .adaptive_observer_set import AdaptiveObserverSet
from .algebraic_optimizer import AlgebraicOptimizer
from .chain import (
    LazyChainBuilder,
    TurboChainBuilder,
    benchmark_all_modes,
    chain_batch,
    find_ultimate_source,
)
from .cow_observer_set import CoWObserverSet, SharedObserverArray
from .function_flyweight import FunctionFlyweight
from .soa_observer_set import SoAObserverSet

__all__ = [
    "SoAObserverSet",
    "CoWObserverSet",
    "SharedObserverArray",
    "FunctionFlyweight",
    "AlgebraicOptimizer",
    "AdaptiveObserverSet",
    "LazyChainBuilder",
    "TurboChainBuilder",
    "find_ultimate_source",
    "chain_batch",
    "benchmark_all_modes",
]
