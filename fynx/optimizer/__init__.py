"""
FynX Reactive Graph Optimizer
============================

This module implements categorical optimization for FynX reactive observable networks,
based on category theory principles. It performs global analysis and transformation
of dependency graphs to minimize computational cost while preserving semantic equivalence.
"""

from .dependency_graph import get_graph_statistics
from .morphism import Morphism, MorphismParser
from .optimizer import (
    DependencyNode,
    OptimizationContext,
    ReactiveGraph,
    optimize_reactive_graph,
)

__all__ = [
    "Morphism",
    "MorphismParser",
    "DependencyNode",
    "OptimizationContext",
    "ReactiveGraph",
    "optimize_reactive_graph",
    "get_graph_statistics",
]
