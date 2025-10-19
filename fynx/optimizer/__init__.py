"""
FynX Reactive Graph Optimizer
============================

This module implements categorical optimization for FynX reactive observable networks,
based on category theory principles. It performs global analysis and transformation
of dependency graphs to minimize computational cost while preserving semantic equivalence.
"""

from .optimizer import (
    DependencyNode,
    Morphism,
    MorphismParser,
    OptimizationContext,
    ReactiveGraph,
    get_graph_statistics,
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
