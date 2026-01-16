"""
FynX Reactive Graph Optimizer
============================

This module implements categorical optimization for FynX reactive observable networks,
based on category theory principles. It performs global analysis and transformation
of dependency graphs to minimize computational cost while preserving semantic equivalence.

Core Concepts
-------------

**Temporal Category C_T**: Reactive system as a category where:
- Objects: Time-varying values O(A): T → A (reactive observables)
- Morphisms: Structure-preserving transformations between observables
- Composition: Preserves temporal coherence

**Cost Functional C**: Computational expense measure
C(σ) = α·|Dep(σ)| + β·E[Updates(σ)] + γ·depth(σ)

**Optimization Goal**: Find σ' ≅ σ such that C(σ') = min_{τ ≅ σ} C(τ)

Rewrite Rules
-------------

1. **Functor Composition Collapse**: O(g) ∘ O(f) = O(g ∘ f)
   - Fuses sequential transformations into single composed function

2. **Product Factorization**: O(A) × O(B) ≅ O(A × B)
   - Shares common subexpressions across multiple dependents

3. **Pullback Fusion**: Sequential filters combine via conjunction
   - f₁ ∧ f₂ filters fuse into single conjunctive predicate

4. **Materialization Optimization**: Dynamic programming for cost-optimal caching
   - Decides which intermediate results to materialize vs recompute

Implementation
--------------

The optimizer works in phases:

1. **Graph Construction**: Build dependency DAG from observable relationships
2. **Semantic Equivalence**: Identify nodes with identical semantics (DAG quotient)
3. **Rewrite Application**: Apply categorical rewrite rules exhaustively
4. **Cost Optimization**: Use dynamic programming for materialization strategy
5. **Graph Transformation**: Update observable network with optimized structure

Usage
-----

The optimizer runs automatically as part of observable creation and can be
triggered manually for complex reactive graphs:

```python
from fynx.optimizer import optimize_reactive_graph

# Automatic optimization (built into >> operator)
chain = obs >> f >> g >> h  # Automatically fused to obs >> (h ∘ g ∘ f)

# Manual optimization for complex graphs
optimized_roots = optimize_reactive_graph(root_observables)
```

See Also
--------

- `fynx.observable`: Core observable classes
- `fynx.computed`: Computed observable creation
- `fynx.store`: Reactive state containers
"""

import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ..observable.base import Observable
from ..observable.computed import ComputedObservable
from ..observable.conditional import ConditionalObservable
from ..observable.merged import MergedObservable
from .dependency_graph import get_graph_statistics
from .dependency_node import DependencyNode
from .reactive_graph import ReactiveGraph

T = TypeVar("T")


class OptimizationContext:
    """
    Context manager for reactive graph optimization.

    Encapsulates an optimizer instance and provides automatic optimization
    without global state. Uses thread-local storage to maintain
    context per thread.
    """

    _thread_local = threading.local()

    def __init__(self, auto_optimize: bool = True):
        self.optimizer = ReactiveGraph()
        self.auto_optimize = auto_optimize
        self._previous_context = None

    def __enter__(self):
        """Enter the optimization context."""
        self._previous_context = getattr(
            OptimizationContext._thread_local, "current", None
        )
        OptimizationContext._thread_local.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the optimization context."""
        OptimizationContext._thread_local.current = self._previous_context

    @classmethod
    def current(cls) -> Optional["OptimizationContext"]:
        """Get the current optimization context for this thread."""
        return getattr(cls._thread_local, "current", None)

    @classmethod
    def get_optimizer(cls) -> "ReactiveGraph":
        """Get the current optimizer instance, creating one if needed."""
        context = cls.current()
        if context is None:
            # Create a temporary context if none exists
            context = cls()
            context.__enter__()
        return context.optimizer

    def register_observable(self, observable: "Observable") -> None:
        """Register an observable with this optimization context."""
        self.optimizer.get_or_create_node(observable)

        # Trigger incremental optimization if enabled
        if self.auto_optimize and isinstance(observable, ComputedObservable):
            fusions = self.optimizer.apply_functor_composition_fusion()
            if fusions > 0 and len(self.optimizer.nodes) > 20:
                # Run full optimization for large graphs
                results, _ = optimize_reactive_graph(list(self.optimizer.nodes.keys()))
                # Update our optimizer with the optimized graph
                self.optimizer = ReactiveGraph()
                for obs in self.optimizer.nodes.keys():
                    self.optimizer.get_or_create_node(obs)
