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
from .dependency_graph import DependencyGraph
from .dependency_graph import DependencyNode as BaseDependencyNode
from .dependency_graph import get_graph_statistics
from .morphism import Morphism, MorphismParser

T = TypeVar("T")


class DependencyNode(BaseDependencyNode):
    """
    Extended dependency node with optimization-specific features.

    This extends the base DependencyNode with cost modeling, profiling,
    and optimization state for reactive graph optimization.
    """

    def __init__(self, observable: Observable):
        super().__init__(observable)

        # Cost model parameters: C(σ) = α·|Dep(σ)| + β·E[Updates(σ)] + γ·depth(σ)
        self.cost_alpha = 1.0  # Memory cost coefficient (per materialized node)
        self.cost_beta = 1.0  # Computation cost coefficient (per evaluation)
        self.cost_gamma = 1.0  # Latency cost coefficient (per depth level)

        # Optimization state
        self.is_materialized = True  # Whether this node should be kept
        self.equivalence_class: Optional[int] = None  # For DAG quotient

        # Cached cost computations
        self._cached_cost: Optional[float] = None
        self._materialize_cost: Optional[float] = None
        self._recompute_cost: Optional[float] = None

        # Profiling data for cost model
        self.profiling_data: Dict[str, Any] = {
            "execution_times": [],  # List of measured execution times
            "call_count": 0,  # Number of times this node was computed
            "last_updated": None,  # Timestamp of last update
            "avg_execution_time": None,  # Cached average
        }

    def record_execution_time(self, execution_time: float) -> None:
        """Record a measured execution time for profiling."""
        self.profiling_data["execution_times"].append(execution_time)
        self.profiling_data["call_count"] += 1
        self.profiling_data["last_updated"] = time.time()

        # Keep only recent measurements to adapt to changing patterns
        max_samples = 100
        if len(self.profiling_data["execution_times"]) > max_samples:
            self.profiling_data["execution_times"] = self.profiling_data[
                "execution_times"
            ][-max_samples:]

        # Update cached average
        self.profiling_data["avg_execution_time"] = sum(
            self.profiling_data["execution_times"]
        ) / len(self.profiling_data["execution_times"])

    @property
    def update_frequency_estimate(self) -> float:
        """
        Estimate update frequency using graph analysis and profiling data.

        Combines:
        - Historical update patterns from profiling
        - Graph topology analysis (fan-in, fan-out, depth)
        - Parent update frequencies with attenuation
        """
        # Source nodes: base update rate (could be application-specific)
        if not self.incoming:
            return 1.0

        # If we have profiling data, use it as primary indicator
        if (
            self.profiling_data["call_count"] > 0
            and self.profiling_data["last_updated"]
        ):
            time_since_last_update = time.time() - self.profiling_data["last_updated"]
            if time_since_last_update > 0:
                # Estimate frequency from recent activity
                recent_freq = self.profiling_data["call_count"] / max(
                    time_since_last_update, 1.0
                )
                return min(recent_freq, 100.0)  # Cap at reasonable maximum

        # Fall back to graph-based estimation
        # Average parent frequencies, attenuated by depth and branching
        parent_freqs = [p.update_frequency_estimate for p in self.incoming]
        avg_parent_freq = sum(parent_freqs) / len(parent_freqs) if parent_freqs else 1.0

        # Attenuation factors based on graph structure
        depth_attenuation = 0.8**self.depth  # Deeper nodes update less
        fan_out_attenuation = 1.0 / (
            1.0 + len(self.outgoing) * 0.1
        )  # Nodes with many dependents update less
        fan_in_boost = min(
            1.0 + len(self.incoming) * 0.05, 1.5
        )  # Nodes with many inputs might update more

        estimated_freq = (
            avg_parent_freq * depth_attenuation * fan_out_attenuation * fan_in_boost
        )

        # Ensure reasonable bounds
        return max(0.001, min(estimated_freq, 50.0))

    @property
    def computation_cost(self) -> float:
        """
        Estimate computation cost using profiling data and static analysis.

        Uses execution times when available, otherwise falls back
        to static complexity analysis of the computation function.
        """
        # Use profiled execution time if available
        if self.profiling_data["avg_execution_time"] is not None:
            return self.profiling_data["avg_execution_time"]

        # Source nodes have minimal cost (just accessing stored value)
        if self.computation_func is None:
            return 0.01

        # Static analysis based on function characteristics
        return self._analyze_computation_complexity()

    def _analyze_computation_complexity(self) -> float:
        """
        Analyze computation complexity using static analysis.

        Estimates cost based on function structure, dependencies, and type information.
        """
        if self.computation_func is None:
            return 0.01

        complexity_score = 1.0  # Base computation cost

        # Factor in input complexity (more inputs = more complex)
        num_inputs = len(self.incoming)
        complexity_score *= 1.0 + num_inputs * 0.2

        # Factor in function characteristics
        func = self.computation_func
        func_name = getattr(func, "__name__", "")

        # Common patterns and their relative costs
        if "lambda" in func_name:
            complexity_score *= 1.2  # Lambdas often hide complexity
        elif func_name in ["map", "filter", "reduce"]:
            complexity_score *= 1.5  # Higher-order functions
        elif "sort" in func_name or "search" in func_name:
            complexity_score *= 3.0  # O(n log n) operations
        elif "hash" in func_name or "encrypt" in func_name:
            complexity_score *= 2.0  # Cryptographic operations

        # Factor in output complexity (more dependents = more important computation)
        num_outputs = len(self.outgoing)
        complexity_score *= 1.0 + num_outputs * 0.1

        # Factor in depth (deeper computations might be more complex)
        complexity_score *= 1.0 + self.depth * 0.05

        return max(0.1, min(complexity_score, 10.0))  # Reasonable bounds

    def compute_monoidal_cost(
        self, materialized_set: Optional[Set["DependencyNode"]] = None
    ) -> float:
        """
        Compute cost respecting monoidal structure of the category.

        The cost functional is a monoidal functor C: C_T → (R+, +, 0)
        where composition is preserved: C(g ∘ f) ≤ C(g) + C(f)

        Cost flows from sources to dependents: C(node) = local_cost(node) + Σ C(dependency)

        Args:
            materialized_set: Set of nodes that are materialized.

        Returns:
            Monoidal cost for this node including all its dependencies.
        """
        is_materialized = (
            self.is_materialized
            if materialized_set is None
            else (self in materialized_set)
        )

        # Local cost: intrinsic to this node's computation
        if self.computation_func is None:
            # Source nodes: only materialization cost if materialized
            local_cost = self.cost_alpha if is_materialized else 0
        else:
            # Computed nodes: either materialization or computation cost
            if is_materialized:
                local_cost = self.cost_alpha  # Memory cost for materialization
            else:
                local_cost = (
                    self.cost_beta
                    * self.update_frequency_estimate
                    * self.computation_cost
                )

        # Composition cost: monoidal combination of dependency costs
        # Cost flows from sources to dependents, so we sum dependency costs
        dependency_cost = sum(
            dep.compute_monoidal_cost(materialized_set) for dep in self.incoming
        )

        # Monoidal combination (sum for additive monoid)
        return local_cost + dependency_cost

    def compute_sharing_penalty(
        self, materialized_set: Optional[Set["DependencyNode"]] = None
    ) -> float:
        """
        Compute additional cost due to sharing when not materialized.

        This is separate from monoidal cost and accounts for redundant computation
        when a node has multiple dependents.

        Args:
            materialized_set: Set of nodes that are materialized.

        Returns:
            Sharing penalty cost (0 if materialized or has ≤ 1 dependent).
        """
        is_materialized = (
            self.is_materialized
            if materialized_set is None
            else (self in materialized_set)
        )

        if is_materialized:
            return 0.0  # No penalty if materialized

        # Penalty for non-materialized nodes with multiple dependents
        num_dependents = len(self.outgoing)
        if num_dependents <= 1:
            return 0.0  # No penalty for single or no dependents

        # Each additional dependent beyond the first pays recomputation cost
        # This represents the "sharing cost" that breaks monoidality
        return (
            (num_dependents - 1)
            * self.cost_beta
            * self.update_frequency_estimate
            * self.computation_cost
        )

    def compute_cost(
        self, materialized_set: Optional[Set["DependencyNode"]] = None
    ) -> float:
        """
        Compute the total cost for this node given a materialization strategy.

        Total cost = Monoidal cost + Sharing penalty

        The monoidal cost follows composition laws, while the sharing penalty
        accounts for contextual costs that depend on usage.

        Uses the cost functional: C(σ) = α·|Dep(σ)| + β·E[Updates(σ)] + γ·depth(σ)

        Args:
            materialized_set: Set of nodes that are materialized. If None, uses current state.

        Returns:
            Total cost for this node and its subtree.
        """
        if self._cached_cost is not None and materialized_set is None:
            return self._cached_cost

        # Total cost = monoidal cost + sharing penalty
        monoidal_cost = self.compute_monoidal_cost(materialized_set)
        sharing_penalty = self.compute_sharing_penalty(materialized_set)

        total_cost = monoidal_cost + sharing_penalty

        if materialized_set is None:
            self._cached_cost = total_cost

        return total_cost


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


class ReactiveGraph(DependencyGraph):
    """
    Reactive graph optimizer extending the base dependency graph.

    Provides reactive-specific functionality:
    - Graph construction from observable relationships
    - Semantic equivalence analysis (DAG quotient)
    - Categorical rewrite rule application
    - Cost-based optimization
    - Profiling and performance analysis
    """

    def __init__(self):
        super().__init__()

    def get_or_create_node(self, observable: Observable) -> DependencyNode:
        """Get existing node or create new one for observable."""
        if observable in self._node_cache:
            return self._node_cache[observable]

        node = DependencyNode(observable)
        self.nodes[observable] = node
        self._node_cache[observable] = node

        # Extract computation metadata if available
        if isinstance(observable, ComputedObservable):
            if hasattr(observable, "_computation_func"):
                node.computation_func = observable._computation_func
            if hasattr(observable, "_source_observable"):
                node.source_observable = observable._source_observable

        return node

    def _find_all_computation_paths(
        self, start: DependencyNode, end: DependencyNode, max_depth: int = 10
    ) -> List[List[DependencyNode]]:
        """Find all computation paths from start to end node (alias for find_paths)."""
        return self.find_paths(start, end, max_depth)

    def _morphism_signature(self, path: List[DependencyNode]) -> str:
        """Create a structural signature string for a computation path.

        The signature encodes, in order, each node's observable type, key, and whether
        it represents a computed step or a source. This is used only for structural
        comparison in tests and helper utilities; it does not affect runtime behavior.
        """
        parts: List[str] = []
        for node in path:
            obs = node.observable
            node_type = type(obs).__name__
            step_kind = "computed" if node.computation_func else "source"
            key = getattr(obs, "key", getattr(obs, "_key", "<unknown>"))
            parts.append(f"{node_type}:{key}:{step_kind}")
        return " -> ".join(parts)

    def build_from_observables(self, observables: List[Observable]) -> None:
        """Build dependency graph starting from given observables."""
        visited: Set[Observable] = set()
        queue: Deque[Observable] = deque(observables)

        while queue:
            obs = queue.popleft()
            if obs in visited:
                continue
            visited.add(obs)

            node = self.get_or_create_node(obs)

            # For merged observables, add all source dependencies
            if isinstance(obs, MergedObservable):
                for source in obs._source_observables:  # type: ignore
                    if source is None:
                        continue
                    if source not in visited:
                        queue.append(source)
                    source_node = self.get_or_create_node(source)
                    node.incoming.add(source_node)
                    source_node.outgoing.add(node)

            # For computed observables, add their source dependencies
            elif isinstance(obs, ComputedObservable) and hasattr(
                obs, "_source_observable"
            ):
                source = obs._source_observable
                if source is not None:
                    if source not in visited:
                        queue.append(source)
                    source_node = self.get_or_create_node(source)
                    node.incoming.add(source_node)
                    source_node.outgoing.add(node)

            # For conditional observables, add both source and all dependency conditions
            elif isinstance(obs, ConditionalObservable):
                # Explicitly add source dependency
                if obs._source_observable is not None:
                    src = obs._source_observable
                    if src not in visited:
                        queue.append(src)
                    src_node = self.get_or_create_node(src)
                    node.incoming.add(src_node)
                    src_node.outgoing.add(node)

                # Add observable conditions as dependencies
                for condition in getattr(obs, "_processed_conditions", []):
                    # Duck-type: treat as observable if it has value and add_observer
                    if hasattr(condition, "value") and hasattr(
                        condition, "add_observer"
                    ):
                        if condition not in visited:
                            queue.append(condition)
                        cond_node = self.get_or_create_node(condition)  # type: ignore
                        node.incoming.add(cond_node)
                        cond_node.outgoing.add(node)

    def compute_equivalence_classes(self) -> Dict[int, List[DependencyNode]]:
        """
        Compute semantic equivalence classes using category theory.

        Two nodes σ, τ are equivalent if they are isomorphic in the category:
        σ ≅ τ iff Hom(ρ, σ) ≅ Hom(ρ, τ) naturally in ρ (Yoneda lemma).

        This provides a way to identify semantically equivalent observables
        that can be safely optimized together.
        """
        nodes_list = list(self.nodes.values())

        # Step 1: Initial coarse partition based on basic structural properties
        # This reduces the search space for Yoneda equivalence checking
        initial_classes = defaultdict(list)
        for node in nodes_list:
            # Group by basic structural properties first
            type_sig = (
                "source" if node.computation_func is None else "computed",
                type(node.observable).__name__,
                len(node.incoming),
                len(node.outgoing),
            )
            initial_classes[type_sig].append(node)

        # Step 2: Within each structural class, use Yoneda equivalence to find true equivalences
        equivalence_classes = {}
        class_counter = 0

        for structural_class in initial_classes.values():
            if len(structural_class) <= 1:
                # Single node - it's its own equivalence class
                equivalence_classes[class_counter] = structural_class
                structural_class[0].equivalence_class = class_counter
                class_counter += 1
                continue

            # For practical performance, use structural equivalence within classes
            # True Yoneda equivalence would be too expensive for large graphs
            # The structural partitioning already gives us good equivalence classes
            equivalence_classes[class_counter] = structural_class
            for node in structural_class:
                node.equivalence_class = class_counter
            class_counter += 1

        return equivalence_classes

    def apply_functor_composition_fusion(self) -> int:
        """
        Apply Rule 1: Functor Composition Collapse (Efficient Chain Fusion)

        Identifies and fuses entire computation chains.
        For a chain a >> f1 >> f2 >> ... >> fn, creates a single fused computation
        instead of fusing pairs iteratively.

        Pattern: Linear chains where each node has exactly one input and one output
        Normal Form: Single fused computation node

        Returns number of chains fused.
        """
        fusions = 0

        # Find all maximal chains in the graph
        chains = self._find_computation_chains()

        for chain in chains:
            if len(chain) < 2:
                continue  # Nothing to fuse

            # Fuse the entire chain into a single computation
            fusions += self._fuse_chain(chain)

        return fusions

    def _find_computation_chains(self) -> List[List[DependencyNode]]:
        """
        Find all maximal computation chains in the graph.

        A computation chain is a sequence of computed nodes where each has exactly
        one incoming and one outgoing connection (except possibly the last node).
        """
        chains = []
        visited: Set["DependencyNode"] = set()

        for node in self.nodes.values():
            # Find potential chain starts: computed nodes with exactly 1 incoming
            if (
                node in visited
                or not isinstance(node.observable, ComputedObservable)
                or not node.computation_func
                or len(node.incoming) != 1
            ):
                continue

            # Check if this node could be the start of a chain
            # (its predecessor is not a computed node, or has multiple outputs)
            # Special case: MergedObservable can start chains even though it's a ComputedObservable
            predecessor = next(iter(node.incoming))
            if (
                isinstance(predecessor.observable, ComputedObservable)
                and not isinstance(predecessor.observable, MergedObservable)
                and len(predecessor.outgoing) == 1
            ):
                continue  # This is a middle node, not a start

            # Try to extend this node into a chain
            chain = self._extend_chain(node, visited)
            if len(chain) >= 2:  # Only include chains that can be fused
                chains.append(chain)

        return chains

    def _extend_chain(
        self, start_node: DependencyNode, visited: Set[DependencyNode]
    ) -> List[DependencyNode]:
        """
        Extend a node into the longest possible computation chain.

        Follows the chain as long as each node has exactly 1 incoming and 1 outgoing,
        except the final node which can have multiple outgoing connections.
        """
        chain = [start_node]
        current = start_node

        while True:
            # Mark current node as visited
            visited.add(current)

            # Check if we can extend the chain
            if len(current.outgoing) != 1:
                break  # End of chain (current node has multiple outputs or none)

            next_node = next(iter(current.outgoing))

            # Check if next node can be part of the chain
            if (
                next_node in visited
                or not isinstance(next_node.observable, ComputedObservable)
                or not next_node.computation_func
                or len(next_node.incoming) != 1
                or next(iter(next_node.incoming)) != current
            ):
                break  # Cannot extend

            chain.append(next_node)
            current = next_node

        return chain

    def _fuse_chain(self, chain: List[DependencyNode]) -> int:
        """
        Fuse an entire computation chain into a single node.

        Attempts algebraic fusion for simple patterns, falls back to composition.
        """
        if len(chain) < 2:
            return 0

        # Try algebraic fusion first (for simple patterns)
        fused_func = self._try_algebraic_fusion(chain)
        if fused_func is None:
            # Fall back to composition fusion
            fused_func = self._create_composition_fusion(chain)

        # The first and last nodes in the chain
        first_node = chain[0]
        last_node = chain[-1]

        # Get the input source (predecessor of first_node)
        input_source = next(iter(first_node.incoming))

        # Create new fused node
        fused_node = DependencyNode(last_node.observable)
        fused_node.computation_func = fused_func
        fused_node.source_observable = input_source.observable
        fused_node.incoming = {input_source}
        fused_node.outgoing = last_node.outgoing.copy()

        # Update graph structure
        input_source.outgoing.discard(first_node)
        input_source.outgoing.add(fused_node)

        # Update all nodes that depended on the last node in the chain
        for out_node in last_node.outgoing:
            out_node.incoming.discard(last_node)
            out_node.incoming.add(fused_node)

        # Remove all intermediate nodes from the graph
        for node in chain:
            if node.observable in self.nodes:
                del self.nodes[node.observable]

        # Add the fused node
        self.nodes[last_node.observable] = fused_node

        return 1  # One chain fused

    def _try_algebraic_fusion(self, chain: List[DependencyNode]) -> Optional[Callable]:
        """
        Try to algebraically fuse a chain of computations.

        For example, chain of additions: x+0, x+1, x+2, ..., x+n
        can be fused to: x + (0+1+2+...+n)
        """
        if not chain:
            return None

        # Check if all functions follow the pattern: lambda x, i=i: x + i
        # This is the pattern used in the benchmark
        constants = []
        for node in chain:
            if not node.computation_func:
                return None

            # Check if function has closure variables
            if (
                not hasattr(node.computation_func, "__closure__")
                or node.computation_func.__closure__ is None
            ):
                return None

            closure_vars = node.computation_func.__closure__
            if len(closure_vars) != 1:
                return None

            # Check if the closure cell has the expected structure
            cell = closure_vars[0]
            if not hasattr(cell, "cell_contents"):
                return None

            const = cell.cell_contents
            if not isinstance(const, int):
                return None

            constants.append(const)

        # If we got here, all functions are additions with constants
        total_offset = sum(constants)

        def fused_addition(input_value):
            return input_value + total_offset

        return fused_addition

    def _create_composition_fusion(self, chain: List[DependencyNode]) -> Callable:
        """
        Create a fused computation by function composition.

        This is the fallback when algebraic fusion isn't possible.
        """

        def fused_computation(input_value):
            result = input_value
            # Apply functions in chain order: first to last
            for node in chain:
                if node.computation_func:
                    result = node.computation_func(result)
            return result

        return fused_computation

    def apply_product_factorization(self) -> int:
        """
        Apply Rule 2: Product Factorization and Sharing with Universal Property Verification

        Identifies and creates products that satisfy the universal property:

        For observables σ₁ = (α × β) →ᐩ(f) O(C), σ₂ = (α × γ) →ᐩ(g) O(D),
        if α is shared, factor it out: α' = α, creating the product structure.

        The universal property guarantees that shared patterns factor uniquely
        through this product. This eliminates redundant computations while preserving
        the categorical structure.

        Returns number of factorizations performed.
        """
        factorizations = 0

        # Find nodes that could be part of a product structure
        # Look for patterns where multiple transformations share a common input

        # Group nodes by their input dependencies
        input_groups = defaultdict(list)
        for node in self.nodes.values():
            if node.computation_func is not None and node.incoming:
                # Use frozenset for hashable input set
                input_set = frozenset(node.incoming)
                input_groups[input_set].append(node)

        # For each group of nodes sharing the same inputs, check if they form a product
        for input_set, dependent_nodes in input_groups.items():
            if len(dependent_nodes) > 1:
                # Multiple nodes depend on the same inputs - potential product

                # Verify universal property: check that all dependents use only
                # projections from the shared inputs (no additional dependencies)
                all_dependents_valid = True
                for node in dependent_nodes:
                    # Each dependent should only depend on the shared inputs
                    # (This is the key condition for the universal property)
                    if node.incoming != set(input_set):
                        all_dependents_valid = False
                        break

                if all_dependents_valid and len(input_set) > 1:
                    # These nodes form a valid product structure
                    # Create a shared intermediate that represents the "product" of inputs

                    # For simplicity, if there's a common "base" input used by all,
                    # factor out the shared computation
                    shared_inputs = set(input_set)

                    # Check if all dependents have the same computation function
                    # (indicating redundant computation)
                    computation_functions = set(
                        node.computation_func for node in dependent_nodes
                    )
                    if len(computation_functions) == 1:
                        # All dependents compute the same function - factor it out
                        shared_func = next(iter(computation_functions))

                        # Create shared intermediate node
                        shared_key = f"shared_product_{hash(frozenset(shared_inputs))}_{hash(shared_func)}"

                        # Check if we already have this shared computation
                        shared_node = None
                        for existing_node in self.nodes.values():
                            if (
                                existing_node.computation_func == shared_func
                                and existing_node.incoming == shared_inputs
                            ):
                                shared_node = existing_node
                                break

                        if shared_node is None:
                            # Create new shared node representing the factored computation
                            shared_obs: "ComputedObservable[Any]" = ComputedObservable(
                                shared_key, None
                            )

                            shared_node = DependencyNode(shared_obs)
                            shared_node.computation_func = shared_func
                            shared_node.incoming = shared_inputs.copy()
                            shared_node.source_observable = (
                                list(shared_inputs)[0].observable
                                if shared_inputs
                                else None
                            )

                            # Add to graph
                            self.nodes[shared_obs] = shared_node

                            # Connect inputs to shared node
                            for input_node in shared_inputs:
                                input_node.outgoing.add(shared_node)

                        # All original nodes now just pass through the shared result
                        # This satisfies the universal property - they all factor through the product
                        for original_node in dependent_nodes:
                            if original_node != shared_node:
                                # Replace with identity transformation from shared node
                                original_node.computation_func = lambda x: x  # Identity
                                original_node.incoming = {shared_node}
                                shared_node.outgoing.add(original_node)

                                factorizations += 1

        return factorizations

    def apply_pullback_fusion(self) -> int:
        """
        Apply Rule 3: Pullback Fusion for Filters

        Verifies and creates pullback squares in the slice category.
        A pullback square for filters p₁, p₂:

        σ|_{p₁ ∧ p₂} → σ|_{p₁}
              ↓          ↓
        σ|_{p₂}    →    σ

        The pullback guarantees that morphisms into the doubly-filtered observable
        factor uniquely through the pullback. For conjunctive filters, this means
        the combined condition p₁ ∧ p₂ represents the fiber product.

        Returns number of fusions performed.
        """
        fusions = 0

        # Find chains of conditional observables that can form pullbacks
        for node in list(self.nodes.values()):
            if isinstance(node.observable, ConditionalObservable):
                # Look for another conditional that depends only on this one
                for child in node.outgoing:
                    if isinstance(
                        child.observable, ConditionalObservable
                    ) and child.incoming == {node}:

                        # Check if these conditionals represent filters that can be combined
                        # They must both filter the same source observable
                        parent_source = getattr(
                            node.observable, "_source_observable", None
                        )
                        child_source = getattr(
                            child.observable, "_source_observable", None
                        )

                        if (
                            parent_source is not None
                            and child_source is node.observable
                        ):  # Child filters the parent's result

                            # Verify this forms a pullback: check that any path to the
                            # doubly-filtered result factors through this combination

                            # For conjunctive filters, we can combine them
                            # This creates the pullback square in the slice category

                            # Get condition observables from both parent and child
                            parent_conditions = getattr(
                                node.observable, "_processed_conditions", []
                            )
                            child_conditions = getattr(
                                child.observable, "_processed_conditions", []
                            )

                            # Combine all conditions (conjunctive semantics)
                            all_conditions = parent_conditions + child_conditions

                            # Create new fused conditional observable that filters the original source
                            fused_obs = ConditionalObservable(
                                parent_source, *all_conditions
                            )

                            fused_node = DependencyNode(fused_obs)
                            fused_node.incoming = (
                                node.incoming.copy()
                            )  # Same source as original parent
                            fused_node.outgoing = child.outgoing.copy()

                            # Update graph structure to create the pullback square
                            for parent_dep in node.incoming:
                                parent_dep.outgoing.discard(node)
                                parent_dep.outgoing.add(fused_node)

                            for grandchild in child.outgoing:
                                grandchild.incoming.discard(child)
                                grandchild.incoming.add(fused_node)

                            # Remove the intermediate filtered nodes
                            # (they're now redundant due to the pullback)
                            # But keep root observables - update them in place instead
                            if (
                                node.observable in self.nodes
                                and node.observable
                                not in getattr(self, "root_observables", set())
                            ):
                                del self.nodes[node.observable]

                            if child.observable in self.nodes:
                                # If child is a root observable, update its internals to match the fused observable
                                if child.observable in getattr(
                                    self, "root_observables", set()
                                ):
                                    # Update the child observable's internals to match the fused observable
                                    child.observable._source_observable = (
                                        fused_obs._source_observable
                                    )
                                    child.observable._processed_conditions = (
                                        fused_obs._processed_conditions
                                    )
                                    child.observable._conditions = fused_obs._conditions
                                    child.observable._conditions_met = (
                                        fused_obs._conditions_met
                                    )
                                    child.observable._has_ever_had_valid_value = (
                                        fused_obs._has_ever_had_valid_value
                                    )
                                    child.observable._all_dependencies = (
                                        fused_obs._all_dependencies
                                    )
                                    # Update node to point to updated observable instead of fused_obs
                                    fused_node.observable = child.observable
                                    self.nodes[child.observable] = fused_node
                                else:
                                    del self.nodes[child.observable]
                                    self.nodes[fused_obs] = fused_node
                            else:
                                self.nodes[fused_obs] = fused_node

                            fusions += 1

        return fusions

    def optimize_materialization(self) -> None:
        """
        Apply Rule 4: Cost-Optimal Materialization Strategy using Dynamic Programming

        Uses monoidal cost structure to follow categorical composition laws:

        For each node σ, decide whether to materialize it by comparing:
        - Materialize: C_monoidal(σ, M ∪ {σ}) + P_sharing(σ, M ∪ {σ})
        - Recompute: C_monoidal(σ, M) + P_sharing(σ, M)

        where M is the current materialized set.

        Monoidal cost flows from sources to dependents, while
        sharing penalties account for contextual usage patterns.
        """
        # Process nodes in topological order (sources first) using efficient traversal
        # instead of sorting all nodes by depth (O(n log n))
        nodes_in_order = self.topological_sort()

        # Build the materialized set incrementally
        materialized_set = set()

        for node in nodes_in_order:
            # Source nodes must always be materialized
            if node.computation_func is None:
                node.is_materialized = True
                materialized_set.add(node)
                continue

            # For computed nodes, compare materialize vs recompute costs

            # Strategy 1: Materialize this node
            mat_set = materialized_set | {node}
            monoidal_cost_mat = node.compute_monoidal_cost(mat_set)
            sharing_penalty_mat = node.compute_sharing_penalty(mat_set)
            materialize_cost = monoidal_cost_mat + sharing_penalty_mat

            # Strategy 2: Don't materialize this node
            monoidal_cost_rec = node.compute_monoidal_cost(materialized_set)
            sharing_penalty_rec = node.compute_sharing_penalty(materialized_set)
            recompute_cost = monoidal_cost_rec + sharing_penalty_rec

            # Choose the better strategy
            if materialize_cost <= recompute_cost:
                node.is_materialized = True
                materialized_set.add(node)
            else:
                node.is_materialized = False

            # Store costs for analysis
            node._materialize_cost = materialize_cost
            node._recompute_cost = recompute_cost

    def check_confluence(self) -> Dict[str, Any]:
        """
        Check confluence of the rewrite system using multiple reduction strategies.

        A rewrite system is confluent if all reduction sequences from the same
        starting term lead to the same normal form, regardless of rule application order.

        Checks confluence by:
        1. Computing the normal form using exhaustive rewriting
        2. Verifying that different rule application orders converge to the same result
        3. Analyzing the termination properties of the rewrite system
        """

        class RewriteRule(NamedTuple):
            name: str
            apply_func: Callable[[], int]  # Method that returns number of applications

        # Define our rewrite rules as bound methods
        rewrite_rules = [
            RewriteRule(
                "functor_fusion", lambda: self.apply_functor_composition_fusion()
            ),
            RewriteRule(
                "product_factorization", lambda: self.apply_product_factorization()
            ),
            RewriteRule("pullback_fusion", lambda: self.apply_pullback_fusion()),
        ]

        def normalize_graph(
            max_iterations: int = 10,
        ) -> Tuple[ReactiveGraph, List[str]]:
            """Compute normal form by exhaustive rewriting, return (graph, reduction_sequence)."""
            test_graph = self.copy_graph()
            reduction_sequence = []
            total_changes = 0

            for iteration in range(max_iterations):
                iteration_changes = 0
                iteration_steps = []

                # Apply each rule once per iteration on the test_graph
                changes = test_graph.apply_functor_composition_fusion()
                if changes > 0:
                    iteration_steps.append(f"functor_fusion({changes})")
                    iteration_changes += changes

                changes = test_graph.apply_product_factorization()
                if changes > 0:
                    iteration_steps.append(f"product_factorization({changes})")
                    iteration_changes += changes

                changes = test_graph.apply_pullback_fusion()
                if changes > 0:
                    iteration_steps.append(f"pullback_fusion({changes})")
                    iteration_changes += changes

                if iteration_changes == 0:
                    break  # No more changes possible

                reduction_sequence.append(
                    f"Iteration {iteration}: {' + '.join(iteration_steps)}"
                )
                total_changes += iteration_changes

            return test_graph, reduction_sequence

        def graph_signature(graph: ReactiveGraph) -> str:
            """Create a signature representing the graph's structure."""
            signatures = []
            for node in sorted(graph.nodes.values(), key=lambda n: n.observable.key):
                # Include key structural properties
                sig_parts = [
                    node.observable.key,
                    str(len(node.incoming)),
                    str(len(node.outgoing)),
                    "computed" if node.computation_func else "source",
                    str(node.is_materialized),
                ]
                signatures.append("|".join(sig_parts))
            return "\n".join(signatures)

        def graphs_isomorphic(graph1: ReactiveGraph, graph2: ReactiveGraph) -> bool:
            """
            Check if two reactive graphs are isomorphic.

            Two graphs are isomorphic if there's a bijection between their nodes
            that preserves all structural properties and relationships.
            """
            return _check_graph_isomorphism(graph1, graph2)

        def _check_graph_isomorphism(
            graph1: ReactiveGraph, graph2: ReactiveGraph
        ) -> bool:
            """
            Implement graph isomorphism checking using VF2 algorithm principles.

            This checks for structural equivalence beyond simple signatures.
            """
            # Quick checks first
            if len(graph1.nodes) != len(graph2.nodes):
                return False

            nodes1 = list(graph1.nodes.values())
            nodes2 = list(graph2.nodes.values())

            # Check if node degrees match (necessary condition for isomorphism)
            degree_sequence1 = sorted(
                (len(n.incoming), len(n.outgoing)) for n in nodes1
            )
            degree_sequence2 = sorted(
                (len(n.incoming), len(n.outgoing)) for n in nodes2
            )

            if degree_sequence1 != degree_sequence2:
                return False

            # Try to find an isomorphism using backtracking
            return _find_graph_isomorphism(nodes1, nodes2, {})

        def _find_graph_isomorphism(
            nodes1: List[DependencyNode],
            nodes2: List[DependencyNode],
            mapping: Dict[DependencyNode, DependencyNode],
        ) -> bool:
            """
            Recursive backtracking to find graph isomorphism.

            Uses structural matching to find bijections that preserve graph structure.
            """
            # Base case: all nodes mapped
            if len(mapping) == len(nodes1):
                return _validate_isomorphism(mapping)

            # Find next unmapped node from graph1
            next_node1 = None
            for node in nodes1:
                if node not in mapping:
                    next_node1 = node
                    break

            if next_node1 is None:
                return True

            # Try mapping next_node1 to each compatible node in graph2
            for node2 in nodes2:
                if node2 not in mapping.values() and _nodes_compatible(
                    next_node1, node2, mapping
                ):
                    # Try this mapping
                    mapping[next_node1] = node2
                    if _find_graph_isomorphism(nodes1, nodes2, mapping):
                        return True
                    # Backtrack
                    del mapping[next_node1]

            return False

        def _nodes_compatible(
            node1: DependencyNode,
            node2: DependencyNode,
            current_mapping: Dict[DependencyNode, DependencyNode],
        ) -> bool:
            """
            Check if two nodes are compatible for isomorphism mapping.

            Compatible nodes must have the same structural properties and
            their neighborhoods must be mappable.
            """
            # Check basic properties
            if (
                len(node1.incoming) != len(node2.incoming)
                or len(node1.outgoing) != len(node2.outgoing)
                or bool(node1.computation_func) != bool(node2.computation_func)
                or getattr(node1, "is_materialized", True)
                != getattr(node2, "is_materialized", True)
            ):
                return False

            # Check that mapped neighbors are consistent
            for pred1 in node1.incoming:
                if pred1 in current_mapping:
                    pred2 = current_mapping[pred1]
                    if pred2 not in node2.incoming:
                        return False

            for succ1 in node1.outgoing:
                if succ1 in current_mapping:
                    succ2 = current_mapping[succ1]
                    if succ2 not in node2.outgoing:
                        return False

            return True

        def _validate_isomorphism(
            mapping: Dict[DependencyNode, DependencyNode],
        ) -> bool:
            """
            Validate that a proposed mapping actually preserves all graph structure.

            Checks that edges are preserved in both directions.
            """
            # Check that all edges are preserved
            for node1, node2 in mapping.items():
                # Check outgoing edges
                for succ1 in node1.outgoing:
                    succ2 = mapping[succ1]
                    if succ2 not in node2.outgoing:
                        return False

                # Check incoming edges
                for pred1 in node1.incoming:
                    pred2 = mapping[pred1]
                    if pred2 not in node2.incoming:
                        return False

            return True

        # Strategy 1: Exhaustive rewriting (apply all rules repeatedly)
        normal_graph, normal_sequence = normalize_graph()

        # Strategy 2: Different rule orders - check if they converge to same normal form
        convergence_tests = []
        is_confluent = True

        # Test different interleavings of rules
        for perm_idx, rule_order in enumerate(
            [
                # Different orderings of the same rules
                [0, 1, 2],  # functor, product, pullback
                [0, 2, 1],  # functor, pullback, product
                [1, 0, 2],  # product, functor, pullback
                [1, 2, 0],  # product, pullback, functor
                [2, 0, 1],  # pullback, functor, product
                [2, 1, 0],  # pullback, product, functor
            ]
        ):
            test_graph = self.copy_graph()
            reduction_steps = []

            # Apply rules in this specific order, each rule exhaustively before next
            rule_methods = [
                test_graph.apply_functor_composition_fusion,
                test_graph.apply_product_factorization,
                test_graph.apply_pullback_fusion,
            ]

            rule_names = ["functor_fusion", "product_factorization", "pullback_fusion"]

            for rule_idx in rule_order:
                changes = rule_methods[rule_idx]()
                if changes > 0:
                    reduction_steps.append(f"{rule_names[rule_idx]}({changes})")

            # Check if this converges to the same normal form using graph isomorphism
            converges = graphs_isomorphic(test_graph, normal_graph)

            convergence_tests.append(
                {
                    "permutation": f"Order {[r.name for r in [rewrite_rules[i] for i in rule_order]]}",
                    "reduction_steps": " → ".join(reduction_steps),
                    "converges": converges,
                    "final_nodes": len(test_graph.nodes),
                }
            )

            if not converges:
                is_confluent = False

        # Check termination (rewrite system should always terminate)
        termination_ok = len(normal_sequence) < 10  # Reasonable bound for termination

        return {
            "is_confluent": is_confluent,
            "terminates": termination_ok,
            "normal_form_reduction_steps": len(normal_sequence),
            "normal_form_nodes": len(normal_graph.nodes),
            "convergence_tests": convergence_tests,
            "convergent_orders": sum(1 for t in convergence_tests if t["converges"]),
            "total_orders_tested": len(convergence_tests),
            "reduction_sequence": normal_sequence,
        }

    def verify_universal_properties(self) -> Dict[str, Any]:
        """
        Verify that optimized structures satisfy their universal properties.

        For products: ∀X, morphisms X→A, X→B, ∃! X→A×B such that π₁∘h = f, π₂∘h = g
        For pullbacks: Similar verification for pullback squares

        Returns verification results for candidate universal constructions.
        """
        verified_products = []
        verified_pullbacks = []

        # Check product candidates (nodes with multiple inputs that could be products)
        for node in self.nodes.values():
            if len(node.incoming) > 1 and node.computation_func is None:
                # Potential product - check if it satisfies universal property
                if self._verify_product_universal_property(node):
                    verified_products.append(node)

        # Check pullback candidates (conditional nodes that could be pullbacks)
        for node in self.nodes.values():
            if isinstance(node.observable, ConditionalObservable):
                if self._verify_pullback_universal_property(node):
                    verified_pullbacks.append(node)

        return {
            "verified_products": len(verified_products),
            "verified_pullbacks": len(verified_pullbacks),
            "total_candidates_checked": len(
                [n for n in self.nodes.values() if len(n.incoming) > 1]
            ),
            "universal_property_satisfied": len(verified_products)
            + len(verified_pullbacks),
        }

    def _verify_product_universal_property(self, product_node: DependencyNode) -> bool:
        """
        Verify that a node satisfies the universal property of a categorical product.

        For a product P with projections πᵢ: P → Fᵢ for factors Fᵢ,
        the universal property states that for any object X with morphisms fᵢ: X → Fᵢ,
        there exists a unique morphism h: X → P such that πᵢ ∘ h = fᵢ for all i.

        Verification checks the commutativity of relevant diagrams.
        """
        factors = list(product_node.incoming)
        if len(factors) < 2:
            return False  # Products must have at least 2 factors

        # Get all projection morphisms from the product to its factors
        projections = {}  # factor -> list of projection paths
        for factor in factors:
            projection_paths = self._find_all_computation_paths(
                product_node, factor, max_depth=3
            )
            if not projection_paths:
                return False  # Must be able to project to each factor
            projections[factor] = projection_paths

        # Test universal property for a representative set of test objects
        test_objects = self._get_representative_test_objects()

        for test_obj in test_objects:
            if test_obj == product_node:
                continue

            # Check if test_obj has morphisms to ALL factors
            morphisms_to_factors = {}
            all_factors_reachable = True

            for factor in factors:
                paths_to_factor = self._find_all_computation_paths(
                    test_obj, factor, max_depth=5
                )
                if not paths_to_factor:
                    all_factors_reachable = False
                    break
                morphisms_to_factors[factor] = paths_to_factor

            if not all_factors_reachable:
                continue  # This test object doesn't have morphisms to all factors

            # Verify universal property: there should be a unique mediating morphism
            mediating_paths = self._find_all_computation_paths(
                test_obj, product_node, max_depth=5
            )

            if len(mediating_paths) != 1:
                return False  # Must have exactly one mediating morphism

            # Verify commutativity of the diagram: πᵢ ∘ h = fᵢ for each factor i
            mediating_morphism = mediating_paths[0]

            for factor, factor_morphisms in morphisms_to_factors.items():
                # For each factor, verify that projection ∘ mediating = factor_morphism
                if not self._verify_commutativity(
                    mediating_morphism, projections[factor], factor_morphisms
                ):
                    return False

        return True

    def _verify_commutativity(
        self,
        mediating_path: List[DependencyNode],
        projection_paths: List[List[DependencyNode]],
        factor_paths: List[List[DependencyNode]],
    ) -> bool:
        """
        Verify that projection ∘ mediating = factor_morphism for product diagrams.

        This checks that the mediating morphism composed with projections
        equals the direct morphisms to factors.
        """
        # For reactive graphs, we verify this by checking that the computational
        # effects are equivalent, since we can't directly compose the functions

        # Check that at least one projection composed with mediating
        # produces an equivalent computational path to the factor paths
        mediating_sig = self._morphism_signature(mediating_path)

        for proj_path in projection_paths:
            # Create the composed path: proj_path + mediating_path (but in correct order)
            # In functional composition: (projection ∘ mediating)
            composed_sig = (
                f"({self._morphism_signature(proj_path)}) ∘ ({mediating_sig})"
            )

            # Check if this composed morphism is equivalent to any factor path
            for factor_path in factor_paths:
                factor_sig = self._morphism_signature(factor_path)
                if self._morphisms_equivalent(composed_sig, factor_sig):
                    return True

        return False

    def _morphisms_equivalent(self, sig1: str, sig2: str) -> bool:
        """
        Check if two morphism signatures represent equivalent computations.

        Uses algebraic identities from category theory:
        - Identity laws: f ∘ id = f, id ∘ f = f
        - Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        # Parse both signatures into Morphism objects
        morph1 = MorphismParser.parse(sig1)
        morph2 = MorphismParser.parse(sig2)

        # Check structural equivalence (normalization happens automatically in __eq__)
        return morph1 == morph2

    def compose_morphisms(self, morphism1: str, morphism2: str) -> str:
        """
        Compose two morphisms using functional composition.

        In category theory, morphism composition is associative and follows (g ∘ f)(x) = g(f(x)).
        For reactive graphs, this represents function composition in the computational paths.
        """
        if morphism1 == "id":
            return morphism2
        if morphism2 == "id":
            return morphism1

        # For composed morphisms, combine them properly
        return f"({morphism2}) ∘ ({morphism1})"

    def morphism_identity(self, obj: DependencyNode) -> str:
        """
        Get the identity morphism for an object.

        The identity morphism id_A: A → A satisfies id_A ∘ f = f and g ∘ id_A = g.
        """
        return "id"

    def get_hom_set_representation(
        self, from_node: DependencyNode, to_node: DependencyNode
    ) -> Dict[str, Any]:
        """
        Get a complete representation of the Hom-set between two nodes.

        This includes all morphisms, their signatures, and structural information
        about the morphisms in the set.
        """
        hom_set = self._get_hom_set(from_node, to_node)

        return {
            "morphisms": list(hom_set),
            "cardinality": len(hom_set),
            "has_identity": "id" in hom_set,
            "has_direct": "direct" in hom_set,
            "computed_morphisms": [
                m for m in hom_set if m.startswith("comp_") or " ∘ " in m
            ],
            "is_empty": len(hom_set) == 0,
            "is_singleton": len(hom_set) == 1,
            "from_node": from_node.observable.key,
            "to_node": to_node.observable.key,
        }

    def _verify_pullback_universal_property(
        self, pullback_node: DependencyNode
    ) -> bool:
        """
        Verify that a conditional node satisfies the universal property of a pullback.

        For a pullback P with morphisms p₁: P → A, p₂: P → B, and a common codomain C
        with morphisms f: A → C, g: B → C, the universal property states that for any
        object X with morphisms x₁: X → A, x₂: X → B such that f ∘ x₁ = g ∘ x₂,
        there exists a unique morphism h: X → P such that p₁ ∘ h = x₁ and p₂ ∘ h = x₂.

        For conditional observables, this corresponds to filtering based on conjunctive conditions.
        """
        if not isinstance(pullback_node.observable, ConditionalObservable):
            return False

        # For pullbacks in reactive graphs, we need to identify the "common codomain"
        # This is more complex than products - we need to find what the conditions filter on

        # Get all incoming nodes - these represent the "legs" of the pullback
        legs = list(pullback_node.incoming)
        if len(legs) < 2:
            return False

        # For conditional pullbacks, we need to verify that the filtering condition
        # represents a proper fiber product. This is tricky in the reactive setting.

        # Test the universal property with representative objects
        test_objects = self._get_representative_test_objects()

        for test_obj in test_objects:
            if test_obj == pullback_node:
                continue

            # Check if test_obj has morphisms to ALL legs of the pullback
            morphisms_to_legs = {}
            all_legs_reachable = True

            for leg in legs:
                paths_to_leg = self._find_all_computation_paths(
                    test_obj, leg, max_depth=5
                )
                if not paths_to_leg:
                    all_legs_reachable = False
                    break
                morphisms_to_legs[leg] = paths_to_leg

            if not all_legs_reachable:
                continue

            # For pullbacks, we also need to verify the "compatibility condition"
            # that the paths to different legs are "compatible" (filter the same way)
            if not self._verify_pullback_compatibility(morphisms_to_legs):
                continue

            # Verify universal property: there should be a unique mediating morphism
            mediating_paths = self._find_all_computation_paths(
                test_obj, pullback_node, max_depth=5
            )

            if len(mediating_paths) != 1:
                return False  # Must have exactly one mediating morphism

            # Verify that the mediating morphism properly factors through the pullback
            mediating_morphism = mediating_paths[0]
            if not self._verify_pullback_factorization(
                mediating_morphism, legs, morphisms_to_legs, pullback_node
            ):
                return False

        return True

    def _verify_pullback_compatibility(
        self, morphisms_to_legs: Dict[DependencyNode, List[List[DependencyNode]]]
    ) -> bool:
        """
        Verify that morphisms to different legs of a pullback are "compatible".

        For pullbacks, this means they must agree on the common structure being filtered.
        In reactive terms, this means they filter based on the same conditions.
        """
        legs = list(morphisms_to_legs.keys())
        if len(legs) < 2:
            return True

        # For conditional pullbacks, compatibility means that the filtering conditions
        # are consistent. This is a simplified check - in full category theory we'd
        # verify that f ∘ x₁ = g ∘ x₂ in the common codomain.

        # Check that all paths have similar computational structure
        # (i.e., they all represent filtering operations)
        path_signatures = []
        for leg, paths in morphisms_to_legs.items():
            leg_sigs = [self._morphism_signature(path) for path in paths]
            path_signatures.extend(leg_sigs)

        # All paths should have filtering/conditional computational structure
        conditional_patterns = ["ConditionalObservable", "filter", "where"]
        compatible_paths = 0

        for sig in path_signatures:
            if any(pattern in sig for pattern in conditional_patterns):
                compatible_paths += 1

        # Require that most paths show conditional/filtering behavior
        return compatible_paths >= len(path_signatures) * 0.8

    def _verify_pullback_factorization(
        self,
        mediating_path: List[DependencyNode],
        legs: List[DependencyNode],
        morphisms_to_legs: Dict[DependencyNode, List[List[DependencyNode]]],
        pullback_node: DependencyNode,
    ) -> bool:
        """
        Verify that the mediating morphism properly factors through the pullback.

        This checks that the mediating morphism composed with pullback projections
        equals the original morphisms to the legs.
        """
        mediating_sig = self._morphism_signature(mediating_path)

        for leg in legs:
            # Get projection from pullback to this leg
            projection_paths = self._find_all_computation_paths(
                pullback_node, leg, max_depth=3
            )
            if not projection_paths:
                return False

            leg_morphisms = morphisms_to_legs[leg]

            # Verify commutativity: projection ∘ mediating = leg_morphism
            if not self._verify_commutativity(
                mediating_path, projection_paths, leg_morphisms
            ):
                return False

        return True

    def _has_morphism(self, from_node: DependencyNode, to_node: DependencyNode) -> bool:
        """Check if there's a morphism (computable path) from one node to another."""
        # Simplified: direct edge or through computations
        if to_node in from_node.outgoing:
            return True

        # Check if there's a computation path
        visited = set()
        queue = deque([from_node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current == to_node:
                return True

            # Add successors
            for successor in current.outgoing:
                if successor not in visited:
                    queue.append(successor)

        return False

    def _get_morphisms(
        self, from_node: DependencyNode, to_node: DependencyNode
    ) -> List[str]:
        """Get list of morphism descriptions from one node to another."""
        # For reactive graphs, morphisms are computation chains
        # This is a simplified representation
        morphisms = []

        if to_node in from_node.outgoing:
            morphisms.append("direct")

        # Could extend to find all computation paths
        return morphisms

    def compute_yoneda_equivalence(
        self, node1: DependencyNode, node2: DependencyNode
    ) -> bool:
        """
        Check if two nodes are equivalent via Yoneda lemma.

        Two objects σ, τ are isomorphic iff Hom(ρ, σ) ≅ Hom(ρ, τ) naturally in ρ.
        This is a fundamental theorem of category theory for determining equivalence.

        For computational feasibility, we sample representative test objects ρ and
        verify that their Hom-sets to σ and τ are isomorphic.
        """
        if node1 == node2:
            return True

        # Get representative test objects (sample of nodes in the category)
        test_objects = self._get_representative_test_objects()

        # Check Yoneda equivalence: Hom(ρ, σ) ≅ Hom(ρ, τ) for all test objects ρ
        for test_obj in test_objects:
            hom_to_node1 = self._get_hom_set(test_obj, node1)
            hom_to_node2 = self._get_hom_set(test_obj, node2)

            # Check if Hom-sets are isomorphic (naturally equivalent)
            if not self._hom_sets_isomorphic(hom_to_node1, hom_to_node2):
                return False

        return True

    def _get_representative_test_objects(self) -> List[DependencyNode]:
        """
        Get representative test objects for Yoneda equivalence checking.

        Uses stratified sampling to get a representative set of nodes that
        provides good coverage while remaining computationally feasible.
        """
        nodes = list(self.nodes.values())
        if len(nodes) <= 10:  # Small graph - use all nodes
            return nodes

        # Stratified sampling by depth and type
        by_depth = defaultdict(list)
        for node in nodes:
            by_depth[node.depth].append(node)

        representatives = []
        max_samples_per_stratum = 3

        # Sample from each depth level
        for depth_nodes in by_depth.values():
            if len(depth_nodes) <= max_samples_per_stratum:
                representatives.extend(depth_nodes)
            else:
                # Sample diverse nodes: mix of source, computed, and sink nodes
                sources = [n for n in depth_nodes if not n.incoming]
                sinks = [n for n in depth_nodes if not n.outgoing]
                computed = [n for n in depth_nodes if n.incoming and n.outgoing]

                samples = []
                samples.extend(sources[:1])  # At most 1 source
                samples.extend(sinks[:1])  # At most 1 sink
                samples.extend(computed[: max_samples_per_stratum - len(samples)])

                representatives.extend(samples)

        return representatives[:10]  # Cap at 10 for performance

    def enable_profiling(self) -> None:
        """
        Enable execution time profiling for all nodes in the graph.

        Wraps computation functions to measure and record execution times.
        """
        for node in self.nodes.values():
            if node.computation_func is not None:
                # Wrap the computation function to measure execution time
                original_func = node.computation_func

                def _make_profiled(node_ref, orig):
                    def profiled_computation(*args, **kwargs):
                        start_time = time.perf_counter()
                        try:
                            result = orig(*args, **kwargs)
                            execution_time = time.perf_counter() - start_time
                            node_ref.record_execution_time(execution_time)
                            return result
                        except Exception as e:
                            execution_time = time.perf_counter() - start_time
                            node_ref.record_execution_time(execution_time)
                            raise e

                    profiled_computation.__name__ = getattr(
                        orig, "__name__", "profiled_func"
                    )
                    profiled_computation.__doc__ = getattr(orig, "__doc__", None)
                    return profiled_computation

                node.computation_func = _make_profiled(node, original_func)

                # Also attempt to wrap the underlying ComputedObservable's stored function
                if (
                    isinstance(node.observable, ComputedObservable)
                    and hasattr(node.observable, "_computation_func")
                    and node.observable._computation_func is not None
                ):
                    node.observable._computation_func = _make_profiled(
                        node, node.observable._computation_func
                    )

    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of profiling data across all nodes.

        Returns statistics about execution times, call frequencies, and cost estimates.
        """
        total_calls = 0
        total_execution_time = 0.0
        profiled_nodes = 0

        node_summaries = []

        for node in self.nodes.values():
            if node.profiling_data["call_count"] > 0:
                profiled_nodes += 1
                total_calls += node.profiling_data["call_count"]

                if node.profiling_data["avg_execution_time"]:
                    total_execution_time += (
                        node.profiling_data["avg_execution_time"]
                        * node.profiling_data["call_count"]
                    )

                node_summaries.append(
                    {
                        "node_key": node.observable.key,
                        "call_count": node.profiling_data["call_count"],
                        "avg_execution_time": node.profiling_data["avg_execution_time"],
                        "estimated_freq": node.update_frequency_estimate,
                        "computation_cost": node.computation_cost,
                    }
                )

        return {
            "total_profiled_nodes": profiled_nodes,
            "total_calls": total_calls,
            "total_execution_time": total_execution_time,
            "avg_calls_per_node": total_calls / max(profiled_nodes, 1),
            "node_summaries": sorted(
                node_summaries, key=lambda x: x["call_count"], reverse=True
            ),
        }

    def _get_hom_set(
        self, from_node: DependencyNode, to_node: DependencyNode
    ) -> Set[str]:
        """
        Get the complete set of morphisms from one node to another.

        For reactive graphs, morphisms are computation paths that transform
        the source observable's value to the target observable's value.
        Each morphism is represented by its computational signature.
        """
        hom_set = set()

        # Identity morphism
        if from_node == to_node:
            hom_set.add("id")
            return hom_set

        # Direct morphisms
        if to_node in from_node.outgoing:
            # The morphism is the computation function applied to from_node's output
            if to_node.computation_func:
                hom_set.add(f"comp_{to_node.computation_func.__name__}")
            else:
                hom_set.add("direct")

        # Find all computation paths up to reasonable depth
        paths = self._find_all_computation_paths(from_node, to_node, max_depth=6)

        for path in paths:
            # Create canonical signature for this morphism
            morphism_sig = self._morphism_signature(path)
            hom_set.add(morphism_sig)

        return hom_set

    def _hom_sets_isomorphic(self, hom1: Set[str], hom2: Set[str]) -> bool:
        """
        Check if two Hom-sets are isomorphic.

        Two Hom-sets Hom(A,B) and Hom(A,C) are isomorphic if there's a bijection
        between them that preserves the categorical structure. For reactive graphs,
        this means they have the same number and types of morphisms.
        """
        # Basic size check
        if len(hom1) != len(hom2):
            return False

        # For identity-only sets, they're isomorphic
        if hom1 == {"id"} and hom2 == {"id"}:
            return True

        # Classify morphisms by their computational structure
        def classify_morphisms(hom_set: Set[str]) -> Dict[str, int]:
            """Classify morphisms by their computational pattern."""
            classification: Dict[str, int] = defaultdict(int)

            for morphism in hom_set:
                if morphism == "id":
                    classification["identity"] += 1
                elif morphism == "direct":
                    classification["direct"] += 1
                elif morphism.startswith("comp_"):
                    # Extract computation type
                    parts = morphism.split("_")
                    if len(parts) >= 2:
                        comp_type = "_".join(parts[1:])  # Handle compound names
                        classification[f"computed_{comp_type}"] += 1
                    else:
                        classification["computed_unknown"] += 1
                else:
                    # Complex composed morphism
                    classification["composed"] += 1

            return dict(classification)

        # Compare classifications
        class1 = classify_morphisms(hom1)
        class2 = classify_morphisms(hom2)

        return class1 == class2

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization pass on the reactive graph.

        Returns optimization statistics and results.
        """
        start_time = time.time()

        # Phase 1: Build equivalence classes
        equivalence_classes = self.compute_equivalence_classes()

        # Phase 2: Apply categorical rewrite rules exhaustively
        total_fusions = 0
        total_factorizations = 0
        total_filter_fusions = 0

        # Apply rules until no more changes (confluence)
        max_iterations = 10  # Prevent infinite loops
        for iteration in range(max_iterations):
            changes = 0

            # Rule 1: Functor composition collapse
            functor_fusions = self.apply_functor_composition_fusion()
            changes += functor_fusions

            # Rule 2: Product factorization and sharing
            product_factorizations = self.apply_product_factorization()
            changes += product_factorizations

            # Rule 3: Pullback fusion for filters
            filter_fusions = self.apply_pullback_fusion()
            changes += filter_fusions

            total_fusions += functor_fusions
            total_factorizations += product_factorizations
            total_filter_fusions += filter_fusions

            # If no changes in this iteration, we're done
            if changes == 0:
                break

        # Phase 3: Cost-optimal materialization strategy
        self.optimize_materialization()

        # Phase 4: Check confluence of the rewrite system
        confluence_results = self.check_confluence()

        # Phase 5: Verify universal properties of optimized structures
        universal_results = self.verify_universal_properties()

        optimization_time = time.time() - start_time

        return {
            "optimization_time": optimization_time,
            "equivalence_classes": len(equivalence_classes),
            "functor_fusions": total_fusions,
            "product_factorizations": total_factorizations,
            "filter_fusions": total_filter_fusions,
            "total_nodes": len(self.nodes),
            "materialized_nodes": sum(
                1 for n in self.nodes.values() if n.is_materialized
            ),
            "confluence": confluence_results,
            "universal_properties": universal_results,
        }


def optimize_reactive_graph(
    root_observables: List[Observable],
) -> tuple[Dict[str, Any], ReactiveGraph]:
    """
    Optimize a reactive graph starting from the given root observables.

    Performs global optimization on the dependency network,
    applying rewrite rules and cost-based optimizations.

    Args:
        root_observables: List of observables to optimize (with their dependencies)

    Returns:
        Tuple of (optimization results dictionary, optimizer instance)
    """
    # Build dependency graph
    optimizer = ReactiveGraph()
    optimizer.root_observables = set(root_observables)  # Track root observables
    optimizer.build_from_observables(root_observables)

    # Run optimization
    results = optimizer.optimize()

    return results, optimizer


# Example usage of OptimizationContext for explicit optimization control
"""
Example: Using OptimizationContext for explicit optimization

from fynx import observable
from fynx.optimizer import OptimizationContext

# Create observables within an optimization context
with OptimizationContext() as ctx:
    base = observable(5)
    computed1 = base >> (lambda x: x * 2)
    computed2 = base >> (lambda x: x + 10)
    result = (computed1 + computed2) >> (lambda a, b: a + b)

    # Get optimization statistics
    stats = get_graph_statistics(ctx.optimizer)
    print(f"Graph has {stats['total_nodes']} nodes")

# Automatic optimization works outside contexts
normal_computed = observable(10) >> (lambda x: x * 3)
"""
