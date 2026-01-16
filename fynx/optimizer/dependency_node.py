"""
Dependency Node for Reactive Graph Optimization
==============================================

This module contains the extended DependencyNode class used in reactive graph optimization.
"""

import time
from typing import Any, Dict, List, Optional, Set

from ..observable.base import Observable
from .dependency_graph import DependencyNode as BaseDependencyNode


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
        # Scale sample size with graph complexity (at least 50, up to 200)
        max_samples = min(200, max(50, len(self.incoming) + len(self.outgoing) * 10))
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
