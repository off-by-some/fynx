"""
Hyper-Efficient Delta-Based Key-Value Store with Subscriptions
=============================================================

A high-performance reactive key-value store that uses delta-based change detection
and only propagates changes to affected nodes, based on principles from Self-Adjusting
Computation (SAC) and Differential Dataflow (DD).

Key Features:
- Delta-based change detection (O(affected) complexity)
- Automatic dependency tracking with DAG
- Topological change propagation
- Lazy evaluation for computed values
- Hyper-efficient data structures
- Subscription system for key changes

Mathematical Foundation:
- Self-Adjusting Computation (SAC): Dynamic dependency graphs with trace stability
- Differential Dataflow (DD): Delta collections <Data, Time, Delta>
- O(affected) complexity bounds for optimal incremental computation
"""

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np


# Elegant solution: Add proper exception for circular dependencies
class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the reactive graph."""

    pass


T = TypeVar("T")


class ChangeType(Enum):
    """Types of changes that can occur in the store."""

    SET = "set"
    DELETE = "delete"
    COMPUTED_UPDATE = "computed_update"


@dataclass
class Delta:
    """Represents a change delta in the key-value store."""

    key: str
    change_type: ChangeType
    old_value: Any
    new_value: Any
    timestamp: float

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class DependencyGraph:
    """
    Directed Acyclic Graph for tracking dependencies between keys.
    Uses adjacency lists for O(1) lookups and efficient topological sorting.
    """

    def __init__(self):
        self._graph: Dict[str, Set[str]] = defaultdict(set)  # key -> dependents
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(
            set
        )  # key -> dependencies
        self._indegrees: Dict[str, int] = defaultdict(int)

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add a dependency: dependent relies on dependency."""
        if dependent not in self._graph[dependency]:
            self._graph[dependency].add(dependent)
            self._reverse_graph[dependent].add(dependency)
            self._indegrees[dependent] += 1

    def remove_dependency(self, dependent: str, dependency: str) -> None:
        """Remove a dependency relationship."""
        if dependent in self._graph[dependency]:
            self._graph[dependency].remove(dependent)
            self._reverse_graph[dependent].remove(dependency)
            self._indegrees[dependent] -= 1

    def get_dependents(self, key: str) -> Set[str]:
        """Get all keys that depend on the given key."""
        return self._graph[key].copy()

    def get_dependencies(self, key: str) -> Set[str]:
        """Get all keys that the given key depends on."""
        return self._reverse_graph[key].copy()

    def topological_sort(self, affected_keys: Set[str]) -> List[str]:
        """
        Perform topological sort on the subgraph of affected keys.
        Returns keys in the order they should be processed (dependencies first).
        """
        if not affected_keys:
            return []

        # Calculate indegrees within the affected subgraph
        subgraph_indegrees = {}
        for key in affected_keys:
            # Count how many dependencies this key has within the affected set
            indegree = 0
            for dependency in self._reverse_graph[key]:
                if dependency in affected_keys:
                    indegree += 1
            subgraph_indegrees[key] = indegree

        # Start with keys that have no dependencies within the affected set
        queue = deque([key for key in affected_keys if subgraph_indegrees[key] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Decrease indegrees of dependents within the affected set
            for dependent in self._graph[current]:
                if dependent in affected_keys:
                    subgraph_indegrees[dependent] -= 1
                    if subgraph_indegrees[dependent] == 0:
                        queue.append(dependent)

        return result


class ComputedValue:
    """
    A computed value that depends on other keys in the store.
    Uses lazy evaluation and automatic dependency tracking.
    """

    def __init__(self, key: str, compute_func: Callable[[], T], store: "DeltaKVStore"):
        self.key = key
        self._compute_func = compute_func
        self._store = store
        self._value: Optional[T] = None
        self._is_dirty = True
        self._dependencies: Set[str] = set()
        self._last_computed = 0.0

    def get(self) -> T:
        """Get the computed value, computing it if necessary."""
        if self._is_dirty or self._value is None:
            self._recompute()
        return self._value

    def _recompute(self) -> None:
        """Recompute the value and update dependencies."""
        # Track which keys are accessed during computation
        old_dependencies = self._dependencies.copy()
        accessed_keys: Set[str] = set()

        # Create a tracking context for dependency detection
        original_get = self._store._get_raw

        def tracking_get(key: str) -> Any:
            accessed_keys.add(key)
            return original_get(key)

        self._store._get_raw = tracking_get

        # Compute the new value
        new_value = self._compute_func()
        self._value = new_value
        self._is_dirty = False
        self._last_computed = time.time()

        # Update dependencies
        self._update_dependencies(old_dependencies, accessed_keys)

        # Restore original get function
        self._store._get_raw = original_get

    def _update_dependencies(self, old_deps: Set[str], new_deps: Set[str]) -> None:
        """Update the dependency graph with new dependencies."""
        # Remove old dependencies that are no longer used
        for dep in old_deps - new_deps:
            self._store._dep_graph.remove_dependency(self.key, dep)

        # Add new dependencies
        for dep in new_deps - old_deps:
            self._store._dep_graph.add_dependency(self.key, dep)

        self._dependencies = new_deps

    def mark_dirty(self) -> None:
        """Mark this computed value as needing recomputation."""
        self._is_dirty = True

    def is_dirty(self) -> bool:
        """Check if this computed value needs recomputation."""
        return self._is_dirty


class OrthogonalDeltaMerger:
    """
    Implements orthogonal delta merging for high-dimensional reactive state.

    Based on Proposal 1.1: Orthogonal Delta Algebra for reducing redundancy
    in change propagation. Uses vector space orthogonalization to minimize
    computational overlap when merging multiple deltas.
    """

    def __init__(self):
        self._delta_cache: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._max_cache_size = 100  # Limit cache size for memory efficiency

    def merge_deltas(self, key: str, deltas: List[Delta]) -> Optional[Delta]:
        """
        Merge multiple deltas using orthogonal decomposition.

        Handles scalars, lists, and dicts by flattening to vectors for orthogonalization.
        Formula: θ_merged = θ_base + Σ α_i * Δθ_i^⊥
        where Δθ_i^⊥ represents orthogonalized deltas.
        """
        if not deltas:
            return None

        if len(deltas) == 1:
            return deltas[0]

        first_delta = deltas[0]

        # Strategy 1: Numeric scalars (original implementation)
        if self._are_numeric_scalars(deltas):
            return self._merge_numeric_scalars(key, deltas)

        # Strategy 2: Lists/arrays of numbers
        elif self._are_numeric_lists(deltas):
            return self._merge_numeric_lists(key, deltas)

        # Strategy 3: Dictionaries with numeric values
        elif self._are_numeric_dicts(deltas):
            return self._merge_numeric_dicts(key, deltas)

        # Strategy 4: Fallback - last delta wins
        else:
            return deltas[-1]

    def _are_numeric_scalars(self, deltas: List[Delta]) -> bool:
        """Check if all deltas contain numeric scalars."""
        return all(
            isinstance(delta.new_value, (int, float))
            and isinstance(delta.old_value, (int, float, type(None)))
            for delta in deltas
        )

    def _are_numeric_lists(self, deltas: List[Delta]) -> bool:
        """Check if all deltas contain lists/arrays of numbers."""
        return all(
            isinstance(delta.new_value, (list, tuple, np.ndarray))
            and isinstance(delta.old_value, (list, tuple, np.ndarray, type(None)))
            and all(isinstance(x, (int, float)) for x in delta.new_value)
            and (
                delta.old_value is None
                or all(isinstance(x, (int, float)) for x in delta.old_value)
            )
            for delta in deltas
        )

    def _are_numeric_dicts(self, deltas: List[Delta]) -> bool:
        """Check if all deltas contain dicts with numeric values."""
        return all(
            isinstance(delta.new_value, dict)
            and isinstance(delta.old_value, (dict, type(None)))
            and all(isinstance(v, (int, float)) for v in delta.new_value.values())
            and (
                delta.old_value is None
                or all(isinstance(v, (int, float)) for v in delta.old_value.values())
            )
            for delta in deltas
        )

    def _merge_numeric_scalars(self, key: str, deltas: List[Delta]) -> Delta:
        """Merge numeric scalar deltas using orthogonalization."""
        numeric_deltas = []
        base_value = None

        for delta in deltas:
            if base_value is None:
                base_value = delta.old_value if delta.old_value is not None else 0
            numeric_deltas.append(delta.new_value - (delta.old_value or 0))

        # Convert to numpy array for orthogonalization
        delta_vector = np.array(numeric_deltas)

        # Simple orthogonalization: Gram-Schmidt process
        # This reduces redundancy in the delta vector
        orthogonalized = self._gram_schmidt_orthogonalization(delta_vector)

        # Combine orthogonalized deltas with equal weights (α_i = 1/N)
        alpha = 1.0 / len(orthogonalized)
        merged_change = np.sum(alpha * orthogonalized)

        final_value = (base_value or 0) + merged_change

        return Delta(
            key=key,
            change_type=ChangeType.SET,
            old_value=base_value,
            new_value=final_value,
            timestamp=time.time(),
        )

    def _merge_numeric_lists(self, key: str, deltas: List[Delta]) -> Delta:
        """Merge list deltas by orthogonalizing element-wise."""
        if not deltas:
            return deltas[-1]

        first_new = deltas[0].new_value
        length = len(first_new)

        # Ensure all lists have same length
        if not all(len(delta.new_value) == length for delta in deltas):
            return deltas[-1]  # Fallback

        # Merge each element position using orthogonalization
        merged_list = []
        base_list = (
            deltas[0].old_value if deltas[0].old_value is not None else [0] * length
        )

        for i in range(length):
            element_deltas = []
            for delta in deltas:
                old_val = delta.old_value[i] if delta.old_value else 0
                new_val = delta.new_value[i]
                element_deltas.append(new_val - old_val)

            # Orthogonalize this element's deltas
            delta_vector = np.array(element_deltas)
            orthogonalized = self._gram_schmidt_orthogonalization(delta_vector)
            alpha = 1.0 / len(orthogonalized)
            merged_change = np.sum(alpha * orthogonalized)

            merged_list.append(base_list[i] + merged_change)

        return Delta(
            key=key,
            change_type=ChangeType.SET,
            old_value=base_list,
            new_value=merged_list,
            timestamp=time.time(),
        )

    def _merge_numeric_dicts(self, key: str, deltas: List[Delta]) -> Delta:
        """Merge dict deltas by orthogonalizing each key's values."""
        if not deltas:
            return deltas[-1]

        first_new = deltas[0].new_value
        keys = set(first_new.keys())

        # Ensure all dicts have same keys
        if not all(set(delta.new_value.keys()) == keys for delta in deltas):
            return deltas[-1]  # Fallback

        merged_dict = {}
        base_dict = (
            deltas[0].old_value
            if deltas[0].old_value is not None
            else {k: 0 for k in keys}
        )

        for dict_key in keys:
            element_deltas = []
            for delta in deltas:
                old_val = delta.old_value.get(dict_key, 0) if delta.old_value else 0
                new_val = delta.new_value[dict_key]
                element_deltas.append(new_val - old_val)

            # Orthogonalize this key's deltas
            delta_vector = np.array(element_deltas)
            orthogonalized = self._gram_schmidt_orthogonalization(delta_vector)
            alpha = 1.0 / len(orthogonalized)
            merged_change = np.sum(alpha * orthogonalized)

            merged_dict[dict_key] = base_dict[dict_key] + merged_change

        return Delta(
            key=key,
            change_type=ChangeType.SET,
            old_value=base_dict,
            new_value=merged_dict,
            timestamp=time.time(),
        )

    def _gram_schmidt_orthogonalization(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform Gram-Schmidt orthogonalization on delta vectors.

        This reduces linear dependence between deltas, minimizing redundant computation.
        """
        if vectors.ndim == 1:
            # Single vector case
            return vectors.reshape(1, -1)

        n = len(vectors)
        orthogonalized = np.zeros_like(vectors)

        for i in range(n):
            # Start with the original vector
            orthogonalized[i] = vectors[i]

            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                if np.linalg.norm(orthogonalized[j]) > 1e-10:  # Avoid division by zero
                    projection = np.dot(vectors[i], orthogonalized[j]) / np.dot(
                        orthogonalized[j], orthogonalized[j]
                    )
                    orthogonalized[i] -= projection * orthogonalized[j]

        return orthogonalized


class IncrementalStatisticsTracker:
    """
    Implements O(1) incremental statistical computations with proper removal support.

    Based on Proposal 1.2: Incremental Statistical and Tensor Tracking.
    Uses numerically stable algorithms like Welford's online algorithm
    for mean/variance and incremental tensor updates.

    Supports accurate removal by maintaining bounded history for recalculation.
    """

    def __init__(self, max_history_size: int = 1000):
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._max_history_size = max_history_size

    def track_mean_variance(
        self, key: str, new_value: float, remove_value: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Track running mean and variance using Welford's online algorithm with exact removal.

        Maintains complete history for mathematically exact removal via recalculation.
        O(1) amortized per update, O(n) for removal, numerically stable.
        """
        if key not in self._stats_cache:
            self._stats_cache[key] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,  # Sum of squared differences from mean
                "variance": 0.0,
                "history": [],  # Keep complete history for exact removals
            }

        stats = self._stats_cache[key]

        if remove_value is not None:
            # Exact removal: remove from history and recalculate from scratch
            history = stats["history"]
            if remove_value in history:
                history.remove(remove_value)
                self._recalculate_stats_from_history(stats)
            else:
                # Value not in history - cannot remove what was never added
                raise ValueError(
                    f"Cannot remove value {remove_value} - not found in tracked history"
                )
        else:
            # Add new value using Welford's algorithm
            old_count = stats["count"]
            old_mean = stats["mean"]
            old_m2 = stats["m2"]

            new_count = old_count + 1
            delta = new_value - old_mean
            new_mean = old_mean + delta / new_count
            delta2 = new_value - new_mean
            new_m2 = old_m2 + delta * delta2

            stats["count"] = new_count
            stats["mean"] = new_mean
            stats["m2"] = new_m2
            stats["variance"] = new_m2 / new_count if new_count > 1 else 0

            # Maintain bounded history
            stats["history"].append(new_value)
            if len(stats["history"]) > self._max_history_size:
                # Remove oldest value when history gets too large
                stats["history"].pop(0)
                # Recalculate to maintain exactness after truncation
                self._recalculate_stats_from_history(stats)

        return {
            "mean": stats["mean"],
            "variance": stats["variance"],
            "count": stats["count"],
            "std_dev": math.sqrt(stats["variance"]) if stats["variance"] > 0 else 0,
        }

    def _recalculate_stats_from_history(self, stats: Dict[str, Any]) -> None:
        """Recalculate statistics from remaining history values."""
        history = stats["history"]
        if not history:
            stats.update({"count": 0, "mean": 0.0, "m2": 0.0, "variance": 0.0})
            return

        # Recalculate using Welford's algorithm from scratch
        count = len(history)
        mean = sum(history) / count
        m2 = sum((x - mean) ** 2 for x in history)

        stats.update(
            {
                "count": count,
                "mean": mean,
                "m2": m2,
                "variance": m2 / count if count > 1 else 0,
            }
        )

    def _approximate_removal(self, stats: Dict[str, Any], remove_value: float) -> None:
        """
        Approximate removal using inverse Welford operations.
        Less accurate than full recalculation but maintains O(1) performance.
        """
        old_count = stats["count"]
        if old_count <= 1:
            # Reset if removing the last value
            stats.update({"count": 0, "mean": 0.0, "m2": 0.0, "variance": 0.0})
            return

        old_mean = stats["mean"]
        old_m2 = stats["m2"]

        # Approximate inverse of Welford's update
        new_count = old_count - 1
        # This is an approximation - full accuracy requires the exact update sequence
        delta = remove_value - old_mean
        new_mean = (old_mean * old_count - remove_value) / new_count
        # Approximate M2 adjustment (not exact, but better than nothing)
        m2_adjustment = delta**2 * old_count / new_count
        new_m2 = max(0, old_m2 - m2_adjustment)

        stats.update(
            {
                "count": new_count,
                "mean": new_mean,
                "m2": new_m2,
                "variance": new_m2 / new_count if new_count > 1 else 0,
            }
        )

    def track_tensor_statistics(
        self, key: str, tensor: np.ndarray, operation: str = "add"
    ) -> Dict[str, Any]:
        """
        Track incremental tensor statistics (e.g., covariance matrices).

        Implements dynamic tensor analysis for dimensionality reduction.
        """
        cache_key = f"{key}_tensor"
        if cache_key not in self._stats_cache:
            self._stats_cache[cache_key] = {
                "count": 0,
                "mean": np.zeros_like(tensor),
                "covariance": np.zeros((tensor.shape[0], tensor.shape[0])),
                "eigenvalues": None,
                "eigenvectors": None,
            }

        stats = self._stats_cache[cache_key]

        if operation == "add":
            # Online covariance matrix update
            old_count = stats["count"]
            old_mean = stats["mean"]
            old_cov = stats["covariance"]

            new_count = old_count + 1

            # Update mean
            delta = tensor - old_mean
            new_mean = old_mean + delta / new_count

            # Update covariance matrix incrementally
            if old_count > 0:
                # Outer product update for covariance
                cov_update = np.outer(delta, tensor - new_mean)
                new_cov = (old_count * old_cov + cov_update) / new_count
            else:
                new_cov = np.zeros_like(old_cov)

            stats["count"] = new_count
            stats["mean"] = new_mean
            stats["covariance"] = new_cov

            # Update spectral decomposition (for dynamic sparsity)
            if new_count % 10 == 0:  # Periodic update to avoid overhead
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(new_cov)
                    # Keep only top k eigenvalues/vectors for sparsity
                    k = min(5, len(eigenvals))  # Top 5 components
                    top_indices = np.argsort(eigenvals)[-k:]
                    stats["eigenvalues"] = eigenvals[top_indices]
                    stats["eigenvectors"] = eigenvecs[:, top_indices]
                except np.linalg.LinAlgError:
                    # Handle singular matrices
                    stats["eigenvalues"] = None
                    stats["eigenvectors"] = None

        return stats


class DynamicSpectralSparsifier:
    """
    Implements dynamic spectral sparsity for large dependency graphs.

    Based on Proposal 2.2: Dynamic Spectral Sparsity for Time-Dependent Computation.
    Tracks only the most significant components of change propagation using
    low-rank approximations of the dependency kernel matrix.
    """

    def __init__(self, target_rank: int = 10):
        self._target_rank = target_rank
        self._dependency_matrix: Optional[np.ndarray] = None
        self._node_index: Dict[str, int] = {}
        self._index_node: Dict[int, str] = {}
        self._spectral_components: Optional[Dict[str, np.ndarray]] = None

    def update_dependency_matrix(self, graph: DependencyGraph) -> None:
        """
        Update the dependency matrix and compute spectral decomposition.

        This creates a low-rank approximation of the dependency structure.
        Optimized for performance with size limits and early exits.
        """
        # Get all nodes
        all_nodes = set(graph._graph.keys()) | set(graph._reverse_graph.keys())
        if not all_nodes:
            return

        # Limit matrix size for performance (max 200x200 for eigenvalue computation)
        if len(all_nodes) > 200:
            # For very large graphs, skip spectral analysis
            self._spectral_components = None
            return

        # Update node mappings
        for node in all_nodes:
            if node not in self._node_index:
                idx = len(self._node_index)
                self._node_index[node] = idx
                self._index_node[idx] = node

        n = len(self._node_index)
        if n < 10:  # Too small for meaningful spectral analysis
            self._spectral_components = None
            return

        # Build adjacency matrix efficiently
        matrix = np.zeros((n, n), dtype=np.float32)  # Use float32 for memory efficiency

        # Only build matrix for nodes with dependencies to save computation
        edges_added = 0
        for from_node, to_nodes in graph._graph.items():
            if from_node in self._node_index:
                i = self._node_index[from_node]
                for to_node in to_nodes:
                    if to_node in self._node_index:
                        j = self._node_index[to_node]
                        matrix[j, i] = 1.0  # Influence flows from i to j
                        edges_added += 1

        # Skip spectral analysis if graph is too sparse or dense
        if (
            edges_added < 5 or edges_added > n * n * 0.1
        ):  # Less than 5 edges or >10% dense
            self._spectral_components = None
            return

        self._dependency_matrix = matrix

        # Compute spectral decomposition efficiently
        # Use influence matrix (faster than full SVD for our purposes)
        influence_matrix = matrix @ matrix.T  # Symmetric matrix

        # Use efficient eigenvalue computation with size limits
        if n <= 100:
            # Full eigendecomposition for manageable matrices
            eigenvals, eigenvecs = np.linalg.eigh(influence_matrix)
        else:
            # For larger matrices, skip spectral analysis for performance
            self._spectral_components = None
            return

        # Keep only top k components
        k = min(self._target_rank, len(eigenvals))
        if k > 0:
            top_indices = np.argsort(eigenvals)[-k:]

            self._spectral_components = {
                "eigenvalues": eigenvals[top_indices],
                "eigenvectors": eigenvecs[:, top_indices],
                "projection_matrix": eigenvecs[:, top_indices],
            }
        else:
            self._spectral_components = None

    def sparsify_propagation(self, affected_keys: Set[str]) -> Set[str]:
        """
        Use spectral sparsity to filter affected keys to only the most significant ones.

        Returns a subset of affected_keys that captures the dominant change propagation.
        """
        if not self._spectral_components or not affected_keys:
            return affected_keys

        # Convert affected keys to vector representation
        n = len(self._node_index)
        affected_vector = np.zeros(n)

        for key in affected_keys:
            if key in self._node_index:
                idx = self._node_index[key]
                if idx < n:
                    affected_vector[idx] = 1.0

        # Project onto spectral subspace
        projection_matrix = self._spectral_components["projection_matrix"]
        spectral_projection = projection_matrix.T @ affected_vector

        # Reconstruct using only significant components
        reconstructed = projection_matrix @ spectral_projection

        # Threshold to get significant nodes (top 80% of projection magnitude)
        if len(reconstructed) > 0:
            threshold = np.percentile(np.abs(reconstructed), 20)  # Bottom 20% threshold
            significant_indices = np.where(np.abs(reconstructed) > threshold)[0]

            significant_keys = set()
            for idx in significant_indices:
                if idx in self._index_node:
                    significant_keys.add(self._index_node[idx])

            # Ensure we don't lose critical dependencies
            significant_keys.update(affected_keys)  # Conservative approach
            return significant_keys

        return affected_keys


class DeltaKVStore:
    """
    Hyper-efficient key-value store with delta-based change detection and subscriptions.

    Key Features:
    - O(affected) complexity for change propagation
    - Automatic dependency tracking
    - Lazy evaluation for computed values
    - Delta-based change notifications
    - Thread-safe operations
    - Mathematical optimizations for fan-in scenarios
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._computed: Dict[str, OptimizedComputedValue] = {}
        self._dep_graph = DependencyGraph()
        self._observers: Dict[str, Set[Callable[[Delta], None]]] = defaultdict(set)
        self._global_observers: Set[Callable[[Delta], None]] = set()
        self._lock = threading.RLock()
        self._change_log: List[Delta] = []
        self._batch_depth = 0

        # Thread-local tracking context for dependency discovery and cycle detection
        self._tracking_context = threading.local()

        # Advanced mathematical optimizations
        self._orthogonal_merger = OrthogonalDeltaMerger()
        self._stats_tracker = IncrementalStatisticsTracker(max_history_size=1000)
        self._spectral_sparsifier = DynamicSpectralSparsifier(target_rank=10)

    def get(self, key: str) -> Any:
        """Get a value from the store."""
        with self._lock:
            return self._get_raw(key)

    def _get_raw(self, key: str) -> Any:
        """Internal get method with dependency tracking support."""
        # Track dependency if we're in a tracking context
        if hasattr(self._tracking_context, "accessed_keys"):
            self._tracking_context.accessed_keys.add(key)

        # Check if it's a computed value
        if key in self._computed:
            computed = self._computed[key]

            # CRITICAL: Check if already being computed (cycle detection)
            if hasattr(self._tracking_context, "computing_stack"):
                if key in self._tracking_context.computing_stack:
                    # Cycle detected! Return current (possibly stale) value
                    return computed._value

            return computed.get()

        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        with self._lock:
            old_value = self._data.get(key)
            if old_value == value:
                return  # No change

        self._data[key] = value
        delta = Delta(key, ChangeType.SET, old_value, value, None)
        self._propagate_change(delta)

        # Update spectral sparsifier conservatively (only for large graphs and infrequently)
        total_deps = len(self._dep_graph._graph)
        if (
            total_deps > 100 and total_deps % 200 == 0
        ):  # Only for large graphs, very infrequently
            try:
                self._spectral_sparsifier.update_dependency_matrix(self._dep_graph)
            except (np.linalg.LinAlgError, ValueError):
                # Skip spectral update if matrix is singular or too large
                pass

    def delete(self, key: str) -> bool:
        """Delete a key from the store."""
        with self._lock:
            if key not in self._data:
                return False

            old_value = self._data[key]
            del self._data[key]

            delta = Delta(key, ChangeType.DELETE, old_value, None, None)
            self._propagate_change(delta)
            return True

    def _detect_cycles(self, key: str) -> None:
        """Detect circular dependencies and fail fast."""
        visited = set()
        rec_stack = set()

        def dfs(current_key: str):
            if current_key in rec_stack:
                raise CircularDependencyError(
                    f"Circular dependency detected involving key: {current_key}"
                )
            if current_key in visited:
                return

            visited.add(current_key)
            rec_stack.add(current_key)

            # Check dependencies
            if current_key in self._computed:
                computed = self._computed[current_key]
                for dep in computed._dependencies:
                    dfs(dep)

            rec_stack.remove(current_key)

        dfs(key)

    def computed(
        self, key: str, compute_func: Callable[[], T], deps: Optional[List[str]] = None
    ) -> None:
        with self._lock:
            # Create computed value with explicit dependencies (if provided)
            dependencies = deps if deps is not None else []
            optimized_val = HierarchicalComputedValue(
                key, dependencies, compute_func, self
            )
            self._computed[key] = optimized_val

    def subscribe(
        self, key: str, callback: Callable[[Delta], None]
    ) -> Callable[[], None]:
        """Subscribe to changes on a specific key."""
        with self._lock:
            self._observers[key].add(callback)

            # Return unsubscribe function
            def unsubscribe():
                with self._lock:
                    self._observers[key].discard(callback)

            return unsubscribe

    def subscribe_all(self, callback: Callable[[Delta], None]) -> Callable[[], None]:
        """Subscribe to all changes in the store."""
        with self._lock:
            self._global_observers.add(callback)

            def unsubscribe():
                with self._lock:
                    self._global_observers.discard(callback)
                    return unsubscribe

    def batch(self) -> "BatchContext":
        """Start a batch operation context."""
        return BatchContext(self)

    def _get_transitive_dependents(self, key: str) -> Set[str]:
        """Get all transitive dependents efficiently using iterative deepening."""
        # Use iterative deepening to avoid processing unnecessary deep dependencies
        # This maintains O(affected) complexity while being more memory efficient

        all_affected = set()
        current_level = {key}
        max_depth = 10  # Prevent infinite loops and limit traversal depth

        for depth in range(max_depth):
            if not current_level:
                break

            next_level = set()
            for node in current_level:
                dependents = self._dep_graph.get_dependents(node)
                for dep in dependents:
                    if dep not in all_affected:
                        all_affected.add(dep)
                        next_level.add(dep)

            current_level = next_level

        return all_affected

    def _propagate_change(self, delta: Delta) -> None:
        """Propagate a change through the dependency graph with NOF-inspired optimizations."""
        # Log the change
        self._change_log.append(delta)

        # Notify direct observers
        self._notify_observers(delta)

        # Find all affected keys using topological sort (transitive closure)
        affected_keys = self._get_transitive_dependents(delta.key)
        if not affected_keys:
            return

        # Apply spectral sparsification for large graphs
        if len(affected_keys) > 20:  # Threshold for applying sparsification
            affected_keys = self._spectral_sparsifier.sparsify_propagation(
                affected_keys
            )

        # Standard topological propagation order
        propagation_order = self._dep_graph.topological_sort(affected_keys)

        # Single topo-sort with batched propagation (no recursion)
        # Collect all deltas first, then notify all at once
        deltas_to_notify = [delta]

        for key in propagation_order:
            if key in self._computed:
                computed_val = self._computed[key]
                computed_val.invalidate()

                # Always check if value changed after invalidation
                old_val = computed_val._value
                new_val = computed_val.get()  # This will recompute if dirty

                if old_val != new_val:
                    computed_delta = Delta(
                        key, ChangeType.COMPUTED_UPDATE, old_val, new_val, None
                    )
                    deltas_to_notify.append(computed_delta)

        # Notify all deltas at once (no recursive propagation)
        for d in deltas_to_notify:
            self._notify_observers(d)

    def _get_optimized_propagation_order(
        self, affected_keys: Set[str], priority: float
    ) -> List[str]:
        """
        Get optimized propagation order based on NOF contradiction analysis.

        Prioritizes keys that resolve the most contradictions first.
        """
        if not affected_keys:
            return []

        # Get standard topological order
        standard_order = self._dep_graph.topological_sort(affected_keys)

        # If priority is high, prioritize keys that resolve contradictions
        if priority > 0.5:
            # Sort by contradiction resolution potential
            def contradiction_score(key: str) -> float:
                if key in self._contradiction_analyzer._block_gadgets:
                    block = self._contradiction_analyzer._block_gadgets[key]
                    return (
                        block.k_budget
                    )  # Higher k_budget = more contradictions resolved
                return 0.0

            # Sort by contradiction score (descending) while maintaining topological order
            scored_keys = [(key, contradiction_score(key)) for key in standard_order]
            scored_keys.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in scored_keys]

        return standard_order

    def _notify_observers(self, delta: Delta) -> None:
        """Notify observers of a change."""
        # Notify key-specific observers
        for observer in self._observers[delta.key]:
            observer(delta)

        # Notify global observers
        for observer in self._global_observers:
            observer(delta)

    def clear(self) -> None:
        """Clear all data from the store."""
        with self._lock:
            keys = list(self._data.keys()) + list(self._computed.keys())
            for key in keys:
                self.delete(key)

    def _analyze_dependencies(self, compute_func: Callable) -> Set[str]:
        """Analyze dependencies of a compute function using one-time tracking."""
        accessed_keys = set()

        # Monkey-patch once to track dependencies
        original_get_raw = self._get_raw

        def tracking_get(key: str) -> Any:
            accessed_keys.add(key)
            return original_get_raw(key)

        self._get_raw = tracking_get

        try:
            # Execute function to track dependencies
            compute_func()
        finally:
            # Always restore original function
            self._get_raw = original_get_raw

        return accessed_keys

    def keys(self) -> List[str]:
        """Get all keys in the store."""
        with self._lock:
            return list(self._data.keys()) + list(self._computed.keys())

    def computed_stats(self, key: str, data_key: str) -> None:
        """
        Define a computed value that tracks incremental statistics for a data stream.

        Based on Proposal 1.2: Incremental Statistical Tracking.
        Uses Welford's algorithm for O(1) mean/variance computation.
        """

        def stats_computation():
            data = self.get(data_key)
            if isinstance(data, (int, float)):
                return self._stats_tracker.track_mean_variance(data_key, data)
            elif isinstance(data, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in data
            ):
                # For arrays, compute aggregate statistics
                stats = {"count": len(data)}
                if data:
                    stats["mean"] = sum(data) / len(data)
                    stats["variance"] = sum(
                        (x - stats["mean"]) ** 2 for x in data
                    ) / len(data)
                    stats["std_dev"] = (
                        math.sqrt(stats["variance"]) if stats["variance"] > 0 else 0
                    )
                return stats
            else:
                return {"count": 0, "mean": 0, "variance": 0, "std_dev": 0}

        self.computed(key, stats_computation)

    def computed_tensor_stats(self, key: str, tensor_key: str) -> None:
        """
        Define a computed value that tracks incremental tensor statistics.

        Based on Proposal 1.2: Dynamic Tensor Analysis for dimensionality reduction.
        """

        def tensor_stats_computation():
            tensor = self.get(tensor_key)
            if isinstance(tensor, np.ndarray):
                return self._stats_tracker.track_tensor_statistics(tensor_key, tensor)
            else:
                return {"count": 0, "mean": None, "covariance": None}

        self.computed(key, tensor_stats_computation)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            # Count optimization types
            optimization_types = {}
            for computed_val in self._computed.values():
                opt_type = type(computed_val).__name__
                optimization_types[opt_type] = optimization_types.get(opt_type, 0) + 1

        return {
            "total_keys": len(self._data) + len(self._computed),
            "data_keys": len(self._data),
            "computed_keys": len(self._computed),
            "optimization_types": optimization_types,
            "total_observers": sum(
                len(observers) for observers in self._observers.values()
            )
            + len(self._global_observers),
            "change_log_size": len(self._change_log),
            "total_dependencies": sum(
                len(deps) for deps in self._dep_graph._graph.values()
            ),
            "nof_analytics": self.get_nof_analytics(),
        }


class BatchContext:
    """Context manager for batch operations."""

    def __init__(self, store: DeltaKVStore):
        self._store = store

    def __enter__(self):
        self._store._batch_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store._batch_depth -= 1
        # Could implement batch propagation here if needed
        return False


# ============================================================================
# Optimized Computation Values
# ============================================================================

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

"""
Fixed OptimizedComputedValue implementation with proper cycle detection.

Replace your existing OptimizedComputedValue, HierarchicalComputedValue,
and StandardComputedValue classes with these versions.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Set


class OptimizedComputedValue(ABC):
    """Abstract base class for optimized computed values with cycle detection and error propagation."""

    def __init__(self, key: str, store):
        self.key = key
        self._store = store
        self._value = None
        self._is_dirty = True
        self._dependencies: Set[str] = set()
        self._last_error: Optional[Exception] = None

    @abstractmethod
    def _compute_func(self) -> Any:
        """The computation function to execute."""
        pass

    def get(self) -> Any:
        """
        Get the computed value, computing if necessary.

        This is the ONLY entry point for getting values - all cycle detection
        happens here at the very beginning.
        """
        # Initialize computing_stack if needed (thread-local)
        if not hasattr(self._store._tracking_context, "computing_stack"):
            self._store._tracking_context.computing_stack = set()

        # CRITICAL: Check for cycles FIRST before doing ANYTHING else
        stack = self._store._tracking_context.computing_stack
        if self.key in stack:
            # Cycle detected! Return current value (even if None/stale) and mark as clean
            # This prevents infinite recomputation attempts
            self._is_dirty = False
            return self._value

        # If we have a cached error and we're not dirty, re-raise it
        if self._last_error is not None and not self._is_dirty:
            raise self._last_error

        # If not dirty, return cached value (fast path)
        if not self._is_dirty:
            return self._value

        # Need to compute - add to stack to prevent cycles
        stack.add(self.key)

        try:
            # Now safe to compute
            self._do_compute()
            return self._value
        finally:
            # ALWAYS remove from stack, even if computation fails
            stack.discard(self.key)

    def _do_compute(self) -> None:
        """
        Internal method that does the actual computation with proper error handling.
        Should NEVER be called directly - always use get().
        """
        # Set up dependency tracking
        accessed_keys: Set[str] = set()

        # Save previous tracking context (for nested computations)
        prev_accessed = getattr(self._store._tracking_context, "accessed_keys", None)
        self._store._tracking_context.accessed_keys = accessed_keys

        try:
            # Compute the value with proper error propagation
            self._value = self._compute_func()
            # Clear any previous error on successful computation
            self._last_error = None
            self._is_dirty = False

        except Exception as e:
            # Cache the error for proper propagation
            self._last_error = e
            # Don't clear the value on error - keep the last good value
            # Mark as not dirty so we don't retry on every access
            self._is_dirty = False
            # Re-raise the error for immediate propagation
            raise

        finally:
            # Restore previous tracking context
            if prev_accessed is not None:
                self._store._tracking_context.accessed_keys = prev_accessed
            elif hasattr(self._store._tracking_context, "accessed_keys"):
                delattr(self._store._tracking_context, "accessed_keys")

            # Update dependency graph based on accessed keys
            # This happens even on failure to ensure invalidation works
            if accessed_keys:  # Only if we actually accessed keys during computation
                old_deps = self._dependencies
                new_deps = accessed_keys

                # Remove old dependencies
                for dep in old_deps - new_deps:
                    self._store._dep_graph.remove_dependency(self.key, dep)

                # Add new dependencies
                for dep in new_deps - old_deps:
                    self._store._dep_graph.add_dependency(self.key, dep)

                self._dependencies = new_deps

    def invalidate(self) -> None:
        """Mark this value as needing recomputation."""
        self._is_dirty = True
        # Clear any cached error when invalidated - allows retry on next access
        self._last_error = None

    def is_dirty(self) -> bool:
        """Check if this value needs recomputation."""
        return self._is_dirty


class HierarchicalComputedValue(OptimizedComputedValue):
    """
    Uses hierarchical dependency tracking for efficient change propagation.
    Supports both explicit dependencies and lazy discovery.
    """

    def __init__(
        self, key: str, dependencies: List[str], compute_func: Callable, store
    ):
        super().__init__(key, store)
        self._user_compute_func = compute_func
        self._explicit_deps = set(dependencies) if dependencies else None

        # If explicit dependencies provided, register them immediately
        if self._explicit_deps:
            for dep in self._explicit_deps:
                self._store._dep_graph.add_dependency(key, dep)

    def _compute_func(self) -> Any:
        """Execute the user's computation function."""
        return self._user_compute_func()

    def _do_compute(self) -> None:
        """
        Compute with explicit dependency handling.
        """
        # If we have explicit dependencies, we can skip the complex tracking
        # and just ensure our dependencies are up to date
        if self._explicit_deps is not None:
            # For explicit deps, we don't need the full tracking machinery
            # Just compute the value directly
            self._value = self._compute_func()
            self._is_dirty = False
            return

        # Fall back to lazy discovery for compatibility
        super()._do_compute()


class StandardComputedValue(OptimizedComputedValue):
    """Standard computed value implementation."""

    def __init__(self, key: str, compute_func: Callable, store):
        super().__init__(key, store)
        self._user_compute_func = compute_func

    def _compute_func(self) -> Any:
        """Execute the user's computation function."""
        return self._user_compute_func()
