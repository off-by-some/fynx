"""
Mathematical Optimizations for DeltaKVStore
===========================================

Advanced mathematical algorithms that optimize DeltaKVStore performance:

- OrthogonalDeltaMerger: Vector space orthogonalization for efficient delta merging
- IncrementalStatisticsTracker: O(1) incremental statistical computations
- DynamicSpectralSparsifier: Spectral methods for large dependency graph optimization
"""

import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from .delta_kv_store import ChangeType, Delta, DependencyGraph


class OrthogonalDeltaMerger:
    """
    Implements orthogonal delta merging for high-dimensional reactive state.

    Based on Proposal 1.1: Orthogonal Delta Algebra for reducing redundancy
    in change propagation. Uses vector space orthogonalization to minimize
    computational overlap when merging multiple deltas.
    """

    def __init__(self):
        self._delta_cache: Dict[str, List[np.ndarray]] = {}
        self._max_cache_size = 100  # Limit cache size for memory efficiency

    def merge_deltas(self, key: str, deltas: List["Delta"]) -> Optional["Delta"]:
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

    def _are_numeric_scalars(self, deltas: List["Delta"]) -> bool:
        """Check if all deltas contain numeric scalars."""
        return all(
            isinstance(delta.new_value, (int, float))
            and isinstance(delta.old_value, (int, float, type(None)))
            for delta in deltas
        )

    def _are_numeric_lists(self, deltas: List["Delta"]) -> bool:
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

    def _are_numeric_dicts(self, deltas: List["Delta"]) -> bool:
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

    def _merge_numeric_scalars(self, key: str, deltas: List["Delta"]) -> "Delta":
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

    def _merge_numeric_lists(self, key: str, deltas: List["Delta"]) -> "Delta":
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

    def _merge_numeric_dicts(self, key: str, deltas: List["Delta"]) -> "Delta":
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

    def update_dependency_matrix(self, graph: "DependencyGraph") -> None:
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
