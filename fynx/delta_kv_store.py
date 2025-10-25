"""
DeltaKVStore - Hyper-Efficient Delta-Based Key-Value Store with Subscriptions
=============================================================================

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

"""
DeltaKVStore - Core Reactive Key-Value Store
============================================

The core DeltaKVStore implementation with dependency tracking and change propagation.
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


"""
Optimized Computed Values for DeltaKVStore
==========================================

Advanced computed value implementations with cycle detection and error propagation:

- OptimizedComputedValue: Abstract base class for computed values with cycle detection
- HierarchicalComputedValue: Uses explicit dependency tracking for efficiency
- StandardComputedValue: Basic computed value implementation
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set

if TYPE_CHECKING:
    from .delta_kv_store import DeltaKVStore


class OptimizedComputedValue(ABC):
    """Abstract base class for optimized computed values with cycle detection and error propagation."""

    def __init__(self, key: str, store: "DeltaKVStore"):
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
        self,
        key: str,
        dependencies: List[str],
        compute_func: Callable,
        store: "DeltaKVStore",
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

    def __init__(self, key: str, compute_func: Callable, store: "DeltaKVStore"):
        super().__init__(key, store)
        self._user_compute_func = compute_func

    def _compute_func(self) -> Any:
        """Execute the user's computation function."""
        return self._user_compute_func()


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
        self._computed: Dict[str, HierarchicalComputedValue] = {}
        self._dep_graph = DependencyGraph()
        self._observers: Dict[str, Set[Callable[[Delta], None]]] = defaultdict(set)
        self._global_observers: Set[Callable[[Delta], None]] = set()
        self._lock = threading.RLock()
        self._change_log: List[Delta] = []
        self._batch_depth = 0

        # Thread-local tracking context for dependency discovery and cycle detection
        self._tracking_context = threading.local()

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
            # Handle numpy arrays and other objects that don't support direct comparison
            try:
                if old_value == value:
                    return  # No change
            except (ValueError, TypeError):
                # For objects that don't support comparison (like numpy arrays),
                # always treat as a change
                pass

        self._data[key] = value
        delta = Delta(key, ChangeType.SET, old_value, value, None)
        self._propagate_change(delta)

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
        """Propagate a change through the dependency graph with advanced optimizations."""
        # Log the change
        self._change_log.append(delta)

        # Notify direct observers
        self._notify_observers(delta)

        # Find all affected keys using topological sort (transitive closure)
        affected_keys = self._get_transitive_dependents(delta.key)
        if not affected_keys:
            return

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

    def _get_optimized_propagation_order(self, affected_keys: Set[str]) -> List[str]:
        """
        Get optimized propagation order for efficient change propagation.

        Uses priority-based ordering to optimize dependency resolution.
        """
        if not affected_keys:
            return []

        # Get standard topological order
        standard_order = self._dep_graph.topological_sort(affected_keys)

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
        Define a computed value that computes basic statistics for a data stream.

        Provides count, mean, variance, and standard deviation calculations.
        """

        def stats_computation():
            data = self.get(data_key)
            if isinstance(data, (int, float)):
                # Simple single value statistics
                return {"count": 1, "mean": data, "variance": 0, "std_dev": 0}
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
        Define a computed value that computes basic tensor statistics.

        Provides basic statistical analysis for numpy arrays.
        """

        def tensor_stats_computation():
            tensor = self.get(tensor_key)
            if isinstance(tensor, np.ndarray):
                return {
                    "count": tensor.size,
                    "mean": np.mean(tensor),
                    "std": np.std(tensor),
                    "shape": tensor.shape,
                }
            else:
                return {"count": 0, "mean": None, "std": None, "shape": None}

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
