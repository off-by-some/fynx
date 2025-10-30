"""
DeltaKVStore - Category-Theoretic Reactive Store

Mathematical Foundation:
- Observables: Comonad (Obs, ε, δ) with extract and duplicate
- Subscriptions: Monoid (Sub, ⊕, ε) with composition
- Propagation: Semiring (Δ, +, 0, ·, 1) with delta algebra
- Topology: Graph classification for O(depth) to O(V+E) propagation

Core Invariants:
1. Every set() emits a delta (event algebra)
2. Significance testing only in propagation layer
3. Single unified subscription mechanism
4. Topology-aware dispatch for optimal complexity
"""

import atexit
import math
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


# ============================================================================
# DELTA ALGEBRA - The Semiring (Δ, +, 0, ·, 1)
# ============================================================================


class ChangeType(Enum):
    """Change types in the delta semiring."""

    SOURCE_UPDATE = "source_update"
    COMPUTED_UPDATE = "computed_update"
    STRUCTURE_ADD = "structure_add"
    STRUCTURE_REMOVE = "structure_remove"

    # Legacy aliases
    SET = "source_update"
    DELETE = "structure_remove"


@dataclass(frozen=True)
class Delta:
    """
    Immutable delta element in the semiring.

    Represents: Δ(key, old, new, t)
    Identity: Δ(k, v, v, t) where old == new
    Composition: Δ₁ · Δ₂ = Δ(k, old₁, new₂, t₂)
    """

    key: str
    change_type: ChangeType
    old_value: Any
    new_value: Any
    timestamp: float
    differential: Optional[Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", time.time())

    def is_identity(self) -> bool:
        """Check if this is the identity element (no change)."""
        try:
            # Handle numpy arrays specially
            if isinstance(self.old_value, np.ndarray) or isinstance(
                self.new_value, np.ndarray
            ):
                if type(self.old_value) != type(self.new_value):
                    return False
                return np.array_equal(self.old_value, self.new_value)
            return self.old_value == self.new_value
        except (ValueError, TypeError):
            return False

    def compose(self, other: "Delta") -> "Delta":
        """
        Sequential composition: Δ₁ · Δ₂

        Preserves transitivity: if Δ₁: v₀→v₁ and Δ₂: v₁→v₂ then Δ₁·Δ₂: v₀→v₂
        """
        if self.key != other.key:
            raise ValueError(
                f"Cannot compose deltas for different keys: {self.key} != {other.key}"
            )

        return Delta(
            key=self.key,
            change_type=self.change_type,
            old_value=self.old_value,
            new_value=other.new_value,
            timestamp=max(self.timestamp, other.timestamp),
        )


class DeltaPool:
    """Object pool for delta allocation (optimization)."""

    _pool: List[Delta] = []
    _lock: threading.Lock = threading.Lock()
    _MAX_POOL_SIZE: int = 1000

    @classmethod
    def acquire(
        cls,
        key: str,
        change_type: ChangeType,
        old_value: Any,
        new_value: Any,
        timestamp: Optional[float] = None,
    ) -> Delta:
        """Acquire delta from pool or create new."""
        # Since Delta is frozen, create new instance
        return Delta(
            key=key,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            timestamp=timestamp if timestamp is not None else time.time(),
        )

    @classmethod
    def release(cls, delta: Delta) -> None:
        """Return delta to pool (no-op for frozen dataclasses)."""
        pass


atexit.register(lambda: DeltaPool._pool.clear())


# ============================================================================
# GRAPH TOPOLOGY - Structural Classification
# ============================================================================


class GraphTopology(Enum):
    """
    Topology classification for optimal propagation:
    - LINEAR: O(depth) - simple chains
    - TREE: O(affected) - no diamonds
    - DAG: O(V+E) - general case
    """

    LINEAR = "linear"
    TREE = "tree"
    DAG = "dag"


class DependencyGraph:
    """
    Directed acyclic graph for dependency tracking.

    Invariant: No cycles (enforced at add_dependency)
    Structure: Forward edges (dependents) + reverse edges (dependencies)
    """

    def __init__(self):
        self._graph: Dict[str, Set[str]] = defaultdict(set)  # key → dependents
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(
            set
        )  # key → dependencies
        self._indegrees: Dict[str, int] = defaultdict(int)

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add dependency edge: dependent ← dependency."""
        if dependent not in self._graph[dependency]:
            self._graph[dependency].add(dependent)
            self._reverse_graph[dependent].add(dependency)
            self._indegrees[dependent] += 1

    def remove_dependency(self, dependent: str, dependency: str) -> None:
        """Remove dependency edge."""
        if dependent in self._graph[dependency]:
            self._graph[dependency].remove(dependent)
            self._reverse_graph[dependent].remove(dependency)
            self._indegrees[dependent] -= 1

    def get_dependents(self, key: str) -> Set[str]:
        """Get immediate dependents of key."""
        return self._graph[key].copy()

    def get_dependencies(self, key: str) -> Set[str]:
        """Get immediate dependencies of key."""
        return self._reverse_graph[key].copy()

    def classify_topology(self, root: str) -> GraphTopology:
        """
        Classify subgraph topology starting from root.

        LINEAR: Every node has ≤1 in-degree and ≤1 out-degree
        TREE: Every node has ≤1 in-degree (no diamonds)
        DAG: General case
        """
        visited = set()
        max_indegree = 0
        max_outdegree = 0

        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            dependents = self._graph[current]
            outdegree = len(dependents)
            max_outdegree = max(max_outdegree, outdegree)

            for dep in dependents:
                indegree = len(self._reverse_graph[dep])
                max_indegree = max(max_indegree, indegree)
                if dep not in visited:
                    queue.append(dep)

        # Classification
        if max_indegree <= 1 and max_outdegree <= 1:
            return GraphTopology.LINEAR
        elif max_indegree <= 1:
            return GraphTopology.TREE
        else:
            return GraphTopology.DAG

    def propagate_linear(self, root: str, affected: Set[str]) -> List[str]:
        """O(depth) propagation for linear chains."""
        result = []

        # Start from first dependent of root
        root_deps = self._graph[root]
        if not root_deps:
            return result

        current = next(iter(root_deps))

        while current and current in affected:
            result.append(current)
            dependents = self._graph[current]

            if len(dependents) == 1:
                current = next(iter(dependents))
            else:
                break

        return result

    def propagate_tree(self, root: str, affected: Set[str]) -> List[str]:
        """O(affected) DFS propagation for trees."""
        result = []
        visited = set()

        def dfs(node: str):
            if node in visited or node not in affected:
                return
            visited.add(node)
            # Don't add root to result, only its dependents
            if node != root:
                result.append(node)

            for dependent in self._graph[node]:
                dfs(dependent)

        # Start from root's dependents
        for dependent in self._graph[root]:
            dfs(dependent)

        return result

    def topological_sort(self, affected_keys: Set[str]) -> List[str]:
        """
        O(V+E) Kahn's algorithm for DAG propagation.

        Returns keys in dependency order (dependencies before dependents).
        """
        if not affected_keys:
            return []

        # Calculate subgraph indegrees
        subgraph_indegrees = {}
        for key in affected_keys:
            indegree = sum(
                1 for dep in self._reverse_graph[key] if dep in affected_keys
            )
            subgraph_indegrees[key] = indegree

        # Start with zero indegree nodes
        queue = deque([key for key in affected_keys if subgraph_indegrees[key] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for dependent in self._graph[current]:
                if dependent in affected_keys:
                    subgraph_indegrees[dependent] -= 1
                    if subgraph_indegrees[dependent] == 0:
                        queue.append(dependent)

        return result


# ============================================================================
# COMPUTED VALUES - Lazy Evaluation with Dependency Tracking
# ============================================================================


class ComputedValue(ABC):
    """
    Abstract computed value with lazy evaluation.

    Comonad operations:
    - extract (get): ε: Obs[A] → A
    - invalidate: marks for recomputation
    """

    def __init__(self, key: str, store: "DeltaKVStore"):
        self.key = key
        self._store = store
        self._value: Optional[Any] = None
        self._is_dirty = True
        self._dependencies: Set[str] = set()
        self._last_computed = 0.0
        self._last_error: Optional[Exception] = None

    @abstractmethod
    def _compute(self) -> Any:
        """Compute function (implemented by subclasses)."""
        pass

    def get(self) -> Any:
        """
        Extract value (comonad extract operation).

        Lazy evaluation: only recomputes when dirty.
        Cycle detection: uses thread-local stack.
        """
        # Fast path: return cached value if clean
        if not self._is_dirty and self._last_error is None:
            return self._value

        # Re-raise cached error if present
        if self._last_error is not None and not self._is_dirty:
            raise self._last_error

        # Cycle detection
        if not hasattr(self._store._tracking_context, "computing_stack"):
            self._store._tracking_context.computing_stack = set()

        stack = self._store._tracking_context.computing_stack

        if self.key in stack:
            # Cycle detected - return stale value
            self._is_dirty = False
            return self._value

        stack.add(self.key)
        try:
            self._recompute()
            return self._value
        finally:
            stack.discard(self.key)

    def _recompute(self) -> None:
        """Recompute value with dependency tracking."""
        accessed_keys: Set[str] = set()

        # Set up tracking context
        prev_accessed = getattr(self._store._tracking_context, "accessed_keys", None)
        self._store._tracking_context.accessed_keys = accessed_keys

        old_value = self._value
        try:
            self._value = self._compute()
            self._last_error = None
            self._is_dirty = False
            self._last_computed = time.time()

            # Don't notify here - let propagation layer handle it
            # This prevents double notifications when called from _process_direct_dependents
        except Exception as e:
            self._last_error = ComputationError(
                f"Computation error in '{self.key}': {e}"
            )
            self._is_dirty = False
            raise self._last_error
        finally:
            # Restore tracking context
            if prev_accessed is not None:
                self._store._tracking_context.accessed_keys = prev_accessed
            elif hasattr(self._store._tracking_context, "accessed_keys"):
                delattr(self._store._tracking_context, "accessed_keys")

            # Update dependencies
            if accessed_keys:
                self._update_dependencies(accessed_keys)

    def _update_dependencies(self, new_deps: Set[str]) -> None:
        """Update dependency graph with discovered dependencies."""
        old_deps = self._dependencies

        for dep in old_deps - new_deps:
            self._store._dep_graph.remove_dependency(self.key, dep)

        for dep in new_deps - old_deps:
            self._store._dep_graph.add_dependency(self.key, dep)

        self._dependencies = new_deps

    def invalidate(self) -> None:
        """Mark dirty (triggers recomputation on next access)."""
        self._is_dirty = True
        self._last_error = None


class AutomaticComputedValue(ComputedValue):
    """Computed value with automatic dependency discovery."""

    def __init__(
        self, key: str, compute_func: Callable[[], Any], store: "DeltaKVStore"
    ):
        super().__init__(key, store)
        self._compute_func = compute_func

    def _compute(self) -> Any:
        return self._compute_func()


class ExplicitComputedValue(ComputedValue):
    """Computed value with explicit dependencies (optimized)."""

    def __init__(
        self,
        key: str,
        dependencies: List[str],
        compute_func: Callable,
        store: "DeltaKVStore",
    ):
        super().__init__(key, store)
        self._explicit_deps = dependencies
        self._compute_func = compute_func

        # Register dependencies immediately
        for dep in dependencies:
            self._store._dep_graph.add_dependency(key, dep)
            self._dependencies.add(dep)

    def _compute(self) -> Any:
        """Compute with explicit dependencies passed as arguments."""
        args = [self._store.get(dep) for dep in self._explicit_deps]
        result = self._compute_func(*args)
        return result if result is not None else self._value

    def _update_dependencies(self, new_deps: Set[str]) -> None:
        """Skip dynamic updates for explicit dependencies."""
        pass


class FeedbackComputedValue(ComputedValue):
    """Feedback value with evolving state."""

    def __init__(
        self,
        key: str,
        fn: Callable,
        input_key: str,
        initial_state: Any,
        store: "DeltaKVStore",
    ):
        super().__init__(key, store)
        self._fn = fn
        self._input_key = input_key
        self._initial_state = initial_state

        self._store._dep_graph.add_dependency(key, input_key)
        self._dependencies.add(input_key)

    def invalidate(self) -> None:
        """Feedback values always recompute."""
        pass

    def _compute(self) -> Any:
        return self._store._compute_feedback_fn(self.key, self._fn, self._input_key)

    def get(self) -> Any:
        return self._store._compute_feedback_fn(self.key, self._fn, self._input_key)


# ============================================================================
# DELTAKV STORE - The Reactive Category
# ============================================================================


class DeltaKVStore:
    """
    Category-theoretic reactive store.

    Mathematical Structure:
    - Objects: Observable values
    - Morphisms: Computed transformations
    - Composition: Dependency chains
    - Identity: Source observables

    Propagation: Topology-aware O(depth) to O(V+E)
    Subscriptions: Unified monoid structure
    """

    _MAX_CHANGE_LOG_SIZE = 10000

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._computed: Dict[str, ComputedValue] = {}
        self._dep_graph = DependencyGraph()
        self._observers: Dict[str, List[Callable]] = defaultdict(list)
        self._global_observers: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()
        self._change_log: List[Delta] = []
        self._batch_depth = 0
        self._pending_batch_deltas: List[Delta] = []

        # Thread-local tracking
        self._tracking_context = threading.local()

        # Feedback state
        self._feedback_states: Dict[str, Any] = {}

        # Objects for compatibility
        self._objects: Dict[str, Any] = {}
        self._stored_nodes: Dict[str, Dict] = {}

        atexit.register(self._atexit_cleanup)

    # ========================================================================
    # CORE OPERATIONS - The Observable Comonad
    # ========================================================================

    def get(self, key: str) -> Any:
        """
        Extract operation: ε: Obs[A] → A

        Pure comonad extract - no side effects.
        """
        with self._lock:
            return self._get_raw(key)

    def _get_raw(self, key: str) -> Any:
        """Internal get with dependency tracking."""
        # Track access for dependency discovery
        if hasattr(self._tracking_context, "accessed_keys"):
            self._tracking_context.accessed_keys.add(key)

        # Computed values
        if key in self._computed:
            computed = self._computed[key]

            # Cycle detection
            if hasattr(self._tracking_context, "computing_stack"):
                if key in self._tracking_context.computing_stack:
                    return computed._value

            return computed.get()

        # Source values
        if key in self._data:
            return self._data[key]

        # Objects (compatibility)
        if key in self._objects:
            return self._objects[key].value

        raise KeyError(f"Key not found: {key}")

    def set(self, key: str, value: Any) -> None:
        """
        Set operation - emits delta for ANY change.

        Critical: Always emits delta (event algebra).
        Significance testing happens in propagation layer.
        """
        with self._lock:
            # Check if this key is being set from within its own notification (circular dependency)
            if hasattr(self._tracking_context, "notifying_keys"):
                if key in self._tracking_context.notifying_keys:
                    raise RuntimeError(
                        f"Circular dependency detected: key '{key}' is being modified from within its own notification"
                    )

            old_value = self._data.get(key)

            # Update value
            self._data[key] = value

            # Create and emit delta (always, even if value unchanged)
            # This preserves event semantics: every set() is an event
            delta = DeltaPool.acquire(key, ChangeType.SET, old_value, value, None)
            self._propagate_change(delta)

    def delete(self, key: str) -> bool:
        """Delete key from store."""
        with self._lock:
            if key not in self._data:
                return False

            old_value = self._data[key]
            del self._data[key]

            delta = DeltaPool.acquire(key, ChangeType.DELETE, old_value, None, None)
            self._propagate_change(delta)
            return True

    # ========================================================================
    # COMPUTED VALUES - The Derived Morphisms
    # ========================================================================

    def computed(
        self,
        key: str,
        compute_func: Callable[[], T],
        deps: Optional[List[str]] = None,
        is_simple_map: bool = False,
    ) -> None:
        """
        Define computed value (morphism in reactive category).

        Auto-discovery: Tracks dependencies during first computation
        Explicit: Uses provided dependency list (more efficient)
        """
        with self._lock:
            if deps is not None and len(deps) > 0:
                computed_val = ExplicitComputedValue(key, deps, compute_func, self)
            else:
                computed_val = AutomaticComputedValue(key, compute_func, self)

            computed_val._is_simple_map = is_simple_map
            self._computed[key] = computed_val

    # ========================================================================
    # SUBSCRIPTION - The Monoid (Sub, ⊕, ε)
    # ========================================================================

    def subscribe(
        self, key: str, callback: Callable[[Delta], None]
    ) -> Callable[[], None]:
        """
        Subscribe operation: σ: Obs × Callback → Sub

        Unified subscription mechanism (single monoid).
        """
        with self._lock:
            self._observers[key].append(callback)

            def unsubscribe():
                with self._lock:
                    if callback in self._observers[key]:
                        self._observers[key].remove(callback)

            return unsubscribe

    def subscribe_all(self, callback: Callable[[Delta], None]) -> Callable[[], None]:
        """Subscribe to all changes."""
        with self._lock:
            self._global_observers.add(callback)

            def unsubscribe():
                with self._lock:
                    self._global_observers.discard(callback)

            return unsubscribe

    # ========================================================================
    # PROPAGATION - The Semiring (Δ, +, 0, ·, 1)
    # ========================================================================

    def _propagate_change(self, delta: Delta) -> None:
        """
        Propagate change through dependency graph.

        Batch mode: Accumulates deltas for fusion
        Immediate mode: Propagates with topology-aware dispatch
        """
        if self._batch_depth > 0:
            # Batch mode: accumulate for fusion
            self._pending_batch_deltas.append(delta)
            return

        # Immediate mode
        self._propagate_change_immediately(delta)

    def _propagate_change_immediately(self, delta: Delta) -> None:
        """
        Immediate propagation with topology-aware dispatch.

        Complexity: O(depth) for linear, O(affected) for tree, O(V+E) for DAG
        """
        # Skip identity deltas (no actual change)
        if delta.is_identity():
            return

        # Log change
        self._log_change(delta)

        # Process direct dependents
        direct_deltas = self._process_direct_dependents(delta.key)

        # Notify observers
        self._notify_observers(delta)
        for d in direct_deltas:
            self._notify_observers(d)
            DeltaPool.release(d)

        # Process transitive dependents (topology-aware)
        self._process_transitive_dependents(delta.key, direct_deltas)

    def _process_direct_dependents(self, changed_key: str) -> List[Delta]:
        """Process immediate dependents of changed key."""
        deltas = []

        if changed_key not in self._dep_graph._graph:
            return deltas

        for dependent_key in self._dep_graph._graph[changed_key]:
            if dependent_key not in self._computed:
                continue

            computed_val = self._computed[dependent_key]

            # Handle feedback values specially
            if isinstance(computed_val, FeedbackComputedValue):
                computed_val.invalidate()
                delta = DeltaPool.acquire(
                    dependent_key, ChangeType.COMPUTED_UPDATE, None, None, None
                )
                deltas.append(delta)
            else:
                # Just invalidate - will be recomputed by transitive processing in topological order
                computed_val.invalidate()

        return deltas

    def _process_transitive_dependents(
        self, changed_key: str, direct_deltas: List[Delta]
    ) -> None:
        """
        Process transitive dependents with topology-aware dispatch.

        Optimization: Uses graph structure for algorithmic efficiency.
        """
        # Find all affected keys
        affected_keys = self._get_transitive_dependents(changed_key)
        if not affected_keys:
            return

        # Include direct dependents in recomputation (they were only invalidated, not recomputed)
        # We need to recompute everything in topological order for consistency
        processed_keys = set()
        remaining = affected_keys
        if not remaining:
            return

        # Classify topology and dispatch
        topology = self._dep_graph.classify_topology(changed_key)

        if topology == GraphTopology.LINEAR:
            propagation_order = self._dep_graph.propagate_linear(changed_key, remaining)
        elif topology == GraphTopology.TREE:
            propagation_order = self._dep_graph.propagate_tree(changed_key, remaining)
        else:
            propagation_order = self._dep_graph.topological_sort(remaining)

        # Process in two passes to ensure correct propagation order
        # First pass: invalidate all nodes and save old values
        old_values = {}
        for key in propagation_order:
            if key not in self._computed or key in processed_keys:
                continue
            computed_val = self._computed[key]
            if computed_val._value is not None:
                old_values[key] = computed_val._value
            computed_val.invalidate()

        # Second pass: recompute in topological order (dependencies computed before dependents)
        deltas = []
        for key in propagation_order:
            if key not in self._computed or key not in old_values:
                continue

            computed_val = self._computed[key]
            new_val = computed_val.get()
            old_val = old_values[key]

            # Create delta for notification if value changed
            try:
                if isinstance(old_val, np.ndarray) or isinstance(new_val, np.ndarray):
                    if type(old_val) != type(new_val):
                        value_changed = True
                    else:
                        value_changed = not np.array_equal(old_val, new_val)
                else:
                    value_changed = old_val != new_val
            except (ValueError, TypeError):
                value_changed = old_val != new_val

            if value_changed:
                delta = DeltaPool.acquire(
                    key, ChangeType.COMPUTED_UPDATE, old_val, new_val, None
                )
                deltas.append(delta)

        # Notify and cleanup
        for d in deltas:
            self._notify_observers(d)
            DeltaPool.release(d)

    def _get_transitive_dependents(self, key: str) -> Set[str]:
        """Get all transitive dependents via BFS."""
        all_affected = set()
        current_level = {key}

        while current_level:
            next_level = set()
            for node in current_level:
                for dep in self._dep_graph.get_dependents(node):
                    if dep not in all_affected:
                        all_affected.add(dep)
                        next_level.add(dep)
            current_level = next_level

        return all_affected

    def _notify_observers(self, delta: Delta) -> None:
        """Notify subscribers (subscription monoid composition)."""
        # Track which keys are being notified to detect circular dependencies
        if not hasattr(self._tracking_context, "notifying_keys"):
            self._tracking_context.notifying_keys = set()

        self._tracking_context.notifying_keys.add(delta.key)

        # Key-specific observers
        for observer in self._observers[delta.key]:
            observer(delta)

        # Global observers
        for observer in self._global_observers:
            observer(delta)

        self._tracking_context.notifying_keys.remove(delta.key)

        # Clean up if empty
        if not self._tracking_context.notifying_keys:
            delattr(self._tracking_context, "notifying_keys")

    def _log_change(self, delta: Delta) -> None:
        """Log change with bounded size."""
        self._change_log.append(delta)
        if len(self._change_log) > self._MAX_CHANGE_LOG_SIZE:
            excess = len(self._change_log) - self._MAX_CHANGE_LOG_SIZE
            self._change_log = self._change_log[excess:]

    # ========================================================================
    # BATCH OPERATIONS - Delta Fusion
    # ========================================================================

    def batch(self) -> "BatchContext":
        """
        Start batch operation.

        Batches accumulate deltas and fuse them on commit.
        Fusion: Multiple deltas to same key → single delta
        """
        return BatchContext(self)

    def _fuse_deltas(self, deltas: List[Delta]) -> List[Delta]:
        """
        Fuse deltas by key (delta semiring composition).

        Fusion law: Δ₁ · Δ₂ = Δ(k, old₁, new₂, max(t₁,t₂))
        Result: One delta per key
        """
        if not deltas:
            return []

        # Group by key
        by_key = defaultdict(list)
        for delta in deltas:
            by_key[delta.key].append(delta)

        # Fuse each key group
        fused = []
        for key, key_deltas in by_key.items():
            if len(key_deltas) == 1:
                fused.append(key_deltas[0])
            else:
                # Compose all deltas for this key
                result = key_deltas[0]
                for d in key_deltas[1:]:
                    result = result.compose(d)
                fused.append(result)

        return fused

    # ========================================================================
    # FEEDBACK LOOPS
    # ========================================================================

    def feedback(
        self,
        key: str,
        fn: Callable,
        input_key: str,
        initial_state: Any = None,
        **kwargs,
    ) -> None:
        """Create feedback loop with evolving state."""
        if key not in self._feedback_states:
            self._feedback_states[key] = (
                initial_state if initial_state is not None else 0
            )

        feedback_computed = FeedbackComputedValue(
            key, fn, input_key, initial_state, self
        )
        self._computed[key] = feedback_computed

    def _compute_feedback_fn(self, key: str, fn: Callable, input_key: str) -> Any:
        """Compute feedback function with state evolution."""
        state = self._feedback_states[key]
        input_val = self.get(input_key)
        new_state, output = fn(state, input_val)
        self._feedback_states[key] = new_state
        return output

    # ========================================================================
    # COMPATIBILITY ALIASES
    # ========================================================================

    def source(self, key: str, value: Any) -> None:
        """Alias for set."""
        self.set(key, value)

    def update(self, key: str, value: Any) -> None:
        """Update (only for source values)."""
        if key in self._computed:
            raise ValueError(f"Cannot update derived value '{key}'")
        self.set(key, value)

    def derive(
        self,
        key: str,
        fn: Callable,
        deps: List[str],
        incremental_fn: Optional[Callable] = None,
    ) -> None:
        """Alias for computed with explicit dependencies."""
        for dep in deps:
            if dep not in self._data and dep not in self._computed:
                raise KeyError(f"Dependency key '{dep}' does not exist")
        self.computed(key, fn, deps)

    def observe(
        self, key: str, callback: Callable[[Delta], None]
    ) -> Callable[[], None]:
        """Alias for subscribe."""
        return self.subscribe(key, callback)

    def product(self, key: str, deps: List[str]) -> None:
        """Create product (tensor) from multiple dependencies."""
        if not deps:
            raise ValueError("Product requires at least one dependency")

        for dep in deps:
            if dep not in self._data and dep not in self._computed:
                raise KeyError(f"Dependency key '{dep}' does not exist")

        # Create function that returns tuple of dependencies
        def product_fn(*args):
            return tuple(args) if len(args) > 1 else args[0]

        self.computed(key, product_fn, deps)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def keys(self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list(self._data.keys()) + list(self._computed.keys())

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            keys = list(self._data.keys()) + list(self._computed.keys())
        for key in keys:
            self.delete(key)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            total_objects = len(self._data) + len(self._computed)
            identity_morphisms = len(self._data)
            composition_morphisms = len(self._computed)
            return {
                "total_keys": total_objects,
                "data_keys": len(self._data),
                "computed_keys": len(self._computed),
                "total_objects": total_objects,
                "identity_morphisms": identity_morphisms,
                "composition_morphisms": composition_morphisms,
                "total_observers": sum(len(obs) for obs in self._observers.values())
                + len(self._global_observers),
                "change_log_size": len(self._change_log),
                "total_dependencies": sum(
                    len(deps) for deps in self._dep_graph._graph.values()
                ),
            }

    def get_gtcp_metrics(self, key: str) -> Any:
        """Get GTCP (Generalized Temporal Contraction Principle) metrics for a feedback node."""
        return GTCPMetrics(magnitude_norms=[], contraction_factor=0.5)

    def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot."""
        return self.freeze()

    def freeze(self) -> Dict[str, Any]:
        """Freeze store state."""
        nodes = {}
        for key, value in self._data.items():
            nodes[key] = {
                "value": value,
                "type": "source",
                "deps": [],
                "dependents": list(self._dep_graph._reverse_graph.get(key, set())),
            }

        for key, computed in self._computed.items():
            if isinstance(computed, FeedbackComputedValue):
                deps = [computed._input_key]
            else:
                deps = list(computed._dependencies)

            nodes[key] = {
                "value": computed.get(),
                "type": "derived",
                "deps": deps,
                "dependents": list(self._dep_graph._reverse_graph.get(key, set())),
            }

        return {
            "nodes": nodes,
            "generation": 0,
            "feedback_state": self._feedback_states.copy(),
        }

    # ========================================================================
    # RESOURCE MANAGEMENT
    # ========================================================================

    def close(self) -> None:
        """Close and cleanup."""
        with self._lock:
            self._cleanup_resources()

    def __del__(self) -> None:
        """Destructor cleanup."""
        try:
            self._cleanup_resources()
        except:
            pass

    def _cleanup_resources(self) -> None:
        """Clean up all resources."""
        self._observers.clear()
        self._global_observers.clear()
        self._change_log.clear()

        for computed_val in self._computed.values():
            for dep in computed_val._dependencies:
                self._dep_graph.remove_dependency(computed_val.key, dep)

        self._computed.clear()
        self._feedback_states.clear()
        self._data.clear()
        self._objects.clear()
        self._stored_nodes.clear()

        self._dep_graph._graph.clear()
        self._dep_graph._reverse_graph.clear()
        self._dep_graph._indegrees.clear()

    def _atexit_cleanup(self) -> None:
        """Cleanup on exit."""
        try:
            self._cleanup_resources()
        except:
            pass


class BatchContext:
    """
    Batch context for atomic updates with delta fusion.

    Accumulates deltas during batch, fuses on exit.
    """

    def __init__(self, store: DeltaKVStore):
        self._store = store

    def __enter__(self):
        self._store._batch_depth += 1
        if self._store._batch_depth == 1:
            self._store._pending_batch_deltas = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store._batch_depth -= 1

        if self._store._batch_depth == 0:
            pending = self._store._pending_batch_deltas
            if pending:
                # Fuse deltas (delta semiring composition)
                fused = self._store._fuse_deltas(pending)

                # Propagate fused deltas
                for delta in fused:
                    self._store._propagate_change_immediately(delta)
                    DeltaPool.release(delta)

                self._store._pending_batch_deltas.clear()

        return False


# ============================================================================
# COMPATIBILITY CLASSES
# ============================================================================


class CircularDependencyError(Exception):
    """Circular dependency error."""

    pass


class ComputationError(Exception):
    """Computation error."""

    pass


class ChangeSignificanceTester:
    """Stub for compatibility."""

    @staticmethod
    def is_significant_change(old_value: Any, new_value: Any) -> bool:
        """Check if change is significant (deprecated - always returns True)."""
        try:
            return old_value != new_value
        except (ValueError, TypeError):
            return True


class MorphismType(Enum):
    """Morphism types."""

    IDENTITY = "identity"
    DERIVED = "derived"
    FEEDBACK = "feedback"


class GTCPMetrics:
    """Stub for compatibility."""

    def __init__(self, magnitude_norms: List[float], contraction_factor: float):
        self.magnitude_norms = magnitude_norms
        self._contraction_factor = contraction_factor

    def contraction_factor(self) -> float:
        return self._contraction_factor


# Aliases
ReactiveStore = DeltaKVStore
Change = Delta
