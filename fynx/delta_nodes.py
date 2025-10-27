"""
Node Abstraction Layer for FynX
================================

Provides the three-state node system (Virtual, Tracked, Materialized) as a
frontend for the DeltaKVStore. This layer has no knowledge of observables and
operates purely in terms of computation graphs and state transitions.

Three Node States:
- VirtualNode: Pure computation, no storage, function fusion
- TrackedNode: Registered in store for change propagation
- MaterializedNode: Tracked + cached result for shared computation

State Transitions:
- Virtual → Tracked: When change propagation is needed
- Tracked → Materialized: When fan-out ≥ 2 (multiple consumers)
"""

import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Tuple


class NodeState:
    """Enumeration of node states."""

    VIRTUAL = "virtual"
    TRACKED = "tracked"
    MATERIALIZED = "materialized"


@dataclass
class NodeMetrics:
    """Metrics for a node's usage patterns."""

    access_count: int = 0
    consumer_count: int = 0
    last_access_time: float = 0.0
    computation_cost: float = 0.0


class Node(ABC):
    """
    Abstract base class for computation graph nodes.

    Nodes represent values in the reactive system and can be in one of three states:
    - Virtual: Pure computation, no storage
    - Tracked: Registered for change propagation
    - Materialized: Cached result for sharing
    """

    def __init__(self, node_id: str, store):
        self.node_id = node_id
        self._store = store
        self._state = NodeState.VIRTUAL
        self._consumers: Set[str] = set()
        self._metrics = NodeMetrics()
        self._lock = threading.RLock()

    @abstractmethod
    def read(self) -> Any:
        """Read the node's value."""
        pass

    @abstractmethod
    def write(self, value: Any) -> None:
        """Write a value to the node (only valid for source nodes)."""
        pass

    @abstractmethod
    def invalidate(self) -> None:
        """Mark node as needing recomputation."""
        pass

    def get_state(self) -> str:
        """Get the current state of the node."""
        return self._state

    def add_consumer(self, consumer_id: str) -> None:
        """
        Register a consumer of this node's value.

        Triggers materialization when consumer count reaches 2.
        """
        with self._lock:
            if consumer_id not in self._consumers:
                self._consumers.add(consumer_id)
                self._metrics.consumer_count = len(self._consumers)

                # Materialization threshold: 2+ consumers
                if (
                    self._metrics.consumer_count >= 2
                    and self._state != NodeState.MATERIALIZED
                ):
                    self._transition_to_materialized()

    def remove_consumer(self, consumer_id: str) -> None:
        """Remove a consumer registration."""
        with self._lock:
            self._consumers.discard(consumer_id)
            self._metrics.consumer_count = len(self._consumers)

    def _transition_to_tracked(self) -> None:
        """Transition from Virtual to Tracked state."""
        if self._state == NodeState.VIRTUAL:
            self._state = NodeState.TRACKED
            self._register_in_store()
            # Clear direct dependents when transitioning to tracked
            # Store-based propagation will handle notifications now
            if hasattr(self, "_direct_dependents"):
                self._direct_dependents.clear()

    def _transition_to_materialized(self) -> None:
        """Transition to Materialized state (ensures Tracked first)."""
        if self._state == NodeState.VIRTUAL:
            self._transition_to_tracked()
        if self._state == NodeState.TRACKED:
            self._state = NodeState.MATERIALIZED
            self._materialize_in_store()
            # Clear direct dependents when materializing
            if hasattr(self, "_direct_dependents"):
                self._direct_dependents.clear()

    @abstractmethod
    def _register_in_store(self) -> None:
        """Register this node in the KV store for tracking."""
        pass

    @abstractmethod
    def _materialize_in_store(self) -> None:
        """Materialize this node's value in the KV store."""
        pass


class VirtualNode(Node):
    """
    Virtual node: pure computation with no storage.

    Represents a fused chain of computations that execute on-demand.
    No intermediate storage, optimal for linear chains.
    """

    def __init__(self, node_id: str, compute_func: Callable[[], Any], store):
        super().__init__(node_id, store)
        self._compute_func = compute_func
        self._dependencies: List[str] = []
        self._direct_dependents: List[weakref.ref] = (
            []
        )  # Weak references for O(1) propagation
        self._cached_value: Optional[Any] = None

    def read(self) -> Any:
        """Compute value on-demand by executing the fused function."""
        with self._lock:
            self._metrics.access_count += 1
            # Use cached value if available to avoid recomputation
            if self._cached_value is not None:
                return self._cached_value
            self._cached_value = self._compute_func()
            return self._cached_value

    def write(self, value: Any) -> None:
        """Virtual nodes are read-only."""
        raise ValueError(f"Cannot write to virtual node {self.node_id}")

    def invalidate(self) -> None:
        """Invalidate cached value and notify direct dependents."""
        with self._lock:
            self._cached_value = None
            self.notify_dependents()

    def notify_dependents(self) -> None:
        """
        Fast path: O(1) direct notification without store overhead.

        Notifies all direct dependents via weak references. This avoids
        the DeltaKVStore overhead for linear chains staying in virtual mode.
        """
        valid_refs = []
        for dep_ref in self._direct_dependents:
            dep = dep_ref()  # Unwrap weakref
            if dep is not None:
                valid_refs.append(dep_ref)
                if hasattr(dep, "_on_node_invalidated"):
                    dep._on_node_invalidated()
        # Clean up dead references
        self._direct_dependents = valid_refs

    def fuse_with(self, transform: Callable[[Any], Any]) -> "VirtualNode":
        """
        Fuse this virtual node with another transformation.

        Creates a new virtual node with a composed function:
        new_func = transform ∘ self._compute_func
        """

        def fused_func():
            return transform(self._compute_func())

        new_id = f"{self.node_id}_fused"
        return VirtualNode(new_id, fused_func, self._store)

    def _register_in_store(self) -> None:
        """Register as a computed value in the store."""
        # Convert to tracked node by registering computation
        self._store.computed(self.node_id, self._compute_func, deps=self._dependencies)

    def _materialize_in_store(self) -> None:
        """
        Materialize by ensuring we're registered and will cache results.

        The store's computed values already cache, so this is about
        ensuring the value is eagerly computed and stored.
        """
        if self.node_id not in self._store._computed:
            self._register_in_store()
        # Force computation to populate cache
        self._store.get(self.node_id)


class TrackedNode(Node):
    """
    Tracked node: registered in store for change propagation.

    Participates in the reactive graph but doesn't necessarily cache its value.
    Used when change propagation is needed but sharing hasn't occurred yet.
    """

    def __init__(
        self,
        node_id: str,
        compute_func: Callable[[], Any],
        dependencies: List[str],
        store,
    ):
        super().__init__(node_id, store)
        self._state = NodeState.TRACKED
        self._compute_func = compute_func
        self._dependencies = dependencies

        # Register immediately
        self._register_in_store()

    def read(self) -> Any:
        """Read from store (may trigger computation)."""
        with self._lock:
            self._metrics.access_count += 1
            return self._store.get(self.node_id)

    def write(self, value: Any) -> None:
        """Tracked computed nodes are read-only."""
        raise ValueError(f"Cannot write to tracked computed node {self.node_id}")

    def invalidate(self) -> None:
        """Invalidate the cached computation."""
        if self.node_id in self._store._computed:
            self._store._computed[self.node_id].invalidate()

    def _register_in_store(self) -> None:
        """Register as a computed value in the store."""
        self._store.computed(self.node_id, self._compute_func, deps=self._dependencies)

    def _materialize_in_store(self) -> None:
        """
        Materialize by forcing computation and caching.

        For tracked nodes, materialization means ensuring the value is
        eagerly computed and cached for fast lookup.
        """
        # Force computation to populate cache
        self._store.get(self.node_id)


class MaterializedNode(Node):
    """
    Materialized node: cached value for shared computation.

    Stores computed results for efficient access by multiple consumers.
    Materialization happens automatically when fan-out ≥ 2.
    """

    def __init__(
        self,
        node_id: str,
        compute_func: Callable[[], Any],
        dependencies: List[str],
        store,
    ):
        super().__init__(node_id, store)
        self._state = NodeState.MATERIALIZED
        self._compute_func = compute_func
        self._dependencies = dependencies

        # Register and materialize immediately
        self._register_in_store()
        self._materialize_in_store()

    def read(self) -> Any:
        """Fast O(1) lookup from cached value."""
        with self._lock:
            self._metrics.access_count += 1
            return self._store.get(self.node_id)

    def write(self, value: Any) -> None:
        """Materialized computed nodes are read-only."""
        raise ValueError(f"Cannot write to materialized computed node {self.node_id}")

    def invalidate(self) -> None:
        """Invalidate the cached value."""
        if self.node_id in self._store._computed:
            self._store._computed[self.node_id].invalidate()

    def _register_in_store(self) -> None:
        """Register as a computed value in the store."""
        self._store.computed(self.node_id, self._compute_func, deps=self._dependencies)

    def _materialize_in_store(self) -> None:
        """Ensure value is computed and cached."""
        # Force immediate computation
        self._store.get(self.node_id)


class SourceNode(Node):
    """
    Source node: stores a mutable value.

    Source nodes are always tracked (registered in the store) since they
    need to propagate changes to dependents.
    """

    def __init__(self, node_id: str, initial_value: Any, store):
        super().__init__(node_id, store)
        self._state = NodeState.TRACKED

        # Register immediately with initial value
        self._store.set(node_id, initial_value)

    def read(self) -> Any:
        """Read the stored value."""
        with self._lock:
            self._metrics.access_count += 1
            return self._store.get(self.node_id)

    def write(self, value: Any) -> None:
        """Update the stored value and propagate changes."""
        with self._lock:
            self._store.set(self.node_id, value)

    def invalidate(self) -> None:
        """Source nodes don't need invalidation (they store directly)."""
        pass

    def _register_in_store(self) -> None:
        """Source nodes are registered via set()."""
        pass

    def _materialize_in_store(self) -> None:
        """Source nodes are always materialized (they store values)."""
        pass


class NodeFactory:
    """
    Factory for creating nodes with automatic state management.

    Handles the logic for determining which node type to create based on
    usage patterns and dependencies.
    """

    def __init__(self, store):
        self._store = store
        self._node_registry: dict[str, Node] = {}
        self._lock = threading.RLock()

    def create_source(self, node_id: str, initial_value: Any) -> SourceNode:
        """Create a source node (always tracked)."""
        with self._lock:
            if node_id in self._node_registry:
                raise ValueError(f"Node {node_id} already exists")

            node = SourceNode(node_id, initial_value, self._store)
            self._node_registry[node_id] = node
            return node

    def create_computed(
        self,
        node_id: str,
        compute_func: Callable[[], Any],
        dependencies: Optional[List[str]] = None,
        force_materialize: bool = False,
    ) -> Node:
        """
        Create a computed node with automatic state selection.

        Args:
            node_id: Unique identifier for the node
            compute_func: Function to compute the node's value
            dependencies: Explicit dependencies (if known)
            force_materialize: Force materialization regardless of fan-out

        Returns:
            Node in the appropriate state (Virtual, Tracked, or Materialized)
        """
        with self._lock:
            if node_id in self._node_registry:
                raise ValueError(f"Node {node_id} already exists")

            deps = dependencies or []

            # Start virtual by default (most efficient for linear chains)
            if force_materialize:
                node = MaterializedNode(node_id, compute_func, deps, self._store)
            elif not deps:
                # No dependencies known - start virtual for maximum fusion
                node = VirtualNode(node_id, compute_func, self._store)
            else:
                # Has dependencies but no consumers yet - start virtual
                node = VirtualNode(node_id, compute_func, self._store)

            self._node_registry[node_id] = node
            return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get an existing node by ID."""
        return self._node_registry.get(node_id)

    def transition_to_tracked(self, node_id: str) -> None:
        """
        Transition a node to tracked state.

        Called when change propagation is needed (e.g., subscription added).
        """
        with self._lock:
            node = self._node_registry.get(node_id)
            if node and node.get_state() == NodeState.VIRTUAL:
                node._transition_to_tracked()

    def register_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """
        Register a dependency relationship and trigger materialization if needed.

        This is where fan-out detection happens: when a node gains a second
        consumer, it automatically materializes.
        """
        with self._lock:
            dependency = self._node_registry.get(dependency_id)
            if dependency:
                dependency.add_consumer(dependent_id)

    def stats(self) -> dict:
        """Get statistics about node states."""
        with self._lock:
            stats = {
                NodeState.VIRTUAL: 0,
                NodeState.TRACKED: 0,
                NodeState.MATERIALIZED: 0,
            }

            for node in self._node_registry.values():
                stats[node.get_state()] += 1

            return {
                "total_nodes": len(self._node_registry),
                "by_state": stats,
                "source_nodes": sum(
                    1 for n in self._node_registry.values() if isinstance(n, SourceNode)
                ),
            }


# Example usage demonstrating the three states:
if __name__ == "__main__":
    from delta_kv_store import DeltaKVStore

    store = DeltaKVStore()
    factory = NodeFactory(store)

    # Create source node (always tracked)
    x = factory.create_source("x", 5)
    print(f"x state: {x.get_state()}")  # TRACKED

    # Create virtual computed node (linear chain)
    y = factory.create_computed("y", lambda: x.read() * 2, dependencies=["x"])
    print(f"y state: {y.get_state()}")  # VIRTUAL

    # Add another computed node in the chain (still virtual)
    z = factory.create_computed("z", lambda: y.read() + 3, dependencies=["y"])
    print(f"z state: {z.get_state()}")  # VIRTUAL
    print(f"y state: {y.get_state()}")  # Still VIRTUAL (one consumer)

    # Create branch point - y gains second consumer
    w = factory.create_computed("w", lambda: y.read() - 1, dependencies=["y"])
    factory.register_dependency("w", "y")  # Triggers materialization!

    print(f"y state after branch: {y.get_state()}")  # MATERIALIZED
    print(f"z state: {z.get_state()}")  # Still VIRTUAL
    print(f"w state: {w.get_state()}")  # VIRTUAL

    # Read values
    print(f"z value: {z.read()}")  # 13
    print(f"w value: {w.read()}")  # 9

    print("\nNode statistics:")
    print(factory.stats())
