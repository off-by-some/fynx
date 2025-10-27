"""
Node Abstraction Layer for FynX
================================

This module provides the three-state node system (Virtual, Tracked, Materialized)
as a pure computation graph layer for the DeltaKVStore. The Node layer operates
independently of observables, managing computation graphs, state transitions,
and caching decisions based on structural patterns.

Architecture Overview:
----------------------

The Node layer sits between observables (high-level API) and DeltaKVStore
(low-level storage), providing intelligent caching and state management.
Unlike observables, Nodes operate purely on computation graphs without any
knowledge of the reactive system.

    Observable Layer    (User-facing API with .value, .set, .subscribe)
            ↓
       Node Layer      (Computation graph management, state transitions)
            ↓
    DeltaKVStore       (Change propagation, dependency tracking)

Three Node States:
------------------

1. VirtualNode:
   - Pure computation with no storage
   - Executes fused function chains on-demand
   - Maintains direct dependents for O(1) propagation
   - Caches computed values with dirty tracking
   - Optimal for linear dependency chains

2. TrackedNode:
   - Registered with DeltaKVStore for change propagation
   - No separate cache (uses store's computed value cache)
   - Participates in reactive dependency graph
   - Used when change notifications are needed

3. MaterializedNode:
   - Tracked with eager computation and caching
   - Results are immediately available for fast access
   - Automatic when fan-out ≥ 2 (multiple consumers)
   - Provides shared computation results

State Transitions:
------------------

State transitions happen automatically based on usage patterns:

  Virtual ←→ Tracked: When subscriptions or change propagation is needed
        ↓
  Materialized: When fan-out ≥ 2 (branch point detected)

Virtual → Tracked Transition:
    Triggers when:
    - Subscription added (needs change notifications)
    - Dependency graph requires tracking

    Effects:
    - Node registers with DeltaKVStore
    - Direct dependents list is cleared (store handles propagation)
    - Cache remains valid for fast initial access

Virtual → Materialized Transition:
    Triggers when:
    - Fan-out reaches 2+ consumers (sharing detected)

    Effects:
    - Node becomes tracked if not already
    - Computation result is eagerly cached
    - Direct dependents cleared (store handles updates)
    - Subsequent reads are O(1) lookups

Fan-Out Detection:
------------------

The key insight is that fan-out (multiple consumers) indicates sharing.
When 2+ nodes depend on the same computation, caching becomes beneficial:

    Without caching:  cost = k × computation_cost
    With caching:     cost = computation_cost + k × lookup_cost

For k ≥ 2, caching wins. The threshold of exactly 2 emerges from this
cost model. The Node layer uses consumer counting to detect fan-out:

    consumer_count ≥ 2  ⟹  materialize
    consumer_count < 2  ⟹  stay virtual

Direct Propagation (Virtual Mode):
-----------------------------------

VirtualNodes maintain a list of direct dependents using weak references.
When a virtual node's value changes, it can notify dependents directly
without going through the store:

    x → y → z  (all virtual)

    x.invalidate()
      ↓
    y.invalidate() + y.notify_dependents()
      ↓
    z.invalidate()  (O(1) notification)

This avoids DeltaKVStore overhead for linear chains. The direct
dependents list is automatically cleared when transitioning to
Tracked/Materialized (store handles propagation then).

Cache Management:
-----------------

VirtualNodes own their caching logic:

    read():
        if _is_dirty or _cached_value is None:
            _cached_value = _compute_func()
            _is_dirty = False
        return _cached_value

    invalidate():
        mark_dirty()  # Sets _is_dirty, notifies dependents

The _is_dirty flag tracks cache validity. When a source changes,
invalidate() is called, marking the cache dirty and notifying
direct dependents. The next read() will recompute and cache the
new value.

TrackedNode and MaterializedNode delegate caching to the store's
computed value system, which handles invalidation automatically
through the dependency graph.

Fusion Support:
--------------

VirtualNode.fuse_with() composes computation functions:

    f1: x → intermediate
    f2: intermediate → result

    Fused: x → result  (composed function)

FusedVirtualNode is a specialized VirtualNode that represents
fused computation chains. It behaves identically to VirtualNode
but tracks that it's a fused chain for potential future optimizations.

The NodeFactory can detect fusability by checking:
    - Parent is a VirtualNode
    - Parent has ≤1 consumer (no fan-out yet)
    - Available for fusion

Automatic Cleanup:
-----------------

State transitions automatically clean up resources:

    Virtual → Tracked:
        - Clears _direct_dependents (store handles updates)
        - Cache remains valid for last read

    Virtual → Materialized:
        - Clears _direct_dependents (store handles updates)
        - Eagerly computes and caches value
        - Ensures dependents use cached value

This prevents stale references and ensures efficient propagation.

Example Usage:
-------------

    # Create nodes via factory
    factory = NodeFactory(store)

    # Create source (always tracked)
    x_node = factory.create_source("x", 5)

    # Create virtual computed
    def double(v):
        return v * 2
    y_node = factory.create_computed("y", lambda: double(x_node.read()))

    # Access value (computes on demand)
    print(y_node.read())  # 10

    # Invalidate and recompute
    x_node.write(7)
    # DeltaKVStore automatically invalidates y_node

    print(y_node.read())  # 14 (recomputed)

Design Principles:
-----------------

1. Separation of Concerns:
   - Node layer: computation graphs, state, caching
   - Observable layer: user-facing API, subscriptions
   - DeltaKVStore: change propagation, dependency tracking

2. Automatic State Management:
   - No manual state annotations required
   - Transitions happen automatically based on usage
   - Structural patterns (fan-out) guide decisions

3. Performance by Default:
   - Virtual nodes avoid storage overhead
   - Direct propagation for linear chains (O(1))
   - Caching only where sharing occurs (k ≥ 2)

4. Memory Efficiency:
   - Weak references prevent cycles
   - Automatic cleanup on state transitions
   - No duplication between layers
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

    Every Node tracks:
    - node_id: Unique identifier in the computation graph
    - _state: Current state (VIRTUAL, TRACKED, MATERIALIZED)
    - _consumers: Set of consumer IDs for fan-out detection
    - _metrics: Usage statistics (access count, consumer count, etc.)
    - _lock: Thread safety for concurrent access

    State Management:
    The Node automatically transitions based on consumer count:
    - Virtual: consumer_count < 2 (single consumer or none)
    - Tracked: consumer_count ≥ 1 and needs change propagation
    - Materialized: consumer_count ≥ 2 (fan-out detected)

    This ensures optimal caching: storage only where it provides benefit.
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

    VirtualNode is the default state for computed nodes. It provides:
    - Pure computation: Values computed on-demand by executing functions
    - Function fusion: Adjacent transforms compose into single functions
    - Direct propagation: O(1) notification to dependents without store overhead
    - Caching: Computed results cached until invalidated (dirty tracking)

    Key Properties:
    ---------------
    - _compute_func: The computation function (may be a composed fusion)
    - _direct_dependents: Weak references for fast O(1) direct notification
    - _cached_value: Last computed result (None if never computed)
    - _is_dirty: Flag indicating cache is invalid (requires recomputation)
    - _dependencies: List of dependency node IDs

    Cache Behavior:
    ---------------
    The _is_dirty flag enables efficient cache management:

        read() called:
            if _is_dirty or _cached_value is None:
                _cached_value = _compute_func()  # Compute
                _is_dirty = False                # Mark clean
            return _cached_value                 # Return cached

        invalidate() called:
            _is_dirty = True                     # Mark dirty
            notify_dependents()                 # Propagate

    This avoids unnecessary recomputation when the cache is still valid.

    Direct Propagation:
    -------------------
    VirtualNodes maintain _direct_dependents as a list of weak references.
    When the node is invalidated, it notifies all dependents directly:

        for ref in _direct_dependents:
            dep = ref()  # Unwrap weakref
            if dep and hasattr(dep, '_on_node_invalidated'):
                dep._on_node_invalidated()

    This creates O(1) propagation paths without store overhead. The
    direct dependents list is automatically cleared when transitioning
    to Tracked/Materialized state (store handles propagation then).

    Fusion:
    -------
    VirtualNode.fuse_with() enables function composition:

        VirtualNode(f1) with transform f2:
          → FusedVirtualNode(f2 ∘ f1)

        Composed function executes both f1 and f2 in sequence:
          result = f2(f1(input))

    Fusion continues as long as fan-out remains < 2. Once materialization
    occurs (fan-out ≥ 2), fusion stops and each branch fuses independently.

    State Transitions:
    ------------------
    Virtual → Tracked:
        - Triggered when subscription or tracking needed
        - Calls _register_in_store() to register with DeltaKVStore
        - Clears _direct_dependents (store handles updates)

    Virtual → Materialized:
        - Triggered when consumer_count ≥ 2 (add_consumer detects this)
        - First transitions to Tracked if needed
        - Then calls _materialize_in_store() to eagerly compute
        - Clears _direct_dependents (store handles updates)
        - Future reads are O(1) lookups

    Thread Safety:
    --------------
    All operations are protected by self._lock (RLock for reentrancy).
    Direct dependents list uses weak references to avoid circular
    dependencies and enable automatic garbage collection.
    """

    def __init__(self, node_id: str, compute_func: Callable[[], Any], store):
        super().__init__(node_id, store)
        self._compute_func = compute_func
        self._dependencies: List[str] = []
        self._direct_dependents: List[weakref.ref] = (
            []
        )  # Weak references for O(1) propagation
        self._cached_value: Optional[Any] = None
        self._is_dirty: bool = True  # Cache invalid until first computation

    def read(self) -> Any:
        """
        Compute value on-demand with caching.

        Returns cached value if available and not dirty.
        Otherwise computes and caches the result.
        """
        with self._lock:
            self._metrics.access_count += 1
            # Check if cache is valid
            if self._cached_value is None or self._is_dirty:
                self._cached_value = self._compute_func()
                self._is_dirty = False
            return self._cached_value

    def mark_dirty(self) -> None:
        """Mark cache as dirty and notify direct dependents."""
        with self._lock:
            self._is_dirty = True
            self.notify_dependents()

    def write(self, value: Any) -> None:
        """Virtual nodes are read-only."""
        raise ValueError(f"Cannot write to virtual node {self.node_id}")

    def invalidate(self) -> None:
        """Invalidate cache and notify direct dependents."""
        self.mark_dirty()

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
        # Create FusedVirtualNode to preserve fusion chain
        return FusedVirtualNode(new_id, fused_func, self._store, self._dependencies)


class FusedVirtualNode(VirtualNode):
    """
    Virtual node with composed computation functions.

    Represents a chain of fused transformations (f3 ∘ f2 ∘ f1) that execute
    as a single computation for optimal performance in linear chains.

    FusedVirtualNode is created when VirtualNode.fuse_with() is called.
    It maintains the same behavior as VirtualNode but tracks that it's a fused chain.

    Fusion combines adjacent transforms into a single composed function:

        Before fusion:
            y = f1(x)
            z = f2(y) = f2(f1(x))
            # Two separate computations

        After fusion:
            z = (f2 ∘ f1)(x)
            # Single composed function

    This reduces:
    - Memory: No intermediate storage for y's value
    - Computation: Single function call instead of two
    - Propagation: Direct notification without store overhead

    The fusion chain can be arbitrarily long, creating deeply composed
    functions without intermediate nodes. This is optimal for linear
    computation chains that have no branching.
    """

    def __init__(
        self,
        node_id: str,
        compute_func: Callable[[], Any],
        store,
        dependencies: List[str] = None,
    ):
        # Don't call super().__init__ since we need to set dependencies first
        Node.__init__(self, node_id, store)
        self._compute_func = compute_func
        self._dependencies = dependencies or []
        self._direct_dependents: List[weakref.ref] = []
        self._cached_value: Optional[Any] = None
        self._is_dirty: bool = True

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

    Source nodes cache values locally for fast access and only use the
    store for change propagation when needed.
    """

    def __init__(self, node_id: str, initial_value: Any, store):
        super().__init__(node_id, store)
        self._state = NodeState.TRACKED
        self._local_value = initial_value
        self._is_tracked = False

    def read(self) -> Any:
        """Read the stored value."""
        # Minimal lock check - Python's GIL makes reads safe without full lock
        if not self._is_tracked:
            return self._local_value

        # Tracked path: use store (no lock needed here, store handles it)
        self._metrics.access_count += 1
        return self._store.get(self.node_id)

    def write(self, value: Any) -> None:
        """Update the stored value and propagate changes."""
        # Fast path: identity check for mutable types avoids O(n) comparisons
        if self._local_value is value:
            # Same reference: check if mutable (allows in-place mutations to propagate)
            if isinstance(self._local_value, (list, dict, set)):
                # Mutable with same identity: update anyway for reactivity
                changed = True
            else:
                # Immutable: no change
                return
        else:
            # Different reference: check if value changed (may be O(n) for lists)
            try:
                changed = self._local_value != value
            except (ValueError, TypeError):
                changed = True

        if not changed:
            return

        # Update local cache
        self._local_value = value

        # Tracked path: propagate to store
        if self._is_tracked:
            self._store.set(self.node_id, value)

    def mark_tracked(self):
        """Mark this node as tracked in the store."""
        with self._lock:
            if not self._is_tracked:
                self._is_tracked = True
                self._store.set(self.node_id, self._local_value)

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

    The factory encapsulates the decision logic for which node type to create
    based on usage patterns, dependencies, and fan-out detection. It ensures
    consistent node creation and provides a central point for optimization
    decisions.

    Key Responsibilities:
    ---------------------
    1. Node Creation:
       - create_source(): Always creates SourceNode (tracked)
       - create_computed(): Creates VirtualNode by default, MaterializedNode if forced
       - create_computed_with_fusion(): Handles fusion decisions

    2. State Transitions:
       - transition_to_tracked(): Promotes a node to tracked state
       - register_dependency(): Tracks consumer relationships

    3. Dependency Management:
       - Automatically registers dependencies between nodes
       - Triggers materialization when fan-out detected (≥2 consumers)
       - Maintains a registry of all nodes for state queries

    Fan-Out Detection:
    ------------------
    The factory detects fan-out through register_dependency() which:

        1. Calls node.add_consumer(consumer_id)
        2. Node checks its consumer count
        3. If count ≥ 2, node._transition_to_materialized() is called
        4. Materialized nodes now cache their results

    This automatic detection makes optimal caching decisions without
    manual annotations or configuration.

    Example:
    --------

        factory = NodeFactory(store)

        # Create source
        x = factory.create_source("x", 5)

        # Create virtual computed (consumer_count=0, stays virtual)
        y = factory.create_computed("y", lambda: x.read() * 2)

        # Add consumer (fan-out=1, still virtual)
        factory.register_dependency("z", "y")

        # Add another consumer (fan-out=2, materializes)
        factory.register_dependency("w", "y")
        # Now: y.state == MATERIALIZED
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

    def create_computed_with_fusion(
        self, parent_node: Node, transform: Callable
    ) -> Node:
        """
        Create a computed node with fusion support.

        If parent is a fusable VirtualNode with ≤1 consumer, fuse the functions.
        Otherwise, create a normal computed node.
        """
        # Check if parent is fusable
        is_virtual = isinstance(parent_node, VirtualNode)
        is_fusable = is_virtual and parent_node._metrics.consumer_count <= 1

        if is_fusable:
            # Use parent's fuse_with method to create fused node
            fused_node = parent_node.fuse_with(transform)
            # Register in factory
            self._node_registry[fused_node.node_id] = fused_node
            return fused_node

        # Not fusable - create normal computed node
        # This path requires more context, so delegate to create_computed
        return self.create_computed(
            node_id=self._store._gen_key("computed"),
            compute_func=transform,
            dependencies=[parent_node.node_id],
        )

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
