"""
Dependency Management for Reactive Graphs
========================================

This module provides core dependency graph management functionality that can be
used independently of optimization logic. It handles:

- Dependency graph construction and representation
- Topological sorting with cycle detection
- Graph traversal and path finding
- Graph statistics and analysis
- Graph copying and manipulation

Classes
-------
DependencyNode : Represents a node in the dependency graph
DependencyGraph : Base class for managing dependency graphs

Functions
---------
get_graph_statistics : Get statistics about a dependency graph
"""

import weakref
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
)

from ..observable.interfaces import Observable

T = TypeVar("T")


class DependencyNode:
    """
    Node in the dependency graph representing an observable and its relationships.

    Each node tracks:
    - The observable it represents
    - Incoming dependencies (observables this one depends on)
    - Outgoing dependents (observables that depend on this one)
    - Computation function and metadata
    - Graph traversal state
    """

    def __init__(self, observable: Observable):
        self.observable = observable
        self.incoming: Set["DependencyNode"] = set()  # Dependencies
        self.outgoing: Set["DependencyNode"] = set()  # Dependents
        self.computation_func: Optional[Callable] = None
        self.source_observable: Optional[Observable] = None

        # Graph traversal state
        self.visit_count = 0

        # Cached computations
        self._cached_depth: Optional[int] = None

    @property
    def depth(self) -> int:
        """Maximum depth from source nodes to this node."""
        if self._cached_depth is not None:
            return self._cached_depth

        # Handle cycles by using a visited set
        visited = set()

        def calculate_depth(node: DependencyNode, path: set) -> int:
            if node in path:
                # Cycle detected, return 0 as a safe default
                return 0
            if node._cached_depth is not None:
                return node._cached_depth
            if not node.incoming:
                node._cached_depth = 0
                return 0

            path.add(node)
            try:
                parent_depths = []
                for parent in node.incoming:
                    parent_depths.append(calculate_depth(parent, path))
                node._cached_depth = 1 + max(parent_depths) if parent_depths else 0
                return node._cached_depth
            finally:
                path.discard(node)

        self._cached_depth = calculate_depth(self, set())
        return self._cached_depth

    def __repr__(self) -> str:
        return f"Node({self.observable.key}, depth={self.depth}, deps={len(self.incoming)})"


class DependencyGraph:
    """
    A pythonic dependency graph for managing observable relationships.

    This class provides an elegant, fluent API for working with dependency graphs,
    implementing Python's container protocols for seamless integration.

    Examples
    --------
    >>> graph = DependencyGraph()
    >>> # Add nodes fluently
    >>> graph.add(observable1).add(observable2)
    >>> # Check membership
    >>> observable1 in graph  # True
    >>> # Access by observable
    >>> node = graph[observable1]
    >>> # Iterate over nodes
    >>> for node in graph: print(node)
    >>> # Use as context manager
    >>> with graph.batch_update():
    ...     graph.add(obs1).add(obs2)
    """

    def __init__(self):
        self.nodes: Dict[Observable, DependencyNode] = {}
        self._root_nodes: Set[DependencyNode] = set()
        self._node_cache = weakref.WeakKeyDictionary()
        self._cached_cycles: Optional[List[List[DependencyNode]]] = None
        self._cached_stats: Optional[Dict[str, Any]] = None

    # Container Protocol
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __iter__(self):
        """Iterate over all nodes in the graph."""
        return iter(self.nodes.values())

    def __contains__(self, observable: Observable) -> bool:
        """Check if an observable is in the graph."""
        return observable in self.nodes

    def __getitem__(self, observable: Observable) -> DependencyNode:
        """Get a node by its observable."""
        if observable not in self.nodes:
            raise KeyError(f"Observable {observable} not found in graph")
        return self.nodes[observable]

    def __setitem__(self, observable: Observable, node: DependencyNode) -> None:
        """Set a node for an observable (advanced usage)."""
        if not isinstance(node, DependencyNode):
            raise TypeError("Value must be a DependencyNode")
        if node.observable != observable:
            raise ValueError("Node's observable must match the key")
        self.nodes[observable] = node
        self._node_cache[observable] = node
        self._invalidate_cache()

    # Context Manager Protocol
    def __enter__(self):
        """Context manager entry for batch operations."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - could add cleanup logic here."""
        pass

    # String Representation
    def __str__(self) -> str:
        """Human-readable string representation."""
        stats = self.statistics
        return f"DependencyGraph(nodes={stats['total_nodes']}, edges={stats['total_edges']}, depth={stats['max_depth']})"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"{self.__class__.__name__}({list(self.nodes.keys())})"

    # Properties
    @property
    def is_empty(self) -> bool:
        """Check if the graph has no nodes."""
        return len(self) == 0

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph."""
        if self._cached_stats is None:
            self._cached_stats = self._compute_statistics()
        return self._cached_stats

    @property
    def has_cycles(self) -> bool:
        """Check if the graph contains any cycles."""
        return len(self.cycles) > 0

    @property
    def cycles(self) -> List[List[DependencyNode]]:
        """Get all cycles in the graph (cached)."""
        if self._cached_cycles is None:
            self._cached_cycles = self.detect_cycles()
        return self._cached_cycles

    @property
    def roots(self) -> List[DependencyNode]:
        """Get all root nodes (nodes with no incoming dependencies)."""
        return [node for node in self if not node.incoming]

    @property
    def leaves(self) -> List[DependencyNode]:
        """Get all leaf nodes (nodes with no outgoing dependencies)."""
        return [node for node in self if not node.outgoing]

    # Fluent API Methods
    def add(self, observable: Observable) -> "DependencyGraph":
        """Add an observable to the graph. Returns self for chaining."""
        self.get_or_create_node(observable)
        return self

    def remove(self, observable: Observable) -> "DependencyGraph":
        """Remove an observable from the graph. Returns self for chaining."""
        if observable in self.nodes:
            node = self.nodes[observable]
            # Remove from incoming/outgoing relationships
            for incoming in node.incoming:
                incoming.outgoing.discard(node)
            for outgoing in node.outgoing:
                outgoing.incoming.discard(node)
            del self.nodes[observable]
            if observable in self._node_cache:
                del self._node_cache[observable]
            self._invalidate_cache()
        return self

    def clear(self) -> "DependencyGraph":
        """Clear all nodes from the graph. Returns self for chaining."""
        self.nodes.clear()
        self._root_nodes.clear()
        self._node_cache.clear()
        self._invalidate_cache()
        return self

    def get_or_create_node(self, observable: Observable) -> DependencyNode:
        """Get existing node or create new one for observable."""
        if observable in self._node_cache:
            return self._node_cache[observable]

        node = DependencyNode(observable)
        self.nodes[observable] = node
        self._node_cache[observable] = node
        self._invalidate_cache()

        return node

    def batch_update(self) -> "DependencyGraph":
        """Context manager for batch operations (currently just returns self)."""
        return self

    # Core Graph Operations
    def build_from_observables(
        self, observables: List[Observable]
    ) -> "DependencyGraph":
        """
        Build dependency graph starting from given observables.

        This is an abstract method that should be implemented by subclasses
        to handle the specific dependency extraction logic for their domain.
        """
        raise NotImplementedError("Subclasses must implement build_from_observables")

    def topological_sort(self) -> List[DependencyNode]:
        """
        Perform topological sort using Kahn's algorithm.

        Returns nodes in topological order (sources first).
        """
        # Calculate incoming degrees for current nodes only
        current_nodes = set(self.nodes.values())
        in_degree = {}
        for node in current_nodes:
            # Count only incoming edges from nodes that still exist
            in_degree[node] = len(
                [pred for pred in node.incoming if pred in current_nodes]
            )

        # Start with nodes that have no incoming edges from current nodes (sources)
        queue = [node for node in current_nodes if in_degree[node] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # For each dependent of current node that still exists
            for dependent in list(current.outgoing):
                if dependent in current_nodes and dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check if we got all current nodes
        if len(result) != len(current_nodes):
            # If there are cycles or other issues, fall back to depth-based sorting
            return sorted(current_nodes, key=lambda n: n.depth)

        return result

    def detect_cycles(self) -> List[List[DependencyNode]]:
        """
        Detect cycles in the dependency graph.

        Returns a list of cycles, where each cycle is represented as a list of nodes.
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs_cycle_detect(node: DependencyNode, path: List[DependencyNode]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all neighbors
            for neighbor in node.outgoing:
                if neighbor not in visited:
                    if dfs_cycle_detect(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in self.nodes.values():
            if node not in visited:
                dfs_cycle_detect(node, [])

        return cycles

    def find_paths(
        self, start: DependencyNode, end: DependencyNode, max_depth: int = 10
    ) -> List[List[DependencyNode]]:
        """
        Find all paths from start to end node up to max_depth.

        Uses DFS with cycle detection and early termination.
        """
        paths = []

        def dfs(
            current: DependencyNode,
            path: List[DependencyNode],
            depth: int,
            visited: Set[DependencyNode],
        ):
            if depth > max_depth:
                return

            # Check for cycles
            if current in visited:
                return

            path.append(current)

            if current == end and len(path) > 1:  # Found a valid path
                paths.append(path.copy())
            else:
                # Continue along outgoing edges
                visited.add(current)

                for neighbor in current.outgoing:
                    dfs(neighbor, path, depth + 1, visited)

                visited.remove(current)

            path.pop()

        dfs(start, [], 0, set())

        # Remove duplicate paths
        unique_paths = []
        seen_paths = set()

        for path in paths:
            path_tuple = tuple(n.observable.key for n in path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                unique_paths.append(path)

        return unique_paths

    def can_reach(
        self, target: DependencyNode, current: DependencyNode, max_hops: int = 10
    ) -> bool:
        """
        Check if target is reachable from current within max_hops.

        Uses BFS for reachability checking with depth limit.
        """
        if current == target:
            return True

        visited = set()
        queue = deque([(current, 0)])

        while queue:
            node, depth = queue.popleft()

            if depth >= max_hops:
                continue

            if node in visited:
                continue
            visited.add(node)

            for neighbor in node.outgoing:
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return False

    def copy(self) -> "DependencyGraph":
        """Create a deep copy of the current graph."""
        new_graph = self.__class__()
        node_map = {}

        # Create new nodes
        for obs, node in self.nodes.items():
            new_node = DependencyNode(obs)
            new_node.computation_func = node.computation_func
            new_node.source_observable = node.source_observable
            new_graph.nodes[obs] = new_node
            node_map[node] = new_node

        # Copy edges (only for nodes that still exist in the mapping)
        for obs, node in self.nodes.items():
            new_node = new_graph.nodes[obs]
            new_node.incoming = {
                node_map[pred] for pred in node.incoming if pred in node_map
            }
            new_node.outgoing = {
                node_map[succ] for succ in node.outgoing if succ in node_map
            }

        return new_graph

    def copy_graph(self) -> "DependencyGraph":
        """Alias for copy() method for backward compatibility."""
        return self.copy()

    # Private Methods
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when graph changes."""
        self._cached_cycles = None
        self._cached_stats = None

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics about the graph."""
        if not self.nodes:
            return {
                "total_nodes": 0,
                "max_depth": 0,
                "total_edges": 0,
                "roots": 0,
                "leaves": 0,
                "cycles": 0,
            }

        max_depth = max((node.depth for node in self.nodes.values()), default=0)
        total_edges = sum(len(node.incoming) for node in self.nodes.values())
        cycles = len(self.cycles)

        return {
            "total_nodes": len(self.nodes),
            "max_depth": max_depth,
            "total_edges": total_edges,
            "roots": len(self.roots),
            "leaves": len(self.leaves),
            "cycles": cycles,
        }


def get_graph_statistics(graph: DependencyGraph) -> Dict[str, Any]:
    """Get statistics about a dependency graph."""
    return graph.statistics
