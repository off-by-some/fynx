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
    Tuple,
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
        if not isinstance(observable, Observable):
            raise TypeError("observable must be an instance of Observable")

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
        """
        Maximum depth from source nodes to this node.

        Uses dynamic programming with cycle detection to compute the longest
        path from any root node to this node.
        """
        if self._cached_depth is not None:
            return self._cached_depth

        self._cached_depth = self._calculate_depth_with_cycle_detection()
        return self._cached_depth

    def _calculate_depth_with_cycle_detection(self) -> int:
        """
        Calculate depth using DFS with cycle detection.

        Returns the maximum depth from any source node, or 0 if cycles are detected.
        """
        visited = set()
        path_stack = set()

        def dfs_depth(node: DependencyNode) -> int:
            # Cycle detected - return 0 as safe default
            if node in path_stack:
                return 0

            # Already computed
            if node._cached_depth is not None:
                return node._cached_depth

            # Source node (no incoming dependencies)
            if not node.incoming:
                node._cached_depth = 0
                return 0

            # Mark as visiting
            path_stack.add(node)
            visited.add(node)

            try:
                # Recursively calculate depth of all dependencies
                dependency_depths = []
                for dependency in node.incoming:
                    dep_depth = dfs_depth(dependency)
                    dependency_depths.append(dep_depth)

                # Node depth is 1 + maximum dependency depth
                max_dep_depth = max(dependency_depths) if dependency_depths else 0
                node._cached_depth = 1 + max_dep_depth
                return node._cached_depth

            finally:
                # Clean up path tracking
                path_stack.discard(node)

        return dfs_depth(self)

    def invalidate_depth_cache(self) -> None:
        """Invalidate cached depth computation."""
        self._cached_depth = None

    def __repr__(self) -> str:
        return f"Node({self.observable.key}, depth={self.depth}, deps={len(self.incoming)})"


class DependencyGraph:
    """
    A pythonic dependency graph for managing observable relationships.

    This class provides an elegant, fluent API for working with dependency graphs,
    implementing Python's container protocols for seamless integration.

    The graph maintains mathematical correctness by ensuring:
    - Topological ordering respects dependency relationships
    - Cycle detection prevents infinite loops
    - Depth calculations use proper graph algorithms
    - Path finding algorithms handle cycles appropriately

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

    def __contains__(self, observable: Any) -> bool:
        """Check if an observable is in the graph."""
        return isinstance(observable, Observable) and observable in self.nodes

    def __getitem__(self, observable: Observable) -> DependencyNode:
        """Get a node by its observable."""
        if not isinstance(observable, Observable):
            raise TypeError("Key must be an Observable")
        try:
            return self.nodes[observable]
        except KeyError:
            raise KeyError(f"Observable {observable} not found in graph")

    def __setitem__(self, observable: Observable, node: DependencyNode) -> None:
        """Set a node for an observable (advanced usage)."""
        if not isinstance(observable, Observable):
            raise TypeError("Key must be an Observable")
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
            self._cached_cycles = self._detect_cycles()
        return self._cached_cycles

    @property
    def roots(self) -> List[DependencyNode]:
        """Get all root nodes (nodes with no incoming dependencies)."""
        return [node for node in self.nodes.values() if not node.incoming]

    @property
    def leaves(self) -> List[DependencyNode]:
        """Get all leaf nodes (nodes with no outgoing dependencies)."""
        return [node for node in self.nodes.values() if not node.outgoing]

    # Fluent API Methods
    def add(self, observable: Observable) -> "DependencyGraph":
        """Add an observable to the graph. Returns self for chaining."""
        if not isinstance(observable, Observable):
            raise TypeError("Can only add Observable instances to the graph")
        self.get_or_create_node(observable)
        return self

    def remove(self, observable: Observable) -> "DependencyGraph":
        """Remove an observable from the graph. Returns self for chaining."""
        if not isinstance(observable, Observable):
            raise TypeError("Can only remove Observable instances from the graph")

        if observable in self.nodes:
            self._remove_node_safely(observable)
        return self

    def clear(self) -> "DependencyGraph":
        """Clear all nodes from the graph. Returns self for chaining."""
        # Invalidate depth caches for all nodes
        for node in self.nodes.values():
            node.invalidate_depth_cache()

        self.nodes.clear()
        self._node_cache.clear()
        self._invalidate_cache()
        return self

    def get_or_create_node(self, observable: Observable) -> DependencyNode:
        """Get existing node or create new one for observable."""
        if not isinstance(observable, Observable):
            raise TypeError("observable must be an Observable instance")

        if observable in self._node_cache:
            return self._node_cache[observable]

        node = DependencyNode(observable)
        self.nodes[observable] = node
        self._node_cache[observable] = node
        self._invalidate_cache()

        return node

    def _remove_node_safely(self, observable: Observable) -> None:
        """Safely remove a node and clean up all its relationships."""
        node = self.nodes[observable]

        # Remove this node from all incoming node's outgoing sets
        for incoming_node in node.incoming:
            incoming_node.outgoing.discard(node)

        # Remove this node from all outgoing node's incoming sets
        for outgoing_node in node.outgoing:
            outgoing_node.incoming.discard(node)

        # Invalidate depth caches for affected nodes
        affected_nodes = node.incoming | node.outgoing
        for affected_node in affected_nodes:
            affected_node.invalidate_depth_cache()

        # Remove from graph
        del self.nodes[observable]
        if observable in self._node_cache:
            del self._node_cache[observable]

        self._invalidate_cache()

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

        Returns nodes in topological order (sources first). If cycles exist,
        falls back to depth-based sorting as a safe approximation.

        Time complexity: O(V + E) where V is vertices (nodes) and E is edges.
        """
        if self.is_empty:
            return []

        # Calculate incoming degrees for all current nodes
        in_degree_map = self._compute_in_degrees()
        current_nodes = set(self.nodes.values())

        # Initialize queue with nodes that have no incoming dependencies
        source_queue = self._get_source_nodes(in_degree_map)
        sorted_nodes = []

        # Process nodes using Kahn's algorithm
        while source_queue:
            current_node = source_queue.pop(0)
            sorted_nodes.append(current_node)

            # Update in-degrees of dependent nodes
            self._process_dependents(
                current_node, in_degree_map, source_queue, current_nodes
            )

        # Verify topological ordering is complete
        if len(sorted_nodes) != len(current_nodes):
            # Cycle detected - fall back to depth-based sorting
            return self._fallback_depth_sort(current_nodes)

        return sorted_nodes

    def _compute_in_degrees(self) -> Dict[DependencyNode, int]:
        """Compute in-degrees for all nodes in the current graph."""
        current_nodes = set(self.nodes.values())
        in_degree_map = {}

        for node in current_nodes:
            # Count incoming edges only from nodes that still exist
            in_degree_map[node] = len(
                [pred for pred in node.incoming if pred in current_nodes]
            )

        return in_degree_map

    def _get_source_nodes(
        self, in_degree_map: Dict[DependencyNode, int]
    ) -> List[DependencyNode]:
        """Get nodes with no incoming dependencies (in-degree 0)."""
        return [node for node, degree in in_degree_map.items() if degree == 0]

    def _process_dependents(
        self,
        node: DependencyNode,
        in_degree_map: Dict[DependencyNode, int],
        source_queue: List[DependencyNode],
        current_nodes: Set[DependencyNode],
    ) -> None:
        """Process all dependents of a node, updating their in-degrees."""
        for dependent in list(node.outgoing):
            if dependent in current_nodes and dependent in in_degree_map:
                in_degree_map[dependent] -= 1
                if in_degree_map[dependent] == 0:
                    source_queue.append(dependent)

    def _fallback_depth_sort(self, nodes: Set[DependencyNode]) -> List[DependencyNode]:
        """Fallback sorting by depth when topological sort fails due to cycles."""
        return sorted(nodes, key=lambda n: n.depth)

    def _detect_cycles(self) -> List[List[DependencyNode]]:
        """
        Detect all cycles in the dependency graph using DFS.

        Returns a list of cycles, where each cycle is represented as a list of nodes.
        Uses the standard cycle detection algorithm with three states:
        - not visited
        - visiting (in current path)
        - visited (processed)

        Time complexity: O(V + E) where V is vertices and E is edges.
        """
        cycles = []
        visited = set()
        recursion_stack = set()

        def dfs_visit(node: DependencyNode, path: List[DependencyNode]) -> bool:
            """
            Visit a node during DFS cycle detection.

            Returns True if a cycle is found starting from this node.
            """
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            # Check all outgoing neighbors
            for neighbor in node.outgoing:
                if neighbor not in visited:
                    # Not visited - continue DFS
                    if dfs_visit(neighbor, path):
                        return True
                elif neighbor in recursion_stack:
                    # Found back edge - cycle detected
                    cycle = self._extract_cycle(path, neighbor)
                    cycles.append(cycle)
                    return True

            # Backtrack
            path.pop()
            recursion_stack.remove(node)
            return False

        # Run DFS from each unvisited node
        for node in self.nodes.values():
            if node not in visited:
                dfs_visit(node, [])

        return cycles

    def _extract_cycle(
        self, path: List[DependencyNode], back_edge_target: DependencyNode
    ) -> List[DependencyNode]:
        """
        Extract a cycle from the current path when a back edge is found.

        The cycle starts from the back edge target and includes all nodes
        up to and including the current node.
        """
        cycle_start_index = path.index(back_edge_target)
        cycle = path[cycle_start_index:] + [back_edge_target]
        return cycle

    def find_paths(
        self, start: DependencyNode, end: DependencyNode, max_depth: int = 10
    ) -> List[List[DependencyNode]]:
        """
        Find all paths from start to end node within max_depth limit.

        Uses DFS with cycle detection to enumerate all valid paths.
        Paths are deduplicated based on observable keys.

        Args:
            start: Starting node for path search
            end: Target node to reach
            max_depth: Maximum path length (prevents infinite recursion)

        Returns:
            List of unique paths, where each path is a list of nodes
        """
        if start not in self.nodes.values() or end not in self.nodes.values():
            return []

        if max_depth < 1:
            return []

        paths = []
        self._dfs_path_search(start, end, max_depth, paths)

        # Deduplicate paths based on observable sequence
        return self._deduplicate_paths(paths)

    def _dfs_path_search(
        self,
        current: DependencyNode,
        target: DependencyNode,
        max_depth: int,
        paths: List[List[DependencyNode]],
        path: Optional[List[DependencyNode]] = None,
        visited: Optional[Set[DependencyNode]] = None,
        depth: int = 0,
    ) -> None:
        """
        Perform DFS to find all paths from current to target.

        Uses backtracking with cycle detection via visited set.
        """
        if path is None:
            path = []
        if visited is None:
            visited = set()

        # Depth limit exceeded
        if depth > max_depth:
            return

        # Cycle detected
        if current in visited:
            return

        # Add current node to path
        path.append(current)

        # Found target (but exclude trivial single-node paths)
        if current == target and len(path) > 1:
            paths.append(path.copy())
        else:
            # Continue DFS to neighbors
            visited.add(current)

            for neighbor in current.outgoing:
                self._dfs_path_search(
                    neighbor, target, max_depth, paths, path, visited, depth + 1
                )

            visited.remove(current)

        # Backtrack
        path.pop()

    def _deduplicate_paths(
        self, paths: List[List[DependencyNode]]
    ) -> List[List[DependencyNode]]:
        """Remove duplicate paths based on sequence of observable keys."""
        unique_paths = []
        seen_path_keys = set()

        for path in paths:
            path_key = tuple(node.observable.key for node in path)
            if path_key not in seen_path_keys:
                seen_path_keys.add(path_key)
                unique_paths.append(path)

        return unique_paths

    def can_reach(
        self, target: DependencyNode, start: DependencyNode, max_hops: int = 10
    ) -> bool:
        """
        Check if target is reachable from start within max_hops using BFS.

        This implements a bounded reachability check to prevent excessive
        computation on large graphs.

        Args:
            target: Node to reach
            start: Starting node
            max_hops: Maximum number of hops to search

        Returns:
            True if target is reachable within the hop limit
        """
        if start == target:
            return True

        if max_hops < 1:
            return False

        return self._bfs_reachability_check(start, target, max_hops)

    def _bfs_reachability_check(
        self, start: DependencyNode, target: DependencyNode, max_hops: int
    ) -> bool:
        """
        Perform BFS to check reachability within hop limit.

        Returns True if target is found within max_hops from start.
        """
        visited = set()
        queue = deque([(start, 0)])  # (node, depth)

        while queue:
            current_node, depth = queue.popleft()

            # Exceeded hop limit
            if depth >= max_hops:
                continue

            # Skip already visited nodes
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check all outgoing neighbors
            for neighbor in current_node.outgoing:
                if neighbor == target:
                    return True

                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return False

    def copy(self) -> "DependencyGraph":
        """
        Create a deep copy of the current graph.

        Preserves all node relationships and metadata while creating
        independent node instances.
        """
        new_graph = self.__class__()
        node_mapping = {}

        # Create all nodes first
        for observable, original_node in self.nodes.items():
            new_node = self._create_node_copy(original_node)
            new_graph.nodes[observable] = new_node
            node_mapping[original_node] = new_node

        # Reconstruct all edges
        self._copy_node_relationships(new_graph, node_mapping)

        return new_graph

    def _create_node_copy(self, original_node: DependencyNode) -> DependencyNode:
        """Create a copy of a single node with all its metadata."""
        new_node = DependencyNode(original_node.observable)
        new_node.computation_func = original_node.computation_func
        new_node.source_observable = original_node.source_observable
        new_node.visit_count = original_node.visit_count
        return new_node

    def _copy_node_relationships(
        self,
        new_graph: "DependencyGraph",
        node_mapping: Dict[DependencyNode, DependencyNode],
    ) -> None:
        """Copy all incoming and outgoing relationships between nodes."""
        for original_node, new_node in node_mapping.items():
            # Copy incoming relationships
            new_node.incoming = {
                node_mapping[pred]
                for pred in original_node.incoming
                if pred in node_mapping
            }

            # Copy outgoing relationships
            new_node.outgoing = {
                node_mapping[succ]
                for succ in original_node.outgoing
                if succ in node_mapping
            }

    def copy_graph(self) -> "DependencyGraph":
        """Alias for copy() method for backward compatibility."""
        return self.copy()

    # Private Methods
    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when graph changes."""
        self._cached_cycles = None
        self._cached_stats = None

    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics about the graph structure.

        Returns metrics that characterize the graph's topology and complexity.
        """
        if self.is_empty:
            return self._empty_graph_statistics()

        # Compute basic structural metrics
        nodes = list(self.nodes.values())
        max_depth = max(node.depth for node in nodes)

        # Count edges (each incoming edge contributes to total)
        total_edges = sum(len(node.incoming) for node in nodes)

        # Count topological features
        root_count = len(self.roots)
        leaf_count = len(self.leaves)
        cycle_count = 1 if self.has_cycles else 0

        return {
            "total_nodes": len(nodes),
            "max_depth": max_depth,
            "total_edges": total_edges,
            "roots": root_count,
            "leaves": leaf_count,
            "cycles": cycle_count,
        }

    def _empty_graph_statistics(self) -> Dict[str, Any]:
        """Return statistics for an empty graph."""
        return {
            "total_nodes": 0,
            "max_depth": 0,
            "total_edges": 0,
            "roots": 0,
            "leaves": 0,
            "cycles": 0,
        }


def get_graph_statistics(graph: DependencyGraph) -> Dict[str, Any]:
    """Get statistics about a dependency graph."""
    return graph.statistics
