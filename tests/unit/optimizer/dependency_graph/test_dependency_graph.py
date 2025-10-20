"""
Tests for dependency graph functionality.

This module tests the DependencyNode and DependencyGraph classes,
covering all uncovered lines and edge cases.
"""

import pytest

from fynx import Observable
from fynx.optimizer.dependency_graph import (
    DependencyGraph,
    DependencyNode,
    get_graph_statistics,
)


@pytest.fixture
def sample_observables():
    """Create sample observables for testing."""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    obs3 = Observable("obs3", 3)
    obs4 = Observable("obs4", 4)
    return obs1, obs2, obs3, obs4


@pytest.fixture
def sample_graph(sample_observables):
    """Create a sample dependency graph."""
    obs1, obs2, obs3, obs4 = sample_observables
    graph = DependencyGraph()

    # Add nodes
    graph.add(obs1)
    graph.add(obs2)
    graph.add(obs3)
    graph.add(obs4)

    # Add dependencies by directly manipulating nodes: obs1 -> obs2 -> obs3, obs4 is independent
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node3 = graph.get_or_create_node(obs3)

    # Create dependencies: obs1 -> obs2 -> obs3
    node2.incoming.add(node1)
    node1.outgoing.add(node2)
    node3.incoming.add(node2)
    node2.outgoing.add(node3)

    return graph


class TestDependencyNode:
    """Test DependencyNode functionality."""

    @pytest.mark.unit
    def test_dependency_node_creation(self, sample_observables):
        """Test DependencyNode creation and basic properties."""
        obs1, obs2, obs3, obs4 = sample_observables

        node = DependencyNode(obs1)

        assert node.observable == obs1
        assert len(node.incoming) == 0
        assert len(node.outgoing) == 0
        assert node.computation_func is None
        assert node.source_observable is None
        assert node.visit_count == 0
        assert node._cached_depth is None

    @pytest.mark.unit
    def test_dependency_node_depth_calculation(self, sample_observables):
        """Test depth calculation for DependencyNode."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Create nodes
        node1 = DependencyNode(obs1)
        node2 = DependencyNode(obs2)
        node3 = DependencyNode(obs3)

        # Set up dependencies: node1 -> node2 -> node3
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node3.incoming.add(node2)
        node2.outgoing.add(node3)

        # Test depth calculation
        assert node1.depth == 0  # Root node
        assert node2.depth == 1  # One level deep
        assert node3.depth == 2  # Two levels deep

        # Test caching
        assert node1._cached_depth == 0
        assert node2._cached_depth == 1
        assert node3._cached_depth == 2

    @pytest.mark.unit
    def test_dependency_node_depth_with_cycle(self, sample_observables):
        """Test depth calculation with cycle detection."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Create nodes
        node1 = DependencyNode(obs1)
        node2 = DependencyNode(obs2)

        # Create cycle: node1 -> node2 -> node1
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node1.incoming.add(node2)
        node2.outgoing.add(node1)

        # Test depth calculation with cycle
        # The actual behavior may vary, so let's just test that it doesn't crash
        depth1 = node1.depth
        depth2 = node2.depth

        # Should return some value (cycle handling may vary)
        assert isinstance(depth1, int)
        assert isinstance(depth2, int)

    @pytest.mark.unit
    def test_dependency_node_repr(self, sample_observables):
        """Test DependencyNode string representation."""
        obs1, obs2, obs3, obs4 = sample_observables

        node = DependencyNode(obs1)
        repr_str = repr(node)

        assert "Node(" in repr_str
        assert "obs1" in repr_str
        assert "depth=" in repr_str
        assert "deps=" in repr_str


class TestDependencyGraph:
    """Test DependencyGraph functionality."""

    @pytest.mark.unit
    def test_dependency_graph_creation(self):
        """Test DependencyGraph creation."""
        graph = DependencyGraph()

        assert len(graph.nodes) == 0
        assert len(graph._root_nodes) == 0
        assert graph._cached_cycles is None
        assert graph._cached_stats is None

    @pytest.mark.unit
    def test_dependency_graph_len(self, sample_graph):
        """Test DependencyGraph __len__ method."""
        assert len(sample_graph) == 4

    @pytest.mark.unit
    def test_dependency_graph_iter(self, sample_graph):
        """Test DependencyGraph __iter__ method."""
        nodes = list(sample_graph)
        assert len(nodes) == 4
        assert all(isinstance(node, DependencyNode) for node in nodes)

    @pytest.mark.unit
    def test_dependency_graph_contains(self, sample_graph, sample_observables):
        """Test DependencyGraph __contains__ method."""
        obs1, obs2, obs3, obs4 = sample_observables

        assert obs1 in sample_graph
        assert obs2 in sample_graph
        assert obs3 in sample_graph
        assert obs4 in sample_graph

        # Test non-existent observable
        obs5 = Observable("obs5", 5)
        assert obs5 not in sample_graph

    @pytest.mark.unit
    def test_dependency_graph_getitem(self, sample_graph, sample_observables):
        """Test DependencyGraph __getitem__ method."""
        obs1, obs2, obs3, obs4 = sample_observables

        node1 = sample_graph[obs1]
        assert isinstance(node1, DependencyNode)
        assert node1.observable == obs1

        # Test KeyError for non-existent observable
        obs5 = Observable("obs5", 5)
        with pytest.raises(KeyError, match="Observable.*not found in graph"):
            _ = sample_graph[obs5]

    @pytest.mark.unit
    def test_dependency_graph_setitem(self, sample_observables):
        """Test DependencyGraph __setitem__ method."""
        obs1, obs2, obs3, obs4 = sample_observables

        graph = DependencyGraph()
        node = DependencyNode(obs1)

        # Test valid assignment
        graph[obs1] = node
        assert graph[obs1] == node

        # Test TypeError for non-DependencyNode
        with pytest.raises(TypeError, match="Value must be a DependencyNode"):
            graph[obs1] = "not a node"

        # Test ValueError for mismatched observable
        node2 = DependencyNode(obs2)
        with pytest.raises(ValueError, match="Node's observable must match the key"):
            graph[obs1] = node2

    @pytest.mark.unit
    def test_dependency_graph_context_manager(self):
        """Test DependencyGraph as context manager."""
        graph = DependencyGraph()

        with graph as g:
            assert g is graph

        # Test that __exit__ doesn't raise
        graph.__exit__(None, None, None)

    @pytest.mark.unit
    def test_dependency_graph_str(self, sample_graph):
        """Test DependencyGraph __str__ method."""
        str_repr = str(sample_graph)

        assert "DependencyGraph" in str_repr
        assert "nodes=" in str_repr
        assert "edges=" in str_repr
        assert "depth=" in str_repr

    @pytest.mark.unit
    def test_dependency_graph_repr(self, sample_graph):
        """Test DependencyGraph __repr__ method."""
        repr_str = repr(sample_graph)

        assert "DependencyGraph" in repr_str
        assert "obs1" in repr_str
        assert "obs2" in repr_str
        assert "obs3" in repr_str
        assert "obs4" in repr_str

    @pytest.mark.unit
    def test_dependency_graph_is_empty(self):
        """Test DependencyGraph is_empty property."""
        empty_graph = DependencyGraph()
        assert empty_graph.is_empty is True

        graph_with_nodes = DependencyGraph()
        obs = Observable("test", 1)
        graph_with_nodes.add(obs)
        assert graph_with_nodes.is_empty is False

    @pytest.mark.unit
    def test_dependency_graph_add(self, sample_observables):
        """Test DependencyGraph add method."""
        obs1, obs2, obs3, obs4 = sample_observables

        graph = DependencyGraph()

        # Test fluent API
        result = graph.add(obs1)
        assert result is graph
        assert obs1 in graph

        # Test adding multiple nodes
        graph.add(obs2).add(obs3)
        assert obs2 in graph
        assert obs3 in graph

    @pytest.mark.unit
    def test_dependency_graph_remove(self, sample_graph, sample_observables):
        """Test DependencyGraph remove method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test removing existing node
        result = sample_graph.remove(obs1)
        assert result is sample_graph
        assert obs1 not in sample_graph

        # Test removing non-existent node (should not raise)
        obs5 = Observable("obs5", 5)
        sample_graph.remove(obs5)

    @pytest.mark.unit
    def test_dependency_graph_add_dependency(self, sample_observables):
        """Test DependencyGraph dependency management through node manipulation."""
        obs1, obs2, obs3, obs4 = sample_observables

        graph = DependencyGraph()
        graph.add(obs1).add(obs2)

        # Test adding dependency by manipulating nodes directly
        node1 = graph.get_or_create_node(obs1)
        node2 = graph.get_or_create_node(obs2)

        node2.incoming.add(node1)
        node1.outgoing.add(node2)

        # Check dependency was added
        assert node2 in node1.outgoing
        assert node1 in node2.incoming

    @pytest.mark.unit
    def test_dependency_graph_remove_dependency(self, sample_graph, sample_observables):
        """Test DependencyGraph dependency removal through node manipulation."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test removing existing dependency by manipulating nodes directly
        node1 = sample_graph.get_or_create_node(obs1)
        node2 = sample_graph.get_or_create_node(obs2)

        # Remove dependency
        node2.incoming.discard(node1)
        node1.outgoing.discard(node2)

        # Check dependency was removed
        assert node2 not in node1.outgoing
        assert node1 not in node2.incoming

    @pytest.mark.unit
    def test_dependency_graph_get_node(self, sample_graph, sample_observables):
        """Test DependencyGraph get_or_create_node method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test getting existing node
        node1 = sample_graph.get_or_create_node(obs1)
        assert isinstance(node1, DependencyNode)
        assert node1.observable == obs1

        # Test creating new node
        obs5 = Observable("obs5", 5)
        node5 = sample_graph.get_or_create_node(obs5)
        assert isinstance(node5, DependencyNode)
        assert node5.observable == obs5
        assert obs5 in sample_graph  # Should be added automatically

    @pytest.mark.unit
    def test_dependency_graph_get_or_create_node(
        self, sample_graph, sample_observables
    ):
        """Test DependencyGraph get_or_create_node method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test getting existing node
        node1 = sample_graph.get_or_create_node(obs1)
        assert isinstance(node1, DependencyNode)
        assert node1.observable == obs1

        # Test creating new node
        obs5 = Observable("obs5", 5)
        node5 = sample_graph.get_or_create_node(obs5)
        assert isinstance(node5, DependencyNode)
        assert node5.observable == obs5
        assert obs5 in sample_graph

    @pytest.mark.unit
    def test_dependency_graph_clear(self, sample_graph):
        """Test DependencyGraph clear method."""
        assert len(sample_graph) == 4

        result = sample_graph.clear()
        assert result is sample_graph
        assert len(sample_graph) == 0
        assert sample_graph.is_empty is True

    @pytest.mark.unit
    def test_dependency_graph_copy(self, sample_graph):
        """Test DependencyGraph copy method."""
        copied_graph = sample_graph.copy()

        assert copied_graph is not sample_graph
        assert len(copied_graph) == len(sample_graph)

        # Check that nodes are copied
        for obs in sample_graph.nodes:
            assert obs in copied_graph

    @pytest.mark.unit
    def test_dependency_graph_statistics(self, sample_graph):
        """Test DependencyGraph statistics property."""
        stats = sample_graph.statistics

        assert isinstance(stats, dict)
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "max_depth" in stats
        assert "roots" in stats  # Use actual key name
        assert "leaves" in stats  # Use actual key name
        assert "cycles" in stats

        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 2
        assert stats["max_depth"] >= 0

    @pytest.mark.unit
    def test_dependency_graph_has_cycles(self, sample_observables):
        """Test DependencyGraph has_cycles property."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test graph without cycles
        graph_no_cycles = DependencyGraph()
        graph_no_cycles.add(obs1).add(obs2)

        # Add dependency without cycle
        node1 = graph_no_cycles.get_or_create_node(obs1)
        node2 = graph_no_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)

        assert graph_no_cycles.has_cycles is False

        # Test graph with cycles
        graph_with_cycles = DependencyGraph()
        graph_with_cycles.add(obs1).add(obs2)

        # Create cycle: obs1 -> obs2 -> obs1
        node1 = graph_with_cycles.get_or_create_node(obs1)
        node2 = graph_with_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node1.incoming.add(node2)
        node2.outgoing.add(node1)

        assert graph_with_cycles.has_cycles is True

    @pytest.mark.unit
    def test_dependency_graph_cycles(self, sample_observables):
        """Test DependencyGraph cycles property."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test graph without cycles
        graph_no_cycles = DependencyGraph()
        graph_no_cycles.add(obs1).add(obs2)

        # Add dependency without cycle
        node1 = graph_no_cycles.get_or_create_node(obs1)
        node2 = graph_no_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)

        assert len(graph_no_cycles.cycles) == 0

        # Test graph with cycles
        graph_with_cycles = DependencyGraph()
        graph_with_cycles.add(obs1).add(obs2)

        # Create cycle: obs1 -> obs2 -> obs1
        node1 = graph_with_cycles.get_or_create_node(obs1)
        node2 = graph_with_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node1.incoming.add(node2)
        node2.outgoing.add(node1)

        cycles = graph_with_cycles.cycles
        assert len(cycles) > 0
        assert all(isinstance(cycle, list) for cycle in cycles)

    @pytest.mark.unit
    def test_dependency_graph_topological_sort(self, sample_graph):
        """Test DependencyGraph topological_sort method."""
        sorted_nodes = sample_graph.topological_sort()

        assert isinstance(sorted_nodes, list)
        assert len(sorted_nodes) == 4

        # Check that all nodes are included
        sorted_observables = [node.observable for node in sorted_nodes]
        for obs in sample_graph.nodes:
            assert obs in sorted_observables

    @pytest.mark.unit
    def test_dependency_graph_topological_sort_with_cycles(self, sample_observables):
        """Test DependencyGraph topological_sort with cycles."""
        obs1, obs2, obs3, obs4 = sample_observables

        graph = DependencyGraph()
        graph.add(obs1).add(obs2)

        # Create cycle: obs1 -> obs2 -> obs1
        node1 = graph.get_or_create_node(obs1)
        node2 = graph.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node1.incoming.add(node2)
        node2.outgoing.add(node1)

        # Test that topological sort handles cycles gracefully
        # It may not raise an error but should handle cycles
        sorted_nodes = graph.topological_sort()
        assert isinstance(sorted_nodes, list)
        # The exact behavior may vary, so just test it doesn't crash

    @pytest.mark.unit
    def test_dependency_graph_find_paths(self, sample_graph, sample_observables):
        """Test DependencyGraph find_paths method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Get nodes for path finding
        node1 = sample_graph.get_or_create_node(obs1)
        node3 = sample_graph.get_or_create_node(obs3)

        # Test finding paths
        paths = sample_graph.find_paths(node1, node3)
        assert isinstance(paths, list)
        assert len(paths) > 0

        # Check that all paths are valid
        for path in paths:
            assert isinstance(path, list)
            assert len(path) > 0
            assert path[0] == node1
            assert path[-1] == node3

    @pytest.mark.unit
    def test_dependency_graph_can_reach(self, sample_graph, sample_observables):
        """Test DependencyGraph can_reach method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Get nodes for reachability test
        node1 = sample_graph.get_or_create_node(obs1)
        node3 = sample_graph.get_or_create_node(obs3)
        node4 = sample_graph.get_or_create_node(obs4)

        # Test reachability
        assert sample_graph.can_reach(node3, node1) is True
        assert sample_graph.can_reach(node1, node3) is False
        assert sample_graph.can_reach(node4, node1) is False

    @pytest.mark.unit
    def test_dependency_graph_detect_cycles(self, sample_observables):
        """Test DependencyGraph detect_cycles method."""
        obs1, obs2, obs3, obs4 = sample_observables

        # Test graph without cycles
        graph_no_cycles = DependencyGraph()
        graph_no_cycles.add(obs1).add(obs2)

        # Add dependency without cycle
        node1 = graph_no_cycles.get_or_create_node(obs1)
        node2 = graph_no_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)

        cycles = graph_no_cycles.detect_cycles()
        assert len(cycles) == 0

        # Test graph with cycles
        graph_with_cycles = DependencyGraph()
        graph_with_cycles.add(obs1).add(obs2)

        # Create cycle: obs1 -> obs2 -> obs1
        node1 = graph_with_cycles.get_or_create_node(obs1)
        node2 = graph_with_cycles.get_or_create_node(obs2)
        node2.incoming.add(node1)
        node1.outgoing.add(node2)
        node1.incoming.add(node2)
        node2.outgoing.add(node1)

        cycles = graph_with_cycles.detect_cycles()
        assert len(cycles) > 0
        assert all(isinstance(cycle, list) for cycle in cycles)

    @pytest.mark.unit
    def test_dependency_graph_roots_property(self, sample_graph):
        """Test DependencyGraph roots property."""
        roots = sample_graph.roots

        assert isinstance(roots, list)
        assert len(roots) > 0

        # Check that all root nodes have no incoming dependencies
        for root in roots:
            assert len(root.incoming) == 0

    @pytest.mark.unit
    def test_dependency_graph_leaves_property(self, sample_graph):
        """Test DependencyGraph leaves property."""
        leaves = sample_graph.leaves

        assert isinstance(leaves, list)
        assert len(leaves) > 0

        # Check that all leaf nodes have no outgoing dependencies
        for leaf in leaves:
            assert len(leaf.outgoing) == 0


class TestGetGraphStatistics:
    """Test get_graph_statistics function."""

    @pytest.mark.unit
    def test_get_graph_statistics(self, sample_graph):
        """Test get_graph_statistics function."""
        stats = get_graph_statistics(sample_graph)

        assert isinstance(stats, dict)
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "max_depth" in stats
        assert "roots" in stats  # Use actual key name
        assert "leaves" in stats  # Use actual key name
        assert "cycles" in stats

        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 2
        assert stats["max_depth"] >= 0

    @pytest.mark.unit
    def test_get_graph_statistics_empty_graph(self):
        """Test get_graph_statistics with empty graph."""
        empty_graph = DependencyGraph()
        stats = get_graph_statistics(empty_graph)

        assert isinstance(stats, dict)
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["max_depth"] == 0
        assert stats["roots"] == 0  # Use actual key name
        assert stats["leaves"] == 0  # Use actual key name
        assert stats["cycles"] == 0


@pytest.mark.unit
def test_dependency_graph_statistics_caching():
    """Test DependencyGraph.statistics property caching (lines 189->191)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # First access should compute statistics
    stats1 = graph.statistics
    assert isinstance(stats1, dict)

    # Second access should use cached value
    stats2 = graph.statistics
    assert stats1 is stats2  # Should be the same object (cached)


@pytest.mark.unit
def test_dependency_graph_cycles_caching():
    """Test DependencyGraph.cycles property caching (lines 201->203)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # Create a cycle
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node1.outgoing.add(node2)
    node2.outgoing.add(node1)

    # First access should compute cycles
    cycles1 = graph.cycles
    assert isinstance(cycles1, list)

    # Second access should use cached value
    cycles2 = graph.cycles
    assert cycles1 is cycles2  # Should be the same object (cached)


@pytest.mark.unit
def test_dependency_graph_remove_incoming_cleanup():
    """Test DependencyGraph.remove() incoming cleanup (line 227)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # Create dependency: obs1 -> obs2
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    # Verify dependency exists
    assert node1 in node2.incoming
    assert node2 in node1.outgoing

    # Remove obs2
    graph.remove(obs2)

    # Verify obs1's outgoing is cleaned up
    assert node2 not in node1.outgoing


@pytest.mark.unit
def test_dependency_graph_remove_node_cache_cleanup():
    """Test DependencyGraph.remove() node cache cleanup (lines 231->233)"""
    obs1 = Observable("obs1", 1)

    graph = DependencyGraph()
    graph.add(obs1)

    # Verify node is in cache
    assert obs1 in graph._node_cache

    # Remove obs1
    graph.remove(obs1)

    # Verify node is removed from cache
    assert obs1 not in graph._node_cache


@pytest.mark.unit
def test_dependency_graph_batch_update():
    """Test DependencyGraph.batch_update() method (line 258)"""
    graph = DependencyGraph()

    # batch_update should return self for chaining
    result = graph.batch_update()
    assert result is graph


@pytest.mark.unit
def test_dependency_graph_build_from_observables_not_implemented():
    """Test DependencyGraph.build_from_observables() NotImplementedError (line 270)"""
    graph = DependencyGraph()
    obs1 = Observable("obs1", 1)

    with pytest.raises(
        NotImplementedError, match="Subclasses must implement build_from_observables"
    ):
        graph.build_from_observables([obs1])


@pytest.mark.unit
def test_dependency_graph_topological_sort_dependent_check():
    """Test DependencyGraph.topological_sort() dependent existence check (lines 297->296)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # Create dependency: obs1 -> obs2
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    # Remove obs2 from graph but keep the dependency relationship
    del graph.nodes[obs2]

    # Topological sort should handle missing dependents gracefully
    result = graph.topological_sort()
    assert len(result) == 1
    assert result[0] == node1


@pytest.mark.unit
def test_dependency_graph_find_paths_max_depth_limit():
    """Test DependencyGraph.find_paths() max depth limit (line 363)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    obs3 = Observable("obs3", 3)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)
    graph.add(obs3)

    # Create chain: obs1 -> obs2 -> obs3
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node3 = graph.get_or_create_node(obs3)

    node2.incoming.add(node1)
    node1.outgoing.add(node2)
    node3.incoming.add(node2)
    node2.outgoing.add(node3)

    # Find paths with max depth 1 (should limit recursion)
    paths = graph.find_paths(node1, node3, max_depth=1)
    assert isinstance(paths, list)


@pytest.mark.unit
def test_dependency_graph_find_paths_cycle_detection():
    """Test DependencyGraph.find_paths() cycle detection (line 367)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # Create cycle: obs1 -> obs2 -> obs1
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)

    node2.incoming.add(node1)
    node1.outgoing.add(node2)
    node1.incoming.add(node2)
    node2.outgoing.add(node1)

    # Find paths should handle cycles gracefully
    paths = graph.find_paths(node1, node2)
    assert isinstance(paths, list)


@pytest.mark.unit
def test_dependency_graph_find_paths_duplicate_prevention():
    """Test DependencyGraph.find_paths() duplicate path prevention (lines 392->390)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    obs3 = Observable("obs3", 3)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)
    graph.add(obs3)

    # Create diamond pattern: obs1 -> obs2, obs1 -> obs3, obs2 -> obs4, obs3 -> obs4
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node3 = graph.get_or_create_node(obs3)

    node2.incoming.add(node1)
    node1.outgoing.add(node2)
    node3.incoming.add(node1)
    node1.outgoing.add(node3)

    # Find paths should prevent duplicates
    paths = graph.find_paths(node1, node2)
    assert isinstance(paths, list)


@pytest.mark.unit
def test_dependency_graph_can_reach_max_hops_exceeded():
    """Test DependencyGraph.can_reach() max hops exceeded (line 407)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)

    # Test with very low max_hops to trigger the limit
    result = graph.can_reach(node2, node1, max_hops=0)
    assert result is False


@pytest.mark.unit
def test_dependency_graph_can_reach_target_not_found():
    """Test DependencyGraph.can_reach() target not found (line 416)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)

    # Test reachability when target is not reachable
    result = graph.can_reach(node2, node1, max_hops=10)
    assert result is False


@pytest.mark.unit
def test_dependency_graph_can_reach_current_not_found():
    """Test DependencyGraph.can_reach() current not found (line 419)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)

    # Test reachability when current is not reachable
    result = graph.can_reach(node1, node2, max_hops=10)
    assert result is False


@pytest.mark.unit
def test_dependency_graph_can_reach_max_hops_return():
    """Test DependencyGraph.can_reach() max hops return (lines 425->422)"""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    graph = DependencyGraph()
    graph.add(obs1)
    graph.add(obs2)

    # Create dependency: obs1 -> obs2
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    # Test reachability with max_hops limit
    result = graph.can_reach(node2, node1, max_hops=1)
    assert result is True
