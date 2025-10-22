"""
Tests for IncrementalTopoSort cycle detection algorithm.
"""

import pytest

from fynx.util.cycle_detector import IncrementalTopoSort


class TestIncrementalTopoSort:
    """Test suite for IncrementalTopoSort cycle detection."""

    def test_empty_graph(self):
        """Empty graph has no cycles and no nodes."""
        detector = IncrementalTopoSort()
        assert len(detector) == 0
        assert not detector.has_cycle()
        assert detector.topological_sort() == []

    def test_single_node(self):
        """Single node graph has no cycles."""
        detector = IncrementalTopoSort()
        detector.add_node("A")

        assert len(detector) == 1
        assert "A" in detector
        assert not detector.has_cycle()
        assert detector.topological_sort() == ["A"]

    def test_simple_chain(self):
        """Linear chain A -> B -> C has no cycles."""
        detector = IncrementalTopoSort()

        detector.add_edge("A", "B")  # B depends on A
        detector.add_edge("B", "C")  # C depends on B

        assert len(detector) == 3
        assert not detector.has_cycle()
        assert detector.topological_sort() == ["A", "B", "C"]

    def test_simple_cycle(self):
        """Simple cycle A -> B -> C -> A should be detected."""
        detector = IncrementalTopoSort()

        detector.add_edge("A", "B")
        detector.add_edge("B", "C")

        # Adding C -> A should create a cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            detector.add_edge("C", "A")

    def test_self_loop(self):
        """Self-loop should be detected as a cycle."""
        detector = IncrementalTopoSort()

        with pytest.raises(ValueError, match="would create a cycle"):
            detector.add_edge("A", "A")

    def test_complex_cycle(self):
        """More complex cycle detection."""
        detector = IncrementalTopoSort()

        # Create a graph: A -> B -> C -> D -> B (cycle through D->B)
        detector.add_edge("A", "B")
        detector.add_edge("B", "C")
        detector.add_edge("C", "D")

        # This should create a cycle: D -> B (since B -> C -> D already exists)
        with pytest.raises(ValueError, match="would create a cycle"):
            detector.add_edge("D", "B")

    def test_multiple_components(self):
        """Graph with multiple disconnected components."""
        detector = IncrementalTopoSort()

        # Component 1: X -> Y
        detector.add_edge("X", "Y")

        # Component 2: P -> Q -> R
        detector.add_edge("P", "Q")
        detector.add_edge("Q", "R")

        assert len(detector) == 5
        assert not detector.has_cycle()

        # Topological sort should include all nodes
        topo = detector.topological_sort()
        assert len(topo) == 5
        assert set(topo) == {"X", "Y", "P", "Q", "R"}

        # Check ordering constraints
        assert topo.index("X") < topo.index("Y")
        assert topo.index("P") < topo.index("Q") < topo.index("R")

    def test_remove_edge(self):
        """Test edge removal."""
        detector = IncrementalTopoSort()

        detector.add_edge("A", "B")
        detector.add_edge("B", "C")

        assert detector.get_dependents("A") == {"B"}
        assert detector.get_dependencies("B") == {"A"}

        # Remove edge
        assert detector.remove_edge("A", "B")
        assert detector.get_dependents("A") == set()
        assert detector.get_dependencies("B") == set()

        # Try to remove non-existent edge
        assert not detector.remove_edge("A", "B")

    def test_remove_node(self):
        """Test node removal."""
        detector = IncrementalTopoSort()

        detector.add_edge("A", "B")
        detector.add_edge("B", "C")
        detector.add_edge("D", "B")  # D also points to B

        assert len(detector) == 4

        # Remove B
        detector.remove_node("B")

        assert len(detector) == 3
        assert "B" not in detector
        assert detector.get_dependents("A") == set()
        assert detector.get_dependencies("C") == set()
        assert detector.get_dependents("D") == set()

    def test_dependencies_and_dependents(self):
        """Test dependency relationship queries."""
        detector = IncrementalTopoSort()

        # A -> B -> C
        # D -> B
        detector.add_edge("A", "B")
        detector.add_edge("B", "C")
        detector.add_edge("D", "B")

        # Check dependents (who depends on me)
        assert detector.get_dependents("A") == {"B"}
        assert detector.get_dependents("B") == {"C"}
        assert detector.get_dependents("C") == set()
        assert detector.get_dependents("D") == {"B"}

        # Check dependencies (who do I depend on)
        assert detector.get_dependencies("A") == set()
        assert detector.get_dependencies("B") == {"A", "D"}
        assert detector.get_dependencies("C") == {"B"}
        assert detector.get_dependencies("D") == set()

    def test_cycle_detection_with_existing_graph(self):
        """Test cycle detection when graph already has cycles."""
        detector = IncrementalTopoSort()

        # Create a cycle manually by not using add_edge
        detector.add_node("A")
        detector.add_node("B")
        detector.add_node("C")

        # Manually add edges to create cycle (bypassing cycle detection)
        detector.graph["A"].add("B")
        detector.graph["B"].add("C")
        detector.graph["C"].add("A")

        # Update reverse graph and indegrees
        detector.reverse_graph["B"].add("A")
        detector.reverse_graph["C"].add("B")
        detector.reverse_graph["A"].add("C")

        detector.indegrees["B"] = 1
        detector.indegrees["C"] = 1
        detector.indegrees["A"] = 1

        assert detector.has_cycle()

        with pytest.raises(ValueError, match="contains cycles"):
            detector.topological_sort()

    def test_duplicate_edges(self):
        """Adding the same edge multiple times should not cause issues."""
        detector = IncrementalTopoSort()

        # Add same edge twice
        detector.add_edge("A", "B")
        detector.add_edge("A", "B")  # Should not raise

        assert detector.get_dependents("A") == {"B"}
        assert detector.get_dependencies("B") == {"A"}

    def test_string_representation(self):
        """Test string representation of the detector."""
        detector = IncrementalTopoSort()
        detector.add_edge("A", "B")
        detector.add_edge("B", "C")

        str_repr = str(detector)
        assert "IncrementalTopoSort" in str_repr
        assert "nodes=3" in str_repr
        assert "edges=2" in str_repr

    def test_clear(self):
        """Test clearing the graph."""
        detector = IncrementalTopoSort()

        detector.add_edge("A", "B")
        detector.add_edge("B", "C")
        assert len(detector) == 3

        detector.clear()
        assert len(detector) == 0
        assert not detector.has_cycle()

    def test_fan_in_fan_out(self):
        """Test fan-in and fan-out patterns."""
        detector = IncrementalTopoSort()

        # Fan-out: A -> B, A -> C, A -> D
        detector.add_edge("A", "B")
        detector.add_edge("A", "C")
        detector.add_edge("A", "D")

        # Fan-in: X -> E, Y -> E, Z -> E
        detector.add_edge("X", "E")
        detector.add_edge("Y", "E")
        detector.add_edge("Z", "E")

        assert len(detector) == 8
        assert not detector.has_cycle()

        topo = detector.topological_sort()
        assert topo.index("A") < topo.index("B")
        assert topo.index("A") < topo.index("C")
        assert topo.index("A") < topo.index("D")
        assert topo.index("X") < topo.index("E")
        assert topo.index("Y") < topo.index("E")
        assert topo.index("Z") < topo.index("E")
