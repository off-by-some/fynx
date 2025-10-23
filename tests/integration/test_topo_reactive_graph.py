"""Comprehensive tests for TopoReactiveGraph system - rxpy-style testing.

Tests cover:
- Creations: Node creation patterns and validation
- Updates: Value propagation and computation
- Chain propagation: Long dependency chains
- Fanout: One-to-many relationships
- Fanin: Many-to-one relationships
- Subscriptions: Subscriber lifecycle management
- Error handling: Exception propagation and recovery
- Concurrency: Batch updates and async operations
- Edge cases: Boundary conditions and error states
"""

import asyncio
import time
from typing import Any, Dict, List

import pytest

from prototype import Subscriber, TopoReactiveGraph


class TestTopoReactiveGraphCreations:
    """Test node creation patterns and validation."""

    def test_create_empty_graph(self):
        """Graph starts empty with no nodes."""
        graph = TopoReactiveGraph()
        assert len(graph.nodes) == 0
        assert len(graph.topo_order) == 0
        assert len(graph.dirty_nodes) == 0

    def test_add_input_node(self):
        """Input nodes store values without computation."""
        graph = TopoReactiveGraph()
        graph.add_input("price", 100.0)

        assert "price" in graph.nodes
        assert graph.get_value("price") == 100.0
        assert graph.nodes["price"].deps == []
        assert graph.nodes["price"].compute_fn({"": None}) == 100.0

    def test_add_computed_node(self):
        """Computed nodes have dependencies and compute functions."""
        graph = TopoReactiveGraph()
        graph.add_input("price", 100.0)
        graph.add_input("quantity", 5)

        graph.add_node(
            "total",
            lambda deps: deps["price"] * deps["quantity"],
            ["price", "quantity"],
        )

        assert "total" in graph.nodes
        assert graph.nodes["total"].deps == ["price", "quantity"]
        # Should be marked as dirty initially
        assert "total" in graph.dirty_nodes

    def test_add_node_with_initial_value(self):
        """Computed nodes can have initial values."""
        graph = TopoReactiveGraph()
        graph.add_node("counter", lambda deps: len(deps), [], 0)

        assert graph.get_value("counter") == 0

    def test_duplicate_node_raises_error(self):
        """Adding duplicate node names raises ValueError."""
        graph = TopoReactiveGraph()
        graph.add_input("test", 1)

        with pytest.raises(ValueError, match="Node test already exists"):
            graph.add_input("test", 2)

    def test_invalid_dependency_raises_error(self):
        """Referencing non-existent dependencies raises error during computation."""
        graph = TopoReactiveGraph()
        graph.add_node("bad", lambda deps: deps["missing"], ["missing"])

        # Should fail during computation when dependency is accessed
        with pytest.raises(KeyError):
            graph.get_value("bad")

    def test_circular_dependency_detection(self):
        """Circular dependencies are detected during topological sort."""
        graph = TopoReactiveGraph()

        # Create a circular dependency: A -> B -> A
        graph.add_node("A", lambda deps: deps.get("B", 0) + 1, ["B"])

        # Adding the second node should detect the cycle
        with pytest.raises(ValueError, match="Graph contains cycles"):
            graph.add_node("B", lambda deps: deps.get("A", 0) + 1, ["A"])


class TestTopoReactiveGraphUpdates:
    """Test value propagation and computation updates."""

    def test_input_node_updates(self):
        """Input node values can be updated."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        assert graph.get_value("value") == 10

        graph.set_value("value", 20)
        assert graph.get_value("value") == 20

    def test_computed_node_updates_on_dependency_change(self):
        """Computed nodes update when dependencies change."""
        graph = TopoReactiveGraph()
        graph.add_input("price", 100)
        graph.add_input("quantity", 2)
        graph.add_node(
            "total",
            lambda deps: deps["price"] * deps["quantity"],
            ["price", "quantity"],
        )

        # Initial computation
        asyncio.run(graph.flush_async())
        assert graph.get_value("total") == 200

        # Update dependency
        graph.set_value("price", 150)
        asyncio.run(graph.flush_async())
        assert graph.get_value("total") == 300

    def test_no_update_when_value_unchanged(self):
        """Nodes don't update when set to same value."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        # Set to same value - should not trigger updates
        graph.set_value("value", 10)
        assert len(graph.dirty_nodes) == 0

    def test_batch_updates(self):
        """Multiple updates are batched efficiently."""
        graph = TopoReactiveGraph(batch_size=10)

        # Add many input nodes
        for i in range(20):
            graph.add_input(f"input_{i}", i)

        # Set values (should trigger batching when batch_size reached)
        for i in range(20):
            graph.set_value(f"input_{i}", i * 2)

        # Should have batched updates
        assert len(graph.dirty_nodes) <= 20

    def test_lazy_computation(self):
        """Values are computed lazily on demand."""
        graph = TopoReactiveGraph()
        graph.add_input("a", 1)
        graph.add_input("b", 2)
        graph.add_node("sum", lambda deps: deps["a"] + deps["b"], ["a", "b"])

        # Node should be dirty but not computed yet
        assert "sum" in graph.dirty_nodes
        assert graph.nodes["sum"].value is None

        # First access should compute
        result = graph.get_value("sum")
        assert result == 3
        assert "sum" not in graph.dirty_nodes


class TestTopoReactiveGraphChainPropagation:
    """Test long dependency chains and propagation."""

    def test_long_dependency_chain(self):
        """Long chains of dependencies propagate correctly."""
        graph = TopoReactiveGraph()

        # Create a chain: input -> step1 -> step2 -> ... -> step10
        graph.add_input("input", 1)

        for i in range(1, 11):
            prev = "input" if i == 1 else f"step{i-1}"
            graph.add_node(f"step{i}", lambda deps, prev=prev: deps[prev] * 2, [prev])

        asyncio.run(graph.flush_async())

        # Each step should double the value
        expected = 1
        for i in range(1, 11):
            expected *= 2
            assert graph.get_value(f"step{i}") == expected

    def test_chain_update_propagation(self):
        """Updates propagate through entire dependency chain."""
        graph = TopoReactiveGraph()

        # Create chain: a -> b -> c -> d
        graph.add_input("a", 1)
        graph.add_node("b", lambda deps: deps["a"] + 1, ["a"])
        graph.add_node("c", lambda deps: deps["b"] * 2, ["b"])
        graph.add_node("d", lambda deps: deps["c"] + 10, ["c"])

        asyncio.run(graph.flush_async())
        assert graph.get_value("d") == 14  # ((1+1)*2)+10 = 14

        # Change input and verify propagation
        graph.set_value("a", 5)
        asyncio.run(graph.flush_async())
        assert graph.get_value("d") == 22  # ((5+1)*2)+10 = 22

    def test_multiple_independent_chains(self):
        """Multiple independent chains work simultaneously."""
        graph = TopoReactiveGraph()

        # Chain 1: x1 -> y1 -> z1
        graph.add_input("x1", 10)
        graph.add_node("y1", lambda deps: deps["x1"] * 2, ["x1"])
        graph.add_node("z1", lambda deps: deps["y1"] + 5, ["y1"])

        # Chain 2: x2 -> y2 -> z2
        graph.add_input("x2", 20)
        graph.add_node("y2", lambda deps: deps["x2"] * 3, ["x2"])
        graph.add_node("z2", lambda deps: deps["y2"] + 7, ["y2"])

        asyncio.run(graph.flush_async())

        assert graph.get_value("z1") == 25  # (10*2)+5 = 25
        assert graph.get_value("z2") == 67  # (20*3)+7 = 67


class TestTopoReactiveGraphFanout:
    """Test one-to-many relationships (fanout)."""

    def test_single_input_multiple_outputs(self):
        """One input feeds multiple computed nodes."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        # Multiple computations from same input
        graph.add_node("double", lambda deps: deps["value"] * 2, ["value"])
        graph.add_node("square", lambda deps: deps["value"] ** 2, ["value"])
        graph.add_node("negate", lambda deps: -deps["value"], ["value"])

        asyncio.run(graph.flush_async())

        assert graph.get_value("double") == 20
        assert graph.get_value("square") == 100
        assert graph.get_value("negate") == -10

    def test_fanout_updates_all_dependents(self):
        """Changing input updates all dependent nodes."""
        graph = TopoReactiveGraph()
        graph.add_input("base", 5)

        graph.add_node("add_10", lambda deps: deps["base"] + 10, ["base"])
        graph.add_node("multiply_3", lambda deps: deps["base"] * 3, ["base"])
        graph.add_node("power_2", lambda deps: deps["base"] ** 2, ["base"])

        asyncio.run(graph.flush_async())

        # Change base value
        graph.set_value("base", 8)
        asyncio.run(graph.flush_async())

        assert graph.get_value("add_10") == 18
        assert graph.get_value("multiply_3") == 24
        assert graph.get_value("power_2") == 64

    def test_deep_fanout_tree(self):
        """Test complex fanout tree structure."""
        graph = TopoReactiveGraph()
        graph.add_input("root", 2)

        # Level 1
        graph.add_node("level1_a", lambda deps: deps["root"] + 1, ["root"])
        graph.add_node("level1_b", lambda deps: deps["root"] * 3, ["root"])

        # Level 2
        graph.add_node("level2_a1", lambda deps: deps["level1_a"] * 2, ["level1_a"])
        graph.add_node("level2_a2", lambda deps: deps["level1_a"] + 5, ["level1_a"])
        graph.add_node("level2_b1", lambda deps: deps["level1_b"] - 1, ["level1_b"])
        graph.add_node("level2_b2", lambda deps: deps["level1_b"] / 2, ["level1_b"])

        # Level 3
        graph.add_node(
            "level3_final",
            lambda deps: deps["level2_a1"] + deps["level2_b1"],
            ["level2_a1", "level2_b1"],
        )

        asyncio.run(graph.flush_async())

        # Verify all levels computed correctly
        assert graph.get_value("level1_a") == 3  # 2 + 1
        assert graph.get_value("level1_b") == 6  # 2 * 3
        assert graph.get_value("level2_a1") == 6  # 3 * 2
        assert graph.get_value("level2_a2") == 8  # 3 + 5
        assert graph.get_value("level2_b1") == 5  # 6 - 1
        assert graph.get_value("level2_b2") == 3  # 6 / 2
        assert graph.get_value("level3_final") == 11  # 6 + 5


class TestTopoReactiveGraphFanin:
    """Test many-to-one relationships (fanin)."""

    def test_multiple_inputs_single_output(self):
        """Multiple inputs feed into single computed node."""
        graph = TopoReactiveGraph()
        graph.add_input("a", 1)
        graph.add_input("b", 2)
        graph.add_input("c", 3)

        graph.add_node(
            "sum", lambda deps: deps["a"] + deps["b"] + deps["c"], ["a", "b", "c"]
        )

        asyncio.run(graph.flush_async())
        assert graph.get_value("sum") == 6

    def test_fanin_updates_on_any_input_change(self):
        """Fanin node updates when any input changes."""
        graph = TopoReactiveGraph()
        graph.add_input("width", 10)
        graph.add_input("height", 5)
        graph.add_input("depth", 2)

        graph.add_node(
            "volume",
            lambda deps: deps["width"] * deps["height"] * deps["depth"],
            ["width", "height", "depth"],
        )

        asyncio.run(graph.flush_async())
        assert graph.get_value("volume") == 100

        # Change each input and verify volume updates
        graph.set_value("width", 20)
        asyncio.run(graph.flush_async())
        assert graph.get_value("volume") == 200

        graph.set_value("height", 10)
        asyncio.run(graph.flush_async())
        assert graph.get_value("volume") == 400

        graph.set_value("depth", 5)
        asyncio.run(graph.flush_async())
        assert graph.get_value("volume") == 1000

    def test_complex_fanin_patterns(self):
        """Complex fanin patterns with overlapping dependencies."""
        graph = TopoReactiveGraph()

        # Inputs
        graph.add_input("x", 2)
        graph.add_input("y", 3)
        graph.add_input("z", 4)

        # Intermediate computations
        graph.add_node("xy", lambda deps: deps["x"] * deps["y"], ["x", "y"])
        graph.add_node("yz", lambda deps: deps["y"] * deps["z"], ["y", "z"])
        graph.add_node("xz", lambda deps: deps["x"] * deps["z"], ["x", "z"])

        # Fanin to final computation
        graph.add_node(
            "result",
            lambda deps: deps["xy"] + deps["yz"] + deps["xz"],
            ["xy", "yz", "xz"],
        )

        asyncio.run(graph.flush_async())

        # xy = 2*3 = 6, yz = 3*4 = 12, xz = 2*4 = 8
        # result = 6 + 12 + 8 = 26
        assert graph.get_value("result") == 26


class TestTopoReactiveGraphSubscriptions:
    """Test subscriber lifecycle management."""

    def test_subscribe_to_node(self):
        """Can subscribe to node changes."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        calls = []

        def subscriber(new_value):
            calls.append(new_value)

        unsubscribe = graph.subscribe("value", subscriber)
        assert len(graph.get_subscribers("value")) == 1

        # Trigger notification
        graph.set_value("value", 20)
        assert calls == [20]

    def test_unsubscribe_function(self):
        """Unsubscribe function removes subscriber."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        calls = []

        def subscriber(new_value):
            calls.append(new_value)

        unsubscribe = graph.subscribe("value", subscriber)
        unsubscribe()

        assert len(graph.get_subscribers("value")) == 0

        # Should not get notifications after unsubscribe
        graph.set_value("value", 30)
        assert calls == []

    def test_multiple_subscribers(self):
        """Multiple subscribers can be attached to same node."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        calls1 = []
        calls2 = []

        def sub1(v):
            calls1.append(v)

        def sub2(v):
            calls2.append(v)

        graph.subscribe("value", sub1)
        graph.subscribe("value", sub2)

        assert len(graph.get_subscribers("value")) == 2

        graph.set_value("value", 15)
        assert calls1 == [15]
        assert calls2 == [15]

    def test_async_subscribers(self):
        """Async subscribers are supported."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        async def async_subscriber(new_value):
            await asyncio.sleep(0.001)  # Simulate async work
            async_subscriber.called = True
            async_subscriber.value = new_value

        async_subscriber.called = False

        graph.subscribe("value", async_subscriber)

        # Run async update
        async def test():
            graph.set_value("value", 25)
            # Give time for async subscriber to complete
            await asyncio.sleep(0.01)

        asyncio.run(test())
        assert async_subscriber.called
        assert async_subscriber.value == 25

    def test_computed_node_subscriptions(self):
        """Computed nodes notify subscribers on updates."""
        graph = TopoReactiveGraph()
        graph.add_input("a", 1)
        graph.add_input("b", 2)
        graph.add_node("sum", lambda deps: deps["a"] + deps["b"], ["a", "b"])

        calls = []

        def subscriber(v):
            calls.append(v)

        graph.subscribe("sum", subscriber)

        # Initial computation
        asyncio.run(graph.flush_async())
        assert calls == [3]

        # Update dependency
        graph.set_value("a", 5)
        asyncio.run(graph.flush_async())
        assert calls == [3, 7]

    def test_subscriber_error_handling(self):
        """Subscriber exceptions don't break the system."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        def good_subscriber(v):
            good_subscriber.called = True

        def bad_subscriber(v):
            raise ValueError("Subscriber error")

        good_subscriber.called = False

        graph.subscribe("value", bad_subscriber)
        graph.subscribe("value", good_subscriber)

        # Should not raise exception from bad subscriber
        graph.set_value("value", 20)
        assert good_subscriber.called

    def test_backward_compatibility_on_change(self):
        """Old on_change method still works."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        calls = []

        def callback(v):
            calls.append(v)

        graph.on_change("value", callback)

        graph.set_value("value", 15)
        assert calls == [15]


class TestTopoReactiveGraphErrorHandling:
    """Test error handling and propagation."""

    def test_computation_error_in_node(self):
        """Errors in computation functions are handled gracefully."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        # Computation that raises error
        graph.add_node("bad", lambda deps: 1 / 0, ["value"])

        # Should handle error gracefully
        with pytest.raises(ZeroDivisionError):
            asyncio.run(graph.flush_async())

    def test_subscriber_error_doesnt_break_system(self):
        """Subscriber errors don't prevent other subscribers from running."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 10)

        def good_sub(v):
            good_sub.called = True

        def bad_sub(v):
            raise RuntimeError("Bad subscriber")

        good_sub.called = False

        graph.subscribe("value", good_sub)
        graph.subscribe("value", bad_sub)

        # Good subscriber should still be called despite bad one
        graph.set_value("value", 20)
        assert good_sub.called

    def test_invalid_node_access(self):
        """Accessing non-existent nodes raises KeyError."""
        graph = TopoReactiveGraph()

        with pytest.raises(KeyError, match="Node nonexistent not found"):
            graph.get_value("nonexistent")

    def test_invalid_subscription(self):
        """Subscribing to non-existent nodes raises ValueError."""
        graph = TopoReactiveGraph()

        def subscriber(v):
            pass

        with pytest.raises(ValueError, match="Node nonexistent not found"):
            graph.subscribe("nonexistent", subscriber)


class TestTopoReactiveGraphConcurrency:
    """Test concurrent updates and batching."""

    @pytest.mark.asyncio
    async def test_async_batch_updates(self):
        """Async batch updates work correctly."""
        graph = TopoReactiveGraph(batch_size=5)

        # Add many nodes
        for i in range(10):
            graph.add_input(f"input_{i}", i)

        # Trigger batch updates
        for i in range(10):
            graph.set_value(f"input_{i}", i * 2)

        # Wait for batch processing
        await graph.flush_async()

        # Verify all updates processed
        for i in range(10):
            assert graph.get_value(f"input_{i}") == i * 2

    @pytest.mark.asyncio
    async def test_concurrent_computation(self):
        """Concurrent computations work correctly."""
        graph = TopoReactiveGraph(max_workers=4)

        # Create computation chain
        graph.add_input("start", 1)

        for i in range(1, 6):
            prev = "start" if i == 1 else f"step{i-1}"
            graph.add_node(f"step{i}", lambda deps, i=i: deps[prev] + i, [prev])

        await graph.flush_async()

        # Each step adds its index
        expected = 1 + sum(range(1, 6))  # 1 + (1+2+3+4+5) = 16
        assert graph.get_value("step5") == expected

    def test_performance_large_graph(self):
        """Large graphs perform reasonably well."""
        graph = TopoReactiveGraph(batch_size=1000)

        # Create large fanout graph
        graph.add_input("root", 1)

        # 100 computed nodes all depending on root
        for i in range(100):
            graph.add_node(f"node_{i}", lambda deps, i=i: deps["root"] + i, ["root"])

        start_time = time.time()
        asyncio.run(graph.flush_async())
        end_time = time.time()

        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0

        # Verify all nodes computed correctly
        for i in range(100):
            assert graph.get_value(f"node_{i}") == 1 + i


class TestTopoReactiveGraphEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dependencies(self):
        """Nodes with no dependencies work correctly."""
        graph = TopoReactiveGraph()

        # Node that doesn't depend on anything
        graph.add_node("constant", lambda deps: 42, [])

        asyncio.run(graph.flush_async())
        assert graph.get_value("constant") == 42

    def test_dependency_on_itself_not_allowed(self):
        """Nodes cannot depend on themselves."""
        graph = TopoReactiveGraph()

        # This would create a cycle if allowed
        with pytest.raises(ValueError, match="Graph contains cycles"):
            graph.add_node(
                "self_ref", lambda deps: deps.get("self_ref", 0) + 1, ["self_ref"]
            )

    def test_large_values(self):
        """Large numeric values are handled correctly."""
        graph = TopoReactiveGraph()
        graph.add_input("big_num", 10**18)
        graph.add_node("doubled", lambda deps: deps["big_num"] * 2, ["big_num"])

        asyncio.run(graph.flush_async())
        assert graph.get_value("doubled") == 2 * 10**18

    def test_none_values(self):
        """None values are handled correctly."""
        graph = TopoReactiveGraph()
        graph.add_input("none_val", None)
        graph.add_node("is_none", lambda deps: deps["none_val"] is None, ["none_val"])

        asyncio.run(graph.flush_async())
        assert graph.get_value("is_none") is True

    def test_complex_data_structures(self):
        """Complex data structures work as values."""
        graph = TopoReactiveGraph()

        complex_data = {"nested": {"value": [1, 2, 3]}}
        graph.add_input("data", complex_data)

        graph.add_node(
            "processed",
            lambda deps: {
                "sum": sum(deps["data"]["nested"]["value"]),
                "count": len(deps["data"]["nested"]["value"]),
            },
            ["data"],
        )

        asyncio.run(graph.flush_async())
        result = graph.get_value("processed")
        assert result["sum"] == 6
        assert result["count"] == 3

    def test_zero_dependencies_computed_nodes(self):
        """Computed nodes with empty dependency lists."""
        graph = TopoReactiveGraph()

        # Pure computation nodes
        graph.add_node("pi", lambda deps: 3.14159, [])
        graph.add_node("e", lambda deps: 2.71828, [])

        asyncio.run(graph.flush_async())
        assert abs(graph.get_value("pi") - 3.14159) < 0.001
        assert abs(graph.get_value("e") - 2.71828) < 0.001

    def test_repeated_flushes(self):
        """Multiple flushes don't cause issues."""
        graph = TopoReactiveGraph()
        graph.add_input("value", 1)
        graph.add_node("double", lambda deps: deps["value"] * 2, ["value"])

        # Multiple flushes should be safe
        asyncio.run(graph.flush_async())
        assert graph.get_value("double") == 2

        asyncio.run(graph.flush_async())  # Second flush
        assert graph.get_value("double") == 2

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        graph = TopoReactiveGraph()

        # Initially empty
        stats = graph.stats()
        assert stats["node_count"] == 0
        assert stats["subscriber_count"] == 0

        # Add nodes and subscribers
        graph.add_input("a", 1)
        graph.add_input("b", 2)
        graph.add_node("sum", lambda deps: deps["a"] + deps["b"], ["a", "b"])

        graph.subscribe("sum", lambda v: None)
        graph.subscribe("a", lambda v: None)

        stats = graph.stats()
        assert stats["node_count"] == 3
        assert stats["subscriber_count"] == 2
