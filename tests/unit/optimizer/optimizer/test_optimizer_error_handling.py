"""Unit tests for optimizer edge cases and error conditions."""

import pytest

from fynx import observable
from fynx.optimizer import DependencyNode, OptimizationContext, ReactiveGraph


@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.edge_case
def test_dependency_node_with_no_profiling_data():
    """DependencyNode handles missing profiling data gracefully"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Should handle missing profiling data gracefully
    freq = node.update_frequency_estimate
    assert freq > 0

    cost = node.computation_cost
    assert cost >= 0


@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.edge_case
def test_reactive_graph_with_isolated_nodes():
    """ReactiveGraph handles isolated nodes correctly"""
    obs1 = observable(1)
    obs2 = observable(2)

    graph = ReactiveGraph()
    graph.build_from_observables([obs1, obs2])

    # Should handle isolated nodes
    assert len(graph.nodes) == 2


@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.edge_case
def test_optimization_context_thread_safety():
    """OptimizationContext creates separate instances for thread safety"""
    context1 = OptimizationContext()
    context2 = OptimizationContext()

    # Should be different instances
    assert context1 is not context2
    assert context1.optimizer is not context2.optimizer


@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.edge_case
def test_reactive_graph_with_cycles():
    """ReactiveGraph handles cycles gracefully"""
    obs1 = observable(1)
    obs2 = observable(2)

    graph = ReactiveGraph()
    graph.build_from_observables([obs1, obs2])

    # Manually create a cycle
    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)
    node1.incoming.add(node2)
    node2.incoming.add(node1)
    node1.outgoing.add(node2)
    node2.outgoing.add(node1)

    # Should handle cycles gracefully
    result = graph.check_confluence()
    assert isinstance(result, dict)


@pytest.mark.unit
@pytest.mark.optimizer
@pytest.mark.edge_case
def test_profiling_with_zero_execution_times():
    """DependencyNode profiling handles zero execution times"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Record zero execution times
    node.record_execution_time(0.0)
    node.record_execution_time(0.0)

    # Should handle zero times gracefully
    cost = node.computation_cost
    assert cost >= 0
