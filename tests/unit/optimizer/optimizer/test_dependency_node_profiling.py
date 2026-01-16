"""Unit tests for DependencyNode profiling and cost estimation functionality."""

import time

import pytest

from fynx import observable
from fynx.optimizer import DependencyNode


@pytest.mark.unit
@pytest.mark.optimizer
def test_record_execution_time_updates_profiling_data():
    """DependencyNode record_execution_time updates profiling data correctly"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Record some execution times
    node.record_execution_time(0.1)
    node.record_execution_time(0.2)
    node.record_execution_time(0.3)

    assert len(node.profiling_data["execution_times"]) == 3
    assert node.profiling_data["execution_times"] == [0.1, 0.2, 0.3]
    assert (
        abs(node.profiling_data["avg_execution_time"] - 0.2) < 1e-10
    )  # Handle floating point precision
    assert node.profiling_data["call_count"] == 3


@pytest.mark.unit
@pytest.mark.optimizer
def test_record_execution_time_truncates_old_samples():
    """DependencyNode record_execution_time keeps only recent samples"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Add more than max_samples (100) execution times
    for i in range(150):
        node.record_execution_time(0.1)

    # Should keep only recent samples (scaled with connectivity)
    expected_samples = min(200, max(50, len(node.incoming) + len(node.outgoing) * 10))
    assert len(node.profiling_data["execution_times"]) == expected_samples
    assert node.profiling_data["call_count"] == 150


@pytest.mark.unit
@pytest.mark.optimizer
def test_update_frequency_estimate_with_profiling_data():
    """DependencyNode update_frequency_estimate uses profiling data when available"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Set up profiling data
    node.profiling_data["call_count"] = 10
    node.profiling_data["last_updated"] = time.time() - 1.0  # 1 second ago

    freq = node.update_frequency_estimate
    assert freq > 0
    assert freq <= 100.0  # Should be capped at 100


@pytest.mark.unit
@pytest.mark.optimizer
def test_update_frequency_estimate_fallback_to_graph_based():
    """DependencyNode update_frequency_estimate falls back to graph-based estimation"""
    obs1 = observable(1)
    obs2 = observable(2)
    node1 = DependencyNode(obs1)
    node2 = DependencyNode(obs2)

    # Set up dependency: node1 -> node2
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    # No profiling data, should use graph-based estimation
    freq = node2.update_frequency_estimate
    assert freq > 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_computation_cost_estimation():
    """DependencyNode computation_cost provides cost estimation"""
    obs = observable(1)
    node = DependencyNode(obs)

    # Set up some profiling data
    node.record_execution_time(0.1)
    node.record_execution_time(0.2)

    cost = node.computation_cost
    assert cost > 0
    assert isinstance(cost, float)


@pytest.mark.unit
@pytest.mark.optimizer
def test_compute_monoidal_cost_with_materialized_set():
    """DependencyNode compute_monoidal_cost calculates cost with materialized dependencies"""
    obs1 = observable(1)
    obs2 = observable(2)
    node1 = DependencyNode(obs1)
    node2 = DependencyNode(obs2)

    # Set up dependency
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    # Test with materialized set
    materialized = {node1}
    cost = node2.compute_monoidal_cost(materialized)
    assert cost >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_compute_monoidal_cost_without_materialized_set():
    """DependencyNode compute_monoidal_cost calculates cost without materialized dependencies"""
    obs = observable(1)
    node = DependencyNode(obs)

    cost = node.compute_monoidal_cost()
    assert cost >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_compute_sharing_penalty():
    """DependencyNode compute_sharing_penalty calculates penalty for shared dependencies"""
    obs1 = observable(1)
    obs2 = observable(2)
    obs3 = observable(3)
    node1 = DependencyNode(obs1)
    node2 = DependencyNode(obs2)
    node3 = DependencyNode(obs3)

    # Set up sharing: node1 -> {node2, node3}
    node2.incoming.add(node1)
    node3.incoming.add(node1)
    node1.outgoing.add(node2)
    node1.outgoing.add(node3)

    penalty = node1.compute_sharing_penalty()
    assert penalty >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_compute_cost_integration():
    """DependencyNode compute_cost method provides integrated cost calculation"""
    obs1 = observable(1)
    obs2 = observable(2)
    node1 = DependencyNode(obs1)
    node2 = DependencyNode(obs2)

    # Set up dependency
    node2.incoming.add(node1)
    node1.outgoing.add(node2)

    cost = node2.compute_cost()
    assert cost >= 0
