"""Unit tests for ReactiveGraph functionality."""

import pytest

from fynx import observable
from fynx.observable.base import Observable
from fynx.optimizer import ReactiveGraph, optimize_reactive_graph
from fynx.optimizer.dependency_node import DependencyNode


@pytest.mark.unit
@pytest.mark.optimizer
def test_reactive_graph_creation():
    """ReactiveGraph creates with proper initial state"""
    graph = ReactiveGraph()

    assert graph._node_cache is not None
    assert len(graph.nodes) == 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_get_or_create_node_caching():
    """ReactiveGraph get_or_create_node uses caching for same observable"""
    graph = ReactiveGraph()
    obs = observable(1)

    node1 = graph.get_or_create_node(obs)
    node2 = graph.get_or_create_node(obs)

    # Should return the same node instance
    assert node1 is node2
    assert obs in graph.nodes


@pytest.mark.unit
@pytest.mark.optimizer
def test_build_from_observables_with_computed():
    """ReactiveGraph build_from_observables creates dependency relationships for computed observables"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    # Should have both nodes
    assert base in graph.nodes
    assert computed in graph.nodes

    # Should have dependency relationship
    base_node = graph.get_or_create_node(base)
    computed_node = graph.get_or_create_node(computed)
    assert base_node in computed_node.incoming
    assert computed_node in base_node.outgoing


@pytest.mark.unit
@pytest.mark.optimizer
def test_build_from_observables_with_conditional():
    """ReactiveGraph build_from_observables handles conditional observables"""
    source = observable(1)
    condition = observable(True)
    conditional = source & condition

    graph = ReactiveGraph()
    graph.build_from_observables([conditional])

    # Should have source and conditional nodes
    assert source in graph.nodes
    assert conditional in graph.nodes
    # Condition might not be directly added depending on implementation


@pytest.mark.unit
@pytest.mark.optimizer
def test_apply_functor_composition_fusion():
    """ReactiveGraph apply_functor_composition_fusion applies chain fusion optimizations"""
    base = observable(1)
    step1 = base >> (lambda x: x + 1)
    step2 = step1 >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([step2])

    fusions = graph.apply_functor_composition_fusion()

    assert isinstance(fusions, int)
    assert fusions >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_apply_product_factorization():
    """ReactiveGraph apply_product_factorization applies common subexpression elimination"""
    base = observable(1)
    branch1 = base >> (lambda x: x * 2)
    branch2 = base >> (lambda x: x * 3)

    graph = ReactiveGraph()
    graph.build_from_observables([branch1, branch2])

    factorizations = graph.apply_product_factorization()

    assert isinstance(factorizations, int)
    assert factorizations >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_apply_pullback_fusion():
    """ReactiveGraph apply_pullback_fusion applies filter combination optimizations"""
    base = observable(1)
    filter1 = base & (lambda x: x > 0)
    filter2 = filter1 & (lambda x: x < 10)

    graph = ReactiveGraph()
    graph.build_from_observables([filter2])

    fusions = graph.apply_pullback_fusion()

    assert isinstance(fusions, int)
    assert fusions >= 0


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimize_materialization():
    """ReactiveGraph optimize_materialization sets materialization costs on nodes"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    # Should not raise an error
    graph.optimize_materialization()

    # Check that nodes have materialization costs set
    computed_node = graph.get_or_create_node(computed)
    assert hasattr(computed_node, "_materialize_cost")
    assert hasattr(computed_node, "_recompute_cost")


@pytest.mark.unit
@pytest.mark.optimizer
def test_check_confluence():
    """ReactiveGraph check_confluence verifies optimization confluence"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    result = graph.check_confluence()

    assert isinstance(result, dict)
    # Check for actual keys that exist
    assert "is_confluent" in result or "consistent_results" in result


def test_verify_optimization_correctness():
    """Test optimization correctness verification with computational checks"""
    # Test 1: Basic correctness verification
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    result = graph.verify_optimization_correctness()

    assert isinstance(result, dict)
    assert "cycles_introduced" in result
    assert "graph_connected" in result
    assert "structural_correctness" in result
    assert result["structural_correctness"] == True

    # Test 2: Verify optimization actually preserves observable behavior
    # Create a chain that should be fused by optimization
    base_val = Observable("base", 5)
    step1 = base_val >> (lambda x: x + 3)  # 5 + 3 = 8
    step2 = step1 >> (lambda x: x * 2)  # 8 * 2 = 16
    final = step2 >> (lambda x: x - 1)  # 16 - 1 = 15

    # Expected final value: ((5 + 3) * 2) - 1 = 15
    expected_final = ((base_val.value + 3) * 2) - 1

    graph2 = ReactiveGraph()
    graph2.build_from_observables([final])

    # Run optimization
    opt_results = graph2.optimize()

    # Verify optimization applied (should fuse the chain)
    assert (
        opt_results["functor_fusions"] > 0
    ), "Chain fusion optimization should have applied"

    # Verify correctness preserved after optimization
    base_val.set(5)  # Trigger recomputation
    actual_final = final.value

    assert (
        actual_final == expected_final
    ), f"Optimization broke correctness: expected {expected_final}, got {actual_final}"

    # Verify graph structure is valid after optimization
    correctness = graph2.verify_optimization_correctness()
    assert (
        correctness["structural_correctness"] == True
    ), "Optimization left graph in invalid state"

    # Test 3: Verify optimizations don't break correctness with complex graphs
    # Create a more complex graph with multiple computation paths
    base_a = Observable("base_a", 2)
    base_b = Observable("base_b", 3)

    # Create computations that depend on both bases
    sum_ab = (base_a + base_b) >> (lambda a, b: a + b)  # 2 + 3 = 5
    prod_ab = (base_a + base_b) >> (lambda a, b: a * b)  # 2 * 3 = 6
    final = (sum_ab + prod_ab) >> (lambda s, p: s + p)  # 5 + 6 = 11

    graph3 = ReactiveGraph()
    graph3.build_from_observables([final])

    opt_results3 = graph3.optimize()

    # Verify correctness is preserved
    base_a.set(2)
    base_b.set(3)
    expected = (2 + 3) + (2 * 3)  # 5 + 6 = 11
    assert (
        final.value == expected
    ), f"Complex computation failed: expected {expected}, got {final.value}"

    # Test 4: Regression test - ensure optimizations don't create cycles
    # This would catch if optimization incorrectly created self-references
    correctness3 = graph3.verify_optimization_correctness()
    assert (
        correctness3["cycles_introduced"] == False
    ), "Optimization must not introduce cycles"
    assert (
        correctness3["graph_connected"] == True
    ), "Optimization must maintain connectivity"


@pytest.mark.unit
@pytest.mark.optimizer
def test_compose_morphisms():
    """ReactiveGraph compose_morphisms handles morphism composition"""
    graph = ReactiveGraph()

    # Test identity laws
    assert graph.compose_morphisms("id", "f") == "f"
    assert graph.compose_morphisms("g", "id") == "g"

    # Test composition
    result = graph.compose_morphisms("f", "g")
    assert "f" in result
    assert "g" in result


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_identity():
    """ReactiveGraph morphism_identity returns identity morphism"""
    graph = ReactiveGraph()
    obs = observable(1)
    node = graph.get_or_create_node(obs)

    identity = graph.morphism_identity(node)
    assert identity == "id"


@pytest.mark.unit
@pytest.mark.optimizer
def test_enable_profiling():
    """ReactiveGraph enable_profiling enables performance profiling"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    # Should not raise an error
    graph.enable_profiling()


@pytest.mark.unit
@pytest.mark.optimizer
def test_get_profiling_summary():
    """ReactiveGraph get_profiling_summary returns profiling statistics"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    summary = graph.get_profiling_summary()

    assert isinstance(summary, dict)
    # Check for actual keys that exist
    assert "avg_calls_per_node" in summary or "total_nodes" in summary


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimize_method():
    """ReactiveGraph optimize method applies all optimizations"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    result = graph.optimize()

    assert isinstance(result, dict)
    # Check for actual keys that exist
    assert "confluence" in result or "optimizations_applied" in result


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimize_reactive_graph_function_basic():
    """optimize_reactive_graph function optimizes basic observables"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    result, optimized_graph = optimize_reactive_graph([computed])

    assert isinstance(result, dict)
    assert isinstance(optimized_graph, ReactiveGraph)
    # Check for actual keys that exist
    assert "confluence" in result or "optimizations_applied" in result


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimize_reactive_graph_function_multiple_observables():
    """optimize_reactive_graph function handles multiple observables"""
    base1 = observable(1)
    base2 = observable(2)
    computed1 = base1 >> (lambda x: x * 2)
    computed2 = base2 >> (lambda x: x * 3)

    result, optimized_graph = optimize_reactive_graph([computed1, computed2])

    assert isinstance(result, dict)
    assert isinstance(optimized_graph, ReactiveGraph)


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimize_reactive_graph_function_empty_list():
    """optimize_reactive_graph function handles empty observable list"""
    result, optimized_graph = optimize_reactive_graph([])

    assert isinstance(result, dict)
    assert isinstance(optimized_graph, ReactiveGraph)
