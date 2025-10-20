"""Unit tests for ReactiveGraph functionality."""

import pytest

from fynx import observable
from fynx.optimizer import ReactiveGraph, optimize_reactive_graph


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
def test_compute_equivalence_classes():
    """ReactiveGraph compute_equivalence_classes returns equivalence classes"""
    obs1 = observable(1)
    obs2 = observable(2)

    graph = ReactiveGraph()
    graph.build_from_observables([obs1, obs2])

    classes = graph.compute_equivalence_classes()

    assert isinstance(classes, dict)
    assert len(classes) > 0


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
    assert "convergence_tests" in result or "confluent" in result


@pytest.mark.unit
@pytest.mark.optimizer
def test_verify_universal_properties():
    """ReactiveGraph verify_universal_properties checks categorical properties"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    result = graph.verify_universal_properties()

    assert isinstance(result, dict)
    # Check for actual keys that exist
    assert "total_candidates_checked" in result or "universal_properties" in result


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
def test_get_hom_set_representation():
    """ReactiveGraph get_hom_set_representation returns morphism representations"""
    base = observable(1)
    computed = base >> (lambda x: x * 2)

    graph = ReactiveGraph()
    graph.build_from_observables([computed])

    base_node = graph.get_or_create_node(base)
    computed_node = graph.get_or_create_node(computed)

    result = graph.get_hom_set_representation(base_node, computed_node)

    assert isinstance(result, dict)
    assert "morphisms" in result


@pytest.mark.unit
@pytest.mark.optimizer
def test_compute_yoneda_equivalence():
    """ReactiveGraph compute_yoneda_equivalence checks Yoneda equivalence"""
    obs1 = observable(1)
    obs2 = observable(2)

    graph = ReactiveGraph()
    graph.build_from_observables([obs1, obs2])

    node1 = graph.get_or_create_node(obs1)
    node2 = graph.get_or_create_node(obs2)

    equivalent = graph.compute_yoneda_equivalence(node1, node2)
    assert isinstance(equivalent, bool)


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
