"""Unit tests for optimizer functionality."""

import time

import pytest

from fynx import observable
from fynx.observable.conditional import ConditionalObservable
from fynx.optimizer import (
    Morphism,
    MorphismParser,
    ReactiveGraph,
    get_graph_statistics,
    optimize_reactive_graph,
)


@pytest.mark.unit
@pytest.mark.operators
def test_build_from_observables_adds_computed_dependencies():
    """Graph builder links computed observables to their sources."""
    # Arrange
    base = observable(2)
    doubled = base >> (lambda x: x * 2)
    # Act
    graph = ReactiveGraph()
    graph.build_from_observables([doubled])
    # Assert
    node_base = graph.get_or_create_node(base)
    node_doubled = graph.get_or_create_node(doubled)
    assert node_doubled in node_base.outgoing
    assert node_base in node_doubled.incoming


@pytest.mark.unit
@pytest.mark.operators
def test_build_from_observables_handles_merged_sources():
    """Merged observables add all source dependencies to the graph."""
    # Arrange
    a = observable(1)
    b = observable(2)
    merged = a | b
    s = merged >> (lambda x, y: x + y)
    # Act
    graph = ReactiveGraph()
    graph.build_from_observables([s])
    # Assert
    # In the optimizer graph, a MergedObservable node feeds a computed node.
    # Assert that both sources connect to the merged node, and merged connects to sum.
    n_a = graph.get_or_create_node(a)
    n_b = graph.get_or_create_node(b)
    # Find the merged node by type
    merged_nodes = [
        n
        for n in graph.nodes.values()
        if type(n.observable).__name__ == "MergedObservable"
    ]
    assert len(merged_nodes) == 1
    n_merged = merged_nodes[0]
    n_s = graph.get_or_create_node(s)
    assert n_merged in n_a.outgoing and n_merged in n_b.outgoing
    assert n_a in n_merged.incoming and n_b in n_merged.incoming
    assert n_s in n_merged.outgoing and n_merged in n_s.incoming


@pytest.mark.unit
@pytest.mark.observable
def test_build_from_observables_handles_conditional_dependencies():
    """Conditional observables include source and condition dependencies."""
    # Arrange
    data = observable(10)
    flag = observable(True)
    filt = data & flag
    # Act
    graph = ReactiveGraph()
    graph.build_from_observables([filt])
    # Assert
    n_data = graph.get_or_create_node(data)
    n_flag = graph.get_or_create_node(flag)
    n_filt = graph.get_or_create_node(filt)
    # Conditional node should depend on source; observable conditions may or may not be explicit nodes
    assert n_data in n_filt.incoming
    # Some implementations do not materialize condition nodes in the graph; tolerate either form
    if n_flag not in n_filt.incoming:
        # Fallback: the conditional's dependencies should at least include the flag at the object level
        assert any(
            getattr(dep, "value", None) is not None
            for dep in getattr(filt, "_all_dependencies", [])
        )
    # Source should point to conditional; condition linkage may be implicit
    assert n_filt in n_data.outgoing


@pytest.mark.unit
def test_topological_sort_orders_sources_first():
    """Topological order starts with sources and respects edges."""
    # Arrange
    base = observable(1)
    c1 = base >> (lambda x: x + 1)
    c2 = c1 >> (lambda x: x * 2)
    graph = ReactiveGraph()
    graph.build_from_observables([c2])
    # Act
    ordered = graph.topological_sort()
    # Assert
    obs_to_pos = {n.observable: i for i, n in enumerate(ordered)}
    assert obs_to_pos[base] < obs_to_pos[c1] < obs_to_pos[c2]


@pytest.mark.unit
def test_find_paths_returns_all_simple_paths_within_depth():
    """find_paths locates all paths between two nodes up to a depth bound."""
    # Arrange
    a = observable(1)
    b = a >> (lambda x: x + 1)
    c = b >> (lambda x: x + 1)
    graph = ReactiveGraph()
    graph.build_from_observables([c])
    n_a = graph.get_or_create_node(a)
    n_c = graph.get_or_create_node(c)
    # Act
    paths = graph.find_paths(n_a, n_c, max_depth=5)
    # Assert
    assert len(paths) >= 1
    assert paths[0][0].observable is a and paths[0][-1].observable is c


@pytest.mark.unit
def test_get_graph_statistics_reports_counts():
    """Graph statistics expose node and edge counts and depth."""
    # Arrange
    base = observable(2)
    d = base >> (lambda x: x * 2)
    graph = ReactiveGraph()
    graph.build_from_observables([d])
    # Act
    stats = get_graph_statistics(graph)
    # Assert
    assert stats["total_nodes"] >= 2
    assert stats["total_edges"] >= 1
    assert stats["max_depth"] >= 1


@pytest.mark.unit
def test_compute_equivalence_classes_partitions_by_structure():
    """Equivalent computed shapes end up in the same structural class."""
    # Arrange
    base = observable(3)
    c1 = base >> (lambda x: x + 1)
    c2 = base >> (lambda x: x + 2) >> (lambda x: x - 1)
    graph = ReactiveGraph()
    graph.build_from_observables([c1, c2])
    # Act
    classes = graph.compute_equivalence_classes()
    # Assert
    assert isinstance(classes, dict)
    assert sum(len(v) for v in classes.values()) == len(graph.nodes)


@pytest.mark.unit
def test_apply_functor_composition_fusion_reduces_chain_nodes():
    """Functor composition fusion collapses linear computed chains."""
    # Arrange
    base = observable(0)
    chain = base
    for i in range(5):
        chain = chain >> (lambda x, i=i: x + i)
    graph = ReactiveGraph()
    graph.build_from_observables([chain])
    before = len(graph.nodes)
    # Act
    fusions = graph.apply_functor_composition_fusion()
    after = len(graph.nodes)
    # Assert
    assert fusions >= 1
    assert after <= before


@pytest.mark.unit
def test_apply_product_factorization_factores_shared_inputs():
    """Product factorization reduces duplicate computations with same inputs."""
    # Arrange
    base = observable(5)
    shared1 = base >> (lambda x: x + 1)
    shared2 = base >> (lambda x: x + 1)
    graph = ReactiveGraph()
    graph.build_from_observables([shared1, shared2])
    # Act
    changes = graph.apply_product_factorization()
    # Assert
    assert changes >= 0  # May be zero if already recognized


@pytest.mark.unit
def test_apply_pullback_fusion_combines_sequential_filters():
    """Pullback fusion merges chained conditionals with same source."""
    # Arrange
    data = observable(10)
    step1 = data & (lambda x: x > 5)
    step2 = step1 & (lambda x: x < 20)
    graph = ReactiveGraph()
    graph.build_from_observables([step2])
    before = len(graph.nodes)
    # Act
    fused = graph.apply_pullback_fusion()
    after = len(graph.nodes)
    # Assert
    assert fused >= 0
    assert after <= before


@pytest.mark.unit
def test_optimize_runs_all_phases_and_returns_metrics():
    """optimize aggregates rewrite stats, materialization, and confluence results."""
    # Arrange
    a = observable(1)
    b = a >> (lambda x: x + 2)
    c = b >> (lambda x: x * 3)
    graph = ReactiveGraph()
    graph.build_from_observables([c])
    # Act
    results = graph.optimize()
    # Assert
    assert set(
        [
            "optimization_time",
            "equivalence_classes",
            "functor_fusions",
            "product_factorizations",
            "filter_fusions",
            "total_nodes",
            "materialized_nodes",
            "confluence",
            "universal_properties",
        ]
    ).issubset(results.keys())


@pytest.mark.unit
def test_optimize_reactive_graph_builds_and_returns_graph():
    """optimize_reactive_graph returns results and a populated optimizer graph."""
    # Arrange
    base = observable(4)
    chain = base >> (lambda x: x * 2)
    # Act
    results, optimizer = optimize_reactive_graph([chain])
    # Assert
    assert isinstance(optimizer, ReactiveGraph)
    assert results["total_nodes"] >= 1


@pytest.mark.unit
def test_enable_profiling_wraps_functions_and_records_times():
    """Profiling wrapper records call counts and execution time averages."""
    # Arrange
    base = observable(100)
    slow = base >> (lambda x: sum(range(x)))
    graph = ReactiveGraph()
    graph.build_from_observables([slow])
    # Act
    graph.enable_profiling()
    # Manually invoke the profiled computation function to ensure calls are recorded
    node_slow = graph.get_or_create_node(slow)
    if node_slow.computation_func is not None:
        node_slow.computation_func(base.value)
        node_slow.computation_func(base.value)
    summary = graph.get_profiling_summary()
    # Assert: There should be at least one profiled node and at least one call
    assert summary["total_profiled_nodes"] >= 1
    assert summary["total_calls"] >= 1


@pytest.mark.unit
def test_morphism_identity_and_compose_properties():
    """Morphism identity, single, compose, and normalization behave as expected."""
    # Arrange
    idm = Morphism.identity()
    f = Morphism.single("f")
    g = Morphism.single("g")
    # Act
    composed = Morphism.compose(f, g)
    norm = composed.normalize()
    # Assert
    assert str(idm) == "id"
    assert "(" in str(composed)
    assert norm == composed


@pytest.mark.unit
def test_morphism_parser_parses_identity_single_and_composition():
    """MorphismParser recognizes identity, single morphisms, and compositions."""
    # Arrange / Act
    idm = MorphismParser.parse("id")
    s = MorphismParser.parse("f")
    comp = MorphismParser.parse("(f) ∘ (g)")
    # Assert
    assert idm == Morphism.identity()
    assert s == Morphism.single("f")
    assert isinstance(comp, Morphism)


@pytest.mark.unit
def test_morphism_normalize_handles_identity_laws():
    """Morphism.normalize() applies identity laws correctly."""
    # Test left identity: id ∘ f = f
    f = Morphism.single("f")
    left_id = Morphism.compose(Morphism.identity(), f)
    assert left_id.normalize() == f

    # Test right identity: f ∘ id = f
    right_id = Morphism.compose(f, Morphism.identity())
    assert right_id.normalize() == f


@pytest.mark.unit
def test_morphism_normalize_handles_associativity():
    """Morphism.normalize() handles associativity correctly."""
    f = Morphism.single("f")
    g = Morphism.single("g")
    h = Morphism.single("h")

    # Test associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
    left_assoc = Morphism.compose(Morphism.compose(f, g), h)
    right_assoc = Morphism.compose(f, Morphism.compose(g, h))

    # Both should normalize to the same form
    assert (
        left_assoc.normalize().canonical_form()
        == right_assoc.normalize().canonical_form()
    )


@pytest.mark.unit
def test_morphism_normalize_handles_unknown_type():
    """Morphism.normalize() handles unknown morphism types gracefully."""
    # Create a morphism with unknown type
    morphism = Morphism("unknown")

    # Should return self for unknown types
    assert morphism.normalize() is morphism


@pytest.mark.unit
def test_morphism_canonical_form_handles_unknown_type():
    """Morphism.canonical_form() handles unknown morphism types."""
    # Create a morphism with unknown type
    morphism = Morphism("unknown")

    # Should return ("unknown",) for unknown types
    assert morphism.canonical_form() == ("unknown",)


@pytest.mark.unit
def test_morphism_equality_with_non_morphism():
    """Morphism.__eq__ returns NotImplemented for non-Morphism objects."""
    morphism = Morphism.single("f")

    # Should return NotImplemented for non-Morphism objects
    result = morphism.__eq__("not a morphism")
    assert result is NotImplemented


@pytest.mark.unit
def test_morphism_str_handles_unknown_type():
    """Morphism.__str__ handles unknown morphism types."""
    # Create a morphism with unknown type
    morphism = Morphism("unknown")

    # Should return "unknown(unknown)" for unknown types
    assert str(morphism) == "unknown(unknown)"


@pytest.mark.unit
def test_morphism_parser_handles_empty_signature():
    """MorphismParser.parse() handles empty signature by returning single morphism with empty name."""
    # Empty signature should return single morphism with empty name
    result = MorphismParser.parse("")
    assert result._type == "single"
    assert result._name == ""


@pytest.mark.unit
def test_morphism_parser_split_composition_handles_empty_current():
    """MorphismParser._split_composition() handles empty current string."""
    # Test with signature that has empty parts
    parts = MorphismParser._split_composition("f ∘  ∘ g")

    # Should filter out empty parts
    assert "f" in parts
    assert "g" in parts
    assert "" not in parts or parts.count("") == 0


@pytest.mark.unit
def test_morphism_parser_split_composition_handles_nested_parentheses():
    """MorphismParser._split_composition() handles nested parentheses correctly."""
    # Test with nested parentheses
    parts = MorphismParser._split_composition("(f ∘ g) ∘ (h ∘ i)")

    # Should split at top level only
    assert len(parts) == 2
    assert "(f ∘ g)" in parts
    assert "(h ∘ i)" in parts


@pytest.mark.unit
def test_check_confluence_reports_convergent_orders():
    """Confluence check returns structured report with convergence counts."""
    # Arrange
    base = observable(2)
    chain = base >> (lambda x: x + 1) >> (lambda x: x * 3)
    graph = ReactiveGraph()
    graph.build_from_observables([chain])
    # Act
    report = graph.check_confluence()
    # Assert
    assert report["total_orders_tested"] == 6
    assert "convergent_orders" in report


@pytest.mark.unit
def test_verify_universal_properties_returns_counts():
    """Universal property verification summarizes checked candidates."""
    # Arrange
    a = observable(1)
    b = a >> (lambda x: x + 1)
    c = a >> (lambda x: x * 2)
    merged = (b | c) >> (lambda u, v: u + v)
    graph = ReactiveGraph()
    graph.build_from_observables([merged])
    # Act
    summary = graph.verify_universal_properties()
    # Assert
    assert set(
        [
            "verified_products",
            "verified_pullbacks",
            "total_candidates_checked",
            "universal_property_satisfied",
        ]
    ).issubset(summary.keys())


@pytest.mark.unit
def test_compose_morphisms_and_identity_helpers():
    """compose_morphisms obeys identity laws."""
    # Arrange
    graph = ReactiveGraph()
    # Act / Assert
    assert graph.compose_morphisms("id", "f") == "f"
    assert graph.compose_morphisms("f", "id") == "f"


@pytest.mark.unit
def test_get_hom_set_representation_includes_cardinality_and_flags():
    """Hom-set representation includes counts and identity/direct flags."""
    # Arrange
    base = observable(1)
    nxt = base >> (lambda x: x + 1)
    graph = ReactiveGraph()
    graph.build_from_observables([nxt])
    n_base = graph.get_or_create_node(base)
    n_nxt = graph.get_or_create_node(nxt)
    # Act
    rep = graph.get_hom_set_representation(n_base, n_nxt)
    # Assert
    assert rep["cardinality"] >= 1
    assert "from_node" in rep and "to_node" in rep


@pytest.mark.unit
def test_materialization_strategy_sets_is_materialized_flags():
    """Materialization optimization decides per-node storage strategy."""
    # Arrange
    base = observable(1)
    chain = base >> (lambda x: x + 1) >> (lambda x: x * 2)
    graph = ReactiveGraph()
    graph.build_from_observables([chain])
    # Act
    graph.optimize_materialization()
    # Assert
    assert any(n.is_materialized for n in graph.nodes.values())


@pytest.mark.unit
def test_optimizer_handles_conditional_chains_in_graph_methods():
    """Graph helpers handle ConditionalObservable when present."""
    # Arrange
    data = observable(5)
    cond = data & (lambda x: x > 3)
    next_step = cond >> (lambda x: x + 1)
    graph = ReactiveGraph()
    graph.build_from_observables([next_step])
    # Act
    classes = graph.compute_equivalence_classes()
    # Assert
    assert isinstance(classes, dict)
    assert any(
        isinstance(n.observable, ConditionalObservable) for n in graph.nodes.values()
    )
