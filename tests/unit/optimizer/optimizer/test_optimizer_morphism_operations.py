import pytest

from fynx import observable
from fynx.optimizer import Morphism, MorphismParser, ReactiveGraph


@pytest.mark.unit
def test_compose_morphisms_identity_laws():
    graph = ReactiveGraph()
    assert graph.compose_morphisms("id", "f") == "f"
    assert graph.compose_morphisms("g", "id") == "g"


@pytest.mark.unit
def test_morphism_equality_and_hash_canonicalization():
    # (f ∘ id) ∘ g == f ∘ g
    left = Morphism.compose(
        Morphism.compose(Morphism.single("f"), Morphism.identity()),
        Morphism.single("g"),
    )
    right = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    assert left == right
    assert hash(left) == hash(right)


@pytest.mark.unit
def test_morphism_parser_nested_parentheses():
    parsed = MorphismParser.parse("(((f)) ∘ ((g)))")
    assert isinstance(parsed, Morphism)
    assert str(parsed) == "(f) ∘ (g)"


@pytest.mark.unit
def test_copy_graph_preserves_structure():
    base = observable(1)
    a = base >> (lambda x: x + 1)
    b = a >> (lambda x: x * 2)
    rg = ReactiveGraph()
    rg.build_from_observables([b])
    copy = rg.copy_graph()
    assert len(copy.nodes) == len(rg.nodes)
    # All nodes in copy should have same incoming/outgoing sizes
    for obs, node in rg.nodes.items():
        cnode = copy.nodes[obs]
        assert len(cnode.incoming) == len(node.incoming)
        assert len(cnode.outgoing) == len(node.outgoing)


@pytest.mark.unit
def test_topological_sort_fallback_on_cycle():
    # Create two observables and manually connect nodes to simulate a cycle
    x = observable(1)
    y = x >> (lambda v: v + 1)
    rg = ReactiveGraph()
    rg.build_from_observables([y])
    nx = rg.get_or_create_node(x)
    ny = rg.get_or_create_node(y)
    # Force a cycle in graph structure
    ny.outgoing.add(nx)
    nx.incoming.add(ny)
    ordered = rg.topological_sort()
    # Fallback returns depth-sorted list; ensure both nodes present
    assert set(n.observable for n in ordered) == {x, y}


@pytest.mark.unit
def test_detect_cycles_finds_introduced_cycle():
    a = observable(1)
    b = a >> (lambda v: v + 2)
    rg = ReactiveGraph()
    rg.build_from_observables([b])
    na = rg.get_or_create_node(a)
    nb = rg.get_or_create_node(b)
    # Introduce cycle
    na.incoming.add(nb)
    nb.outgoing.add(na)
    cycles = rg.cycles
    assert cycles  # at least one cycle detected


@pytest.mark.unit
def test_check_confluence_structure_of_report():
    base = observable(2)
    chain = base >> (lambda x: x + 1) >> (lambda x: x * 3)
    rg = ReactiveGraph()
    rg.build_from_observables([chain])
    report = rg.check_confluence()
    assert report["total_orders_tested"] == 6
    assert isinstance(report["order_results"], list)


@pytest.mark.unit
def test_verify_optimization_correctness():
    # Test optimization correctness verification
    base = observable(1)
    c = base >> (lambda x: x + 1)
    rg = ReactiveGraph()
    rg.build_from_observables([c])
    res = rg.verify_optimization_correctness()
    assert set(
        [
            "cycles_introduced",
            "graph_connected",
            "structural_correctness",
            "optimization_invariants_held",
        ]
    ).issubset(res.keys())
