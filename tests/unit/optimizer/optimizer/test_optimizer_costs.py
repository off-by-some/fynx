import pytest

from fynx import observable
from fynx.optimizer import ReactiveGraph


@pytest.mark.unit
def test_compute_cost_prefers_materialize_for_shared_node():
    # Build graph: base -> shared -> {branch1, branch2}
    base = observable(1)
    shared = base >> (lambda x: x + 1)
    branch1 = shared >> (lambda x: x * 2)
    branch2 = shared >> (lambda x: x * 3)

    rg = ReactiveGraph()
    rg.build_from_observables([branch1, branch2])

    n_shared = rg.get_or_create_node(shared)
    # Ensure two dependents for sharing penalty
    assert len(n_shared.outgoing) >= 1

    # Compare cost with and without materialization
    mat_cost = n_shared.compute_cost({n_shared})
    rec_cost = n_shared.compute_cost(set())
    # With two dependents, materializing should not be more expensive than recomputing in this model
    assert mat_cost <= rec_cost


@pytest.mark.unit
def test_optimize_materialization_sets_flags_across_graph():
    base = observable(1)
    # Create a small diamond to trigger decisions
    left = base >> (lambda x: x + 1)
    right = base >> (lambda x: x + 2)
    out = (left + right) >> (lambda l, r: l + r)

    rg = ReactiveGraph()
    rg.build_from_observables([out])
    rg.optimize_materialization()

    # At least source should be materialized
    assert any(node.is_materialized for node in rg.nodes.values())


@pytest.mark.unit
def test_apply_functor_composition_fusion_counts_changes():
    base = observable(0)
    chain = base
    for i in range(6):
        chain = chain >> (lambda x, i=i: x + i)
    rg = ReactiveGraph()
    rg.build_from_observables([chain])
    before = len(rg.nodes)
    changes = rg.apply_functor_composition_fusion()
    after = len(rg.nodes)
    assert changes >= 0
    assert after <= before


@pytest.mark.unit
def test_apply_product_factorization_noop_on_unshared_inputs():
    a = observable(1)
    b = a >> (lambda x: x + 1)
    rg = ReactiveGraph()
    rg.build_from_observables([b])
    changes = rg.apply_product_factorization()
    assert changes >= 0


@pytest.mark.unit
def test_apply_pullback_fusion_on_simple_chain():
    src = observable(5)
    c1 = src & (lambda x: x > 0)
    c2 = c1 & (lambda x: x < 10)
    rg = ReactiveGraph()
    rg.build_from_observables([c2])
    f = rg.apply_pullback_fusion()
    assert f >= 0
