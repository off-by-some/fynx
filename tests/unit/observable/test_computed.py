"""Unit tests for computed observable behavior."""

import pytest

from fynx import Observable


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_then_method_returns_computed_observable_instances():
    """then() method returns ComputedObservable instances"""
    base = Observable("base", 10)
    result = base.then(lambda x: x * 2)

    # Test behavior: result should be a computed observable
    assert hasattr(result, "value")
    assert result.value == 20  # 10 * 2


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_prevents_direct_modification():
    """Computed observables cannot be set directly (readonly protection)"""
    base = Observable("base", 10)
    computed_obs = base.then(lambda x: x * 2)

    # Test behavior: computed observables should be read-only
    with pytest.raises(ValueError, match="ComputedObservable is read-only"):
        computed_obs.set(50)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_stores_computation_function():
    """ComputedObservable stores reference to computation function for optimization"""

    def test_func(x):
        return x * 3

    base = Observable("base", 5)
    computed_obs = base.then(test_func)

    # Test behavior: computed observable should work correctly
    assert computed_obs.value == 15  # 5 * 3

    # Test reactivity: changing base should update computed value
    base.set(10)
    assert computed_obs.value == 30  # 10 * 3
