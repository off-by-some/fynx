"""Unit tests for computed observable implementation specifics."""

import pytest

from fynx.observable import Observable
from fynx.observable.computed import ComputedObservable


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_then_method_returns_computed_observable_instances():
    """then() method returns ComputedObservable instances"""
    base = Observable("base", 10)
    result = base.then(lambda x: x * 2)

    assert isinstance(result, ComputedObservable)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_prevents_direct_modification():
    """Computed observables cannot be set directly (readonly protection)"""
    base = Observable("base", 10)
    computed_obs = base.then(lambda x: x * 2)

    with pytest.raises(ValueError, match="Computed observables are read-only"):
        computed_obs.set(50)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_internal_set_method_bypasses_readonly():
    """Computed observables allow internal updates via _set_computed_value method"""
    base = Observable("base", 10)
    computed_obs = base.then(lambda x: x * 2)

    # Internal method should work
    computed_obs._set_computed_value(42)
    assert computed_obs.value == 42


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_stores_computation_function():
    """ComputedObservable stores reference to computation function for optimization"""

    def test_func(x):
        return x * 3

    base = Observable("base", 5)
    computed_obs = base.then(test_func)

    # Implementation detail: stores computation function
    assert computed_obs._computation_func == test_func


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_computed_observable_stores_source_observable_reference():
    """ComputedObservable stores reference to source observable"""
    source = Observable("source", "data")
    computed_obs = source.then(lambda x: x.upper())

    # Implementation detail: stores source reference
    assert computed_obs._source_observable is source
