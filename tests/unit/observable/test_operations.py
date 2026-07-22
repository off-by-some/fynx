"""Unit tests for observable operations functionality."""

import pytest

from fynx.observable import Observable
from fynx.observable.operations import _ComputedObservable, _ConditionalObservable


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_import():
    """_ConditionalObservable function imports and returns ConditionalObservable class."""
    ConditionalObservable = _ConditionalObservable()

    # Should return the ConditionalObservable class
    assert ConditionalObservable is not None
    from fynx.observable.conditional import ConditionalObservable as ExpectedClass

    assert ConditionalObservable is ExpectedClass


@pytest.mark.unit
@pytest.mark.observable
def test_computed_observable_import():
    """_ComputedObservable function imports and returns ComputedObservable class."""
    ComputedObservable = _ComputedObservable()

    # Should return the ComputedObservable class
    assert ComputedObservable is not None
    from fynx.observable.computed import ComputedObservable as ExpectedClass

    assert ComputedObservable is ExpectedClass


@pytest.mark.unit
@pytest.mark.observable
def test_then_method_creates_computed_value():
    """then() creates computed values without an external optimization context."""
    obs = Observable("test", 5)

    computed = obs.then(lambda x: x * 2)
    assert computed.value == 10


@pytest.mark.unit
@pytest.mark.observable
def test_then_method_with_merged_observable_unpacks_values():
    """then() with a merged observable passes tuple values as arguments."""
    obs1 = Observable("obs1", 3)
    obs2 = Observable("obs2", 4)
    merged = obs1 + obs2

    computed = merged.then(lambda a, b: a + b)

    assert computed.value == 7


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_creates_boolean_observable():
    """either() operator creates a total boolean observable."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", True)

    # Use either operator
    result = obs1.either(obs2)

    # Should produce a boolean value when either side is truthy
    assert result.value is True


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_with_falsy_values():
    """either() operator returns False when both values are falsy."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", 0)

    # Use either operator
    result = obs1.either(obs2)

    assert result.value is False


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_with_truthy_values():
    """either() operator coerces truthy values to True."""
    obs1 = Observable("obs1", "hello")
    obs2 = Observable("obs2", "")

    # Use either operator
    result = obs1.either(obs2)

    assert result.value is True


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_updates_when_sources_change():
    """either() operator updates when source observables change."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", False)

    # Use either operator
    result = obs1.either(obs2)

    # Initially false
    assert result.value is False

    # Make first observable truthy
    obs1.set("truthy")
    assert result.value is True

    # Make first falsy again, but second truthy
    obs1.set(False)
    obs2.set(42)
    assert result.value is True


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_remains_total_when_sources_become_false():
    """either() operator keeps emitting boolean values when sources become false."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", True)

    # Use either operator
    result = obs1.either(obs2)

    assert result.value is True

    # Change to falsy values
    obs1.set(False)
    obs2.set(False)
    assert result.value is False
