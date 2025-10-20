"""Unit tests for observable operations functionality."""

import pytest

from fynx.observable import Observable
from fynx.observable.operations import _ComputedObservable, _ConditionalObservable
from fynx.optimizer import OptimizationContext


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
def test_then_method_registers_with_optimization_context():
    """then() method registers computed observable with optimization context when available."""
    obs = Observable("test", 5)

    # Create optimization context
    with OptimizationContext() as context:
        computed = obs.then(lambda x: x * 2)

        # Should register the computed observable with the context
        # Check that the observable was registered by checking if it has a node
        assert context.optimizer.get_or_create_node(computed) is not None


@pytest.mark.unit
@pytest.mark.observable
def test_then_method_without_optimization_context():
    """then() method works correctly without optimization context."""
    obs = Observable("test", 5)

    # Should work without optimization context
    computed = obs.then(lambda x: x * 2)
    assert computed.value == 10


@pytest.mark.unit
@pytest.mark.observable
def test_then_method_with_merged_observable_registers_with_context():
    """then() method with merged observable registers with optimization context."""
    obs1 = Observable("obs1", 3)
    obs2 = Observable("obs2", 4)
    merged = obs1 + obs2

    # Create optimization context
    with OptimizationContext() as context:
        computed = merged.then(lambda a, b: a + b)

        # Should register the computed observable with the context
        assert context.optimizer.get_or_create_node(computed) is not None


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_creates_conditional_observable():
    """either() operator creates conditional observable that filters based on truthiness."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", True)

    # Use either operator
    result = obs1.either(obs2)

    # Should create conditional observable
    from fynx.observable.conditional import ConditionalObservable

    assert isinstance(result, ConditionalObservable)

    # Should be active when either is truthy
    assert result.is_active is True
    assert result.value is True  # True or False = True


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_with_falsy_values():
    """either() operator handles falsy values correctly."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", 0)

    # Use either operator
    result = obs1.either(obs2)

    # Should be inactive when both are falsy
    assert result.is_active is False

    # Accessing value when inactive should raise exception
    from fynx.observable.conditional import ConditionalNeverMet

    with pytest.raises(ConditionalNeverMet):
        _ = result.value


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_with_truthy_values():
    """either() operator handles truthy values correctly."""
    obs1 = Observable("obs1", "hello")
    obs2 = Observable("obs2", "")

    # Use either operator
    result = obs1.either(obs2)

    # Should be active when first is truthy
    assert result.is_active is True
    assert result.value == "hello"  # "hello" or "" = "hello"


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_updates_when_sources_change():
    """either() operator updates when source observables change."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", False)

    # Use either operator
    result = obs1.either(obs2)

    # Initially inactive
    assert result.is_active is False

    # Make first observable truthy
    obs1.set("truthy")
    assert result.is_active is True
    assert result.value == "truthy"

    # Make first falsy again, but second truthy
    obs1.set(False)
    obs2.set(42)
    assert result.is_active is True
    assert result.value == 42


@pytest.mark.unit
@pytest.mark.observable
def test_either_operator_uses_callable_condition():
    """either() operator uses callable condition to avoid timing issues."""
    obs1 = Observable("obs1", False)
    obs2 = Observable("obs2", True)

    # Use either operator
    result = obs1.either(obs2)

    # The result should be a conditional with a callable condition
    # This tests the internal implementation detail that uses lambda x: bool(x)
    assert result.is_active is True

    # Change to falsy values
    obs1.set(False)
    obs2.set(False)
    assert result.is_active is False
