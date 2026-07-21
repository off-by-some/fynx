"""Focused tests for the algebraic runtime guarantees."""

import pytest

from fynx import Store, observable
from fynx.observable.base import Observable, TransformPurityError


@pytest.mark.unit
@pytest.mark.observable
def test_diamond_convergence_recomputes_once_per_source_update():
    """A converging node in a diamond graph recomputes once after stabilization."""
    source = Observable("source", 1)
    calls = {"left": 0, "right": 0, "joined": 0}

    def left_fn(value):
        calls["left"] += 1
        return value + 1

    def right_fn(value):
        calls["right"] += 1
        return value * 2

    def joined_fn(left, right):
        calls["joined"] += 1
        return left + right

    left = source >> left_fn
    right = source >> right_fn
    joined = (left + right) >> joined_fn
    received = []
    joined.subscribe(received.append)

    calls.update({"left": 0, "right": 0, "joined": 0})

    source.set(2)

    assert joined.value == 7
    assert received == [7]
    assert calls == {"left": 1, "right": 1, "joined": 1}


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_reads_with_hint():
    """Transforms are pure maps over their explicit input values."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    def apply_discount(value):
        return value * (1 - discount.value)

    with pytest.raises(TransformPurityError) as error:
        price >> apply_discount

    message = str(error.value)
    assert "inside a transform" in message
    assert "pass every reactive input explicitly" in message
    assert "price + discount" in message


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_reads_through_helper():
    """The runtime guard catches helpers that static certification cannot inspect."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    def current_discount():
        return discount.value

    with pytest.raises(TransformPurityError, match="inside a transform"):
        price >> (lambda value: value * (1 - current_discount()))


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_defaults():
    """Default arguments cannot smuggle observable reads into a transform."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        price >> (lambda value, hidden=discount: value * (1 - hidden.value))


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_mutations_with_hint():
    """Transform functions cannot smuggle side effects into the reactive graph."""
    source = Observable("source", 1)
    target = Observable("target", 0)

    def mutate_target(value):
        target.set(value)
        return value

    with pytest.raises(TransformPurityError) as error:
        source >> mutate_target

    message = str(error.value)
    assert "inside a transform" in message
    assert "Move side effects and mutations" in message
    assert target.value == 0


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_transparent_observable_reads():
    """Value-like Observable operations are still reads inside transforms."""
    source = Observable("source", 1)
    flag = Observable("flag", True)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: value if flag else 0)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: f"{value}:{flag}")


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_transparent_store_value_reads():
    """Store descriptors keep their friendly syntax outside transforms only."""

    class Flags(Store):
        enabled = observable(True)

    source = observable(1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: value if Flags.enabled else 0)


@pytest.mark.unit
@pytest.mark.observable
def test_transform_accepts_explicit_product_dependencies():
    """Multi-input transforms stay expressive by combining inputs first."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    discounted = (price + discount) >> (
        lambda current_price, current_discount: current_price * (1 - current_discount)
    )

    assert discounted.value == 90.0

    discount.set(0.2)
    assert discounted.value == 80.0

    price.set(50.0)
    assert discounted.value == 40.0


@pytest.mark.unit
@pytest.mark.observable
def test_callable_condition_tracks_external_observable_dependencies():
    """Callable pullbacks subscribe to observables read inside predicates."""
    source = Observable("source", 5)
    limit = Observable("limit", 10)
    filtered = source & (lambda value: value < limit.value)

    assert filtered.is_active is True

    limit.set(3)
    assert filtered.is_active is False

    limit.set(6)
    assert filtered.is_active is True

    source.set(7)
    assert filtered.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_callable_condition_switches_dynamic_dependencies():
    """Predicate dependencies move when a conditional predicate changes branches."""
    source = Observable("source", 5)
    use_left = Observable("use_left", True)
    left_limit = Observable("left_limit", 10)
    right_limit = Observable("right_limit", 3)

    filtered = source & (
        lambda value: value
        < (left_limit.value if use_left.value else right_limit.value)
    )

    assert filtered.is_active is True

    right_limit.set(10)
    assert filtered.is_active is True

    use_left.set(False)
    assert filtered.is_active is True

    right_limit.set(4)
    assert filtered.is_active is False

    left_limit.set(1)
    assert filtered.is_active is False
