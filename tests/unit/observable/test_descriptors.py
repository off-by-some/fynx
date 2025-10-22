"""Unit tests for observable descriptors functionality."""

import pytest

from fynx.observable import Observable
from fynx.observable.value import ObservableValue, SubscriptableDescriptor


@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_stores_and_returns_values():
    """ObservableValue stores values and provides transparent access"""
    # Test with regular value
    val = ObservableValue(42)
    assert val.value == 42
    assert val == 42

    # Test with Observable
    obs = Observable("test", "hello")
    obs_val = ObservableValue(obs)
    assert obs_val.value is obs
    assert obs_val == obs


@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_change_callback():
    """ObservableValue calls on_change callback when value changes"""
    changes = []

    def on_change(old_val, new_val):
        changes.append((old_val, new_val))

    val = ObservableValue(42, on_change=on_change)
    val.value = 100

    assert changes == [(42, 100)]


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_creates_observable():
    """SubscriptableDescriptor creates Observable instances"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="initial")

    result = TestClass.attr
    assert isinstance(result, Observable)
    assert result.value == "initial"


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_shares_across_access():
    """SubscriptableDescriptor shares the same Observable across access patterns"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="shared")

    instance = TestClass()

    # Access through class and instance should return the same Observable
    class_result = TestClass.attr
    instance_result = instance.attr

    assert class_result is instance_result
    assert class_result.value == "shared"


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_assignment():
    """SubscriptableDescriptor supports assignment"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="initial")

    instance = TestClass()
    instance.attr = "updated"

    # Both class and instance access should reflect the change
    assert TestClass.attr.value == "updated"
    assert instance.attr.value == "updated"


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_isolates_classes():
    """SubscriptableDescriptor creates separate observables for different classes"""

    class ClassA:
        attr = SubscriptableDescriptor(initial_value="a")

    class ClassB:
        attr = SubscriptableDescriptor(initial_value="b")

    # Different classes should have different Observable instances
    result_a = ClassA.attr
    result_b = ClassB.attr

    assert result_a is not result_b
    assert result_a.value == "a"
    assert result_b.value == "b"


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_with_provided_observable():
    """SubscriptableDescriptor can use a provided Observable"""
    custom_obs = Observable("custom", "custom_value")

    class TestClass:
        attr = SubscriptableDescriptor(original_observable=custom_obs)

    result = TestClass.attr
    assert result is custom_obs
