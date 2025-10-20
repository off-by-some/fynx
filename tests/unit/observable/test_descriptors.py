"""Unit tests for observable descriptors functionality."""

import pytest

from fynx.observable import Observable
from fynx.observable.conditional import ConditionalNotMet
from fynx.observable.descriptors import ObservableValue, SubscriptableDescriptor


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_provides_transparent_access_to_observable():
    """ObservableValue provides transparent access to underlying observable value"""
    obs = Observable("test", "initial_value")
    obs_value = ObservableValue(obs)

    # Act & Assert - Behaves like the observable value
    assert obs_value.value == "initial_value"
    assert obs_value == "initial_value"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_maintains_value_synchronization():
    """ObservableValue stays synchronized with underlying observable changes"""
    obs = Observable("test", "initial")
    obs_value = ObservableValue(obs)

    # Act - Change underlying observable
    obs.set("updated")

    # Assert - ObservableValue reflects the change
    assert obs_value.value == "updated"
    assert obs_value == "updated"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_enables_setting_observable_value():
    """ObservableValue allows setting the underlying observable value"""
    obs = Observable("test", "initial")
    obs_value = ObservableValue(obs)

    # Act - Set through ObservableValue
    obs_value.set("updated")

    # Assert - Observable is updated
    assert obs.value == "updated"
    assert obs_value.value == "updated"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_supports_subscription_to_changes():
    """ObservableValue enables subscription to underlying observable changes"""
    obs = Observable("test", "initial")
    obs_value = ObservableValue(obs)

    received = []

    def callback(value):
        received.append(value)

    # Act - Subscribe through ObservableValue
    obs_value.subscribe(callback)
    obs.set("updated")

    # Assert - Receives notifications
    assert received == ["updated"]


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_supports_reactive_operators():
    """ObservableValue supports all reactive operators transparently"""
    obs1 = Observable("obs1", 10)
    obs2 = Observable("obs2", 20)

    obs_value1 = ObservableValue(obs1)
    obs_value2 = ObservableValue(obs2)

    # Test >> operator (transform)
    doubled = obs_value1 >> (lambda x: x * 2)
    assert doubled.value == 20

    # Test | operator (merge)
    merged = obs_value1 | obs_value2
    assert merged.value == (10, 20)

    # Test & operator (conditional)
    conditional = obs_value1 & obs_value2
    assert conditional.value == 10  # Both are truthy

    # Change condition to falsy
    obs2.set(0)
    assert not conditional.is_active  # Condition not currently met

    # Accessing value when conditions are unmet should raise ConditionalNotMet
    with pytest.raises(
        ConditionalNotMet, match="Conditions are not currently satisfied"
    ):
        _ = conditional.value


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_creates_class_level_observables():
    """SubscriptableDescriptor creates class-level observables on access"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="initial")

    # Act - Access the descriptor
    result = TestClass.attr

    # Assert - Returns ObservableValue with correct initial value
    assert isinstance(result, ObservableValue)
    assert result.value == "initial"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_shares_observable_across_instances():
    """SubscriptableDescriptor shares observable across class and instance access"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="shared")

    instance = TestClass()

    # Act - Access through both class and instance
    class_result = TestClass.attr
    instance_result = instance.attr

    # Assert - Same observable, same value
    assert class_result.value == instance_result.value == "shared"
    assert class_result.observable is instance_result.observable


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_supports_instance_assignment():
    """SubscriptableDescriptor supports assignment through instances"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value="initial")

    instance = TestClass()

    # Act - Create observable and assign new value
    instance.attr  # Ensure observable exists
    instance.attr = "updated"

    # Assert - Value is updated for both class and instance access
    assert TestClass.attr.value == "updated"
    assert instance.attr.value == "updated"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_isolates_observables_between_classes():
    """SubscriptableDescriptor creates separate observables for different classes"""

    class ClassA:
        attr = SubscriptableDescriptor(initial_value="a")

    class ClassB:
        attr = SubscriptableDescriptor(initial_value="b")

    # Act - Access attributes
    result_a = ClassA.attr
    result_b = ClassB.attr

    # Assert - Different observables, different values
    assert result_a.value == "a"
    assert result_b.value == "b"
    assert result_a.observable is not result_b.observable


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_uses_provided_observable():
    """SubscriptableDescriptor uses provided observable when available"""
    custom_obs = Observable("custom", "custom_value")

    class TestClass:
        attr = SubscriptableDescriptor(original_observable=custom_obs)

    # Act - Access the descriptor
    result = TestClass.attr

    # Assert - Uses the provided observable
    assert result.observable is custom_obs
    assert result.value == "custom_value"


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_observable_value_handles_none_values_transparently():
    """ObservableValue handles None values just like regular observable values"""
    obs = Observable("test", None)
    obs_value = ObservableValue(obs)

    # Act & Assert - None handling works transparently
    assert obs_value.value is None
    assert obs_value == None

    # Setting None works
    obs_value.set("not_none")
    assert obs_value.value == "not_none"
    assert obs_value == "not_none"

    # Setting back to None works
    obs_value.set(None)
    assert obs_value.value is None
    assert obs_value == None


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_subscriptable_descriptor_handles_none_initial_values():
    """SubscriptableDescriptor works correctly with None initial values"""

    class TestClass:
        attr = SubscriptableDescriptor(initial_value=None)

    # Act - Access the descriptor
    result = TestClass.attr

    # Assert - None value is preserved
    assert result.value is None
    assert result == None


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_raises_error_when_not_initialized():
    """SubscriptableDescriptor raises AttributeError when not properly initialized."""
    descriptor = SubscriptableDescriptor("test")
    descriptor._owner_class = None

    with pytest.raises(AttributeError, match="Descriptor not properly initialized"):
        descriptor.__get__(None, None)


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_set_raises_error_when_uninitialized():
    """SubscriptableDescriptor.__set__ raises AttributeError when uninitialized."""
    descriptor = SubscriptableDescriptor("test")
    descriptor._owner_class = None

    with pytest.raises(
        AttributeError, match="Cannot set value on uninitialized descriptor"
    ):
        descriptor.__set__(None, "value")


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_set_uses_instance_type_when_owner_class_none():
    """SubscriptableDescriptor.__set__ uses instance type when _owner_class is None."""
    descriptor = SubscriptableDescriptor("test")
    descriptor._owner_class = None

    class TestClass:
        pass

    instance = TestClass()

    # Should use instance type when _owner_class is None
    descriptor.__set__(instance, "value")

    # Verify observable was created on the instance's class
    obs_key = f"_{descriptor.attr_name}_observable"
    assert hasattr(TestClass, obs_key)


@pytest.mark.unit
@pytest.mark.observable
def test_subscriptable_descriptor_creates_new_observable_when_none_provided():
    """SubscriptableDescriptor creates new observable when no original observable provided."""
    descriptor = SubscriptableDescriptor("test")
    descriptor._original_observable = None

    class TestClass:
        pass

    descriptor._owner_class = TestClass
    descriptor.attr_name = "test_attr"

    # Access descriptor to trigger observable creation
    result = descriptor.__get__(None, TestClass)

    # Should create new observable with initial value
    assert result.value == "test"
    assert descriptor.attr_name == "test_attr"
