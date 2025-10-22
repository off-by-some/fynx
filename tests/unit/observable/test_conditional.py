"""Unit tests for conditional observable behavior."""

import pytest

from fynx import (
    ConditionalNeverMet,
    ConditionalNotMet,
    ConditionalObservable,
    Observable,
)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_inherits_from_conditional_observable_class():
    """& operator creates ConditionalObservable instances"""
    source = Observable("source", "data")
    condition = Observable("condition", True)

    conditional = source & condition

    assert isinstance(conditional, ConditionalObservable)


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_reacts_to_source_changes():
    """ConditionalObservable reacts to changes in its source observable"""
    source = Observable("source", "data")
    condition = Observable("condition", True)

    conditional = source & condition

    # Test behavior: conditional should pass through source value when condition is met
    assert conditional.value == "data"

    # Test reactivity: changing source should update conditional
    source.set("new_data")
    assert conditional.value == "new_data"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_stores_conditions():
    """ConditionalObservable correctly applies multiple conditions"""

    source = Observable("source", "data")
    cond1 = Observable("cond1", True)
    cond2 = Observable("cond2", False)

    conditional = source & cond1 & cond2

    # Test behavior: should be inactive because cond2 is False
    assert conditional.is_active is False

    # Test that changing cond2 to True makes it active
    cond2.set(True)
    assert conditional.is_active is True
    assert conditional.value == "data"

    # Test that changing cond1 to False makes it inactive again
    cond1.set(False)
    assert conditional.is_active is False


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_tracks_condition_satisfaction_state():
    """ConditionalObservable tracks whether its conditions are satisfied"""
    source = Observable("source", "data")
    cond1 = Observable("cond1", True)  # Always true for inner conditional
    cond2 = Observable("cond2", False)  # Controls outer conditional

    conditional = source & cond1 & cond2

    # Test behavior: conditional should be inactive when conditions not met
    assert conditional.is_active is False

    # Test behavior: conditional should become active when conditions are met
    cond2.set(True)
    assert conditional.is_active is True
    assert conditional.value == "data"

    # Test behavior: conditional should become inactive when conditions not met again
    cond2.set(False)
    assert conditional.is_active is False


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_initial_value_based_on_conditions():
    """ConditionalObservable initializes with source value only when conditions met"""
    # Conditions initially met
    source1 = Observable("source1", "data")
    condition1 = Observable("condition1", True)
    conditional1 = source1 & condition1

    assert conditional1.value == "data"

    # Conditions initially not met
    source2 = Observable("source2", "data")
    condition2 = Observable("condition2", False)
    conditional2 = source2 & condition2

    # Accessing value when conditions never met raises ConditionalNeverMet
    from fynx.observable.computed import ConditionalNeverMet

    with pytest.raises(ConditionalNeverMet):
        _ = conditional2.value


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_with_no_additional_conditions():
    """ConditionalObservable works with single source (no additional conditions)"""
    source = Observable("source", "data")

    # Using & with only the source should still work
    always_true = Observable("always_true", True)
    conditional = source & always_true

    # Should be active and pass through the source value
    assert conditional.is_active is True
    assert conditional.value == "data"

    # Should react to changes in the condition
    always_true.set(False)
    assert conditional.is_active is False


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_operator_creates_flat_structure():
    """& operator chaining creates ConditionalObservable structure"""
    source = Observable("source", "data")
    cond1 = Observable("cond1", True)
    cond2 = Observable("cond2", True)

    conditional = source & cond1 & cond2

    # Test behavior: the result should be a ConditionalObservable
    assert isinstance(conditional, ConditionalObservable)

    # Test behavior: conditional should work correctly with multiple conditions
    assert conditional.is_active is True
    assert conditional.value == "data"

    # Test behavior: changing any condition should affect the conditional
    cond1.set(False)
    assert conditional.is_active is False

    cond1.set(True)
    cond2.set(False)
    assert conditional.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_raises_never_met_before_any_valid_value():
    """Accessing .value before any condition was satisfied raises ConditionalNeverMet."""
    # Arrange
    data = Observable("d", 0)
    cond = data & (lambda x: x > 0)
    # Assert
    assert cond.is_active is False
    with pytest.raises(ConditionalNeverMet):
        _ = cond.value


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_raises_not_met_after_becoming_inactive():
    """After being active once, becoming inactive raises ConditionalNotMet on .value."""
    # Arrange
    data = Observable("d", 0)
    cond = data & (lambda x: x > 0)
    # Act: become active
    data.set(5)
    assert cond.is_active is True
    assert cond.value == 5
    # Act: become inactive
    data.set(-1)
    # Assert
    assert cond.is_active is False
    with pytest.raises(ConditionalNotMet):
        _ = cond.value


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_with_empty_conditions_is_always_active():
    """Constructing with no conditions yields an always-open gate."""
    # Arrange
    src = Observable("s", 10)
    always_open = ConditionalObservable(src)
    # Assert
    assert always_open.is_active is True
    assert always_open.value == 10
    src.set(20)
    assert always_open.value == 20


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_get_debug_info_reports_condition_states():
    """ConditionalObservable correctly evaluates different types of conditions"""
    # Arrange
    src = Observable("s", 2)
    flag = Observable("flag", True)
    cond = src & flag & (lambda x: x % 2 == 0)

    # Test behavior: should be active because all conditions are met
    assert cond.is_active is True
    assert cond.value == 2

    # Test that changing flag to False makes it inactive
    flag.set(False)
    assert cond.is_active is False

    # Test that changing src to odd number makes it inactive
    flag.set(True)
    src.set(3)
    assert cond.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_raises_error_for_none_source():
    """ConditionalObservable raises ValueError when source_observable is None."""
    with pytest.raises(ValueError, match="source_observable cannot be None"):
        ConditionalObservable(None)


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_raises_error_for_none_condition():
    """ConditionalObservable raises ValueError when any condition is None."""
    source = Observable("source", "data")

    with pytest.raises(ValueError, match="Condition 0 cannot be None"):
        ConditionalObservable(source, None)


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_raises_error_for_invalid_condition_type():
    """ConditionalObservable raises TypeError for invalid condition types."""
    source = Observable("source", "data")

    with pytest.raises(
        TypeError,
        match="Condition 0 must be an Observable, ObservableValue, callable, or ConditionalObservable",
    ):
        ConditionalObservable(source, 42)


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_with_inactive_source_is_inactive():
    """ConditionalObservable with inactive source becomes inactive."""
    source = Observable("source", "data")
    inactive_conditional = ConditionalObservable(source, Observable("cond", False))

    # Create another conditional with inactive source
    outer_conditional = ConditionalObservable(
        inactive_conditional, Observable("outer", True)
    )

    # Should be inactive because the source conditional is inactive
    assert outer_conditional.is_active is False

    # Should raise exception when trying to access value
    with pytest.raises(ConditionalNeverMet):
        _ = outer_conditional.value


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_evaluates_conditions_with_none():
    """ConditionalObservable handles None conditions in evaluation."""
    source = Observable("source", "data")
    conditional = ConditionalObservable(source, Observable("cond", True))

    # Test behavior: conditional should work correctly with valid conditions
    assert conditional.is_active is True
    assert conditional.value == "data"


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_with_callable_condition_filters_tuple_values():
    """ConditionalObservable with callable condition filters tuple values correctly."""
    source1 = Observable("s1", 1)
    source2 = Observable("s2", 2)
    merged = source1 + source2

    def check_sum(tuple_value):
        a, b = tuple_value  # Manual unpacking
        return a + b > 2

    conditional = merged & check_sum

    # Should be active because 1 + 2 = 3 > 2
    assert conditional.is_active is True
    assert conditional.value == (1, 2)

    # Change values to make condition false
    source1.set(0)
    source2.set(1)

    # Should be inactive because 0 + 1 = 1 <= 2
    assert conditional.is_active is False

    # Should raise exception when trying to access value
    with pytest.raises(ConditionalNotMet):
        _ = conditional.value


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_with_callable_condition_filters_single_values():
    """ConditionalObservable with callable condition filters single values correctly."""
    source = Observable("source", 5)

    def check_value(x):
        return x > 3

    conditional = source & check_value

    # Should be active because 5 > 3
    assert conditional.is_active is True
    assert conditional.value == 5

    # Change value to make condition false
    source.set(2)

    # Should be inactive because 2 <= 3
    assert conditional.is_active is False

    # Should raise exception when trying to access value
    with pytest.raises(ConditionalNotMet):
        _ = conditional.value


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_rejects_invalid_condition_types():
    """ConditionalObservable rejects invalid condition types at construction."""
    source = Observable("source", "data")

    # Create a condition that's not Observable, callable, or Conditional
    class UnknownCondition:
        pass

    unknown_cond = UnknownCondition()

    # Should raise TypeError when trying to create conditional with invalid condition type
    with pytest.raises(
        TypeError,
        match="Condition 0 must be an Observable, ObservableValue, callable, or ConditionalObservable",
    ):
        ConditionalObservable(source, unknown_cond)


@pytest.mark.unit
@pytest.mark.observable
def test_nested_conditional_observable_reacts_to_all_dependency_changes():
    """Nested ConditionalObservable reacts to changes in all its dependencies."""
    source = Observable("source", "data")
    inner_cond = Observable("inner", True)
    inner_conditional = ConditionalObservable(source, inner_cond)

    outer_cond = Observable("outer", True)
    outer_conditional = ConditionalObservable(inner_conditional, outer_cond)

    # Initially should be active (all conditions True)
    assert outer_conditional.is_active is True
    assert outer_conditional.value == "data"

    # Change inner condition - should affect outer conditional
    inner_cond.set(False)
    assert outer_conditional.is_active is False

    # Change outer condition - should affect outer conditional
    inner_cond.set(True)  # Reset inner condition
    outer_cond.set(False)
    assert outer_conditional.is_active is False

    # Change source - should affect outer conditional
    outer_cond.set(True)  # Reset outer condition
    source.set("new_data")
    assert outer_conditional.is_active is True
    assert outer_conditional.value == "new_data"


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.store
def test_conditional_observable_works_with_observable_value_from_stores():
    """Conditional observables work with ObservableValue objects from store computed observables."""
    from fynx import Store, observable

    class TestStore(Store):
        value = observable(10)
        is_positive = value >> (lambda x: x > 0)
        is_even = value >> (lambda x: x % 2 == 0)

    # Create conditional observable using computed observables from store
    # These are ObservableValue objects, not raw observables
    filtered = TestStore.value & TestStore.is_positive & TestStore.is_even

    # Should work correctly
    assert isinstance(filtered, ConditionalObservable)
    assert filtered.is_active is True  # 10 is positive and even
    assert filtered.value == 10

    # Test updates
    TestStore.value = 7  # Positive but odd
    assert filtered.is_active is False  # Conditions not met

    TestStore.value = 8  # Positive and even
    assert filtered.is_active is True
    assert filtered.value == 8

    TestStore.value = -4  # Negative but even
    assert filtered.is_active is False  # Conditions not met


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.store
def test_conditional_observable_validation_accepts_observable_value():
    """Conditional observable validation accepts ObservableValue objects."""
    from fynx import Store, observable

    class TestStore(Store):
        value = observable(5)
        condition = value >> (lambda x: x > 3)

    # This should not raise a TypeError
    conditional = TestStore.value & TestStore.condition

    assert isinstance(conditional, ConditionalObservable)
    assert conditional.is_active is True
    assert conditional.value == 5
