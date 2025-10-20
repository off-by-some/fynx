"""Unit tests for conditional observable implementation specifics."""

import pytest

from fynx.observable import Observable
from fynx.observable.conditional import ConditionalObservable


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
def test_conditional_observable_stores_source_observable_reference():
    """ConditionalObservable stores reference to source observable"""
    source = Observable("source", "data")
    condition = Observable("condition", True)

    conditional = source & condition

    # Implementation detail: stores source reference
    assert conditional._source_observable is source


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_stores_conditions():
    """ConditionalObservable stores conditions (may be nested for optimization)"""
    source = Observable("source", "data")
    cond1 = Observable("cond1", True)
    cond2 = Observable("cond2", False)

    conditional = source & cond1 & cond2

    # Check that all conditions are present in the structure
    all_conditions = []
    current = conditional
    while isinstance(current, ConditionalObservable):
        all_conditions.extend(current._conditions)
        current = current._source_observable
    assert set(all_conditions) == {cond1, cond2}

    # Check that the root source is correct
    root_source = conditional._source_observable
    while isinstance(root_source, ConditionalObservable):
        root_source = root_source._source_observable
    assert root_source is source


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_conditional_observable_tracks_condition_satisfaction_state():
    """ConditionalObservable tracks whether its own conditions are satisfied"""
    source = Observable("source", "data")
    cond1 = Observable("cond1", True)  # Always true for inner conditional
    cond2 = Observable("cond2", False)  # Controls outer conditional

    conditional = source & cond1 & cond2

    # Outer conditional (cond2) initially not satisfied
    assert conditional._conditions_met is False

    # Make outer condition satisfied
    cond2.set(True)
    assert conditional._conditions_met is True

    # Make outer condition unsatisfied again
    cond2.set(False)
    assert conditional._conditions_met is False


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
    from fynx.observable.conditional import ConditionalNeverMet

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
    """& operator chaining creates flat ConditionalObservable structure"""
    source = Observable("source", "data")
    cond1 = Observable("cond1", True)
    cond2 = Observable("cond2", True)

    conditional = source & cond1 & cond2

    # The result should be a ConditionalObservable
    assert isinstance(conditional, ConditionalObservable)

    # The structure may be nested for optimization purposes
    # Check that the root source is the original observable
    root_source = conditional._source_observable
    while isinstance(root_source, ConditionalObservable):
        root_source = root_source._source_observable
    assert root_source is source

    # Check that all conditions are present in the chain
    all_conditions = []
    current = conditional
    while isinstance(current, ConditionalObservable):
        all_conditions.extend(current._conditions)
        current = current._source_observable
    assert set(all_conditions) == {cond1, cond2}


import pytest

from fynx.observable.base import Observable
from fynx.observable.conditional import (
    ConditionalNeverMet,
    ConditionalNotMet,
    ConditionalObservable,
)


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
    """get_debug_info returns per-condition state for observables and callables."""
    # Arrange
    src = Observable("s", 2)
    flag = Observable("flag", True)
    cond = src & flag & (lambda x: x % 2 == 0)
    # Act
    info = cond.get_debug_info()
    # Assert
    assert info["is_active"] is True
    assert info["conditions_count"] == 2
    assert any(
        cs["type"] == "Observable" for cs in info["condition_states"]
    )  # boolean obs
    assert any(cs["type"] == "Callable" for cs in info["condition_states"])  # callable


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
        match="Condition 0 must be an Observable, callable, or ConditionalObservable",
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

    # Test with None conditions (should use processed conditions)
    result = conditional._evaluate_all_conditions("data", None)
    assert result is True


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_with_callable_condition_filters_tuple_values():
    """ConditionalObservable with callable condition filters tuple values correctly."""
    source1 = Observable("s1", 1)
    source2 = Observable("s2", 2)
    merged = source1 | source2

    def check_sum(a, b):
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
        match="Condition 0 must be an Observable, callable, or ConditionalObservable",
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
def test_conditional_observable_extract_condition_dependencies_returns_empty():
    """ConditionalObservable._extract_condition_dependencies returns empty set (deprecated)."""
    source = Observable("source", "data")
    conditional = ConditionalObservable(source, Observable("cond", True))

    # Deprecated method should return empty set
    result = conditional._extract_condition_dependencies(Observable("test", True))
    assert result == set()


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_handles_transition_to_inactive():
    """ConditionalObservable handles transition from active to inactive without notification."""
    source = Observable("source", 5)
    condition = Observable("condition", True)
    conditional = source & condition

    # Make it active first
    source.set(5)
    assert conditional.is_active is True
    assert conditional.value == 5

    # Now make condition false - should transition to inactive
    condition.set(False)
    assert conditional.is_active is False

    # Should not notify observers when transitioning to inactive
    notifications = []
    conditional.subscribe(lambda val: notifications.append(val))

    # Change source while inactive - should not notify
    source.set(10)
    assert len(notifications) == 0


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_debug_info_handles_callable_with_tuple():
    """get_debug_info handles callable conditions with tuple source values."""
    source1 = Observable("s1", 1)
    source2 = Observable("s2", 2)
    merged = source1 | source2

    def check_values(a, b):
        return a + b > 2

    conditional = merged & check_values

    # Get debug info
    info = conditional.get_debug_info()

    # Should have callable condition info
    callable_conditions = [
        cs for cs in info["condition_states"] if cs["type"] == "Callable"
    ]
    assert len(callable_conditions) == 1
    assert callable_conditions[0]["result"] is True  # 1 + 2 > 2


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_debug_info_handles_callable_with_single_value():
    """get_debug_info handles callable conditions with single source values."""
    source = Observable("source", 5)

    def check_value(x):
        return x > 3

    conditional = source & check_value

    # Get debug info
    info = conditional.get_debug_info()

    # Should have callable condition info
    callable_conditions = [
        cs for cs in info["condition_states"] if cs["type"] == "Callable"
    ]
    assert len(callable_conditions) == 1
    assert callable_conditions[0]["result"] is True  # 5 > 3


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_evaluate_single_condition_conditional():
    """Test _evaluate_single_condition() with Conditional condition (line 473)"""
    source = Observable("source", 5)

    # Create a nested conditional
    inner_conditional = source & (lambda x: x > 3)

    # Create outer conditional that depends on inner conditional
    outer_conditional = source & inner_conditional

    # The inner conditional should be active (5 > 3)
    assert inner_conditional.is_active

    # Test evaluation of conditional condition
    result = outer_conditional._evaluate_single_condition(inner_conditional, 5)
    assert result is True


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_evaluate_single_condition_unknown_type():
    """Test _evaluate_single_condition() with unknown condition type (line 519)"""
    source = Observable("source", 5)

    # Create a valid conditional first
    conditional = ConditionalObservable(source, lambda x: x > 3)

    # Create unknown condition type
    class UnknownCondition:
        pass

    unknown_condition = UnknownCondition()

    # Test the method directly with unknown condition type
    # This should treat unknown condition as falsy
    result = conditional._evaluate_single_condition(unknown_condition, 5)
    assert result is False


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_debug_info_handles_callable_with_tuple_source():
    """Test get_debug_info() with callable condition and tuple source (line 698)"""
    source1 = Observable("s1", 1)
    source2 = Observable("s2", 2)
    merged = source1 | source2

    def check_tuple(a, b):
        return a + b > 2

    conditional = merged & check_tuple

    # Get debug info
    info = conditional.get_debug_info()

    # Should handle tuple source values in callable conditions
    callable_conditions = [
        cs for cs in info["condition_states"] if cs["type"] == "Callable"
    ]
    assert len(callable_conditions) == 1
    assert callable_conditions[0]["result"] is True  # 1 + 2 > 2


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_observable_debug_info_handles_callable_with_single_source():
    """Test get_debug_info() with callable condition and single source (line 714->696)"""
    source = Observable("source", 5)

    def check_single(x):
        return x > 3

    conditional = source & check_single

    # Get debug info
    info = conditional.get_debug_info()

    # Should handle single source values in callable conditions
    callable_conditions = [
        cs for cs in info["condition_states"] if cs["type"] == "Callable"
    ]
    assert len(callable_conditions) == 1
    assert callable_conditions[0]["result"] is True  # 5 > 3
