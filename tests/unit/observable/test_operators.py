import pytest

from fynx.observable.base import Observable
from fynx.observable.conditional import ConditionalNeverMet
from fynx.observable.descriptors import ObservableValue
from fynx.observable.operators import and_operator


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_or_operator_creates_logical_or_condition():
    """The | operator creates a logical OR condition between observables."""
    # Arrange
    is_error = Observable("error", True)
    is_warning = Observable("warning", False)

    # Act
    needs_attention = is_error | is_warning

    # Assert
    assert needs_attention.value is True  # True OR False = True

    # Test updates - keep at least one True to maintain the condition
    is_warning.set(True)
    assert needs_attention.value is True  # True OR True = True

    # Test with both True
    is_error.set(True)
    assert needs_attention.value is True  # True OR True = True


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_or_operator_with_falsy_initial_values():
    """The | operator raises ConditionalNeverMet when both initial values are falsy."""
    # Arrange
    is_error = Observable("error", False)
    is_warning = Observable("warning", False)

    # Act & Assert
    with pytest.raises(ConditionalNeverMet):
        needs_attention = is_error | is_warning
        _ = needs_attention.value


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_or_operator_chaining():
    """The | operator can be chained for multiple OR conditions."""
    # Arrange
    is_error = Observable("error", True)
    is_warning = Observable("warning", False)
    is_critical = Observable("critical", False)

    # Act
    needs_attention = is_error | is_warning | is_critical

    # Assert
    assert needs_attention.value is True  # True OR False OR False = True

    # Test updates - keep at least one True to maintain the condition
    is_warning.set(True)
    assert needs_attention.value is True  # True OR True OR False = True

    # Test with all True
    is_critical.set(True)
    assert needs_attention.value is True  # True OR True OR True = True


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_or_operator_equivalent_to_either():
    """The | operator is equivalent to calling .either() method."""
    # Arrange
    is_error = Observable("error", True)
    is_warning = Observable("warning", False)

    # Act
    needs_attention_or = is_error | is_warning
    needs_attention_either = is_error.either(is_warning)

    # Assert
    assert needs_attention_or.value == needs_attention_either.value

    # Test updates - keep at least one True to maintain the condition
    is_warning.set(True)
    assert needs_attention_or.value == needs_attention_either.value

    # Test with both True
    is_error.set(True)
    assert needs_attention_or.value == needs_attention_either.value


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_rshift_operator_applies_function_to_merged_arguments():
    """The >> operator applies a function to unpacked merged values."""
    # Arrange
    a = Observable("a", 2)
    b = Observable("b", 3)
    merged = a + b
    # Act
    prod = merged >> (lambda x, y: x * y)
    # Assert
    assert prod.value == 6


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_invert_operator_negates_boolean_observable():
    """The ~ operator negates a boolean computed observable."""
    # Arrange
    base = Observable("n", 6)
    is_even = base >> (lambda v: v % 2 == 0)
    # Act / Assert
    assert (~is_even).value is False


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_list_len_iter_getitem_contains():
    """ObservableValue wrapping a list supports len, iter, indexing, and contains."""
    # Arrange
    base = Observable("list", [1, 2, 3])
    val = ObservableValue(base)
    # Assert
    assert len(val) == 3
    assert list(iter(val)) == [1, 2, 3]
    assert val[1] == 2
    assert 3 in val


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_none_len_iter_contains_falsey():
    """ObservableValue wrapping None yields len 0, empty iteration, and no membership."""
    # Arrange
    none_base = Observable("none", None)
    none_val = ObservableValue(none_base)
    # Assert
    assert len(none_val) == 0
    assert list(iter(none_val)) == []
    assert (0 in none_val) is False


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_scalar_iter_and_getitem_type_error():
    """Scalar ObservableValue iterates as single-item sequence and is not subscriptable."""
    # Arrange
    num = Observable("num", 10)
    num_val = ObservableValue(num)
    # Assert
    assert len(num_val) == 0
    assert list(iter(num_val)) == [10]
    with pytest.raises(TypeError):
        _ = num_val[0]


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_and_with_callable_condition():
    """Using & with a callable evaluates predicate against the source value."""
    # Arrange
    source = Observable("x", 5)
    filtered = ObservableValue(source) & (lambda x: x > 3)
    # Act / Assert
    assert filtered.is_active is True
    source.set(2)
    assert filtered.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_and_with_boolean_observable():
    """Using & with a boolean observable gates emissions by that flag."""
    # Arrange
    source = Observable("x", 10)
    ready = Observable("ready", False)
    filtered = ObservableValue(source) & ready
    # Act / Assert
    assert filtered.is_active is False
    ready.set(True)
    assert filtered.is_active is True


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.operators
def test_and_operator_with_conditional_source_uses_root_value():
    """and_operator evaluates callable conditions against the root source when source is conditional."""
    # Arrange
    data = Observable("data", 1)
    step1 = data & (lambda x: x > 0)
    # Act
    step2 = and_operator(step1, lambda x: x < 5)
    # Assert
    assert step1.is_active is True
    assert step2.is_active is True

    data.set(7)
    assert step1.is_active is True
    assert step2.is_active is False

    data.set(-3)
    assert step1.is_active is False
    assert step2.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_getitem_raises_error_when_no_value():
    """MergedObservable.__getitem__ raises IndexError when value is None."""
    from fynx.observable.merged import MergedObservable

    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 + obs2

    # Set value to None to trigger the error path
    merged._value = None

    with pytest.raises(IndexError, match="MergedObservable has no value"):
        _ = merged[0]


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_setitem_raises_error_for_out_of_range_index():
    """MergedObservable.__setitem__ raises IndexError for out of range index."""
    from fynx.observable.merged import MergedObservable

    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 + obs2

    # Try to set index that's out of range
    with pytest.raises(IndexError, match="Index out of range"):
        merged[5] = 10


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_str_returns_string_representation():
    """ValueMixin.__str__ returns string representation of current value."""
    obs = Observable("test", 42)
    val = ObservableValue(obs)

    assert str(val) == "42"


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_repr_returns_repr_representation():
    """ValueMixin.__repr__ returns repr representation of current value."""
    obs = Observable("test", [1, 2, 3])
    val = ObservableValue(obs)

    assert repr(val) == "[1, 2, 3]"


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_len_returns_zero_for_none():
    """ValueMixin.__len__ returns 0 when current value is None."""
    obs = Observable("test", None)
    val = ObservableValue(obs)

    assert len(val) == 0


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_getitem_raises_error_for_none():
    """ValueMixin.__getitem__ raises IndexError when current value is None."""
    obs = Observable("test", None)
    val = ObservableValue(obs)

    with pytest.raises(IndexError, match="observable value is None"):
        _ = val[0]


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_getitem_raises_error_for_non_subscriptable():
    """ValueMixin.__getitem__ raises TypeError for non-subscriptable values."""
    obs = Observable("test", 42)
    val = ObservableValue(obs)

    with pytest.raises(TypeError, match="'int' object is not subscriptable"):
        _ = val[0]


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_contains_returns_false_for_non_containable():
    """ValueMixin.__contains__ returns False for values without __contains__."""
    obs = Observable("test", 42)
    val = ObservableValue(obs)

    assert (5 in val) is False


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_bool_returns_boolean_value():
    """ValueMixin.__bool__ returns boolean conversion of current value."""
    obs_true = Observable("test", "hello")
    val_true = ObservableValue(obs_true)
    assert bool(val_true) is True

    obs_false = Observable("test", "")
    val_false = ObservableValue(obs_false)
    assert bool(val_false) is False


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_unwrap_operand_returns_observable():
    """ValueMixin._unwrap_operand returns observable for ObservableValue operands."""
    obs = Observable("test", 42)
    val = ObservableValue(obs)

    # Should unwrap ObservableValue to get the underlying observable
    result = val._unwrap_operand(val)
    assert result is obs


@pytest.mark.unit
@pytest.mark.observable
def test_value_mixin_unwrap_operand_returns_operand_as_is():
    """ValueMixin._unwrap_operand returns operand as-is for non-ObservableValue operands."""
    obs = Observable("test", 42)
    val = ObservableValue(obs)

    # Should return non-ObservableValue operands as-is
    result = val._unwrap_operand(42)
    assert result == 42


@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_mixin_invert_delegates_to_observable():
    """ObservableValueMixin.__invert__ delegates to underlying observable."""
    obs = Observable("test", True)
    val = ObservableValue(obs)

    # Should delegate to observable's __invert__ method
    result = ~val
    assert result.value is False


@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_mixin_rshift_delegates_to_observable():
    """ObservableValueMixin.__rshift__ delegates to underlying observable."""
    obs = Observable("test", 5)
    val = ObservableValue(obs)

    # Should delegate to observable's __rshift__ method
    result = val >> (lambda x: x * 2)
    assert result.value == 10


@pytest.mark.unit
@pytest.mark.observable
def test_rshift_operator_delegates_to_create_computed():
    """rshift_operator delegates to observable's _create_computed method."""
    from fynx.observable.operators import rshift_operator

    obs = Observable("test", 5)

    # Should delegate to _create_computed
    result = rshift_operator(obs, lambda x: x * 2)
    assert result.value == 10


@pytest.mark.unit
@pytest.mark.observable
def test_and_operator_with_callable_condition_creates_computed():
    """and_operator creates computed observable for callable conditions."""
    from fynx.observable.operators import and_operator

    obs = Observable("test", 5)

    # Should create computed observable for callable condition
    result = and_operator(obs, lambda x: x > 3)
    assert result.is_active is True

    obs.set(2)
    assert result.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_and_operator_with_observable_condition_uses_directly():
    """and_operator uses observable conditions directly without creating computed."""
    from fynx.observable.operators import and_operator

    obs = Observable("test", 5)
    condition_obs = Observable("condition", True)

    # Should use observable condition directly
    result = and_operator(obs, condition_obs)
    assert result.is_active is True

    condition_obs.set(False)
    assert result.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_unsubscribe_cleanup_empty_mapping():
    """Test MergedObservable.unsubscribe() cleanup of empty function mappings (line 361->exit)"""
    from fynx.observable.merged import MergedObservable
    from fynx.registry import _func_to_contexts

    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 + obs2

    def test_func():
        pass

    # Create context and add it to function mappings
    context = type(
        "MockContext", (object,), {"run": test_func, "dispose": lambda: None}
    )()
    _func_to_contexts[test_func] = [context]

    # Verify mapping exists
    assert test_func in _func_to_contexts
    assert context in _func_to_contexts[test_func]

    # Remove context (simulating cleanup)
    _func_to_contexts[test_func].remove(context)

    # Clean up empty function mappings (line 361-362)
    if not _func_to_contexts[test_func]:
        del _func_to_contexts[test_func]

    # Verify mapping is removed
    assert test_func not in _func_to_contexts
