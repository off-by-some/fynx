"""Tests for computed observables and transformation operations."""

from fynx import Observable, computed


def test_computed_single_observable_transformation():
    """Test computed observable with single observable input."""
    base = Observable("base", 5)
    doubled = computed(lambda x: x * 2, base)

    assert doubled.value == 10

    base.set(10)
    assert doubled.value == 20


def test_computed_single_observable_method_chaining():
    """Test that computed returns an Observable instance."""
    obs = Observable("test", 1)
    result = computed(lambda x: x + 1, obs)

    assert isinstance(result, Observable)
    assert result.value == 2


def test_computed_single_observable_eager_evaluation():
    """Test that computed observable is evaluated eagerly upon creation."""
    base = Observable("base", 1)
    call_count = 0

    def expensive_computation(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    computed_obs = computed(expensive_computation, base)

    # Should compute immediately upon creation
    assert call_count == 1
    assert computed_obs.value == 2

    # Subsequent accesses don't recompute (cached)
    assert computed_obs.value == 2
    assert call_count == 1


def test_computed_single_observable_updates_on_change():
    """Test that computed observable updates when source changes."""
    source = Observable("source", 3)
    squared = computed(lambda x: x**2, source)

    assert squared.value == 9

    source.set(4)
    assert squared.value == 16

    source.set(5)
    assert squared.value == 25


def test_computed_merged_observables_transformation():
    """Test computed observable with merged observables input."""
    obs1 = Observable("obs1", 2)
    obs2 = Observable("obs2", 3)
    merged = obs1 | obs2

    summed = computed(lambda a, b: a + b, merged)

    assert summed.value == 5

    obs1.set(5)
    assert summed.value == 8

    obs2.set(10)
    assert summed.value == 15


def test_computed_merged_observables_multiple_parameters():
    """Test computed with merged observables using different operations."""
    width = Observable("width", 10)
    height = Observable("height", 5)
    dimensions = width | height

    area = computed(lambda w, h: w * h, dimensions)
    perimeter = computed(lambda w, h: 2 * (w + h), dimensions)

    assert area.value == 50
    assert perimeter.value == 30

    width.set(20)
    assert area.value == 100
    assert perimeter.value == 50


def test_computed_chaining_with_rshift_operator():
    """Test computed observable chaining using >> operator."""
    base = Observable("base", 2)

    # Chain multiple transformations
    result = base >> (lambda x: x + 1) >> (lambda x: x * 3) >> (lambda x: x - 2)

    # ((2 + 1) * 3) - 2 = 7
    assert result.value == 7

    base.set(3)
    # ((3 + 1) * 3) - 2 = 10
    assert result.value == 10


def test_computed_rshift_operator_creates_new_observable():
    """Test that >> operator creates new Observable instances."""
    obs = Observable("test", 1)

    # Each >> should create a new observable
    step1 = obs >> (lambda x: x + 1)
    step2 = step1 >> (lambda x: x * 2)

    assert isinstance(step1, Observable)
    assert isinstance(step2, Observable)
    assert step1 is not obs
    assert step2 is not obs
    assert step2 is not step1

    assert step1.value == 2
    assert step2.value == 4


def test_computed_rshift_with_merged_observables():
    """Test >> operator with merged observables."""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    merged = obs1 | obs2

    # Create computed observable using >>
    result = merged >> (lambda a, b: a + b)

    assert result.value == 3

    obs1.set(10)
    assert result.value == 12

    obs2.set(5)
    assert result.value == 15


def test_computed_rshift_method_chaining():
    """Test that >> operator returns observable for method chaining."""
    obs = Observable("test", 1)

    result = obs >> (lambda x: x * 2) >> (lambda x: x + 1)

    # Should be able to call subscribe on the result
    call_count = 0

    def callback(value):
        nonlocal call_count
        call_count += 1

    result.subscribe(callback)
    obs.set(2)

    assert call_count == 1


def test_computed_initial_computation():
    """Test that computed observables are computed immediately upon creation."""
    source = Observable("source", 1)

    computed_values = []

    def track_computation(x):
        computed_values.append(x * 2)
        return x * 2

    computed_obs = computed(track_computation, source)

    # Should compute immediately
    assert len(computed_values) == 1
    assert computed_values[0] == 2
    assert computed_obs.value == 2


def test_computed_no_recomputation_when_value_unchanged():
    """Test that computed observables don't recompute when source value doesn't change."""
    source = Observable("source", 5)
    computation_count = 0

    def counting_computation(x):
        nonlocal computation_count
        computation_count += 1
        return x * 2

    computed_obs = computed(counting_computation, source)

    assert computation_count == 1  # Initial computation
    assert computed_obs.value == 10

    # Setting same value should not trigger recomputation
    source.set(5)
    assert computation_count == 1  # Still only 1 computation

    # Setting different value should trigger recomputation
    source.set(6)
    assert computation_count == 2
    assert computed_obs.value == 12


def test_computed_with_mixed_observable_types():
    """Test computed observables with different value types."""
    string_obs = Observable("text", "hello")
    number_obs = Observable("number", 3)
    bool_obs = Observable("bool", True)

    # String transformation
    upper = computed(lambda s: s.upper(), string_obs)
    assert upper.value == "HELLO"

    # Number transformation
    squared = computed(lambda n: n**2, number_obs)
    assert squared.value == 9

    # Boolean transformation
    negated = computed(lambda b: not b, bool_obs)
    assert negated.value is False


def test_computed_error_in_computation_function():
    """Test behavior when computation function throws an error."""
    source = Observable("source", 1)

    def failing_computation(x):
        if x == 2:
            raise ValueError("Test error")
        return x * 2

    computed_obs = computed(failing_computation, source)

    # Initial computation should work
    assert computed_obs.value == 2

    # Setting value that causes error should propagate the error
    # (computed observables don't handle errors - they propagate them)
    try:
        source.set(2)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "Test error"


def test_computed_with_none_values():
    """Test computed observables handle None values correctly."""
    obs = Observable("test", None)
    computed_obs = computed(lambda x: x is None, obs)

    assert computed_obs.value is True

    obs.set("not_none")
    assert computed_obs.value is False

    obs.set(None)
    assert computed_obs.value is True


def test_mixed_type_observables_in_computed():
    """Test computed observables with mixed types."""
    str_obs = Observable("string", "hello")
    int_obs = Observable("int", 5)
    merged = str_obs | int_obs

    # Create computed that combines different types
    combined = computed(lambda s, i: f"{s}_{i}", merged)

    assert combined.value == "hello_5"

    str_obs.set("world")
    assert combined.value == "world_5"

    int_obs.set(10)
    assert combined.value == "world_10"
