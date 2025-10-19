"""Tests for watch decorator functionality."""

import io
from contextlib import redirect_stdout

from fynx import observable, watch


def test_watch_single_condition_true():
    """Test watch decorator executes when single condition becomes true."""
    is_ready = observable(False)
    callback_executed = False

    @watch(lambda: is_ready.value)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    assert not callback_executed

    is_ready.set(True)
    assert callback_executed


def test_watch_single_condition_false_to_true():
    """Test watch decorator with single condition changing from false to true."""
    flag = observable(False)
    execution_count = 0

    @watch(lambda: flag.value)
    def callback():
        nonlocal execution_count
        execution_count += 1

    flag.set(True)
    assert execution_count == 1

    # Setting to false should not trigger again
    flag.set(False)
    assert execution_count == 1

    # Setting to true again should trigger
    flag.set(True)
    assert execution_count == 2


def test_watch_multiple_conditions_all_true():
    """Test watch decorator with multiple conditions that are all true."""
    cond1 = observable(True)
    cond2 = observable(True)
    callback_executed = False

    @watch(lambda: cond1.value, lambda: cond2.value)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # Should execute immediately since all conditions are true
    assert callback_executed


def test_watch_multiple_conditions_partial_true():
    """Test watch decorator when only some conditions are initially true."""
    cond1 = observable(True)
    cond2 = observable(False)
    callback_executed = False

    @watch(lambda: cond1.value, lambda: cond2.value)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    assert not callback_executed

    cond2.set(True)
    assert callback_executed


def test_watch_multiple_conditions_becomes_false():
    """Test watch decorator when conditions become unmet."""
    cond1 = observable(True)
    cond2 = observable(True)
    execution_count = 0

    @watch(lambda: cond1.value, lambda: cond2.value)
    def callback():
        nonlocal execution_count
        execution_count += 1

    assert execution_count == 1  # Initial execution

    cond1.set(False)
    assert execution_count == 1  # No additional execution

    cond1.set(True)
    assert execution_count == 2  # Conditions met again


def test_watch_observable_discovery_identifies_used_observables():
    """Test that watch discovers only observables used in conditions."""
    used_obs = observable("used")
    unused_obs = observable("unused")
    callback_count = 0

    @watch(lambda: used_obs.value == "trigger")
    def callback():
        nonlocal callback_count
        callback_count += 1

    assert callback_count == 0

    # Changing unused observable should not trigger
    unused_obs.set("changed")
    assert callback_count == 0

    # Changing used observable to meet condition should trigger
    used_obs.set("trigger")
    assert callback_count == 1


def test_watch_observable_discovery_complex_conditions():
    """Test watch discovers observables in complex condition expressions."""
    name = observable("Alice")
    age = observable(25)
    active = observable(True)
    unused = observable("ignored")

    callback_count = 0

    @watch(lambda: name.value == "Bob", lambda: age.value > 20, lambda: active.value)
    def callback():
        nonlocal callback_count
        callback_count += 1

    assert callback_count == 0  # name != "Bob"

    # Change unused observable - should not trigger
    unused.set("changed")
    assert callback_count == 0

    # Make name condition true - should trigger
    name.set("Bob")
    assert callback_count == 1

    # Change age (still > 20) - conditions still met, no additional trigger
    age.set(30)
    assert callback_count == 1  # No additional trigger on same condition state

    # Make age condition false - conditions no longer met
    age.set(15)
    assert callback_count == 1  # No trigger

    # Make age condition true again - conditions become true, trigger callback
    age.set(25)
    assert callback_count == 2


def test_watch_initial_execution_when_conditions_met():
    """Test that watch executes immediately when initial conditions are met."""
    counter = observable(10)
    callback_executed = False

    @watch(lambda: counter.value > 5)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # Should execute immediately since 10 > 5
    assert callback_executed


def test_watch_initial_execution_when_conditions_not_met():
    """Test that watch does not execute immediately when initial conditions are not met."""
    counter = observable(3)
    callback_executed = False

    @watch(lambda: counter.value > 5)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # Should not execute immediately since 3 is not > 5
    assert not callback_executed

    # But should execute when condition becomes met
    counter.set(7)
    assert callback_executed


def test_watch_no_execution_on_same_value():
    """Test that watch doesn't execute when observable value doesn't change."""
    obs = observable("value")
    callback_count = 0

    @watch(lambda: obs.value == "target")
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Setting same value should not trigger
    obs.set("value")
    assert callback_count == 0

    # Setting target value should trigger
    obs.set("target")
    assert callback_count == 1


def test_watch_error_propagation_in_condition_evaluation():
    """Test that watch propagates errors during condition evaluation."""
    value = observable(5)
    error_obs = observable(None)
    callback_executed = False

    # Should raise an exception during watch decorator setup due to error in condition
    try:

        @watch(lambda: value.value > 3, lambda: error_obs.value.nonexistent_attribute)
        def callback():
            nonlocal callback_executed
            callback_executed = True

        # If we get here, the test failed
        assert False, "Expected AttributeError to be raised during watch setup"
    except AttributeError:
        # Expected - error in condition evaluation now propagates
        pass

    # Callback should not have been executed
    assert not callback_executed


def test_watch_error_propagation_during_discovery():
    """Test that watch propagates errors during condition discovery."""
    error_obs = observable(None)

    # Should raise an exception during watch decorator setup due to error in condition discovery
    try:

        @watch(lambda: error_obs.value.nonexistent_attribute)
        def callback():
            pass

        # If we get here, the test failed
        assert False, "Expected AttributeError to be raised during watch discovery"
    except AttributeError:
        # Expected - error during condition discovery now propagates
        pass


def test_watch_returns_original_function():
    """Test that watch decorator returns the original function."""

    def test_function():
        return "test"

    decorated = watch(lambda: True)(test_function)
    assert decorated is test_function


def test_watch_empty_conditions_list():
    """Test watch decorator with no conditions (edge case)."""
    obs = observable("test")
    callback_executed = False

    # Empty conditions - should this execute immediately or never?
    @watch()
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # With no conditions, it's ambiguous what should happen
    # For now, assume it executes immediately (all conditions met when none exist)
    assert callback_executed


def test_watch_condition_with_complex_expressions():
    """Test watch with complex condition expressions."""
    a = observable(1)
    b = observable(2)
    c = observable(3)

    callback_count = 0

    @watch(lambda: (a.value + b.value) > c.value)
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Initially: (1 + 2) > 3 = 3 > 3 = False
    assert callback_count == 0

    # Make condition true: (1 + 2) > 2 = 3 > 2 = True
    c.set(2)
    assert callback_count == 1

    # Change a: (2 + 2) > 2 = 4 > 2 = True (already true, no additional trigger)
    a.set(2)
    assert callback_count == 1  # No additional trigger on same condition state

    # Make condition false: (2 + 2) > 4 = 4 > 4 = False
    c.set(4)
    assert callback_count == 1

    # Make condition true again: (2 + 2) > 3 = 4 > 3 = True
    c.set(3)
    assert callback_count == 2


def test_watch_multiple_decorators_on_same_function():
    """Test that multiple watch decorators can be applied to the same function."""
    obs1 = observable(False)
    obs2 = observable(False)

    call_sequence = []

    @watch(lambda: obs1.value)
    @watch(lambda: obs2.value)
    def callback():
        call_sequence.append("called")

    # Initially neither condition is met
    assert call_sequence == []

    # Setting obs2 to True should trigger
    obs2.set(True)
    assert len(call_sequence) == 1

    # Setting obs1 to True should trigger again
    obs1.set(True)
    assert len(call_sequence) == 2


def test_watch_single_conditional_observable():
    """Test watch decorator with a single ConditionalObservable."""
    obs1 = observable(False)
    obs2 = observable(False)
    callback_count = 0

    @watch(obs1 & obs2)
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Initially conditions not met
    assert callback_count == 0

    # Make first condition true - still not all met
    obs1.set(True)
    assert callback_count == 0

    # Make second condition true - all conditions met, should trigger
    obs2.set(True)
    assert callback_count == 1

    # Change obs1 from truthy to falsy and back - should trigger on the transition to truthy
    obs1.set(False)
    obs1.set(True)
    assert callback_count == 2

    # Make conditions unmet then met again - should trigger
    obs2.set(False)
    obs2.set(True)
    assert callback_count == 3


def test_watch_multiple_conditions():
    """Test watch decorator with multiple conditions (ConditionalObservable + lambda)."""
    obs1 = observable(False)
    obs2 = observable(False)
    obs3 = observable(False)
    callback_count = 0

    @watch(obs1 & obs2, lambda: obs3.value)  # ConditionalObservable + lambda
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Initially no conditions met
    assert callback_count == 0

    # Make obs3 true - still missing obs1 & obs2
    obs3.set(True)
    assert callback_count == 0

    # Make obs1 & obs2 true - all conditions met, should trigger
    obs1.set(True)
    obs2.set(True)
    assert callback_count == 1

    # Change obs1 to false - conditions no longer met
    obs1.set(False)
    assert callback_count == 1

    # Make obs1 true again - conditions met again, should trigger
    obs1.set(True)
    assert callback_count == 2


def test_watch_mixed_conditional_and_lambda():
    """Test watch decorator with ConditionalObservable and lambda conditions."""
    obs1 = observable(False)
    obs2 = observable(False)
    num_obs = observable(5)
    callback_count = 0

    @watch(obs1 & obs2, lambda: num_obs.value > 10)
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Initially conditions not met (num_obs = 5, not > 10)
    assert callback_count == 0

    # Make num_obs condition true - still missing obs1 & obs2
    num_obs.set(15)
    assert callback_count == 0

    # Make obs1 & obs2 true - all conditions met, should trigger
    obs1.set(True)
    obs2.set(True)
    assert callback_count == 1

    # Change num_obs while conditions still met - no additional trigger
    num_obs.set(20)
    assert callback_count == 1

    # Make num_obs condition false - conditions no longer met
    num_obs.set(5)
    assert callback_count == 1

    # Make num_obs condition true again - conditions met again, should trigger
    num_obs.set(15)
    assert callback_count == 2


def test_watch_conditional_observable_initial_execution():
    """Test that watch executes immediately when ConditionalObservable conditions are initially met."""
    obs1 = observable(True)  # Already true
    obs2 = observable(True)  # Already true
    callback_executed = False

    @watch(obs1 & obs2)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # Should execute immediately since both conditions are already true
    assert callback_executed


def test_watch_conditional_observable_no_initial_execution():
    """Test that watch does not execute immediately when ConditionalObservable conditions are not initially met."""
    obs1 = observable(True)
    obs2 = observable(False)  # Not true initially
    callback_executed = False

    @watch(obs1 & obs2)
    def callback():
        nonlocal callback_executed
        callback_executed = True

    # Should not execute initially since obs2 is False
    assert not callback_executed

    # But should execute when obs2 becomes true
    obs2.set(True)
    assert callback_executed


def test_watch_conditional_observable_transition_behavior():
    """Test that watch with ConditionalObservable only triggers on transitions, not on ongoing changes."""
    obs1 = observable(False)
    obs2 = observable(False)
    data_obs = observable("initial")
    callback_count = 0

    @watch(obs1 & obs2)
    def callback():
        nonlocal callback_count
        callback_count += 1

    # Make conditions met
    obs1.set(True)
    obs2.set(True)
    assert callback_count == 1

    # Change data_obs while conditions still met - should NOT trigger additional times
    data_obs.set("changed1")
    data_obs.set("changed2")
    data_obs.set("changed3")
    assert callback_count == 1  # Still only 1

    # Make conditions unmet then met again - should trigger
    obs1.set(False)
    obs1.set(True)
    assert callback_count == 2
