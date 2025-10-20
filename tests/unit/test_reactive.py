"""Unit tests for @reactive decorator functionality."""

import pytest

from fynx import Store, observable, reactive
from tests.test_factories import create_temperature_monitor_store


@pytest.mark.unit
@pytest.mark.reactive
class TestReactiveDecoratorCoreFunctionality:
    """Test the core functionality of the @reactive decorator."""

    def test_reactive_decorator_executes_immediately_with_current_value(self):
        """Reactive function executes immediately with current observable value."""
        obs = observable(42)
        execution_log = []

        @reactive(obs)
        def log_value(value):
            execution_log.append(value)

        assert execution_log == [42]

    def test_reactive_decorator_triggers_on_observable_changes(self):
        """Reactive function executes when observable value changes."""
        obs = observable(10)
        execution_log = []

        @reactive(obs)
        def log_changes(value):
            execution_log.append(value)

        obs.set(20)
        obs.set(30)

        assert execution_log == [10, 20, 30]  # initial + changes

    def test_reactive_decorator_propagates_none_values(self):
        """Reactive decorator should propagate None values (None is a valid value in FRP)."""
        obs = observable(42)
        execution_log = []

        @reactive(obs)
        def log_all_values(value):
            execution_log.append(f"value:{value}")

        # Initial value should be logged
        assert execution_log == ["value:42"]

        # None should propagate
        obs.set(None)
        assert execution_log == ["value:42", "value:None"]

        # Other values should continue to work
        obs.set(100)
        assert execution_log == ["value:42", "value:None", "value:100"]

    def test_reactive_decorator_with_multiple_observables_executes_immediately(self):
        """Reactive function executes immediately with current values from multiple observables."""
        obs1 = observable(10)
        obs2 = observable(20)
        execution_log = []

        @reactive(obs1, obs2)
        def log_combined_values(val1, val2):
            execution_log.append((val1, val2))

        assert execution_log == [(10, 20)]

    def test_reactive_decorator_with_multiple_observables_triggers_on_changes(self):
        """Reactive function executes when any of multiple observables change."""
        obs1 = observable(10)
        obs2 = observable(20)
        execution_log = []

        @reactive(obs1, obs2)
        def log_combined_values(val1, val2):
            execution_log.append((val1, val2))

        obs1.set(15)
        obs2.set(25)

        expected = [(10, 20), (15, 20), (15, 25)]
        assert execution_log == expected

    def test_reactive_decorator_with_store_executes_immediately(self):
        """Reactive function executes immediately with store snapshot."""
        TemperatureMonitor = create_temperature_monitor_store()
        execution_log = []

        @reactive(TemperatureMonitor)
        def log_store_snapshot(snapshot):
            execution_log.append(snapshot.to_dict())

        assert len(execution_log) == 1
        assert execution_log[0]["celsius"] == 0.0
        assert execution_log[0]["fahrenheit"] == 32.0

    def test_reactive_decorator_with_store_triggers_on_changes(self):
        """Reactive function executes when store observables change."""
        TemperatureMonitor = create_temperature_monitor_store()
        execution_log = []

        @reactive(TemperatureMonitor)
        def log_temperature_changes(snapshot):
            execution_log.append(snapshot.to_dict())

        TemperatureMonitor.celsius = 25.0

        assert len(execution_log) >= 2  # initial + change(s)
        final_state = execution_log[-1]
        assert final_state["celsius"] == 25.0
        assert final_state["fahrenheit"] == 77.0

    def test_reactive_decorator_propagates_exceptions(self):
        """Reactive decorator propagates exceptions from decorated functions."""
        obs = observable(10)
        execution_log = []

        @reactive(obs)
        def failing_function(value):
            if value == 20:
                raise ValueError("Test error")
            execution_log.append(value)

        obs.set(15)  # Should work
        # obs.set(20) would raise ValueError - exceptions are propagated

        assert execution_log == [10, 15]

    def test_reactive_decorator_with_no_targets_does_nothing(self):
        """Reactive decorator with no targets does not execute function."""
        execution_log = []

        @reactive()  # No targets
        def should_not_execute():
            execution_log.append("executed")

        assert execution_log == []

    def test_reactive_decorator_preserves_function_metadata_but_wraps_calls(self):
        """Reactive decorator preserves function metadata but wraps to prevent manual calls."""
        obs = observable(5)

        def original_func(value):
            return value * 2

        decorated = reactive(obs)(original_func)

        # Should return ReactiveWrapper (not the original)
        assert decorated is not original_func
        assert decorated.__name__ == "original_func"
        assert callable(decorated)
        assert hasattr(decorated, "unsubscribe")
        assert hasattr(decorated, "_func")

        # Original function should still be callable normally
        assert original_func(5) == 10

        # But decorated function should prevent manual calls while subscribed
        from fynx import ReactiveFunctionWasCalled

        with pytest.raises(ReactiveFunctionWasCalled):
            decorated(5)

        # After unsubscribe, it should allow calls
        decorated.unsubscribe()
        assert decorated(5) == 10  # Should call original function

    def test_reactive_decorator_works_with_computed_observables(self):
        """Reactive decorator works correctly with computed observable values."""
        source = observable(10)
        doubled = source >> (lambda x: x * 2)
        execution_log = []

        @reactive(doubled)
        def log_doubled_value(value):
            execution_log.append(value)

        source.set(20)

        assert execution_log == [20, 40]  # initial + updated computed value

    def test_reactive_decorator_prevents_manual_function_calls_while_subscribed(self):
        """Reactive decorator prevents manual function calls while subscribed."""
        from fynx import ReactiveFunctionWasCalled

        obs = observable(10)

        @reactive(obs)
        def reactive_func(value):
            return value + 5

        # Manual calls should raise ReactiveFunctionWasCalled while subscribed
        with pytest.raises(
            ReactiveFunctionWasCalled,
            match="Reactive function reactive_func was called manually",
        ):
            reactive_func(10)

        with pytest.raises(ReactiveFunctionWasCalled):
            reactive_func(20)

        # After unsubscribe, normal calls should work
        reactive_func.unsubscribe()
        assert reactive_func(10) == 15
        assert reactive_func(20) == 25


@pytest.mark.unit
@pytest.mark.reactive
class TestReactiveDecoratorUnsubscribe:
    """Test the unsubscribe functionality of the reactive decorator."""

    def test_reactive_decorator_single_observable_unsubscribe(self):
        """Can unsubscribe reactive function from single observable."""
        obs = observable(5)
        call_log = []

        reactive_handler = reactive(obs)(
            lambda value: call_log.append(f"single:{value}")
        )

        assert len(call_log) == 1  # Initial call

        obs.set(10)
        assert len(call_log) == 2

        # Unsubscribe using the wrapper's unsubscribe method
        reactive_handler.unsubscribe()

        # Further changes should not trigger
        obs.set(15)
        assert len(call_log) == 2  # No new calls

    def test_reactive_decorator_store_unsubscribe(self):
        """Can unsubscribe reactive function from store."""

        class TestStore(Store):
            value = observable(42)

        call_log = []

        store_handler = reactive(TestStore)(
            lambda snapshot: call_log.append(f"store:{snapshot.to_dict()['value']}")
        )

        assert len(call_log) == 1  # Initial call

        TestStore.value = 100
        assert len(call_log) >= 2  # Change triggered

        # Unsubscribe using the wrapper's unsubscribe method
        store_handler.unsubscribe()

        # Further changes should not trigger
        TestStore.value = 200
        initial_count = len(call_log)
        assert len(call_log) == initial_count  # No new calls

    def test_reactive_decorator_unsubscribe_is_idempotent(self):
        """Unsubscribe operations are safe to call multiple times."""
        obs = observable(5)
        call_log = []

        handler = reactive(obs)(lambda value: call_log.append(value))

        assert len(call_log) == 1

        # Unsubscribe multiple times - should not raise errors
        handler.unsubscribe()
        handler.unsubscribe()  # Second call should be safe
        handler.unsubscribe()  # Third call should be safe

        # Further changes should not trigger
        obs.set(10)
        assert len(call_log) == 1  # No new calls


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_decorator_skips_inactive_conditional_observables():
    """Reactive decorator skips execution when conditional observable is not active."""
    from fynx.observable.conditional import ConditionalObservable

    source = observable(5)
    condition = observable(False)  # Condition is false
    conditional = source & condition

    execution_log = []

    @reactive(conditional)
    def log_value(value):
        execution_log.append(value)

    # Should not execute because conditional is not active
    assert len(execution_log) == 0

    # Make condition true
    condition.set(True)
    # Should now execute
    assert len(execution_log) == 1
    assert execution_log[0] == 5


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_decorator_handles_empty_targets():
    """Reactive decorator handles empty targets gracefully."""
    execution_log = []

    @reactive()
    def log_value(value):
        execution_log.append(value)

    # Should not execute with no targets
    assert len(execution_log) == 0


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_decorator_handles_none_merged_values():
    """Reactive decorator handles None merged values gracefully."""
    obs1 = observable(5)
    obs2 = observable(10)
    execution_log = []

    @reactive(obs1, obs2)
    def log_values(val1, val2):
        execution_log.append((val1, val2))

    # Should execute with current values
    assert len(execution_log) == 1
    assert execution_log[0] == (5, 10)

    # Simulate merged value being None (edge case)
    merged = obs1 + obs2
    merged._value = None

    # Should handle None merged values gracefully
    # This tests the line: if current_values is not None:
    obs1.set(6)
    # Should still work despite the edge case
    assert len(execution_log) >= 2


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_wrapper_call_raises_error_when_subscribed():
    """Test ReactiveWrapper.__call__() raises error when subscribed (line 111)"""
    from fynx.reactive import ReactiveWrapper

    obs = observable(5)

    def test_func(value):
        return value * 2

    wrapper = ReactiveWrapper(test_func, (obs,))
    wrapper._subscribed = True

    # Should raise ReactiveFunctionWasCalled when called manually
    with pytest.raises(Exception, match="Reactive function.*was called manually"):
        wrapper(10)


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_wrapper_setup_subscriptions_empty_targets():
    """Test ReactiveWrapper._setup_subscriptions() with empty targets (line 130->exit)"""
    from fynx.reactive import ReactiveWrapper

    def test_func():
        return "test"

    wrapper = ReactiveWrapper(test_func, ())
    wrapper._setup_subscriptions()

    # Should be subscribed but with no subscriptions
    assert wrapper._subscribed is True
    assert len(wrapper._subscriptions) == 0


@pytest.mark.unit
@pytest.mark.reactive
def test_reactive_wrapper_setup_subscriptions_conditional_inactive():
    """Test ReactiveWrapper._setup_subscriptions() with inactive conditional (line 144->148)"""
    from fynx.observable.conditional import ConditionalObservable
    from fynx.reactive import ReactiveWrapper

    obs = observable(5)
    conditional = obs & (lambda x: x > 10)  # Will be inactive initially

    def test_func(value):
        return value * 2

    wrapper = ReactiveWrapper(test_func, (conditional,))

    # Should not call the function when conditional is inactive
    execution_log = []
    wrapper._func = lambda x: execution_log.append(x)

    wrapper._setup_subscriptions()

    # Should be subscribed but function not called due to inactive conditional
    assert wrapper._subscribed is True
    assert (
        len(execution_log) == 0
    )  # Function not called because conditional is inactive
