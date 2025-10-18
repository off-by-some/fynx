"""Basic tests for FynX library."""

import pytest

from fynx import Observable, ReactiveContext, Store, computed, observable, reactive


def test_observable_basic():
    """Test basic Observable functionality."""
    obs = Observable("test", "initial")

    assert obs.value == "initial"
    assert str(obs) == "initial"

    obs.set("updated")
    assert obs.value == "updated"


def test_merged_observable():
    """Test MergedObservable functionality."""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")

    merged = obs1 | obs2

    assert merged.value == ("hello", "world")
    assert len(merged) == 2

    # Test context manager and unpacking
    with merged:
        val1, val2 = merged.value
        assert val1 == "hello"
        assert val2 == "world"

    # Test that merged updates when sources change
    obs1.set("hi")
    assert merged.value == ("hi", "world")

    obs2.set("there")
    assert merged.value == ("hi", "there")


def test_chained_merged_observable():
    """Test chaining the | operator for merging multiple observables."""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    obs3 = Observable("third", "!")

    # Test chaining: obs1 | obs2 | obs3
    chained = obs1 | obs2 | obs3

    assert chained.value == ("hello", "world", "!")
    assert len(chained) == 3

    # Test that chained merged updates when any source changes
    obs1.set("hi")
    assert chained.value == ("hi", "world", "!")

    obs3.set("?")
    assert chained.value == ("hi", "world", "?")

    # Test merging an existing merged observable with another observable
    obs4 = Observable("fourth", "extra")
    chained2 = chained | obs4

    assert chained2.value == ("hi", "world", "?", "extra")
    assert len(chained2) == 4

    # Verify the original chained is unaffected
    assert chained.value == ("hi", "world", "?")
    assert len(chained) == 3


def test_store_with_observable():
    """Test Store class with observable() API."""

    class TestStore(Store):
        count = observable(0)
        name = observable("test")

    # Test initial values
    assert TestStore.count.value == 0
    assert TestStore.name.value == "test"

    # Test setting values
    TestStore.count = 5
    TestStore.name = "updated"

    assert TestStore.count.value == 5
    assert TestStore.name.value == "updated"

    # Note: Context manager is only supported for merged observables, not individual ones

    # Test merged observables
    merged = TestStore.count | TestStore.name
    assert merged.value == (5, "updated")

    # Test serialization
    state = TestStore().to_dict()
    assert state == {"count": 5, "name": "updated"}


def test_standalone_observable_subscription():
    """Test subscribing and unsubscribing from standalone observables."""
    # Create standalone observables
    current_name = observable("Alice")

    # Track callback calls
    callback_calls = []

    def on_name_change(name):
        callback_calls.append(f"Name changed to: {name}")

    # Subscribe to changes
    current_name.subscribe(on_name_change)

    # Change should trigger callback
    current_name.set("Smith")
    assert callback_calls == ["Name changed to: Smith"]

    # Unsubscribe
    current_name.unsubscribe(on_name_change)

    # Change should NOT trigger callback anymore
    current_name.set("Bob")
    assert callback_calls == ["Name changed to: Smith"]


def test_merged_observable_subscription():
    """Test subscribing and unsubscribing from merged observables."""
    # Create observables
    current_name = observable("Alice")
    current_age = observable(30)

    # Track callback calls
    callback_calls = []

    def on_combined_change(name, age):
        callback_calls.append(f"Name: {name}, Age: {age}")

    # Create merged observable
    current_name_and_age = current_name | current_age

    # Subscribe to merged observable
    current_name_and_age.subscribe(on_combined_change)

    # Changes should trigger callback
    current_name.set("Charlie")
    assert callback_calls == ["Name: Charlie, Age: 30"]

    current_age.set(31)
    assert callback_calls == ["Name: Charlie, Age: 30", "Name: Charlie, Age: 31"]

    # Unsubscribe
    current_name_and_age.unsubscribe(on_combined_change)

    # Changes should NOT trigger callback anymore
    current_age.set(32)
    assert callback_calls == ["Name: Charlie, Age: 30", "Name: Charlie, Age: 31"]


def test_store_subscription():
    """Test subscribing and unsubscribing from store changes."""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    # Track callback calls
    callback_calls = []

    def on_store_change(store):
        callback_calls.append(
            f"Store: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    # Subscribe to store changes
    TestStore.subscribe(on_store_change)

    # Changes should trigger callback
    TestStore.height_cm = 170.0
    assert len(callback_calls) == 1
    assert "height=170.0" in callback_calls[0]

    TestStore.name = "Bob"
    assert len(callback_calls) == 2
    assert "name=Bob" in callback_calls[1]

    TestStore.age = 31
    assert len(callback_calls) == 3
    assert "age=31" in callback_calls[2]

    # Unsubscribe
    TestStore.unsubscribe(on_store_change)

    # Changes should NOT trigger callback anymore
    TestStore.height_cm = 180.0
    assert len(callback_calls) == 3  # No new calls


def test_reactive_store_decorator():
    """Test @reactive decorator with stores."""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    # Track callback calls
    callback_calls = []

    @reactive(TestStore)
    def on_store_change(store):
        callback_calls.append(
            f"Decorator: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    # Changes should trigger decorated function
    TestStore.height_cm = 170.2
    assert len(callback_calls) == 2  # Called immediately + on change
    assert "height=170.2" in callback_calls[1]

    # Unsubscribe
    TestStore.unsubscribe(on_store_change)

    # Changes should NOT trigger anymore
    TestStore.name = "Charlie"
    assert len(callback_calls) == 2  # No new calls


def test_reactive_multiple_observables_decorator():
    """Test @reactive decorator with multiple observables."""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    # Track callback calls
    callback_calls = []

    @reactive(TestStore.age, TestStore.name)
    def on_multi_change(age, name):
        callback_calls.append(f"Multi: name={name}, age={age}")

    # Changes should trigger decorated function
    TestStore.age = 31
    assert callback_calls == ["Multi: name=Alice, age=31"]

    TestStore.name = "Barbara"
    assert callback_calls == [
        "Multi: name=Alice, age=31",
        "Multi: name=Barbara, age=31",
    ]

    # Unsubscribe
    TestStore.unsubscribe(on_multi_change)

    # Changes should NOT trigger anymore
    TestStore.age = 32
    assert len(callback_calls) == 2  # No new calls


def test_context_manager_merged_observables():
    """Test context manager with merged observables."""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    # Track callback calls
    callback_calls = []

    def on_context_change(name, age):
        callback_calls.append(f"Context: name={name}, age={age}")

    # Use context manager
    with TestStore.name | TestStore.age as react:
        react(on_context_change)

        # Context manager executes callback immediately with current values
        assert callback_calls == ["Context: name=Alice, age=30"]

        # Changes should trigger callback
        TestStore.name = "Bob"
        assert callback_calls == [
            "Context: name=Alice, age=30",
            "Context: name=Bob, age=30",
        ]

        TestStore.age = 31
        assert callback_calls == [
            "Context: name=Alice, age=30",
            "Context: name=Bob, age=30",
            "Context: name=Bob, age=31",
        ]

    # Note: Context manager cleanup is not currently implemented
    # so callbacks continue after exit
    TestStore.name = "Charlie"
    # Just verify the context manager worked during its execution


def test_store_snapshot_repr():
    """Test StoreSnapshot string representation."""
    from fynx.store import StoreSnapshot

    class TestStore(Store):
        pass

    # Create snapshot manually for testing
    snapshot = StoreSnapshot(TestStore, ["name", "age"])
    snapshot._snapshot_values = {"name": "Alice", "age": 30}

    # Test repr
    repr_str = repr(snapshot)
    assert repr_str == "StoreSnapshot(name='Alice', age=30)"

    # Test empty snapshot
    empty_snapshot = StoreSnapshot(TestStore, [])
    assert repr(empty_snapshot) == "StoreSnapshot()"


def test_computed_observables():
    """Test computed/derived observables via functorial map."""
    # Test single observable computed
    base = observable(5)
    doubled = computed(lambda x: x * 2, base)

    assert doubled.value == 10  # Initial computation

    # Change base, computed should update
    base.set(7)
    assert doubled.value == 14

    # Test merged observable computed
    num1 = observable(3)
    num2 = observable(4)
    combined = num1 | num2
    summed = computed(lambda a, b: a + b, combined)

    assert summed.value == 7  # 3 + 4

    # Change one input, computed should update
    num1.set(10)
    assert summed.value == 14  # 10 + 4

    num2.set(6)
    assert summed.value == 16  # 10 + 6

    # Test chaining computed observables
    final = computed(lambda s: s * 2, summed)
    assert final.value == 32  # (10 + 6) * 2

    num1.set(1)
    assert final.value == 14  # (1 + 6) * 2


def test_watch_decorator():
    """Test the watch decorator for conditional reactions."""
    from fynx import watch

    # Create observables
    status = observable("loading")
    count = observable(0)

    # Track callback calls
    callback_calls = []

    @watch(lambda: status.value == "ready", lambda: count.value > 3)
    def conditional_callback():
        callback_calls.append(f"Called when status={status.value}, count={count.value}")

    # Initially conditions not met, so no callback
    assert callback_calls == []

    # Change count but status still not ready
    count.set(5)
    assert callback_calls == []  # count > 3 but status != "ready"

    # Change status to ready (count is still 5, so conditions are now met)
    status.set("ready")
    assert len(callback_calls) == 1  # Conditions met when status became ready
    assert "status=ready, count=5" in callback_calls[0]

    # Change count to 2 (conditions no longer met)
    count.set(2)
    assert len(callback_calls) == 1  # No additional call

    # Change count back to 5 (conditions met again)
    count.set(5)
    assert len(callback_calls) == 2
    assert "status=ready, count=5" in callback_calls[1]

    # Change status - conditions no longer met
    status.set("busy")
    assert len(callback_calls) == 2  # No new call

    # Change back to ready - conditions met again (count is still 5)
    status.set("ready")
    assert len(callback_calls) == 3
    assert "status=ready, count=5" in callback_calls[2]


def test_rshift_operator():
    """Test the >> operator for chaining computed transformations."""
    # Test single observable
    base = observable(5)
    doubled = base >> (lambda x: x * 2)
    tripled = doubled >> (lambda x: x * 3)

    assert doubled.value == 10  # 5 * 2
    assert tripled.value == 30  # (5 * 2) * 3

    # Change base, should propagate through chain
    base.set(10)
    assert doubled.value == 20  # 10 * 2
    assert tripled.value == 60  # (10 * 2) * 3

    # Test merged observable
    num1 = observable(3)
    num2 = observable(4)
    combined = num1 | num2
    summed = combined >> (lambda a, b: a + b)
    formatted = summed >> (lambda s: f"Sum: {s}")

    assert summed.value == 7  # 3 + 4
    assert formatted.value == "Sum: 7"

    # Change inputs, should propagate
    num1.set(10)
    assert summed.value == 14  # 10 + 4
    assert formatted.value == "Sum: 14"


def test_circular_dependency_detection():
    """Test that circular dependencies in computed observables raise RuntimeError."""
    # Create an observable that tries to update itself during its own update
    computed_obs = observable(0)

    def update_func():
        # This creates a circular dependency by updating the observable
        # that's currently being updated
        computed_obs.set(computed_obs.value + 1)

    # Create a context that updates computed_obs when computed_obs changes
    context = ReactiveContext(update_func, lambda: None, computed_obs)
    computed_obs.add_observer(context.run)

    # This should raise RuntimeError due to circular dependency
    with pytest.raises(RuntimeError, match="Circular dependency detected"):
        computed_obs.set(1)  # This triggers the circular update


def test_watch_single_condition():
    """Test the watch decorator with a single condition."""
    from fynx import watch

    # Create observable
    is_ready = observable(False)

    # Track callback calls
    callback_calls = []

    @watch(lambda: is_ready.value)
    def single_condition_callback():
        callback_calls.append(f"Ready: {is_ready.value}")

    # Initially condition not met
    assert callback_calls == []

    # Set to True - condition met
    is_ready.set(True)
    assert len(callback_calls) == 1
    assert "Ready: True" in callback_calls[0]

    # Set back to False - condition not met
    is_ready.set(False)
    assert len(callback_calls) == 1  # No additional call

    # Set to True again - condition met again
    is_ready.set(True)
    assert len(callback_calls) == 2


def test_watch_observable_discovery():
    """Test that watch correctly discovers which observables are accessed."""
    from fynx import watch

    # Create multiple observables
    name = observable("Alice")
    age = observable(25)
    active = observable(True)
    unused = observable("not accessed")  # This should not be watched

    # Track callback calls
    callback_calls = []

    @watch(lambda: name.value == "Bob", lambda: age.value > 20, lambda: active.value)
    def multi_condition_callback():
        callback_calls.append(
            f"Name: {name.value}, Age: {age.value}, Active: {active.value}"
        )

    # Initially conditions not met (name != "Bob")
    assert callback_calls == []

    # Change name to Bob - now all conditions met
    name.set("Bob")
    assert len(callback_calls) == 1
    assert "Name: Bob, Age: 25, Active: True" in callback_calls[0]

    # Change unused observable - should not trigger callback since it's not watched
    unused.set("changed")
    assert len(callback_calls) == 1  # Still only 1 call

    # Change age - conditions become unmet
    age.set(15)
    assert len(callback_calls) == 1  # Still only 1 call

    # Deactivate first, then change age back - conditions still unmet
    active.set(False)
    age.set(25)
    assert len(callback_calls) == 1  # Still only 1 call (active is now False)

    # Reactivate - all conditions met again
    active.set(True)
    assert len(callback_calls) == 2


def test_watch_initial_execution():
    """Test that watch executes immediately if initial conditions are met."""
    from fynx import watch

    # Create observable with initial value that meets condition
    counter = observable(10)

    # Track callback calls
    callback_calls = []

    @watch(lambda: counter.value > 5)
    def immediate_callback():
        callback_calls.append(f"Counter: {counter.value}")

    # Should execute immediately since initial value (10) > 5
    assert len(callback_calls) == 1
    assert "Counter: 10" in callback_calls[0]

    # Change value but still > 5 - no additional trigger (conditions still met)
    counter.set(8)
    assert len(callback_calls) == 1  # No additional trigger

    # Change to value that doesn't meet condition
    counter.set(3)
    assert len(callback_calls) == 1  # No additional call

    # Change back to value that meets condition - should trigger again
    counter.set(7)
    assert len(callback_calls) == 2
    assert "Counter: 7" in callback_calls[1]


def test_watch_error_handling():
    """Test that watch handles errors in condition evaluation gracefully."""
    import sys

    from fynx import watch

    # Create observables
    value = observable(5)
    error_obs = observable(None)

    # Track callback calls and errors
    callback_calls = []
    captured_errors = []

    # Mock stdout to capture print statements
    import io
    from contextlib import redirect_stdout

    stdout_capture = io.StringIO()

    with redirect_stdout(stdout_capture):

        @watch(lambda: value.value > 3, lambda: error_obs.value.nonexistent_attribute)
        def error_callback():
            callback_calls.append(f"Value: {value.value}")

    # Check that warning was printed about condition evaluation failure
    output = stdout_capture.getvalue()
    assert "condition evaluation failed during discovery" in output

    # Should not execute initially since one condition throws an exception
    assert len(callback_calls) == 0  # No initial execution due to error condition

    # Change value - should still not trigger since error condition fails
    value.set(7)
    assert len(callback_calls) == 0  # Still no calls due to error condition


def test_placeholder():
    """Placeholder test to ensure pytest is working."""
    assert True
