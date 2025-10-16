"""Tests for reactive decorators and reactive functionality."""

from fynx import Store, observable, reactive


def test_reactive_decorator_with_store():
    """Test @reactive decorator with a store class."""

    class TestStore(Store):
        value = observable(0)

    store = TestStore()
    callback_count = 0

    @reactive(TestStore)
    def on_store_change(snapshot):
        nonlocal callback_count
        callback_count += 1

    store.value.set(1)
    assert callback_count == 1

    store.value.set(2)
    assert callback_count == 2


def test_reactive_decorator_with_single_observable():
    """Test @reactive decorator with a single observable."""
    obs = observable(10)
    callback_count = 0

    @reactive(obs)
    def on_change(value):
        nonlocal callback_count
        callback_count += 1

    obs.set(20)
    assert callback_count == 1


def test_reactive_decorator_with_multiple_observables():
    """Test @reactive decorator with multiple observables."""
    obs1 = observable("a")
    obs2 = observable("b")
    callback_count = 0

    @reactive(obs1, obs2)
    def on_multiple_changes(*values):
        nonlocal callback_count
        callback_count += 1

    obs1.set("changed")
    assert callback_count == 1

    obs2.set("modified")
    assert callback_count == 2


def test_reactive_decorator_returns_original_function():
    """Test that @reactive decorator returns the original function."""

    def test_function():
        return "test"

    decorated = reactive(observable(1))(test_function)

    assert decorated is test_function
    assert callable(decorated)


def test_reactive_decorator_with_store_passes_snapshot():
    """Test that store reactive decorator passes snapshot to callback."""

    class TestStore(Store):
        counter = observable(0)

    store = TestStore()
    received_snapshot = None

    @reactive(TestStore)
    def capture_snapshot(snapshot):
        nonlocal received_snapshot
        received_snapshot = snapshot

    store.counter.set(5)

    assert received_snapshot is not None
    assert received_snapshot.counter == 5


def test_reactive_decorator_with_store_multiple_attributes():
    """Test reactive decorator with store having multiple observable attributes."""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    store = TestStore()
    callback_count = 0

    @reactive(TestStore)
    def on_any_change(snapshot):
        nonlocal callback_count
        callback_count += 1

    store.name.set("Bob")
    assert callback_count == 1

    store.age.set(30)
    assert callback_count == 2


def test_reactive_decorator_no_execution_on_same_value():
    """Test that reactive decorator doesn't execute when value doesn't change."""
    obs = observable("value")
    callback_count = 0

    @reactive(obs)
    def callback(value):
        nonlocal callback_count
        callback_count += 1

    # Setting same value should not trigger
    obs.set("value")
    assert callback_count == 0

    # Setting different value should trigger
    obs.set("new_value")
    assert callback_count == 1


def test_reactive_decorator_mixed_observables_and_store():
    """Test reactive decorator with mix of individual observables and store."""
    obs = observable("individual")

    class TestStore(Store):
        store_value = observable("store")

    store = TestStore()
    callback_count = 0

    @reactive(obs)
    def obs_callback(value):
        nonlocal callback_count
        callback_count += 1

    @reactive(TestStore)
    def store_callback(snapshot):
        nonlocal callback_count
        callback_count += 10

    obs.set("changed")
    assert callback_count == 1

    store.store_value.set("modified")
    assert callback_count == 11


def test_reactive_decorator_empty_args():
    """Test reactive decorator with no arguments."""
    obs = observable("test")

    # Should not raise an error, but won't do anything reactive
    @reactive()
    def callback():
        pass

    # Changing observable should not trigger callback since no targets specified
    obs.set("changed")
    # No assertion needed, just ensuring no exception is raised


def test_reactive_handler_creation():
    """Test that reactive() creates a ReactiveHandler instance."""
    from fynx import reactive

    handler = reactive(observable(1))
    assert hasattr(handler, "__call__")


def test_reactive_handler_as_decorator():
    """Test using ReactiveHandler directly as decorator."""
    obs = observable(0)
    callback_count = 0

    def callback(value):
        nonlocal callback_count
        callback_count += 1

    # Create handler and use it as decorator
    handler = reactive(obs)
    decorated_callback = handler(callback)

    assert decorated_callback is callback

    obs.set(1)
    assert callback_count == 1
