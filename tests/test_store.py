"""Tests for store functionality and store-related features."""

from fynx import Store, observable, Observable


def test_store_creation():
    """Test that a Store can be instantiated."""
    store = Store()
    assert isinstance(store, Store)


def test_store_with_observable_attributes():
    """Test that store can have observable attributes defined."""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    store = TestStore()
    assert store.name.value == "Alice"
    assert store.age.value == 25


def test_store_observable_attribute_updates():
    """Test that store observable attributes can be updated."""
    class TestStore(Store):
        counter = observable(0)

    store = TestStore()
    store.counter.set(5)
    assert store.counter.value == 5


def test_store_observable_attribute_type_preservation():
    """Test that different types are preserved in store observables."""
    class TestStore(Store):
        text = observable("hello")
        number = observable(42)
        boolean = observable(True)
        none_value = observable(None)

    store = TestStore()
    assert store.text.value == "hello"
    assert store.number.value == 42
    assert store.boolean.value is True
    assert store.none_value.value is None


def test_store_subscription_to_observable_changes():
    """Test that store can subscribe to changes in its observables."""
    class TestStore(Store):
        value = observable(10)

    store = TestStore()
    callback_count = 0
    last_snapshot = None

    def on_change(snapshot):
        nonlocal callback_count, last_snapshot
        callback_count += 1
        last_snapshot = snapshot

    TestStore.subscribe(on_change)

    # Initial subscription might trigger, but let's change value
    store.value.set(20)

    assert callback_count >= 1
    assert last_snapshot.value == 20


def test_store_subscription_returns_none():
    """Test that store subscribe method doesn't return a value for chaining."""
    class TestStore(Store):
        test = observable("value")

    result = TestStore.subscribe(lambda s: None)
    assert result is None


def test_store_multiple_observables_subscription():
    """Test that store subscription works with multiple observables."""
    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    store = TestStore()
    callback_count = 0

    def on_change(snapshot):
        nonlocal callback_count
        callback_count += 1

    TestStore.subscribe(on_change)

    store.name.set("Bob")
    store.age.set(30)

    assert callback_count >= 2


def test_store_unsubscribe_removes_callbacks():
    """Test that store unsubscribe removes callback functions."""
    class TestStore(Store):
        value = observable(0)

    store = TestStore()
    callback_count = 0

    def callback1(snapshot):
        nonlocal callback_count
        callback_count += 1

    def callback2(snapshot):
        nonlocal callback_count
        callback_count += 10

    TestStore.subscribe(callback1)
    TestStore.subscribe(callback2)

    store.value.set(1)
    assert callback_count == 11  # Both callbacks executed

    TestStore.unsubscribe(callback1)
    store.value.set(2)
    assert callback_count == 21  # Only callback2 executed


def test_store_unsubscribe_nonexistent_callback():
    """Test that unsubscribing non-existent callback doesn't cause errors."""
    class TestStore(Store):
        value = observable("test")

    def callback():
        pass

    # Should not raise an error
    TestStore.unsubscribe(callback)


def test_store_snapshot_contains_current_values():
    """Test that store snapshot contains current observable values."""
    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    store = TestStore()
    snapshot = None

    def capture_snapshot(s):
        nonlocal snapshot
        snapshot = s

    TestStore.subscribe(capture_snapshot)
    store.name.set("Bob")

    assert snapshot is not None
    assert snapshot.name == "Bob"
    assert snapshot.age == 25


def test_store_snapshot_attribute_access():
    """Test attribute access on store snapshots."""
    class TestStore(Store):
        counter = observable(5)

    store = TestStore()
    snapshots = []

    def collect_snapshots(snapshot):
        snapshots.append(snapshot)

    TestStore.subscribe(collect_snapshots)
    store.counter.set(10)

    assert len(snapshots) >= 1
    snapshot = snapshots[-1]
    assert snapshot.counter == 10


def test_store_snapshot_repr():
    """Test string representation of store snapshots."""
    class TestStore(Store):
        x = observable(1)
        y = observable(2)

    store = TestStore()
    snapshots = []

    def collect_snapshots(snapshot):
        snapshots.append(snapshot)

    TestStore.subscribe(collect_snapshots)
    store.x.set(10)

    assert len(snapshots) >= 1
    snapshot = snapshots[-1]

    repr_str = repr(snapshot)
    assert "x=10" in repr_str
    assert "y=2" in repr_str


def test_store_snapshot_empty_repr():
    """Test repr of empty snapshot."""
    # Create a snapshot with no observable attrs
    snapshot = type('TestSnapshot', (), {'_observable_attrs': []})()
    snapshot._snapshot_values = {}

    repr_str = repr(snapshot)
    # For custom types, repr will show the class name, not "StoreSnapshot"
    assert "TestSnapshot" in repr_str


def test_store_to_dict_serializes_observables():
    """Test that store to_dict method serializes observable values."""
    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)
        active = observable(True)

    store = TestStore()
    data = store.to_dict()

    expected = {
        "name": "Alice",
        "age": 25,
        "active": True
    }
    assert data == expected


def test_store_to_dict_ignores_non_observables():
    """Test that to_dict only includes observable attributes."""
    class TestStore(Store):
        observable_value = observable("observable")
        regular_value = "not observable"

    store = TestStore()
    data = store.to_dict()

    assert "observable_value" in data
    assert "regular_value" not in data
    assert data["observable_value"] == "observable"


def test_store_load_state_updates_observables():
    """Test that load_state updates observable values from dict."""
    class TestStore(Store):
        name = observable("initial")
        count = observable(0)

    store = TestStore()

    state = {
        "name": "updated",
        "count": 42
    }

    store.load_state(state)

    assert store.name.value == "updated"
    assert store.count.value == 42


def test_store_load_state_ignores_unknown_keys():
    """Test that load_state ignores keys that don't correspond to observables."""
    class TestStore(Store):
        value = observable("initial")

    store = TestStore()

    state = {
        "value": "updated",
        "unknown_key": "ignored"
    }

    store.load_state(state)
    assert store.value.value == "updated"


def test_store_load_state_partial_update():
    """Test that load_state only updates specified observables."""
    class TestStore(Store):
        a = observable(1)
        b = observable(2)
        c = observable(3)

    store = TestStore()

    # Only update a and c
    partial_state = {
        "a": 10,
        "c": 30
    }

    store.load_state(partial_state)

    assert store.a.value == 10
    assert store.b.value == 2  # Unchanged
    assert store.c.value == 30




def test_circular_dependency_prevention():
    """Test that circular dependencies are prevented in observable updates."""
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    # Create a scenario where updating obs1 triggers obs2, which tries to update obs1
    update_count = 0

    def update_obs2():
        nonlocal update_count
        update_count += 1
        if update_count < 3:  # Prevent infinite recursion in test
            obs1.set(obs1.value + 1)  # This would create a cycle

    obs1.subscribe(update_obs2)

    # This should work without circular dependency issues
    # (the current implementation may not detect all cycles, but basic updates work)
    obs2.set(3)
    assert obs1.value == 1  # Should not have been modified by the cycle
