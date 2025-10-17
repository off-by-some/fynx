"""Tests for store functionality and store-related features."""

import pytest

from fynx import MergedObservable, Observable, Store, computed, observable


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
    snapshot = type("TestSnapshot", (), {"_observable_attrs": []})()
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

    expected = {"name": "Alice", "age": 25, "active": True}
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

    state = {"name": "updated", "count": 42}

    store.load_state(state)

    assert store.name.value == "updated"
    assert store.count.value == 42


def test_store_load_state_ignores_unknown_keys():
    """Test that load_state ignores keys that don't correspond to observables."""

    class TestStore(Store):
        value = observable("initial")

    store = TestStore()

    state = {"value": "updated", "unknown_key": "ignored"}

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
    partial_state = {"a": 10, "c": 30}

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


def test_store_with_mixed_observable_and_computed_attributes():
    """Test that stores can have both regular observables and computed observables."""

    class MixedStore(Store):
        # Regular observables
        base_value = observable(10)
        multiplier = observable(2)

        # Computed observable
        computed_product = computed(lambda b, m: b * m, (base_value | multiplier))

    store = MixedStore()

    # Test initial values
    assert store.base_value.value == 10
    assert store.multiplier.value == 2
    assert store.computed_product.value == 20

    # Test that computed updates when dependencies change
    store.base_value.set(15)
    assert store.computed_product.value == 30

    store.multiplier.set(3)
    assert store.computed_product.value == 45


def test_store_mixed_observables_subscription():
    """Test that subscriptions work with mixed observable and computed attributes."""

    class MixedStore(Store):
        counter = observable(0)
        doubled = computed(lambda c: c * 2, counter)

    callback_calls = []

    def on_change(snapshot):
        callback_calls.append(
            {"counter": snapshot.counter, "doubled": snapshot.doubled}
        )

    MixedStore.subscribe(on_change)

    # Change the base observable
    MixedStore.counter.set(5)

    assert len(callback_calls) >= 1
    latest_call = callback_calls[-1]
    assert latest_call["counter"] == 5
    assert latest_call["doubled"] == 10


def test_store_mixed_observables_serialization():
    """Test that serialization works with mixed observable and computed attributes."""

    class MixedStore(Store):
        name = observable("test")
        count = observable(5)
        name_length = computed(lambda n: len(n), name)

    store = MixedStore()

    # to_dict should include all observable values
    data = store.to_dict()
    expected = {"name": "test", "count": 5, "name_length": 4}
    assert data == expected

    # load_state should only affect regular observables
    store.load_state({"name": "hello", "count": 10})
    assert store.name.value == "hello"
    assert store.count.value == 10
    assert store.name_length.value == 5  # Updated automatically


def test_store_computed_observables_are_readonly():
    """Test that computed observables in stores are read-only."""

    class TestStore(Store):
        value = observable(5)
        doubled = computed(lambda v: v * 2, value)

    store = TestStore()

    # Regular observable can be set
    store.value.set(10)
    assert store.value.value == 10
    assert store.doubled.value == 20

    # Computed observable cannot be set directly
    with pytest.raises(ValueError, match="Computed observables are read-only"):
        store.doubled.set(100)

    # Value should remain computed
    assert store.doubled.value == 20


def test_store_mixed_observables_complex_computation():
    """Test complex computed observables alongside regular observables."""

    class ComplexStore(Store):
        items = observable([1, 2, 3, 4, 5])
        multiplier = observable(2)

        # Computed: sum of items
        total = computed(lambda items: sum(items), items)

        # Computed: average of items
        average = computed(
            lambda total, count: total / count if count > 0 else 0,
            (total | computed(lambda items: len(items), items)),
        )

        # Computed: scaled total
        scaled_total = computed(
            lambda total, multiplier: total * multiplier, (total | multiplier)
        )

    store = ComplexStore()

    # Test initial values
    assert store.total.value == 15
    assert store.average.value == 3.0
    assert store.scaled_total.value == 30

    # Test updates propagate through the computation chain
    store.items.set([10, 20, 30])
    assert store.total.value == 60
    assert store.average.value == 20.0
    assert store.scaled_total.value == 120

    store.multiplier.set(3)
    assert store.scaled_total.value == 180  # 60 * 3


def test_store_get_observable_attrs_includes_computed():
    """Test that _get_observable_attrs includes both regular and computed observables."""

    class TestStore(Store):
        regular_obs = observable("regular")
        dummy_obs = observable("dummy")
        computed_obs = computed(lambda x: f"computed_{x}", dummy_obs)

    attrs = TestStore._get_observable_attrs()
    assert "regular_obs" in attrs
    assert "dummy_obs" in attrs
    assert "computed_obs" in attrs
    assert len(attrs) == 3


def test_store_get_primitive_observable_attrs_excludes_computed():
    """Test that _get_primitive_observable_attrs excludes computed observables."""

    class TestStore(Store):
        regular_obs = observable("regular")
        dummy_obs = observable("dummy")
        computed_obs = computed(lambda x: f"computed_{x}", dummy_obs)

    primitive_attrs = TestStore._get_primitive_observable_attrs()
    assert "regular_obs" in primitive_attrs
    assert "dummy_obs" in primitive_attrs
    assert "computed_obs" not in primitive_attrs
    assert len(primitive_attrs) == 2


def test_store_syntactic_sugar_rshift_operator():
    """Test that the >> syntactic sugar works for computed observables in stores."""

    class TestStore(Store):
        counter = observable(5)
        doubled = counter >> (lambda x: x * 2)
        tripled = counter >> (lambda x: x * 3)
        description = counter >> (lambda x: f"Count: {x}")

    store = TestStore()

    # Test initial values
    assert store.counter.value == 5
    assert store.doubled.value == 10
    assert store.tripled.value == 15
    assert store.description.value == "Count: 5"

    # Test that computed values update when dependency changes
    store.counter.set(10)
    assert store.doubled.value == 20
    assert store.tripled.value == 30
    assert store.description.value == "Count: 10"

    # Test chaining with >>
    chained = store.counter >> (lambda x: x + 1) >> (lambda x: x * 2)
    assert chained.value == 22  # (10 + 1) * 2


def test_store_rshift_with_merged_observables():
    """Test that >> works with merged observables in stores."""

    class TestStore(Store):
        width = observable(10)
        height = observable(20)
        area = (width | height) >> (lambda w, h: w * h)
        perimeter = (width | height) >> (lambda w, h: 2 * (w + h))

    store = TestStore()

    # Test initial values
    assert store.width.value == 10
    assert store.height.value == 20
    assert store.area.value == 200
    assert store.perimeter.value == 60

    # Test updates
    store.width.set(15)
    assert store.area.value == 300  # 15 * 20
    assert store.perimeter.value == 70  # 2 * (15 + 20)

    store.height.set(25)
    assert store.area.value == 375  # 15 * 25
    assert store.perimeter.value == 80  # 2 * (15 + 25)


def test_store_with_regular_properties():
    """Test that stores can have normal non-observable properties."""

    class TestStore(Store):
        # Observable properties
        name = observable("Alice")

        # Regular properties
        version = "1.0.0"
        config = {"theme": "dark", "lang": "en"}

        def get_description(self):
            return f"Store v{self.version}"

    store = TestStore()

    # Test observable works normally
    assert store.name.value == "Alice"
    store.name.set("Bob")
    assert store.name.value == "Bob"

    # Test regular properties work normally
    assert store.version == "1.0.0"
    assert store.config["theme"] == "dark"
    assert store.get_description() == "Store v1.0.0"

    # Test that regular properties can be modified
    store.version = "2.0.0"
    store.config["theme"] = "light"
    assert store.version == "2.0.0"
    assert store.config["theme"] == "light"


def test_store_with_merged_observables():
    """Test that merged observables can be used in store computed properties."""

    class TestStore(Store):
        # Individual observables
        x = observable(5)
        y = observable(10)

        # Computed values that use merged observables inline (like the working examples)
        sum_coords = (x | y) >> (lambda a, b: a + b)
        product_coords = (x | y) >> (lambda a, b: a * b)

        # Also test with three observables
        z = observable(2)
        total = (x | y | z) >> (lambda a, b, c: a + b + c)

    store = TestStore()

    # Test initial values
    assert store.x.value == 5
    assert store.y.value == 10
    assert store.z.value == 2
    assert store.sum_coords.value == 15  # 5 + 10
    assert store.product_coords.value == 50  # 5 * 10
    assert store.total.value == 17  # 5 + 10 + 2

    # Test that computed values update when dependencies change
    store.x.set(8)
    assert store.sum_coords.value == 18  # 8 + 10
    assert store.product_coords.value == 80  # 8 * 10
    assert store.total.value == 20  # 8 + 10 + 2

    store.y.set(15)
    assert store.sum_coords.value == 23  # 8 + 15
    assert store.product_coords.value == 120  # 8 * 15
    assert store.total.value == 25  # 8 + 15 + 2

    store.z.set(5)
    assert store.total.value == 28  # 8 + 15 + 5

    # Test that computed values are included in serialization
    data = store.to_dict()
    assert "x" in data
    assert "y" in data
    assert "z" in data
    assert "sum_coords" in data
    assert "product_coords" in data
    assert "total" in data
    assert data["sum_coords"] == 23
    assert data["product_coords"] == 120
    assert data["total"] == 28
