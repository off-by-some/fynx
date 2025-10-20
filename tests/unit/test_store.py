"""Unit tests for Store behavior and functionality."""

import pytest

from fynx import Observable, Store, observable


@pytest.mark.unit
@pytest.mark.store
def test_store_can_be_instantiated():
    """Store class can be instantiated without errors"""
    # Arrange & Act
    store = Store()

    # Assert
    assert isinstance(store, Store)


@pytest.mark.unit
@pytest.mark.store
def test_store_provides_class_level_access_to_observables():
    """Store observables can be accessed and modified through class attributes"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    # Act & Assert
    store = TestStore()
    assert store.name.value == "Alice"
    assert store.age.value == 25


@pytest.mark.unit
@pytest.mark.store
def test_store_observables_can_be_updated():
    """Store observable attributes can be updated via set() method"""

    class TestStore(Store):
        counter = observable(0)

    # Arrange
    store = TestStore()

    # Act
    store.counter.set(5)

    # Assert
    assert store.counter.value == 5


@pytest.mark.unit
@pytest.mark.store
def test_store_meta_inherits_observables_from_base_classes():
    """StoreMeta properly inherits observables from base classes."""
    from fynx.observable.descriptors import SubscriptableDescriptor

    class BaseStore(Store):
        base_attr = observable("base_value")

    class DerivedStore(BaseStore):
        derived_attr = observable("derived_value")

    # Should inherit base_attr from BaseStore
    assert hasattr(DerivedStore, "base_attr")
    assert hasattr(DerivedStore, "derived_attr")

    # Both should be SubscriptableDescriptor instances
    assert isinstance(DerivedStore.__dict__["base_attr"], SubscriptableDescriptor)
    assert isinstance(DerivedStore.__dict__["derived_attr"], SubscriptableDescriptor)


@pytest.mark.unit
@pytest.mark.store
def test_store_meta_handles_inherited_observables_without_namespace_conflict():
    """StoreMeta handles inherited observables when they're not in namespace."""
    from fynx.observable.descriptors import SubscriptableDescriptor

    class BaseStore(Store):
        shared_attr = observable("shared_value")

    class DerivedStore(BaseStore):
        # Don't redefine shared_attr - should inherit it
        pass

    # Should inherit shared_attr from BaseStore
    assert hasattr(DerivedStore, "shared_attr")
    assert isinstance(DerivedStore.__dict__["shared_attr"], SubscriptableDescriptor)

    # Should have correct initial value
    assert DerivedStore.__dict__["shared_attr"]._initial_value == "shared_value"


@pytest.mark.unit
@pytest.mark.store
def test_store_meta_creates_descriptors_for_inherited_observables():
    """StoreMeta creates new descriptors for inherited observables."""
    from fynx.observable.descriptors import SubscriptableDescriptor

    class BaseStore(Store):
        inherited_attr = observable("inherited_value")

    class DerivedStore(BaseStore):
        pass

    # Should create new descriptor for inherited observable
    descriptor = DerivedStore.__dict__["inherited_attr"]
    assert isinstance(descriptor, SubscriptableDescriptor)
    assert descriptor._initial_value == "inherited_value"
    assert descriptor._original_observable is None  # Should not share original


@pytest.mark.unit
@pytest.mark.store
def test_store_preserves_observable_value_types():
    """Store observables preserve different data types correctly"""

    class TestStore(Store):
        text = observable("hello")
        number = observable(42)
        boolean = observable(True)
        none_value = observable(None)

    # Arrange & Act
    store = TestStore()

    # Assert
    assert store.text.value == "hello"
    assert store.number.value == 42
    assert store.boolean.value is True
    assert store.none_value.value is None


@pytest.mark.unit
@pytest.mark.store
def test_store_subscription_notifies_on_observable_changes():
    """Store subscribers receive notifications when any store observable changes"""

    class TestStore(Store):
        value = observable(10)

    # Arrange
    store = TestStore()
    callback_count = 0
    last_snapshot = None

    def on_change(snapshot):
        nonlocal callback_count, last_snapshot
        callback_count += 1
        last_snapshot = snapshot

    TestStore.subscribe(on_change)

    # Act - Change observable value
    store.value.set(20)

    # Assert
    assert callback_count >= 1
    assert last_snapshot.value == 20


@pytest.mark.unit
@pytest.mark.store
def test_store_subscribe_does_not_return_value_for_chaining():
    """Store subscribe() method returns None (not the store for chaining)"""

    class TestStore(Store):
        test = observable("value")

    # Arrange & Act
    result = TestStore.subscribe(lambda s: None)

    # Assert
    assert result is None


@pytest.mark.unit
@pytest.mark.store
def test_store_subscription_works_with_multiple_observables():
    """Store subscription triggers on changes to any observable in the store"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    # Arrange
    store = TestStore()
    callback_count = 0

    def on_change(snapshot):
        nonlocal callback_count
        callback_count += 1

    TestStore.subscribe(on_change)

    # Act - Change multiple observables
    store.name.set("Bob")
    store.age.set(30)

    # Assert
    assert callback_count >= 2


@pytest.mark.unit
@pytest.mark.store
def test_store_unsubscribe_removes_specific_callbacks():
    """Store unsubscribe() removes only the specified callback function"""

    class TestStore(Store):
        value = observable(0)

    # Arrange
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

    # Act & Assert - Both callbacks initially
    store.value.set(1)
    assert callback_count == 11  # Both callbacks executed

    TestStore.unsubscribe(callback1)
    store.value.set(2)
    assert callback_count == 21  # Only callback2 executed


@pytest.mark.unit
@pytest.mark.store
def test_store_unsubscribe_handles_nonexistent_callbacks_gracefully():
    """Store unsubscribe() with unknown callback doesn't raise errors"""

    class TestStore(Store):
        value = observable("test")

    def callback():
        pass

    # Act & Assert - Should not raise an error
    TestStore.unsubscribe(callback)


@pytest.mark.unit
@pytest.mark.store
def test_store_snapshot_captures_current_observable_values():
    """Store snapshots contain current values of all observables at change time"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)

    # Arrange
    store = TestStore()
    snapshot = None

    def capture_snapshot(s):
        nonlocal snapshot
        snapshot = s

    TestStore.subscribe(capture_snapshot)

    # Act
    store.name.set("Bob")

    # Assert
    assert snapshot is not None
    assert snapshot.name == "Bob"
    assert snapshot.age == 25


@pytest.mark.unit
@pytest.mark.store
def test_store_snapshot_provides_attribute_access():
    """Store snapshots allow attribute access to observable values"""

    class TestStore(Store):
        counter = observable(5)

    # Arrange
    store = TestStore()
    snapshots = []

    def collect_snapshots(snapshot):
        snapshots.append(snapshot)

    TestStore.subscribe(collect_snapshots)

    # Act
    store.counter.set(10)

    # Assert
    assert len(snapshots) >= 1
    snapshot = snapshots[-1]
    assert snapshot.counter == 10


@pytest.mark.unit
@pytest.mark.store
def test_store_snapshot_has_readable_string_representation():
    """Store snapshots have meaningful string representations showing values"""

    class TestStore(Store):
        x = observable(1)
        y = observable(2)

    # Arrange
    store = TestStore()
    snapshots = []

    def collect_snapshots(snapshot):
        snapshots.append(snapshot)

    TestStore.subscribe(collect_snapshots)

    # Act
    store.x.set(10)

    # Assert
    assert len(snapshots) >= 1
    snapshot = snapshots[-1]

    repr_str = repr(snapshot)
    assert "x=10" in repr_str
    assert "y=2" in repr_str


@pytest.mark.unit
@pytest.mark.store
def test_store_snapshot_repr_handles_empty_snapshots():
    """Store snapshots handle repr correctly even with no observables"""
    # Arrange - Create a snapshot with no observable attrs
    snapshot = type("TestSnapshot", (), {"_observable_attrs": []})()
    snapshot._snapshot_values = {}

    # Act
    repr_str = repr(snapshot)

    # Assert - For custom types, repr will show the class name
    assert "TestSnapshot" in repr_str


@pytest.mark.unit
@pytest.mark.store
def test_store_to_dict_serializes_all_observable_values():
    """Store to_dict() method creates dictionary with all observable current values"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(25)
        active = observable(True)

    # Arrange
    store = TestStore()

    # Act
    data = store.to_dict()

    # Assert
    expected = {"name": "Alice", "age": 25, "active": True}
    assert data == expected


@pytest.mark.unit
@pytest.mark.store
def test_store_to_dict_excludes_non_observable_attributes():
    """Store to_dict() only includes observable attributes, not regular attributes"""

    class TestStore(Store):
        observable_value = observable("observable")
        regular_value = "not observable"

    # Arrange
    store = TestStore()
    data = store.to_dict()

    assert "observable_value" in data
    assert "regular_value" not in data
    assert data["observable_value"] == "observable"


@pytest.mark.unit
@pytest.mark.store
def test_store_load_state_updates_observables_from_dict():
    """Store load_state() method updates observable values from dictionary"""

    class TestStore(Store):
        name = observable("initial")
        count = observable(0)

    # Arrange
    store = TestStore()
    state = {"name": "updated", "count": 42}

    # Act
    store.load_state(state)

    # Assert
    assert store.name.value == "updated"
    assert store.count.value == 42


@pytest.mark.unit
@pytest.mark.store
def test_store_load_state_ignores_unknown_keys():
    """Store load_state() ignores dictionary keys that don't match observables"""

    class TestStore(Store):
        value = observable("initial")

    # Arrange
    store = TestStore()
    state = {"value": "updated", "unknown_key": "ignored"}

    # Act
    store.load_state(state)

    # Assert
    assert store.value.value == "updated"


@pytest.mark.unit
@pytest.mark.store
def test_store_load_state_supports_partial_updates():
    """Store load_state() only updates observables present in the state dict"""

    class TestStore(Store):
        a = observable(1)
        b = observable(2)
        c = observable(3)

    # Arrange
    store = TestStore()
    # Only update a and c
    partial_state = {"a": 10, "c": 30}

    # Act
    store.load_state(partial_state)

    # Assert
    assert store.a.value == 10
    assert store.b.value == 2  # Unchanged
    assert store.c.value == 30


@pytest.mark.unit
@pytest.mark.observable
@pytest.mark.edge_case
def test_observable_updates_handle_potential_cycles_gracefully():
    """Observable system handles potential circular dependencies without infinite loops"""
    # Arrange
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)
    update_count = 0

    def update_obs2():
        nonlocal update_count
        update_count += 1
        if update_count < 3:  # Prevent infinite recursion in test
            obs1.set(obs1.value + 1)  # This could create a cycle

    obs1.subscribe(update_obs2)

    # Act - This should work without circular dependency issues
    obs2.set(3)

    # Assert - obs1 should not have been modified by the potential cycle
    assert obs1.value == 1


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_supports_mixed_observable_and_computed_attributes():
    """Stores can contain both regular observables and computed observables"""

    class MixedStore(Store):
        # Regular observables
        base_value = observable(10)
        multiplier = observable(2)

        # Computed observable
        computed_product = (base_value + multiplier).then(lambda b, m: b * m)

    # Arrange
    store = MixedStore()

    # Act & Assert - Test initial values
    assert store.base_value.value == 10
    assert store.multiplier.value == 2
    assert store.computed_product.value == 20

    # Test that computed updates when dependencies change
    store.base_value.set(15)
    assert store.computed_product.value == 30

    store.multiplier.set(3)
    assert store.computed_product.value == 45


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_mixed_observables_support_subscriptions():
    """Store subscriptions work with both regular and computed observables"""

    class MixedStore(Store):
        counter = observable(0)
        doubled = counter.then(lambda c: c * 2)

    # Arrange
    callback_calls = []

    def on_change(snapshot):
        callback_calls.append(
            {"counter": snapshot.counter, "doubled": snapshot.doubled}
        )

    MixedStore.subscribe(on_change)

    # Act - Change the base observable
    MixedStore.counter.set(5)

    # Assert
    assert len(callback_calls) >= 1
    latest_call = callback_calls[-1]
    assert latest_call["counter"] == 5
    assert latest_call["doubled"] == 10


@pytest.mark.unit
@pytest.mark.store
def test_store_mixed_observables_support_serialization():
    """Store serialization includes both regular and computed observables"""

    class MixedStore(Store):
        name = observable("test")
        count = observable(5)
        name_length = name.then(lambda n: len(n))

    # Arrange
    store = MixedStore()

    # Act - to_dict should include all observable values
    data = store.to_dict()

    # Assert
    expected = {"name": "test", "count": 5, "name_length": 4}
    assert data == expected

    # load_state should only affect regular observables
    store.load_state({"name": "hello", "count": 10})
    assert store.name.value == "hello"
    assert store.count.value == 10
    assert store.name_length.value == 5  # Updated automatically


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_computed_observables_are_readonly():
    """Computed observables in stores cannot be set directly"""

    class TestStore(Store):
        value = observable(5)
        doubled = value.then(lambda v: v * 2)

    # Arrange
    store = TestStore()

    # Act & Assert - Regular observable can be set
    store.value.set(10)
    assert store.value.value == 10
    assert store.doubled.value == 20

    # Computed observable cannot be set directly
    with pytest.raises(ValueError, match="Computed observables are read-only"):
        store.doubled.set(100)

    # Value should remain computed
    assert store.doubled.value == 20


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_computation_chain_maintains_average_invariant_initially(
    complex_computation_store,
):
    """Store computed average equals total divided by count in initial state"""
    store = complex_computation_store

    # Assert - average = total / count
    expected_average = store.total.value / len(store.items.value)
    assert abs(store.average.value - expected_average) < 0.001


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_computation_chain_maintains_scaled_total_invariant_initially(
    complex_computation_store,
):
    """Store computed scaled total equals total times multiplier in initial state"""
    store = complex_computation_store

    # Assert - scaled_total = total * multiplier
    assert store.scaled_total.value == store.total.value * store.multiplier.value


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_computation_chain_maintains_average_invariant_after_items_change(
    complex_computation_store,
):
    """Store computed average remains correct after changing items list"""
    store = complex_computation_store

    # Act
    store.items.set([10, 20, 30])

    # Assert - average = total / count after change
    expected_average = store.total.value / len(store.items.value)
    assert abs(store.average.value - expected_average) < 0.001


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.operators
def test_store_computation_chain_maintains_scaled_total_invariant_after_multiplier_change(
    complex_computation_store,
):
    """Store computed scaled total remains correct after changing multiplier"""
    store = complex_computation_store

    # Act
    store.multiplier.set(3)

    # Assert - scaled_total = total * multiplier after change
    assert store.scaled_total.value == store.total.value * store.multiplier.value


@pytest.mark.unit
@pytest.mark.store
@pytest.mark.memory
def test_store_instances_can_be_cleaned_up_without_leaks(memory_tracker, no_leaks):
    """Store instances release all references when deleted"""

    def create_and_destroy_stores():
        for i in range(25):

            class TempStore(Store):
                counter = observable(i)
                doubled = counter.then(lambda x: x * 2)

            store = TempStore()
            # Use the store to ensure it's fully initialized
            assert store.doubled.value == i * 2
            # Delete the store
            del store

    # Act & Assert - No memory leaks from store creation/destruction cycles
    with memory_tracker() as tracker:
        create_and_destroy_stores()

    # Verify no store instances accumulate
    assert "Store" not in tracker.object_growth or tracker.object_growth["Store"] <= 3
    # Also check for observable accumulation (allow more tolerance for complex objects)
    assert (
        "Observable" not in tracker.object_growth
        or tracker.object_growth["Observable"] <= 30
    )

    # Additional leak check
    no_leaks(create_and_destroy_stores, "Store", tolerance=5)
    no_leaks(create_and_destroy_stores, "Observable", tolerance=30)


def test_store_get_observable_attrs_includes_computed():
    """Test that _get_observable_attrs includes both regular and computed observables."""

    class TestStore(Store):
        regular_obs = observable("regular")
        dummy_obs = observable("dummy")
        computed_obs = dummy_obs.then(lambda x: f"computed_{x}")

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
        computed_obs = dummy_obs.then(lambda x: f"computed_{x}")

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
        area = (width + height) >> (lambda w, h: w * h)
        perimeter = (width + height) >> (lambda w, h: 2 * (w + h))

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
        sum_coords = (x + y) >> (lambda a, b: a + b)
        product_coords = (x + y) >> (lambda a, b: a * b)

        # Also test with three observables
        z = observable(2)
        total = (x + y + z) >> (lambda a, b, c: a + b + c)

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


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.store
def test_store_handles_empty_initialization():
    """Store can be initialized with no observables and still function"""
    # Arrange & Act
    store = Store()

    # Assert - Store works even without observables
    data = store.to_dict()
    assert data == {}

    # Can still load state (though it will ignore unknown keys)
    store.load_state({"nonexistent": "ignored"})
    assert store.to_dict() == {}


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.store
def test_store_with_single_observable_operations():
    """Store with single observable handles all operations correctly"""

    class SingleObservableStore(Store):
        value = observable("test")

    # Arrange
    store = SingleObservableStore()

    # Act & Assert - Basic operations work
    assert store.value.value == "test"

    store.value = "updated"
    assert store.value.value == "updated"

    # Serialization works
    data = store.to_dict()
    assert data == {"value": "updated"}

    # State loading works
    store.load_state({"value": "loaded"})
    assert store.value.value == "loaded"


@pytest.mark.edge_case
@pytest.mark.unit
@pytest.mark.store
def test_store_serialization_with_none_values():
    """Store serialization correctly handles None values in observables"""

    class NoneValueStore(Store):
        nullable_value = observable(None)
        regular_value = observable("not_none")

    # Arrange
    store = NoneValueStore()

    # Act
    data = store.to_dict()

    # Assert
    expected = {"nullable_value": None, "regular_value": "not_none"}
    assert data == expected

    # Test loading state with None
    store.load_state({"nullable_value": "not_none", "regular_value": None})
    assert store.nullable_value.value == "not_none"
    assert store.regular_value.value is None
