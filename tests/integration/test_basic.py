"""Integration tests for FynX library reactive system interactions."""

import pytest

from fynx import Observable, ReactiveContext, Store, observable, reactive


@pytest.mark.integration
@pytest.mark.observable
def test_observable_returns_initial_value_when_accessed():
    """Observable provides access to its initial value through value property"""
    obs = Observable("test", "initial")

    assert obs.value == "initial"


@pytest.mark.integration
@pytest.mark.observable
def test_observable_string_representation_returns_current_value():
    """Observable string conversion returns its current value"""
    obs = Observable("test", "initial")

    assert str(obs) == "initial"


@pytest.mark.integration
@pytest.mark.observable
def test_observable_updates_value_when_set_is_called():
    """Observable value changes when set() method is called with new value"""
    obs = Observable("test", "initial")

    obs.set("updated")
    assert obs.value == "updated"


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_pipe_operator_combines_observables_into_tuple():
    """Pipe operator (+) combines multiple observables into a tuple of current values"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")

    merged = obs1 + obs2

    assert merged.value == ("hello", "world")
    assert len(merged) == 2


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_context_manager_enables_unpacking():
    """Merged observable context manager enables tuple unpacking of values"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    merged = obs1 + obs2

    with merged:
        val1, val2 = merged.value
        assert val1 == "hello"
        assert val2 == "world"


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_updates_when_any_source_changes():
    """Merged observable updates its tuple value when any source observable changes"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    merged = obs1 + obs2

    obs1.set("hi")
    assert merged.value == ("hi", "world")

    obs2.set("there")
    assert merged.value == ("hi", "there")


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_pipe_operator_chains_for_multiple_observables():
    """Pipe operator can be chained to combine more than two observables"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    obs3 = Observable("third", "!")

    chained = obs1 + obs2 + obs3

    assert chained.value == ("hello", "world", "!")
    assert len(chained) == 3


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_chained_merged_observable_updates_when_any_source_changes():
    """Chained merged observable updates when any source in the chain changes"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    obs3 = Observable("third", "!")
    chained = obs1 + obs2 + obs3

    obs1.set("hi")
    assert chained.value == ("hi", "world", "!")

    obs3.set("?")
    assert chained.value == ("hi", "world", "?")


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_can_be_extended_with_additional_observables():
    """Existing merged observable can be extended with additional observables"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    obs3 = Observable("third", "!")
    obs4 = Observable("fourth", "extra")

    chained = obs1 + obs2 + obs3
    chained2 = chained + obs4

    assert chained2.value == ("hello", "world", "!", "extra")
    assert len(chained2) == 4


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_extending_merged_observable_preserves_original():
    """Extending a merged observable does not affect the original merged observable"""
    obs1 = Observable("first", "hello")
    obs2 = Observable("second", "world")
    obs3 = Observable("third", "!")
    obs4 = Observable("fourth", "extra")

    chained = obs1 + obs2 + obs3
    chained2 = chained + obs4

    # Verify the original chained is unaffected
    assert chained.value == ("hello", "world", "!")
    assert len(chained) == 3


@pytest.mark.integration
@pytest.mark.store
def test_store_provides_class_level_access_to_observables():
    """Store provides class-level access to observable attributes with initial values"""

    class TestStore(Store):
        count = observable(0)
        name = observable("test")

    assert TestStore.count.value == 0
    assert TestStore.name.value == "test"


@pytest.mark.integration
@pytest.mark.store
def test_store_observables_update_via_attribute_assignment():
    """Store observables can be updated through direct attribute assignment"""

    class TestStore(Store):
        count = observable(0)
        name = observable("test")

    TestStore.count = 5
    TestStore.name = "updated"

    assert TestStore.count.value == 5
    assert TestStore.name.value == "updated"


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.operators
def test_store_observables_can_be_merged_with_pipe_operator():
    """Store observables can be combined using the pipe operator"""

    class TestStore(Store):
        count = observable(0)
        name = observable("test")

    TestStore.count = 5
    TestStore.name = "updated"

    merged = TestStore.count + TestStore.name
    assert merged.value == (5, "updated")


@pytest.mark.integration
@pytest.mark.store
def test_store_serialization_includes_current_observable_values():
    """Store serialization captures current values of all observable attributes"""

    class TestStore(Store):
        count = observable(0)
        name = observable("test")

    TestStore.count = 5
    TestStore.name = "updated"

    state = TestStore().to_dict()
    assert state == {"count": 5, "name": "updated"}


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.subscription
def test_observable_subscription_notifies_callback_on_value_change():
    """Observable subscription calls callback function when value changes"""
    current_name = observable("Alice")
    callback_calls = []

    def on_name_change(name):
        callback_calls.append(f"Name changed to: {name}")

    current_name.subscribe(on_name_change)
    current_name.set("Smith")

    assert callback_calls == ["Name changed to: Smith"]


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.subscription
def test_observable_unsubscription_stops_callback_notifications():
    """Observable unsubscription prevents further callback notifications"""
    current_name = observable("Alice")
    callback_calls = []

    def on_name_change(name):
        callback_calls.append(f"Name changed to: {name}")

    current_name.subscribe(on_name_change)
    current_name.set("Smith")
    current_name.unsubscribe(on_name_change)
    current_name.set("Bob")

    assert callback_calls == ["Name changed to: Smith"]


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.subscription
@pytest.mark.operators
def test_merged_observable_subscription_notifies_on_any_source_change():
    """Merged observable subscription calls callback when any source observable changes"""
    current_name = observable("Alice")
    current_age = observable(30)
    callback_calls = []

    def on_combined_change(values):
        name, age = values
        callback_calls.append(f"Name: {name}, Age: {age}")

    current_name_and_age = current_name + current_age
    current_name_and_age.subscribe(on_combined_change)

    current_name.set("Charlie")
    assert callback_calls == ["Name: Charlie, Age: 30"]

    current_age.set(31)
    assert callback_calls == ["Name: Charlie, Age: 30", "Name: Charlie, Age: 31"]


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.subscription
@pytest.mark.operators
def test_merged_observable_unsubscription_stops_notifications():
    """Merged observable unsubscription prevents further callback notifications"""
    current_name = observable("Alice")
    current_age = observable(30)
    callback_calls = []

    def on_combined_change(values):
        name, age = values
        callback_calls.append(f"Name: {name}, Age: {age}")

    current_name_and_age = current_name + current_age
    current_name_and_age.subscribe(on_combined_change)
    current_name.set("Charlie")
    current_age.set(31)
    current_name_and_age.unsubscribe(on_combined_change)
    current_age.set(32)

    assert callback_calls == ["Name: Charlie, Age: 30", "Name: Charlie, Age: 31"]


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.subscription
def test_store_subscription_notifies_callback_on_observable_change():
    """Store subscription calls callback when any observable attribute changes"""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    def on_store_change(store):
        callback_calls.append(
            f"Store: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    TestStore.subscribe(on_store_change)
    TestStore.height_cm = 170.0

    assert len(callback_calls) == 1
    assert "height=170.0" in callback_calls[0]


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.subscription
def test_store_subscription_tracks_multiple_observable_changes():
    """Store subscription tracks changes to multiple observable attributes"""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    def on_store_change(store):
        callback_calls.append(
            f"Store: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    TestStore.subscribe(on_store_change)
    TestStore.height_cm = 170.0
    TestStore.name = "Bob"
    TestStore.age = 31

    assert len(callback_calls) == 3
    assert "height=170.0" in callback_calls[0]
    assert "name=Bob" in callback_calls[1]
    assert "age=31" in callback_calls[2]


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.subscription
def test_store_unsubscription_stops_callback_notifications():
    """Store unsubscription prevents further callback notifications"""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    def on_store_change(store):
        callback_calls.append(
            f"Store: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    TestStore.subscribe(on_store_change)
    TestStore.height_cm = 170.0
    TestStore.name = "Bob"
    TestStore.age = 31
    TestStore.unsubscribe(on_store_change)
    TestStore.height_cm = 180.0

    assert len(callback_calls) == 3  # No new calls after unsubscribe


@pytest.mark.integration
@pytest.mark.reactive
@pytest.mark.store
def test_reactive_decorator_calls_function_on_store_change():
    """Reactive decorator calls decorated function when store observables change"""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    @reactive(TestStore)
    def on_store_change(store):
        callback_calls.append(
            f"Decorator: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    TestStore.height_cm = 170.2

    assert len(callback_calls) == 2  # Called immediately + on change
    assert "height=170.2" in callback_calls[1]


@pytest.mark.integration
@pytest.mark.reactive
@pytest.mark.store
def test_reactive_decorator_unsubscription_stops_calls():
    """Reactive decorator unsubscription prevents further function calls"""

    class TestStore(Store):
        height_cm = observable(160.0)
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    @reactive(TestStore)
    def on_store_change(store):
        callback_calls.append(
            f"Decorator: height={store.height_cm}, name={store.name}, age={store.age}"
        )

    TestStore.height_cm = 170.2
    on_store_change.unsubscribe()
    TestStore.name = "Charlie"

    assert len(callback_calls) == 2  # No new calls after unsubscribe


@pytest.mark.integration
@pytest.mark.reactive
@pytest.mark.observable
def test_reactive_decorator_calls_function_immediately_with_initial_values():
    """Reactive decorator calls function immediately with initial observable values"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    @reactive(TestStore.age, TestStore.name)
    def on_multi_change(age, name):
        callback_calls.append(f"Multi: name={name}, age={age}")

    assert callback_calls == ["Multi: name=Alice, age=30"]


@pytest.mark.integration
@pytest.mark.reactive
@pytest.mark.observable
def test_reactive_decorator_tracks_multiple_observable_changes():
    """Reactive decorator tracks changes to multiple specified observables"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    @reactive(TestStore.age, TestStore.name)
    def on_multi_change(age, name):
        callback_calls.append(f"Multi: name={name}, age={age}")

    TestStore.age = 31
    TestStore.name = "Barbara"

    assert callback_calls == [
        "Multi: name=Alice, age=30",
        "Multi: name=Alice, age=31",
        "Multi: name=Barbara, age=31",
    ]


@pytest.mark.integration
@pytest.mark.reactive
@pytest.mark.observable
def test_reactive_decorator_unsubscription_stops_multiple_observable_tracking():
    """Reactive decorator unsubscription stops tracking multiple observables"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    @reactive(TestStore.age, TestStore.name)
    def on_multi_change(age, name):
        callback_calls.append(f"Multi: name={name}, age={age}")

    TestStore.age = 31
    TestStore.name = "Barbara"
    on_multi_change.unsubscribe()
    TestStore.age = 32

    assert len(callback_calls) == 3  # No new calls after unsubscribe


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_context_manager_executes_callback_immediately():
    """Merged observable context manager executes callback immediately with current values"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    def on_context_change(name, age):
        callback_calls.append(f"Context: name={name}, age={age}")

    with TestStore.name + TestStore.age as react:
        react(on_context_change)
        assert callback_calls == ["Context: name=Alice, age=30"]


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_merged_observable_context_manager_tracks_changes_during_execution():
    """Merged observable context manager tracks changes during context execution"""

    class TestStore(Store):
        name = observable("Alice")
        age = observable(30)

    callback_calls = []

    def on_context_change(name, age):
        callback_calls.append(f"Context: name={name}, age={age}")

    with TestStore.name + TestStore.age as react:
        react(on_context_change)
        TestStore.name = "Bob"
        TestStore.age = 31

        assert callback_calls == [
            "Context: name=Alice, age=30",
            "Context: name=Bob, age=30",
            "Context: name=Bob, age=31",
        ]


@pytest.mark.integration
@pytest.mark.store
def test_store_snapshot_string_representation_includes_values():
    """StoreSnapshot string representation includes current snapshot values"""
    from fynx.store import StoreSnapshot

    class TestStore(Store):
        pass

    snapshot = StoreSnapshot(TestStore, ["name", "age"])
    snapshot._snapshot_values = {"name": "Alice", "age": 30}

    repr_str = repr(snapshot)
    assert repr_str == "StoreSnapshot(name='Alice', age=30)"


@pytest.mark.integration
@pytest.mark.store
def test_store_snapshot_string_representation_handles_empty_snapshot():
    """StoreSnapshot string representation handles empty snapshot correctly"""
    from fynx.store import StoreSnapshot

    class TestStore(Store):
        pass

    empty_snapshot = StoreSnapshot(TestStore, [])
    assert repr(empty_snapshot) == "StoreSnapshot()"


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.computed
def test_then_operator_creates_computed_observable_from_single_source():
    """Then operator creates computed observable that transforms single source observable"""
    base = observable(5)
    doubled = base.then(lambda x: x * 2)

    assert doubled.value == 10  # Initial computation

    base.set(7)
    assert doubled.value == 14


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.computed
@pytest.mark.operators
def test_then_operator_creates_computed_observable_from_merged_sources():
    """Then operator creates computed observable that transforms merged observable sources"""
    num1 = observable(3)
    num2 = observable(4)
    combined = num1 + num2
    summed = combined.then(lambda a, b: a + b)

    assert summed.value == 7  # 3 + 4

    num1.set(10)
    assert summed.value == 14  # 10 + 4

    num2.set(6)
    assert summed.value == 16  # 10 + 6


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.computed
def test_computed_observables_can_be_chained():
    """Computed observables can be chained to create transformation pipelines"""
    num1 = observable(3)
    num2 = observable(4)
    combined = num1 + num2
    summed = combined.then(lambda a, b: a + b)
    final = summed.then(lambda s: s * 2)

    assert final.value == 14  # (3 + 4) * 2

    num1.set(1)
    assert final.value == 10  # (1 + 4) * 2


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_rshift_operator_chains_transformations_on_single_observable():
    """Right shift operator (>>) chains transformations on single observable"""
    base = observable(5)
    doubled = base >> (lambda x: x * 2)
    tripled = doubled >> (lambda x: x * 3)

    assert doubled.value == 10  # 5 * 2
    assert tripled.value == 30  # (5 * 2) * 3

    base.set(10)
    assert doubled.value == 20  # 10 * 2
    assert tripled.value == 60  # (10 * 2) * 3


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.operators
def test_rshift_operator_chains_transformations_on_merged_observables():
    """Right shift operator (>>) chains transformations on merged observables"""
    num1 = observable(3)
    num2 = observable(4)
    combined = num1 + num2
    summed = combined >> (lambda a, b: a + b)
    formatted = summed >> (lambda s: f"Sum: {s}")

    assert summed.value == 7  # 3 + 4
    assert formatted.value == "Sum: 7"

    num1.set(10)
    assert summed.value == 14  # 10 + 4
    assert formatted.value == "Sum: 14"


@pytest.mark.integration
@pytest.mark.observable
@pytest.mark.edge_case
def test_circular_dependency_detection_raises_runtime_error():
    """Circular dependencies in computed observables raise RuntimeError"""

    # Create a true circular dependency
    obs_a = observable(1)

    # Create computed that modifies its source during computation
    computed_obs = obs_a >> (lambda x: (obs_a.set(x + 1), x)[1])

    # Subscribe to make it evaluate immediately when source changes
    computed_obs.subscribe(lambda v: None)

    # The circular dependency is detected when the source is set
    with pytest.raises(RuntimeError, match="Circular dependency detected"):
        obs_a.set(
            5
        )  # This triggers immediate evaluation and circular dependency detection
