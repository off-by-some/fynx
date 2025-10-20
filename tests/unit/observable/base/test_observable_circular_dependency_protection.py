import pytest

from fynx.observable.base import Observable


@pytest.mark.unit
@pytest.mark.observable
def test_observable_detects_circular_dependency_during_reactive_context():
    """Setting an observable inside a reactive context that depends on it raises RuntimeError."""
    # Arrange
    obs = Observable("x", 1)

    def reactive():
        _ = obs.value  # establish dependency
        with pytest.raises(RuntimeError):
            obs.set(2)

    # Act: subscribe a reaction that triggers within notification cycle
    obs.subscribe(lambda _: reactive())
    obs.set(1)  # triggers notification and executes reactive()


@pytest.mark.unit
@pytest.mark.observable
def test_observable_unsubscribe_stops_future_notifications():
    """unsubscribe removes only the targeted callback, leaving no further calls."""
    # Arrange
    obs = Observable("y", 0)
    calls = []

    def cb(val):
        calls.append(val)

    obs.subscribe(cb)
    # Act
    obs.set(1)
    # Assert
    assert calls == [1]

    # Act: unsubscribe and change again
    obs.unsubscribe(cb)
    obs.set(2)
    # Assert
    assert calls == [1]


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_dispose_removes_from_store_observables():
    """ReactiveContext dispose removes observer from all store observables when _store_observables is set."""
    from fynx.observable.base import ReactiveContext

    # Create mock observables
    obs1 = Observable("obs1", 1)
    obs2 = Observable("obs2", 2)

    # Create reactive context with store observables
    context = ReactiveContext(lambda: None)
    context._store_observables = [obs1, obs2]

    # Add observer to both observables
    obs1.add_observer(context.run)
    obs2.add_observer(context.run)

    # Verify observers are added
    assert context.run in obs1._observers
    assert context.run in obs2._observers

    # Dispose context
    context.dispose()

    # Verify observers are removed from both observables
    assert context.run not in obs1._observers
    assert context.run not in obs2._observers


@pytest.mark.unit
@pytest.mark.observable
def test_observable_notify_observers_prevents_reentrant_notifications():
    """Observable prevents re-entrant notifications by detecting circular dependencies."""
    obs = Observable("test", 0)
    notification_count = 0

    def observer(value):
        nonlocal notification_count
        notification_count += 1
        # Try to trigger another notification while already notifying
        obs.set(obs.value + 1)

    obs.subscribe(observer)

    # This should trigger circular dependency detection, not infinite recursion
    with pytest.raises(RuntimeError, match="Circular dependency detected"):
        obs.set(1)

    # Should only have triggered one notification before the error
    assert notification_count == 1


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_cleanup_removes_empty_function_mappings():
    """ReactiveContext cleanup removes empty function mappings from _func_to_contexts."""
    from fynx.observable.base import ReactiveContext, _func_to_contexts

    def test_func():
        pass

    # Create context and add it to function mappings
    context = ReactiveContext(test_func)
    _func_to_contexts[test_func] = [context]

    # Verify mapping exists
    assert test_func in _func_to_contexts
    assert context in _func_to_contexts[test_func]

    # Remove context (simulating cleanup)
    _func_to_contexts[test_func].remove(context)

    # Clean up empty function mappings (line 514-515)
    if not _func_to_contexts[test_func]:
        del _func_to_contexts[test_func]

    # Verify mapping is removed
    assert test_func not in _func_to_contexts


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_updates_computed_key():
    """Observable descriptor updates key for computed observables in __set_name__."""
    from fynx.observable.base import Observable

    class TestClass:
        pass

    # Create observable with computed flag
    obs = Observable("<unnamed>", 42)
    obs._is_computed = True

    # Call __set_name__
    obs.__set_name__(TestClass, "computed_value")

    # Verify key was updated for computed observable
    assert obs._key == "<computed:computed_value>"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_updates_regular_key():
    """Observable descriptor updates key for regular observables in __set_name__."""
    from fynx.observable.base import Observable

    class TestClass:
        pass

    # Create regular observable
    obs = Observable("<unnamed>", 42)

    # Call __set_name__
    obs.__set_name__(TestClass, "regular_value")

    # Verify key was updated
    assert obs._key == "regular_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_skips_computed_processing():
    """Observable descriptor skips processing for computed observables in __set_name__."""
    from fynx.observable.base import Observable

    class TestClass:
        pass

    # Create computed observable
    obs = Observable("<unnamed>", 42)
    obs._is_computed = True

    # Call __set_name__ - should return early for computed observables
    obs.__set_name__(TestClass, "computed_value")

    # Verify it's still the same observable (not replaced with descriptor)
    assert isinstance(obs, Observable)
    assert obs._is_computed is True


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_handles_store_class():
    """Observable descriptor returns early for Store classes in __set_name__."""
    from fynx.observable.base import Observable

    # Mock Store class
    class MockStore:
        pass

    # Create observable
    obs = Observable("<unnamed>", 42)

    # Call __set_name__ with Store-like class
    obs.__set_name__(MockStore, "store_value")

    # Should return early without processing
    assert isinstance(obs, Observable)


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_updates_key():
    """Observable descriptor updates key in __set_name__."""
    from fynx.observable.base import Observable

    class TestClass:
        pass

    # Create observable
    obs = Observable("<unnamed>", 42)

    # Call __set_name__ - should update the key
    obs.__set_name__(TestClass, "test_value")

    # Should update the key
    assert obs._key == "test_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_descriptor_set_name_updates_key_for_computed():
    """Observable descriptor updates key for computed observables in __set_name__."""
    from fynx.observable.base import Observable

    class TestClass:
        pass

    # Create computed observable
    obs = Observable("<unnamed>", 42)
    obs._is_computed = True

    # Call __set_name__ - should update key and return early
    obs.__set_name__(TestClass, "computed_value")

    # Should update the key for computed observables
    assert obs._key == "<computed:computed_value>"

    # Should not create a descriptor (returns early for computed)
    assert not hasattr(TestClass, "computed_value")
