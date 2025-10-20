"""Unit tests for observable interfaces and abstract base classes."""

from abc import ABC

import pytest

from fynx.observable.interfaces import (
    Conditional,
    Mergeable,
    Observable,
    ReactiveContext,
)


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_is_abstract_base_class():
    """ReactiveContext is an abstract base class that cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ReactiveContext()


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_has_abstract_methods():
    """ReactiveContext defines abstract methods that must be implemented."""
    assert hasattr(ReactiveContext, "run")
    assert hasattr(ReactiveContext, "dispose")

    # Check that these are abstract methods
    assert ReactiveContext.run.__isabstractmethod__
    assert ReactiveContext.dispose.__isabstractmethod__


@pytest.mark.unit
@pytest.mark.observable
def test_observable_is_abstract_base_class():
    """Observable is an abstract base class that cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Observable()


@pytest.mark.unit
@pytest.mark.observable
def test_observable_has_abstract_properties_and_methods():
    """Observable defines abstract properties and methods that must be implemented."""
    assert hasattr(Observable, "key")
    assert hasattr(Observable, "value")
    assert hasattr(Observable, "set")
    assert hasattr(Observable, "subscribe")
    assert hasattr(Observable, "add_observer")

    # Check that these are abstract
    assert Observable.key.__isabstractmethod__
    assert Observable.value.__isabstractmethod__
    assert Observable.set.__isabstractmethod__
    assert Observable.subscribe.__isabstractmethod__
    assert Observable.add_observer.__isabstractmethod__


@pytest.mark.unit
@pytest.mark.observable
def test_mergeable_inherits_from_observable():
    """Mergeable inherits from Observable and adds merge-specific functionality."""
    assert issubclass(Mergeable, Observable)
    assert issubclass(Mergeable, ABC)


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_is_abstract_base_class():
    """Conditional is an abstract base class that cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Conditional()


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_has_abstract_properties():
    """Conditional defines abstract properties that must be implemented."""
    assert hasattr(Conditional, "is_active")

    # Check that this is abstract
    assert Conditional.is_active.__isabstractmethod__


@pytest.mark.unit
@pytest.mark.observable
def test_concrete_reactive_context_implementation():
    """Concrete ReactiveContext implementation works correctly."""

    class ConcreteReactiveContext(ReactiveContext):
        def __init__(self):
            self.run_called = False
            self.dispose_called = False

        def run(self) -> None:
            self.run_called = True

        def dispose(self) -> None:
            self.dispose_called = True

    # Should be able to instantiate concrete implementation
    context = ConcreteReactiveContext()
    assert isinstance(context, ReactiveContext)

    # Should be able to call abstract methods
    context.run()
    assert context.run_called is True

    context.dispose()
    assert context.dispose_called is True


@pytest.mark.unit
@pytest.mark.observable
def test_concrete_observable_implementation():
    """Concrete Observable implementation works correctly."""

    class ConcreteObservable(Observable[str]):
        def __init__(self, key: str, initial_value: str):
            self._key = key
            self._value = initial_value
            self._observers = []

        @property
        def key(self) -> str:
            return self._key

        @property
        def value(self) -> str:
            return self._value

        def set(self, value: str) -> None:
            self._value = value
            for observer in self._observers:
                observer()

        def subscribe(self, func):
            self._observers.append(func)
            return self

        def add_observer(self, observer) -> None:
            self._observers.append(observer)

    # Should be able to instantiate concrete implementation
    obs = ConcreteObservable("test", "initial")
    assert isinstance(obs, Observable)

    # Should be able to use abstract methods
    assert obs.key == "test"
    assert obs.value == "initial"

    obs.set("updated")
    assert obs.value == "updated"

    # Test subscription
    received = []
    obs.subscribe(lambda: received.append("notified"))
    obs.set("changed")
    assert received == ["notified"]


@pytest.mark.unit
@pytest.mark.observable
def test_concrete_mergeable_implementation():
    """Concrete Mergeable implementation works correctly."""

    class ConcreteMergeable(Mergeable[str]):
        def __init__(self, key: str, initial_value: str):
            self._key = key
            self._value = initial_value
            self._observers = []

        @property
        def key(self) -> str:
            return self._key

        @property
        def value(self) -> str:
            return self._value

        def set(self, value: str) -> None:
            self._value = value
            for observer in self._observers:
                observer()

        def subscribe(self, func):
            self._observers.append(func)
            return self

        def add_observer(self, observer) -> None:
            self._observers.append(observer)

    # Should be able to instantiate concrete implementation
    merged = ConcreteMergeable("merged", "value")
    assert isinstance(merged, Mergeable)
    assert isinstance(merged, Observable)

    # Should work like regular observable
    assert merged.key == "merged"
    assert merged.value == "value"


@pytest.mark.unit
@pytest.mark.observable
def test_concrete_conditional_implementation():
    """Concrete Conditional implementation works correctly."""

    class ConcreteConditional(Conditional[str]):
        def __init__(self, active: bool = True):
            self._active = active
            self._value = "test"
            self._key = "concrete"
            self._observers = set()

        @property
        def is_active(self) -> bool:
            return self._active

        @property
        def key(self) -> str:
            return self._key

        @property
        def value(self) -> str:
            return self._value

        def set(self, value: str) -> None:
            self._value = value

        def subscribe(self, func) -> "ConcreteConditional":
            self._observers.add(func)
            return self

        def add_observer(self, observer) -> None:
            self._observers.add(observer)

        def remove_observer(self, observer) -> None:
            self._observers.discard(observer)

    # Should be able to instantiate concrete implementation
    conditional = ConcreteConditional(True)
    assert isinstance(conditional, Conditional)

    # Should be able to use abstract property
    assert conditional.is_active is True

    # Test inactive state
    inactive = ConcreteConditional(False)
    assert inactive.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_incomplete_reactive_context_implementation_fails():
    """Incomplete ReactiveContext implementation raises TypeError."""

    class IncompleteReactiveContext(ReactiveContext):
        def run(self) -> None:
            pass

        # Missing dispose method

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteReactiveContext()


@pytest.mark.unit
@pytest.mark.observable
def test_incomplete_observable_implementation_fails():
    """Incomplete Observable implementation raises TypeError."""

    class IncompleteObservable(Observable[str]):
        @property
        def key(self) -> str:
            return "test"

        @property
        def value(self) -> str:
            return "test"

        def set(self, value: str) -> None:
            pass

        # Missing subscribe and add_observer methods

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteObservable()


@pytest.mark.unit
@pytest.mark.observable
def test_incomplete_conditional_implementation_fails():
    """Incomplete Conditional implementation raises TypeError."""

    class IncompleteConditional(Conditional[str]):
        # Missing is_active property
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteConditional()


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_run_abstract_method():
    """Test ReactiveContext.run() abstract method (line 99)"""

    # Create a concrete implementation
    class ConcreteReactiveContext(ReactiveContext):
        def run(self) -> None:
            pass

        def dispose(self) -> None:
            pass

    context = ConcreteReactiveContext()
    # Should be able to call run without error
    context.run()


@pytest.mark.unit
@pytest.mark.observable
def test_reactive_context_dispose_abstract_method():
    """Test ReactiveContext.dispose() abstract method (line 109)"""

    # Create a concrete implementation
    class ConcreteReactiveContext(ReactiveContext):
        def run(self) -> None:
            pass

        def dispose(self) -> None:
            pass

    context = ConcreteReactiveContext()
    # Should be able to call dispose without error
    context.dispose()


@pytest.mark.unit
@pytest.mark.observable
def test_observable_key_abstract_property():
    """Test Observable.key abstract property (line 138)"""

    # Create a concrete implementation
    class ConcreteObservable(Observable[str]):
        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return "test_value"

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteObservable":
            return self

        def add_observer(self, observer) -> None:
            pass

    obs = ConcreteObservable()
    assert obs.key == "test_key"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_value_abstract_property():
    """Test Observable.value abstract property (line 152)"""

    # Create a concrete implementation
    class ConcreteObservable(Observable[str]):
        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return "test_value"

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteObservable":
            return self

        def add_observer(self, observer) -> None:
            pass

    obs = ConcreteObservable()
    assert obs.value == "test_value"


@pytest.mark.unit
@pytest.mark.observable
def test_observable_set_abstract_method():
    """Test Observable.set() abstract method (line 166)"""

    # Create a concrete implementation
    class ConcreteObservable(Observable[str]):
        def _value(self):
            return "test_value"

        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return self._value()

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteObservable":
            return self

        def add_observer(self, observer) -> None:
            pass

    obs = ConcreteObservable()
    # Should be able to call set without error
    obs.set("new_value")


@pytest.mark.unit
@pytest.mark.observable
def test_observable_subscribe_abstract_method():
    """Test Observable.subscribe() abstract method (line 182)"""

    # Create a concrete implementation
    class ConcreteObservable(Observable[str]):
        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return "test_value"

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteObservable":
            return self

        def add_observer(self, observer) -> None:
            pass

    obs = ConcreteObservable()
    # Should be able to call subscribe without error
    result = obs.subscribe(lambda x: None)
    assert result is obs


@pytest.mark.unit
@pytest.mark.observable
def test_observable_add_observer_abstract_method():
    """Test Observable.add_observer() abstract method (line 193)"""

    # Create a concrete implementation
    class ConcreteObservable(Observable[str]):
        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return "test_value"

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteObservable":
            return self

        def add_observer(self, observer) -> None:
            pass

    obs = ConcreteObservable()
    # Should be able to call add_observer without error
    obs.add_observer(lambda: None)


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_is_active_abstract_property():
    """Test Conditional.is_active abstract property (line 231)"""

    # Create a concrete implementation
    class ConcreteConditional(Conditional[str]):
        @property
        def key(self) -> str:
            return "test_key"

        @property
        def value(self) -> str:
            return "test_value"

        def set(self, value: str) -> None:
            pass

        def subscribe(self, func) -> "ConcreteConditional":
            return self

        def add_observer(self, observer) -> None:
            pass

        @property
        def is_active(self) -> bool:
            return True

    conditional = ConcreteConditional()
    assert conditional.is_active is True
