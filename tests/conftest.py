# Pytest configuration and fixtures for FynX tests

import pytest

from fynx.observable.base import Observable
from fynx.optimizer import OptimizationContext
from tests.test_factories import (
    create_counter_with_limits,
    create_diamond_dependency,
    create_memory_test_chain,
    create_reactive_filter_chain,
    create_subscription_tracker,
    create_temperature_monitor_store,
    create_user_profile_store,
    create_watch_condition_observables,
)
from tests.utils.memory_utils import (
    create_memory_tracker_fixture,
    create_no_leaks_fixture,
)

# Memory testing fixtures
memory_tracker = create_memory_tracker_fixture()
no_leaks = create_no_leaks_fixture()


@pytest.fixture(autouse=True)
def reset_observable_state():
    """Reset Observable class state between tests to prevent contamination."""
    # Store original state
    original_pending = Observable._pending_notifications.copy()
    original_scheduled = Observable._notification_scheduled
    original_notifying = Observable._currently_notifying.copy()
    original_context = Observable._current_context
    original_stack = Observable._context_stack.copy()

    # Store optimizer context
    original_optimizer_context = getattr(
        OptimizationContext._thread_local, "current", None
    )

    # Reset to clean state
    Observable._pending_notifications.clear()
    Observable._notification_scheduled = False
    Observable._currently_notifying.clear()
    Observable._current_context = None
    Observable._context_stack.clear()

    # Reset optimizer context for this thread
    OptimizationContext._thread_local.current = None

    yield

    # Restore original state (though this shouldn't be necessary for autouse fixtures)
    Observable._pending_notifications = original_pending
    Observable._notification_scheduled = original_scheduled
    Observable._currently_notifying = original_notifying
    Observable._current_context = original_context
    Observable._context_stack = original_stack

    # Restore optimizer context
    OptimizationContext._thread_local.current = original_optimizer_context


@pytest.fixture
def subscription_tracker():
    """Pytest fixture for tracking subscription notifications"""
    return create_subscription_tracker()


@pytest.fixture
def diamond_dependency():
    """Pytest fixture providing a diamond dependency pattern"""
    return create_diamond_dependency()


@pytest.fixture
def temperature_monitor():
    """Pytest fixture providing a temperature monitor store"""
    return create_temperature_monitor_store()


@pytest.fixture
def user_profile():
    """Pytest fixture providing a user profile store"""
    return create_user_profile_store()


@pytest.fixture
def counter_with_limits():
    """Pytest fixture providing a counter with bounds checking"""
    return create_counter_with_limits()


@pytest.fixture
def reactive_filter_chain():
    """Pytest fixture providing a reactive filter chain"""
    return create_reactive_filter_chain()


@pytest.fixture
def watch_observables():
    """Pytest fixture providing observables for watch decorator testing"""
    return create_watch_condition_observables()


@pytest.fixture
def memory_test_chain():
    """Pytest fixture providing a reactive chain for memory testing"""
    return create_memory_test_chain()


@pytest.fixture
def complex_computation_store():
    """Setup fixture providing store with complex computation chains"""
    from fynx import Store, observable

    class ComplexStore(Store):
        items = observable([1, 2, 3, 4, 5])
        multiplier = observable(2)

        # Computed: sum of items
        total = items.then(lambda items: sum(items))

        # Computed: average of items
        average = items.then(
            lambda items: sum(items) / len(items) if len(items) > 0 else 0
        )

        # Computed: scaled total
        scaled_total = (total + multiplier).then(
            lambda total, multiplier: total * multiplier
        )

    return ComplexStore()
