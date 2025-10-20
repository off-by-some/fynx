"""
Factory functions and fixtures for FynX tests.

These factories eliminate repetition and ensure consistent test setup across
the test suite. They provide common patterns like diamond dependencies,
subscription tracking, and reactive chain creation.
"""

from fynx import Store, observable


def create_diamond_dependency():
    """Creates diamond pattern: source → (a, b) → combined

    Returns:
        tuple: (source, path_a, path_b, combined)
    """
    source = observable(10)
    path_a = source >> (lambda x: x + 5)
    path_b = source >> (lambda x: x * 2)
    combined = (path_a | path_b) >> (lambda a, b: a + b)

    return source, path_a, path_b, combined


def create_subscription_tracker():
    """Provides a helper for tracking subscription notifications

    Returns:
        Tracker: Object with record() method and values list
    """

    class Tracker:
        def __init__(self):
            self.values = []

        def record(self, value):
            self.values.append(value)

    return Tracker()


def create_temperature_monitor_store():
    """Creates a temperature monitor store with celsius/fahrenheit conversion

    Returns:
        TemperatureMonitor: Store class with reactive temperature conversion
    """

    class TemperatureMonitor(Store):
        celsius = observable(0.0)
        fahrenheit = celsius.then(lambda c: c * 9 / 5 + 32)

    return TemperatureMonitor


def create_user_profile_store():
    """Creates a user profile store with computed full name

    Returns:
        UserProfile: Store class with reactive full name computation
    """

    class UserProfile(Store):
        first_name = observable("")
        last_name = observable("")
        full_name = (first_name | last_name).then(lambda f, l: f"{f} {l}".strip())

    return UserProfile


def create_counter_with_limits():
    """Creates a counter with min/max bounds checking

    Returns:
        tuple: (counter_observable, min_val, max_val, is_valid_computed)
    """
    counter = observable(0)
    min_val = observable(0)
    max_val = observable(100)

    is_valid = (counter | min_val | max_val).then(
        lambda c, min_v, max_v: min_v <= c <= max_v
    )

    return counter, min_val, max_val, is_valid


def create_reactive_filter_chain():
    """Creates a chain: source → filter → transform → result

    Returns:
        tuple: (source, predicate, multiplier, filtered_result)
    """
    source = observable(0)
    predicate = observable(lambda x: x > 0)
    multiplier = observable(2)

    # Create filtered observable that only passes values matching predicate
    filtered = (source | predicate).then(lambda val, pred: val if pred(val) else None)

    # Transform non-None values
    result = (filtered | multiplier).then(
        lambda filtered_val, mult: (
            filtered_val * mult if filtered_val is not None else 0
        )
    )

    return source, predicate, multiplier, result


def create_circular_dependency_attempt():
    """Creates observables that could form a circular dependency

    Returns:
        tuple: (obs_a, obs_b) - use carefully to test circular dependency detection
    """
    obs_a = observable(1)
    obs_b = observable(2)

    # Note: Don't actually create the circular dependency here
    # This is just for setup in tests that will attempt to create cycles

    return obs_a, obs_b


def create_watch_condition_observables():
    """Creates observables commonly used in watch decorator tests

    Returns:
        dict: Dictionary with 'status', 'count', 'ready', 'active' observables
    """
    return {
        "status": observable("idle"),
        "count": observable(0),
        "ready": observable(False),
        "active": observable(True),
    }


def create_memory_test_chain():
    """Creates a reactive chain suitable for memory leak testing

    Returns:
        tuple: (source, chain_elements) - chain_elements is list for easy cleanup testing
    """
    source = observable(10)
    chain_elements = []

    # Build a chain of transformations
    step1 = source >> (lambda x: x * 2)
    chain_elements.append(step1)

    step2 = step1 >> (lambda x: x + 5)
    chain_elements.append(step2)

    step3 = step2 >> (lambda x: f"result: {x}")
    chain_elements.append(step3)

    return source, chain_elements
