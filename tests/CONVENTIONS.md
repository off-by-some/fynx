# Testing Reactive Systems: A Guide for FynX

*On writing tests that clarify rather than obscure*

## Quick Reference: Best Practices

**Core Principles**
- [Test relationships, not outputs](#the-central-insight) — Verify that invariants hold over time
- [One test, one question](#one-test-one-question) — Each test should answer exactly one thing
- [Ensure complete isolation](#the-non-negotiable-test-isolation) — Tests must never depend on each other
- [Name tests like specifications](#names-that-tell-stories) — Be precise about behavior, not just components

**Writing Better Tests**
- [Structure in three phases](#the-three-phase-structure) — Arrange, Act, Assert
- [Eliminate repetition](#eliminating-repetition) — Use factories and fixtures
- [Test edge cases explicitly](#the-edges-matter) — Boundaries, errors, and extremes
- [Mock at boundaries](#strategic-mocking) — Isolate external dependencies, not core logic
- [Parameterize wisely](#testing-multiple-scenarios-efficiently) — Same behavior, different inputs

**Organization & Maintenance**
- [Use markers for organization](#organization-as-your-test-suite-grows) — Group and run targeted test subsets
- [Mirror source structure](#directory-structure) — Make tests easy to find
- [Write integration tests narrowly](#integration-testing) — Test 2-3 components, not everything
- [Maintain 80% coverage](#about-code-coverage) — But focus on meaningful tests

**Common Mistakes to Avoid**
- Testing implementation details instead of behavior
- Writing overly broad tests that verify multiple things
- Ignoring error conditions and edge cases
- Using poor test names that don't explain behavior
- Sharing mutable state between tests

---

## A Different Kind of Testing

Think about the last time you debugged a failing test. Not your code—the test itself. Maybe it failed randomly, or only when run after another test, or it passed locally but failed in CI. These moments reveal something important: tests can become as complex as the code they're meant to verify.

Testing reactive systems makes this worse. Values flow through transformation chains, updates cascade through dependency graphs, and timing becomes significant in ways it rarely is with pure functions. Traditional testing approaches—arrange, act, assert on a single function call—start to feel inadequate.

This guide takes a different approach. Instead of treating tests as an obligation or a chore, we'll explore how tests can become the clearest expression of what your code actually does. Not documentation that goes stale, not comments that lie—executable specifications that fail loudly when reality diverges from intent.

FynX's foundations in category theory (functors, products, pullbacks) guarantee certain properties at the architectural level. Your tests verify that these guarantees hold in practice, that edge cases behave predictably, and that the reactive patterns you build compose as expected. But more than that, they become a medium for thinking about your system.

## The Central Insight

Here's something that changed how I think about testing reactive code: you're not just testing that functions return correct values. You're testing that *relationships hold over time*.

Consider this simple FynX expression:

```python
total = (price | quantity) >> (lambda p, q: p * q)
```

Of course you'll check that `total.value` has the right output—that's still important. But what distinguishes testing reactive systems is that you're verifying something deeper: the relationship "total equals price times quantity" remains true as values change, as subscriptions fire, as the dependency graph updates.

Compare these two tests:

```python
# Testing a pure function (one output check)
def test_multiply():
    assert multiply(3, 4) == 12

# Testing a reactive relationship (outputs verify the relationship persists)
def test_total_maintains_price_times_quantity_relationship():
    price.set(10)
    quantity.set(3)
    assert total.value == 30  # First output check
    
    price.set(5)
    assert total.value == 15  # Relationship still holds
    
    quantity.set(10)
    assert total.value == 50  # Still maintained
```

You're still checking outputs in both cases—that's how you verify correctness. But in the reactive test, each output verification confirms that the dependency graph is wired correctly, updates propagate as expected, and the relationship maintains its invariant over time.

This shift in perspective—from "does this return the right value once?" to "does this relationship hold continuously?"—transforms how we structure tests. Instead of asking "does this function return 42?", we ask "does this observable maintain its invariants as the system evolves?"

The difference seems subtle but proves profound in practice.

## One Test, One Question

Every test should answer exactly one question about your system. Not two questions, not "a few related things," but one specific question. When that test fails, you should immediately know what broke without reading the implementation.

Here's what that looks like:

```python
def test_observable_notifies_subscriber_on_value_change():
    """Subscribers receive notifications when observable values change"""
    obs = observable(10)
    received = []
    obs.subscribe(lambda val: received.append(val))
    
    obs.set(20)
    
    assert received == [20]
```

This test asks one question: do subscribers get notified when values change? It doesn't also verify subscription timing, multiple subscribers, or unsubscription. Those are different questions deserving different tests.

Compare this to the kind of test we've all written (and later regretted):

```python
def test_observable_subscription_system():
    """Test the subscription system"""
    obs = observable(10)
    received1, received2 = [], []
    
    # Test basic subscription
    sub1 = obs.subscribe(lambda val: received1.append(val))
    obs.set(20)
    assert received1 == [20]
    
    # Test multiple subscribers
    sub2 = obs.subscribe(lambda val: received2.append(val))
    obs.set(30)
    assert received1 == [20, 30]
    assert received2 == [30]
    
    # Test unsubscription
    sub1.unsubscribe()
    obs.set(40)
    assert received1 == [20, 30]
    assert received2 == [30, 40]
```

When this test fails at the unsubscription assertion, you can't know whether the earlier assertions would pass independently. The behaviors are tangled together. Six months later, when you're hunting down a subscription bug at 2 AM, this test will actively make your life harder.

### The Three-Phase Structure

Structure every test in three clear phases:

**Arrange:** Create the minimal setup needed for this specific behavior. If a line isn't necessary for this particular question, delete it.

**Act:** Execute the behavior being tested. Usually one operation, sometimes a short sequence of closely related operations.

**Assert:** Verify one expectation. If you find yourself checking multiple properties, pause and ask: am I really testing one behavior, or several?

```python
def test_derived_observable_recalculates_when_source_changes():
    # Arrange: Create source and derived observable
    source = observable(5)
    doubled = source >> (lambda x: x * 2)
    
    # Act: Change the source value
    source.set(10)
    
    # Assert: Derived value updates correctly
    assert doubled.value == 20
```

The three phases make the test's intent transparent. Six months from now, when you've forgotten everything about this code, you'll be able to read this test and immediately understand both what it's checking and why that matters.

## Names That Tell Stories

Test names should read like entries in a specification document. Use present tense. Be specific about the behavior, not just the component.

Some transformations:

| Before | After | Why It Matters |
|--------|-------|----------------|
| `test_store` | `test_store_observables_update_via_attribute_assignment` | Tells you the mechanism |
| `test_combination` | `test_pipe_operator_combines_observables_into_tuple` | Specifies the transformation |
| `test_filtering` | `test_ampersand_operator_gates_emission_by_predicate` | Explains the behavior |
| `test_computed` | `test_computed_property_recalculates_on_dependency_change` | Identifies the trigger |

Good names document your system. In fact, you should be able to understand what FynX does just by reading test names:

```python
def test_store_provides_class_level_access_to_observables():
    """Store observables can be accessed and modified through class attributes"""
    
def test_reactive_decorator_triggers_on_observable_change():
    """Functions decorated with @reactive execute when their observable updates"""
    
def test_functor_composition_preserves_transformation_order():
    """Chaining transformations with >> maintains left-to-right evaluation"""
```

Each name tells a small story. Together, they describe a system. The docstring can elaborate, but the name alone should convey the essential behavior.

## Eliminating Repetition

Repeated setup code obscures what makes each test unique. When every test starts with fifteen lines of boilerplate, the actual behavior being tested gets lost in the noise.

### Factory Functions

Extract common patterns into factory functions:

```python
def create_diamond_dependency():
    """Creates diamond pattern: source → (a, b) → combined"""
    source = observable(10)
    path_a = source >> (lambda x: x + 5)
    path_b = source >> (lambda x: x * 2)
    combined = (path_a | path_b) >> (lambda a, b: a + b)
    
    return source, path_a, path_b, combined
```

Now tests become clear about what they're actually testing:

```python
def test_diamond_dependency_updates_from_single_source():
    source, path_a, path_b, combined = create_diamond_dependency()
    
    source.set(20)
    
    assert combined.value == 65  # (20 + 5) + (20 * 2)
```

The factory name describes the structure. The test focuses on the behavior within that structure.

### Pytest Fixtures

Use fixtures for setup that needs cleanup or fresh instances:

```python
@pytest.fixture
def subscription_tracker():
    """Provides a helper for tracking subscription notifications"""
    class Tracker:
        def __init__(self):
            self.values = []
            
        def record(self, value):
            self.values.append(value)
    
    return Tracker()
```

Then inject it cleanly:

```python
def test_multiple_subscribers_receive_independent_notifications(subscription_tracker):
    obs = observable(0)
    obs.subscribe(subscription_tracker.record)
    
    obs.set(5)
    
    assert subscription_tracker.values == [5]
```

Use fixtures when:
- Setup appears in three or more tests
- You need guaranteed cleanup
- Complex initialization obscures test intent
- You're managing resources that should be fresh each time

## The Non-Negotiable: Test Isolation

Every test must be completely independent. A failure in Test A should never cause Test B to fail. This isn't a suggestion or a guideline—it's a requirement. Without isolation, your test suite becomes unreliable, and unreliable tests are worse than no tests.

Consider what happens without proper isolation. Test A modifies a shared store. Test B assumes a clean state. Test A fails and leaves the store corrupted. Now Test B fails too—not because its behavior is broken, but because of Test A's failure. You have two failing tests, one actual bug, and no clear way to distinguish cause from effect.

FynX's reactive nature makes this even more critical. Observables maintain subscriptions, stores hold state, reactive chains keep references to their sources. If tests share these structures, updates propagate in unexpected ways.

### Achieving Isolation

Create fresh instances for each test:

```python
def test_observable_subscription_receives_updates():
    obs = observable(0)  # Fresh for this test
    received = []
    obs.subscribe(lambda val: received.append(val))
    
    obs.set(5)
    
    assert received == [5]

def test_observable_subscription_can_unsubscribe():
    obs = observable(0)  # Completely independent
    received = []
    sub = obs.subscribe(lambda val: received.append(val))
    
    sub.unsubscribe()
    obs.set(5)
    
    assert received == []
```

These tests could run in any order—even in parallel—without interfering with each other.

Contrast this with shared state:

```python
# ANTI-PATTERN: Don't do this
shared_store = create_simple_store()

def test_store_name_updates():
    shared_store.name = "Alice"
    assert shared_store.name.value == "Alice"

def test_store_count_updates():
    # Mysteriously fails if test_store_name_updates ran first
    shared_store.count = 5
    assert shared_store.count.value == 5
```

### Verifying Isolation

Run your tests in random order:

```bash
pytest --random-order
```

If anything fails in random order but passes normally, you have an isolation problem. Fix it immediately. Unreliable tests destroy trust in your test suite, and once that trust is gone, tests become obstacles rather than allies.

## Strategic Mocking

Mocking replaces complex or external dependencies with controlled substitutes. This isolates what you're testing and prevents cascading failures when dependencies break.

### When to Mock

Mock things outside your test's scope:

- **External resources:** File systems, databases, network calls
- **Expensive operations:** Complex computations that slow tests
- **Non-deterministic behavior:** Random generators, timestamps, variable response times
- **Unready dependencies:** Code that's still being developed

### Mocking in Practice

Suppose you're testing a reactive pipeline that fetches data, transforms it, and caches results:

```python
from unittest.mock import Mock

def test_reactive_pipeline_caches_transformation_results():
    mock_fetcher = Mock()
    mock_fetcher.fetch_data.return_value = {"value": 100}
    
    source = observable(None)
    transformed = source >> (lambda _: mock_fetcher.fetch_data()) \
                         >> (lambda data: data["value"] * 2)
    
    source.set("trigger")
    
    assert transformed.value == 200
    mock_fetcher.fetch_data.assert_called_once()
```

The test verifies the reactive wiring without depending on the actual fetcher's implementation.

### Mocking Principles

**Mock at boundaries.** Don't mock FynX's core behavior—that's what you're testing. Mock external dependencies that FynX interacts with.

**Verify interactions.** Use `assert_called_once()` and similar methods to confirm your reactive code calls dependencies correctly.

**Keep mocks simple.** If you're mocking more than two dependencies in a single test, your code probably has too many responsibilities.

**Name mocks clearly.** Use `mock_database` or `mock_api_client`, not just `mock`.

## Testing Multiple Scenarios Efficiently

Parameterized tests let you verify the same behavior across different inputs without duplicating logic:

```python
@pytest.mark.parametrize("initial,update,expected", [
    (0, 5, 5),
    (10, -3, 7),
    (100, 100, 200),
])
def test_observable_updates_to_new_value(initial, update, expected):
    obs = observable(initial)
    
    obs.set(obs.value + update)
    
    assert obs.value == expected
```

Each tuple runs as a separate test. If the middle case fails, the others still run, helping you identify patterns.

### Adding Clarity with IDs

Descriptive IDs make test output readable:

```python
@pytest.mark.parametrize("operator,values,expected", [
    (">>", (5, lambda x: x * 2), 10),
    ("|", (observable(5), observable(3)), (5, 3)),
    ("&", (observable(5), lambda x: x > 0), 5),
], ids=["transform", "combine", "filter"])
def test_operator_behavior(operator, values, expected):
    # Test implementation
    pass
```

Now failures show `test_operator_behavior[transform]` instead of `test_operator_behavior[0]`.

### The Readability Tradeoff

Parameterized tests are concise and prevent duplication, but they can be harder to debug when failures occur with unfamiliar input combinations. Consider the failing test:

```python
# Failure: test_value_categorization_handles_range[1000000-positive] failed
# Where did 1000000 come from? What was the expected behavior?
```

For complex logic, prefer explicit separate tests that make the intent and expected behavior crystal clear. Use parameterization for simple, mechanical variations where the relationship between input and output is obvious.

### When to Parameterize

Parameterize when:
- Testing the same behavior with different inputs
- Verifying boundary conditions systematically
- Testing error conditions with various invalid inputs

Don't parameterize when:
- Test logic differs significantly between cases
- You're testing fundamentally different behaviors

## The Edges Matter

Edge cases and error conditions hide bugs that only surface in production. Test them explicitly.

### Edge Cases in Reactive Systems

**Empty inputs and initial states:**

```python
def test_derived_observable_handles_initial_none_value():
    source = observable(None)
    transformed = source >> (lambda x: x or "default")
    
    assert transformed.value == "default"
```

**Rapid successive updates:**

```python
def test_observable_handles_rapid_successive_updates():
    obs = observable(0)
    received = []
    obs.subscribe(lambda val: received.append(val))
    
    for i in range(100):
        obs.set(i)
    
    assert received[-1] == 99
    assert len(received) == 100
```

**Circular dependencies:**

```python
def test_circular_dependency_raises_error():
    a = observable(1)
    
    with pytest.raises(ValueError, match="circular"):
        # Attempting to create a circular dependency
        # Your implementation should detect and prevent this
        # For example: a = a.then(lambda x: x + 1)  # would be circular
```

### Error Conditions

**Subscription errors:**

```python
def test_subscription_with_failing_callback_doesnt_break_observable():
    obs = observable(0)
    successful_calls = []
    
    def failing_callback(val):
        raise RuntimeError("Subscriber error")
    
    def working_callback(val):
        successful_calls.append(val)
    
    obs.subscribe(failing_callback)
    obs.subscribe(working_callback)
    
    obs.set(5)
    
    assert successful_calls == [5]
```

### Boundary Conditions

Test extremes systematically:

```python
@pytest.mark.parametrize("value,expected_category", [
    (0, "zero"),
    (1, "positive"),
    (-1, "negative"),
    (1000000, "positive"),
    (-1000000, "negative"),
])
def test_value_categorization_handles_range(value, expected_category):
    obs = observable(value)
    category = obs >> (lambda x: "zero" if x == 0 else 
                                 "positive" if x > 0 else "negative")
    
    assert category.value == expected_category
```

## Organization as Your Test Suite Grows

As tests multiply, organization becomes critical. Pytest markers and directory structure help maintain clarity.

### Using Markers

Define markers in `pytest.ini`:

```ini
[pytest]
markers =
    unit: Unit tests for individual components
    integration: Tests for reactive system interactions
    store: Tests related to Store functionality
    observable: Tests for Observable behavior
    operators: Tests for reactive operators
```

Apply them judiciously:

```python
@pytest.mark.unit
@pytest.mark.observable
def test_observable_notifies_subscriber_on_change():
    obs = observable(0)
    received = []
    obs.subscribe(lambda val: received.append(val))
    
    obs.set(5)
    
    assert received == [5]
```

Run targeted subsets:

```bash
# Run only unit tests
pytest -m unit

# Run observable tests, excluding performance tests
pytest -m "observable and not performance"
```

### Directory Structure

Mirror your source structure:

```
fynx/
├── fynx/
│   ├── observable/*
│   ├── store.py
│   └── operators.py
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── test_factories.py    # Factory functions
    ├── unit/
    │   ├── test_observable.py
    │   ├── test_store.py
    │   └── test_operators.py
    └── integration/
        ├── test_store_with_computed.py
        └── test_reactive_chains.py
```

This makes tests easy to find and navigate. If you must write N test files for a given file in source, you should create a folder with the name of the file you're testing.

For example, if you're testing observable/base.py you should create a folder at `tests/unit/observable/base.py/<test files>`

## Integration Testing

Integration tests verify that multiple components work together correctly. They follow the same principles as unit tests but with broader scope.

### Keep Scope Minimal

Test two or three components together, not your entire application:

```python
@pytest.mark.integration
def test_store_computed_properties_react_to_observable_changes():
    """Verifies that Store computed properties update when observables change"""
    class TemperatureMonitor(Store):
        celsius = observable(0.0)
        fahrenheit = celsius.then(lambda c: c * 9/5 + 32)
    
    TemperatureMonitor.celsius = 100.0
    
    assert TemperatureMonitor.fahrenheit.value == 212.0
```

### Test Specific Interaction Patterns

```python
@pytest.mark.integration
def test_diamond_dependency_resolves_correctly():
    """Verifies that diamond-shaped dependency graphs compute correctly"""
    source = observable(10)
    
    path_a = source >> (lambda x: x + 5)
    path_b = source >> (lambda x: x * 2)
    combined = (path_a | path_b) >> (lambda a, b: a + b)
    
    assert combined.value == 35  # (10 + 5) + (10 * 2)
    
    source.set(20)
    assert combined.value == 65  # (20 + 5) + (20 * 2)
```

### When Integration Tests Are Overkill

Don't write integration tests for scenarios that unit tests already cover comprehensively:

```python
# UNNECESSARY: Integration test for simple composition
def test_observable_and_transform_compose_via_pipe_operator():
    """Integration test - probably unnecessary"""
    obs = observable(5)
    doubled = obs >> (lambda x: x * 2)
    tripled = doubled >> (lambda x: x * 3)

    assert tripled.value == 30
    obs.set(10)
    assert tripled.value == 60

# BETTER: Unit tests for each behavior
def test_pipe_operator_applies_transformation():
    """Unit test for pipe operator"""
    obs = observable(5)
    result = obs >> (lambda x: x * 2)

    assert result.value == 10

def test_observable_notifies_derived_on_change():
    """Unit test for notification propagation"""
    obs = observable(5)
    derived = obs >> (lambda x: x * 2)

    obs.set(10)
    assert derived.value == 20
```

Integration tests shine when verifying *interactions* between components, not just their individual behaviors. If you can test each piece in isolation and be confident they compose correctly, skip the integration test.

## Testing Cleanup and Disposal

FynX provides memory testing utilities to verify cleanup:

```python
from tests.utils.memory_utils import assert_no_object_leak, MemoryTracker
```

Reactive systems create complex dependency graphs that can lead to memory leaks if not properly cleaned up. Subscriptions, reactive chains, and observers hold references to each other—test that these references are released when components are disposed.

### Testing Subscription Cleanup

Verify that unsubscribing actually removes references:

```python
from tests.utils.memory_utils import assert_cleaned_up

def test_subscription_cleanup_removes_references():
    """Ensure unsubscribing prevents memory leaks"""
    obs = observable(0)
    received = []

    def callback(val):
        received.append(val)

    sub = obs.subscribe(callback)

    # Verify subscription is active
    obs.set(5)
    assert received == [5]

    # Unsubscribe and verify cleanup
    sub.unsubscribe()
    assert_cleaned_up(received, "Subscriber should be cleaned up after unsubscribe")
```

### Testing Observable Disposal

Test that observables can be disposed of completely:

```python
def test_observable_disposal_clears_all_subscriptions():
    """Disposed observables should release all subscriber references"""
    obs = observable(0)
    subscriber_refs = []

    # Create multiple subscribers
    for i in range(3):
        received = []
        subscriber_refs.append(weakref.ref(received))

        def make_callback(rec):
            return lambda val: rec.append(val)

        obs.subscribe(make_callback(received))

    # Dispose of the observable
    obs.dispose()  # Assuming FynX has a dispose method

    # Delete local references and force GC
    del subscriber_refs
    gc.collect()

    # All subscriber references should be cleaned up
    # (This assumes your dispose method clears subscriber lists)
```

### Testing Reactive Chain Cleanup

Complex reactive chains can create circular references. Test that chains are properly dismantled:

```python
def test_reactive_chain_cleanup_breaks_cycles():
    """Ensure derived observables don't create circular references"""
    source = observable(10)
    chain_refs = []

    # Build a reactive chain
    derived1 = source >> (lambda x: x * 2)
    derived2 = derived1 >> (lambda x: x + 5)
    final = (derived1 | derived2) >> (lambda a, b: a + b)

    # Keep weak references to all chain elements
    chain_refs = [
        weakref.ref(source),
        weakref.ref(derived1),
        weakref.ref(derived2),
        weakref.ref(final)
    ]

    # Delete the chain and force cleanup
    del source, derived1, derived2, final
    gc.collect()

    # All references should be cleaned up (no cycles)
    assert all(ref() is None for ref in chain_refs)
```

### Testing Store Cleanup

Stores often hold complex state. Test that store disposal cleans up all observables:

```python
@pytest.mark.integration
def test_store_disposal_cleans_up_all_observables():
    """Store disposal should clean up all contained observables"""

    class TestStore(Store):
        counter = observable(0)
        doubled = counter.then(lambda c: c * 2)

        def dispose(self):
            # Custom disposal logic if needed
            super().dispose()

    store = TestStore()
    store_ref = weakref.ref(store)

    # Use the store
    store.counter.set(5)
    assert store.doubled.value == 10

    # Dispose and clean up
    store.dispose()
    del store
    gc.collect()

    assert store_ref() is None
```

### Testing Cleanup After Errors

Ensure cleanup happens even when errors occur:

```python
def test_subscription_cleanup_occurs_even_after_errors():
    """Cleanup should work even when subscribers throw exceptions"""
    obs = observable(0)
    error_subscriber = Mock(side_effect=RuntimeError("Subscriber failed"))

    # Subscribe with error-throwing callback
    sub = obs.subscribe(error_subscriber)

    # This should trigger the error but not break cleanup
    with pytest.raises(RuntimeError):
        obs.set(5)

    # Cleanup should still work
    sub.unsubscribe()

    # Verify subscription is actually removed
    error_subscriber.reset_mock()
    obs.set(10)
    error_subscriber.assert_not_called()
```

### Testing for Memory Leaks in Long-Running Systems

Test that memory growth is sublinear (doesn't grow proportionally with operation count):

```python
def test_repeated_operations_show_bounded_memory_growth():
    """Memory growth should be sublinear with operation count"""
    import gc
    import sys

    obs = observable(0)
    measurements = []

    # Track total memory usage across all objects
    def get_total_size():
        gc.collect()
        return sum(sys.getsizeof(obj) for obj in gc.get_objects()
                  if hasattr(obj, '__dict__'))  # Focus on objects with attributes

    for batch in [100, 200, 400]:
        size_before = get_total_size()

        for i in range(batch):
            temp = obs >> (lambda x: x + i)
            obs.set(i)
            del temp

        size_after = get_total_size()
        growth = size_after - size_before
        measurements.append((batch, growth))

    # Growth rate should decrease (or stay constant)
    growth_rates = [m[1] / m[0] for m in measurements]

    # Each doubling of operations shouldn't cause proportional memory growth
    assert growth_rates[1] < growth_rates[0] * 1.5
    assert growth_rates[2] < growth_rates[0] * 1.5
```

### Testing Object Accumulation

Count actual observable instances to detect leaks more reliably than byte measurements:

```python
from tests.utils.memory_utils import assert_no_object_leak

def test_reactive_chain_disposal_prevents_object_accumulation():
    """Verify temporary chains don't accumulate observable instances"""
    obs = observable(0)

    def create_and_destroy_chains():
        # Create and destroy many reactive chains
        for i in range(100):
            chain = obs >> (lambda x: x + 1) >> (lambda x: x * 2)
            obs.set(i)
            del chain

    assert_no_object_leak(create_and_destroy_chains, 'Observable',
                         tolerance=10,
                         description="Reactive chains should not accumulate observables")
```

### Cleanup Testing Patterns

**Use the memory utilities** for consistent cleanup testing:

```python
from tests.utils.memory_utils import assert_cleaned_up, MemoryTracker, with_memory_tracking

# Simple cleanup verification
def test_component_cleanup():
    component = create_component()
    # ... use component ...
    component.dispose()
    assert_cleaned_up(component)

# Track memory during complex operations
def test_operation_memory_usage():
    with MemoryTracker('Observable') as tracker:
        # ... perform operations that might leak ...
        pass

    tracker.assert_no_growth('Observable')

# Decorate functions for automatic memory leak detection
@with_memory_tracking('Observable')
def my_reactive_operation():
    # ... operation that should not leak observables ...
    pass
```

**FynX provides pytest fixtures for convenient memory testing:**

```python
# These fixtures are available in all tests (configured in tests/conftest.py)
```

Use them in tests:

```python
def test_observable_operations_dont_leak(no_leaks):
    def operation():
        obs = observable(0)
        for i in range(100):
            temp = obs >> (lambda x: x + i)
            del temp

    no_leaks(operation, 'Observable')

def test_store_disposal_cleans_up_observables(memory_tracker):
    with memory_tracker() as tracker:
        store = MyStore()
        store.value.set(100)
        store.dispose()
        del store

    # Verify no growth in any tracked types
    growth = tracker.object_growth
    assert 'Observable' not in growth or abs(growth['Observable']) < 3
    assert 'Store' not in growth or abs(growth['Store']) < 2
```

**Test disposal methods exist and work:**

```python
def test_all_disposable_objects_have_dispose_method():
    """Ensure objects that need cleanup have dispose methods"""
    obs = observable(5)

    # If your observable needs disposal, it should have a dispose method
    assert hasattr(obs, 'dispose'), "Observable should have dispose method"

    # The dispose method should be callable
    obs.dispose()  # Should not raise

    # Disposing twice should be safe (idempotent)
    obs.dispose()  # Should not raise
```

**Test cleanup in fixture teardown:**

```python
@pytest.fixture
def reactive_store():
    store = TestStore()
    yield store
    # Ensure cleanup happens even if test fails
    store.dispose()
```

## Common Pitfalls

### Testing Implementation Instead of Behavior

**Anti-pattern:**
```python
def test_observable_stores_value_in_underscore_value_attribute():
    obs = observable(5)
    assert obs._value == 5  # Testing internal detail
```

**Better:**
```python
def test_observable_returns_current_value():
    obs = observable(5)
    assert obs.value == 5  # Testing public interface
```

### Overly Broad Tests

Don't test six behaviors in one 120-line test. Split into focused tests.

### Ignoring Error Conditions

Test error paths explicitly. Bugs hide in edge cases.

### Poor Names

Use descriptive names that explain the behavior being verified.

### Shared Mutable State

Create fresh instances for each test. Always.

## About Code Coverage

FynX maintains strict coverage requirements: **CI fails if coverage drops below 80%**. This ensures core reactive behaviors are thoroughly tested.

But remember: coverage is a tool for finding untested code, not a goal in itself. You can have 100% coverage with terrible tests that verify nothing. Focus on writing meaningful tests; coverage follows naturally.

### Checking Coverage

Before pushing:

```bash
# Run tests with coverage report
pytest --cov=fynx --cov-report=term-missing

# Generate detailed HTML report
pytest --cov=fynx --cov-report=html
open htmlcov/index.html
```

The `term-missing` flag shows which lines aren't covered.

### What Coverage Means (and Doesn't)

**Coverage indicates:**
- Which lines execute during tests
- Whether branches are exercised
- Untested code paths

**Coverage doesn't indicate:**
- Whether assertions are meaningful
- If tests verify correct behavior
- Test quality or maintainability

A test that calls every line but asserts nothing gives perfect coverage and zero confidence.

### Addressing Gaps

When coverage shows untested code:

1. Identify what behavior isn't tested
2. Write focused tests that exercise the untested path
3. Consider whether the code is testable (difficult testing often signals design issues)
4. Don't test just for coverage—verify meaningful behavior

Example:

```python
# Coverage shows this branch is untested
def get_value_or_default(obs, default=None):
    if obs.value is None:
        return default  # Untested
    return obs.value

# Add focused test
def test_get_value_or_default_returns_default_when_observable_is_none():
    obs = observable(None)
    
    result = get_value_or_default(obs, default="fallback")
    
    assert result == "fallback"
```

## A Final Thought

Testing reactive systems requires discipline. The investment pays off because well-tested FynX applications remain maintainable as complexity grows. Tests document behavior, prevent regressions, and guide refactoring.

Remember:

- **Write focused tests.** One behavior, one test.
- **Ensure isolation.** Tests must be completely independent.
- **Name precisely.** Specify the behavior being verified.
- **Test edges.** Bugs hide in boundaries and error conditions.
- **Organize systematically.** Use markers, fixtures, clear structure.
- **Maintain coverage.** But focus on meaningful tests over metrics.

Your tests are living documentation of how FynX components should behave. Treat them with the same care you give production code, and they'll serve you well as your reactive systems grow in sophistication.

The goal isn't just to verify that your code works—it's to create a feedback loop that makes your code better. Tests that clarify, tests that guide, tests that make the next change easier. That's the practice worth cultivating.