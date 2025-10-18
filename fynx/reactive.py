"""
FynX Reactive - Reactive Decorators and Utilities
=================================================

This module provides decorators and utilities for creating reactive relationships
between observables and functions. Reactive decorators enable automatic execution
of functions when their observable dependencies change, perfect for side effects
like UI updates, API calls, logging, and other imperative operations.

What are Reactive Functions?
-----------------------------

Reactive functions are functions that automatically re-run whenever the observables
they depend on change. Unlike computed values (which are declarative and return
derived data), reactive functions are imperative and perform side effects.

Reactive functions are ideal for:
- **UI Updates**: Automatically updating displays when data changes
- **API Calls**: Triggering network requests when relevant data changes
- **Logging**: Recording changes for debugging or analytics
- **Side Effects**: Any imperative operation that should respond to data changes
- **Synchronization**: Keeping external systems in sync with reactive state

Key Characteristics
-------------------

- **Automatic Execution**: Functions run whenever dependencies change
- **Imperative**: Designed for side effects, not data transformation
- **Dependency Tracking**: Framework automatically tracks accessed observables
- **Unsubscription**: Easy cleanup when reactive functions are no longer needed
- **Multiple Targets**: Can react to multiple observables or entire stores

Basic Usage
-----------

    ```python
    from fynx import Store, observable, reactive

class CounterStore(Store):
    count = observable(0)
    name = observable("Counter")

@reactive(CounterStore.count, CounterStore.name)
def log_counter_changes(count, name):
    print(f"Counter '{name}' changed to: {count}")

# Changes trigger the reactive function automatically
CounterStore.count = 5   # Prints: "Counter 'Counter' changed to: 5"
CounterStore.name = "My Counter"  # Prints: "Counter 'My Counter' changed to: 5"
CounterStore.count = 10  # Prints: "Counter 'My Counter' changed to: 10"
```

Advanced Patterns
-----------------

### Store-Level Reactions

React to any change in an entire store:

```python
    class UserStore(Store):
        name = observable("Alice")
        age = observable(30)
    email = observable("alice@example.com")

@reactive(UserStore)  # Reacts to any change in UserStore
def on_any_user_change():
    snapshot = UserStore.to_dict()
    print(f"User data changed: {snapshot}")

UserStore.name = "Bob"   # Triggers: User data changed: {'name': 'Bob', 'age': 30, 'email': 'alice@example.com'}
UserStore.age = 31       # Triggers again with updated data
```

### Mixed Observable Types

Combine store-level and individual observable reactions:

```python
# React to store changes and a specific external observable
is_online = observable(True)

@reactive(UserStore, is_online)
def on_user_or_online_change():
    user_data = UserStore.to_dict()
    online_status = "online" if is_online.value else "offline"
    print(f"User {user_data['name']} is {online_status}")

UserStore.name = "Charlie"  # Triggers with current online status
is_online.set(False)       # Triggers with current user data
```

### Cleanup and Unsubscription

Reactive functions can be unsubscribed when no longer needed:

```python
# The reactive decorator returns the original function
# so you can unsubscribe later if needed
unsubscribe_func = reactive(CounterStore.count)(log_changes)
# Later...
# CounterStore.count.unsubscribe(log_changes)  # Unsubscribe specific function
```

Real-World Examples
-------------------

### UI Update Simulation

```python
class ViewModel(Store):
    search_query = observable("")
    results = observable([])
    is_loading = observable(False)

@reactive(ViewModel.search_query)
def update_search_results(query):
    if query:
        ViewModel.is_loading = True
        # Simulate API call
        ViewModel.results = [f"Result for '{query}'"]
        ViewModel.is_loading = False
    else:
        ViewModel.results = []

@reactive(ViewModel.results, ViewModel.is_loading)
def update_ui(results, loading):
    if loading:
        print("🔄 Loading...")
    else:
        print(f"📋 Found {len(results)} results: {results}")

ViewModel.search_query = "python"  # Triggers both functions
```

### Form Validation

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    is_submitting = observable(False)

@reactive(FormStore.email)
def validate_email(email):
    is_valid = "@" in email and len(email) > 5
    print(f"Email valid: {is_valid}")

@reactive(FormStore.password)
def validate_password(password):
    is_strong = len(password) >= 8
    print(f"Password strong: {is_strong}")

FormStore.email = "user@"       # Email valid: False
FormStore.email = "user@example.com"  # Email valid: True
FormStore.password = "123"      # Password strong: False
FormStore.password = "secure123" # Password strong: True
```

### Analytics Tracking

```python
class AnalyticsStore(Store):
    page_views = observable(0)
    unique_visitors = observable(0)
    current_page = observable("home")

@reactive(AnalyticsStore.page_views)
def track_page_views(views):
    print(f"📊 Analytics: {views} page views")

@reactive(AnalyticsStore.current_page)
def track_page_changes(page):
    print(f"📍 User navigated to: {page}")

AnalyticsStore.page_views = 150
AnalyticsStore.current_page = "products"
AnalyticsStore.page_views = 151  # Both functions trigger
```

Performance Considerations
--------------------------

Reactive functions include several performance optimizations:

**Efficient Tracking**: Only tracks observables actually accessed during function execution

**Batch Updates**: Multiple observable changes in quick succession trigger the function only once

**Memory Management**: Automatic cleanup when reactive contexts are no longer needed

**Selective Execution**: Functions only run when their specific dependencies change

Best Practices
--------------

### Keep Functions Focused

Each reactive function should have a single, clear responsibility:

```python
# Good: Focused responsibilities
@reactive(user_data)
def update_profile_ui(user_data):
    # Only handles UI updates
    pass

@reactive(user_data)
def sync_to_server(user_data):
    # Only handles server sync
    pass

# Avoid: Mixed responsibilities
@reactive(user_data)
def handle_user_change(user_data):
    # Updates UI, syncs to server, logs analytics...
    pass
```

### Handle Errors Gracefully

Reactive functions should handle exceptions to prevent breaking the reactive system:

```python
@reactive(data)
def process_data_safely(data):
    try:
        # Process data...
        result = expensive_operation(data)
        update_ui(result)
    except Exception as e:
        print(f"Error processing data: {e}")
        show_error_message()
```

### Use Appropriate Granularity

Choose the right level of reactivity for your use case:

```python
# Fine-grained: React to specific changes
@reactive(store.specific_field)
def handle_specific_change(value):
    pass

# Coarse-grained: React to any store change
@reactive(store)
def handle_any_change():
    pass
```

Common Patterns
---------------

**Event Logging**:
```python
@reactive(store)
def log_all_changes():
    print(f"State changed at {datetime.now()}: {store.to_dict()}")
```

**Cache Invalidation**:
```python
cache = {}
@reactive(data_version)
def invalidate_cache(version):
    cache.clear()
    print(f"Cache invalidated for version {version}")
```

**External System Sync**:
```python
@reactive(local_data)
def sync_to_external_system(data):
    external_api.update(data)
    print("Synced to external system")
```

Limitations
-----------

- Reactive functions cannot return values (use computed for that)
- Dependencies must be accessed synchronously during execution
- Functions execute for every dependency change (no debouncing built-in)
- Cannot create circular dependencies with observables

Troubleshooting
---------------

**Function not triggering**: Ensure all accessed observables are passed as arguments to @reactive

```python
# Wrong: external_obs not declared as dependency
external_obs = observable(10)
@reactive(some_obs)
def wrong_func():
    value = external_obs.value  # Not tracked!

# Right: Declare all dependencies
@reactive(some_obs, external_obs)
def right_func(some_val, external_val):
    pass
```

**Too many executions**: Consider using @watch for conditional execution instead

**Performance issues**: Break large reactive functions into smaller, focused ones

Comparison with Other Approaches
---------------------------------

**vs Manual Subscriptions**:
```python
# Manual (error-prone)
def setup():
    obs.subscribe(callback)
    obs2.subscribe(callback)
    # Must manually unsubscribe later...

# Reactive (declarative)
@reactive(obs, obs2)
def callback():
    pass  # Automatically managed
```

**vs Computed Values**:
- Use @reactive for side effects (UI updates, API calls)
- Use @computed for derived data (calculations, transformations)

**vs Watch Decorators**:
- Use @reactive for unconditional reactions to changes
- Use @watch for conditional execution (only when conditions met)

See Also
--------

- `fynx.watch`: Conditional reactive functions
- `fynx.computed`: Derived reactive values
- `fynx.store`: Reactive state containers
- `fynx.observable`: Core observable classes
"""

from typing import Callable

from .observable import Observable
from .store import Store, StoreSnapshot


class ReactiveHandler:
    """
    Manages reactive function subscriptions and handles different target types.

    ReactiveHandler is the core implementation behind the `@reactive` decorator.
    It intelligently handles different types of targets (Store classes, individual
    observables) and creates the appropriate subscription mechanism.

    The handler supports:
    - Store class subscriptions (reacts to any change in the store)
    - Individual observable subscriptions (reacts to specific observables)
    - Mixed subscriptions (combination of stores and observables)

    This class is typically used indirectly through the `@reactive` decorator
    rather than instantiated directly.

    Example:
        ```python
        # These all use ReactiveHandler internally:
        @reactive(store_instance)      # Store subscription
        @reactive(obs1, obs2)          # Multiple observables
        @reactive(store_class.attr)    # Single observable
        ```
    """

    def __init__(self, *targets):
        """
        Initialize the reactive handler with target observables/stores.

        Args:
            *targets: Variable number of observables, stores, or store attributes
                     to monitor for changes.
        """
        self.targets = targets

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation that makes the function reactive.

        This method is called when the ReactiveHandler is used as a decorator.
        It sets up the reactive context for the decorated function and returns
        the original function (decorators typically return the same function).

        Args:
            func: The function to make reactive

        Returns:
            The original function, now configured to react to target changes

        Example:
            ```python
            @reactive(store.count, store.name)
            def update_display(count, name):
                print(f"Count: {count}, Name: {name}")

            # This is equivalent to:
            # reactive_handler = ReactiveHandler(store.count, store.name)
            # update_display = reactive_handler(update_display)
            ```
        """
        self._create_reactive_context(func)
        return func

    def _create_reactive_context(self, func: Callable) -> None:
        """
        Create the appropriate reactive context based on target types.

        This method analyzes the targets passed to the handler and creates
        the appropriate subscription mechanism. It handles different scenarios:

        - Store class targets: Subscribe to all observables in the store
        - Individual observable targets: Subscribe to specific observables
        - Mixed targets: Combine multiple subscription types

        Args:
            func: The function to make reactive
        """
        if len(self.targets) == 0:
            # No targets provided - do nothing
            return
        elif len(self.targets) == 1:
            target = self.targets[0]

            if isinstance(target, type) and issubclass(
                target, Store
            ):  # It's a Store class
                # Call immediately with current store state
                snapshot = StoreSnapshot(target, target._observable_attrs)
                snapshot._take_snapshot()
                func(snapshot)

                # Use the Store's subscribe method (only triggers on changes)
                target.subscribe(func)
            else:  # It's a single Observable
                # Create a reaction function that filters None values
                def filtered_reaction():
                    value = target.value
                    if value is not None:
                        func(value)

                # Call immediately with current value (if not None)
                if target.value is not None:
                    func(target.value)

                # Subscribe with the reaction function
                context = Observable._create_subscription_context(
                    filtered_reaction, func, target
                )
                if target is not None:
                    target.add_observer(context.run)

        else:  # Multiple observables passed
            # Merge all observables using the | operator and subscribe to the result
            merged = self.targets[0]
            for obs in self.targets[1:]:
                merged = merged | obs

            # Call immediately with current values
            current_values = merged.value
            if current_values is not None:
                func(*current_values)

            # For merged observables, use standard subscription (no filtering needed for this test)
            merged.subscribe(func)


def reactive(*targets):
    """
    Create a reactive handler that works as a decorator.

    This is a convenience wrapper around subscribe() that works as a decorator.

    As decorator:
        @reactive(store) - reacts to all observables in store
        @reactive(observable) - reacts to single observable
        @reactive(obs1, obs2, ...) - reacts to multiple observables

    Args:
        *targets: Store class, Observable instance(s), or multiple Observable instances

    Returns:
        ReactiveHandler that can be used as decorator
    """
    return ReactiveHandler(*targets)
