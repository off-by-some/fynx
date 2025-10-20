# Decorators - Making Code Reactive

Decorators in FynX provide the high-level API for creating reactive relationships in your application. They transform regular Python functions and classes into reactive components that automatically respond to state changes.

## Overview

FynX decorators fall into two main categories:

- **Function Decorators**: Transform functions into reactive components (`@reactive`)
- **Class Decorators**: Make class attributes reactive (`@observable`)

## Function Decorators

### `@reactive` - Automatic Function Execution

The `@reactive` decorator makes functions automatically execute whenever their observable dependencies change. It's perfect for side effects like UI updates, logging, and API calls.

#### Basic Usage

```python
from fynx import reactive, observable

counter = observable(0)

@reactive(counter)
def log_counter_changes(new_value):
    print(f"Counter changed to: {new_value}")

counter = 5  # Prints: "Counter changed to: 5"
counter = 10 # Prints: "Counter changed to: 10"
```

#### Multiple Dependencies

```python
user_name = observable("Alice")
user_age = observable(30)

@reactive(user_name, user_age)
def update_user_display(name, age):
    print(f"User: {name}, Age: {age}")

user_name = "Bob"  # Triggers with current age
user_age = 31      # Triggers with current name
```

#### Store-Level Reactions

React to any change in an entire store:

```python
from fynx import Store, observable, reactive

class UserStore(Store):
    name = observable("Alice")
    age = observable(30)
    email = observable("alice@example.com")

@reactive(UserStore)
def on_any_user_change(store_snapshot):
    print(f"User data changed: {store_snapshot.name}, {store_snapshot.age}")

UserStore.name = "Bob"     # Triggers reaction
UserStore.email = "bob@example.com"  # Also triggers reaction
```

#### Cleanup

Reactive functions can be unsubscribed:

```python
# The decorator returns the original function
# so you can unsubscribe later
unsubscribe_func = reactive(UserStore)(on_any_user_change)

# Later, stop the reaction
UserStore.unsubscribe(on_any_user_change)
```

### Conditional Reactions with @reactive

You can use `@reactive` with conditional observables to create event-driven reactions:

```python
from fynx import reactive, observable

count = observable(5)

# Create a conditional observable
is_above_threshold = count >> (lambda c: c > 10)

@reactive(is_above_threshold)
def on_threshold(is_above):
    if is_above:
        print("Count exceeded threshold!")

count.set(15)  # Triggers the reaction
count.set(8)   # No reaction (condition not met)
count.set(12)  # Triggers the reaction again
```

## Class Decorators

### `@observable` - Reactive Class Attributes

The `@observable` decorator makes class attributes reactive. It's used within Store classes to create observable properties.

#### Basic Usage in Stores

```python
from fynx import Store, observable

class CounterStore(Store):
    count = observable(0)    # Reactive attribute
    step = observable(1)     # Another reactive attribute

# Direct assignment triggers reactivity
CounterStore.count = 5
CounterStore.step = 2
```

#### Computed Properties

While not a decorator itself, the `>>` operator (used with `observable` attributes) creates computed properties:

```python
class CounterStore(Store):
    count = observable(0)
    doubled = count >> (lambda x: x * 2)  # Computed property

CounterStore.count = 5
print(CounterStore.doubled)  # 10
```

## Decorator Patterns

### When to Use Each Decorator

| Decorator | Use Case | Triggers On |
|-----------|----------|-------------|
| `@reactive` | Side effects, UI updates, logging, conditional reactions | Every change to dependencies |
| `@observable` | Reactive state in classes | N/A (attribute decorator) |

### Combining Decorators

```python
from fynx import Store, observable, reactive

class TodoStore(Store):
    todos = observable([])
    filter_mode = observable("all")

    # Reactive: Update UI on any change
    @reactive(todos, filter_mode)
    def update_ui(todos_list, mode):
        print(f"UI updated: {len(todos_list)} todos, filter: {mode}")

    # Conditional reactive: Only when todos exist and filter changes to "completed"
    completed_filter = (todos >> (lambda t: len(t) > 0)) & (filter_mode >> (lambda f: f == "completed"))
    @reactive(completed_filter)
    def show_completion_message(should_show):
        if should_show:
            completed_count = len([t for t in TodoStore.todos if t["completed"]])
            print(f"🎉 {completed_count} todos completed!")
```

### Error Handling

Decorators handle errors gracefully:

```python
@reactive(some_observable)
def potentially_failing_reaction(value):
    if value < 0:
        raise ValueError("Negative values not allowed!")
    print(f"Processed: {value}")

# Errors in reactive functions don't break the reactive system
some_observable = -5  # Error logged, but reactivity continues
some_observable = 10  # Continues working: "Processed: 10"
```

### Performance Considerations

#### Lazy vs Eager Execution

- **Reactive methods** (`@reactive`): Execute eagerly when dependencies change
- **Computed properties**: Execute lazily when accessed

```python
expensive_calc = observable(0)

# Eager: Runs immediately when expensive_calc changes
@reactive(expensive_calc)
def eager_update(val):
    slow_operation(val)  # Runs immediately

# Lazy: Only runs when result is accessed
lazy_result = expensive_calc >> (lambda val: slow_operation(val))

# lazy_result.value  # Only runs slow_operation here
```

#### Memory Management

Always clean up subscriptions:

```python
class Component:
    def __init__(self):
        # Store subscription references for cleanup
        self._cleanup = reactive(store)(self._on_change)

    def destroy(self):
        # Clean up when component is destroyed
        store.unsubscribe(self._on_change)
```

## Common Patterns

### UI State Management

```python
class UIStore(Store):
    sidebar_open = observable(False)
    modal_visible = observable(False)
    loading = observable(False)

    @reactive(sidebar_open)
    def update_layout(open_state):
        if open_state:
            print("📱 Adjusting layout for sidebar")
        else:
            print("📱 Restoring full-width layout")

    # Conditional reactive for loading state
    @reactive(loading)
    def handle_loading_state(is_loading):
        if is_loading:
            print("⏳ Showing loading spinner")
        else:
            print("✅ Hiding loading spinner")
```

### Form Handling

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    confirm_password = observable("")

    # Reactive validation
    @reactive(email)
    def validate_email(email_val):
        if email_val and "@" not in email_val:
            print("❌ Invalid email format")

    # Conditional reactive for complete form
    form_valid = (
        email >> (lambda e: e and "@" in e) &
        password >> (lambda p: len(p) >= 8) &
        (password | confirm_password) >> (lambda p, c: p == c)
    )
    @reactive(form_valid)
    def enable_submit(is_valid):
        if is_valid:
            print("✅ Form is valid and ready to submit")
```

### API Integration

```python
class ApiStore(Store):
    is_loading = observable(False)
    data = observable(None)
    error = observable(None)

    @reactive(is_loading)
    def update_ui_state(loading):
        if loading:
            print("⏳ Showing loading indicator")
        else:
            print("✅ Hiding loading indicator")

    # Conditional reactive for error state
    @reactive(error)
    def handle_error_state(error_val):
        if error_val is not None:
            print(f"❌ Error: {error_val}")

    # Conditional reactive for successful response
    @reactive(data)
    def process_successful_response(data_val):
        if data_val is not None:
            print(f"📦 Processing data: {len(data_val)} items")
```

## Best Practices

### 1. Use Descriptive Function Names

```python
# Good
@reactive(user_data)
def update_user_profile_display(user):
    pass

# Avoid
@reactive(user_data)
def func1(user):
    pass
```

### 2. Keep Reactive Functions Focused

```python
# Good: Single responsibility
@reactive(shopping_cart)
def update_cart_total(cart):
    calculate_total(cart)

@reactive(shopping_cart)
def update_cart_item_count(cart):
    update_counter(len(cart))

# Avoid: Multiple responsibilities
@reactive(shopping_cart)
def handle_cart_changes(cart):
    calculate_total(cart)
    update_counter(len(cart))
    send_analytics(cart)
    update_ui(cart)
```

### 3. Handle Errors Appropriately

```python
@reactive(api_response)
def handle_api_response(response):
    try:
        if response["error"]:
            show_error(response["error"])
        else:
            process_data(response["data"])
    except Exception as e:
        log_error(f"Failed to handle API response: {e}")
        show_generic_error()
```

### 4. Use Conditional Observables for State Machines

```python
current_state = observable("idle")

# Create conditional observables for state transitions
is_loading = current_state >> (lambda s: s == "loading")
is_success = current_state >> (lambda s: s == "success")
is_error = current_state >> (lambda s: s == "error")

@reactive(is_loading)
def enter_loading_state(is_loading_state):
    if is_loading_state:
        show_spinner()

@reactive(is_success)
def enter_success_state(is_success_state):
    if is_success_state:
        hide_spinner()
        show_success_message()

@reactive(is_error)
def enter_error_state(is_error_state):
    if is_error_state:
        hide_spinner()
        show_error_message()
```

### 5. Document Side Effects

```python
@reactive(user_preferences)
def update_application_theme(preferences):
    """
    Updates the application theme based on user preferences.

    Side effects:
    - Modifies CSS custom properties
    - Updates localStorage
    - Triggers re-render of styled components
    """
    apply_theme(preferences["theme"])
    save_to_localstorage("theme", preferences["theme"])
```

Decorators are the bridge between reactive state and imperative code. They allow you to write declarative relationships while maintaining clean, maintainable code. Use `@reactive` for all reactive behavior, combining it with conditional observables for event-driven reactions. Remember that reactive functions should focus on side effects while computed properties handle pure transformations.
