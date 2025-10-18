# Decorators - Making Code Reactive

Decorators in FynX provide the high-level API for creating reactive relationships in your application. They transform regular Python functions and classes into reactive components that automatically respond to state changes.

## Overview

FynX decorators fall into two main categories:

- **Function Decorators**: Transform functions into reactive components (`@reactive`, `@watch`)
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

### `@watch` - Conditional Reactions

The `@watch` decorator creates reactions that only trigger when specific conditions are met. Unlike `@reactive` which triggers on every change, `@watch` only runs when conditions transition from unmet to met.

#### Basic Conditional Watching

```python
from fynx import watch, observable

user_online = observable(False)
message_count = observable(0)

@watch(
    lambda: user_online.value,           # User must be online
    lambda: message_count.value > 0      # Must have messages
)
def send_notifications():
    print(f"📬 Sending {message_count.value} notifications!")

# No reaction yet - conditions not met
message_count = 5       # User offline
user_online = True      # Now both conditions met - triggers!

# No reaction - already met conditions
message_count = 3       # Still triggers (count changed)
user_online = False     # Conditions no longer met

user_online = True      # Triggers again - conditions newly met
```

#### Complex Conditions

```python
is_logged_in = observable(False)
has_permissions = observable(False)
data_loaded = observable(False)

@watch(
    lambda: is_logged_in.value,
    lambda: has_permissions.value,
    lambda: data_loaded.value
)
def enable_feature():
    print("🎉 Feature enabled - all conditions met!")

# Only triggers when ALL conditions become true after being false
is_logged_in = True     # Not yet - missing permissions
has_permissions = True  # Not yet - missing data
data_loaded = True      # Now triggers!
```

#### Real-World Example: Form Validation

```python
email = observable("")
password = observable("")
terms_accepted = observable(False)

@watch(
    lambda: email.value and "@" in email.value,
    lambda: len(password.value) >= 8,
    lambda: terms_accepted.value
)
def enable_submit_button():
    print("✅ Form is valid - submit button enabled")

email = "user@example.com"  # Not yet - password too short
password = "secure123"      # Not yet - terms not accepted
terms_accepted = True       # All conditions met - triggers!
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
| `@reactive` | Side effects, UI updates, logging | Every change to dependencies |
| `@watch` | Conditional logic, state machines, validation | Conditions becoming newly met |
| `@observable` | Reactive state in classes | N/A (attribute decorator) |

### Combining Decorators

```python
from fynx import Store, observable, reactive, watch

class TodoStore(Store):
    todos = observable([])
    filter_mode = observable("all")

    # Reactive: Update UI on any change
    @reactive(todos, filter_mode)
    def update_ui(todos_list, mode):
        print(f"UI updated: {len(todos_list)} todos, filter: {mode}")

    # Watch: Only when todos exist and filter changes to "completed"
    @watch(
        lambda: len(TodoStore.todos.value) > 0,
        lambda: TodoStore.filter_mode.value == "completed"
    )
    def show_completion_message():
        completed_count = len([t for t in TodoStore.todos.value if t["completed"]])
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
- **Watch methods** (`@watch`): Execute eagerly only when conditions are newly met
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

    @watch(lambda: UIStore.loading.value)
    def show_loading_spinner():
        print("⏳ Showing loading spinner")

    @watch(lambda: not UIStore.loading.value)
    def hide_loading_spinner():
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

    # Watch for complete form
    @watch(
        lambda: FormStore.email.value and "@" in FormStore.email.value,
        lambda: len(FormStore.password.value) >= 8,
        lambda: FormStore.password.value == FormStore.confirm_password.value
    )
    def enable_submit():
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

    @watch(lambda: ApiStore.error.value is not None)
    def show_error_message():
        print(f"❌ Error: {ApiStore.error.value}")

    @watch(lambda: ApiStore.data.value is not None)
    def process_successful_response():
        print(f"📦 Processing data: {len(ApiStore.data.value)} items")
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

### 4. Prefer `@watch` for State Machines

```python
current_state = observable("idle")

@watch(lambda: current_state.value == "loading")
def enter_loading_state():
    show_spinner()

@watch(lambda: current_state.value == "success")
def enter_success_state():
    hide_spinner()
    show_success_message()

@watch(lambda: current_state.value == "error")
def enter_error_state():
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

Decorators are the bridge between reactive state and imperative code. They allow you to write declarative relationships while maintaining clean, maintainable code. Choose the right decorator for your use case, and remember that reactive functions should focus on side effects while computed properties handle pure transformations.
