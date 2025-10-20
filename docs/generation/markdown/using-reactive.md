# @reactive: Automatic Reactions to Change

Observables hold state, and Stores organize it. But how do you actually respond when that state changes? How do you keep UI, databases, and external systems in sync?

Right now, if you want to respond to changes, you write this:

```python
count = observable(0)

def log_count(value):
    print(f"Count: {value}")

count.subscribe(log_count)
```

This works. But as your application grows, subscription management becomes tedious:

```python
# Subscriptions scattered everywhere
count.subscribe(update_ui)
count.subscribe(save_to_database)
count.subscribe(notify_analytics)
name.subscribe(update_greeting)
email.subscribe(validate_email)
(first_name | last_name).subscribe(update_display_name)

# Later... did you remember to unsubscribe?
count.unsubscribe(update_ui)
# Wait, which function was subscribed to which observable?
```

You're back to manual synchronization, just with a different syntax. The subscriptions themselves become state you have to manage.

There's a better way.

## Introducing @reactive

The `@reactive` decorator turns functions into automatic reactions. Instead of manually subscribing, you declare *what observables matter* and FynX handles the rest:

```python
from fynx import observable, reactive

count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")

count.set(5)   # Prints: "Count: 5"
count.set(10)  # Prints: "Count: 10"
```

That's it. No manual subscription. No cleanup. Just a declaration: "this function reacts to this observable."

The decorator does three things:

1. **Subscribes automatically** — No need to call `.subscribe()`
2. **Runs immediately** — The function executes once when decorated, giving you the initial state
3. **Runs on every change** — Whenever the observable changes, the function runs with the new value

This is the bridge from passive state management (observables and stores) to active behavior (side effects that respond to changes).

## How It Works: The Execution Model

Understanding when `@reactive` functions run is crucial:

```python
count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")

# At this point, log_count has already run once with the initial value (0)
# Output so far: "Count: 0"

count.set(5)   # log_count runs again
# Output: "Count: 5"

count.set(5)   # Same value - does log_count run?
# Output: (no additional output - only runs when value actually changes)
```

The function runs:
- **Every time `.set()` is called with a different value** — Only when the value actually changes
- **Synchronously** — The function completes before `.set()` returns

This synchronous execution is important. When you write `count.set(5)`, you know that all reactive functions have finished by the time the next line of code runs. This makes reactive code predictable and debuggable.

## The Mental Model: Declarative Side Effects

Traditional programming separates "doing" from "reacting":

```python
# Traditional: Manual coordination
def update_count(new_value):
    count = new_value
    update_ui(count)           # Remember to call this
    save_to_database(count)    # Remember to call this
    log_change(count)          # Remember to call this
```

Every time you modify state, you must remember all the dependent actions. Miss one and your application falls out of sync.

With `@reactive`, you declare the relationships once:

```python
# Reactive: Declare what should happen
@reactive(count)
def update_ui(value):
    print(f"UI: {value}")

@reactive(count)
def save_to_database(value):
    print(f"Saving: {value}")

@reactive(count)
def log_change(value):
    print(f"Log: {value}")

# Now just update state
count.set(42)
# All three functions run automatically
# UI: 42
# Saving: 42
# Log: 42
```

You've moved from "remember to update everything" to "declare what should stay synchronized." The burden of coordination shifts from you to FynX.

## Reacting to Multiple Observables

Most real-world reactions depend on multiple pieces of state. `@reactive` accepts multiple observables:

```python
first_name = observable("Alice")
last_name = observable("Smith")

@reactive(first_name, last_name)
def greet(first, last):
    print(f"Hello, {first} {last}!")

# Runs immediately: "Hello, Alice Smith!"

first_name.set("Bob")
# Runs again: "Hello, Bob Smith!"

last_name.set("Jones")
# Runs again: "Hello, Bob Jones!"
```

When you pass multiple observables, the function receives their values as separate arguments, in the same order you listed them. Change any observable, and the function runs with all current values.

This makes coordinating multiple state sources trivial:

```python
class CartStore(Store):
    items = observable([])
    tax_rate = observable(0.08)

@reactive(CartStore.items, CartStore.tax_rate)
def update_total_display(items, rate):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    tax = subtotal * rate
    total = subtotal + tax
    print(f"Total: ${total:.2f}")

# Runs when items change OR when tax_rate changes
```

You don't write separate subscriptions for each observable. You don't coordinate between them. You just declare: "this function needs these values, run it when any change."

## Reacting to Entire Stores

Sometimes you want to react to *any* change in a Store, regardless of which specific observable changed. Pass the Store class itself:

```python
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)
    email = observable("alice@example.com")

@reactive(UserStore)
def sync_to_server(store_snapshot):
    print(f"Syncing: {store_snapshot.name}, {store_snapshot.email}")

# Runs immediately with initial state

# Runs when name changes:
UserStore.name = "Bob"

# Runs when age changes:
UserStore.age = 31

# Runs when email changes:
UserStore.email = "bob@example.com"
```

The function receives a snapshot of the entire Store. This is perfect for operations that need to consider the complete state—saving to a database, logging changes, synchronizing with a server.

Note the subtle difference: when reacting to individual observables, you get the *values* as arguments. When reacting to a Store, you get the *Store snapshot itself* as a single argument, and you access observables through it.

## Reacting to Computed Observables

Everything that's an observable—including computed ones—works with `@reactive`:

```python
class CartStore(Store):
    items = observable([])

# Computed observable
item_count = CartStore.items >> (lambda items: len(items))

@reactive(item_count)
def update_badge(count):
    print(f"Cart badge: {count}")

# Runs immediately with initial computed value (0)

CartStore.items = [{'name': 'Widget', 'price': 10}]
# Computed value recalculates: 1
# Reactive function runs: "Cart badge: 1"

CartStore.items = CartStore.items + [{'name': 'Gadget', 'price': 15}]
# Computed value recalculates: 2
# Reactive function runs: "Cart badge: 2"
```

You don't react to `CartStore.items` directly. You react to the *computed* value. This is powerful: it means you only care about changes in the *derived* state, not every modification to the underlying data.

## Practical Example: Form Validation

Here's where `@reactive` really shines—coordinating complex UI behavior:

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    confirm_password = observable("")

# Computed validations
email_valid = FormStore.email >> (
    lambda e: '@' in e and '.' in e.split('@')[-1]
)

password_valid = FormStore.password >> (
    lambda p: len(p) >= 8
)

passwords_match = (FormStore.password | FormStore.confirm_password) >> (
    lambda pwd, confirm: pwd == confirm and pwd != ""
)

form_valid = (email_valid | password_valid | passwords_match) >> (
    lambda ev, pv, pm: ev and pv and pm
)

# Reactive UI updates
@reactive(email_valid)
def update_email_indicator(is_valid):
    status = "✓" if is_valid else "✗"
    print(f"Email: {status}")

@reactive(password_valid)
def update_password_indicator(is_valid):
    status = "✓" if is_valid else "✗"
    print(f"Password strength: {status}")

@reactive(passwords_match)
def update_match_indicator(match):
    status = "✓" if match else "✗"
    print(f"Passwords match: {status}")

@reactive(form_valid)
def update_submit_button(is_valid):
    state = "enabled" if is_valid else "disabled"
    print(f"Submit button: {state}")

# Reactive functions run immediately with initial validation states
# Then update the form fields:
FormStore.email = "alice@example.com"
# Email: ✓ (email indicator runs)

FormStore.password = "secure123"
# Password strength: ✓ (password indicator runs)
# Passwords match: ✗ (match indicator runs - passwords don't match yet)

FormStore.confirm_password = "secure123"
# Passwords match: ✓ (match indicator runs)
# Submit button: enabled (form becomes valid)
```

Every UI element updates automatically in response to the relevant state changes. You never write "when email changes, check if it's valid and update the indicator." You just declare the relationship and FynX handles the orchestration.

## When @reactive Runs: Understanding Execution Order

When multiple observables change in quick succession, reactive functions run in a predictable order:

```python
count = observable(0)

@reactive(count)
def first_reaction(value):
    print(f"First: {value}")

@reactive(count)
def second_reaction(value):
    print(f"Second: {value}")

count.set(5)
# Output (order may vary):
# First: 5
# Second: 5
```

Reactive functions run in the order they were decorated. This ordering is deterministic but fragile—if reaction order matters to your application, you're probably doing something wrong. Each reaction should be independent, responding only to the observable values it receives.

If you have reactions that depend on each other, consider using computed observables instead:

```python
# Don't do this: reactions that depend on other reactions
shared_state = []

@reactive(count)
def reaction_one(value):
    shared_state.append(value)

@reactive(count)
def reaction_two(value):
    # This assumes reaction_one has already run
    print(f"Total accumulated: {sum(shared_state)}")

# Do this instead: express dependencies through computed observables
accumulated = count >> (lambda c: sum(range(c + 1)))

@reactive(accumulated)
def show_total(total):
    print(f"Total: {total}")
```

## @reactive vs. Manual Subscriptions

When should you use `@reactive` instead of calling `.subscribe()` directly?

**Use `@reactive` when:**

```python
# You want declarative, self-documenting code
@reactive(user_count)
def update_dashboard(count):
    print(f"Users: {count}")

# You need the function to run immediately with initial state
@reactive(theme)
def apply_theme(theme_name):
    load_css(theme_name)  # Runs right away

# You're defining reactions at module level or class definition
class UIController:
    @reactive(AppStore.mode)
    def sync_mode(mode):
        update_ui_mode(mode)
```

**Use `.subscribe()` when:**

```python
# You need dynamic subscriptions that change at runtime
if user_wants_notifications:
    count.subscribe(send_notification)

# You need to unsubscribe conditionally
subscription_func = count.subscribe(handler)
if some_condition:
    count.unsubscribe(subscription_func)

# You're building a library that accepts observables
def create_widget(data_observable):
    data_observable.subscribe(widget.update)
```

The rule of thumb: `@reactive` for static, declarative reactions that exist for the lifetime of your application. `.subscribe()` for dynamic, programmatic subscriptions that you manage explicitly.

## Common Patterns

**Pattern 1: Syncing to external systems**

```python
@reactive(AppStore)
def save_state(store):
    serialized = {
        'user': store.user,
        'settings': store.settings
    }
    save_to_local_storage('app_state', serialized)
```

**Pattern 2: Logging and debugging**

```python
@reactive(UserStore.login_count)
def log_logins(count):
    print(f"[DEBUG] Login count: {count}")
    if count > 100:
        print("[WARN] Unusual login activity detected")
```

**Pattern 3: Cross-store coordination**

```python
@reactive(ThemeStore.mode)
def update_editor_theme(mode):
    EditorStore.syntax_theme = "dark" if mode == "dark" else "light"
```

**Pattern 4: Analytics and tracking**

```python
@reactive(CartStore.items)
def track_cart_changes(items):
    analytics.track('cart_updated', {
        'item_count': len(items),
        'total_value': sum(item['price'] for item in items)
    })
```

## Gotchas and Edge Cases

**1. Infinite loops are possible**

```python
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)  # DON'T DO THIS

# This will hang your program
```

FynX doesn't prevent infinite loops. If your reactive function modifies an observable it's reacting to, you create a cycle. The solution: reactive functions should perform *side effects* (UI updates, logging, network calls), not modify the observables they're watching.

**2. Reactive functions don't track .value reads**

```python
other_count = observable(10)

@reactive(count)
def show_sum(value):
    print(f"Sum: {value + other_count.value}")

count.set(5)  # Prints: "Sum: 15"
other_count.set(20)  # Doesn't trigger show_sum
```

The function only reacts to observables passed to `@reactive()`. Reading `other_count.value` inside the function doesn't create a dependency. If you want to react to both, pass both:

```python
@reactive(count, other_count)
def show_sum(value, other):
    print(f"Sum: {value + other}")
```

**3. Reactive functions receive values, not observables**

```python
@reactive(count)
def try_to_modify(value):
    value.set(100)  # ERROR: value is an int, not an observable

# If you need the observable, access it directly:
@reactive(count)
def correct_approach(value):
    if value < 0:
        count.set(0)  # Access count directly, not through the argument
```

**4. Store reactions receive snapshots**

```python
@reactive(UserStore)
def save_user(store):
    # store is a snapshot of UserStore at this moment
    # store.name is the current value, not an observable
    save_to_db(store.name)  # Correct

    # This won't give you an observable:
    store.name.subscribe(handler)  # ERROR
```

## Performance Considerations

Reactive functions run synchronously on every change. For expensive operations, consider:

**Debouncing through computed observables:**

```python
search_query = observable("")

# Computed observable that only changes when meaningful
filtered_results = search_query >> (
    lambda q: search_database(q) if len(q) >= 3 else []
)

@reactive(filtered_results)
def update_ui(results):
    display_results(results)  # Only runs when filter criteria met
```

**Conditional logic inside reactions:**

```python
@reactive(mouse_position)
def update_tooltip(position):
    if should_show_tooltip(position):  # Guard clause
        expensive_tooltip_render(position)
```

**Batching updates:**

```python
pending_saves = []

@reactive(DocumentStore.content)
def queue_save(content):
    pending_saves.append(content)
    # Actual save happens elsewhere, periodically
```

## What's Next

`@reactive` gives you automatic reactions to state changes. You can combine it with conditional observables to create event-driven reactions when specific conditions are met. This covers the full spectrum of reactive behaviors—from "keep this in sync" to "do this when that happens."

With observables, stores, and `@reactive`, you have everything you need to build sophisticated reactive applications where state changes automatically propagate through your system, and important transitions trigger the right behaviors at the right times.

## Summary

The `@reactive` decorator transforms functions into automatic reactions that run whenever observables change:

- **Declarative subscriptions** — No manual `.subscribe()` calls to manage
- **Runs immediately and on changes** — Get initial state and all updates
- **Works with any observable** — Standalone, Store attributes, computed values, merged observables, conditional observables
- **Multiple observable support** — React to several sources, receive values as arguments
- **Store-level reactions** — React to any change in an entire Store
- **Conditional reactions** — Use with conditional observables for event-driven behavior
- **Side effects, not state changes** — Reactive functions should perform effects, not modify watched observables

With `@reactive`, you declare *what should happen* when state changes. FynX ensures it happens automatically, in the right order, every time. This eliminates a whole category of synchronization bugs and makes your reactive systems self-maintaining.
