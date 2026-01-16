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
(first_name + last_name).subscribe(update_display_name)

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

That's it. No manual subscription. No cleanup to remember. Just a declaration: "this function reacts to this observable."

The decorator does two things:

1. **Subscribes automatically** — No need to call `.subscribe()`
2. **Runs on every change** — Whenever the observable changes, the function runs with the new value

This is the bridge from passive state management (observables and stores) to active behavior (side effects that respond to changes).

## A Critical Detail: When Reactions Fire

When you create a reactive function, it fires immediately with the current value of its dependencies, and then again whenever any dependency changes.

```python
ready = observable(True)  # Already true

@reactive(ready)
def on_ready(value):
    print(f"Ready: {value}")

# Prints: "Ready: True" (fires immediately with current value)

ready.set(False)  # Prints: "Ready: False"
ready.set(True)   # Prints: "Ready: True"
```

This behavior has deep roots in category theory—reactive functions form what's called a "pullback" in categorical semantics. The initial state isn't captured because you haven't pulled back through a change yet. You're observing the flow of changes, not the snapshot of current state.

This matters enormously for initialization logic. If you need something to run immediately based on current state, you'll need to handle that separately, perhaps by calling the function once manually before decorating it, or by setting up your initial state in a way that triggers the reaction. Reactive functions are about responding to transitions, not about reflecting static state.

Understanding this execution model is crucial:

```python
count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")

# At this point, log_count has NOT run yet - no initial trigger

count.set(5)   # log_count runs for the first time
# Output: "Count: 5"

count.set(5)   # Same value - does log_count run?
# Output: (no additional output - only runs when value actually changes)
```

The function runs every time `.set()` is called with a different value—only when the value actually changes. The execution is synchronous—the function completes before `.set()` returns. This makes reactive code predictable and debuggable. When you write `count.set(5)`, you know that all reactive functions have finished by the time the next line runs.

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

## Conditional Reactions: Boolean Logic for When to React

Here's where `@reactive` becomes truly powerful. You can combine observables with logical operators to create conditional reactions that only fire when specific conditions are met:

```python
is_logged_in = observable(False)
has_data = observable(False)
is_loading = observable(True)
should_sync = observable(False)

# React only when logged in AND has data AND NOT loading OR should sync
@reactive(is_logged_in & has_data & ~is_loading + should_sync)
def sync_to_server(should_run):
    if should_run:
        perform_sync()
```

The operators work exactly as you'd expect:

* `&` is logical AND
* `+` is logical OR (when used with observables on both sides)
* `~` is logical NOT (negation)

These create composite observables that emit values based on boolean logic. The critical insight: the reaction still follows the change-only semantics. Even if your condition is `True` when you attach the reactive function, it won't fire until something changes *and* the condition evaluates.

```python
logged_in = observable(True)
verified = observable(True)

# Even though both are already True, this doesn't fire yet
@reactive(logged_in & verified)
def enable_premium_features(both_true):
    print(f"Premium features: {both_true}")

# Nothing printed yet - waiting for first change

logged_in.set(False)  # Condition now False, triggers reaction
# Prints: "Premium features: False"

verified.set(False)  # Both False, triggers reaction
# Prints: "Premium features: False"

logged_in.set(True)  # One is True, one is False, triggers reaction
# Prints: "Premium features: False"

verified.set(True)  # Both True now, triggers reaction
# Prints: "Premium features: True"
```

This mirrors MobX's `when` behavior, but with more compositional flexibility. You're not limited to simple conditions—you can build arbitrarily complex boolean expressions that describe exactly when your side effect should consider running.

Think of it as event-driven reactions with declarative conditions. Instead of checking conditions inside your reaction function, you express them in the observable composition itself:

```python
# Instead of this:
@reactive(status)
def maybe_sync(status):
    if status.logged_in and status.has_data and not status.loading:
        perform_sync()

# You can write this:
@reactive(logged_in & has_data & ~is_loading)
def sync_when_ready(should_sync):
    if should_sync:
        perform_sync()
```

The second version is clearer about *when* the sync happens—the condition is part of the observable dependency declaration, not buried in the function body.

## Reacting to Multiple Observables

Most real-world reactions depend on multiple pieces of state. When you need values from several observables without boolean logic, use the `+` operator differently—for combining observables into value tuples:

```python
first_name = observable("Alice")
last_name = observable("Smith")

# Derive a combined observable first
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")

# Then react to changes in the derivation
@reactive(full_name)
def update_display(display_name):
    print(f"Display: {display_name}")

# Nothing prints yet - waiting for first change

first_name.set("Bob")  # Triggers with "Bob Smith"
last_name.set("Jones")  # Triggers with "Bob Jones"
```

Notice the pattern: derive first, react second. The `+` operator combines observables into a stream of value pairs. The `>>` operator transforms that stream. Only after you've created a derived observable do you attach the reaction.

This is a fundamental principle: most of the time, you don't react directly to raw observables. You react to *derived* observables—computed values that represent the meaningful state for your side effect.

```python
class CartStore(Store):
    items = observable([])
    tax_rate = observable(0.08)

# Derive the meaningful state
total = (CartStore.items + CartStore.tax_rate) >> (
    lambda items, rate: sum(item['price'] * item['qty'] for item in items) * (1 + rate)
)

# React to the derived state
@reactive(total)
def update_total_display(total_amount):
    print(f"Total: ${total_amount:.2f}")
```

The reaction only cares about the final computed total, not about whether items changed or tax rate changed. This separation of concerns—derive meaning, then react to it—keeps your reactive functions simple and focused.

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

# Doesn't run immediately - waits for first change

UserStore.name = "Bob"  # Triggers reaction
# Prints: "Syncing: Bob, alice@example.com"

UserStore.age = 31  # Triggers reaction
# Prints: "Syncing: Bob, alice@example.com"

UserStore.email = "bob@example.com"  # Triggers reaction
# Prints: "Syncing: Bob, bob@example.com"
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

# Doesn't run immediately - waits for first change

CartStore.items = [{'name': 'Widget', 'price': 10}]
# Computed value recalculates: 1
# Reactive function runs: "Cart badge: 1"

CartStore.items = CartStore.items + [{'name': 'Gadget', 'price': 15}]
# Computed value recalculates: 2
# Reactive function runs: "Cart badge: 2"
```

You don't react to `CartStore.items` directly. You react to the *computed* value. This is powerful: it means you only care about changes in the *derived* state, not every modification to the underlying data.

If the computed value doesn't change, the reaction doesn't fire:

```python
items = observable([1, 2, 3])
length = items >> (lambda i: len(i))

@reactive(length)
def log_length(l):
    print(f"Length: {l}")

items.set([4, 5, 6])  # Length is still 3, reaction doesn't fire
items.set([7, 8, 9, 10])  # Length is now 4, reaction fires
```

This is exactly what you want—reactions tied to semantic meaning, not raw data changes.

## The Commitment: What You Gain and What You Give Up

Once you decorate a function with `@reactive`, you're making a commitment. The function becomes automatic—it runs when its dependencies change. In exchange, you lose the ability to call it manually:

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")

log_count(10)  # Raises fynx.reactive.ReactiveFunctionWasCalled exception
```

This isn't an arbitrary restriction. It's protecting you from confusion. If you could call `log_count()` manually *and* have it trigger automatically, which version of the value is authoritative? The manual call or the reactive update? The framework eliminates this ambiguity by enforcing one mode at a time.

You can always change your mind, though. Call `.unsubscribe()` to sever the reactive connection and return the function to normal, non-reactive behavior:

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")

count.set(5)   # Prints: "Count: 5"

log_count.unsubscribe()  # Severs the reactive connection

count.set(10)  # No output—the function is no longer reactive
log_count(15)  # Prints: "Count: 15"—now works as a normal function
```

After unsubscribing, the function reverts to its original, non-reactive form. You can call it manually again, and it will no longer respond to changes in its former dependencies.

This lifecycle management is particularly important for component-based architectures:

```python
class UIComponent:
    def __init__(self):
        self.count = observable(0)

        @reactive(self.count)
        def update_display(value):
            print(f"Display: {value}")

        self._update_display = update_display

    def destroy(self):
        # Clean up when component is destroyed
        self._update_display.unsubscribe()
```

The pattern is simple: create reactive functions when you need them, unsubscribe when you're done. This prevents memory leaks and ensures reactions don't outlive the components they serve.

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

passwords_match = (FormStore.password + FormStore.confirm_password) >> (
    lambda pwd, confirm: pwd == confirm and pwd != ""
)

form_valid = (email_valid & password_valid & passwords_match) >> (lambda x: x)

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

# Reactive functions don't run immediately
# Now update the form fields:
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

Notice how we use the `&` operator to create `form_valid`—it only becomes true when all three conditions are met. The reactive function on `form_valid` only fires when something changes, giving you precise control over when the submit button updates.

## The Core Insight: Where @reactive Belongs

Here's the fundamental principle that makes reactive systems maintainable: **`@reactive` is for side effects, not for deriving state.**

When you're tempted to use `@reactive`, ask yourself: "Am I computing a new value from existing data, or am I sending information outside my application?" If you're computing, you want `>>`, `+`, `&`, or `~` operators. If you're communicating with the outside world, you want `@reactive`.

This distinction creates what we call the "functional core, reactive shell" pattern. Your core is pure transformations—testable, predictable, composable. Your shell is reactions—the unavoidable side effects that make your application actually do something.

```python
# ===== FUNCTIONAL CORE (Pure) =====
class OrderCore(Store):
    items = observable([])
    shipping_address = observable(None)
    payment_method = observable(None)
    is_processing = observable(False)

    # Pure derivations—no side effects anywhere
    subtotal = items >> (lambda i: sum(x['price'] * x['qty'] for x in i))
    has_items = items >> (lambda i: len(i) > 0)
    has_address = shipping_address >> (lambda a: a is not None)
    has_payment = payment_method >> (lambda p: p is not None)

    # Boolean logic for conditions
    can_checkout = (has_items & has_address & has_payment & ~is_processing) >> (lambda x: x)

    tax = subtotal >> (lambda s: s * 0.08)
    total = (subtotal + tax) >> (lambda s, t: s + t)

# ===== REACTIVE SHELL (Impure) =====
@reactive(OrderCore.can_checkout)
def update_checkout_button(can_checkout):
    button.disabled = not can_checkout

@reactive(OrderCore.total)
def update_display(total):
    render_total(f"${total:.2f}")

# Only auto-save when we have items and aren't processing
@reactive(OrderCore.has_items & ~OrderCore.is_processing)
def auto_save(should_save):
    if should_save:
        save_to_db(OrderCore.to_dict())
```

Notice how the core is entirely composed of derivations—values computed from other values. No database calls, no DOM manipulation, no network requests. These pure transformations are easy to test, easy to understand, and easy to change.

The reactions appear only at the boundary. They're where your perfect functional world meets reality: updating a button's state, rendering to the screen, persisting to a database. The conditional operators let you express exactly when these side effects should occur, without polluting your core logic.

## Best Practices for @reactive

### Use @reactive for Side Effects Only

**✅ USE @reactive for:**

**Side Effects at Application Boundaries**

```python
@reactive(UserStore)
def sync_to_server(store):
    # Network I/O
    api.post('/user/update', store.to_dict())

@reactive(settings_changed)
def save_to_local_storage(settings):
    # Browser storage I/O
    localStorage.setItem('settings', json.dumps(settings))
```

**UI Updates** (DOM manipulation, rendering)

```python
@reactive(cart_total)
def update_total_display(total):
    # DOM manipulation
    render_total(f"${total:.2f}")
```

**Logging and Monitoring**

```python
@reactive(error_observable)
def log_errors(error):
    # Logging side effect
    logger.error(f"Application error: {error}")
    analytics.track('error', {'message': str(error)})
```

**Cross-System Coordination** (when one reactive system needs to update another)

```python
@reactive(ThemeStore.dark_mode)
def update_editor_theme(is_dark):
    # Coordinating separate systems
    EditorStore.theme = 'dark' if is_dark else 'light'
```

**❌ AVOID @reactive for:**

**Deriving State** (use `>>`, `+`, `&`, `~` instead)

```python
# BAD: Using @reactive for transformation
@reactive(count)
def doubled_count(value):
    doubled.set(value * 2)  # Modifying another observable

# GOOD: Use functional transformation
doubled = count >> (lambda x: x * 2)
```

**State Coordination** (use computed observables)

```python
# BAD: Coordinating state in reactive functions
@reactive(first_name, last_name)
def update_full_name(first, last):
    full_name.set(f"{first} {last}")

# GOOD: Express as derived state
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")
```

**Business Logic** (keep logic in pure transformations)

```python
# BAD: Business logic in reactive function
@reactive(order_items)
def calculate_total(items):
    total = sum(item.price * item.quantity for item in items)
    tax = total * 0.08
    order_total.set(total + tax)

# GOOD: Business logic in pure transformation
order_total = order_items >> (lambda items:
    sum(i.price * i.quantity for i in items) * 1.08
)
```

### Anti-Patterns to Avoid

**The infinite loop.** When a reaction modifies what it's watching, you've created a feedback cycle:

```python
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)  # Every change triggers another change
```

This is obvious in toy examples but can hide in real code when the dependency is indirect. The change semantics don't save you here—each change triggers the reaction, which causes another change, ad infinitum.

**The hidden cache.** When reactions maintain their own state, you've split your application's state across two systems:

```python
results_cache = {}

@reactive(query)
def update_cache(query_string):
    results_cache[query_string] = fetch_results(query_string)
```

Now you have to remember that `results_cache` exists and keep it synchronized. Better to make the cache itself observable and derive from it.

**The sequential assumption.** When reactions depend on each other's execution order, you've created fragile coupling:

```python
shared_list = []

@reactive(data)
def reaction_one(value):
    shared_list.append(value)

@reactive(data)
def reaction_two(value):
    # Assumes reaction_one has already run
    print(f"List has {len(shared_list)} items")
```

The second reaction assumes the first has already run. But that's an implementation detail, not a guarantee. If execution order changes, your code breaks silently.

The fix for all three is the same: keep reactions independent and stateless. Let the observable system coordinate state. Keep reactions purely about effects.

## Advanced Patterns: Conditional Guards and Cleanup

The conditional operators shine when you need to guard expensive or sensitive operations:

```python
user = observable(None)
has_permission = observable(False)
is_online = observable(False)

# Only sync when user is logged in, has permission, and is online
@reactive(user & has_permission & is_online)
def sync_sensitive_data(should_sync):
    if should_sync and user.get():
        api.sync_user_data(user.get().id)

# Later, when you want to stop syncing entirely:
sync_sensitive_data.unsubscribe()
```

The unsubscribe mechanism becomes particularly important in cleanup scenarios. If your reactive function represents a resource that needs explicit teardown (like a WebSocket connection or a file handle), you can unsubscribe when you're done to prevent further reactions and then perform cleanup in the function itself.

## @reactive vs. Manual Subscriptions

When should you use `@reactive` instead of calling `.subscribe()` directly?

**Use `@reactive` when:**

You want declarative, self-documenting code:

```python
@reactive(user_count)
def update_dashboard(count):
    print(f"Users: {count}")
```

You're defining reactions at module level or class definition:

```python
class UIController:
    @reactive(AppStore.mode)
    def sync_mode(mode):
        update_ui_mode(mode)
```

You need lifecycle management with unsubscribe capability:

```python
@reactive(data_stream)
def process_data(data):
    handle_data(data)

# Later, when component is destroyed
process_data.unsubscribe()  # Clean up
```

**Use `.subscribe()` when:**

You need dynamic subscriptions that change at runtime:

```python
if user_wants_notifications:
    count.subscribe(send_notification)
```

You need to unsubscribe conditionally:

```python
subscription_func = count.subscribe(handler)
if some_condition:
    count.unsubscribe(subscription_func)
```

You're building a library that accepts observables:

```python
def create_widget(data_observable):
    data_observable.subscribe(widget.update)
```

The rule of thumb: `@reactive` for static, declarative reactions with optional cleanup via `.unsubscribe()`. `.subscribe()` for dynamic, programmatic subscriptions that you manage explicitly.

## Gotchas and Edge Cases

**Infinite loops are possible**

```python
# BAD: Modifying what you're watching
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)  # DON'T DO THIS
```

Solution: Reactive functions should perform *side effects*, not modify the observables they're watching. Use computed observables for transformations.

**Reactive functions don't track .get() or .value reads**

```python
# BAD: Hidden dependency
other_count = observable(10)

@reactive(count)
def show_sum(value):
    print(f"Sum: {value + other_count.get()}")  # Hidden dependency

count.set(5)  # Prints: "Sum: 15"
other_count.set(20)  # Doesn't trigger show_sum - bug!
```

Solution: Make all dependencies explicit in the decorator.

**Reactive functions receive values, not observables**

```python
# BAD: Trying to modify the value parameter
@reactive(count)
def try_to_modify(value):
    value.set(100)  # ERROR: value is an int, not an observable
```

Solution: Access the observable directly if you need to modify it (though you usually shouldn't).

**Store reactions receive snapshots**

```python
@reactive(UserStore)
def save_user(store):
    # store is a snapshot of UserStore at this moment
    # store.name is the current value, not an observable
    save_to_db(store.name)  # Correct

    # This won't work:
    store.name.subscribe(handler)  # ERROR
```

**Order-dependent reactions are fragile**

Express dependencies through computed observables instead of relying on execution order between multiple reactions.

## Performance Considerations

Reactive functions run synchronously on every change. For expensive operations, consider:

**Reacting to derived state that filters changes:**

```python
search_query = observable("")

# Only changes when meaningful
filtered_results = search_query >> (
    lambda q: search_database(q) if len(q) >= 3 else []
)

@reactive(filtered_results)
def update_ui(results):
    display_results(results)
```

**Using conditional observables to limit when reactions fire:**

```python
should_update = (user_active & ~is_loading) >> (lambda x: x)

@reactive(should_update)
def update_display(should):
    if should:
        expensive_render()
```

**Conditional logic inside reactions:**

```python
@reactive(mouse_position)
def update_tooltip(position):
    if should_show_tooltip(position):
        expensive_tooltip_render(position)
```

## Summary

The `@reactive` decorator transforms functions into automatic reactions that run whenever observables change:

* **Declarative subscriptions** — No manual `.subscribe()` calls to manage
* **Runs on changes only** — No initial execution; waits for first change (pullback semantics)
* **Works with any observable** — Standalone, Store attributes, computed values, merged observables
* **Boolean operators for conditions** — Use `&`, `+`, `~` to create conditional reactions (like MobX's `when`)
* **Multiple observable support** — Derive combined observables first, then react
* **Store-level reactions** — React to any change in an entire Store
* **Lifecycle management** — Use `.unsubscribe()` to stop reactive behavior and restore normal function calls
* **Prevents manual calls** — Raises `fynx.reactive.ReactiveFunctionWasCalled` if called manually while subscribed
* **Side effects, not state changes** — Reactive functions should perform effects, not modify watched observables

With `@reactive`, you declare *what should happen* when state changes. FynX ensures it happens automatically, in the right order, every time. This eliminates a whole category of synchronization bugs and makes your reactive systems self-maintaining.

The rule of thumb here is that most of your code should be pure derivations using `>>`, `+`, `&`, and `~`. Reactions with `@reactive` appear only at the edges, where your application must interact with something external. This separation—the functional core, reactive shell pattern—is what makes reactive systems both powerful and maintainable.
