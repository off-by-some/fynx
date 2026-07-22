# @reactive: Automatic Reactions to Change

Observables hold state, and Stores organize it, but neither one says how to actually respond when that state changes - how to keep UI, databases, and external systems in sync.

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
first_name.alongside(last_name).subscribe(update_display_name)

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
# Prints immediately: "Count: 0"

count.set(5)   # Prints: "Count: 5"
count.set(10)  # Prints: "Count: 10"
```

That's it. No manual subscription. No cleanup to remember. Just a declaration: "this function reacts to this observable."

The decorator does two things:

1. **Subscribes automatically** — No need to call `.subscribe()`
2. **Runs on every change** — Whenever the observable changes, the function runs with the new value

This is the bridge from passive state management (observables and stores) to active behavior (side effects that respond to changes).

## A Critical Detail: When Reactions Fire

When you create a reactive function with an active observable or store, it fires immediately with the current value of its dependencies, and then again whenever any dependency changes. Conditional observables are the exception: if the condition is inactive at setup time, the function waits until the condition becomes active.

```python
ready = observable(True)  # Already true

@reactive(ready)
def on_ready(value):
    print(f"Ready: {value}")

# Prints: "Ready: True" (fires immediately with current value)

ready.set(False)  # Prints: "Ready: False"
ready.set(True)   # Prints: "Ready: True"
```

This immediate run is useful for initialization: attach the reaction, and the outside world can be brought into sync with the current state right away. For conditionals, inactive setup means there is no valid gated value to deliver yet, so the first run occurs when the gate opens.

Understanding this execution model is crucial:

```python
count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")

# At this point, log_count has already run once with 0

count.set(5)   # log_count runs again
# Output: "Count: 5"

count.set(5)   # Same value - does log_count run?
# Output: (no additional output - only runs when value actually changes)
```

The function runs only when the value actually changes, and it runs synchronously: `.set()` doesn't return until every reactive function watching that value has finished. By the time `count.set(5)` returns, all of them have already run.

## The Mental Model: Declarative Side Effects

Traditional programming fuses "doing" and "reacting" into the same place: the function that changes the value is also the function that has to remember everything downstream of it:

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
# Each @reactive fires immediately with the current value (0), then again on
# every later change - so decorating these three already prints three lines
# before count.set(42) is even called.
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
# All three functions run automatically, printing "UI: 42", "Saving: 42",
# and "Log: 42" - but not necessarily in that order, since execution order
# between independent subscribers isn't guaranteed.
```

You've moved from "remember to update everything" to "declare what should stay synchronized." The burden of coordination shifts from you to FynX.

## Conditional Reactions: Boolean Logic for When to React

You can combine observables with logical operators to create conditions, then use those conditions directly or gate values with `@` / `.requiring()`:

Use `&` / `.all()` for Boolean AND, `|` / `.either()` for OR, and `~` / `.negate()` for NOT. Use `@` / `.requiring()` when you want to pass a source value through only while a condition is true.

For a pure gating use case - suppress a side effect while a condition doesn't hold - `.requiring()` is exactly right:

```python
is_logged_in = observable(True)
is_verified = observable(True)

# Emits is_logged_in's value only while is_verified is true.
@reactive(is_logged_in @ is_verified)
def show_gated_value(value):
    print(f"Gated value: {value}")
# Fires immediately since the gate starts open: "Gated value: True"

is_logged_in.set(False)
# Gate is still open (is_verified is True) and the value changed -> fires
# Prints: "Gated value: False"

is_verified.set(False)
# Gate closes. Nothing passes through, so nothing prints.

is_logged_in.set(True)
# Gate is still closed (is_verified is still False) -> nothing prints,
# even though is_logged_in changed.

is_verified.set(True)
# Gate reopens, and the current value (True) differs from what was last
# delivered (False) -> fires
# Prints: "Gated value: True"
```

Notice `is_verified.set(False)` and `is_logged_in.set(True)` print nothing - the gate stays closed across both.

When you need a plain boolean that notifies on every relevant change - including complex expressions that mix AND/OR/NOT - build it with the Boolean operators:

```python
is_logged_in = observable(False)
has_data = observable(False)
is_loading = observable(True)
should_sync = observable(False)

# Ready when logged in AND has data AND (NOT loading OR should_sync)
ready_to_sync = is_logged_in & has_data & (~is_loading | should_sync)

@reactive(ready_to_sync)
def sync_to_server(should_run):
    if should_run:
        perform_sync()
```

This mirrors MobX's `when` behavior, but composes more freely: the condition can be any boolean expression, built from as many observables as the logic actually needs.

That composability changes where conditions live: instead of checking them inside your reaction function, you express them in the observable composition itself:

```python
# Instead of this:
@reactive(status)
def maybe_sync(status):
    if status.logged_in and status.has_data and not status.loading:
        perform_sync()

# You can write this:
is_ready = is_logged_in.alongside(has_data).alongside(is_loading).then(
    lambda logged_in, data, loading: logged_in and data and not loading
)

@reactive(is_ready)
def sync_when_ready(should_sync):
    if should_sync:
        perform_sync()
```

The second version states *when* the sync happens in the `@reactive(...)` call itself, so it's visible without reading the function body.

## Reacting to Multiple Observables

Most real-world reactions depend on multiple pieces of state. When you need values from several observables without boolean logic, use `.alongside()`—for combining observables into value tuples:

```python
first_name = observable("Alice")
last_name = observable("Smith")

# Derive a combined observable first
full_name = first_name.alongside(last_name).then(lambda f, l: f"{f} {l}")

# Then react to changes in the derivation
@reactive(full_name)
def update_display(display_name):
    print(f"Display: {display_name}")
# Fires immediately: "Display: Alice Smith"

first_name.set("Bob")  # Triggers with "Bob Smith"
last_name.set("Jones")  # Triggers with "Bob Jones"
```

Notice the pattern: derive first, react second. `.alongside()` combines observables into a stream of value pairs. `.then()` transforms that stream. Only after you've created a derived observable do you attach the reaction.

This is a fundamental principle: most of the time, you don't react directly to raw observables. You react to *derived* observables—computed values that represent the meaningful state for your side effect.

```python
class CartStore(Store):
    items = observable([])
    tax_rate = observable(0.08)

# Derive the meaningful state
total = CartStore.items.alongside(CartStore.tax_rate).then(
    lambda items, rate: sum(item['price'] * item['qty'] for item in items) * (1 + rate)
)

# React to the derived state
@reactive(total)
def update_total_display(total_amount):
    print(f"Total: ${total_amount:.2f}")
```

The reaction only cares about the final computed total, not about whether items changed or tax rate changed. Deriving the meaningful value first, then reacting to that, is what keeps it simple.

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
# Fires immediately with a snapshot: "Syncing: Alice, alice@example.com"

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

Computed observables work with `@reactive` the same way standalone ones do:

```python
class CartStore(Store):
    items = observable([])

# Computed observable
item_count = CartStore.items.then(lambda items: len(items))

@reactive(item_count)
def update_badge(count):
    print(f"Cart badge: {count}")
# Fires immediately: "Cart badge: 0"

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
length = items.then(lambda i: len(i))

@reactive(length)
def log_length(l):
    print(f"Length: {l}")
# Fires immediately: "Length: 3"

items.set([4, 5, 6])  # Length is still 3, reaction doesn't fire
items.set([7, 8, 9, 10])  # Length is now 4, reaction fires
```

The reaction is tied to the length changing, not to the list being reassigned.

## The Commitment: What You Gain and What You Give Up

Once a function is decorated with `@reactive`, it runs automatically when its dependencies change, and it can no longer be called manually:

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")
# Prints immediately with the current value of count

log_count(10)  # Raises fynx.reactive.ReactiveFunctionWasCalled exception
```

If manual calls were allowed while subscribed, `log_count(10)` and the next `count.set()` could both fire with different arguments, and nothing says which result should win. FynX raises instead of picking one silently.

You can always change your mind, though. Call `.unsubscribe()` to sever the reactive connection and return the function to normal, non-reactive behavior:

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")
# Prints immediately with the current value of count

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

A signup form makes the case well - three independent checks feeding one submit button:

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    confirm_password = observable("")

# Computed validations
email_valid = FormStore.email.then(
    lambda e: '@' in e and '.' in e.split('@')[-1]
)

password_valid = FormStore.password.then(
    lambda p: len(p) >= 8
)

passwords_match = FormStore.password.alongside(FormStore.confirm_password).then(
    lambda pwd, confirm: pwd == confirm and pwd != ""
)

form_valid = email_valid & password_valid & passwords_match

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
# email_valid, password_valid, and passwords_match are ordinary computed
# observables, so all three fire immediately here:
# Email: ✗
# Password strength: ✗
# Passwords match: ✗
# form_valid is a total boolean, so the submit button starts disabled.
# Submit button: disabled

# Now update the form fields:
FormStore.email = "alice@example.com"
# Email: ✓ (email indicator runs)

FormStore.password = "secure123"
# Password strength: ✓ (password indicator runs)
# passwords_match does NOT run here: it was already False (empty == empty
# is true, but the "pwd != ''" check made it False) and is still False now
# ("secure123" != "" and "secure123" != confirm_password), so its value
# hasn't changed and it doesn't notify.

FormStore.confirm_password = "secure123"
# Passwords match: ✓ (match indicator runs - value changed from False to True)
# Submit button: enabled (form becomes valid for the first time)
```

Every UI element updates automatically in response to the relevant state changes. You never write "when email changes, check if it's valid and update the indicator." You just declare the relationship and FynX handles the orchestration.

`form_valid` uses Boolean AND, so it tracks both valid and invalid states. Like any computed observable, the reactive function fires only when the delivered value actually changes - which is why the submit button doesn't flicker on every keystroke.

## The Core Insight: Where @reactive Belongs

Here's the fundamental principle that makes reactive systems maintainable: **`@reactive` is for side effects, not for deriving state.**

When you're tempted to use `@reactive`, ask yourself: "Am I computing a new value from existing data, or am I sending information outside my application?" If you're computing, you want `.then()`, `.alongside()`, `.all()`, `.either()`, or `.negate()`. If you're gating a value, you want `.requiring()` or `@`. If you're communicating with the outside world, you want `@reactive`.

This is the "functional core, reactive shell" pattern: the core is built from pure `.then()` transforms, explicit `.alongside()` products, total boolean conditions with `.all()` / `.either()` / `.negate()`, and gates with `.requiring()` / `@`. The shell is the small set of `@reactive` functions that actually touch the outside world.

```python
# ===== FUNCTIONAL CORE (Pure) =====
class OrderCore(Store):
    items = observable([])
    shipping_address = observable(None)
    payment_method = observable(None)
    is_processing = observable(False)

    # Pure derivations—no side effects anywhere
    subtotal = items.then(lambda i: sum(x['price'] * x['qty'] for x in i))
    has_items = items.then(lambda i: len(i) > 0)
    has_address = shipping_address.then(lambda a: a is not None)
    has_payment = payment_method.then(lambda p: p is not None)

    # Boolean logic for conditions
    can_checkout = has_items & has_address & has_payment & ~is_processing

    tax = subtotal.then(lambda s: s * 0.08)
    total = subtotal.alongside(tax).then(lambda s, t: s + t)

# ===== REACTIVE SHELL (Impure) =====
@reactive(OrderCore.can_checkout)
def update_checkout_button(can_checkout):
    button.disabled = not can_checkout

@reactive(OrderCore.total)
def update_display(total):
    render_total(f"${total:.2f}")

# Only auto-save when we have items and aren't processing
@reactive(OrderCore.has_items @ OrderCore.is_processing.negate())
def auto_save(should_save):
    if should_save:
        save_to_db(OrderCore.to_dict())
```

The core has no database calls, no DOM manipulation, no network requests - only derivations, which you can test by calling them with plain values and checking the result.

The reactions appear only at the boundary. They're where your perfect functional world meets reality: updating a button's state, rendering to the screen, persisting to a database. The conditional operators let you express exactly when these side effects should occur, without polluting your core logic.

See the [Best Practices](best-practices.md) page for guidance on when to reach for `@reactive`, along with the anti-patterns and cleanup patterns that come up around it.

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

See the [Best Practices](best-practices.md) page for gotchas around self-mutation, implicit `.value` reads, and Store snapshots.

## Performance Considerations

Reactive functions run synchronously on every change. For expensive operations, consider:

**Reacting to derived state that filters changes:**

```python
search_query = observable("")

# Only changes when meaningful
filtered_results = search_query.then(
    lambda q: search_database(q) if len(q) >= 3 else []
)

@reactive(filtered_results)
def update_ui(results):
    display_results(results)
```

**Using conditional observables to limit when reactions fire:**

```python
should_update = user_active & ~is_loading

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
* **Runs immediately when active** — Active observables and stores run once at setup, then again on changes; inactive conditionals wait until their gate opens
* **Works with any observable** — Standalone, Store attributes, computed values, merged observables
* **[Conditional operators](conditionals.md) for conditions** — Use `.all()`, `.either()`, and `.negate()` to create boolean conditions, then `.requiring()` / `@` to gate values when you need a conditional reaction
* **Multiple observable support** — Derive combined observables first, then react
* **Store-level reactions** — React to any change in an entire Store
* **Lifecycle management** — Use `.unsubscribe()` to stop reactive behavior and restore normal function calls
* **Prevents manual calls** — Raises `fynx.reactive.ReactiveFunctionWasCalled` if called manually while subscribed
* **Side effects, not state changes** — Reactive functions should perform effects, not modify watched observables

Most of your code should still be pure derivations built with [`.then()`](derived-observables.md), `.alongside()`, and [`.all()`, `.either()`, and `.negate()`](conditionals.md), plus `.requiring()` / `@` for gates. `@reactive` belongs only at the edges, where the application has to touch something outside itself. See [Best Practices](best-practices.md) for the anti-patterns to watch for as your reactions grow.
