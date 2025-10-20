# Understanding the @reactive Decorator

The `@reactive` decorator bridges your pure, functional data transformations with the messy, real world of side effects. Think of it as the membrane between your application's logic and everything outside it—the UI, the network, the file system, the console.

## Starting Simple

Let's see what reactive functions look like in practice:

```python
from fynx import reactive, observable

count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")

count.set(5)   # Prints: "Count: 5"
count.set(10)  # Prints: "Count: 10"
```

The function runs automatically whenever `count` changes. You declare what should happen when data changes, and the framework handles the timing.

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

## A Crucial Detail: Initial State and Change Semantics

Here's something that might surprise you: when you create a reactive function, it doesn't fire immediately with the current value. It only fires when the value *changes*.

```python
ready = observable(True)  # Already true

@reactive(ready)
def on_ready(value):
    print(f"Ready: {value}")

# Nothing prints yet, even though ready is True

ready.set(False)  # Prints: "Ready: False"
ready.set(True)   # Prints: "Ready: True"
```

This behavior has deep roots in category theory—reactive functions form what's called a "pullback" in categorical semantics. The initial state isn't captured because you haven't pulled back through a change yet. You're observing the flow of changes, not the snapshot of current state.

This matters enormously for initialization logic. If you need something to run immediately based on current state, you'll need to handle that separately. Reactive functions are about responding to transitions, not about reflecting static state.

## Conditional Reactions: The MobX `when` Pattern

Here's where things get powerful. You can combine observables with logical operators to create conditional reactions that only fire when specific conditions are met:

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

The operators work as you'd expect:

* `&` is logical AND
* `+` is logical OR
* `~` is logical NOT (negation)

These create composite observables that emit values based on boolean logic applied to their constituent observables. The critical insight: the reaction still follows the change-only semantics. Even if your condition is `True` at the moment you attach the reactive function, it won't fire until something changes *and* the condition is met.

```python
logged_in = observable(True)
verified = observable(True)

# Even though both are already True, this doesn't fire yet
@reactive(logged_in & verified)
def enable_premium_features(both_true):
    print("Premium features enabled")

# Nothing printed yet

logged_in.set(False)  # Condition now False, triggers reaction
# Prints: "Premium features enabled" with value False

verified.set(False)  # Both False, triggers reaction
# Prints: "Premium features enabled" with value False

logged_in.set(True)  # One is True, one is False, triggers reaction
# Prints: "Premium features enabled" with value False

verified.set(True)  # Both True now, triggers reaction
# Prints: "Premium features enabled" with value True
```

This mirrors MobX's `when` behavior, but with more compositional flexibility. You're not limited to simple conditions—you can build arbitrarily complex boolean expressions that describe exactly when your side effect should consider running.

## Multiple Dependencies Without Conditions

Sometimes you just want a reaction to fire whenever any of several observables change, without boolean logic:

```python
name = observable("Alice")
age = observable(30)

# Derive a combined observable first
full_name = (name + age) >> (lambda n, a: f"{n} ({a} years old)")

# Then react to changes in the derivation
@reactive(full_name)
def update_display(display_name):
    print(f"Display: {display_name}")

name.set("Bob")  # Triggers with "Bob (30 years old)"
age.set(31)      # Triggers with "Bob (31 years old)"
```

Notice the pattern: derive first, react second. The `+` operator here isn't doing boolean OR—it's combining observables into a tuple-like stream. The `>>` operator then transforms that stream. Only after you've created a derived observable do you attach the reaction.

## The Core Insight: Where @reactive Belongs

Here's the fundamental principle that makes reactive systems maintainable: **`@reactive` is for side effects, not for deriving state.**

When you're tempted to use `@reactive`, ask yourself: "Am I computing a new value from existing data, or am I sending information outside my application?" If you're computing, you want `>>` or `+` operators. If you're communicating with the outside world, you want `@reactive`.

This distinction creates what we call the "functional core, reactive shell" pattern. Your core is pure transformations—testable, predictable, composable. Your shell is reactions—the unavoidable side effects that make your application actually do something.

Let's see this in a real example:

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

## The Trap of Clever Reactions

The biggest pitfall with `@reactive` is trying to be too clever. Three patterns consistently cause problems:

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

## Store-Level Reactions

Stores collect related observables, and you can react to derived properties on stores just like standalone observables:

```python
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)
    is_active = observable(True)

    user_summary = (name + age) >> (lambda n, a: f"{n}, {a}")
    should_display = is_active & (age >> (lambda a: a >= 18))

@reactive(UserStore.user_summary)
def sync_to_server(summary):
    api.post('/user/update', {'summary': summary})

@reactive(UserStore.should_display)
def toggle_profile_visibility(should_show):
    profile_element.visible = should_show

UserStore.name = "Bob"  # Triggers first reaction
UserStore.age = 31      # Triggers both reactions
UserStore.is_active = False  # Triggers second reaction only
```

The store becomes your functional core. The reactions watching it become your shell. This separation makes testing straightforward—test the store logic in isolation, mock the side effects in the reactions.

***

**The Big Picture:** Use `@reactive` sparingly. Most of your code should be pure derivations using `>>`, `+`, `&`, and `~`. Reactions appear only at the edges, where your application must interact with something external. The conditional operators let you express exactly when these interactions should happen without mixing conditions into your business logic. When you find yourself reaching for `@reactive`, pause and ask: "Is this really a side effect, or am I just deriving new state?" That question alone will guide you toward cleaner, more maintainable reactive systems.

::: fynx.reactive
