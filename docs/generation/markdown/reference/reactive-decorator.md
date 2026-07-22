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
# Prints immediately: "Count: 0"

count.set(5)   # Prints: "Count: 5"
count.set(10)  # Prints: "Count: 10"
```

The function runs automatically whenever `count` changes. You declare what should happen when data changes, and the framework handles the timing. Notice that decorating `log_count` runs it immediately with the current value (`0`) - `@reactive` always fires eagerly when it can. See [A Crucial Detail](#a-crucial-detail-initial-state-and-change-semantics) below for the exact rules.

## The Commitment: What You Gain and What You Give Up

Once you decorate a function with `@reactive`, you're making a commitment. The function becomes automatic—it runs when its dependencies change. In exchange, you lose the ability to call it manually:

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")
# Prints immediately with the current value of count

log_count(10)  # Raises fynx.reactive.ReactiveFunctionWasCalled exception
```

This isn't an arbitrary restriction. It's protecting you from confusion. If you could call `log_count()` manually *and* have it trigger automatically, which version of the value is authoritative? The manual call or the reactive update? The framework eliminates this ambiguity by enforcing one mode at a time.

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

## A Crucial Detail: Initial State and Change Semantics

`@reactive` is eager whenever it can be, so it fires immediately on decoration and then again on every later qualifying change:

- **Store targets** (`@reactive(SomeStore)`) fire immediately with a `StoreSnapshot` of the store's current state.
- **Ordinary observable, computed, or merged targets** fire immediately with their current value.
- **Conditional targets** (built with `&`) fire immediately only if the gate is *already active* at decoration time - if it's closed, the function waits until the gate opens.

Then, in every case, the function runs again on each later change that qualifies (a value change for ordinary/computed/merged targets; the gate opening or its passed-through value changing for conditional targets).

```python
ready = observable(True)  # Already true

@reactive(ready)
def on_ready(value):
    print(f"Ready: {value}")
# Prints immediately: "Ready: True"

ready.set(False)  # Prints: "Ready: False"
ready.set(True)   # Prints: "Ready: True"
```

This eager-first-run behavior is useful in practice: it means attaching a reaction is enough to bring an external system (a UI, a log, a cache) into sync with the current state - you don't need a separate "initialize once" call before the reactive updates take over.

## Conditional Reactions: Gating, Not Boolean Logic

FynX gives you operators for building conditional reactions, but it's important to get their meaning right up front:

* `&` **gates** a value by a condition - `data & condition` means "emit `data`'s value while `condition` is true." It is *not* boolean AND, and the two sides aren't interchangeable: `data & is_ready` gates `data` by `is_ready`, while `is_ready & data` gates `is_ready` by `data`.
* `+` **combines** observables into a tuple so a transform can read several of them at once (`(a + b) >> f`). It is *not* boolean OR.
* `|` is logical OR, and `~` is logical NOT. Both produce plain boolean observables that notify on every value change, not gates.

Because `&` is a gate rather than an AND, chaining it (`a & b & c`) only works cleanly when you want "pass through `a`'s value while `b` and `c` both hold" - a genuine gating relationship. Reach for `&` when you want to suppress a side effect while some condition doesn't hold:

```python
is_logged_in = observable(True)
is_verified = observable(True)

# Emits is_logged_in's value only while is_verified is true.
gated = is_logged_in & is_verified

@reactive(gated)
def show_gated_value(value):
    print(f"Gated value: {value}")
# Prints immediately since the gate starts open: "Gated value: True"

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

Notice `verified.set(False)` and `logged_in.set(True)` print nothing at all - the gate stays closed across both, since closing (or staying closed) never delivers a value. This is the asymmetric, stateful nature of a gate, and it's easy to mistake for boolean AND if you don't watch closely.

When what you actually want is a plain boolean that updates on every relevant change - for a UI flag, a button's enabled state, or any "total" condition - build it with `+` and `>>` instead of `&`:

```python
is_logged_in = observable(True)
is_verified = observable(True)

# A total boolean: recomputes (and notifies) on every actual change to the
# logical AND of the two inputs, with no gating quirks.
both_ready = (is_logged_in + is_verified) >> (
    lambda logged_in, verified: logged_in and verified
)

@reactive(both_ready)
def on_both_ready(ready):
    print(f"Both ready: {ready}")
# Prints immediately: "Both ready: True"
```

The same `+` / `>>` pattern scales to more complex boolean expressions, including ones that mix AND/OR/NOT:

```python
is_logged_in = observable(True)
has_data = observable(True)
is_loading = observable(True)
should_sync = observable(False)

# Ready when logged in AND has data AND (NOT loading OR should_sync)
ready_to_sync = (is_logged_in + has_data + is_loading + should_sync) >> (
    lambda logged_in, data, loading, sync: logged_in and data and (not loading or sync)
)

@reactive(ready_to_sync)
def sync_to_server(should_run):
    if should_run:
        perform_sync()
```

As a rule of thumb: use `&` to gate a value or a side effect ("only do this while X holds"); use `+` with `>>` (optionally combined with `|` / `~`) whenever you need a real boolean expression that should notify on every change to its truth value.

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
# Prints immediately: "Display: Alice (30 years old)"

name.set("Bob")  # Triggers with "Bob (30 years old)"
age.set(31)      # Triggers with "Bob (31 years old)"
```

Notice the pattern: derive first, react second. The `+` operator here isn't doing boolean OR—it's combining observables into a tuple-like stream. The `>>` operator then transforms that stream. Only after you've created a derived observable do you attach the reaction.

## The Core Insight: Where @reactive Belongs

Here's the fundamental principle that makes reactive systems maintainable: **`@reactive` is for side effects, not for deriving state.**

When you're tempted to use `@reactive`, ask yourself: "Am I computing a new value from existing data, or am I sending information outside my application?" If you're computing, you want `>>` or `+` operators. If you're communicating with the outside world, you want `@reactive`.

This distinction creates what we call the "functional core, reactive shell" pattern. Your core is pure transformations—testable, predictable, composable. A transform should only use the values it receives as arguments; combine extra observables first with `+` / `.alongside()`. Your shell is reactions—the unavoidable side effects that make your application actually do something.

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

**Self-mutation.** When a reaction modifies the very thing it's watching, you'd expect an infinite feedback cycle:

```python
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)  # Modifying the observable this reaction watches
```

FynX doesn't actually let this loop forever. The decoration itself runs once immediately (`0 → 1`) before the subscription is even registered, so that first call succeeds quietly. But once the reaction is subscribed, any *later* external change is a different story:

```python
count.set(5)
# Raises RuntimeError: Circular dependency detected in reactive computation!
# FynX detects that increment_forever is trying to modify `count` while
# running in response to a `count` change, and refuses rather than looping.
```

So this pattern doesn't silently run away - it fails loudly and immediately the first time it would actually recurse. That said, it's still a trap worth avoiding: relying on the framework to catch a self-mutation is worse than not writing one, and the failure can be harder to spot when the dependency is indirect (reaction A changes something that reaction B watches, and B changes what A watches).

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

# Gates `user`'s value by has_permission and is_online: the function only
# runs while both conditions hold, and receives user's current value
# (which may still be None if no one has logged in yet).
@reactive(user & has_permission & is_online)
def sync_sensitive_data(current_user):
    if current_user is not None:
        api.sync_user_data(current_user.id)

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

    # should_display is a total boolean UI flag, so it's built with `+` / `>>`
    # rather than `&` - it should notify on every real change to the value,
    # not just when a gate opens.
    is_adult = age >> (lambda a: a >= 18)
    should_display = (is_active + is_adult) >> (lambda active, adult: active and adult)

@reactive(UserStore.user_summary)
def sync_to_server(summary):
    api.post('/user/update', {'summary': summary})
# Fires immediately: sync_to_server("Alice, 30")

@reactive(UserStore.should_display)
def toggle_profile_visibility(should_show):
    profile_element.visible = should_show
# Fires immediately: toggle_profile_visibility(True)

UserStore.name = "Bob"  # Triggers first reaction (user_summary changed)
UserStore.age = 31      # Triggers first reaction only (is_adult is still True, so should_display is unchanged)
UserStore.is_active = False  # Triggers second reaction only (should_display flips to False)
```

The store becomes your functional core. The reactions watching it become your shell. This separation makes testing straightforward—test the store logic in isolation, mock the side effects in the reactions.

***

**The Big Picture:** Use `@reactive` sparingly. Most of your code should be pure derivations using `>>` and `+` (plus `|` / `~` for boolean logic), with `&` reserved for gating a value or a side effect rather than computing AND. Reactions appear only at the edges, where your application must interact with something external. When you find yourself reaching for `@reactive`, pause and ask: "Is this really a side effect, or am I just deriving new state?" That question alone will guide you toward cleaner, more maintainable reactive systems.

::: fynx.reactive
