# API Reference

This reference provides comprehensive documentation of FynX's public API. FynX is a reactive programming library that makes your application state respond automatically to changes—think of it as a spreadsheet for your code, where updating one cell automatically recalculates all the formulas that depend on it.

## A Mental Model for FynX

Before diving into the API details, it helps to understand FynX's core philosophy:

**Traditional programming is imperative**: You tell your code exactly when to update things. Change a variable here, update the UI there, recalculate this value over there. You're responsible for remembering all the connections.

**FynX is declarative**: You describe *relationships* between values, and FynX handles the updates automatically. Change a value once, and everything that depends on it updates correctly, in the right order, every time.

This mental shift—from managing updates to declaring relationships—is what makes thinking reactively so powerful.

## Your Learning Path

FynX's API is designed to be learned progressively, with each concept building on the previous one:

1. **Start with Observables** — Learn to create reactive values that notify subscribers when they change
2. **Organize with Stores** — Group related observables into cohesive units with clean APIs
3. **Filter Conditionally** — Control when reactions occur based on runtime conditions
4. **Automate with Decorators** — Eliminate boilerplate and express reactive relationships declaratively

Each section in this reference assumes you understand the previous sections. If you encounter an unfamiliar concept, backtrack to the earlier pages.

## Core Concepts

### Observables: Reactive Values

Observables are containers for values that change over time. Unlike regular variables, they automatically notify anyone who's interested when their values change.

**[Observable](observable.md)** — The foundation of FynX. Create observables with `observable(initial_value)`, read them with `.value`, write them with `.set(new_value)`. Every other FynX feature builds on this simple primitive.

**[ComputedObservable](computed-observable.md)** — Values that automatically recalculate when their dependencies change. Create them with the `>>` operator: `full_name = (first + last) >> (lambda f, l: f"{f} {l}")`. The `>>` operator transforms observables through functions, creating a new computed observable. Alternatively, use the `.then(func)` method on observables for the same result. FynX tracks dependencies automatically and ensures computed values always stay up-to-date.

**[MergedObservable](merged-observable.md)** — Combine multiple observables into a single reactive tuple using the `+` operator: `position = x + y + z`. When any source changes, subscribers receive all values as a tuple. This is the foundation for reactive relationships that depend on multiple values.

**[ConditionalObservable](conditional-observable.md)** — Observables that emit when conditions are satisfied. Create them with the `&` operator: `valid_submission = form_data & is_valid`. This enables sophisticated reactive logic without cluttering your code with conditional checks.

**[Observable Descriptors](observable-descriptors.md)** — The mechanism behind Store class attributes. When you write `name = observable("Alice")` in a Store class, you're creating a descriptor that provides clean property access without `.value` or `.set()`.

**[Observable Operators](observable-operators.md)** — The operators (`+`, `>>`, `&`, `~`) and methods (`.then()`, `.also()`) that let you compose observables into reactive pipelines. The `>>` operator is the primary way to transform observables, passing values through functions. Understanding these operators unlocks FynX's full expressive power.

### Stores: Organizing State

While standalone observables are useful for small scripts, real applications need structure. Stores group related observables and computed values into cohesive units.

**[Store & @observable](store.md)** — Create Store classes that encapsulate related state. Use `@observable` to make class attributes reactive, or use `observable()` as a class attribute descriptor. This gives you clean property access: `UserStore.name = "Alice"` instead of `user_name.set("Alice")`. Stores are where FynX really shines in application development.

### Decorators: Declarative Reactions

Decorators let you declare what should happen when observables change, without manually managing subscriptions.

**[@reactive](reactive-decorator.md)** — Run functions automatically when dependencies change. This is how you implement side effects—logging, UI updates, API calls—that should happen in response to state changes. The function runs immediately and again whenever any observable it reads changes. Use with conditional observables for event-driven reactions: `@reactive(condition & other_condition)`.

## API Quick Reference

### Creating Reactive State

```python
from fynx import observable, Store

# Standalone observables
count = observable(0)
name = observable("Alice")

# Store-based observables
class AppStore(Store):
    count = observable(0)
    name = observable("Alice")
```

### Reading and Writing

```python
# Standalone observables
current = count.value          # Read
count.set(current + 1)         # Write

# Store observables
current = AppStore.count       # Read
AppStore.count = current + 1   # Write
```

### Deriving Values

```python
# Using the >> operator (recommended)
doubled = count >> (lambda c: c * 2)
full_name = (first + last) >> (lambda f, l: f"{f} {l}")

# Using .then() method (alternative syntax)
doubled = count.then(lambda c: c * 2)
full_name = (first + last).then(lambda f, l: f"{f} {l}")
```

### Reacting to Changes

```python
# Manual subscription
count.subscribe(lambda val: print(f"Count: {val}"))

# Using @reactive decorator
@reactive(count)
def log_count(val):
    print(f"Count: {val}")

# Using @reactive with conditional observables for event-driven reactions
is_above_threshold = count >> (lambda c: c > 10)
@reactive(is_above_threshold)
def on_threshold(is_above):
    if is_above:
        print("Count exceeded 10!")
```

### Composing Observables

```python
# Merge multiple sources
position = x + y + z

# Transform values with >> operator
doubled = count >> (lambda c: c * 2)

# Or use .then() method
doubled = count.then(lambda c: c * 2)

# Filter conditionally
should_save = has_changes & is_valid

# Negate conditions
is_idle = ~is_busy
```

## Complete Example: Putting It All Together

Here's how these concepts work together in a realistic scenario:

```python
from fynx import Store, observable, reactive

class ShoppingCartStore(Store):
    # Basic reactive state
    items = observable([])
    discount_code = observable(None)

# Computed values using >> operator
subtotal = ShoppingCartStore.items >> (
    lambda items: sum(item['price'] * item['quantity'] for item in items)
)

discount_amount = (ShoppingCartStore.items + ShoppingCartStore.discount_code) >> (
    lambda items, code: sum(item['price'] * item['quantity'] for item in items) * 0.20
    if code == "SAVE20" else 0.0
)

total = (subtotal + discount_amount) >> (
    lambda sub, disc: sub - disc
)

# Conditional observable for checkout eligibility
has_items = ShoppingCartStore.items >> (lambda i: len(i) > 0)
total_positive = total >> (lambda t: t > 0)
can_checkout = has_items & total_positive

# React to changes automatically
@reactive(total)
def update_ui_total(t):
    print(f"💰 New total: ${t:.2f}")

# React to checkout eligibility using conditional observables
@reactive(can_checkout)
def enable_checkout_button(can_checkout_val):
    if can_checkout_val:
        print("✅ Checkout button enabled")

# Use the store
ShoppingCartStore.items = [
    {'name': 'Widget', 'price': 10.00, 'quantity': 2}
]
# Output: 💰 New total: $20.00
# Output: ✅ Checkout button enabled

ShoppingCartStore.discount_code = "SAVE20"
# Output: 💰 New total: $16.00
```

## Documentation Conventions

Throughout this reference, we follow consistent patterns:

* **Type signatures** use Python type hints for clarity and enable IDE autocomplete
* **Examples progress from simple to complex** within each page
* **Notes highlight gotchas** that trip up newcomers
* **Performance tips** appear when relevant to optimization decisions
* **See also links** connect related concepts and alternative approaches

## Navigating This Reference

### New to FynX?

Read in order: [Observable](observable.md) → [Store](store.md) → [@reactive](reactive-decorator.md) → [ConditionalObservable](conditional-observable.md)

### Building an application?

Focus on: [Store](store.md), [Observable Operators](observable-operators.md) (especially `>>`), [@reactive](reactive-decorator.md)

### Need complex state logic?

Dive into: [Observable Operators](observable-operators.md), [ConditionalObservable](conditional-observable.md), [@reactive](reactive-decorator.md)

### Performance optimization?

See: [ComputedObservable](computed-observable.md) for memoization, [Observable](observable.md) for subscription management

### Curious about implementation?

Explore: [Observable Descriptors](observable-descriptors.md) to understand how the magic works

***

For conceptual introductions and tutorials, return to the [main documentation](../../index.md).
