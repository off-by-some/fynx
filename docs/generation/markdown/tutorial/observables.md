# Observables: The Foundation of Reactivity

An observable is a value that changes over time and tells you when it changes. That's it. Everything else in FynX—every operator, every decorator, every pattern—builds on this simple idea.

Think of a regular Python variable:

```python
count = 0
count = 1
count = 2
```

When `count` changes, nothing happens. The value updates silently, and any code that depends on `count` has no idea. You have to manually check for changes, manually update dependent state, manually keep everything synchronized.

Now think of an observable:

```python
from fynx import observable

count = observable(0)
count.set(1)
count.set(2)
```

When an observable changes, it notifies anyone who's listening. Dependent computations recalculate automatically. UI updates happen without explicit instructions. The reactive graph maintains its own consistency.

This is the fundamental shift: from **telling your code when to update** to **declaring what should be true**. FynX handles the rest.

## Creating Your First Observable

Start by importing the `observable` function and giving it an initial value:

```python
from fynx import observable

# Create observables with initial values of any type
name = observable("Alice")
age = observable(30)
scores = observable([85, 92, 78])
user = observable({"id": 1, "active": True})
```

The initial value can be anything—strings, numbers, lists, dictionaries, custom objects, even `None`. FynX doesn't care about the type. It just wraps the value and makes it reactive.

## Reading Observable Values

To read what's inside an observable, use the `.value` property:

```python
name = observable("Alice")

print(name.value)  # "Alice"
print(age.value)   # 30
```

This looks like extra syntax compared to regular variables, and it is. But that `.value` access does something important: it registers that your code depends on this observable. When you read `.value` inside a reactive context (we'll get to those soon), FynX automatically tracks the dependency and ensures your code re-runs when the observable changes.

## Writing Observable Values

To change what's inside an observable, use the `.set()` method:

```python
name = observable("Alice")

name.set("Bob")
print(name.value)  # "Bob"

name.set("Charlie")
print(name.value)  # "Charlie"
```

Each call to `.set()` does two things:

1. Updates the internal value
2. Notifies all subscribers that the value changed

This notification is what makes observables reactive. Without it, they'd just be awkward wrappers around regular values.

## Subscribing to Changes

The real power of observables emerges when you subscribe to them. A subscription is a function that runs whenever the observable changes:

```python
name = observable("Alice")

def greet(new_name):
    print(f"Hello, {new_name}!")

# Subscribe to changes
name.subscribe(greet)

# Now changes trigger the subscriber
name.set("Bob")      # Prints: "Hello, Bob!"
name.set("Charlie")  # Prints: "Hello, Charlie!"
```

Your subscriber function receives the new value as its argument. The function runs immediately when you call `.set()`, after the value updates but before `.set()` returns.

You can subscribe multiple functions to the same observable:

```python
counter = observable(0)

counter.subscribe(lambda n: print(f"Count: {n}"))
counter.subscribe(lambda n: print(f"Double: {n * 2}"))
counter.subscribe(lambda n: print(f"Square: {n ** 2}"))

counter.set(5)
# Output:
# Count: 5
# Double: 10
# Square: 25
```

All subscribers receive the same value, and they run in the order you subscribed them.

## Unsubscribing

When you no longer need to listen to an observable, unsubscribe to prevent memory leaks:

```python
def logger(value):
    print(f"Value: {value}")

counter = observable(0)
counter.subscribe(logger)

counter.set(1)  # Prints: "Value: 1"

# Unsubscribe when done
counter.unsubscribe(logger)

counter.set(2)  # No output - logger no longer subscribed
```

You must pass the exact same function reference to `unsubscribe()` that you passed to `subscribe()`. This is why lambda functions can be tricky—you can't easily unsubscribe them later. For cleanup-critical code, use named functions.

## Transactions: Safe Reentrant Updates

Sometimes you need to update an observable from within one of its own subscribers. This creates a potential circular dependency—the observable is already notifying subscribers, and now you're trying to notify them again.

FynX prevents infinite loops with **circular dependency detection**:

```python
counter = observable(0)

def increment_on_change(value):
    # This would cause a circular dependency error
    counter.set(value + 1)  # ❌ RuntimeError: Circular dependency detected

counter.subscribe(increment_on_change)
counter.set(5)  # Raises RuntimeError
```

But what if you legitimately need to update an observable from within a subscriber? Use **transactions** for safe, controlled reentrant updates:

```python
counter = observable(0)

def increment_on_change(value):
    # Use transaction for safe reentrant updates
    with counter.transaction():
        counter.set(value + 1)  # ✅ Safe!

counter.subscribe(increment_on_change)
counter.set(5)  # Works correctly
```

### How Transactions Work

Transactions defer notifications until the transaction completes, then send a single notification with the final value:

```python
counter = observable(0)
notifications = []

def track_notifications(value):
    notifications.append(value)

counter.subscribe(track_notifications)

# Without transaction - multiple notifications
counter.set(1)
counter.set(2)
counter.set(3)
print(notifications)  # [1, 2, 3]

notifications.clear()

# With transaction - single notification
with counter.transaction():
    counter.set(1)
    counter.set(2)
    counter.set(3)
print(notifications)  # [3] - only the final value
```

### When to Use Transactions

Use transactions when you need to:

* **Update an observable from within its own subscriber** — Safe reentrant updates
* **Batch multiple changes** — Reduce notification overhead
* **Ensure atomic updates** — Prevent intermediate inconsistent states
* **Coordinate complex state changes** — Multiple observables updating together

```python
# Example: Form validation with batched updates
class FormStore(Store):
    name = observable("")
    email = observable("")
    is_valid = observable(False)

def validate_form():
    with FormStore.is_valid.transaction():
        # Multiple validation checks
        name_valid = len(FormStore.name.value) > 0
        email_valid = "@" in FormStore.email.value
        FormStore.is_valid.set(name_valid and email_valid)
        # Only one notification sent when transaction completes

# Subscribe to validation changes
FormStore.is_valid.subscribe(lambda valid: print(f"Form valid: {valid}"))

# Update form fields
FormStore.name = "Alice"  # Triggers validation
FormStore.email = "alice@example.com"  # Triggers validation
# Output: "Form valid: True" (only once, after both updates)
```

### Transaction Best Practices

* **Keep transactions short** — Long transactions can make debugging harder
* **Use for legitimate reentrancy** — Don't use transactions to work around design issues
* **Prefer declarative patterns** — Often, computed observables are cleaner than manual transactions
* **Test transaction behavior** — Ensure your batched updates work as expected

```python
# Good: Using transactions for legitimate coordination
def update_user_profile(user_data):
    with UserStore.transaction():  # If Store had transaction support
        UserStore.name = user_data['name']
        UserStore.email = user_data['email']
        UserStore.last_updated = datetime.now()

# Better: Using computed observables for derived state
class UserStore(Store):
    name = observable("")
    email = observable("")
    display_name = name >> (lambda n: n.title())
    is_complete = (name + email) >> (lambda n, e: bool(n and e))
```

## Observables vs. Regular Variables

Let's make the difference concrete. Here's state management without observables:

```python
# Traditional approach
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.total = 0

    def add_item(self, item):
        self.items.append(item)
        self.total = sum(item['price'] for item in self.items)
        self.update_ui()  # Manual synchronization
        self.save_to_storage()  # Manual synchronization
        self.notify_analytics()  # Manual synchronization
```

Every time you modify state, you must remember to update everything that depends on it. Miss one call and your application falls out of sync.

Here's the same thing with observables:

```python
# Reactive approach
items = observable([])
total = items >> (lambda item_list: sum(item['price'] for item in item_list))

items.subscribe(update_ui)
items.subscribe(save_to_storage)
items.subscribe(notify_analytics)

# Now just update the observable
items.set(items.value + [{'name': 'Widget', 'price': 10}])
# All three functions run automatically
# total recalculates automatically
```

You declare the relationships once. Changes propagate automatically. There's no way to forget a synchronization step because there are no synchronization steps—just state changes.

## What Observables Enable

Observables are containers, but they're programmable containers. They carry their dependencies with them. When you build reactive systems, you're constructing graphs where nodes (observables) automatically update when their upstream dependencies change.

Consider a simple example:

```python
base_price = observable(100)
quantity = observable(2)

# This creates a computed observable (we'll explore these deeply in the next section)
total = (base_price + quantity) >> (lambda price, qty: price * qty)

total.subscribe(lambda t: print(f"Total: ${t}"))

base_price.set(150)  # Prints: "Total: $300"
quantity.set(3)      # Prints: "Total: $450"
```

Notice what didn't happen: you never manually recalculated `total`. You never explicitly called the subscriber. The reactive graph did all that work for you. You just changed the base values and watched the effects cascade.

This is what observables enable: **declarative state management**. You describe what relationships should exist, and FynX ensures they hold.

## When to Use Observables

Observables shine in situations where:

* **Multiple things depend on the same state** — One change needs to update several downstream systems
* **State changes frequently** — User interactions, real-time data, animated values
* **Dependencies are complex** — Value A depends on B and C, which depend on D and E
* **You want to avoid manual synchronization** — Eliminating update code reduces bugs

Observables add overhead compared to plain variables. For simple scripts or one-off calculations, that overhead isn't worth it. But for interactive applications, data pipelines, or anything with non-trivial state management, observables pay for themselves quickly.

## What's Next

Observables are more than containers—they're nodes in a reactive graph. But standalone observables are just the beginning. The real power emerges when you learn to:

* **Transform observables** using the `>>` operator to create derived values that update automatically
* **Combine observables** using the `+` operator to work with multiple sources of data
* **Filter observables** using the `&` operator to apply conditional logic and control when data flows
* **Create logical OR conditions** using the `|` operator to combine boolean observables
* **Organize observables** into Stores for cleaner application architecture
* **Automate reactions** with decorators that eliminate subscription boilerplate

Each of these builds on the foundation you've just learned. Observables are simple, but their composition creates sophisticated reactive systems.

The insight to carry forward: **observables aren't just containers—they're nodes in a reactive graph**. When you change one node, effects ripple through the entire structure automatically. That's the power FynX gives you.
