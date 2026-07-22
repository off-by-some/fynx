# Observables: The Foundation of Reactivity

An observable is a value that changes over time and tells you when it changes. That's it. Every operator, decorator, and pattern in FynX builds on that one idea.

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

All subscribers receive the same value. The execution order is not guaranteed and may vary between runs.

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

You must pass the exact same function reference to `unsubscribe()` that you passed to `subscribe()`, so a lambda passed inline can't be unsubscribed later - there's no reference to it. Use named functions for anything you'll need to clean up.

## Observables vs. Regular Variables

Let's make the difference concrete. Here's state management without observables:

```python
# Traditional approach
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.total = 0

    def add_item(self, item):
        self.items = self.items + [item]  # Immutable update
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
total = items.then(lambda item_list: sum(item['price'] for item in item_list))

# Or, to use fynx's syntactic sugar:
# total = items >> (lambda item_list: sum(item['price'] for item in item_list))

items.subscribe(update_ui)
items.subscribe(save_to_storage)
items.subscribe(notify_analytics)

# Now just update the observable
items.set(items.value + [{'name': 'Widget', 'price': 10}])
# All three functions run automatically
# total recalculates automatically
```

You declare the relationships once, and changes propagate automatically - there's no separate synchronization step to remember, because there isn't one.

## What Observables Enable

Observables are containers, but they're programmable containers. They carry their dependencies with them. When you build reactive systems, you're constructing graphs where nodes (observables) automatically update when their upstream dependencies change.

Consider a simple example:

```python
base_price = observable(100)
quantity = observable(2)

# This creates a computed observable (we'll explore these deeply in the next section)
total = base_price.alongside(quantity).then(lambda price, qty: price * qty)

# Or, to use fynx's syntactic sugar:
# total = (base_price + quantity) >> (lambda price, qty: price * qty)

total.subscribe(lambda t: print(f"Total: ${t}"))

base_price.set(150)  # Prints: "Total: $300"
quantity.set(3)      # Prints: "Total: $450"
```

`total` was never manually recalculated, and the subscriber was never explicitly called - both happened because `base_price` and `quantity` changed.

## When to Use Observables

Observables shine in situations where:

* **Multiple things depend on the same state** — One change needs to update several downstream systems
* **State changes frequently** — User interactions, real-time data, animated values
* **Dependencies are complex** — Value A depends on B and C, which depend on D and E
* **You want to avoid manual synchronization** — Eliminating update code reduces bugs

Observables add overhead compared to plain variables. For simple scripts or one-off calculations, that overhead isn't worth it. But for interactive applications, data pipelines, or anything with non-trivial state management, observables pay for themselves quickly.

## What's Next

Standalone observables are the starting point. From here you'll learn to:

* **[Transform observables](derived-observables.md)** using the `>>` operator to create derived values that update automatically
* **[Combine observables](derived-observables.md)** using the `+` operator to work with multiple sources of data
* **[Build boolean conditions](conditionals.md)** using the `&`, `|`, and `~` operators
* **[Gate observables](conditionals.md)** using the `@` operator to control when data flows
* **[Organize observables](stores.md)** into Stores for cleaner application architecture
* **[Automate reactions](using-reactive.md)** with decorators that eliminate subscription boilerplate

Each of these builds on what's here: an observable holds a value and notifies whoever's listening when it changes. Everything else in FynX is built out of that.
