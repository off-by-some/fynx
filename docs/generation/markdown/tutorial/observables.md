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
