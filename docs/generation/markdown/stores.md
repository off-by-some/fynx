# Stores: Organizing Reactive State

Observables form the foundation of reactivity, but scattered reactive values create chaos. As applications grow, you need structure—containers that group related state, define behaviors, and provide clean boundaries.

Stores solve these problems by giving structure to your reactive state.

## What is a Store?

A Store is a class that groups related observables together. That's the essence. Everything else—the computed properties, the methods, the reactive behaviors—builds on this simple organizational principle.

Think of a Store as a namespace for related state. If you're building a shopping cart, all the cart-related observables live in `CartStore`. If you're managing user authentication, all auth state lives in `AuthStore`. Each Store becomes a clear, testable boundary in your application.

Here's the simplest possible Store:

```python
from fynx import Store, observable

class CounterStore(Store):
    count = observable(0)
```

That's it. You've created a Store with one reactive attribute. The magic is in what this gives you:

```python
# Read the value
current = CounterStore.count

# Write the value
CounterStore.count = 5

# The value is an observable under the hood
CounterStore.count.subscribe(lambda c: print(f"Count: {c}"))

CounterStore.count = 10  # Prints: "Count: 10"
```

Notice the asymmetry: you read with direct access (`CounterStore.count`), but the value is still an observable. You can still subscribe to it, transform it with `>>`, merge it with `|`. The Store class uses Python descriptors to give you clean syntax while preserving all of observable's power.

## Why Stores Matter

Before Stores, you might write code like this:

```python
# Scattered observables
user_first_name = observable("Alice")
user_last_name = observable("Smith")
user_age = observable(30)
user_email = observable("alice@example.com")
is_authenticated = observable(False)

# Where does this logic live?
def update_user_profile(first, last, age, email):
    user_first_name.set(first)
    user_last_name.set(last)
    user_age.set(age)
    user_email.set(email)
```

This works, but it doesn't scale. Which observables relate to each other? Where should validation logic go? How do you reset user state? How do you test user-related functionality without affecting other parts of your application?

With Stores:

```python
class UserStore(Store):
    first_name = observable("Alice")
    last_name = observable("Smith")
    age = observable(30)
    email = observable("alice@example.com")
    is_authenticated = observable(False)

    @classmethod
    def update_profile(cls, first, last, age, email):
        cls.first_name = first
        cls.last_name = last
        cls.age = age
        cls.email = email

    @classmethod
    def logout(cls):
        cls.is_authenticated = False
        cls.first_name = ""
        cls.last_name = ""
        cls.email = ""
```

Now everything about users lives in one place. The boundaries are clear. Testing is straightforward. Other parts of your application import `UserStore` and interact with it through its public methods.

## Store Attributes: Observable Descriptors

When you write `count = observable(0)` inside a Store class, you're creating an observable descriptor. This descriptor gives you convenient syntax:

```python
class MyStore(Store):
    value = observable(100)

# These are equivalent:
print(MyStore.value)         # Direct access
print(MyStore.value.value)   # Explicit .value access

# Writing is clean:
MyStore.value = 200           # Direct assignment

# But you can still use it as an observable:
MyStore.value.subscribe(lambda v: print(v))
doubled = MyStore.value >> (lambda v: v * 2)
```

The descriptor pattern means you don't need `.value` and `.set()` for Store attributes—just read and write naturally. But the underlying observable is still there, ready for transformations and subscriptions.

**Important:** This clean syntax only works for Store class attributes. Standalone observables (not in a Store) still require `.value` and `.set()`:

```python
# Standalone observable
counter = observable(0)
print(counter.value)  # Must use .value
counter.set(5)        # Must use .set()

# Store observable
class CounterStore(Store):
    counter = observable(0)

print(CounterStore.counter)  # No .value needed
CounterStore.counter = 5     # No .set() needed
```

## Adding Computed Values

The real power of Stores emerges when you add derived state using the `>>` operator:

```python
class CartStore(Store):
    items = observable([])
    tax_rate = observable(0.08)

    # Computed: recalculates when items changes
    item_count = items >> (lambda items: len(items))

    # Computed: recalculates when items changes
    subtotal = items >> (
        lambda items: sum(item['price'] * item['quantity'] for item in items)
    )
```

These computed values update automatically. When you change `CartStore.items`, both `item_count` and `subtotal` recalculate. But they only recalculate when you actually access them—this lazy evaluation means computed values have zero cost until you need them.

```python
CartStore.items = [
    {'name': 'Widget', 'price': 10, 'quantity': 2},
    {'name': 'Gadget', 'price': 15, 'quantity': 1}
]

print(CartStore.item_count)  # 2 (computes now)
print(CartStore.subtotal)    # 35.0 (computes now)

# Access again without changes
print(CartStore.item_count)  # 2 (returns cached value)
```

Computed values memoize their results. After the first access, they return the cached value until their dependencies change.

## Combining Multiple Observables

Most computed values depend on more than one observable. Use the `|` operator to merge observables:

```python
class CartStore(Store):
    items = observable([])
    tax_rate = observable(0.08)

    subtotal = items >> (
        lambda items: sum(item['price'] * item['quantity'] for item in items)
    )

    # Merge subtotal and tax_rate
    tax_amount = (subtotal | tax_rate) >> (
        lambda sub, rate: sub * rate
    )

    # Merge subtotal and tax_amount
    total = (subtotal | tax_amount) >> (
        lambda sub, tax: sub + tax
    )
```

The `|` operator creates a merged observable that emits a tuple. When you transform it with `>>`, the function receives one argument per observable:

```python
CartStore.items = [{'name': 'Widget', 'price': 20, 'quantity': 1}]

print(CartStore.subtotal)     # 20.0
print(CartStore.tax_amount)   # 1.6
print(CartStore.total)        # 21.6

CartStore.tax_rate = 0.10
print(CartStore.tax_amount)   # 2.0 (recalculated)
print(CartStore.total)        # 22.0 (recalculated)
```

Any change to a merged observable triggers recomputation. This makes `|` perfect for values that need to coordinate multiple pieces of state.

## Methods: Encapsulating State Changes

Stores become truly useful when they encapsulate the logic for modifying their own state:

```python
class CartStore(Store):
    items = observable([])

    @classmethod
    def add_item(cls, name, price, quantity=1):
        """Add an item to the cart or update quantity if it exists."""
        current_items = cls.items

        # Find existing item
        existing = next((item for item in current_items if item['name'] == name), None)

        if existing:
            # Update quantity
            cls.items = [
                {**item, 'quantity': item['quantity'] + quantity}
                if item['name'] == name else item
                for item in current_items
            ]
        else:
            # Add new item
            cls.items = current_items + [{'name': name, 'price': price, 'quantity': quantity}]

    @classmethod
    def remove_item(cls, name):
        """Remove an item from the cart."""
        cls.items = [item for item in cls.items if item['name'] != name]

    @classmethod
    def clear(cls):
        """Remove all items."""
        cls.items = []
```

Now cart manipulation is clean and explicit:

```python
CartStore.add_item('Widget', 10.0, 2)
CartStore.add_item('Gadget', 15.0)
print(CartStore.item_count)  # 2

CartStore.remove_item('Widget')
print(CartStore.item_count)  # 1

CartStore.clear()
print(CartStore.item_count)  # 0
```

Methods define your Store's public API. Users don't manipulate observables directly—they call methods that express intent. This encapsulation makes your code more maintainable and testable.

## A Critical Pattern: Immutable Updates

Notice the pattern in the methods above—we never mutate values in place:

```python
# Wrong: Mutation doesn't trigger reactivity
cls.items.append(new_item)

# Right: Create new list
cls.items = cls.items + [new_item]
```

FynX detects changes through assignment. When you mutate an observable's value in place, nothing triggers because from FynX's perspective, the reference hasn't changed. Always create new values:

```python
# Lists: Create new list
cls.items = cls.items + [new_item]
cls.items = [item for item in cls.items if condition]

# Dicts: Create new dict
cls.user = {**cls.user, 'name': 'New Name'}

# Nested structures: Reconstruct the path
cls.items = [
    {**item, 'quantity': item['quantity'] + 1} if item['id'] == target_id else item
    for item in cls.items
]
```

This immutable update pattern is crucial. It ensures reactivity works correctly and makes your state changes predictable.

## Chaining Computed Values

Computed values can depend on other computed values, creating transformation pipelines:

```python
class AnalyticsStore(Store):
    values = observable([10, 20, 30, 40, 50])

    # Level 1: Basic stats
    count = values >> (lambda v: len(v))
    total = values >> (lambda v: sum(v))

    # Level 2: Depends on count and total
    mean = (total | count) >> (
        lambda t, c: t / c if c > 0 else 0
    )

    # Level 3: Depends on values and mean
    variance = (values | mean | count) >> (
        lambda vals, avg, n: (
            sum((x - avg) ** 2 for x in vals) / (n - 1) if n > 1 else 0
        )
    )

    # Level 4: Depends on variance
    std_dev = variance >> (lambda v: v ** 0.5)
```

When `values` changes, FynX propagates updates through the entire chain in the correct order. Each level recalculates only if its dependencies actually changed:

```python
print(f"Mean: {AnalyticsStore.mean:.2f}")      # 30.00
print(f"Std Dev: {AnalyticsStore.std_dev:.2f}") # 15.81

AnalyticsStore.values = [5, 10, 15, 20, 25]
print(f"Mean: {AnalyticsStore.mean:.2f}")      # 15.00
print(f"Std Dev: {AnalyticsStore.std_dev:.2f}") # 7.91
```

This chaining pattern lets you build complex derived state from simple transformations. Each step is testable and easy to understand.

## Practical Example: User Profile Store

Let's build a realistic Store that demonstrates all these concepts:

```python
from fynx import Store, observable

class UserProfileStore(Store):
    # Basic observables
    first_name = observable("")
    last_name = observable("")
    email = observable("")
    age = observable(0)
    is_premium = observable(False)

    # Computed: full name
    full_name = (first_name | last_name) >> (
        lambda first, last: f"{first} {last}".strip()
    )

    # Computed: display name (falls back if no name)
    display_name = full_name >> (
        lambda name: name if name else "Anonymous User"
    )

    # Computed: email validation
    is_email_valid = email >> (
        lambda e: '@' in e and '.' in e.split('@')[-1] if e else False
    )

    # Computed: age validation
    is_adult = age >> (lambda a: a >= 18)

    # Computed: profile completeness
    is_complete = (first_name | last_name | email | is_email_valid) >> (
        lambda first, last, email_addr, email_valid:
            bool(first and last and email_addr and email_valid)
    )

    # Computed: user tier
    user_tier = (is_premium | is_complete) >> (
        lambda premium, complete: (
            "Premium" if premium else
            "Complete" if complete else
            "Basic"
        )
    )

    @classmethod
    def update_name(cls, first, last):
        """Update the user's name."""
        cls.first_name = first.strip()
        cls.last_name = last.strip()

    @classmethod
    def update_email(cls, email):
        """Update the user's email."""
        cls.email = email.strip().lower()

    @classmethod
    def set_age(cls, age):
        """Update the user's age."""
        if age >= 0:
            cls.age = age

    @classmethod
    def upgrade_to_premium(cls):
        """Upgrade the user to premium status."""
        cls.is_premium = True

    @classmethod
    def reset(cls):
        """Reset all profile data."""
        cls.first_name = ""
        cls.last_name = ""
        cls.email = ""
        cls.age = 0
        cls.is_premium = False
```

Usage demonstrates how everything updates automatically:

```python
# Initial state
print(UserProfileStore.display_name)  # "Anonymous User"
print(UserProfileStore.user_tier)     # "Basic"

# Update name
UserProfileStore.update_name("Alice", "Smith")
print(UserProfileStore.display_name)  # "Alice Smith"
print(UserProfileStore.full_name)     # "Alice Smith"

# Update email
UserProfileStore.update_email("alice@example.com")
print(UserProfileStore.is_email_valid)  # True

# Set age
UserProfileStore.set_age(25)
print(UserProfileStore.is_adult)      # True
print(UserProfileStore.is_complete)   # True
print(UserProfileStore.user_tier)     # "Complete"

# Upgrade
UserProfileStore.upgrade_to_premium()
print(UserProfileStore.user_tier)     # "Premium"
```

Every computed value updates automatically when its dependencies change. You never write synchronization code—just modify observables and watch the effects cascade.

## Cross-Store Dependencies

Stores can reference observables from other Stores, enabling modular architecture:

```python
class ThemeStore(Store):
    mode = observable("light")  # "light" or "dark"
    font_size = observable(16)

class UIStore(Store):
    sidebar_open = observable(True)

    # Depends on ThemeStore
    background_color = ThemeStore.mode >> (
        lambda mode: "#ffffff" if mode == "light" else "#1a1a1a"
    )

    text_color = ThemeStore.mode >> (
        lambda mode: "#000000" if mode == "light" else "#ffffff"
    )

    # Depends on multiple observables from ThemeStore
    css_vars = (ThemeStore.mode | ThemeStore.font_size) >> (
        lambda mode, size: {
            '--bg': "#ffffff" if mode == "light" else "#1a1a1a",
            '--text': "#000000" if mode == "light" else "#ffffff",
            '--font-size': f"{size}px"
        }
    )
```

This pattern keeps Stores focused while allowing coordination:

```python
ThemeStore.mode = "dark"
print(UIStore.background_color)  # "#1a1a1a"
print(UIStore.text_color)        # "#ffffff"

ThemeStore.font_size = 18
print(UIStore.css_vars['--font-size'])  # "18px"
```

Each Store maintains its own domain, but computed values can reach across Store boundaries to create relationships.

## When to Use Stores

Use Stores when you have:

**Related state that belongs together:**
```python
# Good: Cart-related state in CartStore
class CartStore(Store):
    items = observable([])
    discount_code = observable(None)
    shipping_address = observable(None)
```

**State that needs derived values:**
```python
# Good: Computed values with their source state
class FormStore(Store):
    email = observable("")
    password = observable("")

    email_valid = email >> (lambda e: '@' in e)
    password_valid = password >> (lambda p: len(p) >= 8)
    form_valid = (email_valid | password_valid) >> (lambda e, p: e and p)
```

**State that needs encapsulated modification:**
```python
# Good: Methods that maintain invariants
class AccountStore(Store):
    balance = observable(0)

    @classmethod
    def deposit(cls, amount):
        if amount > 0:
            cls.balance = cls.balance + amount

    @classmethod
    def withdraw(cls, amount):
        if 0 < amount <= cls.balance:
            cls.balance = cls.balance - amount
```

Don't use Stores for truly independent, single-purpose observables:

```python
# Overkill: Just use a standalone observable
class IsLoadingStore(Store):
    value = observable(False)

# Better:
is_loading = observable(False)
```

## Store Inheritance: Clean State Isolation

Store classes support inheritance, but with important nuances for state management:

```python
class BaseStore(Store):
    count = observable(0)
    name = observable("Base")

class ChildStore(BaseStore):
    pass  # Inherits count and name observables

# Each class gets completely independent state
BaseStore.count = 5
ChildStore.count = 10

print(BaseStore.count)   # 5
print(ChildStore.count)  # 10 (completely separate)
```

**Key Behavior:** Unlike standard Python inheritance where child classes share parent attributes, Store inheritance creates separate observable instances for each class. This ensures clean state isolation:

- `BaseStore.count` and `ChildStore.count` are completely independent
- Changes to one don't affect the other
- Each class maintains its own reactive state

**Explicit Overrides:** You can still override inherited observables:

```python
class CustomStore(BaseStore):
    count = observable(100)  # Completely replaces parent's count
    name = observable("Custom")  # Completely replaces parent's name

print(CustomStore.count)  # 100 (not 0)
print(BaseStore.count)    # 5 (unchanged)
```

Store inheritance prioritizes predictability and state isolation. Since Stores are typically global singletons, shared state through inheritance could lead to unexpected coupling. Each Store class gets its own clean state namespace.

**Best Practice:** Use inheritance for shared behavior (methods, computed properties), but define separate observables for each Store class that needs independent state.

## Best Practices

**1. Keep Stores focused on a single domain**

Each Store should represent one cohesive area of your application:

```python
# Good: Focused domains
class AuthStore(Store): ...
class CartStore(Store): ...
class UIStore(Store): ...

# Bad: Everything in one Store
class AppStore(Store):
    user = observable(None)
    cart_items = observable([])
    modal_open = observable(False)
    ...  # 50 more observables
```

**2. Use class methods for state modifications**

Encapsulate how state changes:

```python
# Good: Clear API
@classmethod
def add_item(cls, item):
    cls.items = cls.items + [item]

# Bad: Direct manipulation everywhere
SomeStore.items = SomeStore.items + [item]
```

**3. Always create new values, never mutate**

```python
# Good: New list
cls.items = cls.items + [new_item]
cls.items = [item for item in cls.items if item['id'] != id]

# Bad: Mutation (won't trigger reactivity)
cls.items.append(new_item)
cls.items.remove(item)
```

**4. Handle edge cases in computed values**

Computed values should be defensive:

```python
# Good: Handles empty list
average = values >> (
    lambda vals: sum(vals) / len(vals) if len(vals) > 0 else 0
)

# Good: Handles None
user_name = user >> (
    lambda u: u['name'] if u and 'name' in u else "Guest"
)
```

**5. Name computed values clearly**

Use names that indicate derivation:

```python
# Good: Clear that these are derived
is_valid = email >> (lambda e: '@' in e)
item_count = items >> len
total_price = items >> (lambda items: sum(item['price'] for item in items))

# Less clear:
valid = email >> (lambda e: '@' in e)
count = items >> len
price = items >> (lambda items: sum(item['price'] for item in items))
```

## Summary

Stores organize your reactive state into cohesive, testable units. They combine observables, computed values, and methods into structures that represent distinct domains of your application.

Key concepts:

- **Stores group related observables** — Keep state that belongs together in the same Store
- **Observable descriptors enable clean syntax** — Read and write Store attributes naturally
- **The `>>` operator creates computed values** — Derived state updates automatically
- **The `|` operator merges observables** — Combine multiple sources for multi-input computations
- **Always create new values** — Never mutate observable contents in place
- **Methods encapsulate state changes** — Define clear APIs for modifying state
- **Stores can depend on other Stores** — Build modular applications with cross-Store relationships

With Stores, you can build reactive applications that scale from simple counters to complex, multi-domain state management systems. The key is organization: each Store owns its domain, exposes a clean API, and lets FynX handle all the synchronization automatically.
