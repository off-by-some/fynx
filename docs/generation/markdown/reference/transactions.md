# Transactions: Safe Reentrant Updates

Transactions provide a safe way to update observables from within their own subscribers, preventing circular dependencies while enabling sophisticated state coordination patterns.

## The Problem: Circular Dependencies

When an observable's subscriber tries to update the same observable, it creates a circular dependency:

```python
counter = observable(0)

def increment_on_change(value):
    # This causes a circular dependency
    counter.set(value + 1)  # ❌ RuntimeError: Circular dependency detected

counter.subscribe(increment_on_change)
counter.set(5)  # Raises RuntimeError
```

FynX detects these circular dependencies and prevents infinite loops by raising a `RuntimeError`. This protects your application from hanging or crashing due to recursive updates.

## The Solution: Transactions

Transactions defer notifications until the transaction completes, allowing safe reentrant updates:

```python
counter = observable(0)

def increment_on_change(value):
    # Use transaction for safe reentrant updates
    with counter.transaction():
        counter.set(value + 1)  # ✅ Safe!

counter.subscribe(increment_on_change)
counter.set(5)  # Works correctly
```

## How Transactions Work

### Deferred Notifications

Within a transaction, notifications are deferred until the transaction commits:

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

### Nested Transactions

Transactions can be nested safely. Only the outermost transaction triggers notifications:

```python
counter = observable(0)
notifications = []

def track_notifications(value):
    notifications.append(value)

counter.subscribe(track_notifications)

with counter.transaction():  # Outer transaction
    counter.set(1)
    with counter.transaction():  # Inner transaction
        counter.set(2)
        counter.set(3)
    counter.set(4)
# Only one notification: [4]
```

### Transaction Stack

FynX maintains a transaction stack per observable class. When the stack is empty, notifications are sent immediately. When the stack has transactions, notifications are deferred.

## Use Cases

### 1. Safe Reentrant Updates

The primary use case is updating an observable from within its own subscriber:

```python
class AutoCounter:
    def __init__(self):
        self.count = observable(0)
        self.count.subscribe(self._on_count_change)

    def _on_count_change(self, value):
        if value < 10:  # Auto-increment until 10
            with self.count.transaction():
                self.count.set(value + 1)

counter = AutoCounter()
counter.count.set(5)  # Will auto-increment to 10
```

### 2. Batched Updates

Reduce notification overhead by batching multiple changes:

```python
class FormStore(Store):
    name = observable("")
    email = observable("")
    is_valid = observable(False)

def update_form_data(name, email):
    with FormStore.is_valid.transaction():
        FormStore.name.set(name)
        FormStore.email.set(email)
        # Validate after both updates
        is_valid = bool(name and email and "@" in email)
        FormStore.is_valid.set(is_valid)
    # Only one validation notification sent

update_form_data("Alice", "alice@example.com")
```

### 3. Atomic State Changes

Ensure multiple observables are updated atomically:

```python
class UserProfile:
    def __init__(self):
        self.name = observable("")
        self.email = observable("")
        self.last_updated = observable(None)

    def update_profile(self, name, email):
        # Update all fields atomically
        with self.name.transaction():
            self.name.set(name)
            self.email.set(email)
            self.last_updated.set(datetime.now())
        # All subscribers notified after all updates complete

profile = UserProfile()
profile.update_profile("Alice", "alice@example.com")
```

### 4. Complex State Coordination

Coordinate updates across multiple observables:

```python
class ShoppingCart:
    def __init__(self):
        self.items = observable([])
        self.total = observable(0.0)
        self.item_count = observable(0)

    def add_item(self, item):
        with self.items.transaction():
            new_items = self.items.value + [item]
            self.items.set(new_items)
            self.total.set(sum(item['price'] for item in new_items))
            self.item_count.set(len(new_items))
        # All three observables updated atomically

cart = ShoppingCart()
cart.add_item({'name': 'Widget', 'price': 10.0})
```

## Best Practices

### 1. Keep Transactions Short

Long transactions can make debugging harder and reduce responsiveness:

```python
# Good: Short, focused transaction
def update_counter(value):
    with counter.transaction():
        counter.set(value + 1)

# Avoid: Long transaction with complex logic
def complex_update(value):
    with counter.transaction():
        # ... lots of complex logic ...
        counter.set(complex_calculation(value))
```

### 2. Use for Legitimate Reentrancy

Don't use transactions to work around design issues. Consider if computed observables would be cleaner:

```python
# Good: Using transactions for coordination
def sync_with_external_system(value):
    with local_state.transaction():
        local_state.set(value)
        external_api.update(value)

# Better: Using computed observables for derived state
class UserStore(Store):
    first_name = observable("")
    last_name = observable("")
    full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")
```

### 3. Test Transaction Behavior

Ensure your batched updates work as expected:

```python
def test_transaction_batching():
    counter = observable(0)
    notifications = []

    def track_notifications(value):
        notifications.append(value)

    counter.subscribe(track_notifications)

    # Test batching
    with counter.transaction():
        counter.set(1)
        counter.set(2)
        counter.set(3)

    assert notifications == [3]  # Only final value
    assert counter.value == 3
```

### 4. Prefer Declarative Patterns

Often, computed observables are cleaner than manual transactions:

```python
# Manual transaction approach
class ManualStore(Store):
    items = observable([])
    total = observable(0)

    def add_item(self, item):
        with self.items.transaction():
            new_items = self.items.value + [item]
            self.items.set(new_items)
            self.total.set(sum(item['price'] for item in new_items))

# Declarative computed approach
class ComputedStore(Store):
    items = observable([])
    total = items >> (lambda items: sum(item['price'] for item in items))

    def add_item(self, item):
        self.items.set(self.items.value + [item])
        # total updates automatically
```

## Error Handling

### Circular Dependency Detection

FynX automatically detects circular dependencies and raises a `RuntimeError`:

```python
counter = observable(0)

def bad_update(value):
    counter.set(value + 1)  # ❌ RuntimeError: Circular dependency detected

counter.subscribe(bad_update)
counter.set(5)  # Raises RuntimeError
```

### Transaction Error Recovery

If an exception occurs within a transaction, the transaction is automatically rolled back:

```python
counter = observable(0)

try:
    with counter.transaction():
        counter.set(1)
        raise ValueError("Something went wrong")
        counter.set(2)  # This won't execute
except ValueError:
    pass

print(counter.value)  # Still 0 - transaction was rolled back
```

## Performance Considerations

### Notification Batching

Transactions reduce notification overhead by batching updates:

```python
# Without transaction: 3 notifications
counter.set(1)  # Notification 1
counter.set(2)  # Notification 2
counter.set(3)  # Notification 3

# With transaction: 1 notification
with counter.transaction():
    counter.set(1)  # Deferred
    counter.set(2)  # Deferred
    counter.set(3)  # Deferred
# Single notification with value 3
```

### Memory Usage

Transactions use minimal memory overhead. The transaction stack is shared across all observables of the same class, and deferred notifications are stored per observable instance.

## Integration with Other FynX Features

### Computed Observables

Transactions work seamlessly with computed observables:

```python
base_price = observable(100)
quantity = observable(2)
total = (base_price + quantity) >> (lambda p, q: p * q)

# Batch updates to multiple dependencies
with base_price.transaction():
    base_price.set(150)
    quantity.set(3)
# total recalculates once with final values
```

### Conditional Observables

Transactions can be used with conditional observables for complex state coordination:

```python
is_valid = observable(False)
has_changes = observable(False)
should_save = is_valid & has_changes

def update_form_validity(valid):
    with is_valid.transaction():
        is_valid.set(valid)
        has_changes.set(True)
    # should_save updates once after both changes

update_form_validity(True)
```

### Reactive Decorators

Transactions work with reactive decorators:

```python
@reactive(counter)
def log_changes(value):
    if value > 5:
        with counter.transaction():
            counter.set(0)  # Reset counter safely

counter.set(10)  # Will reset to 0 without circular dependency
```

## Summary

Transactions provide a powerful mechanism for:

* **Safe reentrant updates** — Update observables from within their own subscribers
* **Batched notifications** — Reduce overhead by grouping multiple updates
* **Atomic state changes** — Ensure multiple observables update together
* **Complex state coordination** — Handle sophisticated update patterns

Use transactions when you need controlled reentrancy, but prefer computed observables for derived state. Keep transactions short and focused, and always test their behavior to ensure they work as expected.

The key insight: **transactions defer notifications until completion, enabling safe reentrant updates while maintaining FynX's reactive guarantees**.
