"""
FynX Computed - Computed Observable Utilities
============================================

This module provides the `computed` function for creating derived observables
whose values are automatically calculated from other observables. Computed values
enable you to create reactive properties that depend on other reactive values,
automatically updating whenever their dependencies change.

What are Computed Values?
-------------------------

Computed values are read-only observables that derive their value from other observables.
They automatically recalculate when their dependencies change, providing a way to create
derived state without manual synchronization.

Computed values are essential for:
- **Derived State**: Properties that depend on other properties
- **Data Transformation**: Converting or formatting data reactively
- **Business Logic**: Calculated values based on raw data
- **Performance**: Avoiding redundant computations through memoization

Key Characteristics
-------------------

- **Read-only**: Computed values cannot be set directly
- **Lazy Evaluation**: Only recalculate when accessed and dependencies changed
- **Automatic Dependencies**: Framework tracks what observables are accessed
- **Pure Functions**: Computation functions should be pure (no side effects)
- **Memoization**: Results are cached until dependencies change

Basic Usage
-----------

```python
from fynx import observable, computed

# Create base observables
price = observable(10.0)
quantity = observable(5)

# Create computed value
total = computed(lambda p, q: p * q, price | quantity)
print(total.value)  # 50.0

# Changes propagate automatically
price.set(12.0)
print(total.value)  # 60.0 (automatically recalculated)
```

Advanced Patterns
-----------------

### Single Observable Computations

```python
counter = observable(5)
doubled = computed(lambda x: x * 2, counter)
print(doubled.value)  # 10

counter.set(7)
print(doubled.value)  # 14
```

### Complex Computations

```python
# Multiple observables
width = observable(10)
height = observable(20)
depth = observable(5)

# Computed volume
volume = computed(lambda w, h, d: w * h * d, width | height | depth)
print(volume.value)  # 1000

# Computed surface area
surface_area = computed(
    lambda w, h, d: 2 * (w*h + h*d + d*w),
    width | height | depth
)
print(surface_area.value)  # 400
```

### Chaining Computations

```python
base_price = observable(100)
tax_rate = observable(0.08)

# First computation
subtotal = computed(lambda price: price * 1.1, base_price)  # 10% markup

# Second computation based on first
tax = computed(lambda subtotal: subtotal * tax_rate.value, subtotal)

# Final computation
total = computed(lambda subtotal: subtotal + tax.value, subtotal)

print(f"Base: ${base_price.value}, Total: ${total.value}")
# Base: $100, Total: $118.8
```

### Dictionary/Object Computations

```python
user = observable({"name": "Alice", "age": 30})

# Computed display name
display_name = computed(
    lambda u: f"{u['name']} ({u['age']} years old)",
    user
)
print(display_name.value)  # "Alice (30 years old)"

user.set({"name": "Bob", "age": 25})
print(display_name.value)  # "Bob (25 years old)"
```

Store Integration
-----------------

Computed values work seamlessly with Stores:

```python
from fynx import Store, observable, computed

class CartStore(Store):
    items = observable([
        {"name": "Widget", "price": 10, "quantity": 2},
        {"name": "Gadget", "price": 15, "quantity": 1}
    ])

    @computed
    def total_items(self):
        return sum(item["quantity"] for item in self.items)

    @computed
    def total_price(self):
        return sum(item["price"] * item["quantity"] for item in self.items)

print(f"Items: {CartStore.total_items}, Total: ${CartStore.total_price}")
# Items: 3, Total: $35

# Add new item reactively
CartStore.items = CartStore.items + [{"name": "Tool", "price": 20, "quantity": 1}]
print(f"Items: {CartStore.total_items}, Total: ${CartStore.total_price}")
# Items: 4, Total: $55
```

Performance Optimization
-------------------------

Computed values include several performance optimizations:

**Lazy Evaluation**: Values only recalculate when accessed after dependencies change:

```python
expensive_calc = computed(lambda: slow_computation(), some_obs)
# slow_computation() only runs when expensive_calc.value is accessed
```

**Dependency Tracking**: Only tracks observables actually accessed during computation:

```python
result = computed(
    lambda: some_obs.value if condition else default_value,
    some_obs  # Only some_obs is tracked as dependency
)
```

**Memoization**: Results are cached until dependencies actually change:

```python
# This computation runs once, then is cached
memoized = computed(lambda: expensive_op(), obs)
print(memoized.value)  # Runs expensive_op()
print(memoized.value)  # Returns cached result

obs.set(new_value)     # Cache invalidated
print(memoized.value)  # Runs expensive_op() again
```

Best Practices
--------------

### Function Purity
Computed functions should be pure (no side effects):

```python
# Good: Pure function
clean_total = computed(lambda items: sum(item.price for item in items), items)

# Bad: Side effects
dirty_total = computed(
    lambda items: print("Calculating...") or sum(item.price for item in items),
    items
)
```

### Error Handling
Handle potential errors gracefully:

```python
safe_division = computed(
    lambda a, b: a / b if b != 0 else 0,
    num | denom
)
```

### Complex Dependencies
For complex dependency logic, consider breaking into smaller computations:

```python
# Instead of one complex computation
complex_calc = computed(
    lambda a, b, c, d: expensive_calc(a + b, c * d),
    obs_a | obs_b | obs_c | obs_d
)

# Break into smaller, more efficient computations
sum_ab = computed(lambda a, b: a + b, obs_a | obs_b)
prod_cd = computed(lambda c, d: c * d, obs_c | obs_d)
final = computed(lambda ab, cd: expensive_calc(ab, cd), sum_ab | prod_cd)
```

Common Patterns
---------------

**Filtering and Transformation**:
```python
numbers = observable([1, 2, 3, 4, 5])
evens = computed(lambda nums: [n for n in nums if n % 2 == 0], numbers)
sum_even = computed(lambda evens: sum(evens), evens)
```

**Validation**:
```python
email = observable("user@example.com")
is_valid_email = computed(
    lambda e: "@" in e and "." in e.split("@")[1],
    email
)
```

**Formatting**:
```python
amount = observable(1234.56)
formatted = computed(lambda a: f"${a:,.2f}", amount)
```

**Conditional Logic**:
```python
status = observable("loading")
is_loading = computed(lambda s: s == "loading", status)
is_error = computed(lambda s: s == "error", status)
is_success = computed(lambda s: s == "success", status)
```

Limitations
-----------

- Computed functions cannot modify observables (would create circular dependencies)
- Dependencies must be accessed synchronously during computation
- Computed values cannot depend on external state that changes independently

Troubleshooting
---------------

**Computation not updating**: Check that all accessed observables are properly passed to computed()

```python
# Wrong: external_obs not passed as dependency
external_obs = observable(10)
wrong = computed(lambda: external_obs.value * 2, some_obs)  # Won't track external_obs

# Right: include all dependencies
right = computed(lambda: external_obs.value * 2, some_obs | external_obs)
```

**Performance issues**: Break complex computations into smaller ones to enable better caching

**Circular dependencies**: Computed values cannot depend on themselves or create cycles

See Also
--------

- `fynx.observable`: Core observable classes and operators
- `fynx.store`: Organizing observables into reactive state containers
- `fynx.watch`: Conditional reactive functions
"""

from typing import Callable

from .observable import MergedObservable
from .observable.computed import ComputedObservable


def computed(func: Callable, observable) -> ComputedObservable:
    """
        Create a computed observable that derives its value from other observables.

        The `computed` function creates a new observable whose value is automatically
        calculated by applying the given function to the values of the input observable(s).
        When the input observable(s) change, the computed observable automatically updates.

        This implements the functorial map operation over observables, allowing you to
        transform observable values through pure functions while preserving reactivity.

        Args:
            func: A pure function that computes the derived value. For merged observables,
                  the function receives individual values as separate arguments. For single
                  observables, it receives the single value.
            observable: The source observable(s) to compute from. Can be a single Observable
                       or a MergedObservable (created with the `|` operator).

        Returns:
            A new ComputedObservable containing the computed values. The observable will
            automatically update whenever the source observable(s) change.

        Examples:
            ```python
            from fynx import observable, computed

            # Single observable computation
            counter = observable(5)
            doubled = computed(lambda x: x * 2, counter)
            print(doubled.value)  # 10

            counter.set(7)
            print(doubled.value)  # 14

            # Merged observable computation
            width = observable(10)
            height = observable(20)
            dimensions = width | height

            area = computed(lambda w, h: w * h, dimensions)
    print(area.value)  # 200

            # More complex computation
            person = observable({"name": "Alice", "age": 30})
            greeting = computed(
                lambda p: f"Hello {p['name']}, you are {p['age']} years old!",
                person
            )
            print(greeting.value)  # "Hello Alice, you are 30 years old!"
            ```

        Note:
            Computed functions should be pure (no side effects) and relatively fast,
            as they may be called frequently when dependencies change.

        See Also:
            observable: Create basic observables
            ComputedObservable: The returned observable type
            MergedObservable: For combining multiple observables
    """
    if isinstance(observable, MergedObservable):
        # For merged observables, apply func to the tuple values
        merged_computed_obs: ComputedObservable = ComputedObservable(None, None)

        def update_merged_computed():
            values = tuple(obs.value for obs in observable._source_observables)
            result = func(*values)
            merged_computed_obs._set_computed_value(result)

        # Initial computation
        update_merged_computed()

        # Subscribe to changes in the source observable
        observable.subscribe(lambda *args: update_merged_computed())

        return merged_computed_obs
    else:
        # For single observables
        single_computed_obs: ComputedObservable = ComputedObservable(None, None)

        def update_single_computed():
            result = func(observable.value)
            single_computed_obs._set_computed_value(result)

        # Initial computation
        update_single_computed()

        # Subscribe to changes
        observable.subscribe(lambda val: update_single_computed())

        return single_computed_obs
