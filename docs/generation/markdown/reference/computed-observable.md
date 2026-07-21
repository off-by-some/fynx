# Derived Observables

Derived observables are created using the `.then()` method or `>>` operator to transform values from source observables.

## Creating Derived Observables

```python
from fynx import observable

# Source observable
count = observable(0)

# Derived observable using .then()
doubled = count.then(lambda x: x * 2)

# Derived observable using >> operator
tripled = count >> (lambda x: x * 3)

# Subscribe to changes
doubled.subscribe(lambda x: print(f"Doubled: {x}"))
tripled.subscribe(lambda x: print(f"Tripled: {x}"))

# Update source
count.set(5)  # Prints: Doubled: 10, Tripled: 15
```

## Chaining Transformations

```python
# Chain multiple transformations
processed = count.then(lambda x: x * 2).then(lambda x: x + 1)

# Or using >> operator
processed = count >> (lambda x: x * 2) >> (lambda x: x + 1)

processed.subscribe(lambda x: print(f"Processed: {x}"))
count.set(3)  # Prints: Processed: 7
```

## Using Named Functions

For better readability and reusability, use named functions instead of lambdas:

```python
def double(x):
    return x * 2

def add_one(x):
    return x + 1

# Clean, readable transformations
doubled = count.then(double)
incremented = doubled.then(add_one)

# Or with >> operator
result = count >> double >> add_one
```

## Transform Purity

Transform functions receive plain values. They should not read `.value` from another observable, and they should not call `.set()` on any observable. Put every reactive input on the left side of the transform:

```python
price = observable(100.0)
discount = observable(0.1)

# Good: both values are passed as arguments
discounted = (price + discount) >> (lambda p, d: p * (1 - d))

# Error: discount.value is hidden from the graph
discounted = price >> (lambda p: p * (1 - discount.value))
```

When this rule is violated, FynX raises `TransformPurityError` with a hint to combine inputs using `+` / `.alongside()` or move mutations and other effects to `.subscribe()` / `@reactive`.

## Key Properties

* **Reactive**: Automatically update when source observables change
* **Immutable**: Don't modify source values, create new derived values
* **Composable**: Can be chained and combined with other observables
* **Lazy**: Only compute when subscribed to or when source changes
