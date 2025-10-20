# ConditionalObservable

Observables that only emit values when specific conditions are met.

## Creating Conditional Observables

Use the `&` operator to create conditional observables:

```python
from fynx import observable

# Source observable
count = observable(0)

# Conditional observable - only emits when count > 5
filtered = count & (lambda x: x > 5)

filtered.subscribe(lambda x: print(f"Filtered: {x}"))

count.set(3)  # No output
count.set(7)  # Prints: Filtered: 7
count.set(2)  # No output
```

## Multiple Conditions

Chain multiple conditions using the `&` operator:

```python
# Only emit when count is between 5 and 10
range_filtered = count & (lambda x: x >= 5) & (lambda x: x <= 10)

range_filtered.subscribe(lambda x: print(f"In range: {x}"))

count.set(3)   # No output
count.set(7)   # Prints: In range: 7
count.set(15)  # No output
```

## Using Named Functions

For better readability, use named functions:

```python
def is_positive(x):
    return x > 0

def is_even(x):
    return x % 2 == 0

# Only emit positive even numbers
positive_even = count & is_positive & is_even

positive_even.subscribe(lambda x: print(f"Positive even: {x}"))

count.set(-2)  # No output
count.set(3)   # No output
count.set(4)   # Prints: Positive even: 4
```

## Key Properties

* **Filtering**: Only emit values that satisfy all conditions
* **Reactive**: Automatically re-evaluate conditions when source changes
* **Composable**: Can be combined with other observables using `&`, `|`, and `>>`
* **Efficient**: Conditions are only evaluated when source values change
