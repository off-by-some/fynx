# Derived Observables: Transforming Data with `>>`

Observables hold reactive values, and conditionals filter them. But what truly unlocks FynX's power is transformation—the ability to derive new values from existing ones automatically.

That's what the `>>` operator does. It's FynX's transformation engine. It takes an observable and a function, and creates a new observable that automatically stays in sync. When the source changes, the function runs, and the derived observable updates.

This is where reactive programming stops being about "responding to changes" and starts being about "declaring relationships." You describe how values relate to each other, and FynX handles the synchronization.

## The Problem: Manual Recalculation

Consider a shopping cart where you need to calculate totals, taxes, and shipping:

```python
cart_items = [{'name': 'Widget', 'price': 10, 'quantity': 2}]
tax_rate = 0.08
shipping_threshold = 50

# Manual calculations
subtotal = sum(item['price'] * item['quantity'] for item in cart_items)
tax = subtotal * tax_rate
shipping = 0 if subtotal >= shipping_threshold else 5.99
total = subtotal + tax + shipping

print(f"Subtotal: ${subtotal:.2f}")
print(f"Tax: ${tax:.2f}")
print(f"Shipping: ${shipping:.2f}")
print(f"Total: ${total:.2f}")
```

Now add an item to the cart. You have to manually recalculate everything:

```python
cart_items.append({'name': 'Gadget', 'price': 15, 'quantity': 1})

# Recalculate everything again
subtotal = sum(item['price'] * item['quantity'] for item in cart_items)
tax = subtotal * tax_rate
shipping = 0 if subtotal >= shipping_threshold else 5.99
total = subtotal + tax + shipping

print(f"Subtotal: ${subtotal:.2f}")  # Have to remember to do this
print(f"Tax: ${tax:.2f}")          # Have to remember to do this
print(f"Shipping: ${shipping:.2f}") # Have to remember to do this
print(f"Total: ${total:.2f}")       # Have to remember to do this
```

Every time state changes, you have to remember to update all the derived values. Miss one and your display goes stale. This is the synchronization problem that plagues traditional applications.

## The Solution: Declarative Derivation

With FynX's `>>` operator, you declare the relationships once:

```python
from fynx import observable

cart_items = observable([{'name': 'Widget', 'price': 10, 'quantity': 2}])
tax_rate = observable(0.08)
shipping_threshold = observable(50)

# Declarative transformations
subtotal = cart_items >> (lambda items: sum(item['price'] * item['quantity'] for item in items))
tax = subtotal >> (lambda st: st * tax_rate.value)
shipping = subtotal >> (lambda st: 0 if st >= shipping_threshold.value else 5.99)
total = (subtotal | tax | shipping) >> (lambda st, t, s: st + t + s)

# Subscribe to see results
subtotal.subscribe(lambda s: print(f"Subtotal: ${s:.2f}"))
tax.subscribe(lambda t: print(f"Tax: ${t:.2f}"))
shipping.subscribe(lambda s: print(f"Shipping: ${s:.2f}"))
total.subscribe(lambda t: print(f"Total: ${t:.2f}"))

# Now just change the source data
cart_items.set(cart_items.value + [{'name': 'Gadget', 'price': 15, 'quantity': 1}])
# All derived values update automatically!
```

You declare what each value means in terms of others. Changes propagate automatically. No manual recalculation. No stale data. No forgotten updates.

## How `>>` Works: Function Application

The `>>` operator creates a computed observable: `source_observable >> transformation_function`

- Takes the current value from the source observable
- Passes it to your transformation function immediately (eager evaluation)
- Wraps the result in a new observable
- Automatically re-runs the transformation when the source changes

```python
numbers = observable([1, 2, 3])

# Transformation runs immediately with initial value
total = numbers >> (lambda nums: sum(nums))  # total.value is 6

# Transformation re-runs when source changes
numbers.set([4, 5, 6])  # total.value becomes 15
```

### Chaining and Multiple Transformations

Since `>>` returns a new observable, you can chain transformations:

```python
numbers = observable([1, 2, 3])

# Chain: transform → transform again
total = numbers >> (lambda nums: sum(nums))
description = total >> (lambda t: f"Total: {t}")

description.subscribe(lambda d: print(f"Result: {d}"))

numbers.set([4, 5, 6])
# Prints: "Result: Total: 15"
```

### Function Signatures

Your transformation functions receive the source observable's value as their first argument:

```python
# Single observable transformation
name = observable("alice")
greeting = name >> (lambda n: f"Hello, {n.title()}!")

# Multiple observables (using | first)
first = observable("John")
last = observable("Doe")
full_name = (first | last) >> (lambda f, l: f"{f} {l}")
```

### Return Values

Your functions can return anything—a number, string, list, dictionary, even another observable:

```python
data = observable({'users': [{'name': 'Alice'}, {'name': 'Bob'}]})

# Extract user count
user_count = data >> (lambda d: len(d['users']))

# Extract user names
user_names = data >> (lambda d: [u['name'] for u in d['users']])

# Create a derived observable
user_count_obs = data >> (lambda d: observable(len(d['users'])))
```

## Chaining Transformations

Since `>>` returns a new observable, you can chain transformations:

```python
raw_data = observable([1, 2, 3, None, 4, None])

# Chain: filter → clean → sum → format
result = (raw_data
    >> (lambda data: [x for x in data if x is not None])  # Filter out None
    >> (lambda clean: [x for x in clean if x > 0])        # Filter positive
    >> (lambda filtered: sum(filtered))                    # Sum
    >> (lambda total: f"Total: {total}"))                  # Format

result.subscribe(lambda r: print(r))

raw_data.set([5, None, -1, 10])
# Prints: "Total: 15"
```

Each step in the chain is reactive. Change the input and the entire pipeline recalculates automatically.

## Combining with Other Operators

The `>>` operator works beautifully with FynX's other operators:

```python
prices = observable([10, 20, 30])
discount_rate = observable(0.1)

# Use | to combine, then >> to transform
discounted_total = (prices | discount_rate) >> (
    lambda p, r: sum(price * (1 - r) for price in p)
)

# Use & for conditions, then >> for formatting
is_expensive = discounted_total >> (lambda t: t > 50)
expensive_message = (discounted_total & is_expensive) >> (
    lambda t: f"High-value order: ${t:.2f}"
)
```

## Performance Characteristics

Derived observables are lazy and efficient:

- **Memoization**: Results are cached until source values change
- **Selective Updates**: Only recalculates when dependencies actually change
- **No Redundant Work**: If a transformation result hasn't changed, downstream observers don't re-run

```python
# This transformation only runs when expensive_data changes
expensive_result = expensive_data >> (lambda d: slow_computation(d))

# If expensive_data stays the same, slow_computation doesn't re-run
expensive_data.set(same_value)  # No recalculation
```

## Error Handling in Transformations

Transformations can fail. FynX evaluates transformations eagerly when they're created, so errors in your transformation functions will be thrown immediately:

```python
data = observable({'value': 42})

# This will throw a KeyError immediately when the transformation is created
result = data >> (lambda d: d['missing_key'] * 2)  # KeyError here!
```

Handle errors by ensuring your data is in the expected format before creating transformations, or by transforming the data to a safe format first.

## When to Use `.value` (and When Not To)

Understanding when to access `.value` versus passing the observable itself is crucial for writing effective reactive code.

**Use `.value` when you need the actual data for immediate use:**

```python
name = observable("alice")
age = observable(30)

# Reading for immediate use
print(f"Current user: {name.value}, age {age.value}")

# Passing to non-reactive functions
result = some_function(name.value, age.value)

# Conditionals based on current state
if age.value >= 18:
    print("Adult user")
```

When you access `.value`, you're saying "I need this data right now for a calculation or decision." This is perfect for one-time reads, immediate computations, or interfacing with code that doesn't understand observables.

**Don't use `.value` when building reactive relationships:**

```python
# Bad: Reads .value immediately, loses reactivity
total = items.value.reduce(sum)  # Just a number, won't update

# Good: Keeps reactivity by transforming the observable
total = items >> (lambda item_list: sum(item_list))  # Updates when items changes
```

The moment you call `.value`, you extract the data and break the reactive chain. If you're building something that should update automatically when the source changes, work with the observable itself, not its value.

**Pass observables to reactive operators:**

```python
# These operators expect observables, not values
derived = count >> (lambda c: c * 2)           # Pass count, not count.value
merged = first_name | last_name                # Pass observables, not .value
filtered = items & is_valid                    # Pass observables, not .value
```

The operators (`>>`, `|`, `&`, `~`) are designed to work with observables and maintain reactivity. When you pass `.value` to them, you're passing a static snapshot instead of a reactive stream.

**Inside subscribers and reactive functions, `.value` is fine:**

```python
counter = observable(0)

# The function receives the value directly as an argument
counter.subscribe(lambda count: print(f"Count: {count}"))

# But if you need to read OTHER observables inside, use .value
counter.subscribe(lambda count: print(f"Double count: {count}, Age: {age.value}"))
```

When your function is already being called reactively (through a subscription or decorator), using `.value` inside it to read other observables is perfectly appropriate. You're already in a reactive context.

**Rule of thumb:** If you want something to update automatically when the observable changes, don't use `.value`. If you just need to read the current value for immediate use, `.value` is correct.

## Observable Mutation Detection

FynX can't automatically detect changes to the contents of observables:

```python
items = observable([1, 2, 3])

# This does NOT trigger subscribers
items.value.append(4)

# You must explicitly call .set()
items.set(items.value + [4])  # This DOES trigger subscribers
```

FynX can't detect mutations to the objects inside observables. When you modify a list, dictionary, or custom object in place, subscribers won't know. You must call `.set()` with the updated value—even if it's the same object reference—to trigger reactivity.

## External State Dependencies

Derived observables don't track external variables:

```python
external_multiplier = 2

counter = observable(0)

# This depends on external_multiplier, but FynX doesn't know
doubled = counter >> (lambda c: c * external_multiplier)

external_multiplier = 3
counter.set(5)  # Still uses old multiplier value (2), result = 10
```

If your transformation depends on variables outside the observable, FynX won't track those dependencies. Keep all reactive state inside observables for predictable behavior.

## Best Practices for Transformations

### 1. Keep Transformations Pure

```python
# Good - Pure function, same input always gives same output
uppercase = name >> (lambda n: n.upper())

# Avoid - Impure function, depends on external state
random_case = name >> (lambda n: n.upper() if random.random() > 0.5 else n.lower())
```

Pure functions make your reactive system predictable and testable.

### 2. Handle Edge Cases

```python
# Good - Handles empty lists gracefully
average = numbers >> (lambda nums: sum(nums) / len(nums) if nums else 0)

# Avoid - Will crash on empty list
average = numbers >> (lambda nums: sum(nums) / len(nums))
```

Defensive programming prevents runtime errors in your reactive pipelines.

### 3. Name Your Transformations

```python
# Clear intent
user_age = birth_date >> (lambda date: calculate_age(date))
is_adult = user_age >> (lambda age: age >= 18)
eligible_for_voting = is_adult & has_citizenship

# Unclear intent
transformed = birth_date >> (lambda d: calculate_age(d))
filtered = transformed & has_citizenship
```

Descriptive names make your reactive graphs self-documenting.

### 4. Avoid Deep Nesting

```python
# Good - Break complex transformations into steps
user_data = api_response >> (lambda r: r['user'])
user_age = user_data >> (lambda u: u['age'])
is_adult = user_age >> (lambda a: a >= 18)

# Avoid - Hard to debug and modify
is_adult = api_response >> (lambda r: r['user']['age'] >= 18)
```

Small, focused transformations are easier to test and maintain.

### 5. Consider Performance

```python
# Good - Efficient for large lists
summed = large_list >> (lambda lst: sum(lst))

# Better - Lazy evaluation with generator
summed = large_list >> (lambda lst: sum(x for x in lst))
```

Be mindful of performance, especially with large data structures.

## Common Transformation Patterns

### Data Validation

```python
email = observable("user@")

is_valid_email = email >> (lambda e: "@" in e and "." in e.split("@")[1])
email_feedback = is_valid_email >> (lambda valid: "Valid" if valid else "Invalid")
```

### Data Formatting

```python
price = observable(29.99)
formatted_price = price >> (lambda p: f"${p:.2f}")
```

### Collection Operations

```python
items = observable([1, 2, 3, 4, 5])

# Filter
evens = items >> (lambda lst: [x for x in lst if x % 2 == 0])

# Map
doubled = items >> (lambda lst: [x * 2 for x in lst])

# Reduce
total = items >> (lambda lst: sum(lst))
```

### State Derivation

```python
app_state = observable("loading")

is_loading = app_state >> (lambda s: s == "loading")
is_error = app_state >> (lambda s: s == "error")
is_ready = app_state >> (lambda s: s == "ready")
```

## The Big Picture

The `>>` operator transforms FynX from a simple notification system into a powerful data transformation engine. You stop writing imperative update code and start declaring relationships:

- **From**: "When X changes, update Y, then update Z"
- **To**: "Y is a transformation of X, Z is a transformation of Y"

This declarative approach eliminates entire categories of bugs:

- **No stale data**: Derived values always reflect current source values
- **No forgotten updates**: The reactive graph handles all propagation
- **No manual synchronization**: Relationships are maintained automatically

Combined with conditionals (`&`) and merging (`|`), derived observables give you a complete toolkit for building reactive data pipelines. You describe what your data should look like, and FynX ensures it stays that way.

The next step is organizing these reactive pieces into reusable units called **Stores**—the architectural pattern that brings everything together.
