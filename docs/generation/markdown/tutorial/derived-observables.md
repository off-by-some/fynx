# Transforming Data with `.then()` and `>>`

Observables hold reactive values. What truly unlocks FynX's power is transformation—the ability to derive new values from existing ones automatically.

FynX provides two ways to create derived observables: the `.then()` method and the `>>` operator. Both create computed observables that automatically stay in sync with their sources. When the source changes, the transformation function runs, and the derived observable updates.

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

With FynX's `.then()` method, you declare the relationships once:

```python
from fynx import observable

cart_items = observable([{'name': 'Widget', 'price': 10, 'quantity': 2}])
tax_rate = observable(0.08)
shipping_threshold = observable(50)

# Define transformation functions
def calculate_subtotal(items):
    return sum(item['price'] * item['quantity'] for item in items)

def calculate_tax(subtotal, rate):
    return subtotal * rate

def calculate_shipping(subtotal, threshold):
    return 0 if subtotal >= threshold else 5.99

def calculate_total(subtotal, tax, shipping):
    return subtotal + tax + shipping

# Declarative transformations using .then() and .alongside()
subtotal = cart_items.then(calculate_subtotal)
tax = subtotal.alongside(tax_rate).then(calculate_tax)
shipping = subtotal.alongside(shipping_threshold).then(calculate_shipping)
total = subtotal.alongside(tax).alongside(shipping).then(calculate_total)

# Subscribe to see results
def print_subtotal(s):
    print(f"Subtotal: ${s:.2f}")

def print_tax(t):
    print(f"Tax: ${t:.2f}")

def print_shipping(s):
    print(f"Shipping: ${s:.2f}")

def print_total(t):
    print(f"Total: ${t:.2f}")

subtotal.subscribe(print_subtotal)
tax.subscribe(print_tax)
shipping.subscribe(print_shipping)
total.subscribe(print_total)

# Now just change the source data
cart_items.set(cart_items.value + [{'name': 'Gadget', 'price': 15, 'quantity': 1}])
# All derived values update automatically!
```

You declare what each value means in terms of others. Changes propagate automatically. No manual recalculation. No stale data. No forgotten updates.

## How `.then()` and `>>` Work: Function Application

`.then()` has a shorthand: the `>>` operator does exactly the same thing.

* **`.then()`**: `source_observable.then(transformation_function)` - Method syntax
* **`>>`**: `source_observable >> transformation_function` - Operator syntax

Both approaches:

* Take the current value from the source observable
* Pass it to your transformation function immediately (eager evaluation)
* Wrap the result in a new observable
* Automatically re-run the transformation when the source changes

```python
numbers = observable([1, 2, 3])

def sum_numbers(nums):
    return sum(nums)

total = numbers.then(sum_numbers)

# Or, to use fynx's syntactic sugar:
# total = numbers >> sum_numbers

print(total.value)  # 6

# Transformation re-runs when source changes
numbers.set([4, 5, 6])
print(total.value)  # 15
```

The rest of this page uses `.then()` throughout, but reach for `>>` any time you'd rather read the pipeline left to right.

### Function Signatures

Your transformation functions receive the source observable's value as their first argument:

```python
# Single observable transformation
name = observable("alice")

def create_greeting(n):
    return f"Hello, {n.title()}!"

greeting = name.then(create_greeting)

# Multiple observables (combine first)
first = observable("John")
last = observable("Doe")

def combine_names(first_name, last_name):
    return f"{first_name} {last_name}"

full_name = first.alongside(last).then(combine_names)
```

### Return Values

Your functions can return anything—a number, string, list, dictionary, even another observable:

```python
data = observable({'users': [{'name': 'Alice'}, {'name': 'Bob'}]})

def extract_user_count(d):
    return len(d['users'])

def extract_user_names(d):
    return [u['name'] for u in d['users']]

def create_count_observable(d):
    return observable(len(d['users']))

# Extract user count
user_count = data.then(extract_user_count)

# Extract user names
user_names = data.then(extract_user_names)

# Create a derived observable
user_count_obs = data.then(create_count_observable)
```

## Chaining Transformations

Since `.then()` returns a new observable, you can chain calls to build a pipeline:

```python
raw_data = observable([1, 2, 3, None, 4, None])

def filter_none(data):
    return [x for x in data if x is not None]

def filter_positive(clean):
    return [x for x in clean if x > 0]

def sum_values(filtered):
    return sum(filtered)

def format_result(total):
    return f"Total: {total}"

result = (raw_data
    .then(filter_none)      # Filter out None
    .then(filter_positive)  # Filter positive
    .then(sum_values)       # Sum
    .then(format_result))   # Format

result.subscribe(print)

raw_data.set([5, None, -1, 10])
# Prints: "Total: 15"
```

Each step in the chain is reactive. Change the input and the entire pipeline recalculates automatically.

## Combining with Other Operators

Chaining builds a pipeline from one source. Real transformations often need more than one source combined first, and `.then()` combines naturally with `.alongside()` for that:

```python
prices = observable([10, 20, 30])
discount_rate = observable(0.1)

# A merged observable always unpacks its tuple into separate arguments, so
# the transform function takes one parameter per merged source - not a
# single tuple parameter.
def calculate_discounted_total(prices, rate):
    return sum(price * (1 - rate) for price in prices)

def is_expensive(total):
    return total > 50

def format_expensive_message(total, is_exp):
    return f"High-value order: ${total:.2f}"

# Combine, then transform
discounted_total = prices.alongside(discount_rate).then(calculate_discounted_total)

# Derive a boolean, then format
is_expensive_order = discounted_total.then(is_expensive)

expensive_message = discounted_total.alongside(is_expensive_order).then(format_expensive_message)
```

Each derived value here still only recalculates when its own sources change - `is_expensive_order` doesn't recompute just because `prices` changed, if the discounted total it depends on didn't move.

## Performance Characteristics

That selectivity isn't incidental - it's the main reason derived observables stay cheap at scale:

* **Memoization**: Results are cached until source values change
* **Selective Updates**: Unobserved values recalculate only when read after dependencies actually change
* **Demand-Driven Notifications**: Subscribers create the effect boundary needed to deliver updates automatically
* **No Redundant Work**: If a transformation result hasn't changed, downstream observers don't re-run

```python
def slow_computation(data):
    # Simulate expensive operation
    time.sleep(0.1)
    return data * 2

# This transformation only runs when expensive_data changes
expensive_result = expensive_data.then(slow_computation)

# If expensive_data stays the same, slow_computation doesn't re-run
expensive_data.set(same_value)  # No recalculation
```

## Error Handling in Transformations

Transformations can fail. FynX evaluates transformations eagerly when they're created, so errors in your transformation functions will be thrown immediately:

```python
data = observable({'value': 42})

def access_missing_key(d):
    return d['missing_key'] * 2  # KeyError here!

# This raises a KeyError immediately, when the transformation is created
result = data.then(access_missing_key)  # KeyError!
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
def sum_items(item_list):
    return sum(item_list)

total = items.then(sum_items)  # Updates when items changes
```

The moment you call `.value`, you extract the data and break the reactive chain. If you're building something that should update automatically when the source changes, work with the observable itself, not its value.

**Pass observables to reactive operators:**

```python
# These operators expect observables, not values
def double_count(c):
    return c * 2

derived = count.then(double_count)          # Pass count, not count.value
merged = first_name.alongside(last_name)    # Pass observables, not .value
filtered = items @ is_valid                 # Pass observables, not .value
```

These operators are designed to work with observables and maintain reactivity. When you pass `.value` to them, you're passing a static snapshot instead of a reactive stream. (`@` is the gate operator - [Conditionals](conditionals.md) covers it in depth.)

**Inside transforms, use the arguments FynX gives you:**

```python
price = observable(100.0)
discount = observable(0.1)

# Bad: discount.value is hidden inside the transform
discounted = price.then(lambda p: p * (1 - discount.value))

# Good: combine inputs first, then transform plain values
discounted = price.alongside(discount).then(lambda p, d: p * (1 - d))
```

If a transform reads `.value` or calls `.set()` on any observable, FynX raises `TransformPurityError` and points you toward an explicit `.alongside()` form. This keeps transforms easy to reason about: everything they depend on appears as an argument.

**Inside subscribers and reactive functions, `.value` is fine:**

```python
counter = observable(0)

def print_count(count):
    print(f"Count: {count}")

# The function receives the value directly as an argument
counter.subscribe(print_count)

def print_double_count_and_age(count):
    print(f"Double count: {count}, Age: {age.value}")

# This is an ordinary side read. It does not make age a source of this subscription.
counter.subscribe(print_double_count_and_age)
```

Inside subscribers and reactive functions, using `.value` to read another observable is allowed. With `subscribe()`, the subscription still runs when its subscribed observable changes. With `@reactive`, observables read during the function become tracked dependencies.

**Rule of thumb:** In `.then()` transforms, use only the arguments FynX passes in. In `@reactive` functions and subscription callbacks, `.value` is for reading current state at the application's effect boundary.

## Observable Mutation Detection

FynX can't automatically detect changes to the contents of observables—only to the reference stored in them:

```python
items = observable([1, 2, 3])

# This does NOT trigger subscribers
items.value.append(4)

# You must explicitly call .set()
items.set(items.value + [4])  # This DOES trigger subscribers
```

When you modify a list, dictionary, or custom object in place, subscribers won't know. You must call `.set()` with the updated value—even if it's the same object reference—to trigger reactivity. Stores follow the same rule; see [Stores: Immutable Updates](stores.md#a-critical-pattern-immutable-updates) for the Store-specific version of this.

## External State in Transforms

Derived observables can close over ordinary Python values, but those values are not reactive inputs:

```python
external_multiplier = 2

counter = observable(0)

def multiply_by_external(c):
    return c * external_multiplier

# This depends on external_multiplier, which is not an observable
doubled = counter.then(multiply_by_external)

external_multiplier = 3
counter.set(5)  # Uses current multiplier value (3), result = 15
```

If the extra value should be reactive, make it an observable and combine it explicitly:

```python
multiplier = observable(2)

scaled = counter.alongside(multiplier).then(lambda count, factor: count * factor)
```

Transforms may not read `.value` from observables or call `.set()` on observables—the same `TransformPurityError` from earlier applies here too.

## The Big Picture

`.then()` (and its `>>` shorthand) transform FynX from a simple notification system into a powerful data transformation engine. You stop writing imperative update code and start declaring relationships:

* **From**: "When X changes, update Y, then update Z"
* **To**: "Y is a transformation of X, Z is a transformation of Y"

This declarative approach eliminates entire categories of bugs:

* **No stale data**: Derived values always reflect current source values
* **No forgotten updates**: The reactive graph handles all propagation
* **No manual synchronization**: Relationships are maintained automatically

Combined with [gates](conditionals.md) (`.requiring()` / `@`) and products (`.alongside()` / `+`), derived observables give you a complete toolkit for building reactive data pipelines. You describe what your data should look like, and FynX ensures it stays that way.

For practical guidance on naming, edge cases, and common transformation patterns, see the [Best Practices](best-practices.md) page. The next step is organizing these reactive pieces into reusable units called [Stores](stores.md)—the architectural pattern that brings everything together.
