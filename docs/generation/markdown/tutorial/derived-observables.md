# Transforming Data with `.then()` and `>>`

Observables hold reactive values, and conditionals filter them. But what truly unlocks FynX's power is transformation—the ability to derive new values from existing ones automatically.

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

With FynX's `.then()` method and `>>` operator, you declare the relationships once:

```python
from fynx import observable

cart_items = observable([{'name': 'Widget', 'price': 10, 'quantity': 2}])
tax_rate = observable(0.08)
shipping_threshold = observable(50)

# Define transformation functions
def calculate_subtotal(items):
    return sum(item['price'] * item['quantity'] for item in items)

def calculate_tax(subtotal):
    return subtotal * tax_rate.value

def calculate_shipping(subtotal):
    return 0 if subtotal >= shipping_threshold.value else 5.99

def calculate_total(subtotal, tax, shipping):
    return subtotal + tax + shipping

# Declarative transformations using .then()
subtotal = cart_items.then(calculate_subtotal)
tax = subtotal.then(calculate_tax)
shipping = subtotal.then(calculate_shipping)
total = (subtotal + tax + shipping).then(calculate_total)

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

Both `.then()` and `>>` create computed observables, but with slightly different syntax:

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

# Both approaches work identically
total_method = numbers.then(sum_numbers)  # Using .then()
total_operator = numbers >> sum_numbers    # Using >>

# Both total_method.value and total_operator.value are 6

# Transformation re-runs when source changes
numbers.set([4, 5, 6])  # Both become 15
```

### Chaining and Multiple Transformations

Since both `.then()` and `>>` return new observables, you can chain transformations:

```python
numbers = observable([1, 2, 3])

def sum_numbers(nums):
    return sum(nums)

def format_total(total):
    return f"Total: {total}"

# Chain using .then()
total_method = numbers.then(sum_numbers)
description_method = total_method.then(format_total)

# Chain using >> (more concise)
description_operator = numbers >> sum_numbers >> format_total

description_method.subscribe(print)
description_operator.subscribe(print)

numbers.set([4, 5, 6])
# Both print: "Total: 15"
```

### Function Signatures

Your transformation functions receive the source observable's value as their first argument:

```python
# Single observable transformation
name = observable("alice")

def create_greeting(n):
    return f"Hello, {n.title()}!"

greeting_method = name.then(create_greeting)
greeting_operator = name >> create_greeting

# Multiple observables (using + first)
first = observable("John")
last = observable("Doe")

def combine_names(first_name, last_name):
    return f"{first_name} {last_name}"

full_name_method = (first + last).then(combine_names)
full_name_operator = (first + last) >> combine_names
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

Since both `.then()` and `>>` return new observables, you can chain transformations:

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

# Chain using .then() - explicit and readable
result_method = (raw_data
    .then(filter_none)      # Filter out None
    .then(filter_positive)  # Filter positive
    .then(sum_values)       # Sum
    .then(format_result))   # Format

# Chain using >> - more concise
result_operator = (raw_data
    >> filter_none
    >> filter_positive
    >> sum_values
    >> format_result)

result_method.subscribe(print)
result_operator.subscribe(print)

raw_data.set([5, None, -1, 10])
# Both print: "Total: 15"
```

Each step in the chain is reactive. Change the input and the entire pipeline recalculates automatically.

## Combining with Other Operators

Both `.then()` and `>>` work beautifully with FynX's other operators:

```python
prices = observable([10, 20, 30])
discount_rate = observable(0.1)

def calculate_discounted_total(prices_and_rate):
    prices, rate = prices_and_rate
    return sum(price * (1 - rate) for price in prices)

def is_expensive(total):
    return total > 50

def format_expensive_message(total_and_is_expensive):
    total, is_exp = total_and_is_expensive
    return f"High-value order: ${total:.2f}"

# Use + to combine, then transform
discounted_total_method = (prices + discount_rate).then(calculate_discounted_total)
discounted_total_operator = (prices + discount_rate) >> calculate_discounted_total

# Use & for conditions, then format
is_expensive_method = discounted_total_method.then(is_expensive)
is_expensive_operator = discounted_total_method >> is_expensive

expensive_message_method = (discounted_total_method + is_expensive_method).then(format_expensive_message)
expensive_message_operator = (discounted_total_method + is_expensive_operator) >> format_expensive_message
```

## Performance Characteristics

Derived observables are lazy and efficient:

* **Memoization**: Results are cached until source values change
* **Selective Updates**: Only recalculates when dependencies actually change
* **No Redundant Work**: If a transformation result hasn't changed, downstream observers don't re-run

```python
def slow_computation(data):
    # Simulate expensive operation
    time.sleep(0.1)
    return data * 2

# This transformation only runs when expensive_data changes
expensive_result_method = expensive_data.then(slow_computation)
expensive_result_operator = expensive_data >> slow_computation

# If expensive_data stays the same, slow_computation doesn't re-run
expensive_data.set(same_value)  # No recalculation
```

## Error Handling in Transformations

Transformations can fail. FynX evaluates transformations eagerly when they're created, so errors in your transformation functions will be thrown immediately:

```python
data = observable({'value': 42})

def access_missing_key(d):
    return d['missing_key'] * 2  # KeyError here!

# This will throw a KeyError immediately when the transformation is created
result_method = data.then(access_missing_key)  # KeyError!
result_operator = data >> access_missing_key   # KeyError!
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

total_method = items.then(sum_items)  # Updates when items changes
total_operator = items >> sum_items   # Updates when items changes
```

The moment you call `.value`, you extract the data and break the reactive chain. If you're building something that should update automatically when the source changes, work with the observable itself, not its value.

**Pass observables to reactive operators:**

```python
# These operators expect observables, not values
def double_count(c):
    return c * 2

derived_method = count.then(double_count)  # Pass count, not count.value
derived_operator = count >> double_count   # Pass count, not count.value
merged = first_name + last_name            # Pass observables, not .value
filtered = items & is_valid                # Pass observables, not .value
```

The operators (`.then()`, `>>`, `+`, `&`, `~`) are designed to work with observables and maintain reactivity. When you pass `.value` to them, you're passing a static snapshot instead of a reactive stream.

**Inside subscribers and reactive functions, `.value` is fine:**

```python
counter = observable(0)

def print_count(count):
    print(f"Count: {count}")

# The function receives the value directly as an argument
counter.subscribe(print_count)

def print_double_count_and_age(count):
    print(f"Double count: {count}, Age: {age.value}")

# But if you need to read OTHER observables inside, use .value
counter.subscribe(print_double_count_and_age)
```

When your function is already being called reactively (through a subscription or decorator), using `.value` inside it to read other observables is perfectly appropriate. You're already in a reactive context.

**Rule of thumb:** If you want something to update automatically when the observable changes, don't use `.value`. If you just need to read the current value for immediate use, `.value` is correct.

## Observable Mutation Detection

FynX requires explicit notification via `.set()` to trigger reactivity. In-place mutations don't automatically trigger notifications:

```python
items = observable([1, 2, 3])

# This does NOT trigger subscribers
# The mutation happens, but .set() is never called
items.value.append(4)

# You must explicitly call .set() to notify subscribers
items.set(items.value + [4])  # This DOES trigger subscribers
```

**How it works:** When you call `.set()`, FynX computes structural changes (deltas) between the old and new values. For lists, dictionaries, sets, and other supported types, FynX detects what changed and propagates these changes efficiently through the reactive graph.

**The key point:** The limitation is about the notification trigger (calling `.set()`), not about detection capability. When you do call `.set()`, FynX's delta system automatically detects structural changes:

```python
items = observable([1, 2, 3])

# Delta-aware update: FynX detects this is an append operation
items.set(items.value + [4])  # Computes delta: [insert(3, 4)]

# Delta-aware update: FynX detects dictionary changes
profile = observable({"name": "Alice"})
profile.set({**profile.value, "age": 30})  # Computes delta: {age: (None, 30)}
```

In-place mutations bypass the notification mechanism entirely—they modify the object without calling `.set()`, so subscribers never get notified. Always use immutable update patterns and call `.set()` to trigger reactivity.

## External State Dependencies

Derived observables don't track external variables:

```python
external_multiplier = 2

counter = observable(0)

def multiply_by_external(c):
    return c * external_multiplier

# This depends on external_multiplier, but FynX doesn't know
doubled_method = counter.then(multiply_by_external)
doubled_operator = counter >> multiply_by_external

external_multiplier = 3
counter.set(5)  # Still uses old multiplier value (2), result = 10
```

If your transformation depends on variables outside the observable, FynX won't track those dependencies. Keep all reactive state inside observables for predictable behavior.

## Best Practices for Transformations

### 1. Keep Transformations Pure

```python
# Good - Pure function, same input always gives same output
def to_uppercase(n):
    return n.upper()

uppercase_method = name.then(to_uppercase)
uppercase_operator = name >> to_uppercase

# Avoid - Impure function, depends on external state
import random

def random_case(n):
    return n.upper() if random.random() > 0.5 else n.lower()

random_case_method = name.then(random_case)  # Unpredictable
random_case_operator = name >> random_case   # Unpredictable
```

Pure functions make your reactive system predictable and testable.

### 2. Handle Edge Cases

```python
# Good - Handles empty lists gracefully
def safe_average(nums):
    return sum(nums) / len(nums) if nums else 0

average_method = numbers.then(safe_average)
average_operator = numbers >> safe_average

# Avoid - Will crash on empty list
def unsafe_average(nums):
    return sum(nums) / len(nums)

unsafe_method = numbers.then(unsafe_average)  # Crashes on empty list
unsafe_operator = numbers >> unsafe_average   # Crashes on empty list
```

Defensive programming prevents runtime errors in your reactive pipelines.

### 3. Name Your Transformations

```python
# Clear intent
def calculate_age(date):
    return (datetime.now() - date).days // 365

def is_adult(age):
    return age >= 18

user_age_method = birth_date.then(calculate_age)
user_age_operator = birth_date >> calculate_age

is_adult_method = user_age_method.then(is_adult)
is_adult_operator = user_age_operator >> is_adult

eligible_for_voting = is_adult_method & has_citizenship

# Unclear intent
def transform(d):
    return calculate_age(d)

def filter_age(age):
    return age >= 18

transformed_method = birth_date.then(transform)
filtered_method = transformed_method.then(filter_age)
```

Descriptive names make your reactive graphs self-documenting.

### 4. Avoid Deep Nesting

```python
# Good - Break complex transformations into steps
def extract_user_data(response):
    return response['user']

def extract_user_age(user_data):
    return user_data['age']

def is_adult(age):
    return age >= 18

# Each of these are identical
user_data_method = api_response.then(extract_user_data)
user_data_operator = api_response >> extract_user_data

user_age_method = user_data_method.then(extract_user_age)
user_age_operator = user_data_operator >> extract_user_age

is_adult_method = user_age_method.then(is_adult)
is_adult_operator = user_age_operator >> is_adult

# Avoid - Hard to debug and modify
def complex_extraction(response):
    return response['user']['age'] >= 18

complex_method = api_response.then(complex_extraction)
complex_operator = api_response >> complex_extraction
```

Small, focused transformations are easier to test and maintain.

### 5. Consider Performance

```python
# Good - Efficient for large lists
def sum_list(lst):
    return sum(lst)

summed_method = large_list.then(sum_list)
summed_operator = large_list >> sum_list

# Better - Lazy evaluation with generator
def sum_generator(lst):
    return sum(x for x in lst)

summed_lazy_method = large_list.then(sum_generator)
summed_lazy_operator = large_list >> sum_generator
```

Be mindful of performance, especially with large data structures.

## Common Transformation Patterns

### Data Validation

```python
email = observable("user@")

def validate_email(e):
    return "@" in e and "." in e.split("@")[1]

def email_feedback(valid):
    return "Valid" if valid else "Invalid"

is_valid_email_method = email.then(validate_email)
is_valid_email_operator = email >> validate_email

email_feedback_method = is_valid_email_method.then(email_feedback)
email_feedback_operator = is_valid_email_operator >> email_feedback
```

### Data Formatting

```python
price = observable(29.99)

def format_price(p):
    return f"${p:.2f}"

formatted_price_method = price.then(format_price)
formatted_price_operator = price >> format_price
```

### Collection Operations

```python
items = observable([1, 2, 3, 4, 5])

def filter_evens(lst):
    return [x for x in lst if x % 2 == 0]

def double_items(lst):
    return [x * 2 for x in lst]

def sum_items(lst):
    return sum(lst)

# Filter
evens_method = items.then(filter_evens)
evens_operator = items >> filter_evens

# Map
doubled_method = items.then(double_items)
doubled_operator = items >> double_items

# Reduce
total_method = items.then(sum_items)
total_operator = items >> sum_items
```

### State Derivation

```python
app_state = observable("loading")

def is_loading_state(s):
    return s == "loading"

def is_error_state(s):
    return s == "error"

def is_ready_state(s):
    return s == "ready"

is_loading_method = app_state.then(is_loading_state)
is_loading_operator = app_state >> is_loading_state

is_error_method = app_state.then(is_error_state)
is_error_operator = app_state >> is_error_state

is_ready_method = app_state.then(is_ready_state)
is_ready_operator = app_state >> is_ready_state
```

## The Big Picture

Both `.then()` and `>>` transform FynX from a simple notification system into a powerful data transformation engine. You stop writing imperative update code and start declaring relationships:

* **From**: "When X changes, update Y, then update Z"
* **To**: "Y is a transformation of X, Z is a transformation of Y"

This declarative approach eliminates entire categories of bugs:

* **No stale data**: Derived values always reflect current source values
* **No forgotten updates**: The reactive graph handles all propagation
* **No manual synchronization**: Relationships are maintained automatically

Combined with conditionals (`&`) and merging (`+`), derived observables give you a complete toolkit for building reactive data pipelines. You describe what your data should look like, and FynX ensures it stays that way.

The next step is organizing these reactive pieces into reusable units called **Stores**—the architectural pattern that brings everything together.
