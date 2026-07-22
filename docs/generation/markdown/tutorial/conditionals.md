# Conditionals: Filtering and Logic on Observables

Observables track changes, and [`.then()`](derived-observables.md) transforms data. Neither one controls *when* data should flow - a temperature reading you only care about past 30°C, a form field you only want to validate once it's non-empty, a sync that should skip while a request is already in flight.

FynX's conditional operators handle that: they create observables that only produce values when conditions are met, and let you combine those conditions with ordinary boolean logic.

## The Problem: Not All Data Should Pass Through

Consider this scenario:

```python
temperature = observable(20)
weather_alerts = []

# You only want alerts when temperature is extreme
def on_temperature_change(temp):
    if temp > 30 or temp < 0:
        weather_alerts.append(f"Temperature alert: {temp}°C")

temperature.subscribe(on_temperature_change)
```

This works, but it's verbose. You have to write filtering logic in every subscriber. The filtering is mixed with the reaction logic. And if multiple subscribers need the same filtering, you repeat yourself.

## The Solution: Conditional Observables

FynX gives you operators that create filtered, conditional observables, keeping the condition logic separate from the filtering operation itself.

```python
temperature = observable(20)

# First, create a boolean observable representing the condition
is_extreme = temperature.then(lambda temp: temp > 30 or temp < 0)

# Then, filter the original observable based on that condition
extreme_temps = temperature.requiring(is_extreme)

# Or, to use fynx's syntactic sugar:
# is_extreme = temperature >> (lambda temp: temp > 30 or temp < 0)
# extreme_temps = temperature @ is_extreme

extreme_temps.subscribe(lambda temp: print(f"Temperature alert: {temp}°C"))
```

Now the filtering is declarative and reusable. The `extreme_temps` observable only produces values when the condition is met.

## `.all()`: Boolean Composition

A single condition like `is_extreme` is just the starting point - most real filters need to combine several at once. `.all()` (shorthand: `&`) combines multiple boolean observables into a total boolean condition. The result is `True` when every input is truthy and `False` otherwise:

```python
from fynx import observable

authenticated = observable(True)
connected = observable(True)
loading = observable(False)

ready = authenticated.all(connected, loading.negate())
# Or: ready = authenticated & connected & ~loading

ready.subscribe(lambda is_ready: print(f"Ready: {is_ready}"))
```

A boolean condition on its own doesn't filter anything, though - it's just a value that's `True` or `False`. Pairing it with `.requiring()` or `@` is what turns a condition into a gate:

```python
scores = observable(85)
is_high_score = scores.then(lambda score: score > 90)

high_scores = scores @ is_high_score

high_scores.subscribe(lambda score: print(f"🎉 High score achieved: {score}"))

scores.set(88)  # No emission (condition became False)
scores.set(95)  # Prints: "🎉 High score achieved: 95" (condition became True)
scores.set(87)  # No emission (condition became False)
```

`.all()` creates an ordinary `Observable[bool]`. `.requiring()` creates a **ConditionalObservable** that only emits when conditions are met.

## Using Conditional Observables with @reactive

`.requiring()` creates conditional observables that work perfectly with `@reactive` for event-driven reactions:

```python
from fynx import observable, reactive

scores = observable(85)

# Create a boolean condition observable
is_high_score = scores.then(lambda score: score > 90)

# Use with @reactive for event-driven reactions
@reactive(is_high_score)
def on_high_score(is_high):
    if is_high:
        print(f"🎉 High score achieved: {scores.value}")

scores.set(88)  # No output (condition not met)
scores.set(95)  # Prints: "🎉 High score achieved: 95"
scores.set(87)  # No output (condition no longer met)
```

This pattern gives you event-driven reactions while maintaining the reactive paradigm. [Using @reactive](using-reactive.md) covers reactions in depth.

### Complex Predicates

Your condition functions can be as complex as needed:

```python
user = observable({"name": "Alice", "age": 30, "country": "US"})

# Complex validation logic as a boolean observable
is_valid_user = user.then(lambda u: (
    u["age"] >= 18 and
    u["country"] in ["US", "CA", "UK"] and
    len(u["name"]) > 2 and
    "@" not in u["name"]  # No emails in names
))

# Gate: only reacts while the user is valid
valid_users = user @ is_valid_user

valid_users.subscribe(lambda user_data: print(f"✅ Valid user: {user_data['name']}"))

user.set({"name": "Bob", "age": 15, "country": "US"})  # No output - gate closes silently
user.set({"name": "Carol", "age": 25, "country": "CA"})  # Prints: "✅ Valid user: Carol"
```

`.requiring()` doesn't emit `None` when the condition fails - it simply doesn't call your subscriber at all while the gate is closed. If you need to react to *both* the valid and invalid states (for example, to show a validation error in the UI), subscribe to the boolean condition itself instead of gating:

```python
def on_validity_change(is_valid):
    if is_valid:
        print(f"✅ Valid user: {user.value['name']}")
    else:
        print("❌ User is not valid")

is_valid_user.subscribe(on_validity_change)
```

## `.negate()`: Logical Negation

`.negate()` (shorthand: `~`) inverts boolean conditions. It works on boolean observables:

```python
is_online = observable(True)

# Create a negated boolean observable
is_offline = is_online.negate()

# Or, to use fynx's syntactic sugar:
# is_offline = ~is_online

# is_offline is a plain boolean - it notifies on every change, both
# True and False - so check the value if you only care about one direction.
def on_offline_change(offline):
    if offline:
        print("User went offline")

is_offline.subscribe(on_offline_change)

is_online.set(False)  # is_offline becomes True, prints: "User went offline"
is_online.set(True)   # is_offline becomes False, no output (guarded by the if)
```

## `.either()`: Logical OR

`.either()` (shorthand: `|`) creates total boolean OR observables. The result is `True` when any condition is truthy and `False` otherwise:

```python
is_error = observable(False)
is_warning = observable(False)
is_critical = observable(False)

# Logical OR using .either()
needs_attention = is_error.either(is_warning).either(is_critical)

# Or, to use fynx's syntactic sugar:
# needs_attention = is_error | is_warning | is_critical

def on_attention_change(needs_attention_val):
    if needs_attention_val:
        print("⚠️ System needs attention!")

needs_attention.subscribe(on_attention_change)

# Like any computed observable, needs_attention only notifies when its
# actual value changes - not on every .set() call to one of its inputs.
is_error.set(True)    # False -> True: prints "⚠️ System needs attention!"
is_warning.set(True)  # True -> True (still True): no notification at all
is_error.set(False)   # True -> True (is_warning keeps it True): no notification
is_warning.set(False) # True -> False: notifies, but the `if` guard hides it
```

`.either()` does not gate away falsy results. It produces an ordinary boolean observable whose value can be safely read as `True` or `False`. Use `.requiring()` or `@` when you want that boolean condition to gate another observable.

### Combining OR with Other Operators

You can combine `.all()`, `.either()`, and `.negate()` for complex logical expressions:

```python
user_input = observable("")
is_admin = observable(False)
is_moderator = observable(True)

# Create boolean conditions
has_input = user_input.then(lambda u: len(u) > 0)
has_permission = is_admin.either(is_moderator)  # OR condition

can_submit = has_input & has_permission

def on_can_submit_change(can_submit_val):
    if can_submit_val:
        print("✅ User can submit")
    else:
        print("❌ User cannot submit")

can_submit.subscribe(on_can_submit_change)

user_input.set("Hello")  # Prints: "✅ User can submit"
is_moderator.set(False)  # Prints: "❌ User cannot submit" (no permission)
is_admin.set(True)       # Prints: "✅ User can submit" (admin permission)
```

### Combining Negation with Filtering

Create "everything except" patterns:

```python
status = observable("loading")

# Create boolean condition for non-loading states
is_not_loading = status.then(lambda s: s != "loading")

# React to any status except "loading"
non_loading_status = status @ is_not_loading

non_loading_status.subscribe(lambda status_val: print(f"Status changed to: {status_val}"))

status.set("loading")    # No output (filtered out)
status.set("success")    # Prints: "Status changed to: success"
status.set("error")      # Prints: "Status changed to: error"
```

## Real-World Example: Form Validation

Form validation is perfect for conditional observables:

```python
email = observable("")
password = observable("")
terms_accepted = observable(False)

# Validation conditions as boolean observables
email_valid = email.then(lambda e: "@" in e and "." in e.split("@")[1])
password_strong = password.then(lambda p: len(p) >= 8)
terms_checked = terms_accepted.then(lambda t: t == True)

# We need to show both "valid" and "invalid" states in the UI, so this is a
# total boolean built with `.all()` / `&`, not a gate built with `.requiring()`.
form_valid = email_valid.all(password_strong, terms_checked)

def on_form_valid_change(is_valid):
    if is_valid:
        print("✅ Form is complete and valid!")
    else:
        print("❌ Form validation failed")

form_valid.subscribe(on_form_valid_change)

# Simulate form filling. form_valid starts False and only notifies when its
# value actually changes, so these first three calls print nothing - the
# form was already invalid and stays invalid.
email.set("user@")           # Still invalid (no "." after @)
password.set("pass")         # Still invalid (too short)
terms_accepted.set(True)     # Still invalid (email and password still bad)

email.set("user@example.com")  # Still invalid (password too short)
password.set("secure123")      # False -> True: prints "✅ Form is complete and valid!"
password.set("short")          # True -> False: prints "❌ Form validation failed"
```

As before, `.requiring()` would stay silent while its gate is closed - this example uses `.all()` instead specifically because it needs to observe both the valid and invalid states.

## Advanced Patterns: State Machines with Conditionals

Form validation combines a handful of independent conditions. State machines push the same idea further, gating an entire application state on several conditions holding at once:

```python
app_state = observable("initializing")
user_authenticated = observable(False)
data_loaded = observable(False)

# Define state conditions as boolean observables
is_app_ready = app_state.then(lambda s: s == "ready")
is_user_auth = user_authenticated.then(lambda a: a == True)
is_data_loaded = data_loaded.then(lambda d: d == True)
is_app_error = app_state.then(lambda s: s == "error")

# Combine conditions - app is ready when all are true
ready_state = app_state @ (is_app_ready & is_user_auth & is_data_loaded)

# Error state
error_state = app_state @ is_app_error

# React to state transitions
ready_state.subscribe(lambda _: print("🚀 Application is fully ready!"))
error_state.subscribe(lambda _: print("❌ Application encountered an error"))

# Simulate app lifecycle
app_state.set("authenticating")
user_authenticated.set(True)
app_state.set("loading_data")
data_loaded.set(True)
app_state.set("ready")  # Triggers "fully ready" message
```

## Conditional Operators with Derived Values

Gates compose with `.then()` just as readily as they compose with each other. Filter first, then derive from whatever passes through:

```python
sensor_readings = observable([])

# Create condition for sufficient data
has_enough_data = sensor_readings.then(lambda readings: len(readings) >= 3)

# Only process readings when we have enough data
valid_readings = sensor_readings @ has_enough_data

# Then calculate statistics
average_reading = valid_readings.then(lambda readings: sum(readings) / len(readings))

average_reading.subscribe(lambda avg: print(f"Average sensor reading: {avg:.2f}"))

sensor_readings.set([1, 2])        # No output (not enough data)
sensor_readings.set([1, 2, 3, 4])  # Prints: "Average sensor reading: 2.50"
```

## Performance Benefits

That last example hints at a broader payoff: conditional observables improve performance by:

1. **Reducing unnecessary computations** - Only process data that meets criteria
2. **Filtering at the source** - Don't pass unwanted data to subscribers
3. **Early termination** - Stop reactive chains when conditions aren't met

```python
# Without conditionals - expensive operation runs on every change
raw_data = observable("some data")
processed_data = raw_data.then(lambda d: expensive_cleanup(d))
final_result = processed_data.then(lambda d: expensive_analysis(d))

# With conditionals - expensive operations only run when needed
worth_processing = raw_data.then(is_worth_processing)
clean_data = raw_data @ worth_processing
processed_data = clean_data.then(lambda d: expensive_cleanup(d))
final_result = processed_data.then(lambda d: expensive_analysis(d))
```

Naming `worth_processing` this way, rather than inlining `is_worth_processing` straight into `@`, is worth doing on its own even with a single gate: it can be tested independently, and if a second gate ever needs the same check, it's already there to reuse - see [Best Practices](best-practices.md#give-reused-conditions-a-name) for what that looks like with more than one consumer.

## The Big Picture

Conditionals transform your reactive system from "process everything" to "process only what matters":

* **`.all()`** (`&`): Create logical AND conditions between boolean observables
* **`.either()`** (`|`): Create logical OR conditions between boolean observables
* **`.negate()`** (`~`): Invert boolean conditions
* **`.requiring()`** (`@`): Gate data streams based on predicates or boolean conditions
* **Performance**: Skip unnecessary computations
* **Clarity**: Separate filtering logic from reaction logic
* **Composition**: Combine conditions with other operators

Think of conditionals as reactive filters. They let you create observables that only emit valuable data, reducing noise and improving performance. Combined with transformations (`.then()`) and reactions (`@reactive`), they give you a complete toolkit for building sophisticated reactive applications.

For common conditional patterns and gotchas, see the [Best Practices](best-practices.md) page. The next step is organizing these reactive pieces into reusable units called [Stores](stores.md)—the architectural pattern that brings everything together.
