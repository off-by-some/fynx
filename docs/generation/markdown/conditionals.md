# Conditionals: Filtering and Logic on Observables

Observables track changes, and the `>>` operator transforms data. But what about controlling when data flows through your reactive system? What if you only want certain values to pass through?

What if you only want to react when certain conditions are true? What if you want to combine multiple conditions? What if you need to filter out unwanted values?

That's where FynX's conditional operators come in. They let you create observables that only produce values when conditions are met, filter data streams, and combine logical expressions.

## The Problem: Not All Data Should Pass Through

Consider this scenario:

```python
temperature = observable(20)
weather_alerts = []

# You only want alerts when temperature is extreme
temperature.subscribe(lambda temp: {
    if temp > 30 or temp < 0:
        weather_alerts.append(f"Temperature alert: {temp}°C")
})
```

This works, but it's verbose. You have to write filtering logic in every subscriber. The filtering is mixed with the reaction logic. And if multiple subscribers need the same filtering, you repeat yourself.

## The Solution: Conditional Observables

FynX gives you operators that create filtered, conditional observables. The key insight: separate the condition logic from the filtering operation.

```python
temperature = observable(20)

# First, create a boolean observable representing the condition
is_extreme = temperature >> (lambda temp: temp > 30 or temp < 0)

# Then, filter the original observable based on that condition
extreme_temps = temperature & is_extreme

extreme_temps.subscribe(lambda temp: {
    print(f"Temperature alert: {temp}°C")
})
```

Now the filtering is declarative and reusable. The `extreme_temps` observable only produces values when the condition is met.

## The & Operator: Boolean Composition

The `&` operator combines multiple boolean observables into compound boolean conditions. The result emits the source observable's value when ALL conditions are `True`, and does not emit when any condition becomes `False`:

```python
from fynx import observable

scores = observable(85)

# Create a boolean condition observable
is_high_score = scores >> (lambda score: score > 90)

# Filter the original scores based on the condition
high_scores = scores & is_high_score

high_scores.subscribe(lambda score: {
    print(f"🎉 High score achieved: {score}")
})

scores.set(88)  # No emission (condition became False)
scores.set(95)  # Prints: "🎉 High score achieved: 95" (condition became True)
scores.set(87)  # No emission (condition became False)
```

The `&` operator creates a **ConditionalObservable** that only emits when conditions are met, making it ideal for reactive boolean logic and state management.

## Using Conditional Observables with @reactive

The `&` operator creates conditional observables that work perfectly with `@reactive` for event-driven reactions:

```python
from fynx import observable, reactive

scores = observable(85)

# Create a boolean condition observable
is_high_score = scores >> (lambda score: score > 90)

# Use with @reactive for event-driven reactions
@reactive(is_high_score)
def on_high_score(is_high):
    if is_high:
        print(f"🎉 High score achieved: {scores.value}")

scores.set(88)  # No output (condition not met)
scores.set(95)  # Prints: "🎉 High score achieved: 95"
scores.set(87)  # No output (condition no longer met)
```

This pattern gives you event-driven reactions while maintaining the reactive paradigm.

### Complex Predicates with &

Your condition functions can be as complex as needed with the `&` operator:

```python
user = observable({"name": "Alice", "age": 30, "country": "US"})

# Complex validation logic as a boolean observable
is_valid_user = user >> (lambda u: {
    u["age"] >= 18 and
    u["country"] in ["US", "CA", "UK"] and
    len(u["name"]) > 2 and
    "@" not in u["name"]  # No emails in names
})

# Create compound boolean condition with & operator
valid_users = user & is_valid_user

valid_users.subscribe(lambda user_data: {
    if user_data is not None:
        print(f"✅ Valid user: {user_data['name']}")
    else:
        print("❌ User no longer valid")
})
```

The `&` operator emits `None` when validation fails, allowing you to react to both valid and invalid states.

## The ~ Operator: Logical Negation

The `~` operator inverts boolean conditions. It works on boolean observables:

```python
is_online = observable(True)

# Create a negated boolean observable
is_offline = ~is_online

# React when user goes offline (when is_offline becomes True)
offline_events = is_offline

offline_events.subscribe(lambda _: {
    print("User went offline")
})

is_online.set(False)  # is_offline becomes True, prints: "User went offline"
is_online.set(True)   # is_offline becomes False, no output
```

### Combining Negation with Filtering

Create "everything except" patterns:

```python
status = observable("loading")

# Create boolean condition for non-loading states
is_not_loading = status >> (lambda s: s != "loading")

# React to any status except "loading"
non_loading_status = status & is_not_loading

non_loading_status.subscribe(lambda status_val: {
    print(f"Status changed to: {status_val}")
})

status.set("loading")    # No output (filtered out)
status.set("success")    # Prints: "Status changed to: success"
status.set("error")      # Prints: "Status changed to: error"
```

## Real-World Example: Form Validation

Form validation is perfect for conditional observables with the `&` operator:

```python
email = observable("")
password = observable("")
terms_accepted = observable(False)

# Validation conditions as boolean observables
email_valid = email >> (lambda e: "@" in e and "." in e.split("@")[1])
password_strong = password >> (lambda p: len(p) >= 8)
terms_checked = terms_accepted >> (lambda t: t == True)

# Form is valid only when all conditions are true
form_valid = email & email_valid & password_strong & terms_checked

form_valid.subscribe(lambda is_valid: {
    if is_valid is not None:
        print("✅ Form is complete and valid!")
    else:
        print("❌ Form validation failed")
})

# Simulate form filling
email.set("user@")           # Prints: "❌ Form validation failed" (email invalid)
password.set("pass")         # Still invalid
terms_accepted.set(True)     # Still invalid

email.set("user@example.com")  # Still invalid (password too short)
password.set("secure123")      # Prints: "✅ Form is complete and valid!"
password.set("short")          # Prints: "❌ Form validation failed" (password weak)
```

The `&` operator emits `None` when validation fails, allowing you to handle both valid and invalid states in your UI.

## Advanced Patterns: State Machines with Conditionals

Build state machines using conditional logic:

```python
app_state = observable("initializing")
user_authenticated = observable(False)
data_loaded = observable(False)

# Define state conditions as boolean observables
is_app_ready = app_state >> (lambda s: s == "ready")
is_user_auth = user_authenticated >> (lambda a: a == True)
is_data_loaded = data_loaded >> (lambda d: d == True)
is_app_error = app_state >> (lambda s: s == "error")

# Combine conditions - app is ready when all are true
ready_state = app_state & is_app_ready & is_user_auth & is_data_loaded

# Error state
error_state = app_state & is_app_error

# React to state transitions
ready_state.subscribe(lambda _: {
    print("🚀 Application is fully ready!")
})

error_state.subscribe(lambda _: {
    print("❌ Application encountered an error")
})

# Simulate app lifecycle
app_state.set("authenticating")
user_authenticated.set(True)
app_state.set("loading_data")
data_loaded.set(True)
app_state.set("ready")  # Triggers "fully ready" message
```

## Conditional Operators with Derived Values

Combine conditionals with the `>>` operator for powerful data processing:

```python
sensor_readings = observable([])

# Create condition for sufficient data
has_enough_data = sensor_readings >> (lambda readings: len(readings) >= 3)

# Only process readings when we have enough data
valid_readings = sensor_readings & has_enough_data

# Then calculate statistics
average_reading = valid_readings >> (lambda readings: sum(readings) / len(readings))

average_reading.subscribe(lambda avg: {
    print(f"Average sensor reading: {avg:.2f}")
})

sensor_readings.set([1, 2])        # No output (not enough data)
sensor_readings.set([1, 2, 3, 4])  # Prints: "Average sensor reading: 2.50"
```

## Performance Benefits

Conditional observables improve performance by:

1. **Reducing unnecessary computations** - Only process data that meets criteria
2. **Filtering at the source** - Don't pass unwanted data to subscribers
3. **Early termination** - Stop reactive chains when conditions aren't met

```python
# Without conditionals - expensive operation runs on every change
raw_data = observable("some data")
processed_data = raw_data >> (lambda d: expensive_cleanup(d))
final_result = processed_data >> (lambda d: expensive_analysis(d))

# With conditionals - expensive operations only run when needed
clean_data = raw_data & (lambda d: is_worth_processing(d))
processed_data = clean_data >> (lambda d: expensive_cleanup(d))
final_result = processed_data >> (lambda d: expensive_analysis(d))
```

## Common Patterns

### Pattern 1: Threshold Monitoring

```python
temperature = observable(20)

# Create threshold conditions
is_hot = temperature >> (lambda t: t > 25)
is_cold = temperature >> (lambda t: t < 10)

# Alert when temperature crosses thresholds
hot_weather = temperature & is_hot
cold_weather = temperature & is_cold

hot_weather.subscribe(lambda t: print(f"🌡️ Hot: {t}°C"))
cold_weather.subscribe(lambda t: print(f"🧊 Cold: {t}°C"))
```

### Pattern 2: Data Quality Gates

```python
api_response = observable(None)

# Create validation condition
is_valid_response = api_response >> (lambda resp: {
    resp is not None and
    resp.get("status") == "success" and
    resp.get("data") is not None
})

# Only process successful responses with data
valid_responses = api_response & is_valid_response

valid_responses.subscribe(lambda resp: {
    process_data(resp["data"])
})
```

### Pattern 3: Feature Flags with Conditions

```python
feature_enabled = observable(False)
user_premium = observable(False)
experiment_active = observable(True)

# Create boolean conditions
is_feature_on = feature_enabled >> (lambda e: e == True)
is_premium_user = user_premium >> (lambda p: p == True)
is_experiment_on = experiment_active >> (lambda a: a == True)

# Feature is available only under specific conditions
can_use_feature = feature_enabled & is_feature_on & is_premium_user & is_experiment_on

can_use_feature.subscribe(lambda _: {
    enable_premium_feature()
})
```

## Gotchas and Best Practices

### Gotcha 1: Condition Functions Run Frequently

```python
# Bad - expensive condition runs on every change
slow_condition = data & (lambda d: expensive_validation(d))

# Better - cache expensive conditions
is_valid = data >> (lambda d: expensive_validation(d))
valid_data = is_valid & (lambda v: v == True)
```

### Gotcha 2: Negation Can Be Confusing

```python
flag = observable(True)

# This creates an observable that emits when flag becomes False
not_flag = ~flag

# But this doesn't do what you might expect
wrong_not_flag = flag & (lambda f: not f)  # Less clear than ~
```

### Best Practice: Keep Conditions Simple

```python
# Good - simple, focused conditions
is_adult = age & (lambda a: a >= 18)
has_permission = role & (lambda r: r in ["admin", "moderator"])

# Avoid - complex conditions
complex_check = user & (lambda u: {
    u["age"] >= 18 and
    u["role"] in ["admin", "moderator"] and
    u["verified"] and
    not u["banned"]
})
```

### Best Practice: Name Your Conditions

```python
def is_valid_email(email):
    return "@" in email and "." in email.split("@")[1]

def is_strong_password(pwd):
    return len(pwd) >= 8

# Much clearer than inline lambdas
email_ok = email & is_valid_email
password_ok = password & is_strong_password
form_valid = (email_ok & (lambda _: True)) & (password_ok & (lambda _: True))
```

## The Big Picture

Conditionals transform your reactive system from "process everything" to "process only what matters":

- **`&` operator**: Filter data streams based on predicates
- **`~` operator**: Invert boolean conditions
- **Performance**: Skip unnecessary computations
- **Clarity**: Separate filtering logic from reaction logic
- **Composition**: Combine conditions with other operators

Think of conditionals as reactive filters. They let you create observables that only emit valuable data, reducing noise and improving performance. Combined with transformations (`>>`) and reactions (`@reactive`), they give you a complete toolkit for building sophisticated reactive applications.

The next step is organizing these reactive pieces into reusable units called **Stores**—the architectural pattern that brings everything together.
