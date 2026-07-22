# Best Practices

This page collects the best practices, common patterns, and gotchas from across the tutorial in one place, organized by topic. If you're looking for the core concepts themselves, see [Transforming Data](derived-observables.md), [Conditionals](conditionals.md), [Stores](stores.md), and [@reactive](using-reactive.md).

## Transformations

### Keep Transformations Pure

```python
# Good - Pure function, same input always gives same output
def to_uppercase(n):
    return n.upper()

uppercase = name.then(to_uppercase)

# Avoid - Impure function, depends on external state
import random

def random_case(n):
    return n.upper() if random.random() > 0.5 else n.lower()

random_case_result = name.then(random_case)  # Unpredictable
```

Pure functions make your reactive system predictable and testable.

### Handle Edge Cases

```python
# Good - Handles empty lists gracefully
def safe_average(nums):
    return sum(nums) / len(nums) if nums else 0

average = numbers.then(safe_average)

# Avoid - Will crash on empty list
def unsafe_average(nums):
    return sum(nums) / len(nums)

unsafe_result = numbers.then(unsafe_average)  # Crashes on empty list
```

Defensive programming prevents runtime errors in your reactive pipelines.

### Name Your Transformations

```python
# Clear intent
def calculate_age(date):
    return (datetime.now() - date).days // 365

def is_adult(age):
    return age >= 18

user_age = birth_date.then(calculate_age)
is_adult_result = user_age.then(is_adult)

eligible_for_voting = is_adult_result & has_citizenship

# Unclear intent
def transform(d):
    return calculate_age(d)

def filter_age(age):
    return age >= 18

transformed = birth_date.then(transform)
filtered = transformed.then(filter_age)
```

Descriptive names make your reactive graphs self-documenting.

### Avoid Deep Nesting

```python
# Good - Break complex transformations into steps
def extract_user_data(response):
    return response['user']

def extract_user_age(user_data):
    return user_data['age']

def is_adult(age):
    return age >= 18

user_data = api_response.then(extract_user_data)
user_age = user_data.then(extract_user_age)
is_adult_result = user_age.then(is_adult)

# Avoid - Hard to debug and modify
def complex_extraction(response):
    return response['user']['age'] >= 18

complex_result = api_response.then(complex_extraction)
```

Small, focused transformations are easier to test and maintain.

### Consider Performance

```python
# Good - Efficient for large lists
def sum_list(lst):
    return sum(lst)

summed = large_list.then(sum_list)

# Better - Lazy evaluation with generator
def sum_generator(lst):
    return sum(x for x in lst)

summed_lazy = large_list.then(sum_generator)
```

### Common Transformation Patterns

**Data Validation**

```python
email = observable("user@")

def validate_email(e):
    return "@" in e and "." in e.split("@")[1]

def email_feedback(valid):
    return "Valid" if valid else "Invalid"

is_valid_email = email.then(validate_email)
feedback = is_valid_email.then(email_feedback)
```

**Data Formatting**

```python
price = observable(29.99)

def format_price(p):
    return f"${p:.2f}"

formatted_price = price.then(format_price)
```

**Collection Operations**

```python
items = observable([1, 2, 3, 4, 5])

def filter_evens(lst):
    return [x for x in lst if x % 2 == 0]

def double_items(lst):
    return [x * 2 for x in lst]

def sum_items(lst):
    return sum(lst)

# Filter
evens = items.then(filter_evens)

# Map
doubled = items.then(double_items)

# Reduce
total = items.then(sum_items)
```

**State Derivation**

```python
app_state = observable("loading")

def is_loading_state(s):
    return s == "loading"

def is_error_state(s):
    return s == "error"

def is_ready_state(s):
    return s == "ready"

is_loading = app_state.then(is_loading_state)
is_error = app_state.then(is_error_state)
is_ready = app_state.then(is_ready_state)
```

## Conditionals

### Give Reused Conditions a Name

A condition inlined as a lambda inside `@` still only re-evaluates when its dependencies change - it isn't recomputed on every read. But if more than one gate needs the same expensive check, an inline lambda runs it once per gate, since each `@` sees a separate, unrelated function object:

```python
# Bad - expensive_validation runs twice per change, once per gate
valid_for_display = data @ (lambda d: expensive_validation(d))
valid_for_export = data @ (lambda d: expensive_validation(d))

# Better - name the condition once, reuse it everywhere it's needed
is_valid = data.then(lambda d: expensive_validation(d))
valid_for_display = data @ is_valid
valid_for_export = data @ is_valid
```

### Negation Can Be Confusing

```python
flag = observable(True)

# A total boolean: notifies on every change, in either direction
not_flag = flag.negate()

# A gate, not a negation: only emits flag's value while flag is falsy - it
# stays silent while flag is True, and silent again once flag flips back
# to True, rather than tracking the negated value the way .negate() does
wrong_not_flag = flag @ (lambda f: not f)  # Not equivalent to flag.negate()
```

### Keep Conditions Simple

```python
# Good - simple, focused conditions
is_adult = age.then(lambda a: a >= 18)
has_permission = role.then(lambda r: r in ["admin", "moderator"])

# Avoid - complex conditions
complex_check = user @ (lambda u: (
    u["age"] >= 18 and
    u["role"] in ["admin", "moderator"] and
    u["verified"] and
    not u["banned"]
))
```

### Name Your Conditions

```python
def is_valid_email(email):
    return "@" in email and "." in email.split("@")[1]

def is_strong_password(pwd):
    return len(pwd) >= 8

# Much clearer than inline lambdas
email_ok = email @ is_valid_email             # gate: emits email while it's valid
password_ok = password @ is_strong_password   # gate: emits password while it's strong

# To combine named predicates into a single "form is valid" boolean, build
# them as plain booleans and combine with `.all()` / `&` -
# don't nest `.requiring()`:
email_valid = email.then(is_valid_email)
password_valid = password.then(is_strong_password)
form_valid = email_valid & password_valid
```

### Common Conditional Patterns

**Threshold Monitoring**

```python
temperature = observable(20)

# Create threshold conditions
is_hot = temperature.then(lambda t: t > 25)
is_cold = temperature.then(lambda t: t < 10)

# Alert when temperature crosses thresholds
hot_weather = temperature @ is_hot
cold_weather = temperature @ is_cold

hot_weather.subscribe(lambda t: print(f"🌡️ Hot: {t}°C"))
cold_weather.subscribe(lambda t: print(f"🧊 Cold: {t}°C"))
```

**Data Quality Gates**

```python
api_response = observable(None)

# Create validation condition
is_valid_response = api_response.then(lambda resp: (
    resp is not None and
    resp.get("status") == "success" and
    resp.get("data") is not None
))

# Only process successful responses with data
valid_responses = api_response @ is_valid_response

valid_responses.subscribe(lambda resp: process_data(resp["data"]))
```

**Feature Flags with Conditions**

```python
feature_enabled = observable(False)
user_premium = observable(False)
experiment_active = observable(True)

# Create boolean conditions
is_feature_on = feature_enabled.then(lambda e: e == True)
is_premium_user = user_premium.then(lambda p: p == True)
is_experiment_on = experiment_active.then(lambda a: a == True)

# Feature is available only under specific conditions
can_use_feature = is_feature_on & is_premium_user & is_experiment_on

can_use_feature.subscribe(lambda _: enable_premium_feature())
```

## Stores

### Keep Stores Focused on a Single Domain

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

### Use Class Methods for State Modifications

Encapsulate how state changes:

```python
# Good: Clear API
@classmethod
def add_item(cls, item):
    cls.items = cls.items + [item]

# Bad: Direct manipulation everywhere
SomeStore.items = SomeStore.items + [item]
```

### Always Create New Values, Never Mutate

Store attributes follow the same immutable-update rule as standalone observables - see [Stores: Immutable Updates](stores.md#a-critical-pattern-immutable-updates) for the full explanation and the `AttributeError` / silent-no-op pitfalls that come with it.

### Handle Edge Cases in Computed Values

Computed values should be defensive:

```python
# Good: Handles empty list
average = values.then(
    lambda vals: sum(vals) / len(vals) if len(vals) > 0 else 0
)

# Good: Handles None
user_name = user.then(
    lambda u: u['name'] if u and 'name' in u else "Guest"
)
```

### Name Computed Values Clearly

Use names that indicate derivation:

```python
# Good: Clear that these are derived
is_valid = email.then(lambda e: '@' in e)
item_count = items.then(len)
total_price = items.then(lambda items: sum(item['price'] for item in items))

# Less clear:
valid = email.then(lambda e: '@' in e)
count = items.then(len)
price = items.then(lambda items: sum(item['price'] for item in items))
```

## @reactive

### Use @reactive for Side Effects Only

**✅ USE @reactive for:**

**Side Effects at Application Boundaries**

```python
@reactive(UserStore)
def sync_to_server(store):
    # Network I/O
    api.post('/user/update', store.to_dict())

@reactive(settings_changed)
def save_to_local_storage(settings):
    # Browser storage I/O
    localStorage.setItem('settings', json.dumps(settings))
```

**UI Updates** (DOM manipulation, rendering)

```python
@reactive(cart_total)
def update_total_display(total):
    # DOM manipulation
    render_total(f"${total:.2f}")
```

**Logging and Monitoring**

```python
@reactive(error_observable)
def log_errors(error):
    # Logging side effect
    logger.error(f"Application error: {error}")
    analytics.track('error', {'message': str(error)})
```

**Cross-System Coordination** (when one reactive system needs to update another)

```python
@reactive(ThemeStore.dark_mode)
def update_editor_theme(is_dark):
    # Coordinating separate systems
    EditorStore.theme = 'dark' if is_dark else 'light'
```

**❌ AVOID @reactive for:**

**Deriving State** (use `.then()`, `.alongside()`, `.all()`, `.either()`, `.negate()`, and `.requiring()` / `@` instead)

```python
# BAD: Using @reactive for transformation
@reactive(count)
def doubled_count(value):
    doubled.set(value * 2)  # Modifying another observable

# GOOD: Use functional transformation
doubled = count.then(lambda x: x * 2)
```

**State Coordination** (use computed observables)

```python
# BAD: Coordinating state in reactive functions
@reactive(first_name, last_name)
def update_full_name(first, last):
    full_name.set(f"{first} {last}")

# GOOD: Express as derived state
full_name = first_name.alongside(last_name).then(lambda f, l: f"{f} {l}")
```

**Business Logic** (keep logic in pure transformations)

```python
# BAD: Business logic in reactive function
@reactive(order_items)
def calculate_total(items):
    total = sum(item.price * item.quantity for item in items)
    tax = total * 0.08
    order_total.set(total + tax)

# GOOD: Business logic in pure transformation
order_total = order_items.then(lambda items:
    sum(i.price * i.quantity for i in items) * 1.08
)
```

### Anti-Patterns to Avoid

**Self-mutation.** When a reaction modifies the very thing it's watching, you'd expect an infinite feedback cycle:

```python
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)  # Modifying the observable this reaction watches
```

FynX doesn't actually let this loop forever. Decoration runs the function once immediately (`0 → 1`) before the subscription is even registered, so that first call succeeds quietly. But once subscribed, any *later* external change is different:

```python
count.set(5)
# Raises RuntimeError: Circular dependency detected in reactive computation!
# FynX detects that increment_forever is trying to modify `count` while
# running in response to a `count` change, and refuses rather than looping.
```

So this doesn't silently run away - it fails loudly the first time it would actually recurse. Still worth avoiding: relying on the framework to catch a self-mutation is worse than not writing one, and the failure is harder to spot when the dependency is indirect.

**The hidden cache.** When reactions maintain their own state, you've split your application's state across two systems:

```python
results_cache = {}

@reactive(query)
def update_cache(query_string):
    results_cache[query_string] = fetch_results(query_string)
```

Now you have to remember that `results_cache` exists and keep it synchronized. Better to make the cache itself observable and derive from it.

**The sequential assumption.** When reactions depend on each other's execution order, you've created fragile coupling:

```python
shared_list = []

@reactive(data)
def reaction_one(value):
    shared_list.append(value)

@reactive(data)
def reaction_two(value):
    # Assumes reaction_one has already run
    print(f"List has {len(shared_list)} items")
```

The second reaction assumes the first has already run. But that's an implementation detail, not a guarantee. If execution order changes, your code breaks silently.

The fix for all three is the same: keep reactions independent and stateless. Let the observable system coordinate state. Keep reactions purely about effects.

### Advanced Pattern: Conditional Guards and Cleanup

The conditional operators shine when you need to guard expensive or sensitive operations:

```python
user = observable(None)
has_permission = observable(False)
is_online = observable(False)

# Gates `user`'s value by has_permission and is_online: the function only
# runs while both conditions hold, and receives user's current value
# (which may still be None if no one has logged in yet).
@reactive(user @ (has_permission & is_online))
def sync_sensitive_data(current_user):
    if current_user is not None:
        api.sync_user_data(current_user.id)

# Later, when you want to stop syncing entirely:
sync_sensitive_data.unsubscribe()
```

The unsubscribe mechanism becomes particularly important in cleanup scenarios. If your reactive function represents a resource that needs explicit teardown (like a WebSocket connection or a file handle), you can unsubscribe when you're done to prevent further reactions and then perform cleanup in the function itself.

### Gotchas

**Prefer explicit reaction inputs**

```python
# Harder to scan: the second dependency is inside the function body
other_count = observable(10)

@reactive(count)
def show_sum(value):
    print(f"Sum: {value + other_count.value}")
# Prints immediately: "Sum: 10"

count.set(5)  # Prints: "Sum: 15"
other_count.set(20)  # Prints: "Sum: 25"
```

FynX tracks `.value` reads during reactive execution, so this still updates. For code that teammates can understand at a glance, prefer making reaction inputs explicit:

```python
@reactive(count, other_count)
def show_sum(value, other):
    print(f"Sum: {value + other}")
```

**Reactive functions receive values, not observables**

```python
# BAD: Trying to modify the value parameter
@reactive(count)
def try_to_modify(value):
    value.set(100)  # ERROR: value is an int, not an observable
```

Solution: Access the observable directly if you need to modify it (though you usually shouldn't).

**Store reactions receive snapshots**

```python
@reactive(UserStore)
def save_user(store):
    # store is a snapshot of UserStore at this moment
    # store.name is the current value, not an observable
    save_to_db(store.name)  # Correct

    # This won't work:
    store.name.subscribe(handler)  # ERROR
```

**Order-dependent reactions are fragile**

Express dependencies through computed observables instead of relying on execution order between multiple reactions.

### Performance

Reactive functions run synchronously on every change. For expensive operations, consider:

**Reacting to derived state that filters changes:**

```python
search_query = observable("")

# Only changes when meaningful
filtered_results = search_query.then(
    lambda q: search_database(q) if len(q) >= 3 else []
)

@reactive(filtered_results)
def update_ui(results):
    display_results(results)
```

**Using conditional observables to limit when reactions fire:**

```python
should_update = user_active & ~is_loading

@reactive(should_update)
def update_display(should):
    if should:
        expensive_render()
```

**Conditional logic inside reactions:**

```python
@reactive(mouse_position)
def update_tooltip(position):
    if should_show_tooltip(position):
        expensive_tooltip_render(position)
```
