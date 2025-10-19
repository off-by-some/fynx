# @watch: Conditional Reactions

`@reactive` keeps everything synchronized, but not every change deserves a reaction. Some moments matter more than others—specific transitions, threshold crossings, state changes that unlock new capabilities.

But not every reaction should happen on *every* change. Sometimes you only care about specific moments:

- When form validation *passes* for the first time
- When a shopping cart *becomes* eligible for checkout
- When a user's login count *crosses* a threshold
- When a download *completes*
- When all required fields *become* filled

These aren't continuous reactions to every change. They're reactions to *events*—specific transitions in your application state. Writing these with `@reactive` leads to awkward code:

```python
# Awkward: Manual state tracking with @reactive
last_valid_state = False

@reactive(form_valid)
def on_validation_change(is_valid):
    global last_valid_state
    if is_valid and not last_valid_state:
        # Form just became valid
        print("✅ Validation passed!")
        enable_submit_button()
    last_valid_state = is_valid
```

You're tracking state manually, checking for transitions, maintaining flags. This is exactly the kind of bookkeeping that reactive programming should eliminate.

There's a better way.

## Introducing @watch

The `@watch` decorator runs functions when conditions become true. Instead of reacting to every change, you react to specific state transitions:

```python
from fynx import watch

@watch(lambda: form_valid.value)
def on_validation_pass():
    print("✅ Validation passed!")
    enable_submit_button()
```

That's it. No manual state tracking. No comparing previous and current values. Just a declaration: "when this condition becomes true, run this function."

The function runs *once* when the condition transitions from false to true, and then stays dormant until the condition becomes false and true again. This makes `@watch` perfect for event-driven reactions—things that should happen at specific moments, not continuously.

## How It Works: The Transition Model

Understanding when `@watch` functions run is crucial. They don't run on every change—they run on *transitions to true*:

```python
count = observable(0)

@watch(lambda: count.value > 5)
def on_threshold():
    print("Count exceeded 5!")

# Nothing happens yet - condition is false

count.set(3)   # Condition still false, nothing happens
count.set(6)   # Condition becomes true!
# Output: "Count exceeded 5!"

count.set(7)   # Condition stays true, nothing happens
count.set(8)   # Condition still true, nothing happens

count.set(2)   # Condition becomes false, nothing happens
count.set(10)  # Condition becomes true again!
# Output: "Count exceeded 5!"
```

The pattern is clear:
- **False → False** — Nothing happens
- **False → True** — Function runs
- **True → True** — Nothing happens
- **True → False** — Nothing happens (function doesn't run)
- **False → True** — Function runs again

The function fires on the *rising edge* of the condition—the moment it transitions from false to true. This makes `@watch` ideal for threshold crossings, status changes, and event detection.

## The Mental Model: State Transitions as Events

Traditional programming handles state transitions imperatively:

```python
# Traditional: Manual transition detection
def check_cart_status():
    if can_checkout() and not previously_could_checkout:
        show_checkout_button()
        previously_could_checkout = True
    elif not can_checkout() and previously_could_checkout:
        hide_checkout_button()
        previously_could_checkout = False
```

You're explicitly tracking previous state, comparing it to current state, and remembering to check on every relevant update. Miss a check and your UI gets out of sync.

With `@watch`, you declare the transition points:

```python
# Reactive: Declare what should happen at state transitions
@watch(lambda: can_checkout.value)
def on_checkout_available():
    show_checkout_button()

@watch(lambda: not can_checkout.value)
def on_checkout_unavailable():
    hide_checkout_button()
```

You've moved from "track and compare state" to "declare what happens when." The transition detection is automatic. The state tracking is automatic. You just express the events you care about.

## Writing Condition Functions

The `@watch` decorator accepts conditions that return boolean values. You have two main options for writing conditions:

### Lambda Conditions (Most Flexible)

Use lambda functions for any Python expression:

```python
age = observable(16)

@watch(lambda: age.value >= 18)
def on_becomes_adult():
    print("User is now an adult!")

age.set(17)  # Nothing happens
age.set(18)  # Prints: "User is now an adult!"
```

Lambda conditions can contain:
- **Complex boolean logic**: `lambda: temp > 30 or (humidity > 80 and ac_on)`
- **Mathematical calculations**: `lambda: cart.total * 0.9 > 100`
- **Function calls**: `lambda: user.is_eligible() and data.is_valid()`
- **Multi-variable expressions**: `lambda: (width * height) > min_area`

The condition function is checked every time any observable it reads changes. FynX automatically tracks dependencies and re-evaluates when observables change.

### ConditionalObservable Conditions (AND-Only)

Use the `&` operator to create AND conditions from observables:

```python
user_logged_in = observable(False)
data_loaded = observable(False)
notifications_enabled = observable(True)

# Simple AND combination
@watch(user_logged_in & data_loaded & notifications_enabled)
def show_dashboard():
    print("Welcome to your dashboard!")

# Can be complex with chained conditions
@watch(user_logged_in & data_loaded & notifications_enabled & cart_has_items)
def enable_checkout():
    print("Ready to checkout!")
```

The `&` operator creates ConditionalObservable objects that represent AND logic. You can chain as many conditions as needed - there's no limitation on complexity:

```python
# Complex chained conditions work perfectly
@watch(
    user_logged_in &
    data_loaded &
    notifications_enabled &
    (cart_total >> lambda x: x > 50) &  # Even with computed values
    (account_age >> lambda x: x >= 7)
)
def unlock_premium_features():
    print("Premium features unlocked!")
```

### Choosing Between Lambda and ConditionalObservable Conditions

**Use lambda conditions when you need:**
- OR logic: `lambda: temp > 30 or humidity > 80`
- Complex calculations: `lambda: cart.total * 0.9 > 100`
- Function calls: `lambda: user.is_eligible() and data.is_valid()`

**Use ConditionalObservable conditions (`&`) when you need:**
- Simple AND combinations: `a & b & c & d`
- Chain many conditions: `cond1 & cond2 & cond3 & ...`
- Mix with computed values: `logged_in & (age >> lambda x: x >= 18)`

Both approaches work equally well - choose based on readability and your specific needs.

## Multiple Conditions: AND Logic

You have two ways to achieve AND logic with `@watch`:

### Method 1: Multiple Lambda Conditions

Pass multiple condition functions - all must be true:

```python
has_items = observable(False)
is_logged_in = observable(False)

@watch(
    lambda: has_items.value,
    lambda: is_logged_in.value
)
def on_ready_to_checkout():
    print("Ready to checkout!")

has_items.set(True)      # Only one condition true, nothing happens
is_logged_in.set(True)   # Both now true!
# Output: "Ready to checkout!"
```

### Method 2: ConditionalObservable with `&` Operator

Use the `&` operator to combine conditions into a single ConditionalObservable:

```python
has_items = observable(False)
is_logged_in = observable(False)
payment_valid = observable(False)

@watch(has_items & is_logged_in & payment_valid)
def on_ready_to_checkout():
    print("Ready to checkout!")

has_items.set(True)      # Only one condition true, nothing happens
is_logged_in.set(True)   # Two conditions true, nothing happens
payment_valid.set(True)  # All three now true!
# Output: "Ready to checkout!"
```

Both approaches create AND logic where all conditions must be true. The ConditionalObservable approach can be more readable for simple boolean combinations, while multiple lambdas give you more flexibility for complex expressions.

The function fires when the *combined* condition transitions from false to true. If any individual condition becomes false and then true, but the others remain true, nothing happens. The entire combined condition must transition:

```python
@watch(
    lambda: has_items.value,
    lambda: is_logged_in.value
)
def on_ready():
    print("Ready!")

# Initial state: both false
has_items.set(True)        # Combined: False (one still false)
is_logged_in.set(True)     # Combined: True (both now true)
# Output: "Ready!"

has_items.set(False)       # Combined: False
has_items.set(True)        # Combined: True (back to all true)
# Output: "Ready!"
```

## Practical Example: Form Submission Flow

This example shows how @watch can be used to manage complex state transitions in a form submission process, from validation to completion:

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    terms_accepted = observable(False)
    is_submitting = observable(False)
    submission_complete = observable(False)

# Computed validations using the >> operator
email_valid = FormStore.email >> (
    lambda e: '@' in e and len(e) > 3
)

password_valid = FormStore.password >> (
    lambda p: len(p) >= 8
)

# Combine all validations
all_valid = (email_valid | password_valid | FormStore.terms_accepted) >> (
    lambda e, p, t: e and p and t
)

@watch(lambda: all_valid.value)
def on_form_valid():
    print("Form validation passed - submit button enabled")

@watch(lambda: FormStore.is_submitting.value)
def on_submit_start():
    print("Form submission started")

@watch(lambda: FormStore.submission_complete.value)
def on_submit_complete():
    print("Form submission completed successfully")

# Simulate user filling out the form
print("User fills out form:")
FormStore.email = "user@example.com"
print("Email set")

FormStore.password = "secure123"
print("Password set")

FormStore.terms_accepted = True
print("Terms accepted (all validations now pass, triggers form valid)")

# Simulate submission process
print("\nUser submits form:")
FormStore.is_submitting = True
print("Submission initiated")

# Simulate server response
print("\nServer processes submission:")
FormStore.submission_complete = True
print("Submission completed")

# Output:
# User fills out form:
# Email set
# Password set
# Terms accepted (all validations now pass, triggers form valid)
# Form validation passed - submit button enabled
#
# User submits form:
# Submission initiated
# Form submission started
#
# Server processes submission:
# Submission completed
# Form submission completed successfully
```

Each `@watch` declaration captures one transition in your workflow. The complete behavior emerges from these individual transition handlers, with no manual orchestration code needed.

## Watching Complex Conditions

Condition functions can contain arbitrary logic. Read multiple observables, perform calculations, call helper functions—anything that returns a boolean:

```python
class AnalyticsStore(Store):
    page_views = observable(0)
    time_on_site = observable(0)  # seconds
    items_clicked = observable(0)

@watch(
    lambda: (
        AnalyticsStore.page_views.value > 5 and
        AnalyticsStore.time_on_site.value > 60 and
        AnalyticsStore.items_clicked.value > 3
    )
)
def on_engaged_user():
    print("🎯 User is highly engaged!")
    show_upgrade_prompt()

@watch(
    lambda: (
        AnalyticsStore.page_views.value > 10 or
        AnalyticsStore.time_on_site.value > 300
    )
)
def on_power_user():
    print("⭐ Power user detected!")
    enable_advanced_features()
```

FynX tracks all observable reads inside your condition, regardless of how complex the logic. The condition re-evaluates whenever any of those observables change.

## One-Time Events vs. Repeating Events

`@watch` naturally handles both one-time events and repeating events:

```python
login_count = observable(0)

# One-time event: fires once, never again (unless condition resets)
@watch(lambda: login_count.value == 1)
def on_first_login():
    print("🎉 Welcome! This is your first login!")
    show_tutorial()

# Repeating event: fires every 10 logins
last_milestone = 0

@watch(lambda: login_count.value >= last_milestone + 10)
def on_login_milestone():
    global last_milestone
    last_milestone = login_count.value
    print(f"🏆 Milestone: {login_count.value} logins!")

# Simulate logins
for i in range(1, 25):
    login_count.set(i)

# Output:
# 🎉 Welcome! This is your first login!
# 🏆 Milestone: 10 logins!
# 🏆 Milestone: 20 logins!
```

For truly one-time events, you might track state manually. But for most cases, the natural false→true→false→true cycle of conditions handles repetition elegantly.

## Watching Store-Level Changes

Unlike `@reactive`, you can't pass entire Stores to `@watch`. Conditions must be explicit boolean expressions:

```python
# This won't work - @watch needs a condition function
@watch(UserStore)  # ERROR
def on_user_change():
    print("User changed")

# Instead, watch specific conditions derived from the Store
@watch(lambda: UserStore.is_authenticated.value)
def on_login():
    print("User logged in!")

@watch(
    lambda: (
        UserStore.profile_complete.value and
        not UserStore.is_premium.value
    )
)
def on_eligible_for_upgrade():
    print("User eligible for premium upgrade!")
```

This is by design. `@watch` is for *transitions*, and transitions require boolean conditions. If you need to react to any Store change, use `@reactive` instead.

## Combining @watch with Computed Observables

The real power emerges when you combine `@watch` with computed observables. Compute complex state, then watch for transitions:

```python
class ShoppingCartStore(Store):
    items = observable([])
    shipping_address = observable(None)
    payment_method = observable(None)

# Computed: Is cart ready for checkout?
has_items = ShoppingCartStore.items >> (lambda items: len(items) > 0)
has_shipping = ShoppingCartStore.shipping_address >> (lambda addr: addr is not None)
has_payment = ShoppingCartStore.payment_method >> (lambda pm: pm is not None)

can_checkout = (has_items | has_shipping | has_payment) >> (
    lambda items, shipping, payment: items and shipping and payment
)

# Watch the computed condition - lambda approach
@watch(lambda: can_checkout.value)
def on_checkout_ready():
    print("🛒 Ready to checkout!")
    enable_checkout_button()
    send_abandoned_cart_recovery_cancellation()

@watch(lambda: not can_checkout.value)
def on_checkout_not_ready():
    print("⏸️ Checkout not available")
    disable_checkout_button()
    schedule_abandoned_cart_email()

# Alternative: Use ConditionalObservables for cleaner AND logic
checkout_conditions = has_items & has_shipping & has_payment

@watch(checkout_conditions)
def on_checkout_ready_alt():
    print("🛒 Ready to checkout (ConditionalObservable)!")
    enable_checkout_button()

# Can even mix computed values in ConditionalObservables
cart_total = ShoppingCartStore.items >> (lambda items: sum(item['price'] for item in items))
has_minimum_total = cart_total >> (lambda total: total >= 25)

@watch(has_items & has_shipping & has_payment & has_minimum_total)
def on_full_checkout_ready():
    print("🛒 Ready to checkout with minimum total!")
    enable_checkout_button()
    show_free_shipping_banner()
```

The computed observable encapsulates the business logic. The `@watch` decorator handles the transition detection. Each piece has a single, clear responsibility.

## Common Patterns

**Pattern 1: Threshold crossings**

```python
score = observable(0)

@watch(lambda: score.value >= 100)
def on_level_complete():
    print("🎊 Level complete!")
    advance_to_next_level()
```

**Pattern 2: Status changes**

```python
@watch(lambda: ConnectionStore.is_connected.value)
def on_connect():
    print("✅ Connected to server")
    sync_data()

@watch(lambda: not ConnectionStore.is_connected.value)
def on_disconnect():
    print("❌ Disconnected from server")
    show_offline_mode()
```

**Pattern 3: Eligibility detection**

```python
@watch(
    lambda: UserStore.age.value >= 18,
    lambda: UserStore.has_verified_email.value,
    lambda: UserStore.account_age_days.value >= 7
)
def on_eligible_for_feature():
    print("🎁 New feature unlocked!")
    show_feature_announcement()
```

**Pattern 4: Completion tracking**

```python
@watch(
    lambda: DownloadStore.progress.value >= 100,
    lambda: not DownloadStore.has_error.value
)
def on_download_complete():
    print("✅ Download complete!")
    notify_user()
    start_installation()
```

**Pattern 5: Multi-step workflows**

```python
# Step 1: User starts onboarding
@watch(lambda: OnboardingStore.started.value)
def on_onboarding_start():
    print("👋 Starting onboarding...")
    show_welcome_screen()

# Step 2: User completes profile
@watch(lambda: OnboardingStore.profile_complete.value)
def on_profile_complete():
    print("✅ Profile complete!")
    show_preference_screen()

# Step 3: User sets preferences
@watch(lambda: OnboardingStore.preferences_complete.value)
def on_preferences_complete():
    print("✅ Preferences saved!")
    show_dashboard()
    OnboardingStore.finished = True

# Step 4: Onboarding finished
@watch(lambda: OnboardingStore.finished.value)
def on_onboarding_complete():
    print("🎉 Onboarding complete!")
    track_onboarding_completion()
```

## @watch vs. @reactive: When to Use Each

The choice between `@watch` and `@reactive` depends on what you're modeling:

**Use `@reactive` when you want continuous synchronization:**

```python
# Run on EVERY change to keep things in sync
@reactive(ThemeStore.mode)
def sync_theme(mode):
    update_css_variables(mode)
    save_preference('theme', mode)
```

**Use `@watch` when you want event-driven reactions:**

```python
# Run ONCE when condition becomes true
@watch(lambda: ThemeStore.mode.value == 'dark')
def on_dark_mode_enabled():
    print("🌙 Dark mode activated!")
    show_notification("Dark mode enabled")
```

**Use `@reactive` for:** UI updates, data synchronization, logging every change, keeping derived state current

**Use `@watch` for:** Notifications, workflow transitions, milestone tracking, one-time setup, threshold alerts

A rule of thumb: if you care about *what the value is*, use `@reactive`. If you care about *when something becomes true*, use `@watch`.

## Practical Example: User Engagement System

This example shows how @watch can be used to create sophisticated engagement tracking with multiple threshold levels and different types of events:

```python
class UserActivityStore(Store):
    page_views = observable(0)
    actions_taken = observable(0)
    time_on_site = observable(0)  # seconds
    has_account = observable(False)
    is_premium = observable(False)

# Computed engagement score: min(100, (views * 5) + (actions * 10) + (time / 6))
engagement_score = (
    UserActivityStore.page_views |
    UserActivityStore.actions_taken |
    UserActivityStore.time_on_site
) >> (
    lambda views, actions, time: min(
        100,
        (views * 5) + (actions * 10) + (time / 6)
    )
)

@watch(lambda: engagement_score.value >= 25)
def on_low_engagement():
    print(f"Low engagement reached (score: {engagement_score.value:.1f})")

@watch(lambda: engagement_score.value >= 50)
def on_medium_engagement():
    print(f"Medium engagement reached (score: {engagement_score.value:.1f})")

@watch(lambda: engagement_score.value >= 75)
def on_high_engagement():
    print(f"High engagement reached (score: {engagement_score.value:.1f})")

@watch(lambda: UserActivityStore.has_account.value)
def on_account_created():
    print("Account created - user registration completed")

# Simulate user engagement progression
print("User engagement progression:")

# Stage 1: Initial browsing (score: 22.5)
UserActivityStore.page_views = 3
UserActivityStore.time_on_site = 45
print("User browses 3 pages for 45 seconds (score: 22.5)")

# Stage 2: Low engagement reached (score: 42.5)
UserActivityStore.actions_taken = 2
print("User takes 2 actions (score: 42.5, triggers low engagement)")

# Stage 3: Medium engagement reached (score: 57.5)
UserActivityStore.page_views = 6
print("User views 3 more pages (score: 57.5, triggers medium engagement)")

# Stage 4: Account creation (separate event)
UserActivityStore.has_account = True
print("User creates account (triggers account creation event)")

# Stage 5: Extended engagement (score: 70, still medium)
UserActivityStore.time_on_site = 120
print("User spends more time (score: 70, still medium engagement)")

# Stage 6: High engagement reached (score: 80)
UserActivityStore.time_on_site = 180
print("User spends even more time (score: 80, triggers high engagement)")

# Output:
# User engagement progression:
# User browses 3 pages for 45 seconds (score: 22.5)
# Low engagement reached (score: 42.5)
# User takes 2 actions (score: 42.5, triggers low engagement)
# Medium engagement reached (score: 57.5)
# User views 3 more pages (score: 57.5, triggers medium engagement)
# User creates account (triggers account creation event)
# Account created - user registration completed
# User spends more time (score: 70, still medium engagement)
# High engagement reached (score: 80.0)
# User spends even more time (score: 80, triggers high engagement)
```

The `@reactive` decorators handle continuous monitoring—updating displays and tracking metrics. The `@watch` decorators handle discrete events—milestones, state transitions, business logic triggers. Together they create a complete reactive system.

## Gotchas and Edge Cases

**1. Initial state doesn't trigger @watch**

```python
# Condition starts as true
ready = observable(True)

@watch(lambda: ready.value)
def on_ready():
    print("Ready!")

# Nothing prints - no transition occurred
# Function only runs when condition goes false → true
```

If you need to handle initial state, check it explicitly or use `@reactive` instead:

```python
# Option 1: Check initial state
@watch(lambda: ready.value)
def on_ready():
    print("Ready!")

if ready.value:
    on_ready()  # Call manually for initial state

# Option 2: Use @reactive for immediate execution
@reactive(ready)
def on_ready_immediate(is_ready):
    if is_ready:
        print("Ready!")
```

**2. Condition functions should be pure**

```python
# Bad: Side effects in condition
@watch(lambda: (print("Checking..."), count.value > 5)[1])
def on_threshold():
    print("Threshold reached!")

# The print runs every time the condition is checked,
# not just when it becomes true
```

Condition functions are checked frequently. Keep them pure—no side effects, just boolean logic. Save side effects for the decorated function.

**3. Complex conditions can mask transitions**

```python
count = observable(0)

@watch(lambda: count.value > 5 and count.value < 10)
def in_range():
    print("In range!")

count.set(7)   # Prints: "In range!"
count.set(8)   # Nothing (condition still true)
count.set(15)  # Nothing (condition becomes false)
count.set(7)   # Prints: "In range!" (condition true again)
count.set(5)   # Nothing (condition becomes false)
count.set(6)   # Prints: "In range!" (condition true again)
```

The condition only cares about true/false transitions, not the specific values that made it true. Design your conditions to capture the transitions you care about.

**4. Watch decorators don't compose**

```python
# This doesn't do what you might expect
@watch(lambda: condition_a.value)
@watch(lambda: condition_b.value)
def on_either():
    print("Something happened!")

# Only the innermost @watch takes effect
```

Each `@watch` creates a separate conditional reaction. To watch multiple independent conditions, create separate functions:

```python
@watch(lambda: condition_a.value)
def on_condition_a():
    handle_event()

@watch(lambda: condition_b.value)
def on_condition_b():
    handle_event()
```

**5. Beware of condition evaluation cost**

```python
# Expensive condition checked on every change
@watch(lambda: expensive_computation(data.value) > threshold.value)
def on_expensive_trigger():
    print("Triggered!")
```

The condition function runs every time any observable it reads changes. For expensive checks, compute the result once and watch the computed observable:

```python
# Better: Compute once, watch the result
expensive_result = data >> expensive_computation
is_above_threshold = (expensive_result | threshold) >> (
    lambda result, thresh: result > thresh
)

@watch(lambda: is_above_threshold.value)
def on_trigger():
    print("Triggered!")
```

## Debugging Watch Conditions

When a `@watch` function isn't firing as expected, debug the condition:

```python
count = observable(0)

condition = lambda: count.value > 5

# Debug version
def debug_condition():
    result = count.value > 5
    print(f"Condition check: count={count.value}, result={result}")
    return result

@watch(debug_condition)
def on_threshold():
    print("Threshold crossed!")

count.set(3)  # See: Condition check: count=3, result=False
count.set(7)  # See: Condition check: count=7, result=True
              # Then: Threshold crossed!
```

Or extract the condition logic for testing:

```python
def is_eligible(user):
    return (
        user.age >= 18 and
        user.email_verified and
        user.account_age_days >= 7
    )

# Test the condition independently
assert is_eligible(test_user) == True

# Use it in @watch
@watch(lambda: is_eligible(UserStore))
def on_eligible():
    unlock_feature()
```

## Performance Considerations

`@watch` is efficient—conditions are only re-evaluated when observables they read actually change. But consider:

**Many watchers on the same observables:**

```python
# Each watcher triggers independently
@watch(lambda: score.value >= 100) def on_level_1(): ...
@watch(lambda: score.value >= 200) def on_level_2(): ...
@watch(lambda: score.value >= 300) def on_level_3(): ...
# ... 50 more watchers

score.set(150)  # All 50 conditions are checked
```

For many similar conditions, consider a single reaction with conditional logic:

```python
@reactive(score)
def check_milestones(current_score):
    if current_score >= 300 and not unlocked['level_3']:
        on_level_3()
        unlocked['level_3'] = True
    elif current_score >= 200 and not unlocked['level_2']:
        on_level_2()
        unlocked['level_2'] = True
    # ...
```

**Frequently changing observables:**

```python
mouse_position = observable((0, 0))

@watch(lambda: is_hovering_button(mouse_position.value))
def on_hover():
    show_tooltip()

# Condition checked on EVERY mouse move
```

For high-frequency updates, debounce or throttle:

```python
# Debounced hover detection
hover_stable = mouse_position >> (
    lambda pos: debounce(lambda: is_hovering_button(pos), 100)
)

@watch(lambda: hover_stable.value)
def on_stable_hover():
    show_tooltip()
```

## Real-World Example: Order Processing Pipeline

This example shows how @watch can orchestrate complex multi-step workflows where each stage depends on the completion of previous stages:

```python
class OrderStore(Store):
    items = observable([])
    payment_verified = observable(False)
    inventory_reserved = observable(False)
    shipping_label_created = observable(False)
    shipped = observable(False)
    delivered = observable(False)

# Pipeline stages - each @watch represents a transition to the next stage
@watch(lambda: len(OrderStore.items.value) > 0)
def on_order_created():
    print("Order created - items added to cart")

@watch(
    lambda: OrderStore.payment_verified.value,
    lambda: len(OrderStore.items.value) > 0
)
def on_payment_verified():
    print("Payment verified - proceeding to inventory check")
    # Simulate automatic inventory reservation
    OrderStore.inventory_reserved = True

@watch(lambda: OrderStore.inventory_reserved.value)
def on_inventory_reserved():
    print("Inventory reserved - generating shipping label")
    # Simulate automatic label creation
    OrderStore.shipping_label_created = True

@watch(lambda: OrderStore.shipping_label_created.value)
def on_label_created():
    print("Shipping label created - order ready for shipment")

@watch(lambda: OrderStore.shipped.value)
def on_shipped():
    print("Order shipped - tracking number generated")

@watch(lambda: OrderStore.delivered.value)
def on_delivered():
    print("Order delivered - customer notified")

# Execute the order processing pipeline
print("Processing customer order:")

# Stage 1: Customer adds items
OrderStore.items = [
    {'name': 'Widget', 'price': 29.99},
    {'name': 'Gadget', 'price': 49.99}
]
print("Items added to cart")

# Stage 2: Customer completes payment
OrderStore.payment_verified = True
print("Payment processed")

# Subsequent stages are triggered automatically by the @watch decorators
# Each stage advances the order through the pipeline

# Stage 5: Warehouse ships the order
OrderStore.shipped = True
print("Order shipped from warehouse")

# Stage 6: Order is delivered
OrderStore.delivered = True
print("Order delivered to customer")

# Output:
# Processing customer order:
# Order created - items added to cart
# Items added to cart
# Payment verified - proceeding to inventory check
# Inventory reserved - generating shipping label
# Shipping label created - order ready for shipment
# Payment processed
# Order shipped - tracking number generated
# Order shipped from warehouse
# Order delivered - customer notified
# Order delivered to customer
```

Each stage of the pipeline is a separate `@watch` declaration. The complete workflow emerges from these independent transition handlers, with no central orchestration code. Add a new stage? Just add another `@watch`. Remove a stage? Delete the corresponding watch. The pipeline is self-documenting and maintainable.

## Summary

The `@watch` decorator runs functions when conditions transition from false to true:

- **Event-driven reactions** — Respond to specific state transitions, not every change
- **Boolean conditions** — Pass lambda functions or ConditionalObservables (created with `&`)
- **Automatic dependency tracking** — Conditions re-evaluate when observables they read change
- **Multiple conditions** — Combine conditions with AND logic using multiple lambdas or `&` operator
- **One-time per transition** — Function runs once when condition becomes true, then waits for false→true again
- **Compose with computed observables** — Separate business logic (computed) from event handling (watch)

When `@reactive` gives you continuous synchronization, `@watch` gives you precise event detection. Together they cover the full spectrum of reactive behaviors—from "keep this in sync" to "do this when that happens."

With observables, stores, `@reactive`, and `@watch`, you have everything you need to build sophisticated reactive applications where state changes automatically propagate through your system, and important transitions trigger the right behaviors at the right times.
