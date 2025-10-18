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

The `@watch` decorator accepts a function that returns a boolean—this is your condition. Inside this function, you read observable values using `.value`:

```python
age = observable(16)

@watch(lambda: age.value >= 18)
def on_becomes_adult():
    print("User is now an adult!")

age.set(17)  # Nothing happens
age.set(18)  # Prints: "User is now an adult!"
```

The condition function is checked every time any observable it reads changes. When you access `age.value` inside the lambda, `@watch` automatically tracks that dependency and re-evaluates the condition whenever `age` changes.

This automatic dependency tracking is what makes `@watch` powerful. You don't manually list which observables to watch—just write a condition that reads them, and FynX figures out the dependencies.

## Multiple Conditions: AND Logic

You can pass multiple condition functions to `@watch`. The decorated function runs when *all* conditions become true simultaneously:

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

This is AND logic: `condition1 AND condition2 AND condition3...`. All must be true for the function to run. This is perfect for complex state requirements like "user is logged in AND has items in cart AND has a valid payment method."

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

Here's where `@watch` shines—expressing multi-step workflows as declarative transitions:

```python
class FormStore(Store):
    email = observable("")
    password = observable("")
    terms_accepted = observable(False)
    is_submitting = observable(False)
    submission_complete = observable(False)

# Computed validations
email_valid = FormStore.email >> (
    lambda e: '@' in e and len(e) > 3
)

password_valid = FormStore.password >> (
    lambda p: len(p) >= 8
)

all_valid = (email_valid | password_valid | FormStore.terms_accepted) >> (
    lambda e, p, t: e and p and t
)

# Watch for form becoming submittable
@watch(lambda: all_valid.value)
def on_form_valid():
    print("✅ Form is valid - submit button enabled")
    enable_submit_button()

# Watch for form becoming invalid (using NOT)
@watch(lambda: not all_valid.value)
def on_form_invalid():
    print("❌ Form is invalid - submit button disabled")
    disable_submit_button()

# Watch for submission starting
@watch(lambda: FormStore.is_submitting.value)
def on_submit_start():
    print("🔄 Submitting form...")
    show_loading_spinner()
    actual_submit_to_server()

# Watch for submission completing
@watch(lambda: FormStore.submission_complete.value)
def on_submit_complete():
    print("✅ Submission complete!")
    hide_loading_spinner()
    show_success_message()
    FormStore.is_submitting = False

# Simulate the flow
print("Initial state:")
FormStore.email = "user@example.com"
FormStore.password = "secure123"
FormStore.terms_accepted = True
# Output: "✅ Form is valid - submit button enabled"

print("\nUser clicks submit:")
FormStore.is_submitting = True
# Output: "🔄 Submitting form..."

print("\nServer responds:")
FormStore.submission_complete = True
# Output: "✅ Submission complete!"
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

# Watch the computed condition
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

Let's build a complete engagement tracking system that demonstrates both decorators working together:

```python
class UserActivityStore(Store):
    page_views = observable(0)
    actions_taken = observable(0)
    time_on_site = observable(0)  # seconds
    has_account = observable(False)
    is_premium = observable(False)

# Computed engagement score (0-100)
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

# Continuous monitoring with @reactive
@reactive(engagement_score)
def update_engagement_display(score):
    print(f"📊 Engagement: {score:.0f}/100")
    update_progress_bar(score)

@reactive(UserActivityStore.page_views)
def track_page_views(views):
    analytics.track('page_view_count', views)

# Event-driven reactions with @watch

# Engagement milestones
@watch(lambda: engagement_score.value >= 25)
def on_low_engagement():
    print("🟡 User is browsing")

@watch(lambda: engagement_score.value >= 50)
def on_medium_engagement():
    print("🟠 User is engaged!")
    show_tooltip("Enjoying the site? Create an account!")

@watch(lambda: engagement_score.value >= 75)
def on_high_engagement():
    print("🔴 User is highly engaged!")
    show_modal("Love what you see? Try Premium!")

# Account creation flow
@watch(lambda: UserActivityStore.has_account.value)
def on_account_created():
    print("🎉 Account created!")
    send_welcome_email()
    unlock_saved_features()

# Premium conversion
@watch(
    lambda: UserActivityStore.has_account.value,
    lambda: engagement_score.value >= 60,
    lambda: not UserActivityStore.is_premium.value
)
def on_premium_eligible():
    print("💎 User eligible for Premium!")
    show_upgrade_offer()

@watch(lambda: UserActivityStore.is_premium.value)
def on_premium_conversion():
    print("🚀 User upgraded to Premium!")
    send_thank_you_email()
    enable_premium_features()
    track_conversion()

# Simulate user journey
print("=== User starts browsing ===")
UserActivityStore.page_views = 3
UserActivityStore.time_on_site = 45

print("\n=== User interacts more ===")
UserActivityStore.actions_taken = 2
UserActivityStore.page_views = 6

print("\n=== User creates account ===")
UserActivityStore.has_account = True
UserActivityStore.time_on_site = 120

print("\n=== User upgrades to premium ===")
UserActivityStore.is_premium = True

# Output shows both continuous updates and discrete events:
# 📊 Engagement: 45/100
# 🟡 User is browsing
# 📊 Engagement: 60/100
# 🟠 User is engaged!
# 🎉 Account created!
# 💎 User eligible for Premium!
# 📊 Engagement: 80/100
# 🔴 User is highly engaged!
# 🚀 User upgraded to Premium!
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

Let's build a complete order processing system that shows the full power of `@watch`:

```python
class OrderStore(Store):
    items = observable([])
    payment_verified = observable(False)
    inventory_reserved = observable(False)
    shipping_label_created = observable(False)
    shipped = observable(False)
    delivered = observable(False)

# Computed: Order has items
has_items = OrderStore.items >> (lambda items: len(items) > 0)

# Stage 1: Order ready for payment
@watch(lambda: has_items.value)
def on_order_created():
    print("📝 Order created - awaiting payment")
    send_order_confirmation_email()

# Stage 2: Payment verified
@watch(
    lambda: OrderStore.payment_verified.value,
    lambda: has_items.value
)
def on_payment_verified():
    print("💳 Payment verified - reserving inventory")
    reserve_inventory_for_order()
    # Simulating async inventory reservation
    OrderStore.inventory_reserved = True

# Stage 3: Inventory reserved
@watch(lambda: OrderStore.inventory_reserved.value)
def on_inventory_reserved():
    print("📦 Inventory reserved - creating shipping label")
    create_shipping_label()
    OrderStore.shipping_label_created = True

# Stage 4: Shipping label created
@watch(lambda: OrderStore.shipping_label_created.value)
def on_label_created():
    print("🏷️ Shipping label created - ready to ship")
    notify_warehouse_to_ship()

# Stage 5: Order shipped
@watch(lambda: OrderStore.shipped.value)
def on_shipped():
    print("🚚 Order shipped!")
    send_shipping_notification()
    start_delivery_tracking()

# Stage 6: Order delivered
@watch(lambda: OrderStore.delivered.value)
def on_delivered():
    print("✅ Order delivered!")
    send_delivery_confirmation()
    request_review()
    close_order()

# Simulate the order pipeline
print("=== Customer adds items ===")
OrderStore.items = [
    {'name': 'Widget', 'price': 29.99},
    {'name': 'Gadget', 'price': 49.99}
]

print("\n=== Payment processed ===")
OrderStore.payment_verified = True

print("\n=== Warehouse ships order ===")
OrderStore.shipped = True

print("\n=== Order delivered ===")
OrderStore.delivered = True

# Output:
# === Customer adds items ===
# 📝 Order created - awaiting payment
#
# === Payment processed ===
# 💳 Payment verified - reserving inventory
# 📦 Inventory reserved - creating shipping label
# 🏷️ Shipping label created - ready to ship
#
# === Warehouse ships order ===
# 🚚 Order shipped!
#
# === Order delivered ===
# ✅ Order delivered!
```

Each stage of the pipeline is a separate `@watch` declaration. The complete workflow emerges from these independent transition handlers, with no central orchestration code. Add a new stage? Just add another `@watch`. Remove a stage? Delete the corresponding watch. The pipeline is self-documenting and maintainable.

## Summary

The `@watch` decorator runs functions when conditions transition from false to true:

- **Event-driven reactions** — Respond to specific state transitions, not every change
- **Boolean conditions** — Pass lambda functions that return true/false
- **Automatic dependency tracking** — Conditions re-evaluate when observables they read change
- **Multiple conditions** — Combine conditions with AND logic for complex requirements
- **One-time per transition** — Function runs once when condition becomes true, then waits for false→true again
- **Compose with computed observables** — Separate business logic (computed) from event handling (watch)

When `@reactive` gives you continuous synchronization, `@watch` gives you precise event detection. Together they cover the full spectrum of reactive behaviors—from "keep this in sync" to "do this when that happens."

With observables, stores, `@reactive`, and `@watch`, you have everything you need to build sophisticated reactive applications where state changes automatically propagate through your system, and important transitions trigger the right behaviors at the right times.
