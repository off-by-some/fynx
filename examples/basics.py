from fynx import Store, observable, reactive

# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Defining an observable")
print("-" * 100)
print()

# You can freely define observables as you like, and they will be reactive.
current_age = observable(30)
current_name = observable("Alice")

log_on_change = lambda name: print(f"Name changed to: {name}")

# You can call functions that run when the observable changes.
current_name.subscribe(log_on_change)
current_name.set("Smith")  # This will trigger the reactive context


current_name.unsubscribe(log_on_change)
current_name.set("Bob")  # This will not trigger the reactive context

# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Combining observables")
print("-" * 100)
print()


# Subscribing to a merged observable will be passed the values of the individual observables
# as arguments to the function.
def log_name_and_age_change(name, age):
    print(f"Name: {name}, Age: {age}")


# Define a merged observable with the | operator.
current_name_and_age = current_name | current_age

# Subscribe to the merged observable.
current_name_and_age.subscribe(log_name_and_age_change)

# Set the values of one each of the observables.
current_name.set("Charlie")
current_age.set(31)

# Unsubscribe from the merged observable.
current_name_and_age.unsubscribe(log_name_and_age_change)

# This should NOT trigger the reactive context anymore
current_age.set(32)
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Defining the store")
print("-" * 100)
print()


# This is how you create a store.
# Store is a base class that allows you to define observable attributes.
# and includes a few convenience methods such as subscribing to changes and unsubscribing from them.
class ExampleStore(Store):
    height_cm = observable(160.0)
    name = observable("Alice")
    age = observable(30)


# You can subscribe to changes in the store.
def on_store_snapshot_change(store):
    print(f"Store changed, current snapshot: {store}")


ExampleStore.subscribe(on_store_snapshot_change)

# You can change the store and the function will be called.
# This will cause 3 print statements.
ExampleStore.height_cm = 170.0
ExampleStore.name = "Bob"
ExampleStore.age = 31

# You can unsubscribe from changes in the store.
ExampleStore.unsubscribe(on_store_snapshot_change)

# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Using @reactive(ExampleStore) decorator")
print("-" * 100)
print()


# @reactive(store): subscribes to ALL items. Passes a snapshot of the store as param.
# Functionally equivalent to ExampleStore.subscribe(on_store_change)
@reactive(ExampleStore)
def on_store_change(store):
    print(
        f"Store changed - Height: {store.height_cm}, Name: {store.name}, Age: {store.age}"
    )


# Changing the store will trigger the on_store_change function
ExampleStore.height_cm = 170.2

# Unsubscribe the function
ExampleStore.unsubscribe(on_store_change)


# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Using @reactive(*observables) decorator")
print("-" * 100)
print()


# @reactive(single): subscribes to single observable (or more) passes value as param
@reactive(ExampleStore.age, ExampleStore.name)
def on_age_change(age, name):
    print(f"Name: {name}, Age: {age}")


ExampleStore.age = 333

# Test unsubscribing from individual observables
ExampleStore.unsubscribe(on_age_change)

# This should not trigger on_age_change anymore
ExampleStore.name = "Barbara"


# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Using with (ExampleStore.name | ExampleStore.age) as react")
print("-" * 100)
print()


def on_name_age_change(name, age):
    print(f"Name: {name}, Age: {age}")


# Subscribe to multiple observables at once
with ExampleStore.name | ExampleStore.age as react:
    react(on_name_age_change)

ExampleStore.name = "Bob"


# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Conditional reactions with @reactive and & operator")
print("-" * 100)
print()

# You can create conditional reactions that only trigger when specific conditions are met
user_status = observable("offline")
message_count = observable(0)

# Create computed observables for the conditions
is_online = user_status >> (lambda s: s == "online")
has_messages = message_count >> (lambda c: c is not None and c > 0)


@reactive(is_online & has_messages)
def notify_user():
    print(f"📬 Notifying user: {message_count.value} new messages while online!")


# Show that it doesn't trigger initially
user_status.set("online")  # User comes online but no messages
message_count.set(3)  # Messages arrive - should trigger notification

# Change conditions
user_status.set("away")  # User goes away
message_count.set(5)  # More messages, but user is away - no notification

user_status.set("online")  # User comes back online - should trigger again

print("\nConditional reactions only run when ALL conditions are met!")


# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Mathematical API: Functorial computed observables")
print("-" * 100)
print()


# Step 1: Create reactive values (they update automatically)
height = observable(170.0)  # Height in centimeters
weight = observable(70.0)  # Weight in kilograms

# Step 2: Combine related values into a single reactive unit
# This creates a reactive pair: (height, weight)
bmi_data = height | weight


# Step 3: Transform the combined data using the >> operator
def calculate_bmi(height, weight):
    """Calculate BMI from height (cm) and weight (kg)."""
    return weight / (height / 100) ** 2


def calculate_bmi_category(bmi):
    """Categorize BMI value into health categories."""
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


# Each >> creates a new computed observable that updates when inputs change
bmi = bmi_data >> calculate_bmi  # Calculate BMI from height & weight
bmi_category = bmi >> calculate_bmi_category  # Categorize BMI value

# Alternative syntax (same result):
# bmi = bmi_data.then(calculate_bmi)
# bmi_category = bmi.then(calculate_bmi_category)

# Show initial calculated values first
print("Initial BMI calculation:")
bmi_val = bmi.value
if bmi_val is not None:
    print(f"  BMI: {round(bmi_val, 1)}, Category: {bmi_category.value}")
else:
    print(f"  BMI: N/A, Category: {bmi_category.value}")

# Now subscribe to see live updates
print("\nSubscribing to live updates...")
bmi.subscribe(
    lambda val: print(
        f"  → BMI changed to: {round(val, 1) if val is not None else 'N/A'}"
    )
)
bmi_category.subscribe(lambda cat: print(f"  → Category changed to: {cat}"))

print("\nChanging weight to 80kg...")
weight.set(80.0)

print("\nChanging height to 175cm...")
height.set(175.0)

print("\nChanging weight to 65kg...")
weight.set(65.0)
