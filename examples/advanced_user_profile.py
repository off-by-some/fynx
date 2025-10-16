"""
Advanced UserProfile Example - Demonstrating Complex Reactive Systems

This example shows how Fynx enables building sophisticated reactive applications
through store-level reactions, multiple subscription patterns, and composable
transformations. It models a user profile system with validation, notifications,
and state persistence.
"""

from datetime import datetime
from typing import Optional

from fynx import Store, computed, observable, reactive, watch


class UserProfile(Store):
    """A reactive user profile with validation and computed properties."""

    # Basic profile information
    first_name = observable("")
    last_name = observable("")
    email = observable("")
    age = observable(0)
    phone = observable("")

    # Account settings
    is_active = observable(True)
    is_verified = observable(False)
    subscription_tier = observable("free")  # free, premium, enterprise

    # Activity tracking
    last_login: Optional[datetime] = observable(None)
    login_count = observable(0)

    # Preferences
    theme = observable("light")  # light, dark
    notifications_enabled = observable(True)


# ------------------------------------------------------------------------------------------------
# Store-Level Reactions: React to any change in the entire store
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Store-Level Reactions: Reacting to Any Profile Change")
print("-" * 100)
print()


@reactive(UserProfile)
def on_profile_change(profile_snapshot):
    """Called whenever ANY observable in UserProfile changes."""
    print(
        f"🔄 Profile updated: {profile_snapshot.first_name} {profile_snapshot.last_name}"
    )
    print(
        f"   Status: {'✓ Verified' if profile_snapshot.is_verified else '⚠ Unverified'}"
    )
    print(f"   Tier: {profile_snapshot.subscription_tier}")
    print()


# ------------------------------------------------------------------------------------------------
# Computed Properties: Building transformations from simpler components
# ------------------------------------------------------------------------------------------------

print("=" * 100)
print("Computed Properties: Derived Values from Base Observables")
print("-" * 100)
print()

# Computed properties will be defined after initial data setup to avoid module-level issues

print("\nSetting initial profile data...")
UserProfile.first_name = "Alice"
UserProfile.last_name = "Johnson"
UserProfile.email = "alice.johnson@email.com"
UserProfile.age = 28
UserProfile.phone = "+1-555-0123"
UserProfile.is_verified = True
UserProfile.subscription_tier = "premium"

print("\nUpdating profile...")
UserProfile.age = 29
UserProfile.phone = ""  # This should decrease completeness

# Now define computed properties after the Store is initialized
print("\nCreating computed properties...")

# Build complex computed properties from simpler transformations
full_name = computed(
    lambda f, l: f"{f} {l}".strip(), (UserProfile.first_name | UserProfile.last_name)
)

# Age category based on age
age_category = computed(
    lambda age: (
        "unknown"
        if age is None
        else ("minor" if age < 18 else "adult" if age < 65 else "senior")
    ),
    UserProfile.age,
)

# Account status combining multiple factors
account_status = computed(
    lambda active, verified, tier: (
        "premium_active"
        if active and verified and tier == "premium"
        else "active" if active else "inactive"
    ),
    (UserProfile.is_active | UserProfile.is_verified | UserProfile.subscription_tier),
)

# Profile completeness score (0-100)
profile_completeness = computed(
    lambda fn, ln, em, ph: sum([bool(fn), bool(ln), bool(em), bool(ph)]) / 4 * 100,
    (
        UserProfile.first_name
        | UserProfile.last_name
        | UserProfile.email
        | UserProfile.phone
    ),
)

# Display name with fallback logic
display_name = computed(
    lambda name, email: (
        name if name.strip() else email.split("@")[0] if email else "Anonymous"
    ),
    (full_name | UserProfile.email),
)

# Subscribe to computed properties
full_name.subscribe(lambda name: print(f"👤 Full name: {name}"))
profile_completeness.subscribe(lambda pct: print(".0f"))

# ------------------------------------------------------------------------------------------------
# Validation System: Conditional reactions with complex logic
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Validation System: Conditional Reactions")
print("-" * 100)
print()

# Email validation
is_valid_email = computed(
    lambda email: "@" in email and "." in email.split("@")[1], UserProfile.email
)

# Age validation (reasonable range)
is_valid_age = computed(lambda age: 0 <= age <= 150, UserProfile.age)

# Phone validation (basic format check)
is_valid_phone = computed(
    lambda phone: not phone
    or (
        len(phone) >= 10
        and phone.replace("-", "").replace("+", "").replace(" ", "").isdigit()
    ),
    UserProfile.phone,
)

# Overall profile validity - using conditional observables for complex logic
profile_is_valid = is_valid_email & is_valid_age & is_valid_phone


@reactive(profile_is_valid)
def validate_profile(is_valid):
    """React to profile validation status changes."""
    status = "✅ Valid" if is_valid else "❌ Invalid"
    print(f"Profile validation: {status}")


# Premium feature access validation
can_access_premium = computed(
    lambda tier: tier in ["premium", "enterprise"], UserProfile.subscription_tier
)

premium_access_granted = (
    can_access_premium & UserProfile.is_verified & UserProfile.is_active
)


@reactive(premium_access_granted)
def check_premium_access(has_access):
    """React to premium access status."""
    if has_access:
        print("🎉 Premium features unlocked!")
    else:
        print("🔒 Premium features locked")


print("\nTesting validation...")
# Test validation by changing email
UserProfile.email = "invalid-email"  # Should trigger invalid state
UserProfile.email = "valid@email.com"  # Should trigger valid state

# ------------------------------------------------------------------------------------------------
# Notification System: Multiple subscription patterns
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Notification System: Multiple Subscription Patterns")
print("-" * 100)
print()


# Pattern 1: Individual observable subscription
def on_email_change(new_email):
    print(f"📧 Email changed to: {new_email}")


UserProfile.email.subscribe(on_email_change)


# Pattern 2: Combined observables subscription
def on_name_change(first, last):
    print(f"🏷️  Name updated: {first} {last}")


name_observables = UserProfile.first_name | UserProfile.last_name
name_observables.subscribe(on_name_change)


# Pattern 3: Conditional notifications with @watch
@watch(
    lambda: UserProfile.age.value is not None and UserProfile.age.value >= 18,
    lambda: UserProfile.subscription_tier.value == "premium",
)
def on_eligible_user():
    print("🎯 User is now eligible for premium features!")


@watch(
    lambda: UserProfile.login_count.value is not None
    and UserProfile.login_count.value > 5,
    lambda: UserProfile.notifications_enabled.value is not None
    and not UserProfile.notifications_enabled.value,
)
def on_suspicious_activity():
    print("🚨 Suspicious activity detected - many logins with notifications disabled")


# Pattern 4: Context manager for scoped reactions
def on_theme_change(theme):
    print(f"🎨 Theme changed to: {theme}")


print("\nTesting notifications...")
# Subscribe to theme changes
UserProfile.theme.subscribe(on_theme_change)

UserProfile.theme = "dark"
UserProfile.theme = "light"

# Unsubscribe from theme changes
UserProfile.theme.unsubscribe(on_theme_change)
UserProfile.theme = "auto"  # This won't trigger on_theme_change

# ------------------------------------------------------------------------------------------------
# Activity Tracking: Building complex behavior from simple components
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Activity Tracking: Complex Behavior from Simple Components")
print("-" * 100)
print()


# Simulate login activity
def simulate_login():
    """Simulate a user login event."""
    UserProfile.login_count = UserProfile.login_count.value + 1
    UserProfile.last_login = "2024-01-15 10:30:00"
    print(f"🔐 User logged in (total: {UserProfile.login_count.value})")


# Welcome message for new users
@watch(lambda: UserProfile.login_count.value == 1)
def welcome_new_user():
    print("🎊 Welcome! This is your first login!")


# Reward milestones
@watch(lambda: UserProfile.login_count.value == 10)
def reward_milestone():
    print("🏆 Milestone reached: 10 logins! Here's a virtual badge!")


# Activity-based recommendations
login_streak = UserProfile.login_count >> (
    lambda count: "active" if count > 3 else "casual"
)


@reactive(login_streak)
def suggest_features(streak):
    if streak == "active":
        print("💡 Recommendation: Try our advanced analytics features!")
    else:
        print("💡 Recommendation: Complete your profile for better experience!")


print("\nSimulating user activity...")
simulate_login()  # First login - should trigger welcome
simulate_login()  # Second login
simulate_login()  # Third login
simulate_login()  # Fourth login - should trigger active streak recommendation

# ------------------------------------------------------------------------------------------------
# State Persistence: Serialization and restoration
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("State Persistence: Save and Restore Profile State")
print("-" * 100)
print()

# Save current state
print("💾 Saving profile state...")
saved_state = UserProfile.to_dict()
print("Saved state keys:", list(saved_state.keys()))

# Modify profile
print("\n🔄 Modifying profile before restoration...")
UserProfile.first_name = "Modified"
UserProfile.subscription_tier = "enterprise"
UserProfile.login_count = 50

# Restore from saved state
print("\n📂 Restoring profile state...")
UserProfile.load_state(saved_state)
print("Profile restored to previous state")

# ------------------------------------------------------------------------------------------------
# Complex Reactive Chains: Building sophisticated systems
# ------------------------------------------------------------------------------------------------

print()
print("=" * 100)
print("Complex Reactive Chains: Sophisticated System Behavior")
print("-" * 100)
print()

# User engagement score based on multiple factors
engagement_factors = computed(
    lambda completeness, logins, verified, age_cat: (
        completeness / 100 * 0.4
    )  # 40% weight on profile completeness
    + (min(logins / 20, 1) * 0.3)  # 30% weight on activity (capped at 20 logins)
    + (0.2 if verified else 0)  # 20% bonus for verification
    + (0.1 if age_cat == "adult" else 0),  # 10% bonus for adult users
    (
        profile_completeness
        | UserProfile.login_count
        | UserProfile.is_verified
        | age_category
    ),
)

# Auto-upgrade eligibility (complex business logic)
can_auto_upgrade = computed(
    lambda engagement, tier, active: active and tier == "free" and engagement > 0.7,
    (engagement_factors | UserProfile.subscription_tier | UserProfile.is_active),
)


@reactive(can_auto_upgrade)
def check_auto_upgrade(eligible):
    if eligible:
        print("🚀 User eligible for automatic premium upgrade!")
        print("   High engagement score detected")


# Dynamic feature access based on multiple conditions
age_eligible = computed(lambda age: age >= 13, UserProfile.age)
advanced_features_access = computed(
    lambda premium, valid, age_ok: premium and valid and age_ok,
    (can_access_premium | profile_is_valid | age_eligible),
)


@reactive(advanced_features_access)
def manage_feature_access(has_access):
    if has_access:
        print("🔓 Advanced features enabled")
    else:
        print("🔒 Advanced features disabled")


print("\nTesting complex reactive chains...")
print(f"Engagement score: {engagement_factors.value:.2f}")
print(f"Auto-upgrade eligible: {can_auto_upgrade.value}")
print(f"Advanced features access: {advanced_features_access.value}")

print("\n" + "=" * 100)
print("🎉 Advanced UserProfile Example Complete!")
print("=" * 100)
print("\nThis example demonstrated:")
print("• Store-level reactions (@reactive(UserProfile))")
print("• Multiple subscription patterns (individual, combined, conditional)")
print("• Building transformations from simpler components (computed properties)")
print("• Complex reactive systems (validation, notifications, state persistence)")
print("• Composable operators (>>, |, &) for sophisticated logic")
