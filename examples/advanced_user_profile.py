"""
Advanced UserProfile Example - Demonstrating Complex Reactive Systems

This example shows how FynX enables building sophisticated reactive applications
through store-level reactions, multiple subscription patterns, and composable
transformations. It models a user profile system with validation, notifications,
and state persistence.
"""

from datetime import datetime
from typing import Optional

from fynx import Store, observable, reactive, watch
from fynx.observable.computed import ComputedObservable


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
full_name = (UserProfile.first_name | UserProfile.last_name).then(
    lambda f, l: f"{f} {l}".strip()
)

# Age category based on age
age_category = UserProfile.age.then(
    lambda age: (
        "unknown"
        if age is None
        else ("minor" if age < 18 else "adult" if age < 65 else "senior")
    )
)

# Account status combining multiple factors
account_status = (
    UserProfile.is_active | UserProfile.is_verified | UserProfile.subscription_tier
).then(
    lambda active, verified, tier: (
        "premium_active"
        if active and verified and tier == "premium"
        else "active" if active else "inactive"
    )
)

# Profile completeness score (0-100)
profile_completeness = (
    UserProfile.first_name
    | UserProfile.last_name
    | UserProfile.email
    | UserProfile.phone
).then(lambda fn, ln, em, ph: sum([bool(fn), bool(ln), bool(em), bool(ph)]) / 4 * 100)

# Display name with fallback logic
display_name = (full_name | UserProfile.email).then(
    lambda name, email: (
        name if name.strip() else email.split("@")[0] if email else "Anonymous"
    )
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
is_valid_email: ComputedObservable[bool] = UserProfile.email.then(
    lambda email: "@" in email and "." in email.split("@")[1]
)

# Age validation (reasonable range)
is_valid_age: ComputedObservable[bool] = UserProfile.age.then(
    lambda age: 0 <= age <= 150
)

# Phone validation (basic format check)
is_valid_phone: ComputedObservable[bool] = UserProfile.phone.then(
    lambda phone: not phone
    or (
        len(phone) >= 10
        and phone.replace("-", "").replace("+", "").replace(" ", "").isdigit()
    )
)

# Overall profile validity - using conditional observables for complex logic
profile_is_valid = is_valid_email & is_valid_age & is_valid_phone  # type: ignore


@reactive(profile_is_valid)
def validate_profile(is_valid):
    """React to profile validation status changes."""
    status = "✅ Valid" if is_valid else "❌ Invalid"
    print(f"Profile validation: {status}")


# Premium feature access validation
can_access_premium = UserProfile.subscription_tier.then(
    lambda tier: tier in ["premium", "enterprise"]
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


# Pattern 3: Conditional notifications with @reactive and &
# Create computed observables for conditions
is_adult = UserProfile.age >> (lambda age: age is not None and age >= 18)
is_premium = UserProfile.subscription_tier >> (lambda tier: tier == "premium")


@reactive(is_adult & is_premium)
def on_eligible_user():
    print("🎯 User is now eligible for premium features!")


# More complex conditions
many_logins = UserProfile.login_count >> (lambda count: count is not None and count > 5)
notifications_disabled = UserProfile.notifications_enabled >> (
    lambda enabled: enabled is not None and not enabled
)


@reactive(many_logins & notifications_disabled)
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
first_login = UserProfile.login_count >> (lambda count: count == 1)


@reactive(first_login)
def welcome_new_user():
    print("🎊 Welcome! This is your first login!")


# Reward milestones
tenth_login = UserProfile.login_count >> (lambda count: count == 10)


@reactive(tenth_login)
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
engagement_factors = (
    profile_completeness
    | UserProfile.login_count
    | UserProfile.is_verified
    | age_category
).then(
    lambda completeness, logins, verified, age_cat: (
        (completeness / 100 * 0.4)  # 40% weight on profile completeness
        + (min(logins / 20, 1) * 0.3)  # 30% weight on activity (capped at 20 logins)
        + (0.2 if verified else 0)  # 20% bonus for verification
        + (0.1 if age_cat == "adult" else 0)  # 10% bonus for adult users
    )
)

# Auto-upgrade eligibility (complex business logic)
can_auto_upgrade = (
    engagement_factors | UserProfile.subscription_tier | UserProfile.is_active
).then(lambda engagement, tier, active: active and tier == "free" and engagement > 0.7)


@reactive(can_auto_upgrade)
def check_auto_upgrade(eligible):
    if eligible:
        print("🚀 User eligible for automatic premium upgrade!")
        print("   High engagement score detected")


# Dynamic feature access based on multiple conditions
age_eligible = UserProfile.age.then(lambda age: age >= 13)
advanced_features_access = (can_access_premium | profile_is_valid | age_eligible).then(
    lambda premium, valid, age_ok: premium and valid and age_ok
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
