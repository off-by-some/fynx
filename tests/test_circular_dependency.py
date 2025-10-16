"""
Test for circular dependency detection in complex reactive chains.
"""

import pytest

from fynx import Store, computed, observable


class TestStore(Store):
    email = observable("test@example.com")
    age = observable(25)


def test_circular_dependency_with_conditional_chains():
    """Test that complex conditional observable chains don't create false circular dependencies."""

    # Use regular observables instead of Store
    email_obs = observable("test@example.com")
    age_obs = observable(25)

    # Create validation computed observables
    is_valid_email = computed(
        lambda email: "@" in email and "." in email.split("@")[1], email_obs
    )
    is_valid_age = computed(lambda age: 0 <= age <= 150, age_obs)

    # Create conditional observable chain
    profile_is_valid = is_valid_email & is_valid_age

    # Create a computed observable that depends on the conditional observable
    validation_status = computed(
        lambda valid: "valid" if valid else "invalid", profile_is_valid
    )

    # This should not cause a circular dependency
    print("Setting email to invalid-email...")
    email_obs.set("invalid-email")  # Should trigger invalid state
    print("Done setting to invalid-email")

    # The conditional observable should update
    assert profile_is_valid.value == False
    assert validation_status.value == "invalid"

    # This should also not cause a circular dependency
    print("Setting email to valid@example.com...")
    email_obs.set("valid@example.com")  # Should trigger valid state
    print("Done setting to valid@example.com")

    # The conditional observable should update
    assert profile_is_valid.value == True
    assert validation_status.value == "valid"


def test_circular_dependency_with_complex_chains():
    """Test more complex chains that were causing circular dependencies in the example."""

    # Simulate the pattern from advanced_user_profile.py
    is_valid_email = computed(
        lambda email: "@" in email and "." in email.split("@")[1], TestStore.email
    )
    is_valid_age = computed(lambda age: 0 <= age <= 150, TestStore.age)

    # Conditional chain
    profile_is_valid = is_valid_email & is_valid_age

    # Another computed that depends on the conditional
    can_access_feature = computed(lambda valid: valid, profile_is_valid)

    # This change should trigger updates through the chain without circular dependency
    TestStore.email = "new@example.com"

    assert profile_is_valid.value == True
    assert can_access_feature.value == True

    # Another change
    TestStore.email = "invalid"

    assert profile_is_valid.value == False
    assert can_access_feature.value == False


def test_circular_dependency_should_be_detected():
    """Test that actual circular dependencies are still detected."""

    # Create a true circular dependency
    obs_a = observable(1)

    # Create computed that modifies its source
    computed_obs = computed(lambda x: (obs_a.set(x + 1), x)[1], obs_a)

    # The circular dependency is detected when the computed is triggered
    with pytest.raises(RuntimeError, match="Circular dependency detected"):
        obs_a.set(5)  # This should trigger the circular dependency error
