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


def test_long_computed_chain_no_recursion():
    """Test a long chain of computed observables to ensure no recursion issues."""
    # Create a base observable
    base = observable(1)

    # Create a chain of computed observables
    chain = [base]
    for i in range(10000):  # Create 10000 levels deep to stress test
        next_obs = computed(lambda x, i=i: x + i + 1, chain[-1])
        chain.append(next_obs)

    # Setting the base should propagate through the entire chain
    base.set(2)

    # Verify the final value in the chain
    expected = 2  # base value
    for i in range(10000):
        expected += i + 1
    assert chain[-1].value == expected


def test_complex_computed_web():
    """Test a web of computed observables that might cause recursion issues."""
    # Create several base observables
    a = observable(1)
    b = observable(2)
    c = observable(3)

    # Create computed observables that depend on each other
    sum_ab = computed(lambda x, y: x + y, a | b)
    sum_bc = computed(lambda x, y: x + y, b | c)
    sum_ac = computed(lambda x, y: x + y, a | c)

    # Create higher-level computed that depend on the sums
    total1 = computed(lambda x, y: x + y, sum_ab | sum_bc)
    total2 = computed(lambda x, y: x + y, sum_bc | sum_ac)
    total3 = computed(lambda x, y: x + y, sum_ac | sum_ab)

    # Create final aggregator
    grand_total = computed(lambda x, y, z: x + y + z, total1 | total2 | total3)

    # Change one base observable and verify everything updates
    a.set(10)

    # Verify all values are correct
    assert sum_ab.value == 12  # 10 + 2
    assert sum_bc.value == 5  # 2 + 3
    assert sum_ac.value == 13  # 10 + 3
    assert total1.value == 17  # 12 + 5
    assert total2.value == 18  # 5 + 13
    assert total3.value == 25  # 13 + 12
    assert grand_total.value == 60  # 17 + 18 + 25


def test_fan_out_many_dependents():
    """Test fan-out pattern with many computed observables depending on one base."""
    base = observable(42)
    dependents = []

    # Create many computed observables that depend on the base
    for i in range(100):  # Create 100 dependents
        dep = computed(lambda x, i=i: x + i, base)
        dependents.append(dep)

    # Change the base - this should trigger all dependents to update
    base.set(100)

    # Verify all dependents updated correctly
    for i, dep in enumerate(dependents):
        assert dep.value == 100 + i
