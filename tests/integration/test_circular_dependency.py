"""
Test for circular dependency detection in complex reactive chains.
"""

import pytest

from fynx import Store, observable


class TestStore(Store):
    email = observable("test@example.com")
    age = observable(25)


def test_circular_dependency_with_conditional_chains():
    """Test that complex conditional observable chains don't create false circular dependencies."""

    # Use regular observables instead of Store
    email_obs = observable("test@example.com")
    age_obs = observable(25)

    # Create validation computed observables
    is_valid_email = email_obs >> (
        lambda email: "@" in email and "." in email.split("@")[1]
    )
    is_valid_age = age_obs >> (lambda age: 0 <= age <= 150)

    # Create combined boolean observable for profile validity
    profile_is_valid = (is_valid_email + is_valid_age) >> (
        lambda email_valid, age_valid: email_valid and age_valid
    )

    # Create a computed observable that depends on the boolean observable
    validation_status = profile_is_valid >> (
        lambda valid: "valid" if valid else "invalid"
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
    is_valid_email = TestStore.email >> (
        lambda email: "@" in email and "." in email.split("@")[1]
    )
    is_valid_age = TestStore.age >> (lambda age: 0 <= age <= 150)

    # Conditional chain
    profile_is_valid = (is_valid_email + is_valid_age) >> (
        lambda email_valid, age_valid: email_valid and age_valid
    )

    # Another computed that depends on the boolean
    can_access_feature = profile_is_valid >> (lambda valid: valid)

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
    computed_obs = obs_a >> (lambda x: (obs_a.set(x + 1), x)[1])

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
        next_obs = chain[-1] >> (lambda x, i=i: x + i + 1)
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
    sum_ab = (a + b) >> (lambda x, y: x + y)
    sum_bc = (b + c) >> (lambda x, y: x + y)
    sum_ac = (a + c) >> (lambda x, y: x + y)

    # Create higher-level computed that depend on the sums
    total1 = (sum_ab + sum_bc) >> (lambda x, y: x + y)
    total2 = (sum_bc + sum_ac) >> (lambda x, y: x + y)
    total3 = (sum_ac + sum_ab) >> (lambda x, y: x + y)

    # Create final aggregator
    grand_total = (total1 + total2 + total3) >> (lambda x, y, z: x + y + z)

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
        dep = base >> (lambda x, i=i: x + i)
        dependents.append(dep)

    # Change the base - this should trigger all dependents to update
    base.set(100)

    # Verify all dependents updated correctly
    for i, dep in enumerate(dependents):
        assert dep.value == 100 + i
