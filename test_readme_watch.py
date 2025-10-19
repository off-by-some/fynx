"""
Test the watch example from README.md to ensure it works correctly.
"""

from fynx import observable, watch


def test_readme_watch_example():
    """Test the conditional @watch example from the README."""
    print("Testing README watch example...")

    # Setup conditions as shown in README
    condition1 = observable(True)
    condition2 = observable(False)

    triggered = []

    @watch(condition1 & condition2)
    def on_conditions_met():
        triggered.append("conditions_met")
        print("All conditions satisfied!")

    # Initially, condition2 is False, so watch shouldn't trigger
    assert len(triggered) == 0, "Watch should not trigger initially"
    print("✓ Watch correctly doesn't trigger initially")

    # Set condition2 to True - now both conditions are True
    condition2.set(True)
    assert len(triggered) == 1, "Watch should trigger when both conditions become True"
    print("✓ Watch triggers when both conditions become True")

    # Reset and test again
    triggered.clear()
    condition2.set(False)
    condition1.set(False)

    # Neither condition is True now
    assert len(triggered) == 0, "Watch should not trigger when conditions are reset"
    print("✓ Watch correctly doesn't trigger when conditions are False")

    # Set condition1 back to True - still shouldn't trigger since condition2 is False
    condition1.set(True)
    assert len(triggered) == 0, "Watch should not trigger when only condition1 is True"
    print("✓ Watch correctly doesn't trigger when only condition1 is True")

    # Finally set condition2 to True - should trigger
    condition2.set(True)
    assert (
        len(triggered) == 1
    ), "Watch should trigger when both conditions become True again"
    print("✓ Watch triggers when both conditions become True again")

    print("All tests passed! ✅")


if __name__ == "__main__":
    test_readme_watch_example()
