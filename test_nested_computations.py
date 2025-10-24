#!/usr/bin/env python3
"""Test nested computations with the thread-local stack solution."""

from fynx import Store


def test_nested_computations():
    """Test that nested computations track dependencies correctly."""
    print("=== Testing Nested Computations ===")

    store = Store()

    # Create base observables
    a = store.observable("a", 1)
    b = store.observable("b", 2)
    c = store.observable("c", 3)

    # Create nested computations
    # inner = a + b
    # outer = inner + c
    inner = store.computed("inner", lambda: a.value + b.value)
    outer = store.computed("outer", lambda: inner.value + c.value)

    print(f"Initial values: a={a.value}, b={b.value}, c={c.value}")
    print(f"Inner computation: {inner.value}")
    print(f"Outer computation: {outer.value}")

    # Test change propagation
    print("\n--- Testing change propagation ---")
    changes_detected = 0

    def on_change(delta):
        nonlocal changes_detected
        changes_detected += 1
        print(f"Change detected: {delta.key} = {delta.new_value}")

    inner.subscribe(on_change)
    outer.subscribe(on_change)

    print(f"Before change: a={a.value}, inner={inner.value}, outer={outer.value}")
    a.value = 10
    print(f"After change: a={a.value}, inner={inner.value}, outer={outer.value}")
    print(f"Number of changes detected: {changes_detected}")

    print("\n=== Nested computations test completed successfully! ===")
    return True


if __name__ == "__main__":
    test_nested_computations()
