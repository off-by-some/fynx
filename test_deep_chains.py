#!/usr/bin/env python3
"""
Test script to verify that deep observable chains work without recursion issues.
"""

from fynx import Store, observable


def test_deep_chains():
    """Test that deep observable chains work without stack overflow."""
    print("=== Testing Deep Observable Chains ===")

    # Create a store
    store = Store()

    # Create a base observable
    base = store.observable("base", 1)

    # Create a very deep chain: base >> f1 >> f2 >> f3 >> f4 >> f5 >> f6 >> f7 >> f8 >> f9 >> f10
    def add_one(x):
        return x + 1

    def multiply_two(x):
        return x * 2

    def subtract_one(x):
        return x - 1

    # Create very deep chain (100 levels) - test the thread-local stack solution
    chain = base
    for i in range(100):
        if i % 3 == 0:
            chain = chain >> add_one
        elif i % 3 == 1:
            chain = chain >> multiply_two
        else:
            chain = chain >> subtract_one

    print(f"Deep chain created successfully!")
    print(f"Base value: {base.value}")
    print(f"Final chain value: {chain.value}")

    # Test change propagation
    print("\n--- Testing change propagation ---")
    changes = []

    def track_changes(value):
        changes.append(value)
        print(f"Change detected: {value}")

    chain.subscribe(track_changes)

    print(f"Before change: base={base.value}, chain={chain.value}")
    base.value = 5
    print(f"After change: base={base.value}, chain={chain.value}")
    print(f"Number of changes detected: {len(changes)}")

    print("\n=== Deep chain test completed successfully! ===")
    return True


if __name__ == "__main__":
    test_deep_chains()
