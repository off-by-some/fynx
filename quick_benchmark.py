#!/usr/bin/env python3
"""
Quick Performance Analysis for Arena-Based Reactive System
=========================================================

Let's identify the performance bottlenecks in the current implementation.
"""

import time

from prototype import computed, observable


def quick_creation_test():
    """Test observable creation speed."""
    print("Testing observable creation...")

    start = time.perf_counter()
    obs = observable(42)
    end = time.perf_counter()

    print(f"Single observable creation: {(end - start) * 1000:.3f} ms")
    return obs


def quick_access_test(obs):
    """Test value access speed."""
    print("Testing value access...")

    start = time.perf_counter()
    for _ in range(1000):
        _ = obs.value
    end = time.perf_counter()

    print(f"1000 accesses: {(end - start) * 1000:.3f} ms")
    print(f"Per access: {(end - start) * 1000:.3f} μs")


def quick_update_test(obs):
    """Test value update speed."""
    print("Testing value updates...")

    start = time.perf_counter()
    for i in range(1000):
        obs.set(i)
    end = time.perf_counter()

    print(f"1000 updates: {(end - start) * 1000:.3f} ms")
    print(f"Per update: {(end - start) * 1000:.3f} μs")


def quick_computed_test():
    """Test computed observable creation."""
    print("Testing computed observable creation...")

    x = observable(10)

    start = time.perf_counter()
    y = computed(lambda val: val * 2, x)
    end = time.perf_counter()

    print(f"Computed creation: {(end - start) * 1000:.3f} ms")

    # Test access
    start = time.perf_counter()
    for _ in range(1000):
        _ = y.value
    end = time.perf_counter()

    print(f"1000 computed accesses: {(end - start) * 1000:.3f} ms")
    print(f"Per computed access: {(end - start) * 1000:.3f} μs")


def analyze_bottlenecks():
    """Analyze where the performance bottlenecks are."""
    print("=" * 60)
    print("PERFORMANCE BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Test 1: Basic creation
    obs = quick_creation_test()
    print()

    # Test 2: Basic access
    quick_access_test(obs)
    print()

    # Test 3: Basic updates
    quick_update_test(obs)
    print()

    # Test 4: Computed observables
    quick_computed_test()
    print()

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_bottlenecks()
