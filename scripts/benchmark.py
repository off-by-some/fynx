#!/usr/bin/env python3
"""
FynX Performance Benchmarks

This script provides performance testing for the FynX reactive system.
It includes adaptive benchmarks that automatically scale workload until time limits are exceeded,
finding the exact performance boundaries of FynX's reactive observables.

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --quick           # Run only performance summary
    python benchmark.py --adaptive        # Run only adaptive scaling tests
    python benchmark.py --help            # Show help

Configuration:
    Adjust the constants at the top of the file to change benchmark parameters.
"""

import argparse
import gc
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, ".")

from fynx import Store, computed, observable

# Configuration constants - adjust these to change benchmark behavior
TIME_LIMIT_SECONDS = 1.0  # Maximum time allowed per operation
STARTING_N = 10  # Starting number of observables/operations
SCALE_FACTOR = 2.0  # How much to multiply N by each iteration


class AdaptiveBenchmark:
    """Adaptive benchmark that scales workload until time limit is exceeded."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def run_adaptive_benchmark(
        self, operation_func: Callable[[int], Tuple[float, float]], operation_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run an adaptive benchmark that scales N until time T is exceeded.

        Args:
            operation_func: Function that takes N and returns (setup_time, operation_time)
            operation_name: Descriptive name for the operation

        Returns:
            dict: Results including max_N, final_time, etc., or None if failed
        """
        n = STARTING_N
        iteration = 0

        if not self.quiet:
            print(f"\n=== {operation_name} ===")
            print(f"Starting with N={n}, Time limit: {TIME_LIMIT_SECONDS}s")

        # Progress tracking
        last_reported_n = 0
        significant_steps = []

        while True:
            setup_time, operation_time = operation_func(n)

            # Only print at significant milestones (powers of 2, or every 10000 operations)
            should_report = (
                n >= 1000 and (n % 10000 == 0 or n.bit_count() == 1)
            ) or iteration < 3
            if should_report and n != last_reported_n and not self.quiet:
                total_time = setup_time + operation_time
                print(
                    f"N={n:6d} | Total: {total_time:.4f}s | {n/(total_time):.0f} ops/sec"
                )
                last_reported_n = n
                significant_steps.append((n, total_time))

            # Check if we've exceeded the time limit
            if setup_time + operation_time > TIME_LIMIT_SECONDS:
                if not self.quiet:
                    print(f"Time limit exceeded at N={n}")
                # Go back to previous N that was within limits
                n = max(int(n / SCALE_FACTOR), STARTING_N)
                break

            # Scale up for next iteration
            new_n_float = n * SCALE_FACTOR
            # Check for overflow - if multiplication results in infinity
            if new_n_float == float("inf") or new_n_float > 2**63:
                if not self.quiet:
                    print(f"Reached maximum representable N: {n}")
                break
            new_n = int(new_n_float)
            n = new_n
            iteration += 1

        # Final benchmark with the maximum N that was within time limits
        if n > STARTING_N:
            setup_time, operation_time = operation_func(n)
            result = {
                "max_n": n,
                "setup_time": setup_time,
                "operation_time": operation_time,
                "total_time": setup_time + operation_time,
                "operations_per_second": n / (setup_time + operation_time),
            }
            if not self.quiet:
                print(
                    f"✓ Maximum sustainable N: {n} ({result['operations_per_second']:.0f} ops/sec)"
                )
            return result
        else:
            if not self.quiet:
                print("Could not find sustainable N above starting value")
            return None


class FynxBenchmarks:
    """Main benchmark suite for FynX performance testing."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.adaptive_benchmark = AdaptiveBenchmark(quiet=quiet)

    def run_all_benchmarks(self) -> None:
        """Run the complete benchmark suite."""
        if not self.quiet:
            print("🚀 FynX Performance Benchmark Suite")
            print("=" * 50)

        # Quick performance summary
        self.run_performance_summary()

        # Adaptive scaling benchmarks
        self.run_adaptive_benchmarks()

        if not self.quiet:
            print("\n🏁 Benchmark suite completed!")

    def run_performance_summary(self) -> None:
        """Run a quick summary of key performance metrics."""
        if not self.quiet:
            print("\n=== PERFORMANCE SUMMARY ===")

        scenarios = [
            (
                "100 Observables",
                lambda: self._time_operation(
                    lambda: [observable(i) for i in range(100)]
                ),
            ),
            (
                "100 Updates",
                lambda: self._time_operation(
                    lambda: [
                        obs.set(i * 2)
                        for i, obs in enumerate([observable(i) for i in range(100)])
                    ]
                ),
            ),
            (
                "Chain of 50",
                lambda: self._time_operation(lambda: self._create_chain(50)),
            ),
            (
                "Fan-out 50",
                lambda: self._time_operation(lambda: self._create_fan_out(50)),
            ),
        ]

        for name, scenario_func in scenarios:
            time_taken = scenario_func()
            if not self.quiet:
                print(f"{name}: {time_taken:.4f}s")

    def run_adaptive_benchmarks(self) -> None:
        """Run all adaptive scaling benchmarks."""
        if not self.quiet:
            print("\n=== ADAPTIVE SCALING BENCHMARKS ===")
            print(
                "These tests automatically scale workload N until time limit T is exceeded."
            )

        results = []

        # Observable creation scaling
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_n_observables, "Observable Creation"
        )
        if result:
            results.append(("Observable Creation", result))

        # Observable updates scaling
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._update_n_observables, "Observable Updates"
        )
        if result:
            results.append(("Observable Updates", result))

        # Computed chain scaling
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_chain_of_length, "Computed Chain"
        )
        if result:
            results.append(("Computed Chain", result))

        # Fan-out scaling
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_fan_out, "Fan-out Dependencies"
        )
        if result:
            results.append(("Fan-out Dependencies", result))

        # Print summary
        if results:
            if not self.quiet:
                print("\n=== BENCHMARK RESULTS SUMMARY ===")
            for name, result in results:
                print(f"{name:<25} {result['operations_per_second']:>8.0f} ops/sec")

    # Benchmark operation functions

    def _create_n_observables(self, n: int) -> Tuple[float, float]:
        """Create n observables and verify them."""
        start_time = time.time()
        observables = [observable(i) for i in range(n)]
        setup_time = time.time() - start_time

        # Verify they work
        operation_start = time.time()
        for i, obs in enumerate(observables):
            assert obs.value == i
        operation_time = time.time() - operation_start

        return setup_time, operation_time

    def _update_n_observables(self, n: int) -> Tuple[float, float]:
        """Update n pre-created observables."""
        # Pre-create observables for update testing
        if not hasattr(self, "_pre_created_obs"):
            self._pre_created_obs = [observable(0) for _ in range(10000)]

        setup_time = 0  # Already created

        operation_start = time.time()
        for i in range(min(n, len(self._pre_created_obs))):
            self._pre_created_obs[i].set(i * 2)
        operation_time = time.time() - operation_start

        # Verify
        for i in range(min(n, len(self._pre_created_obs))):
            assert self._pre_created_obs[i].value == i * 2

        return setup_time, operation_time

    def _create_chain_of_length(self, n: int) -> Tuple[float, float]:
        """Create a computed observable chain of length n."""
        start_time = time.time()
        base = observable(1)
        current = base

        for i in range(n):
            current = computed(lambda x, i=i: x + i, current)

        setup_time = time.time() - start_time

        # Test update propagation
        operation_start = time.time()
        base.set(2)
        operation_time = time.time() - operation_start

        # Verify: final value should be 2 + sum(range(n))
        expected = 2 + sum(range(n))
        assert current.value == expected

        return setup_time, operation_time

    def _create_fan_out(self, n: int) -> Tuple[float, float]:
        """Create n computed observables depending on a single base."""
        start_time = time.time()
        base = observable(42)
        dependents = []

        for i in range(n):
            dep = computed(lambda x, i=i: x + i, base)
            dependents.append(dep)

        setup_time = time.time() - start_time

        # Test update propagation to all dependents
        operation_start = time.time()
        base.set(100)
        operation_time = time.time() - operation_start

        # Verify all dependents updated
        for i, dep in enumerate(dependents):
            assert dep.value == 100 + i

        return setup_time, operation_time

    # Helper methods

    def _time_operation(self, operation: Callable) -> float:
        """Time a single operation."""
        start = time.time()
        operation()
        return time.time() - start

    def _create_chain(self, length: int) -> Any:
        """Create a chain of computed observables."""
        base = observable(1)
        current = base
        for i in range(length):
            current = computed(lambda x, i=i: x + i, current)
        return current

    def _create_fan_out_list(self, count: int) -> list:
        """Create fan-out dependencies."""
        base = observable(42)
        dependents = [computed(lambda x, i=i: x + i, base) for i in range(count)]
        return dependents


def print_config():
    """Print the current benchmark configuration."""
    print("FynX Benchmark Configuration:")
    print(f"  TIME_LIMIT_SECONDS: {TIME_LIMIT_SECONDS}")
    print(f"  STARTING_N: {STARTING_N}")
    print(f"  SCALE_FACTOR: {SCALE_FACTOR}")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="FynX Performance Benchmarks")
    parser.add_argument(
        "--quick", action="store_true", help="Run only the quick performance summary"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Run only the adaptive scaling benchmarks",
    )
    parser.add_argument(
        "--config", action="store_true", help="Show current benchmark configuration"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output, show only final results",
    )

    args = parser.parse_args()

    if args.config:
        print_config()
        return

    # Print configuration for all benchmark runs (unless quiet)
    if not args.quiet:
        print_config()
        print()

    benchmarks = FynxBenchmarks(quiet=args.quiet)

    if args.quick:
        benchmarks.run_performance_summary()
    elif args.adaptive:
        benchmarks.run_adaptive_benchmarks()
    else:
        benchmarks.run_all_benchmarks()


if __name__ == "__main__":
    main()
