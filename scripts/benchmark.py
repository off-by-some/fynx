#!/usr/bin/env python3
"""
FynX Performance Benchmarks

This script provides performance testing for the FynX reactive system.
It includes adaptive benchmarks that automatically increase workload size until time limits are reached,
finding the exact performance boundaries of FynX's reactive data handling.

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
        self,
        operation_func: Callable[[int], Tuple[float, float]],
        operation_name: str,
        operations_performed: Optional[Callable[[int], int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run an adaptive benchmark that scales N until time T is exceeded.

        Args:
            operation_func: Function that takes N and returns (setup_time, operation_time)
            operation_name: Descriptive name for the operation
            operations_performed: Function that takes N and returns actual operations performed

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
                actual_ops = operations_performed(n) if operations_performed else n
                # For benchmarks that perform few operations, show operations/second based on operation time
                if actual_ops <= n / 1000:  # If performing << N operations
                    perf_metric = f"{actual_ops/(operation_time):.1f} operations/second (reactive)"
                else:
                    perf_metric = f"{actual_ops/(total_time):.0f} operations/second"
                print(
                    f"Workload size: {n:6d} | Time: {total_time:.4f}s | {perf_metric}"
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
            actual_ops = operations_performed(n) if operations_performed else n
            # For reactive benchmarks (few operations), use operation_time for ops/sec
            if actual_ops <= n / 1000:
                ops_per_sec = actual_ops / operation_time
            else:
                ops_per_sec = actual_ops / (setup_time + operation_time)

            result = {
                "max_n": n,
                "setup_time": setup_time,
                "operation_time": operation_time,
                "total_time": setup_time + operation_time,
                "operations_per_second": ops_per_sec,
            }
            if not self.quiet:
                print(
                    f"Maximum sustainable workload: {n} ({result['operations_per_second']:.0f} operations/second)"
                )
            return result
        else:
            if not self.quiet:
                print("Could not find sustainable N above starting value")
            return None


class FynxBenchmarks:
    """Main benchmark suite for FynX performance testing."""

    # Docstrings containing code patterns with markdown formatting
    CREATE_PATTERN_DOCSTRING = """
```python
observables = [observable(i) for i in range(N)]
```
Creates N independent reactive data items
"""

    UPDATE_PATTERN_DOCSTRING = """
```python
obs.set(new_value)  # for each of N observables
```
Updates N separate reactive data items
"""

    CHAIN_PATTERN_DOCSTRING = """
```python
base = observable(1)
current = base
for i in range(N):
    current = computed(lambda x: x + i, current)
```
Creates a chain of N computed values, each depending on the previous
"""

    FANOUT_PATTERN_DOCSTRING = """
```python
base = observable(42)
dependents = [computed(lambda x: x + i, base) for i in range(N)]
```
Creates N computed values all depending on one base observable
"""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.adaptive_benchmark = AdaptiveBenchmark(quiet=quiet)

    def run_all_benchmarks(self) -> None:
        """Run the complete benchmark suite."""
        if not self.quiet:
            print("FynX Performance Benchmark Suite")
            print("=" * 50)

        # Quick performance summary
        self.run_performance_summary()

        # Adaptive scaling benchmarks
        self.run_adaptive_benchmarks()

        if not self.quiet:
            print("\nBenchmark suite completed successfully!")

    def run_performance_summary(self) -> None:
        """Run a quick summary of key performance metrics."""
        if not self.quiet:
            print("\n=== PERFORMANCE SUMMARY ===")
            print("Testing basic FynX operations with fixed workload sizes:")
            print("- Create 100 data items")
            print("- Update 100 data items")
            print("- Process 50-step dependency chain")
            print("- Update 50 dependent data items")
            print()

        scenarios = [
            (
                "Create 100 data items",
                lambda: self._time_operation(
                    lambda: [observable(i) for i in range(100)]
                ),
            ),
            (
                "Update 100 data items",
                lambda: self._time_operation(
                    lambda: [
                        obs.set(i * 2)
                        for i, obs in enumerate([observable(i) for i in range(100)])
                    ]
                ),
            ),
            (
                "Process 50-step dependency chain",
                lambda: self._time_operation(lambda: self._create_chain(50)),
            ),
            (
                "Update 50 dependent data items",
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
                "These tests automatically increase the workload until the time limit is reached."
            )
            print("The workload size (N) represents different things in each test:")
            print(
                "- Creation/Update tests: N = number of data items to create or update"
            )
            print("- Chain test: N = length of the dependency chain to process")
            print("- Fan-out test: N = number of dependent items to update")
            print()

        results = []

        # Observable creation scaling
        if not self.quiet:
            print("Testing pattern:")
            print(self.CREATE_PATTERN_DOCSTRING.strip())
            print()
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_n_observables,
            "Create & Verify N Data Items",
            operations_performed=lambda n: n,
        )
        if result:
            results.append(("Create & Verify Data Items", result))

        # Observable updates scaling
        if not self.quiet:
            print("Testing pattern:")
            print(self.UPDATE_PATTERN_DOCSTRING.strip())
            print()
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._update_n_observables,
            "Update N Individual Data Items",
            operations_performed=lambda n: n,
        )
        if result:
            results.append(("Update Individual Data Items", result))

        # Computed chain scaling
        if not self.quiet:
            print("Testing pattern:")
            print(self.CHAIN_PATTERN_DOCSTRING.strip())
            print()
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_chain_of_length,
            "Process N-Step Dependency Chain",
            operations_performed=lambda n: 1,
        )
        if result:
            results.append(("Process Dependency Chain", result))

        # Fan-out scaling
        if not self.quiet:
            print("Testing pattern:")
            print(self.FANOUT_PATTERN_DOCSTRING.strip())
            print()
        result = self.adaptive_benchmark.run_adaptive_benchmark(
            self._create_fan_out,
            "Update N Dependent Data Items",
            operations_performed=lambda n: 1,
        )
        if result:
            results.append(("Update Dependent Data Items", result))

        # Print summary
        if results:
            if not self.quiet:
                print("\n=== BENCHMARK RESULTS SUMMARY ===")
                print("Maximum sustainable workload size with operations per second:")
                print(
                    "- Creation/Update tests: operations completed per second (higher numbers = faster)"
                )
                print("- Reactive tests: data updates processed per second")
                print()
            for name, result in results:
                print(
                    f"{name:<30} {result['operations_per_second']:>8.0f} operations/second"
                )

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
        """Update n observables and measure update performance."""
        # Create n observables for this test
        start_time = time.time()
        observables = [observable(0) for _ in range(n)]
        setup_time = time.time() - start_time

        # Perform n updates (one per observable)
        operation_start = time.time()
        for i, obs in enumerate(observables):
            obs.set(i * 2)
        operation_time = time.time() - operation_start

        # Verify all updates worked
        for i, obs in enumerate(observables):
            assert obs.value == i * 2

        return setup_time, operation_time

    def _create_chain_of_length(self, n: int) -> Tuple[float, float]:
        """Test propagation speed through a chain of n computed observables."""
        # Create chain of n computed observables
        start_time = time.time()
        base = observable(1)
        current = base

        for i in range(n):
            current = computed(lambda x, i=i: x + i, current)

        setup_time = time.time() - start_time

        # Measure propagation time (one update through the entire chain)
        operation_start = time.time()
        base.set(2)
        operation_time = time.time() - operation_start

        # Verify propagation worked: final value should be 2 + sum(range(n))
        expected = 2 + sum(range(n))
        assert current.value == expected

        return setup_time, operation_time

    def _create_fan_out(self, n: int) -> Tuple[float, float]:
        """Test update speed for n computed observables depending on one base."""
        # Create n computed observables all depending on the same base
        start_time = time.time()
        base = observable(42)
        dependents = []

        for i in range(n):
            dep = computed(lambda x, i=i: x + i, base)
            dependents.append(dep)

        setup_time = time.time() - start_time

        # Measure update time (one base change updates all n dependents)
        operation_start = time.time()
        base.set(100)
        operation_time = time.time() - operation_start

        # Verify all n dependents were updated correctly
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
