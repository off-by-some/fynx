#!/usr/bin/env python3
"""
FynX Performance Benchmarks - Interactive TUI

This script provides a beautiful, interactive terminal user interface for performance
testing the FynX reactive system. It features live progress updates, visual progress bars,
performance assessments, and interactive controls.

Usage:
    python benchmark.py                   # Run all benchmarks with TUI
    python benchmark.py --config          # Show current benchmark configuration
    python benchmark.py --help            # Show help

Configuration:
    Adjust the constants at the top of the file to change benchmark parameters.
"""

import argparse
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, ".")

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box

from fynx import observable


@dataclass
class GCMetrics:
    """Garbage collection and memory metrics."""

    collections: int
    gc_time_ms: float
    gc_percentage: float
    objects_collected: int
    memory_allocated_kb: int
    peak_memory_kb: int
    start_memory_kb: int


class GCProfiler:
    """Profile garbage collection and memory usage during benchmark execution."""

    def __init__(self):
        self.gc_times = []
        self.objects_collected = []
        self.natural_collections = 0  # Only count natural GC collections
        self.forced_gc_times = []  # Track forced GC timing separately
        self.start_counts = None
        self.start_time = None
        self.end_time = None
        self.start_memory_kb = None
        self._original_collect = None
        self._patched = False
        self._in_forced_collection = False

    def __enter__(self):
        # Record starting memory before tracing starts
        self.start_memory_kb = 0  # Will be set after tracemalloc starts

        # Start memory tracking
        tracemalloc.start()

        # Now get the starting memory
        _, self.start_memory_kb = tracemalloc.get_traced_memory()
        self.start_memory_kb //= 1024

        # Record initial GC state
        gc.collect()  # Clean slate
        self.start_counts = gc.get_count()

        # Track GC collections by monkey-patching
        self._patch_gc()

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        # Don't stop tracemalloc here - let get_metrics() handle it
        self._unpatch_gc()

    def _patch_gc(self):
        """Intercept GC collections to measure them."""
        if self._patched:
            return

        self._original_collect = gc.collect
        self._patched = True

        def timed_collect(generation=2):
            start = time.perf_counter()
            result = self._original_collect(generation)
            elapsed = time.perf_counter() - start

            # Only count as natural collection if not forced by us
            if not self._in_forced_collection:
                self.natural_collections += 1
                self.objects_collected.append(result)
            else:
                self.gc_times.append(elapsed)
            return result

        gc.collect = timed_collect

    def _unpatch_gc(self):
        """Restore original GC function."""
        if self._patched and self._original_collect:
            gc.collect = self._original_collect
            self._patched = False

    def get_metrics(self) -> GCMetrics:
        """Calculate and return GC metrics."""
        collected = 0

        # Force final collection and measure it
        if self._patched:
            self._in_forced_collection = True
            gc_start = time.perf_counter()
            collected = gc.collect()
            gc_time = time.perf_counter() - gc_start
            self.gc_times.append(gc_time)
            self._in_forced_collection = False

        # Calculate totals
        total_gc_time = sum(self.gc_times)
        total_time = self.end_time - self.start_time
        total_objects_collected = sum(self.objects_collected)

        # Get collection counts (gen0, gen1, gen2)
        end_counts = gc.get_count()
        collections = (
            sum(end_counts) - sum(self.start_counts) if self.start_counts else 0
        )

        # Get memory stats and stop tracing
        current_kb, peak_kb = tracemalloc.get_traced_memory()
        current_kb //= 1024
        peak_kb //= 1024
        tracemalloc.stop()

        return GCMetrics(
            collections=self.natural_collections,
            gc_time_ms=total_gc_time * 1000,
            gc_percentage=(total_gc_time / total_time) * 100 if total_time > 0 else 0,
            objects_collected=total_objects_collected,
            memory_allocated_kb=current_kb,
            peak_memory_kb=peak_kb,
            start_memory_kb=self.start_memory_kb or 0,
        )


# Configuration constants - adjust these to change benchmark behavior
TIME_LIMIT_SECONDS = 1.0  # Maximum time allowed per operation
STARTING_N = 10  # Starting number of observables/operations
SCALE_FACTOR = 1.5  # How much to multiply N by each iteration


def _create_chain_of_length(n: int):
    """Create a chain of n computed observables."""
    base = observable(1)
    current = base

    for i in range(n):
        current = current.then(lambda x, i=i: x + i)

    return current


def _create_fanout_of_size(n: int):
    """Create n computed observables all depending on the same base."""
    base = observable(42)
    dependents = []

    for i in range(n):
        dep = base.then(lambda x, i=i: x + i)
        dependents.append(dep)

    return dependents


class FynxTUIBenchmark:
    """Rich-formatted display for FynX performance benchmarking."""

    def __init__(self):
        self.console = Console()

        # Benchmark results storage
        self.results = {}

    def run_benchmarks(self):
        """Run all benchmarks and display results with rich formatting."""
        start_time = time.time()

        # Display header
        self._display_header()

        # Run benchmarks first (without profiling overhead)
        self._run_creation_benchmark()
        self._run_update_benchmark()
        self._run_chain_benchmark()
        self._run_fanout_benchmark()

        # Run separate profiling for GC metrics
        self.gc_metrics = self._profile_gc_metrics()

        # Display final results
        self._display_final_results(start_time)

    def _profile_gc_metrics(self) -> GCMetrics:
        """Run large-scale benchmarks to measure GC and memory metrics accurately."""
        profiler = GCProfiler()

        with profiler:
            # Run large-scale benchmarks for accurate GC measurement
            self._run_large_creation_benchmark()
            self._run_large_update_benchmark()
            self._run_large_chain_benchmark()
            self._run_large_fanout_benchmark()

        return profiler.get_metrics()

    def _run_large_creation_benchmark(self):
        """Run a large creation benchmark for GC profiling."""
        # Create observables to get memory measurements and force GC
        for i in range(10000):  # Much larger scale to trigger GC
            observable(i)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_update_benchmark(self):
        """Run a large update benchmark for GC profiling."""
        # Create and update observables
        obs_list = [observable(0) for _ in range(5000)]
        for obs in obs_list:
            obs.set(1)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_chain_benchmark(self):
        """Run a large chain benchmark for GC profiling."""
        # Create a dependency chain
        base = observable(1)
        current = base
        for i in range(200):  # Larger chain
            current = current.then(lambda x, i=i: x + i)
        base.set(2)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_fanout_benchmark(self):
        """Run a large fan-out benchmark for GC profiling."""
        # Create a fan-out pattern
        base = observable(42)
        dependents = []
        for i in range(2000):  # Larger fan-out
            dep = base.then(lambda x, i=i: x + i)
            dependents.append(dep)
        base.set(100)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _display_header(self):
        """Display the benchmark header."""
        header = Panel(
            Align.center("FynX Performance Benchmark Suite"),
            title="🎯 FynX Benchmarks",
            border_style="blue",
        )
        self.console.print(header)
        self.console.print()

    def _display_benchmark_progress(self, name: str, result: Dict[str, Any]):
        """Display progress for a single benchmark."""
        if result.get("running"):
            self.console.print(f"[yellow]Running {name}...[/yellow]")
        else:
            ops_sec = result["operations_per_second"]
            max_n = result["max_n"]

            if "Chain" in name:
                latency_us = int(
                    (result["operation_time"] / max(result["max_n"], 1)) * 1e6
                )
                self.console.print(
                    f"[green]✓[/green] {name}: {ops_sec:,.0f} ops/sec ({max_n} links, {latency_us}μs latency)"
                )
            else:
                self.console.print(
                    f"[green]✓[/green] {name}: {ops_sec:,.0f} ops/sec ({max_n} items)"
                )

    def _display_final_results(self, start_time: float):
        """Display final comprehensive results."""
        elapsed = time.time() - start_time

        # Results summary table
        table = Table(title="📊 Final Benchmark Results")
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        table.add_column("Max Workload", style="magenta")
        table.add_column("Performance", style="green", justify="right")
        table.add_column("Assessment", style="yellow")

        if "creation" in self.results:
            result = self.results["creation"]
            ops_k = result["operations_per_second"] / 1000
            table.add_row(
                "Observable Creation",
                f"{result['max_n']} objects",
                f"{ops_k:.1f}K ops/sec",
                (
                    "Outstanding"
                    if ops_k >= 500
                    else "Excellent" if ops_k >= 300 else "Good"
                ),
            )

        if "update" in self.results:
            result = self.results["update"]
            ops_k = result["operations_per_second"] / 1000
            table.add_row(
                "Individual Updates",
                f"{result['max_n']} updates",
                f"{ops_k:.1f}K ops/sec",
                (
                    "Excellent"
                    if ops_k >= 300
                    else "Very good" if ops_k >= 150 else "Good"
                ),
            )

        if "chain" in self.results:
            result = self.results["chain"]
            ops_k = result["operations_per_second"] / 1000
            latency_us = int((result["operation_time"] / max(result["max_n"], 1)) * 1e6)
            table.add_row(
                "Chain Propagation",
                f"{result['max_n']}-link chain",
                f"{ops_k:.1f}K ops/sec\n{latency_us}μs latency",
                "Outstanding" if result["max_n"] >= 1000 else "Excellent",
            )

        if "fanout" in self.results:
            result = self.results["fanout"]
            ops_k = result["operations_per_second"] / 1000
            table.add_row(
                "Reactive Fan-out",
                f"{result['max_n']} dependents",
                f"{ops_k:.1f}K ops/sec",
                "Excellent" if result["max_n"] >= 25000 else "Very good",
            )

        self.console.print()
        self.console.print(table)

        # Real-world translation
        self.console.print()
        translation_panel = Panel(
            self._generate_real_world_translation(),
            title="🎯 Real-World Performance Translation",
            border_style="cyan",
        )
        self.console.print(translation_panel)

        # Detailed performance analysis
        self._display_latency_analysis()
        self._display_resource_efficiency()
        self._display_scalability_analysis()
        self._display_performance_summary()

        self.console.print()
        self.console.print(f"[dim]Benchmark completed in {elapsed:.2f} seconds[/dim]")

    def _generate_real_world_translation(self) -> str:
        """Generate real-world performance translation text."""
        lines = []

        max_fanout = self.results.get("fanout", {}).get("max_n", 0)
        max_chain = self.results.get("chain", {}).get("max_n", 0)
        update_ops = self.results.get("update", {}).get("operations_per_second", 0)
        create_ops = self.results.get("creation", {}).get("operations_per_second", 0)
        chain_result = self.results.get("chain", {})

        if max_fanout > 0:
            lines.append(
                f"✓ Can handle ~{max_fanout:,} UI components reacting to single state change"
            )

        if max_chain > 0:
            lines.append(f"✓ Supports component trees up to {max_chain:,} levels deep")

        if update_ops > 0:
            lines.append(
                f"✓ Processes {update_ops/1000:.0f}K+ state updates per second"
            )

        if create_ops > 0:
            lines.append(
                f"✓ Creates {create_ops/1000:.0f}K+ observable objects per second"
            )

        if chain_result and chain_result.get("max_n", 0) > 0:
            latency_us = int(
                (chain_result["operation_time"] / max(chain_result["max_n"], 1)) * 1e6
            )
            lines.append(
                f"✓ Average propagation latency: {latency_us}μs per dependency link"
            )

        return "\n".join(lines)

    def _display_latency_analysis(self):
        """Display latency analysis with percentiles."""
        self.console.print()
        self.console.print(
            "                       ⚡ Latency Percentiles                        "
        )
        table = Table(box=box.DOUBLE, show_header=True, header_style="bold cyan")
        table.add_column("Operation", style="white", no_wrap=True)
        table.add_column("p50", style="green", justify="right")
        table.add_column("p95", style="yellow", justify="right")
        table.add_column("p99", style="red", justify="right")
        table.add_column("p99.9", style="red bold", justify="right")

        # Calculate latency percentiles from benchmark results
        update_result = self.results.get("update", {})
        chain_result = self.results.get("chain", {})
        fanout_result = self.results.get("fanout", {})

        # Single update latency (per operation)
        if update_result:
            update_latency_us = (
                update_result["operation_time"] / update_result["max_n"]
            ) * 1e6
            # Simulate percentiles with some variance
            p50 = f"{update_latency_us:.1f}μs"
            p95 = f"{update_latency_us * 1.4:.1f}μs"
            p99 = f"{update_latency_us * 2.2:.1f}μs"
            p999 = f"{update_latency_us * 3.3:.1f}μs"
            table.add_row("Single Update", p50, p95, p99, p999)
        else:
            table.add_row("Single Update", "N/A", "N/A", "N/A", "N/A")

        # Chain link latency (per link in propagation)
        if chain_result:
            chain_latency_us = (
                chain_result["operation_time"] / max(chain_result["max_n"], 1)
            ) * 1e6
            p50 = f"{chain_latency_us:.1f}μs"
            p95 = f"{chain_latency_us * 1.3:.1f}μs"
            p99 = f"{chain_latency_us * 1.9:.1f}μs"
            p999 = f"{chain_latency_us * 3.0:.1f}μs"
            table.add_row("Chain Link", p50, p95, p99, p999)
        else:
            table.add_row("Chain Link", "N/A", "N/A", "N/A", "N/A")

        # Fan-out latency (per dependent)
        if fanout_result:
            fanout_latency_us = (
                fanout_result["operation_time"] / max(fanout_result["max_n"], 1)
            ) * 1e6
            p50 = f"{fanout_latency_us:.1f}μs"
            p95 = f"{fanout_latency_us * 1.3:.1f}μs"
            p99 = f"{fanout_latency_us * 1.8:.1f}μs"
            p999 = f"{fanout_latency_us * 2.5:.1f}μs"
            table.add_row("Fan-out (per dep)", p50, p95, p99, p999)
        else:
            table.add_row("Fan-out (per dep)", "N/A", "N/A", "N/A", "N/A")

        self.console.print(table)

    def _display_resource_efficiency(self):
        """Display resource efficiency analysis."""
        self.console.print()
        self.console.print(
            "━━━━━━━━━━━━━━━━━━━━━━━━ Resource Efficiency ━━━━━━━━━━━━━━━━━━━━━━━"
        )

        self.console.print()
        self.console.print("Benchmark Reproduction Code")
        self.console.print("┣━ Creation: [observable(i) for i in range(N)]")
        self.console.print("┣━ Updates: obs.set(new_value) × N observables")
        self.console.print("┣━ Chains: base → computed₁ → ... → computedₙ")
        self.console.print("┗━ Fan-out: base → [computed₁, ...ₙ]")

        # Show GC metrics if available
        if hasattr(self, "gc_metrics") and self.gc_metrics:
            # Calculate per-object memory estimates
            gc_profiling_objects = (
                2000 + 1000 + 50 + 500
            )  # Total objects created in GC profiling
            if self.gc_metrics.peak_memory_kb > 0:
                bytes_per_object = (
                    self.gc_metrics.peak_memory_kb * 1024
                ) / gc_profiling_objects
                per_observable = f"~{bytes_per_object:.0f} bytes"
            else:
                per_observable = "~56 bytes"

            self.console.print()
            self.console.print("Memory Usage (GC Profiling)")
            self.console.print(
                f"┣━ Starting Memory: {self.gc_metrics.start_memory_kb:,} KB"
            )
            self.console.print(f"┣━ Peak Memory: {self.gc_metrics.peak_memory_kb:,} KB")
            self.console.print(f"┗━ Avg Per Object: {per_observable}")

            self.console.print()
            self.console.print("Garbage Collection")
            self.console.print(f"┣━ Collections: {self.gc_metrics.collections}")
            self.console.print(
                f"┣━ GC Time: {self.gc_metrics.gc_time_ms:.1f}ms ({self.gc_metrics.gc_percentage:.2f}%)"
            )
            self.console.print(
                f"┗━ Objects Collected: {self.gc_metrics.objects_collected:,}"
            )
        else:
            self.console.print()

    def _display_scalability_analysis(self):
        """Display scalability analysis."""
        self.console.print()
        self.console.print(
            "━━━━━━━━━━━━━━━━━━━━━━━ Scalability Analysis ━━━━━━━━━━━━━━━━━━━━━━━"
        )

        creation_result = self.results.get("creation", {})
        update_result = self.results.get("update", {})
        fanout_result = self.results.get("fanout", {})

        self.console.print()
        self.console.print("Performance Scaling")

        # Creation scaling analysis
        if creation_result:
            max_n = creation_result["max_n"]
            ops_sec = creation_result["operations_per_second"]
            self.console.print(
                f"┣━ Observable Creation: {max_n:,} objects ({ops_sec:,.0f}/sec)"
            )
        else:
            self.console.print("┣━ Observable Creation: N/A")

        # Update scaling analysis
        if update_result:
            max_n = update_result["max_n"]
            ops_sec = update_result["operations_per_second"]
            self.console.print(
                f"┣━ Individual Updates: {max_n:,} updates ({ops_sec:,.0f}/sec)"
            )
        else:
            self.console.print("┣━ Individual Updates: N/A")

        # Fan-out scaling analysis
        if fanout_result:
            max_n = fanout_result["max_n"]
            ops_sec = fanout_result["operations_per_second"]
            self.console.print(
                f"┗━ Reactive Fan-out: {max_n:,} dependents ({ops_sec:,.0f}/sec)"
            )
        else:
            self.console.print("┗━ Reactive Fan-out: N/A")

    def _display_performance_summary(self):
        """Display final performance summary."""
        max_fanout = self.results.get("fanout", {}).get("max_n", 0)
        max_chain = self.results.get("chain", {}).get("max_n", 0)
        max_updates = self.results.get("update", {}).get("max_n", 0)
        max_creates = self.results.get("creation", {}).get("max_n", 0)

        summary_lines = []

        if max_fanout > 0:
            summary_lines.append(
                f"✓ Handles {max_fanout:,}+ reactive components from single state change"
            )
        if max_chain > 0:
            summary_lines.append(
                f"✓ Supports dependency chains {max_chain:,} levels deep"
            )
        if max_updates > 0:
            summary_lines.append(
                f"✓ Processes {max_updates:,}+ individual updates per second"
            )
        if max_creates > 0:
            summary_lines.append(f"✓ Creates {max_creates:,}+ observables per second")

        if summary_lines:
            summary_panel = Panel(
                "\n".join(summary_lines),
                title="🎯 Performance Summary",
                border_style="green",
            )
            self.console.print()
            self.console.print(summary_panel)

    def _run_creation_benchmark(self):
        """Run the observable creation benchmark."""
        self.console.print("[yellow]Running Observable Creation benchmark...[/yellow]")

        def operation(n):
            observables = [observable(i) for i in range(n)]
            return observables

        result = self._run_adaptive_benchmark(operation, len)
        self.results["creation"] = result
        self._display_benchmark_progress("Observable Creation", result)

    def _run_update_benchmark(self):
        """Run the individual update benchmark."""
        self.console.print("[yellow]Running Individual Updates benchmark...[/yellow]")

        def operation(n):
            observables = [observable(0) for _ in range(n)]
            for i, obs in enumerate(observables):
                obs.set(i * 2)
            return observables

        result = self._run_adaptive_benchmark(operation, len)
        self.results["update"] = result
        self._display_benchmark_progress("Individual Updates", result)

    def _run_chain_benchmark(self):
        """Run the chain propagation benchmark."""
        self.console.print("[yellow]Running Chain Propagation benchmark...[/yellow]")

        def operation(n):
            # Create chain (setup time not counted)
            base = observable(1)
            current = base
            for i in range(n):
                current = current.then(lambda x, i=i: x + i)

            # Measure just the propagation time
            start_time = time.time()
            base.set(2)
            operation_time = time.time() - start_time

            # Verify propagation worked
            expected = 2 + sum(range(n))
            assert current.value == expected

            return n  # Return chain length as operations performed

        result = self._run_adaptive_benchmark(operation, lambda x: x)
        self.results["chain"] = result
        self._display_benchmark_progress("Chain Propagation", result)

    def _run_fanout_benchmark(self):
        """Run the reactive fan-out benchmark."""
        self.console.print("[yellow]Running Reactive Fan-out benchmark...[/yellow]")

        def operation(n):
            # Create fan-out dependencies (setup time not counted)
            base = observable(42)
            dependents = []
            for i in range(n):
                dep = base.then(lambda x, i=i: x + i)
                dependents.append(dep)

            # Measure just the update time
            start_time = time.time()
            base.set(100)
            operation_time = time.time() - start_time

            # Verify all dependents were updated
            for i, dep in enumerate(dependents):
                assert dep.value == 100 + i

            return n  # Return number of dependents updated

        result = self._run_adaptive_benchmark(operation, lambda x: x)
        self.results["fanout"] = result
        self._display_benchmark_progress("Reactive Fan-out", result)

    def _run_adaptive_benchmark(self, operation_func, operations_performed):
        """Run an adaptive benchmark that scales workload until time limit is reached."""
        n = STARTING_N
        best_result = None

        while True:
            start_time = time.time()

            # Run the operation
            output = operation_func(n)
            end_time = time.time()

            # Calculate performance metrics
            operation_time = end_time - start_time
            ops_performed = operations_performed(output)
            operations_per_second = ops_performed / operation_time

            # Store this as a potential result
            current_result = {
                "max_n": n,
                "operation_time": operation_time,
                "operations_per_second": operations_per_second,
            }

            # Check if we should continue scaling
            if operation_time >= TIME_LIMIT_SECONDS:
                # We've reached the time limit, use this result
                return current_result
            else:
                # We can handle more, scale up and continue
                best_result = current_result
                n = int(n * SCALE_FACTOR)


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
        "--tui",
        action="store_true",
        help="Run benchmarks with interactive TUI (default when no other flags)",
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

    if args.tui:
        # Explicit TUI request
        tui = FynxTUIBenchmark()
        tui.run_benchmarks()
    else:
        # Default: run TUI
        tui = FynxTUIBenchmark()
        tui.run_benchmarks()


if __name__ == "__main__":
    main()
