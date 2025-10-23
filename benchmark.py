"""
FynX Optimized Tape System Benchmarks
"""

import argparse
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, List, Protocol, TypeVar, runtime_checkable

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prototype import Obs, OptimizedTapeGraph, observable

# Type variable for generic observables
T = TypeVar("T")

# =============================================================================
# Benchmark Configuration and Metrics
# =============================================================================

# Configuration
TIME_LIMIT_SECONDS = 1.0
STARTING_N = 10
SCALE_FACTOR = 1.5
NUM_ITERATIONS = 3  # Run multiple times to average out GC variance


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


@dataclass
class BenchmarkMetrics:
    """Complete metrics from a benchmark run."""

    operation: str
    max_n: int
    operation_time: float
    operations_per_second: float

    # Memory metrics
    memory_start_kb: int
    memory_end_kb: int
    memory_peak_kb: int
    memory_allocated_kb: int

    # GC metrics
    gc_count_gen0: int
    gc_count_gen1: int
    gc_count_gen2: int
    gc_total_collections: int

    # Object tracking
    objects_before: int
    objects_after: int
    objects_delta: int


class BenchmarkProfiler:
    """Profile performance, memory, and GC during benchmark execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.gc_stats_before = None
        self.gc_stats_after = None
        self.memory_start = None
        self.memory_end = None
        self.memory_peak = None
        self.objects_before = None
        self.objects_after = None

    def __enter__(self):
        # Collect garbage before starting to get clean baseline
        gc.collect()
        gc.collect()
        gc.collect()

        # Start memory tracking
        tracemalloc.start()
        self.memory_start, _ = tracemalloc.get_traced_memory()

        # Record GC stats and object count
        self.gc_stats_before = gc.get_count()
        self.objects_before = len(gc.get_objects())

        # Start timing
        self.start_time = time.perf_counter()

        return self

    def __exit__(self, *args):
        # Stop timing
        self.end_time = time.perf_counter()

        # Record GC stats after
        self.gc_stats_after = gc.get_count()
        self.objects_after = len(gc.get_objects())

        # Get memory stats
        self.memory_end, self.memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time

    def get_metrics(
        self, operation: str, n: int, operations_performed: int
    ) -> BenchmarkMetrics:
        """Calculate and return all metrics."""
        elapsed = self.get_elapsed_time()
        ops_per_sec = operations_performed / elapsed if elapsed > 0 else 0

        # Calculate GC collections that occurred during benchmark
        gc_gen0 = max(0, self.gc_stats_after[0] - self.gc_stats_before[0])
        gc_gen1 = max(0, self.gc_stats_after[1] - self.gc_stats_before[1])
        gc_gen2 = max(0, self.gc_stats_after[2] - self.gc_stats_before[2])

        return BenchmarkMetrics(
            operation=operation,
            max_n=n,
            operation_time=elapsed,
            operations_per_second=ops_per_sec,
            memory_start_kb=self.memory_start // 1024,
            memory_end_kb=self.memory_end // 1024,
            memory_peak_kb=self.memory_peak // 1024,
            memory_allocated_kb=(self.memory_end - self.memory_start) // 1024,
            gc_count_gen0=gc_gen0,
            gc_count_gen1=gc_gen1,
            gc_count_gen2=gc_gen2,
            gc_total_collections=gc_gen0 + gc_gen1 + gc_gen2,
            objects_before=self.objects_before,
            objects_after=self.objects_after,
            objects_delta=self.objects_after - self.objects_before,
        )


def run_adaptive_benchmark(
    operation: str,
    operation_func: Callable[[int], T],
    operations_counter: Callable[[T], int],
    time_limit: float = TIME_LIMIT_SECONDS,
    starting_n: int = STARTING_N,
    scale_factor: float = SCALE_FACTOR,
) -> BenchmarkMetrics:
    """
    Run adaptive benchmark that scales workload to target time with GC profiling.
    """
    n = starting_n

    # First, find the optimal workload size
    while True:
        start = time.perf_counter()
        result = operation_func(n)
        elapsed = time.perf_counter() - start

        if elapsed >= time_limit:
            break

        n = int(n * scale_factor)
        if n > 10_000_000:  # Safety limit
            break

    # Now run the benchmark at optimal size with full profiling
    # Run multiple iterations and average
    all_metrics = []

    for iteration in range(NUM_ITERATIONS):
        with BenchmarkProfiler() as profiler:
            result = operation_func(n)
            ops_performed = operations_counter(result)

        metrics = profiler.get_metrics(operation, n, ops_performed)
        all_metrics.append(metrics)

    # Return averaged metrics
    return average_metrics(all_metrics)


def average_metrics(metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Average multiple benchmark runs."""
    if not metrics_list:
        raise ValueError("No metrics to average")

    first = metrics_list[0]

    return BenchmarkMetrics(
        operation=first.operation,
        max_n=first.max_n,
        operation_time=sum(m.operation_time for m in metrics_list) / len(metrics_list),
        operations_per_second=sum(m.operations_per_second for m in metrics_list)
        / len(metrics_list),
        memory_start_kb=sum(m.memory_start_kb for m in metrics_list)
        // len(metrics_list),
        memory_end_kb=sum(m.memory_end_kb for m in metrics_list) // len(metrics_list),
        memory_peak_kb=sum(m.memory_peak_kb for m in metrics_list) // len(metrics_list),
        memory_allocated_kb=sum(m.memory_allocated_kb for m in metrics_list)
        // len(metrics_list),
        gc_count_gen0=sum(m.gc_count_gen0 for m in metrics_list) // len(metrics_list),
        gc_count_gen1=sum(m.gc_count_gen1 for m in metrics_list) // len(metrics_list),
        gc_count_gen2=sum(m.gc_count_gen2 for m in metrics_list) // len(metrics_list),
        gc_total_collections=sum(m.gc_total_collections for m in metrics_list)
        // len(metrics_list),
        objects_before=sum(m.objects_before for m in metrics_list) // len(metrics_list),
        objects_after=sum(m.objects_after for m in metrics_list) // len(metrics_list),
        objects_delta=sum(m.objects_delta for m in metrics_list) // len(metrics_list),
    )


class OptimizedTapeBenchmark:
    """Benchmark suite for the optimized tape-based reactive system."""

    def __init__(self):
        self.console = Console()
        self.results: List[BenchmarkMetrics] = []
        self.gc_metrics: Any = None
        self._in_forced_collection = False

    def run_comprehensive_benchmark(self):
        """Run all benchmark categories."""
        start_time = time.time()

        self._display_header()

        # Run benchmarks first (without profiling overhead)
        self._run_creation_benchmark()
        self._run_update_benchmark()
        self._run_chain_benchmark()
        self._run_fanout_benchmark()
        self._run_diamond_benchmark()
        self._run_dsl_composition_benchmark()
        self._run_parallel_execution_benchmark()
        self._run_fused_operations_benchmark()
        self._run_immediate_operands_benchmark()

        # Run separate profiling for GC metrics
        self.gc_metrics = self._profile_gc_metrics()

        # Display results
        self._display_performance_results()
        self._display_memory_results()
        self._display_gc_results()
        self._display_summary()

        elapsed = time.time() - start_time
        self.console.print(
            f"\n[dim]Benchmark suite completed in {elapsed:.2f} seconds[/dim]"
        )

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
            observable(float(i))
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_update_benchmark(self):
        """Run a large update benchmark for GC profiling."""
        # Create and update observables
        obs_list = [observable(0.0) for _ in range(5000)]
        for obs in obs_list:
            obs.set(1.0)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_chain_benchmark(self):
        """Run a large chain benchmark for GC profiling."""
        # Create a dependency chain
        base = observable(1.0)
        current = base
        for i in range(200):  # Larger chain
            current = current + float(i % 10)
        base.set(2.0)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _run_large_fanout_benchmark(self):
        """Run a large fan-out benchmark for GC profiling."""
        # Create a fan-out pattern
        base = observable(42.0)
        dependents = []
        for i in range(2000):  # Larger fan-out
            dep = base + float(i)
            dependents.append(dep)
        base.set(100.0)
        # Force GC collection to measure it (marked as forced)
        self._in_forced_collection = True
        gc.collect()
        self._in_forced_collection = False

    def _display_header(self):
        """Display the benchmark header."""
        header = Panel(
            "FynX Optimized Tape System Performance Benchmarks\n"
            f"{NUM_ITERATIONS} iterations per benchmark with GC profiling",
            title="Optimized Tape Benchmarks",
            border_style="blue",
        )
        self.console.print(header)
        self.console.print()

    def _run_creation_benchmark(self):
        """Benchmark observable creation performance."""
        self.console.print("[yellow]Running Observable Creation benchmark...[/yellow]")

        def creation_operation(n):
            graph = OptimizedTapeGraph()
            observables = [observable(float(i), graph) for i in range(n)]
            return observables

        result = run_adaptive_benchmark("Creation", creation_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Observable Creation: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_update_benchmark(self):
        """Benchmark individual update performance."""
        self.console.print("[yellow]Running Individual Updates benchmark...[/yellow]")

        def update_operation(n):
            graph = OptimizedTapeGraph()
            observables = [observable(0.0, graph) for _ in range(n)]
            graph.compile()

            for i, obs in enumerate(observables):
                obs.set(float(i * 2))
            return observables

        result = run_adaptive_benchmark("Updates", update_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Individual Updates: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_chain_benchmark(self):
        """Benchmark reactive chain propagation."""
        self.console.print("[yellow]Running Chain Propagation benchmark...[/yellow]")

        def chain_operation(n):
            graph = OptimizedTapeGraph()
            base = observable(1.0, graph)
            current = base

            # Build reactive dependency chain
            for i in range(n):
                current = current + float(i % 10)

            graph.compile()
            base.set(2.0)  # Trigger reactive propagation
            return n

        result = run_adaptive_benchmark("Chain", chain_operation, lambda x: x)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Chain Propagation: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_fanout_benchmark(self):
        """Benchmark reactive fan-out performance."""
        self.console.print("[yellow]Running Reactive Fan-out benchmark...[/yellow]")

        def fanout_operation(n):
            graph = OptimizedTapeGraph()
            base = observable(42.0, graph)
            dependents = []

            # Create fan-out dependencies
            for i in range(n):
                # Each creates a reactive computation: (base + i) * (i + 1) + (i * 2)
                temp1 = base + float(i)
                temp2 = temp1 * float(i + 1)
                result = temp2 + float(i * 2)
                dependents.append(result)

            graph.compile()
            base.set(100.0)  # Trigger fan-out propagation
            return dependents

        result = run_adaptive_benchmark("Fan-out", fanout_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Reactive Fan-out: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_diamond_benchmark(self):
        """Benchmark diamond dependency pattern."""
        self.console.print("[yellow]Running Diamond Pattern benchmark...[/yellow]")

        def diamond_operation(n):
            graph = OptimizedTapeGraph()

            for _ in range(n):
                x = observable(0.0, graph)
                a = x * 2.0  # x -> a
                b = x + 10.0  # x -> b
                result = a + b  # a,b -> result (diamond)

            graph.compile()

            # Trigger all diamonds
            for obs in graph.input_indices:
                graph.set_input_value(obs, 1.0)

            graph.execute()
            return n

        result = run_adaptive_benchmark("Diamond", diamond_operation, lambda x: x)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Diamond Pattern: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_dsl_composition_benchmark(self):
        """Benchmark DSL composition performance."""
        self.console.print("[yellow]Running DSL Composition benchmark...[/yellow]")

        def dsl_operation(n):
            graph = OptimizedTapeGraph()
            observables = []

            for i in range(n):
                x = observable(0.0, graph)
                # Complex composition: x -> (x*2) -> ((x*2)+10) -> (((x*2)+10)*3)
                result = ((x * 2.0) + 10.0) * 3.0
                observables.append(result)

            graph.compile()

            # Trigger all compositions
            for obs in observables:
                obs.set(float(i))

            return observables

        result = run_adaptive_benchmark("DSL Composition", dsl_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] DSL Composition: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_parallel_execution_benchmark(self):
        """Benchmark parallel execution performance."""
        self.console.print("[yellow]Running Parallel Execution benchmark...[/yellow]")

        def parallel_operation(n):
            graph = OptimizedTapeGraph()
            observables = []

            # Create independent reactive computations
            for i in range(n):
                x = observable(0.0, graph)
                # Independent computation: (x * i) + (i * 10)
                result = (x * float(i)) + float(i * 10)
                observables.append(result)

            graph.compile(enable_parallel=True)

            # Trigger all computations
            for i, obs in enumerate(observables):
                obs.set(float(i))

            return observables

        result = run_adaptive_benchmark("Parallel Execution", parallel_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Parallel Execution: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_fused_operations_benchmark(self):
        """Benchmark fused operations performance."""
        self.console.print("[yellow]Running Fused Operations benchmark...[/yellow]")

        def fused_operation(n):
            graph = OptimizedTapeGraph()
            base = observable(1.0, graph)

            # Create fused multiply-add operations
            fused_results = []
            for i in range(n):
                # FMADD: base * i + (i * 2)
                result = base.fmadd(float(i), float(i * 2))
                fused_results.append(result)

            graph.compile()
            base.set(2.0)  # Trigger all fused operations
            return fused_results

        result = run_adaptive_benchmark("Fused Operations", fused_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Fused Operations: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _run_immediate_operands_benchmark(self):
        """Benchmark immediate operands performance."""
        self.console.print("[yellow]Running Immediate Operands benchmark...[/yellow]")

        def immediate_operation(n):
            graph = OptimizedTapeGraph()
            base = observable(1.0, graph)

            # Create operations with immediate operands (no node dependencies)
            immediate_results = []
            for i in range(n):
                # All operations use immediate values
                temp1 = base + float(i)  # ADD with immediate
                temp2 = temp1 * float(i + 1)  # MUL with immediate
                result = temp2 + float(i * 2)  # ADD with immediate
                immediate_results.append(result)

            graph.compile()
            base.set(2.0)  # Trigger all operations
            return immediate_results

        result = run_adaptive_benchmark("Immediate Operands", immediate_operation, len)
        self.results.append(result)

        self.console.print(
            f"[green]✓[/green] Immediate Operands: {result.operations_per_second:,.0f} ops/sec "
            f"(n={result.max_n:,})"
        )

    def _display_performance_results(self):
        """Display performance comparison results."""
        self.console.print()

        table = Table(title="Performance Results")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Operations/sec", style="green", justify="right")
        table.add_column("Max N", style="yellow", justify="right")
        table.add_column("Time (sec)", style="blue", justify="right")

        for result in self.results:
            table.add_row(
                result.operation,
                f"{result.operations_per_second:,.0f}",
                f"{result.max_n:,}",
                f"{result.operation_time:.4f}",
            )

        self.console.print(table)

    def _display_memory_results(self):
        """Display memory usage results."""
        self.console.print()

        table = Table(title="Memory Usage Results")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Peak Memory", style="yellow", justify="right")
        table.add_column("Allocated", style="green", justify="right")
        table.add_column("Object Δ", style="blue", justify="right")

        for result in self.results:
            table.add_row(
                result.operation,
                f"{result.memory_peak_kb:,} KB",
                f"{result.memory_allocated_kb:,} KB",
                f"{result.objects_delta:,}",
            )

        self.console.print(table)

        # Show detailed GC metrics if available
        if hasattr(self, "gc_metrics") and self.gc_metrics:
            # Calculate per-object memory estimates
            gc_profiling_objects = (
                10000 + 5000 + 50 + 2000
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

    def _display_gc_results(self):
        """Display GC statistics."""
        self.console.print()

        table = Table(title="Garbage Collection Statistics")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Total GCs", style="red", justify="right")
        table.add_column("Gen0", style="yellow", justify="right")
        table.add_column("Gen1", style="yellow", justify="right")
        table.add_column("Gen2", style="yellow", justify="right")

        for result in self.results:
            table.add_row(
                result.operation,
                str(result.gc_total_collections),
                str(result.gc_count_gen0),
                str(result.gc_count_gen1),
                str(result.gc_count_gen2),
            )

        self.console.print(table)

        # Show detailed GC analysis if available
        if hasattr(self, "gc_metrics") and self.gc_metrics:
            self.console.print()
            self.console.print(
                "━━━━━━━━━━━━━━━━━━━━━━━ GC Analysis ━━━━━━━━━━━━━━━━━━━━━━━"
            )

            self.console.print()
            self.console.print("Detailed GC Profiling Results")
            self.console.print(f"┣━ Natural Collections: {self.gc_metrics.collections}")
            self.console.print(f"┣━ Total GC Time: {self.gc_metrics.gc_time_ms:.2f}ms")
            self.console.print(
                f"┣━ GC Time Percentage: {self.gc_metrics.gc_percentage:.3f}%"
            )
            self.console.print(
                f"┗━ Objects Collected: {self.gc_metrics.objects_collected:,}"
            )
        else:
            self.console.print()

    def _display_summary(self):
        """Display overall summary."""
        self.console.print()

        total_ops_per_sec = sum(r.operations_per_second for r in self.results)
        avg_ops_per_sec = total_ops_per_sec / len(self.results)

        total_memory_peak = sum(r.memory_peak_kb for r in self.results)
        avg_memory_peak = total_memory_peak / len(self.results)

        total_gc = sum(r.gc_total_collections for r in self.results)
        avg_gc = total_gc / len(self.results)

        summary = Panel(
            f"Average Performance: {avg_ops_per_sec:,.0f} ops/sec\n"
            f"Average Peak Memory: {avg_memory_peak:,.0f} KB\n"
            f"Average GC Collections: {avg_gc:.1f}\n"
            f"Total Benchmarks: {len(self.results)}",
            title="Benchmark Summary",
            border_style="green",
        )
        self.console.print(summary)


def print_config():
    """Print the current benchmark configuration."""
    print("FynX Optimized Tape Benchmark Configuration:")
    print(f"  TIME_LIMIT_SECONDS: {TIME_LIMIT_SECONDS}")
    print(f"  STARTING_N: {STARTING_N}")
    print(f"  SCALE_FACTOR: {SCALE_FACTOR}")
    print(f"  NUM_ITERATIONS: {NUM_ITERATIONS}")
    print("\nBenchmark Categories:")
    print("  - Observable Creation")
    print("  - Individual Updates")
    print("  - Chain Propagation")
    print("  - Reactive Fan-out")
    print("  - Diamond Pattern")
    print("  - DSL Composition")
    print("  - Parallel Execution")
    print("  - Fused Operations")
    print("  - Immediate Operands")


def main():
    """Main entry point for the optimized tape benchmark suite."""
    parser = argparse.ArgumentParser(
        description="FynX Optimized Tape System Performance Benchmarks"
    )
    parser.add_argument(
        "--config", action="store_true", help="Show current benchmark configuration"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (reduced time limits)",
    )

    args = parser.parse_args()

    if args.config:
        print_config()
        return

    if args.quick:
        global TIME_LIMIT_SECONDS
        TIME_LIMIT_SECONDS = 0.5  # Faster for quick runs

    if not args.quick:
        print_config()
        print()

    benchmark = OptimizedTapeBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
