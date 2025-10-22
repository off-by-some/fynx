#!/usr/bin/env python3
"""
FynX vs RxPY Performance Comparison with Proper GC Analysis

This script compares FynX reactive system performance against RxPY with accurate
garbage collection and memory profiling during actual benchmark execution.

Benchmark Categories:
- Observable Creation: Creating observables/subjects
- Individual Updates: Setting values on observables
- Chain Propagation: Chaining transformations
- Reactive Fan-out: One-to-many reactive relationships
- Stream Combination: Merge and zip operations
- Throttling/Debouncing: Rate limiting operations
- Buffering/Windowing: Batch processing operations
- Conditional Emission: Take until/skip until operations
"""

import argparse
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, TypeVar

sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fynx import observable

try:
    import rx
except ImportError:
    RXPY_AVAILABLE = False
    print("RxPY not available. Install with: poetry install --with benchmark")
    sys.exit(1)

from rx import interval, merge, of
from rx import operators as ops
from rx import zip as rx_zip
from rx.subject import Subject

RXPY_AVAILABLE = True

T = TypeVar("T")

# Configuration
TIME_LIMIT_SECONDS = 1.0
STARTING_N = 10
SCALE_FACTOR = 1.5
NUM_ITERATIONS = 1  # Run multiple times to average out GC variance


@dataclass
class BenchmarkMetrics:
    """Complete metrics from a benchmark run."""

    library: str
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
        self, library: str, operation: str, n: int, operations_performed: int
    ) -> BenchmarkMetrics:
        """Calculate and return all metrics."""
        elapsed = self.get_elapsed_time()
        ops_per_sec = operations_performed / elapsed if elapsed > 0 else 0

        # Calculate GC collections that occurred during benchmark
        gc_gen0 = max(0, self.gc_stats_after[0] - self.gc_stats_before[0])
        gc_gen1 = max(0, self.gc_stats_after[1] - self.gc_stats_before[1])
        gc_gen2 = max(0, self.gc_stats_after[2] - self.gc_stats_before[2])

        return BenchmarkMetrics(
            library=library,
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
    library: str,
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

        metrics = profiler.get_metrics(library, operation, n, ops_performed)
        all_metrics.append(metrics)

    # Return averaged metrics
    return average_metrics(all_metrics)


def average_metrics(metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Average multiple benchmark runs."""
    if not metrics_list:
        raise ValueError("No metrics to average")

    first = metrics_list[0]

    return BenchmarkMetrics(
        library=first.library,
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


class FynxRxpyComparison:
    """Compare FynX and RxPY performance with proper GC analysis."""

    def __init__(self):
        self.console = Console()
        self.results: List[BenchmarkMetrics] = []

    def run_comparison(self):
        """Run all comparison benchmarks."""
        start_time = time.time()

        self._display_header()

        # Run benchmarks
        self._run_creation_comparison()
        self._run_update_comparison()
        self._run_chain_comparison()
        self._run_fanout_comparison()
        self._run_stream_combination_comparison()
        self._run_throttling_debouncing_comparison()
        self._run_buffering_windowing_comparison()
        self._run_conditional_emission_comparison()

        # Display results
        self._display_comparison_results()
        self._display_memory_comparison()
        self._display_gc_comparison()
        self._display_summary()

        elapsed = time.time() - start_time
        self.console.print(
            f"\n[dim]Comparison completed in {elapsed:.2f} seconds[/dim]"
        )

    def _display_header(self):
        """Display the comparison header."""
        header = Panel(
            f"FynX vs RxPY Performance Comparison\n{NUM_ITERATIONS} iterations per benchmark",
            title="⚔️  Library Comparison",
            border_style="blue",
        )
        self.console.print(header)
        self.console.print()

    def _run_creation_comparison(self):
        """Compare observable creation performance."""
        self.console.print("[yellow]Running Observable Creation comparison...[/yellow]")

        # FynX creation
        def fynx_operation(n):
            return [observable(i) for i in range(n)]

        fynx_result = run_adaptive_benchmark("FynX", "Creation", fynx_operation, len)
        self.results.append(fynx_result)

        # RxPY creation
        def rxpy_operation(n):
            subjects = []
            for i in range(n):
                subject = Subject()
                subject.on_next(i)
                subjects.append(subject)
            return subjects

        rxpy_result = run_adaptive_benchmark("RxPY", "Creation", rxpy_operation, len)
        self.results.append(rxpy_result)

        self._display_progress("Observable Creation", fynx_result, rxpy_result)

    def _run_update_comparison(self):
        """Compare update performance."""
        self.console.print("[yellow]Running Individual Updates comparison...[/yellow]")

        # FynX updates
        def fynx_operation(n):
            observables = [observable(0) for _ in range(n)]
            for i, obs in enumerate(observables):
                obs.set(i * 2)
            return observables

        fynx_result = run_adaptive_benchmark("FynX", "Updates", fynx_operation, len)
        self.results.append(fynx_result)

        # RxPY updates
        def rxpy_operation(n):
            subjects = [Subject() for _ in range(n)]
            for i, subject in enumerate(subjects):
                subject.on_next(i * 2)
            return subjects

        rxpy_result = run_adaptive_benchmark("RxPY", "Updates", rxpy_operation, len)
        self.results.append(rxpy_result)

        self._display_progress("Individual Updates", fynx_result, rxpy_result)

    def _run_chain_comparison(self):
        """Compare chain propagation performance."""
        self.console.print("[yellow]Running Chain Propagation comparison...[/yellow]")

        # FynX chain
        def fynx_operation(n):
            base = observable(1)
            current = base
            for i in range(n):
                current = current.then(lambda x, i=i: x + i)
            base.set(2)
            return n

        fynx_result = run_adaptive_benchmark(
            "FynX", "Chain", fynx_operation, lambda x: x
        )
        self.results.append(fynx_result)

        # RxPY chain
        def rxpy_operation(n):
            base = Subject()
            current = base
            for i in range(n):
                current = current.pipe(ops.map(lambda x, i=i: x + i))
            base.on_next(2)
            return n

        rxpy_result = run_adaptive_benchmark(
            "RxPY", "Chain", rxpy_operation, lambda x: x
        )
        self.results.append(rxpy_result)

        self._display_progress("Chain Propagation", fynx_result, rxpy_result)

    def _run_fanout_comparison(self):
        """Compare fan-out performance."""
        self.console.print("[yellow]Running Reactive Fan-out comparison...[/yellow]")

        # FynX fan-out
        def fynx_operation(n):
            base = observable(42)
            dependents = [base.then(lambda x, i=i: x + i) for i in range(n)]
            base.set(100)
            return n

        fynx_result = run_adaptive_benchmark(
            "FynX", "Fan-out", fynx_operation, lambda x: x
        )
        self.results.append(fynx_result)

        # RxPY fan-out
        def rxpy_operation(n):
            base = Subject()
            dependents = [base.pipe(ops.map(lambda x, i=i: x + i)) for i in range(n)]
            base.on_next(100)
            return n

        rxpy_result = run_adaptive_benchmark(
            "RxPY", "Fan-out", rxpy_operation, lambda x: x
        )
        self.results.append(rxpy_result)

        self._display_progress("Reactive Fan-out", fynx_result, rxpy_result)

    def _run_stream_combination_comparison(self):
        """Compare stream combination (merge/zip) performance."""
        self.console.print("[yellow]Running Stream Combination comparison...[/yellow]")

        # FynX merge (using alongside)
        def fynx_merge_operation(n):
            obs1 = observable(1)
            obs2 = observable(2)
            merged = obs1.alongside(obs2)
            # Trigger updates to test merge performance
            for i in range(n):
                obs1.set(i)
                obs2.set(i * 2)
            return n

        fynx_merge_result = run_adaptive_benchmark(
            "FynX", "Stream Merge", fynx_merge_operation, lambda x: x
        )
        self.results.append(fynx_merge_result)

        # RxPY merge
        def rxpy_merge_operation(n):
            obs1 = Subject()
            obs2 = Subject()
            merged = merge(obs1, obs2)
            # Trigger updates to test merge performance
            for i in range(n):
                obs1.on_next(i)
                obs2.on_next(i * 2)
            return n

        rxpy_merge_result = run_adaptive_benchmark(
            "RxPY", "Stream Merge", rxpy_merge_operation, lambda x: x
        )
        self.results.append(rxpy_merge_result)

        self._display_progress("Stream Merge", fynx_merge_result, rxpy_merge_result)

        # FynX zip (using new zip method)
        def fynx_zip_operation(n):
            obs1 = observable(1)
            obs2 = observable(2)
            zipped = obs1.zip(obs2)
            # Trigger updates to test zip performance
            for i in range(n):
                obs1.set(i)
                obs2.set(i * 2)
            return n

        fynx_zip_result = run_adaptive_benchmark(
            "FynX", "Stream Zip", fynx_zip_operation, lambda x: x
        )
        self.results.append(fynx_zip_result)

        # RxPY zip
        def rxpy_zip_operation(n):
            obs1 = Subject()
            obs2 = Subject()
            zipped = rx_zip(obs1, obs2, lambda x, y: (x, y))
            # Trigger updates to test zip performance
            for i in range(n):
                obs1.on_next(i)
                obs2.on_next(i * 2)
            return n

        rxpy_zip_result = run_adaptive_benchmark(
            "RxPY", "Stream Zip", rxpy_zip_operation, lambda x: x
        )
        self.results.append(rxpy_zip_result)

        self._display_progress("Stream Zip", fynx_zip_result, rxpy_zip_result)

    def _run_throttling_debouncing_comparison(self):
        """Compare throttling/debouncing performance."""
        self.console.print(
            "[yellow]Running Throttling/Debouncing comparison...[/yellow]"
        )

        # FynX debounce (using new debounce method)
        def fynx_debounce_operation(n):
            obs = observable(0)
            # Use proper time-based debouncing
            debounced = obs.debounce(1.0)  # 1ms debounce window
            # Trigger rapid updates
            for i in range(n):
                obs.set(i)
            return n

        fynx_debounce_result = run_adaptive_benchmark(
            "FynX", "Debounce", fynx_debounce_operation, lambda x: x
        )
        self.results.append(fynx_debounce_result)

        # RxPY debounce
        def rxpy_debounce_operation(n):
            obs = Subject()
            debounced = obs.pipe(ops.debounce(timedelta(milliseconds=1)))
            # Trigger rapid updates
            for i in range(n):
                obs.on_next(i)
            return n

        rxpy_debounce_result = run_adaptive_benchmark(
            "RxPY", "Debounce", rxpy_debounce_operation, lambda x: x
        )
        self.results.append(rxpy_debounce_result)

        self._display_progress("Debounce", fynx_debounce_result, rxpy_debounce_result)

        # FynX throttle simulation (using conditional with timing)
        def fynx_throttle_operation(n):
            obs = observable(0)
            # Simulate throttle by filtering rapid updates
            throttled = obs.requiring(
                lambda x: x % 5 == 0
            )  # Only emit every 5th update
            # Trigger rapid updates
            for i in range(n):
                obs.set(i)
            return n

        fynx_throttle_result = run_adaptive_benchmark(
            "FynX", "Throttle", fynx_throttle_operation, lambda x: x
        )
        self.results.append(fynx_throttle_result)

        # RxPY throttle
        def rxpy_throttle_operation(n):
            obs = Subject()
            throttled = obs.pipe(ops.throttle_with_timeout(timedelta(milliseconds=1)))
            # Trigger rapid updates
            for i in range(n):
                obs.on_next(i)
            return n

        rxpy_throttle_result = run_adaptive_benchmark(
            "RxPY", "Throttle", rxpy_throttle_operation, lambda x: x
        )
        self.results.append(rxpy_throttle_result)

        self._display_progress("Throttle", fynx_throttle_result, rxpy_throttle_result)

    def _run_buffering_windowing_comparison(self):
        """Compare buffering/windowing performance."""
        self.console.print("[yellow]Running Buffering/Windowing comparison...[/yellow]")

        # FynX buffer simulation (using computed with accumulation)
        def fynx_buffer_operation(n):
            obs = observable([])
            # Simulate buffering by accumulating values
            buffered = obs.then(lambda x: x if len(x) >= 10 else None)
            # Trigger updates to fill buffer
            for i in range(n):
                current = obs.value or []
                current.append(i)
                if len(current) >= 10:
                    obs.set(current)
                    obs.set([])  # Reset buffer
                else:
                    obs.set(current)
            return n

        fynx_buffer_result = run_adaptive_benchmark(
            "FynX", "Buffer", fynx_buffer_operation, lambda x: x
        )
        self.results.append(fynx_buffer_result)

        # RxPY buffer
        def rxpy_buffer_operation(n):
            obs = Subject()
            buffered = obs.pipe(ops.buffer_with_count(10))
            # Trigger updates to fill buffer
            for i in range(n):
                obs.on_next(i)
            return n

        rxpy_buffer_result = run_adaptive_benchmark(
            "RxPY", "Buffer", rxpy_buffer_operation, lambda x: x
        )
        self.results.append(rxpy_buffer_result)

        self._display_progress("Buffer", fynx_buffer_result, rxpy_buffer_result)

        # FynX window simulation (using computed with windowing)
        def fynx_window_operation(n):
            obs = observable([])
            # Simulate windowing by creating sliding windows
            windowed = obs.then(
                lambda x: (
                    [x[i : i + 5] for i in range(0, len(x), 5)] if len(x) >= 5 else None
                )
            )
            # Trigger updates to create windows
            for i in range(n):
                current = obs.value or []
                current.append(i)
                obs.set(current)
            return n

        fynx_window_result = run_adaptive_benchmark(
            "FynX", "Window", fynx_window_operation, lambda x: x
        )
        self.results.append(fynx_window_result)

        # RxPY window
        def rxpy_window_operation(n):
            obs = Subject()
            windowed = obs.pipe(ops.window_with_count(5))
            # Trigger updates to create windows
            for i in range(n):
                obs.on_next(i)
            return n

        rxpy_window_result = run_adaptive_benchmark(
            "RxPY", "Window", rxpy_window_operation, lambda x: x
        )
        self.results.append(rxpy_window_result)

        self._display_progress("Window", fynx_window_result, rxpy_window_result)

    def _run_conditional_emission_comparison(self):
        """Compare conditional emission (take_until/skip_until) performance."""
        self.console.print(
            "[yellow]Running Conditional Emission comparison...[/yellow]"
        )

        # FynX take_until simulation (using conditional with counter)
        def fynx_take_until_operation(n):
            obs = observable(0)
            trigger = observable(False)
            # Simulate take_until by stopping when trigger is True
            taken = obs.requiring(lambda x: not trigger.value)
            # Trigger updates and eventually trigger stop
            for i in range(n):
                obs.set(i)
                if i >= n // 2:  # Trigger stop halfway through
                    trigger.set(True)
            return n

        fynx_take_until_result = run_adaptive_benchmark(
            "FynX", "Take Until", fynx_take_until_operation, lambda x: x
        )
        self.results.append(fynx_take_until_result)

        # RxPY take_until
        def rxpy_take_until_operation(n):
            obs = Subject()
            trigger = Subject()
            taken = obs.pipe(ops.take_until(trigger))
            # Trigger updates and eventually trigger stop
            for i in range(n):
                obs.on_next(i)
                if i >= n // 2:  # Trigger stop halfway through
                    trigger.on_next(True)
            return n

        rxpy_take_until_result = run_adaptive_benchmark(
            "RxPY", "Take Until", rxpy_take_until_operation, lambda x: x
        )
        self.results.append(rxpy_take_until_result)

        self._display_progress(
            "Take Until", fynx_take_until_result, rxpy_take_until_result
        )

        # FynX skip_until simulation (using conditional with counter)
        def fynx_skip_until_operation(n):
            obs = observable(0)
            trigger = observable(False)
            # Simulate skip_until by only emitting after trigger is True
            skipped = obs.requiring(lambda x: trigger.value)
            # Trigger updates and eventually trigger start
            for i in range(n):
                obs.set(i)
                if i >= n // 2:  # Trigger start halfway through
                    trigger.set(True)
            return n

        fynx_skip_until_result = run_adaptive_benchmark(
            "FynX", "Skip Until", fynx_skip_until_operation, lambda x: x
        )
        self.results.append(fynx_skip_until_result)

        # RxPY skip_until
        def rxpy_skip_until_operation(n):
            obs = Subject()
            trigger = Subject()
            skipped = obs.pipe(ops.skip_until(trigger))
            # Trigger updates and eventually trigger start
            for i in range(n):
                obs.on_next(i)
                if i >= n // 2:  # Trigger start halfway through
                    trigger.on_next(True)
            return n

        rxpy_skip_until_result = run_adaptive_benchmark(
            "RxPY", "Skip Until", rxpy_skip_until_operation, lambda x: x
        )
        self.results.append(rxpy_skip_until_result)

        self._display_progress(
            "Skip Until", fynx_skip_until_result, rxpy_skip_until_result
        )

    def _display_progress(
        self,
        operation_name: str,
        fynx_result: BenchmarkMetrics,
        rxpy_result: BenchmarkMetrics,
    ):
        """Display progress for a comparison."""
        fynx_ops = fynx_result.operations_per_second
        rxpy_ops = rxpy_result.operations_per_second

        if fynx_ops > rxpy_ops:
            speedup = fynx_ops / rxpy_ops
            winner_text = f"[green]FynX {speedup:.1f}x faster[/green]"
        else:
            speedup = rxpy_ops / fynx_ops
            winner_text = f"[blue]RxPY {speedup:.1f}x faster[/blue]"

        self.console.print(
            f"[green]✓[/green] {operation_name}: "
            f"FynX {fynx_ops:,.0f} ops/sec vs RxPY {rxpy_ops:,.0f} ops/sec ({winner_text})"
        )

    def _display_comparison_results(self):
        """Display detailed performance comparison."""
        self.console.print()

        table = Table(title="📊 Performance Comparison")
        table.add_column("Operation", style="cyan")
        table.add_column("FynX ops/sec", style="green", justify="right")
        table.add_column("RxPY ops/sec", style="blue", justify="right")
        table.add_column("Winner", style="yellow", justify="center")
        table.add_column("Speedup", style="magenta", justify="right")

        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = {}
            operations[result.operation][result.library] = result

        for op_name, lib_results in operations.items():
            fynx = lib_results.get("FynX")
            rxpy = lib_results.get("RxPY")

            if fynx and rxpy:
                if fynx.operations_per_second > rxpy.operations_per_second:
                    winner = "FynX"
                    speedup = f"{fynx.operations_per_second / rxpy.operations_per_second:.2f}x"
                else:
                    winner = "RxPY"
                    speedup = f"{rxpy.operations_per_second / fynx.operations_per_second:.2f}x"

                table.add_row(
                    op_name,
                    f"{fynx.operations_per_second:,.0f}",
                    f"{rxpy.operations_per_second:,.0f}",
                    winner,
                    speedup,
                )

        self.console.print(table)

    def _display_memory_comparison(self):
        """Display memory usage comparison."""
        self.console.print()

        table = Table(title="💾 Memory Usage Comparison")
        table.add_column("Operation", style="cyan")
        table.add_column("Library", style="white")
        table.add_column("Peak Memory", style="yellow", justify="right")
        table.add_column("Allocated", style="green", justify="right")
        table.add_column("Object Δ", style="blue", justify="right")

        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = {}
            operations[result.operation][result.library] = result

        for op_name, lib_results in operations.items():
            for lib_name in ["FynX", "RxPY"]:
                if lib_name in lib_results:
                    r = lib_results[lib_name]
                    table.add_row(
                        op_name if lib_name == "FynX" else "",
                        lib_name,
                        f"{r.memory_peak_kb:,} KB",
                        f"{r.memory_allocated_kb:,} KB",
                        f"{r.objects_delta:,}",
                    )

        self.console.print(table)

    def _display_gc_comparison(self):
        """Display GC statistics comparison."""
        self.console.print()

        table = Table(title="🗑️  Garbage Collection Statistics")
        table.add_column("Operation", style="cyan")
        table.add_column("Library", style="white")
        table.add_column("Total GCs", style="red", justify="right")
        table.add_column("Gen0", style="yellow", justify="right")
        table.add_column("Gen1", style="yellow", justify="right")
        table.add_column("Gen2", style="yellow", justify="right")

        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = {}
            operations[result.operation][result.library] = result

        for op_name, lib_results in operations.items():
            for lib_name in ["FynX", "RxPY"]:
                if lib_name in lib_results:
                    r = lib_results[lib_name]
                    table.add_row(
                        op_name if lib_name == "FynX" else "",
                        lib_name,
                        str(r.gc_total_collections),
                        str(r.gc_count_gen0),
                        str(r.gc_count_gen1),
                        str(r.gc_count_gen2),
                    )

        self.console.print(table)

    def _display_summary(self):
        """Display overall summary."""
        self.console.print()

        fynx_wins = sum(
            1
            for i in range(0, len(self.results), 2)
            if self.results[i].operations_per_second
            > self.results[i + 1].operations_per_second
        )
        rxpy_wins = len(self.results) // 2 - fynx_wins

        winner = (
            "FynX"
            if fynx_wins > rxpy_wins
            else "RxPY" if rxpy_wins > fynx_wins else "Tie"
        )
        winner_color = (
            "green" if winner == "FynX" else "blue" if winner == "RxPY" else "yellow"
        )

        summary = Panel(
            f"Overall Winner: {winner}\nFynX wins: {fynx_wins}\nRxPY wins: {rxpy_wins}",
            title="🏆 Performance Summary",
            border_style=winner_color,
        )
        self.console.print(summary)


def print_config():
    """Print the current benchmark configuration."""
    print("FynX vs RxPY Comparison Configuration:")
    print(f"  TIME_LIMIT_SECONDS: {TIME_LIMIT_SECONDS}")
    print(f"  STARTING_N: {STARTING_N}")
    print(f"  SCALE_FACTOR: {SCALE_FACTOR}")
    print(f"  NUM_ITERATIONS: {NUM_ITERATIONS}")
    print("\nBenchmark Categories:")
    print("  - Observable Creation")
    print("  - Individual Updates")
    print("  - Chain Propagation")
    print("  - Reactive Fan-out")
    print("  - Stream Combination (Merge/Zip)")
    print("  - Throttling/Debouncing")
    print("  - Buffering/Windowing")
    print("  - Conditional Emission (Take Until/Skip Until)")


def main():
    """Main entry point for the comparison script."""
    parser = argparse.ArgumentParser(description="FynX vs RxPY Performance Comparison")
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

    if not args.quiet:
        print_config()
        print()

    comparison = FynxRxpyComparison()
    comparison.run_comparison()


if __name__ == "__main__":
    main()
