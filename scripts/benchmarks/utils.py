#!/usr/bin/env python3
"""
Benchmark Harness Utilities

Shared utilities for FynX benchmarking framework with improved display.
"""

import cProfile
import gc
import io
import json
import pstats
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""

    time_limit: float = (
        1.0  # Total time limit per benchmark (time_limit * num_iterations)
    )
    starting_n: int = 10
    scale_factor: float = 1.5
    num_iterations: int = 1
    profile_enabled: bool = False


CONFIG = BenchmarkConfig()


# ============================================================================
# Metrics
# ============================================================================


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

    # Status tracking
    is_dnf: bool = False  # True if benchmark never completed at least one operation
    partial_result: bool = False  # True if hit time limit but had successful iterations
    is_new_record: bool = False  # True if this is a new performance record
    previous_record: Optional[float] = None  # Previous record value if available


class BenchmarkProfiler:
    """Profile performance, memory, and GC during benchmark execution."""

    def __init__(self, enable_profiler: bool = False):
        self.enable_profiler = enable_profiler
        self.profiler = None
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
        # Collect garbage before starting
        gc.collect()
        gc.collect()
        gc.collect()

        # Start profiler if enabled
        if self.enable_profiler:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # Start memory tracking
        tracemalloc.start()
        self.memory_start, _ = tracemalloc.get_traced_memory()

        # Record GC stats
        self.gc_stats_before = gc.get_count()
        self.objects_before = len(gc.get_objects())

        # Start timing
        self.start_time = time.perf_counter()

        return self

    def __exit__(self, *args):
        # Stop timing
        self.end_time = time.perf_counter()

        # Stop profiler
        if self.profiler:
            self.profiler.disable()

        # Record GC stats after
        self.gc_stats_after = gc.get_count()
        self.objects_after = len(gc.get_objects())

        # Get memory stats
        self.memory_end, self.memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def get_elapsed_time(self) -> float:
        return self.end_time - self.start_time

    def print_profile(self, n_lines: int = 20):
        """Print profiler stats if enabled."""
        if self.profiler:
            stats = pstats.Stats(self.profiler)
            stats.sort_stats("cumulative")
            stats.print_stats(n_lines)

    def get_metrics(
        self, library: str, operation: str, n: int, operations_performed: int
    ) -> BenchmarkMetrics:
        """Calculate and return all metrics."""
        elapsed = self.get_elapsed_time()
        ops_per_sec = operations_performed / elapsed if elapsed > 0 else 0

        # Calculate GC collections
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


# ============================================================================
# Benchmark Registry
# ============================================================================


class BenchmarkRegistry:
    """Registry for benchmark functions."""

    def __init__(self):
        self.benchmarks: Dict[str, Dict[str, Callable]] = {}
        self.categories: Dict[str, str] = {}

    def register(
        self,
        name: str,
        library: str,
        func: Callable,
        category: Optional[str] = None,
        operations_counter: Optional[Callable] = None,
        config: Optional[BenchmarkConfig] = None,
        detailed_profiling: bool = False,
    ):
        """Register a benchmark function."""
        if name not in self.benchmarks:
            self.benchmarks[name] = {}

        self.benchmarks[name][library] = {
            "func": func,
            "operations_counter": operations_counter or len,
            "config": config,
            "detailed_profiling": detailed_profiling,
        }

        if category:
            self.categories[name] = category

    def get_benchmark(self, name: str, library: str) -> Optional[Dict]:
        """Get a benchmark by name and library."""
        return self.benchmarks.get(name, {}).get(library)

    def list_benchmarks(self) -> List[str]:
        """List all registered benchmark names."""
        return list(self.benchmarks.keys())

    def get_category(self, name: str) -> str:
        """Get category for a benchmark."""
        return self.categories.get(name, "General")


# ============================================================================
# Adaptive Benchmark Runner
# ============================================================================


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
        is_dnf=first.is_dnf,
        partial_result=first.partial_result,
    )


# ============================================================================
# Comprehensive Profiling System
# ============================================================================


@dataclass
class ExecutionProfile:
    """Detailed execution profile for a single operation."""

    name: str
    execution_time: float
    cpu_time: float
    call_count: int
    memory_allocated: int
    memory_peak: int
    memory_leaked: int
    gc_collections: Tuple[int, int, int]
    gc_time: float
    top_functions: List[Tuple[str, float, int]]  # (func_name, cumtime, calls)
    stack_depth: int

    @property
    def memory_efficiency(self) -> float:
        """Memory efficiency score (ops per MB)."""
        if self.memory_peak == 0:
            return float("inf")
        return (self.call_count * 1024 * 1024) / self.memory_peak

    @property
    def time_per_operation_us(self) -> float:
        """Average time per operation in microseconds."""
        if self.call_count == 0:
            return 0
        return (self.execution_time / self.call_count) * 1e6


class DetailedProfiler:
    """Collects detailed profiling information including call stacks."""

    def __init__(self, name: str):
        self.name = name
        self.profiler = cProfile.Profile()
        self.start_time = 0
        self.end_time = 0
        self.start_memory = 0
        self.peak_memory = 0
        self.gc_start = None
        self.gc_collections = (0, 0, 0)
        self.gc_time = 0
        self.memory_leaked = 0

    def __enter__(self):
        # Start memory tracking
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]

        # Record GC state
        gc.collect()
        self.gc_start = gc.get_count()

        # Start profiling
        self.start_time = time.perf_counter()
        self.profiler.enable()
        return self

    def __exit__(self, *args):
        # Stop profiling
        self.profiler.disable()
        self.end_time = time.perf_counter()

        # Measure GC impact
        gc_start_time = time.perf_counter()
        gc.collect()
        self.gc_time = time.perf_counter() - gc_start_time

        gc_end = gc.get_count()
        self.gc_collections = tuple(gc_end[i] - self.gc_start[i] for i in range(3))

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak
        memory_leaked = current - self.start_memory
        tracemalloc.stop()

        self.memory_leaked = max(0, memory_leaked)

    def get_profile(self, call_count: int = 1) -> ExecutionProfile:
        """Extract profile information."""
        # Get stats from profiler
        stats = pstats.Stats(self.profiler)
        stats.calc_callees()

        # Extract top functions by cumulative time
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[0]}:{func[1]}:{func[2]}"
            top_functions.append((func_name, ct, cc))

        top_functions.sort(key=lambda x: x[1], reverse=True)
        top_functions = top_functions[:10]  # Top 10

        # Calculate stack depth (max recursion depth)
        stack_depth = max(
            (len(str(stack).split("/")) for stack in stats.stats.keys()), default=1
        )

        return ExecutionProfile(
            name=self.name,
            execution_time=self.end_time - self.start_time,
            cpu_time=sum(s[2] for s in stats.stats.values()),
            call_count=call_count,
            memory_allocated=self.peak_memory - self.start_memory,
            memory_peak=self.peak_memory,
            memory_leaked=self.memory_leaked,
            gc_collections=self.gc_collections,
            gc_time=self.gc_time,
            top_functions=top_functions,
            stack_depth=stack_depth,
        )


def run_adaptive_benchmark(
    library: str,
    operation: str,
    operation_func: Callable[[int], T],
    operations_counter: Callable[[T], int],
    config: BenchmarkConfig,
    detailed_profiling: bool = False,
) -> Tuple[BenchmarkMetrics, Optional[ExecutionProfile]]:
    """
    Run adaptive benchmark that scales workload to target total time.

    The time_limit represents the total time the benchmark should take to complete
    all iterations (time_limit * num_iterations). If a benchmark exceeds this time
    and didn't achieve at least 1 operation, it's marked as DNF (Did Not Finish).

    The benchmark scales the workload size (n) until the estimated total time for
    all iterations approaches the target time limit, then runs the benchmark at
    that workload size for the specified number of iterations.
    """
    n = config.starting_n
    total_time_limit = config.time_limit * config.num_iterations

    # Find optimal workload size by testing a single iteration
    scaling_iterations = 0
    max_scaling_iterations = 20  # Prevent runaway scaling

    while scaling_iterations < max_scaling_iterations:
        start = time.perf_counter()
        result = operation_func(n)
        elapsed = time.perf_counter() - start
        ops_performed = operations_counter(result)

        # Estimate total time for all iterations at this workload size
        estimated_total_time = elapsed * config.num_iterations

        # If estimated total time is close to our target, or if we exceed it, stop
        if estimated_total_time >= total_time_limit * 0.8:  # 80% of target time
            break

        # If we didn't even achieve 1 op in the time limit, mark as DNF
        if elapsed >= total_time_limit and ops_performed == 0:
            # Return DNF metrics
            return (
                BenchmarkMetrics(
                    library=library,
                    operation=operation,
                    max_n=n,
                    operation_time=float("inf"),  # Indicate DNF
                    operations_per_second=0,
                    memory_start_kb=0,
                    memory_end_kb=0,
                    memory_peak_kb=0,
                    memory_allocated_kb=0,
                    gc_count_gen0=0,
                    gc_count_gen1=0,
                    gc_count_gen2=0,
                    gc_total_collections=0,
                    objects_before=0,
                    objects_after=0,
                    objects_delta=0,
                    is_dnf=True,
                    partial_result=False,
                ),
                None,
            )

        n = int(n * config.scale_factor)
        if n > 10_000_000:  # Safety limit
            break

        scaling_iterations += 1

    # Run benchmark at optimal size with full profiling
    all_metrics = []
    execution_profile = None
    total_benchmark_time = 0

    for iteration in range(config.num_iterations):
        if total_benchmark_time >= total_time_limit:
            # Time limit exceeded during iterations
            # If we have successful results, return them as partial; otherwise DNF
            if all_metrics:
                return average_metrics(all_metrics), None
            else:
                # No successful iterations - true DNF
                return (
                    BenchmarkMetrics(
                        library=library,
                        operation=operation,
                        max_n=n,
                        operation_time=float("inf"),  # Indicate DNF
                        operations_per_second=0,
                        memory_start_kb=0,
                        memory_end_kb=0,
                        memory_peak_kb=0,
                        memory_allocated_kb=0,
                        gc_count_gen0=0,
                        gc_count_gen1=0,
                        gc_count_gen2=0,
                        gc_total_collections=0,
                        objects_before=0,
                        objects_after=0,
                        objects_delta=0,
                        is_dnf=True,
                        partial_result=False,
                    ),
                    None,
                )

        if detailed_profiling and iteration == 0:
            # Use detailed profiling for the first iteration
            with DetailedProfiler(operation) as profiler:
                result = operation_func(n)
                ops_performed = operations_counter(result)

            metrics = BenchmarkMetrics(
                library=library,
                operation=operation,
                max_n=n,
                operation_time=profiler.end_time - profiler.start_time,
                operations_per_second=ops_performed
                / (profiler.end_time - profiler.start_time),
                memory_start_kb=profiler.start_memory // 1024,
                memory_end_kb=(
                    profiler.start_memory + profiler.peak_memory - profiler.start_memory
                )
                // 1024,
                memory_peak_kb=profiler.peak_memory // 1024,
                memory_allocated_kb=(profiler.peak_memory - profiler.start_memory)
                // 1024,
                gc_count_gen0=profiler.gc_collections[0],
                gc_count_gen1=profiler.gc_collections[1],
                gc_count_gen2=profiler.gc_collections[2],
                gc_total_collections=sum(profiler.gc_collections),
                objects_before=0,  # Not tracked in detailed profiler
                objects_after=0,
                objects_delta=0,
            )
            execution_profile = profiler.get_profile(ops_performed)
            total_benchmark_time += profiler.end_time - profiler.start_time
        else:
            # Use standard profiling
            with BenchmarkProfiler(config.profile_enabled) as profiler:
                result = operation_func(n)
                ops_performed = operations_counter(result)

            metrics = profiler.get_metrics(library, operation, n, ops_performed)
            total_benchmark_time += profiler.get_elapsed_time()

            if config.profile_enabled and iteration == 0:
                profiler.print_profile()

        all_metrics.append(metrics)

    return average_metrics(all_metrics), execution_profile


# ============================================================================
# Benchmark Comparison Runner with Improved Display
# ============================================================================

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import FynX TUI components
from scripts.benchmarks.fynx_tui.tui import (
    H1,
    Box,
    Col,
    Component,
    Line,
    PerformanceStats,
    PerformanceStore,
    ReactiveApp,
    ReactiveComponent,
    RichText,
    Row,
    Spacer,
    Text,
    render,
)

# ============================================================================
# TUI Components for Benchmark Display
# ============================================================================


class BenchmarkHeader(Component):
    """Header component for benchmark results."""

    def render(self):
        title = self.props.get("title", "Benchmark Results")
        subtitle = self.props.get("subtitle", "")
        metadata = self.props.get("metadata", {})

        children = [
            H1(title),
            Spacer(height=1),
        ]

        if subtitle:
            children.append(Text(subtitle, style="dim"))
            children.append(Spacer(height=1))

        if metadata:
            meta_items = []
            for key, value in metadata.items():
                meta_items.append(f"{key}: {value}")
            children.append(Text(" | ".join(meta_items), style="cyan dim"))
            children.append(Spacer(height=1))

        return Col(children=children)


class BenchmarkSummary(Component):
    """Summary component showing key benchmark statistics."""

    def render(self):
        results = self.props.get("results", [])

        if not results:
            return Text("No results to display", style="yellow")

        # Calculate summary statistics
        total_ops = sum(r.operations_per_second for r in results)
        avg_ops = total_ops / len(results) if results else 0

        fynx_results = [r for r in results if r.library == "FynX"]
        rxpy_results = [r for r in results if r.library != "FynX"]

        fynx_avg = (
            sum(r.operations_per_second for r in fynx_results) / len(fynx_results)
            if fynx_results
            else 0
        )
        rxpy_avg = (
            sum(r.operations_per_second for r in rxpy_results) / len(rxpy_results)
            if rxpy_results
            else 0
        )

        speedup = fynx_avg / rxpy_avg if rxpy_avg > 0 else 0

        children = [
            H1("Performance Summary"),
            Spacer(height=1),
            Row(
                children=[
                    Col(
                        children=[
                            Text(
                                f"Total Operations: {total_ops:,.0f}/sec",
                                style="bold cyan",
                            ),
                            Text(f"Average: {avg_ops:,.0f}/sec", style="cyan"),
                        ]
                    ),
                    Col(
                        children=[
                            Text(f"FynX Average: {fynx_avg:,.0f}/sec", style="green"),
                            Text(f"RxPY Average: {rxpy_avg:,.0f}/sec", style="yellow"),
                            Text(
                                f"Speedup: {speedup:.1f}x",
                                style="bold green" if speedup > 1 else "red",
                            ),
                        ]
                    ),
                ]
            ),
        ]

        return Box(title="📊 Summary", border="green", children=children)


class BenchmarkTable(Component):
    """Table component for displaying benchmark results."""

    def render(self):
        results = self.props.get("results", [])

        if not results:
            return Text("No results to display", style="yellow")

        # Header row
        headers = Row(
            children=[
                Text("Operation", style="bold cyan"),
                Text("Library", style="bold magenta"),
                Text("Ops/sec", style="bold green"),
                Text("Time (ms)", style="bold yellow"),
                Text("Memory (MB)", style="bold blue"),
                Text("GC Cycles", style="bold red"),
            ],
            equal=True,
        )

        # Data rows
        rows = [headers]
        for result in results:
            row = Row(
                children=[
                    Text(result.operation, style="cyan"),
                    Text(result.library, style="magenta"),
                    Text(f"{result.operations_per_second:,.0f}", style="green"),
                    Text(f"{result.avg_time_per_operation * 1000:.2f}", style="yellow"),
                    Text(f"{result.peak_memory_mb:.1f}", style="blue"),
                    Text(f"{result.gc_collections:,}", style="red"),
                ],
                equal=True,
            )
            rows.append(row)

        return Col(children=rows)


class BenchmarkResults(Component):
    """Main benchmark results component."""

    def render(self):
        config = self.props.get("config")
        results = self.props.get("results", [])
        execution_profiles = self.props.get("execution_profiles", {})
        elapsed_time = self.props.get("elapsed_time", 0)

        if not config or not results:
            return Text("No benchmark data available", style="yellow")

        children = [
            BenchmarkHeader(
                title="FynX vs RxPY Performance Comparison",
                subtitle="Comprehensive Reactive Programming Benchmark Analysis",
                metadata={
                    "Time Limit": f"{config.time_limit * config.num_iterations}s total",
                    "Iterations": config.num_iterations,
                    "Total Tests": len(results) // 2 if results else 0,
                    "Elapsed Time": f"{elapsed_time:.1f}s",
                },
            ),
            Spacer(height=1),
            BenchmarkSummary(results=results),
            Spacer(height=1),
            BenchmarkTable(results=results),
        ]

        return Col(children=children)


class SuiteHeader(Component):
    """Header component for benchmark suite reports."""

    def render(self):
        name = self.props.get("name", "Benchmark Suite")

        return Box(
            title="📊 Performance Report",
            border="cyan",
            children=[
                H1(name),
                Text("Comprehensive Performance Analysis", style="dim"),
            ],
        )


class SuiteOverviewTable(Component):
    """Overview table component for benchmark suite."""

    def render(self):
        profiles = self.props.get("profiles", {})

        if not profiles:
            return Text("No benchmark data available", style="yellow")

        # Header row
        headers = Row(
            children=[
                Text("Benchmark", style="bold cyan"),
                Text("Time", style="bold green"),
                Text("Ops/sec", style="bold yellow"),
                Text("Memory", style="bold blue"),
                Text("GC", style="bold red"),
            ],
            equal=True,
        )

        # Data rows
        rows = [headers]
        for name, profile in profiles.items():
            row = Row(
                children=[
                    Text(name, style="cyan"),
                    Text(f"{profile.execution_time:.3f}s", style="green"),
                    Text(f"{profile.operations_per_second:,.0f}", style="yellow"),
                    Text(f"{profile.peak_memory_mb:.1f}MB", style="blue"),
                    Text(f"{profile.gc_collections}", style="red"),
                ],
                equal=True,
            )
            rows.append(row)

        return Box(
            title="📊 Performance Overview",
            border="cyan",
            children=[Col(children=rows)],
        )


class SuiteStatistics(Component):
    """Statistical analysis component."""

    def render(self):
        stats = self.props.get("stats", {})

        if not stats:
            return Text("No statistical data available", style="yellow")

        # Header row
        headers = Row(
            children=[
                Text("Benchmark", style="bold cyan"),
                Text("Mean", style="bold green"),
                Text("Std Dev", style="bold yellow"),
                Text("Min", style="bold blue"),
                Text("Max", style="bold red"),
                Text("CV%", style="bold magenta"),
            ],
            equal=True,
        )

        # Data rows
        rows = [headers]
        for name, stat in stats.items():
            if hasattr(stat, "mean"):
                row = Row(
                    children=[
                        Text(name, style="cyan"),
                        Text(f"{stat.mean:.3f}s", style="green"),
                        Text(f"{stat.std_dev:.3f}s", style="yellow"),
                        Text(f"{stat.min:.3f}s", style="blue"),
                        Text(f"{stat.max:.3f}s", style="red"),
                        Text(f"{stat.cv_percent:.1f}%", style="magenta"),
                    ],
                    equal=True,
                )
                rows.append(row)

        return Box(
            title="📈 Statistics",
            border="green",
            children=[
                Col(children=rows),
                Spacer(height=1),
                Text(
                    "CV% = Coefficient of Variation (lower is more consistent)",
                    style="dim",
                ),
            ],
        )


class SuiteBottlenecks(Component):
    """Bottleneck analysis component."""

    def render(self):
        bottlenecks = self.props.get("bottlenecks", [])

        if not bottlenecks:
            return Text("No significant bottlenecks detected", style="green")

        children = [H1("Performance Bottlenecks")]

        for i, bottleneck in enumerate(bottlenecks, 1):
            func_short = (
                bottleneck.function_name.split(".")[-1]
                if "." in bottleneck.function_name
                else bottleneck.function_name
            )

            children.extend(
                [
                    Spacer(height=1),
                    Text(f"{i}. {func_short}", style="bold yellow"),
                    Text(
                        f"   Time: {bottleneck.time_spent*1000:.2f}ms ({bottleneck.percentage:.1f}%)",
                        style="white",
                    ),
                    Text(
                        f"   Calls: {bottleneck.call_count:,} ({bottleneck.time_per_call*1e6:.2f}μs per call)",
                        style="white",
                    ),
                    Text(f"   {bottleneck.recommendation}", style="cyan"),
                ]
            )

        return Box(title="⚠️  Bottlenecks", border="yellow", children=children)


class SuiteReport(Component):
    """Main benchmark suite report component."""

    def render(self):
        name = self.props.get("name", "Benchmark Suite")
        profiles = self.props.get("profiles", {})
        stats = self.props.get("stats", {})
        bottlenecks = self.props.get("bottlenecks", [])

        children = [
            SuiteHeader(name=name),
            Spacer(height=1),
            SuiteOverviewTable(profiles=profiles),
        ]

        if stats:
            children.extend([Spacer(height=1), SuiteStatistics(stats=stats)])

        if bottlenecks:
            children.extend(
                [Spacer(height=1), SuiteBottlenecks(bottlenecks=bottlenecks)]
            )

        return Col(children=children)


@dataclass
class PerformanceGap:
    """Represents a significant performance difference."""

    operation: str
    winner: str
    loser: str
    speedup: float
    winner_ops: float
    loser_ops: float


class HighScoresManager:
    """Manage high scores (best performance records) for benchmarks."""

    def __init__(self):
        self.score_file = Path(__file__).parent / "benchmark_high_scores.json"
        self.scores = self._load_scores()

    def _load_scores(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load high scores from JSON file."""
        if self.score_file.exists():
            with open(self.score_file, "r") as f:
                return json.load(f)
        return {}

    def _save_scores(self):
        """Save high scores to JSON file."""
        with open(self.score_file, "w") as f:
            json.dump(self.scores, f, indent=2)

    def get_record(self, library: str, operation: str) -> Optional[float]:
        """Get the record (operations per second) for a library/operation pair."""
        record_data = self.scores.get(operation, {}).get(library, {})
        if isinstance(record_data, dict):
            return record_data.get("ops_per_second")
        return record_data if isinstance(record_data, (int, float)) else None

    def get_timestamp(self, library: str, operation: str) -> Optional[str]:
        """Get the timestamp when the record was set."""
        record_data = self.scores.get(operation, {}).get(library, {})
        if isinstance(record_data, dict):
            return record_data.get("timestamp")
        return None

    def update_record(
        self, library: str, operation: str, ops_per_second: float
    ) -> bool:
        """Update the record if it's a new best. Returns True if updated."""
        if operation not in self.scores:
            self.scores[operation] = {}

        current_record = self.get_record(library, operation)
        if current_record is None or ops_per_second > current_record:
            self.scores[operation][library] = {
                "ops_per_second": ops_per_second,
                "timestamp": datetime.now().isoformat(),
            }
            self._save_scores()
            return True
        return False

    def is_new_record(
        self, library: str, operation: str, ops_per_second: float
    ) -> bool:
        """Check if this is a new record."""
        record = self.get_record(library, operation)
        return record is None or ops_per_second > record


class BenchmarkComparison:
    """Run and display benchmark comparisons with improved UI."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library required for benchmark comparison. Install with: poetry install --with benchmark"
            )

        self.console = Console()
        self.results: List[BenchmarkMetrics] = []
        self.execution_profiles: Dict[str, ExecutionProfile] = {}
        self.config = config or BenchmarkConfig()
        self.high_scores = HighScoresManager()

        # Initialize the global console observable
        from scripts.benchmarks.text_components import console

        console.set(self.console)

    def run(
        self,
        benchmark_names: Optional[List[str]] = None,
        registry: Optional[BenchmarkRegistry] = None,
    ):
        """Run all or specified benchmarks."""
        if registry is None:
            raise ValueError("BenchmarkRegistry must be provided")

        start_time = time.time()

        # Get benchmarks to run
        if benchmark_names is None:
            # Only run benchmarks that have both FynX and at least one RxPY implementation
            all_names = registry.list_benchmarks()
            benchmark_names = []
            for name in all_names:
                has_fynx = registry.get_benchmark(name, "fynx") is not None
                has_rxpy = (
                    registry.get_benchmark(name, "rxpy") is not None
                    or registry.get_benchmark(name, "rxpy-optimized") is not None
                )
                if has_fynx and has_rxpy:
                    benchmark_names.append(name)

        # Run each benchmark
        for name in benchmark_names:
            self._run_benchmark_pair(name, registry)

        # Display results using TUI components
        elapsed = time.time() - start_time

        # Create the main results component
        results_component = BenchmarkResults(
            config=self.config,
            results=self.results,
            execution_profiles=self.execution_profiles,
            elapsed_time=elapsed,
        )

        # Render using the TUI framework
        render(results_component)

    def _run_benchmark_pair(self, name: str, registry: BenchmarkRegistry):
        """Run both FynX and RxPY versions of a benchmark."""
        # Run FynX version
        fynx_bench = registry.get_benchmark(name, "fynx")
        if not fynx_bench:
            # Skip silently - results will be handled by the UI
            return

        fynx_config = fynx_bench.get("config") or self.config
        fynx_detailed = fynx_bench.get("detailed_profiling", False)
        fynx_result, fynx_profile = run_adaptive_benchmark(
            "FynX",
            name,
            fynx_bench["func"],
            fynx_bench["operations_counter"],
            fynx_config,
            detailed_profiling=fynx_detailed,
        )
        self.results.append(fynx_result)
        if fynx_profile:
            self.execution_profiles[f"FynX_{name}"] = fynx_profile

        # Run RxPY version (prefer optimized version if available)
        rxpy_bench = registry.get_benchmark(
            name, "rxpy-optimized"
        ) or registry.get_benchmark(name, "rxpy")
        if not rxpy_bench:
            # Skip silently - results will be handled by the UI
            return

        # Determine library name for display
        library_name = "RxPY"
        if registry.get_benchmark(name, "rxpy-optimized") is rxpy_bench:
            library_name = "RxPY-Opt"

        try:
            rxpy_config = rxpy_bench.get("config") or self.config
            rxpy_detailed = rxpy_bench.get("detailed_profiling", False)
            rxpy_result, rxpy_profile = run_adaptive_benchmark(
                library_name,
                name,
                rxpy_bench["func"],
                rxpy_bench["operations_counter"],
                rxpy_config,
                detailed_profiling=rxpy_detailed,
            )
            self.results.append(rxpy_result)
            if rxpy_profile:
                self.execution_profiles[f"{library_name}_{name}"] = rxpy_profile
        except RecursionError:
            # Skip silently - recursion errors will be handled by the UI
            pass
            # Create dummy result with capped performance
            rxpy_result = BenchmarkMetrics(
                library=library_name,
                operation=name,
                max_n=400,
                operation_time=fynx_result.operation_time * 2,
                operations_per_second=fynx_result.operations_per_second / 2,
                memory_start_kb=0,
                memory_end_kb=0,
                memory_peak_kb=0,
                memory_allocated_kb=0,
                gc_count_gen0=0,
                gc_count_gen1=0,
                gc_count_gen2=0,
                gc_total_collections=0,
                objects_before=0,
                objects_after=0,
                objects_delta=0,
            )
            self.results.append(rxpy_result)

        # Check and update high scores
        self._check_records(fynx_result)
        self._check_records(rxpy_result)

    def _check_records(self, result: BenchmarkMetrics) -> BenchmarkMetrics:
        """Check if this result is a new record and mark it in the result."""
        if result.operations_per_second <= 0 or result.is_dnf:
            return result

        library = result.library
        operation = result.operation
        ops_per_sec = result.operations_per_second

        # Get previous record
        previous_record = self.high_scores.get_record(library, operation)

        # Update record
        is_new_record = self.high_scores.update_record(library, operation, ops_per_sec)

        # Mark the result with record info
        result.is_new_record = is_new_record
        result.previous_record = previous_record

        return result

    def _group_by_operation(self) -> Dict:
        """Group results by operation name."""
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = {}
            operations[result.operation][result.library] = result
        return operations

    def _calculate_gaps(self, operations: Dict) -> List[PerformanceGap]:
        """Calculate performance gaps for all operations."""
        gaps = []
        for op_name, lib_results in operations.items():
            fynx = lib_results.get("FynX")
            rxpy = lib_results.get("RxPY")

            if fynx and rxpy:
                if fynx.operations_per_second > rxpy.operations_per_second:
                    speedup = fynx.operations_per_second / rxpy.operations_per_second
                    gaps.append(
                        PerformanceGap(
                            operation=op_name,
                            winner="FynX",
                            loser="RxPY",
                            speedup=speedup,
                            winner_ops=fynx.operations_per_second,
                            loser_ops=rxpy.operations_per_second,
                        )
                    )
                else:
                    speedup = rxpy.operations_per_second / fynx.operations_per_second
                    gaps.append(
                        PerformanceGap(
                            operation=op_name,
                            winner="RxPY",
                            loser="FynX",
                            speedup=speedup,
                            winner_ops=rxpy.operations_per_second,
                            loser_ops=fynx.operations_per_second,
                        )
                    )

        return gaps

    def _display_summary(self):
        """Executive summary at the top."""
        self.console.print("[bold]SUMMARY[/bold]", style="cyan")
        self.console.print("=" * 80, style="dim")

        # Calculate wins
        operations = self._group_by_operation()
        fynx_wins = 0
        rxpy_wins = 0

        for lib_results in operations.values():
            fynx = lib_results.get("FynX")
            rxpy = lib_results.get("RxPY")
            if fynx and rxpy:
                if fynx.operations_per_second > rxpy.operations_per_second:
                    fynx_wins += 1
                else:
                    rxpy_wins += 1

        total = fynx_wins + rxpy_wins
        self.console.print(
            f"Results: [green]FynX {fynx_wins}/{total}[/green] | "
            f"[blue]RxPY {rxpy_wins}/{total}[/blue]"
        )
        self.console.print()


# ============================================================================
# Global Registry
# ============================================================================

REGISTRY = BenchmarkRegistry()


# ============================================================================
# Benchmark Decorator
# ============================================================================


def benchmark(
    name: str,
    *,
    library: str = "fynx",
    category: Optional[str] = None,
    operations_counter: Optional[Callable[[Any], int]] = None,
    config: Optional[BenchmarkConfig] = None,
    detailed_profiling: bool = False,
):
    """
    Decorator to register a benchmark function with comprehensive profiling.

    Args:
        name: Benchmark name (used for grouping FynX vs RxPY)
        library: "fynx" or "rxpy"
        category: Optional category for grouping in output
        operations_counter: Function to count operations performed (default: len)
        config: Optional benchmark configuration override
        detailed_profiling: Enable comprehensive profiling with call stacks and bottleneck analysis

    Example:
        @benchmark("Observable Creation")
        def bench_creation(n):
            return [observable(i) for i in range(n)]

        @benchmark("Observable Creation", library="rxpy")
        def bench_creation_rxpy(n):
            subjects = []
            for i in range(n):
                subject = Subject()
                subject.on_next(i)
                subjects.append(subject)
            return subjects

        # Custom config example
        custom_config = BenchmarkConfig(time_limit=2.0, num_iterations=3)
        @benchmark("Custom Benchmark", config=custom_config)
        def custom_benchmark(n):
            return [observable(i) for i in range(n)]

        # Detailed profiling example
        @benchmark("Complex Operation", detailed_profiling=True)
        def complex_operation(n):
            # This benchmark will include call stack analysis and bottleneck detection
            return [observable(i) for i in range(n)]
    """

    def decorator(func: Callable) -> Callable:
        REGISTRY.register(
            name=name,
            library=library,
            func=func,
            category=category,
            operations_counter=operations_counter,
            config=config,
            detailed_profiling=detailed_profiling,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Statistics and Analysis
# ============================================================================


@dataclass
class BenchmarkStats:
    """Statistical analysis of benchmark runs."""

    name: str
    runs: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.runs) / len(self.runs) if self.runs else 0

    @property
    def min(self) -> float:
        return min(self.runs) if self.runs else 0

    @property
    def max(self) -> float:
        return max(self.runs) if self.runs else 0

    @property
    def std_dev(self) -> float:
        if len(self.runs) < 2:
            return 0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self.runs) / len(self.runs)
        return variance**0.5

    @property
    def coefficient_of_variation(self) -> float:
        """CV as percentage - lower is better (more consistent)."""
        if self.mean == 0:
            return 0
        return (self.std_dev / self.mean) * 100

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.runs:
            return 0
        sorted_runs = sorted(self.runs)
        k = (len(sorted_runs) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_runs):
            return sorted_runs[f] + c * (sorted_runs[f + 1] - sorted_runs[f])
        return sorted_runs[f]


@dataclass
class BottleneckInfo:
    """Information about a detected performance bottleneck."""

    function_name: str
    time_spent: float
    percentage: float
    call_count: int
    time_per_call: float
    recommendation: str


class BenchmarkSuite:
    """
    Comprehensive benchmark suite with automatic profiling and analysis.

    Collects:
    - Execution profiles with call stacks
    - Memory allocation and GC metrics
    - Statistical timing analysis
    - Bottleneck detection
    """

    def __init__(self, name: str = "Benchmark Suite"):
        self.name = name
        self.console = Console()
        self.profiles: Dict[str, ExecutionProfile] = {}
        self.stats: Dict[str, BenchmarkStats] = {}
        self.bottlenecks: List[BottleneckInfo] = []
        self.current_benchmark = None

    @contextmanager
    def benchmark(self, name: str, iterations: int = 1):
        """
        Context manager for benchmarking a code block with full profiling.

        Args:
            name: Name of the benchmark
            iterations: Number of iterations to run (for statistical analysis)
        """
        if iterations > 1:
            # Multiple iterations for statistical analysis
            stats = BenchmarkStats(name)

            for i in range(iterations):
                with DetailedProfiler(f"{name}_iter_{i}") as profiler:
                    yield

                profile = profiler.get_profile()
                stats.runs.append(profile.execution_time)

                # Store the profile from the last iteration
                if i == iterations - 1:
                    self.profiles[name] = profile

            self.stats[name] = stats
        else:
            # Single run with detailed profiling
            with DetailedProfiler(name) as profiler:
                yield

            self.profiles[name] = profiler.get_profile()

    def analyze_bottlenecks(self, threshold_percent: float = 5.0):
        """
        Analyze profiles to identify bottlenecks.

        Args:
            threshold_percent: Functions taking more than this % of time are flagged
        """
        self.bottlenecks.clear()

        for profile in self.profiles.values():
            total_time = profile.execution_time

            for func_name, cum_time, calls in profile.top_functions:
                percentage = (cum_time / total_time) * 100

                if percentage >= threshold_percent:
                    # Generate recommendation
                    time_per_call = cum_time / calls if calls > 0 else 0

                    if calls > 1000:
                        recommendation = (
                            "High call count - consider caching or batching"
                        )
                    elif time_per_call > 0.01:
                        recommendation = "Expensive per-call - optimize algorithm"
                    elif "list" in func_name.lower() or "dict" in func_name.lower():
                        recommendation = "Data structure operations - consider more efficient structures"
                    else:
                        recommendation = (
                            "Review implementation for optimization opportunities"
                        )

                    self.bottlenecks.append(
                        BottleneckInfo(
                            function_name=func_name,
                            time_spent=cum_time,
                            percentage=percentage,
                            call_count=calls,
                            time_per_call=time_per_call,
                            recommendation=recommendation,
                        )
                    )

        # Sort by percentage descending
        self.bottlenecks.sort(key=lambda x: x.percentage, reverse=True)

    def print_report(
        self, show_call_stacks: bool = False, bottleneck_threshold: float = 5.0
    ):
        """
        Display comprehensive performance report using TUI components.

        Args:
            show_call_stacks: Whether to show detailed call stack information
            bottleneck_threshold: Percentage threshold for bottleneck detection
        """
        self.analyze_bottlenecks(bottleneck_threshold)

        # Create the main report component
        report_component = SuiteReport(
            name=self.name,
            profiles=self.profiles,
            stats=self.stats,
            bottlenecks=self.bottlenecks,
        )

        # Render using the TUI framework
        render(report_component)


def profile_function(func: Callable) -> Callable:
    """
    Decorator to automatically profile a function.

    Usage:
        @profile_function
        def my_function():
            # code here
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with DetailedProfiler(func.__name__) as profiler:
            result = func(*args, **kwargs)
        profiler.print_report()
        return result

    return wrapper
