#!/usr/bin/env python3
"""
Benchmark Harness - FynX-TUI Integration

Benchmark utilities with reactive terminal UI for performance analysis.
"""

import cProfile
import gc
import json
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

# Import FynX-TUI components
from fynx_tui.tui import (
    H1,
    Box,
    Col,
    Line,
    PerformanceStats,
    ReactiveComponent,
    Row,
    Spacer,
    Tag,
    Text,
    render,
)

from fynx import Store, observable

T = TypeVar("T")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Benchmark configuration parameters."""

    time_limit: float = 1.0
    starting_n: int = 10
    scale_factor: float = 1.5
    num_iterations: int = 1
    profile_enabled: bool = False


CONFIG = BenchmarkConfig()


# ============================================================================
# Reactive Stores
# ============================================================================


class BenchmarkStore(Store):
    """Centralized benchmark state management"""

    # Config
    time_limit = observable(0.0)
    num_iterations = observable(0)
    total_tests = observable(0)

    # Current state
    current_operation = observable("")
    current_library = observable("")
    current_phase = observable("idle")

    # Results
    results = observable([])
    completed_operations = observable(0)

    # Statistics
    fynx_wins = observable(0)
    rxpy_wins = observable(0)
    ties = observable(0)

    # Metrics
    avg_fynx_speedup = observable(0.0)
    avg_rxpy_speedup = observable(0.0)
    max_speedup = observable(0.0)

    # Timing
    elapsed_time = observable(0.0)
    start_time = observable(0.0)

    # Profiling - track slowest operations
    slowest_operations = observable([])  # List of (operation, time) tuples


class TierStore(Store):
    """Performance tier tracking"""

    extreme = observable([])  # 10x+
    excellent = observable([])  # 5-10x
    good = observable([])  # 2-5x
    moderate = observable([])  # 1-2x
    poor = observable([])  # <1x


# ============================================================================
# Metrics
# ============================================================================


@dataclass
class BenchmarkMetrics:
    """Benchmark execution metrics."""

    library: str
    operation: str
    max_n: int
    operation_time: float
    operations_per_second: float

    # Memory
    memory_start_kb: int
    memory_end_kb: int
    memory_peak_kb: int
    memory_allocated_kb: int

    # GC
    gc_count_gen0: int
    gc_count_gen1: int
    gc_count_gen2: int
    gc_total_collections: int

    # Objects
    objects_before: int
    objects_after: int
    objects_delta: int

    # Status
    is_dnf: bool = False
    partial_result: bool = False
    is_new_record: bool = False
    previous_record: Optional[float] = None


class BenchmarkProfiler:
    """Performance profiling context manager."""

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
        gc.collect()
        gc.collect()
        gc.collect()

        if self.enable_profiler:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        tracemalloc.start()
        self.memory_start, _ = tracemalloc.get_traced_memory()

        self.gc_stats_before = gc.get_count()
        self.objects_before = len(gc.get_objects())

        self.start_time = time.perf_counter()

        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

        if self.profiler:
            self.profiler.disable()

        self.gc_stats_after = gc.get_count()
        self.objects_after = len(gc.get_objects())

        self.memory_end, self.memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def get_elapsed_time(self) -> float:
        return self.end_time - self.start_time

    def get_metrics(
        self, library: str, operation: str, n: int, operations_performed: int
    ) -> BenchmarkMetrics:
        elapsed = self.get_elapsed_time()
        ops_per_sec = operations_performed / elapsed if elapsed > 0 else 0

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
# Registry
# ============================================================================


class BenchmarkRegistry:
    """Benchmark function registry."""

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
        return self.benchmarks.get(name, {}).get(library)

    def list_benchmarks(self) -> List[str]:
        return list(self.benchmarks.keys())

    def get_category(self, name: str) -> str:
        return self.categories.get(name, "General")


# ============================================================================
# Adaptive Benchmark Runner
# ============================================================================


def run_adaptive_benchmark(
    library: str,
    operation: str,
    operation_func: Callable[[int], T],
    operations_counter: Callable[[T], int],
    config: BenchmarkConfig,
    detailed_profiling: bool = False,
) -> Tuple[BenchmarkMetrics, Optional["ExecutionProfile"]]:
    """
    Adaptive benchmark runner that scales workload to target time.
    """
    n = config.starting_n
    total_time_limit = config.time_limit * config.num_iterations

    # Update state
    BenchmarkStore.current_library = library
    BenchmarkStore.current_operation = operation
    BenchmarkStore.current_phase = "scaling"

    # Find optimal workload
    scaling_iterations = 0
    max_scaling_iterations = 20

    while scaling_iterations < max_scaling_iterations:
        start = time.perf_counter()
        result = operation_func(n)
        elapsed = time.perf_counter() - start
        ops_performed = operations_counter(result)

        estimated_total_time = elapsed * config.num_iterations

        if estimated_total_time >= total_time_limit * 0.8:
            break

        if elapsed >= total_time_limit and ops_performed == 0:
            BenchmarkStore.current_phase = "failed"
            return (
                BenchmarkMetrics(
                    library=library,
                    operation=operation,
                    max_n=n,
                    operation_time=float("inf"),
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
        if n > 10_000_000:
            break

        scaling_iterations += 1

    # Run benchmark
    BenchmarkStore.current_phase = "running"
    all_metrics = []

    for iteration in range(config.num_iterations):
        with BenchmarkProfiler(config.profile_enabled) as profiler:
            result = operation_func(n)
            ops_performed = operations_counter(result)

        metrics = profiler.get_metrics(library, operation, n, ops_performed)
        all_metrics.append(metrics)

    BenchmarkStore.current_phase = "complete"
    return average_metrics(all_metrics), None


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
# UI Components - Simplified and Grounded
# ============================================================================


class StatusHeader(ReactiveComponent):
    """Current benchmark status with test statistics"""

    def get_dependencies(self):
        return [
            BenchmarkStore.current_operation,
            BenchmarkStore.current_library,
            BenchmarkStore.current_phase,
            BenchmarkStore.completed_operations,
            BenchmarkStore.total_tests,
        ]

    def render_component(self):
        # Access values directly - the caching mechanism handles change detection
        operation = BenchmarkStore.current_operation.value
        library = BenchmarkStore.current_library.value
        phase = BenchmarkStore.current_phase.value
        done = BenchmarkStore.completed_operations.value
        total = BenchmarkStore.total_tests.value

        if not operation:
            return Text(text="Initializing...", color="dim")

        phase_text = {
            "idle": "Ready",
            "scaling": "Calibrating",
            "running": "Testing",
            "complete": "Done",
            "failed": "Error",
        }.get(phase, phase)

        return Box(
            title="Current Test",
            border="blue",
            children=[
                Text(text=f"{operation}", color="white", bold=True),
                Text(text=f"{library} - {phase_text}", color="cyan"),
                Spacer(height=1),
                Text(text=f"Progress: {done}/{total} tests", color="dim"),
            ],
        )


class ResultsSummary(ReactiveComponent):
    """Summary statistics"""

    def get_dependencies(self):
        return [
            BenchmarkStore.fynx_wins,
            BenchmarkStore.rxpy_wins,
            BenchmarkStore.elapsed_time,
        ]

    def render_component(self):
        fynx = BenchmarkStore.fynx_wins.value
        rxpy = BenchmarkStore.rxpy_wins.value
        elapsed = BenchmarkStore.elapsed_time.value

        return Box(
            title="Results Summary",
            border="green",
            children=[
                Text(text=f"FynX: {fynx} wins", color="green"),
                Text(text=f"RxPY: {rxpy} wins", color="blue"),
                Spacer(height=1),
                Text(text=f"Elapsed: {elapsed:.1f}s", color="dim"),
            ],
        )


class ResultsTable(ReactiveComponent):
    """Detailed results table with proper column alignment"""

    def get_dependencies(self):
        return [BenchmarkStore.results]

    def render_component(self):
        results = BenchmarkStore.results.value

        if not results:
            return Text(text="No results yet", color="dim")

        # Group by operation
        operations = {}
        for r in results:
            if r.operation not in operations:
                operations[r.operation] = {}
            operations[r.operation][r.library] = r

        # Build table rows
        rows = []

        # Header
        rows.append(
            Text(
                text=f"{'Operation':<35} {'FynX ops/s':>15} {'RxPY ops/s':>15} {'Ratio':>10} {'Result':>10}",
                color="cyan",
                bold=True,
            )
        )
        rows.append(Line(char="─", width=100, style="dim"))

        # Data rows
        for op_name, libs in operations.items():
            fynx = libs.get("FynX")
            rxpy = libs.get("RxPY") or libs.get("RxPY-Opt")

            if not fynx or not rxpy:
                continue

            fynx_ops = fynx.operations_per_second
            rxpy_ops = rxpy.operations_per_second

            if fynx_ops > rxpy_ops and rxpy_ops > 0:
                ratio = fynx_ops / rxpy_ops
                result = "FynX"
                color = "green"
            elif rxpy_ops > fynx_ops and fynx_ops > 0:
                ratio = rxpy_ops / fynx_ops
                result = "RxPY"
                color = "blue"
            else:
                ratio = 1.0
                result = "Tie"
                color = "yellow"

            # Format numbers
            op_display = op_name[:35]
            fynx_display = f"{fynx_ops:,.0f}" if fynx_ops > 0 else "DNF"
            rxpy_display = f"{rxpy_ops:,.0f}" if rxpy_ops > 0 else "DNF"
            ratio_display = f"{ratio:.2f}x"

            rows.append(
                Text(
                    text=f"{op_display:<35} {fynx_display:>15} {rxpy_display:>15} {ratio_display:>10} {result:>10}",
                    color=color,
                )
            )

        return Box(title="Detailed Results", border="cyan", children=rows)


class ProfileLogs(ReactiveComponent):
    """Show slowest operations as they complete"""

    def get_dependencies(self):
        return [BenchmarkStore.slowest_operations]

    def render_component(self):
        slowest = BenchmarkStore.slowest_operations.value

        if not slowest:
            return Box(
                title="Performance Profile",
                border="yellow",
                children=[Text(text="No profile data yet", color="dim")],
            )

        # Sort by time descending and take top 10
        sorted_ops = sorted(slowest, key=lambda x: x[1], reverse=True)[:10]

        rows = [
            Text(
                text=f"{'Operation':<40} {'Time':>12} {'Library':>10}",
                color="yellow",
                bold=True,
            ),
            Line(char="─", width=70, style="dim"),
        ]

        for op_name, op_time, lib_name in sorted_ops:
            time_ms = op_time * 1000
            color = "red" if time_ms > 100 else "yellow" if time_ms > 50 else "green"

            rows.append(
                Text(
                    text=f"{op_name[:40]:<40} {time_ms:>10.2f}ms {lib_name:>10}",
                    color=color,
                )
            )

        return Box(
            title="Performance Profile - Slowest Operations",
            border="yellow",
            children=rows,
        )


class BenchmarkUI(ReactiveComponent):
    """Main benchmark UI layout"""

    def get_dependencies(self):
        return [BenchmarkStore.current_operation]

    def render_component(self):
        return Col(
            children=[
                Text(text="FynX Benchmark Suite", color="cyan", bold=True),
                Spacer(height=1),
                Row(
                    equal=True,
                    children=[
                        StatusHeader(),
                        ResultsSummary(),
                    ],
                ),
                Spacer(height=1),
                ProfileLogs(),
                Spacer(height=1),
                ResultsTable(),
            ]
        )


# ============================================================================
# High Scores
# ============================================================================


class HighScoresManager:
    """Track best performance results."""

    def __init__(self):
        self.score_file = Path(__file__).parent / "benchmark_high_scores.json"
        self.scores = self._load_scores()

    def _load_scores(self) -> Dict:
        if self.score_file.exists():
            try:
                with open(self.score_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_scores(self):
        try:
            with open(self.score_file, "w") as f:
                json.dump(self.scores, f, indent=2)
        except:
            pass

    def get_record(self, library: str, operation: str) -> Optional[float]:
        record_data = self.scores.get(operation, {}).get(library, {})
        if isinstance(record_data, dict):
            return record_data.get("ops_per_second")
        return record_data if isinstance(record_data, (int, float)) else None

    def update_record(
        self, library: str, operation: str, ops_per_second: float
    ) -> bool:
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


# ============================================================================
# Benchmark Runner
# ============================================================================


class BenchmarkComparison:
    """Benchmark comparison runner with live UI."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.high_scores = HighScoresManager()

    def run(
        self,
        benchmark_names: Optional[List[str]] = None,
        registry: Optional[BenchmarkRegistry] = None,
        fps: int = 30,
    ):
        """Run benchmarks with live UI."""
        if registry is None:
            raise ValueError("Registry required")

        # Initialize stores
        BenchmarkStore.time_limit = self.config.time_limit
        BenchmarkStore.num_iterations = self.config.num_iterations
        BenchmarkStore.results = []
        BenchmarkStore.completed_operations = 0
        BenchmarkStore.fynx_wins = 0
        BenchmarkStore.rxpy_wins = 0
        BenchmarkStore.start_time = time.time()
        BenchmarkStore.slowest_operations = []

        # Get benchmarks
        if benchmark_names is None:
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

        BenchmarkStore.total_tests = len(benchmark_names)

        # Start UI
        ui = BenchmarkUI()
        app = render(ui, fps=fps)

        with app.start():
            for name in benchmark_names:
                self._run_benchmark_pair(name, registry)

                BenchmarkStore.elapsed_time = (
                    time.time() - BenchmarkStore.start_time.value
                )
                time.sleep(0.05)

            BenchmarkStore.current_phase = "complete"
            BenchmarkStore.current_operation = "Complete"

            time.sleep(2)

    def _run_benchmark_pair(self, name: str, registry: BenchmarkRegistry):
        """Run FynX and RxPY versions."""
        # FynX
        fynx_bench = registry.get_benchmark(name, "fynx")
        if not fynx_bench:
            return

        BenchmarkStore.current_operation = name
        BenchmarkStore.current_library = "FynX"

        fynx_config = fynx_bench.get("config") or self.config
        fynx_result, _ = run_adaptive_benchmark(
            "FynX",
            name,
            fynx_bench["func"],
            fynx_bench["operations_counter"],
            fynx_config,
        )

        # RxPY
        rxpy_bench = registry.get_benchmark(
            name, "rxpy-optimized"
        ) or registry.get_benchmark(name, "rxpy")
        if not rxpy_bench:
            return

        BenchmarkStore.current_library = "RxPY"

        library_name = (
            "RxPY-Opt"
            if registry.get_benchmark(name, "rxpy-optimized") is rxpy_bench
            else "RxPY"
        )

        try:
            rxpy_config = rxpy_bench.get("config") or self.config
            rxpy_result, _ = run_adaptive_benchmark(
                library_name,
                name,
                rxpy_bench["func"],
                rxpy_bench["operations_counter"],
                rxpy_config,
            )
        except RecursionError:
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

        # Update stores
        self._update_results(fynx_result, rxpy_result)

    def _update_results(self, fynx: BenchmarkMetrics, rxpy: BenchmarkMetrics):
        """Update reactive stores with results."""
        current = BenchmarkStore.results.value.copy()
        current.append(fynx)
        current.append(rxpy)
        BenchmarkStore.results = current

        BenchmarkStore.completed_operations = (
            BenchmarkStore.completed_operations.value + 1
        )

        # Update wins
        if fynx.operations_per_second > rxpy.operations_per_second:
            BenchmarkStore.fynx_wins = BenchmarkStore.fynx_wins.value + 1
        elif rxpy.operations_per_second > fynx.operations_per_second:
            BenchmarkStore.rxpy_wins = BenchmarkStore.rxpy_wins.value + 1

        # Update slowest operations log
        slowest = BenchmarkStore.slowest_operations.value.copy()
        slowest.append((fynx.operation, fynx.operation_time, fynx.library))
        slowest.append((rxpy.operation, rxpy.operation_time, rxpy.library))
        BenchmarkStore.slowest_operations = slowest


# ============================================================================
# Global Registry
# ============================================================================

REGISTRY = BenchmarkRegistry()


# ============================================================================
# Decorator
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
    """Register a benchmark function."""

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
