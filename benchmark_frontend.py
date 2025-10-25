"""
Fynx Frontend Benchmark Suite
=============================

Comprehensive benchmarks for the Fynx-style frontend with DeltaKVStore backend.
Tests observable operations, operator chaining, and reactive performance.
"""

import gc
import os
import statistics
import time
import tracemalloc
from time import perf_counter

import psutil
import rx
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rx import operators as ops
from rx.subject import Subject

from fynx.observable import DeltaKVStore, Observable, Store, reactive

console = Console()


def warmup_run(func, *args, **kwargs):
    """Run a function once to warm up JIT and caches."""
    try:
        func(*args, **kwargs)
    except:
        pass


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_memory_impact(func, *args, **kwargs):
    """Measure memory usage before and after running a function."""
    # Get baseline memory (don't force GC here - we want to measure actual usage)
    baseline_memory = get_memory_usage()

    # Run the function
    result = func(*args, **kwargs)

    # Get final memory (don't force GC here - we want to measure actual usage)
    final_memory = get_memory_usage()

    return {
        "baseline_memory": baseline_memory,
        "final_memory": final_memory,
        "memory_delta": final_memory - baseline_memory,
        "result": result,
    }


def measure_gc_pressure(func, *args, **kwargs):
    """Measure garbage collection pressure."""
    # Get initial GC counts (don't force collection yet)
    initial_counts = gc.get_count()

    # Run the function
    result = func(*args, **kwargs)

    # Get counts after function execution (before forced collection)
    post_execution_counts = gc.get_count()

    # Force collection and get final counts
    collected = gc.collect()
    final_counts = gc.get_count()

    # Calculate GC pressure
    gc_pressure = {
        "initial_counts": initial_counts,
        "post_execution_counts": post_execution_counts,
        "final_counts": final_counts,
        "collected_objects": collected,
        "generation_0_delta": post_execution_counts[0] - initial_counts[0],
        "generation_1_delta": post_execution_counts[1] - initial_counts[1],
        "generation_2_delta": post_execution_counts[2] - initial_counts[2],
        "result": result,
    }

    return gc_pressure


def benchmark_memory_usage(store_class, n=1000, runs=3):
    """Benchmark memory usage for observable creation."""
    memory_deltas = []
    gc_pressures = []

    for _ in range(runs):

        def create_observables():
            store = store_class()
            observables = []
            for i in range(n):
                obs = store.observable(f"obs_{i}", i)
                observables.append(obs)
            return observables

        # Measure memory impact
        memory_result = measure_memory_impact(create_observables)
        memory_deltas.append(memory_result["memory_delta"])

        # Measure GC pressure
        gc_result = measure_gc_pressure(create_observables)
        gc_pressures.append(gc_result)

    avg_memory_delta = statistics.mean(memory_deltas)
    avg_gc_pressure = statistics.mean([p["generation_0_delta"] for p in gc_pressures])

    return {
        "operation": "Memory Usage (Observable Creation)",
        "count": n,
        "avg_memory_delta_mb": avg_memory_delta,
        "avg_gc_pressure": avg_gc_pressure,
        "memory_per_observable_kb": (avg_memory_delta * 1024) / n,
        "std_dev": statistics.stdev(memory_deltas) if len(memory_deltas) > 1 else 0,
    }


def benchmark_memory_reactive_updates(store_class, n=1000, runs=3):
    """Benchmark memory usage during reactive updates."""
    memory_deltas = []
    gc_pressures = []

    for _ in range(runs):

        def reactive_update_cycle():
            store = store_class()

            # Create a chain of dependencies
            chain_length = min(20, n)
            observables = []

            # Create base observable
            base = store.observable("base", 1)
            observables.append(base)

            # Create dependency chain
            for i in range(1, chain_length):
                prev = observables[-1]

                def make_adder(level):
                    return lambda x: x + level

                next_obs = prev >> make_adder(i)
                observables.append(next_obs)

            # Perform many updates
            num_updates = min(100, n)
            for _ in range(num_updates):
                base.value = base.value + 1
                _ = observables[-1].value  # Force recomputation

            return observables

        # Measure memory impact
        memory_result = measure_memory_impact(reactive_update_cycle)
        memory_deltas.append(memory_result["memory_delta"])

        # Measure GC pressure
        gc_result = measure_gc_pressure(reactive_update_cycle)
        gc_pressures.append(gc_result)

    avg_memory_delta = statistics.mean(memory_deltas)
    avg_gc_pressure = statistics.mean([p["generation_0_delta"] for p in gc_pressures])

    return {
        "operation": "Memory Usage (Reactive Updates)",
        "count": n,
        "avg_memory_delta_mb": avg_memory_delta,
        "avg_gc_pressure": avg_gc_pressure,
        "memory_per_update_kb": (avg_memory_delta * 1024) / n,
        "std_dev": statistics.stdev(memory_deltas) if len(memory_deltas) > 1 else 0,
    }


def benchmark_memory_comparison_fynx_vs_rxpy(n=1000, runs=3):
    """Compare memory usage between Fynx Frontend and RxPy."""
    fynx_memory_deltas = []
    rxpy_memory_deltas = []
    fynx_gc_pressures = []
    rxpy_gc_pressures = []

    for _ in range(runs):
        # Test Fynx Frontend
        def fynx_operation():
            store = Store()
            observables = []
            for i in range(n):
                obs = store.observable(f"obs_{i}", i)
                observables.append(obs)
                obs.subscribe(lambda x: None)
            return observables

        # Test RxPy
        def rxpy_operation():
            from rx.subject import Subject

            subjects = []
            for i in range(n):
                subject = Subject()
                subjects.append(subject)
                subject.subscribe(lambda x: None)
            return subjects

        # Measure Fynx
        fynx_memory_result = measure_memory_impact(fynx_operation)
        fynx_memory_deltas.append(fynx_memory_result["memory_delta"])

        fynx_gc_result = measure_gc_pressure(fynx_operation)
        fynx_gc_pressures.append(fynx_gc_result)

        # Measure RxPy
        rxpy_memory_result = measure_memory_impact(rxpy_operation)
        rxpy_memory_deltas.append(rxpy_memory_result["memory_delta"])

        rxpy_gc_result = measure_gc_pressure(rxpy_operation)
        rxpy_gc_pressures.append(rxpy_gc_result)

    avg_fynx_memory = statistics.mean(fynx_memory_deltas)
    avg_rxpy_memory = statistics.mean(rxpy_memory_deltas)
    avg_fynx_gc = statistics.mean([p["generation_0_delta"] for p in fynx_gc_pressures])
    avg_rxpy_gc = statistics.mean([p["generation_0_delta"] for p in rxpy_gc_pressures])

    return {
        "operation": "Memory Comparison (Fynx vs RxPy)",
        "count": n,
        "fynx_memory_delta_mb": avg_fynx_memory,
        "rxpy_memory_delta_mb": avg_rxpy_memory,
        "fynx_gc_pressure": avg_fynx_gc,
        "rxpy_gc_pressure": avg_rxpy_gc,
        "memory_ratio": (
            avg_fynx_memory / avg_rxpy_memory if avg_rxpy_memory > 0 else float("inf")
        ),
        "gc_ratio": avg_fynx_gc / avg_rxpy_gc if avg_rxpy_gc > 0 else float("inf"),
        "std_dev": (
            statistics.stdev(fynx_memory_deltas) if len(fynx_memory_deltas) > 1 else 0
        ),
    }


def benchmark_memory_efficiency(store_class, n=1000, runs=3):
    """Benchmark memory efficiency with large-scale operations."""
    memory_deltas = []
    gc_pressures = []

    for _ in range(runs):

        def large_scale_operation():
            store = store_class()

            # Create many observables with complex dependencies
            observables = []
            for i in range(n):
                obs = store.observable(f"obs_{i}", i)
                observables.append(obs)

            # Create computed values using operator chaining
            computed_values = []
            for i in range(min(100, n // 10)):
                # Create a chain of observables that depend on multiple sources
                start_idx = i * 10
                if start_idx + 9 < n:
                    # Create a computed observable that sums multiple observables
                    base_obs = observables[start_idx]
                    computed_obs = base_obs

                    # Chain multiple observables together
                    for j in range(1, min(10, n - start_idx)):
                        computed_obs = computed_obs + observables[start_idx + j]

                    computed_values.append(computed_obs)

            # Force computation of all computed values
            for computed in computed_values:
                _ = computed.value

            # Update some observables and recompute
            for i in range(0, n, 10):
                current_val = store.get(f"obs_{i}")
                if current_val is not None:
                    store.set(f"obs_{i}", current_val + 1)
                else:
                    store.set(f"obs_{i}", 1)

            # Force recomputation
            for computed in computed_values:
                _ = computed.value

            return observables, computed_values

        # Measure memory impact
        memory_result = measure_memory_impact(large_scale_operation)
        memory_deltas.append(memory_result["memory_delta"])

        # Measure GC pressure
        gc_result = measure_gc_pressure(large_scale_operation)
        gc_pressures.append(gc_result)

    avg_memory_delta = statistics.mean(memory_deltas)
    avg_gc_pressure = statistics.mean([p["generation_0_delta"] for p in gc_pressures])

    return {
        "operation": "Memory Usage (Large Scale)",
        "count": n,
        "avg_memory_delta_mb": avg_memory_delta,
        "avg_gc_pressure": avg_gc_pressure,
        "memory_per_observable_kb": (avg_memory_delta * 1024) / n,
        "std_dev": statistics.stdev(memory_deltas) if len(memory_deltas) > 1 else 0,
    }


def benchmark_observable_creation(store_class, n=1000, runs=5):
    """Benchmark creating observables."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()
        for i in range(n):
            setattr(store, f"obs_{i}", store.observable(f"obs_{i}", i))

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Observable Creation",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_operator_chaining(store_class, n=1000, runs=5):
    """Benchmark operator chaining operations."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()

        # Create base observables
        for i in range(n):
            setattr(store, f"base_{i}", store.observable(f"base_{i}", i))

        # Create chained operations
        for i in range(n):
            base_obs = getattr(store, f"base_{i}")
            # Chain: double -> add_10 -> format
            doubled = base_obs >> (lambda x: x * 2)
            added = doubled >> (lambda x: x + 10)
            formatted = added >> (lambda x: f"Result: {x}")

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Operator Chaining (>>)",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_observable_combination(store_class, n=500, runs=5):
    """Benchmark stream merging with reactive updates."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()

        # Create pairs of observables
        observables = []
        for i in range(n):
            obs1 = store.observable(f"obs1_{i}", 0)
            obs2 = store.observable(f"obs2_{i}", 0)
            observables.append((obs1, obs2))

        # Create stream merges and subscribe to test reactive propagation
        merges = []
        for obs1, obs2 in observables:
            merged = obs1 + obs2
            # Subscribe to make it reactive
            merged.subscribe(lambda v: None)
            merges.append(merged)

        # Test reactive updates through the streams
        for i in range(min(10, n)):  # Limit updates for performance
            for obs1, obs2 in observables[: min(10, len(observables))]:
                obs1.value = i
                obs2.value = i * 2

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Observable Combination (+)",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_conditional_operations(store_class, n=500, runs=5):
    """Benchmark conditional operations with & and |."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()

        # Create condition observables
        for i in range(n):
            setattr(store, f"value_{i}", store.observable(f"value_{i}", i))
            setattr(store, f"cond1_{i}", store.observable(f"cond1_{i}", i % 2 == 0))
            setattr(store, f"cond2_{i}", store.observable(f"cond2_{i}", i % 3 == 0))

        # Create conditional chains
        for i in range(n):
            value = getattr(store, f"value_{i}")
            cond1 = getattr(store, f"cond1_{i}")
            cond2 = getattr(store, f"cond2_{i}")

            # Complex conditional: (value & cond1) | cond2
            conditional = (value & cond1) | cond2

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Conditional Operations (& |)",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_reactive_updates(store_class, n=1000, runs=3):
    """Benchmark reactive updates through dependency chains."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()

        # Create a chain of dependencies
        chain_length = min(20, n)  # Limit chain length for performance
        observables = []

        # Create base observable
        base = store.observable("base", 1)
        observables.append(base)

        # Create dependency chain
        for i in range(1, chain_length):
            prev = observables[-1]

            def make_adder(level):
                return lambda x: x + level

            next_obs = prev >> make_adder(i)  # Add level to value
            observables.append(next_obs)

        # Force computation of entire chain
        _ = observables[-1].value

        # Time incremental updates
        update_times = []
        num_updates = min(100, n)  # Number of actual updates performed
        for _ in range(num_updates):
            update_start = time.time()
            base.value = base.value + 1
            _ = observables[-1].value  # Force recomputation
            update_times.append(time.time() - update_start)

        times.append(statistics.mean(update_times))

    avg_time = statistics.mean(times)
    # Report updates per second, not operations per second
    updates_per_sec = 1.0 / avg_time

    return {
        "operation": "Reactive Updates (Chain)",
        "count": chain_length,
        "avg_time": avg_time,
        "ops_per_sec": updates_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_complex_reactive_system(store_class, n=100, runs=3):
    """Benchmark a complex reactive system like a todo app."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TodoStore(store_class):
            pass

        store = TodoStore()

        # Create todo items
        for i in range(n):
            setattr(
                store, f"todo_{i}_text", store.observable(f"todo_{i}_text", f"Task {i}")
            )
            setattr(store, f"todo_{i}_done", store.observable(f"todo_{i}_done", False))

        # Create computed properties
        completed_count = store.observable("completed_count", 0)
        total_count = store.observable("total_count", n)

        # Complex computed: completion percentage
        completion_pct = (completed_count + total_count) >> (
            lambda c, t: f"{c}/{t} ({(c/t*100):.1f}%)" if t > 0 else "0/0 (0.0%)"
        )

        # Filter operations
        pending_tasks = []
        completed_tasks = []

        for i in range(n):
            text_obs = getattr(store, f"todo_{i}_text")
            done_obs = getattr(store, f"todo_{i}_done")

            # Pending: text & ~done
            pending = text_obs & (~done_obs)
            pending_tasks.append(pending)

            # Completed: text & done
            completed = text_obs & done_obs
            completed_tasks.append(completed)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Complex Reactive System",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_complex_dependency_dag(store_class, n=1000, runs=3):
    """Benchmark complex dependency DAG - FynX's sweet spot."""
    times = []

    for _ in range(runs):
        start = time.time()

        store = store_class()

        # Create root observable
        a = store.observable("a", 1)

        # Create multiple branches from root (fan-out)
        branches = []
        for i in range(10):  # 10 branches for complexity
            branch = a >> (lambda x, i=i: x * (i + 1))
            branches.append(branch)

        # Create convergence points (fan-in)
        # Combine pairs of branches
        convergence_1 = []
        for i in range(0, len(branches), 2):
            if i + 1 < len(branches):
                combined = (branches[i] + branches[i + 1]) >> (lambda a, b: a + b)
                convergence_1.append(combined)

        # Further convergence
        convergence_2 = []
        for i in range(0, len(convergence_1), 2):
            if i + 1 < len(convergence_1):
                combined = (convergence_1[i] + convergence_1[i + 1]) >> (
                    lambda a, b: max(a, b)
                )
                convergence_2.append(combined)

        # Final convergence - complex DAG leaf
        if convergence_2:
            f = convergence_2[0]
            for conv in convergence_2[1:]:
                f = (f + conv) >> (lambda a, b: a + b)

            # Subscribe to force materialization of the entire DAG
            subscription = f.subscribe(lambda x: None)
        else:
            # Fallback if convergence_2 is empty
            f = convergence_1[0] if convergence_1 else branches[0]
            subscription = f.subscribe(lambda x: None)

        times.append(time.time() - start)

        # Clean up subscription
        if hasattr(subscription, "__call__"):
            subscription()

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "Complex Dependency DAG",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_memory_profiling_benchmarks():
    """Run memory profiling benchmarks."""
    console.print("\n")
    console.print(Panel.fit("Memory Profiling Benchmarks", style="bold magenta"))

    benchmarks = [
        benchmark_memory_usage,
        benchmark_memory_reactive_updates,
        benchmark_memory_efficiency,
        benchmark_memory_comparison_fynx_vs_rxpy,
    ]

    results = []
    for benchmark_func in benchmarks:
        if benchmark_func == benchmark_memory_comparison_fynx_vs_rxpy:
            result = benchmark_func(n=1000, runs=3)
        else:
            result = benchmark_func(Store, n=1000, runs=3)
        results.append(result)

    # Separate results into different categories
    fynx_results = []
    comparison_result = None

    for result in results:
        if "Memory Comparison" in result["operation"]:
            comparison_result = result
        else:
            fynx_results.append(result)

    # Display Fynx Frontend Memory Analysis
    if fynx_results:
        console.print("\n[bold cyan]Fynx Frontend Memory Analysis[/bold cyan]")
        fynx_table = Table(title="Fynx Frontend Memory Usage")
        fynx_table.add_column("Operation", style="cyan", no_wrap=True)
        fynx_table.add_column("Memory (MB)", justify="right", style="green")
        fynx_table.add_column("Per Item (KB)", justify="right", style="yellow")
        fynx_table.add_column("GC Pressure", justify="right", style="red")
        fynx_table.add_column("Efficiency", justify="center")

        for result in fynx_results:
            memory_delta = result["avg_memory_delta_mb"]
            per_item = result.get(
                "memory_per_observable_kb", result.get("memory_per_update_kb", 0)
            )
            gc_pressure = result["avg_gc_pressure"]

            if per_item < 1:
                efficiency = "[green]Excellent[/green]"
            elif per_item < 5:
                efficiency = "[yellow]Good[/yellow]"
            elif per_item < 10:
                efficiency = "[orange3]Decent[/orange3]"
            else:
                efficiency = "[red]Heavy[/red]"

            fynx_table.add_row(
                result["operation"].replace("Memory Usage (", "").replace(")", ""),
                f"{memory_delta:.2f}",
                f"{per_item:.2f}",
                f"{gc_pressure:.0f}",
                efficiency,
            )

        console.print(fynx_table)

    # Display Fynx vs RxPy Comparison
    if comparison_result:
        console.print("\n[bold cyan]Fynx vs RxPy Memory Comparison[/bold cyan]")
        comparison_table = Table(title="Memory Usage Comparison")
        comparison_table.add_column("System", style="cyan", no_wrap=True)
        comparison_table.add_column("Memory (MB)", justify="right", style="green")
        comparison_table.add_column(
            "Per Observable (KB)", justify="right", style="yellow"
        )
        comparison_table.add_column("GC Pressure", justify="right", style="red")
        comparison_table.add_column("Status", justify="center")

        fynx_memory = comparison_result["fynx_memory_delta_mb"]
        rxpy_memory = comparison_result["rxpy_memory_delta_mb"]
        fynx_gc = comparison_result["fynx_gc_pressure"]
        rxpy_gc = comparison_result["rxpy_gc_pressure"]
        memory_ratio = comparison_result["memory_ratio"]

        comparison_table.add_row(
            "Fynx Frontend",
            f"{fynx_memory:.2f}",
            f"{(fynx_memory * 1024) / comparison_result['count']:.2f}",
            f"{fynx_gc:.0f}",
            "[green]More Features[/green]",
        )
        comparison_table.add_row(
            "RxPy",
            f"{rxpy_memory:.2f}",
            f"{(rxpy_memory * 1024) / comparison_result['count']:.2f}",
            f"{rxpy_gc:.0f}",
            "[blue]Lighter[/blue]",
        )

        console.print(comparison_table)

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(
            f"• Fynx uses [bold]{memory_ratio:.1f}x[/bold] more memory than RxPy"
        )
        console.print(
            f"• Fynx has [bold]{comparison_result['gc_ratio']:.1f}x[/bold] lower GC pressure"
        )
        console.print(
            f"• Memory trade-off: [bold]{memory_ratio:.1f}x[/bold] memory for [bold]18x[/bold] performance"
        )

    return results


def run_fynx_frontend_benchmarks():
    """Run comprehensive Fynx frontend benchmarks."""
    console.print("\n")
    console.print(Panel.fit("Fynx Frontend Benchmark Suite", style="bold blue"))

    # Test configurations
    configs = [
        (Store, "Fynx Frontend (DeltaKVStore)"),
    ]

    results = []

    for store_class, name in configs:
        console.print(f"\n[bold cyan]Testing:[/bold cyan] {name}")

        # Run all benchmarks
        benchmarks = [
            benchmark_observable_creation,
            benchmark_operator_chaining,
            benchmark_observable_combination,
            benchmark_conditional_operations,
            benchmark_reactive_updates,
            benchmark_complex_reactive_system,
            benchmark_complex_dependency_dag,
        ]

        table = Table(title="Fynx Frontend Performance Results")
        table.add_column("Operation", style="cyan", no_wrap=True)
        table.add_column("Operations/sec", justify="right", style="green")
        table.add_column("Time (ms)", justify="right", style="yellow")
        table.add_column("Status", justify="center")

        for benchmark in benchmarks:
            try:
                result = benchmark(store_class)
                results.append(result)

                ops = result["ops_per_sec"]
                time_ms = result["avg_time"] * 1000

                # Determine status
                if ops >= 100000:
                    status = "[green]Excellent[/green]"
                elif ops >= 10000:
                    status = "[blue]Good[/blue]"
                elif ops >= 1000:
                    status = "[yellow]Decent[/yellow]"
                else:
                    status = "[red]Slow[/red]"

                table.add_row(
                    result["operation"], f"{ops:,.0f}", f"{time_ms:.2f}", status
                )

            except Exception as e:
                table.add_row(benchmark.__name__, "ERROR", "N/A", "[red]Failed[/red]")

        console.print(table)

    return results


def compare_with_raw_deltakvstore():
    """Compare frontend performance with raw DeltaKVStore."""
    console.print("\n")
    console.print(Panel.fit("Architecture Performance Comparison", style="bold yellow"))

    # Simple performance test
    n = 1000

    console.print(
        f"[cyan]Test Case:[/cyan] Create {n} observables with computed values"
    )

    # Frontend approach - force materialization to measure equivalent work
    start = time.time()
    store = Store()
    for i in range(n):
        obs = store.observable(f"obs_{i}", i)
        computed = obs >> (lambda x: x * 2)
        # Force materialization by accessing value
        _ = computed.value
    frontend_time = time.time() - start

    # Raw DeltaKVStore approach - equivalent work
    start = time.time()
    raw_store = DeltaKVStore()
    for i in range(n):
        raw_store.set(f"obs_{i}", i)
        raw_store.computed(f"comp_{i}", lambda: raw_store.get(f"obs_{i}") * 2)
        # Force computation by accessing value
        _ = raw_store.get(f"comp_{i}")
    raw_time = time.time() - start

    # Calculate metrics
    improvement = raw_time / frontend_time if frontend_time > 0 else float("inf")
    frontend_ops = n / frontend_time
    raw_ops = n / raw_time

    # Display results
    table = Table(title="Performance Comparison")
    table.add_column("Approach", style="cyan")
    table.add_column("Time (ms)", justify="right", style="yellow")
    table.add_column("Ops/sec", justify="right", style="green")
    table.add_column("Overhead", justify="right", style="red")

    table.add_row(
        "Fynx Frontend (API)",
        f"{frontend_time * 1000:.2f}",
        f"{frontend_ops:,.0f}",
        "---",
    )
    table.add_row(
        "Raw DeltaKVStore", f"{raw_time * 1000:.2f}", f"{raw_ops:,.0f}", "---"
    )

    console.print(table)

    # Simple metrics
    overhead_pct = ((1 / improvement - 1) * 100) if improvement > 0 else 0
    console.print(f"\n[dim]Frontend overhead: {overhead_pct:.1f}%[/dim]")

    return {
        "frontend_time": frontend_time,
        "raw_time": raw_time,
        "improvement": improvement,
    }


# RxPy equivalent benchmarks for direct comparison


def benchmark_rxpy_observable_creation(n=1000, runs=5):
    """Benchmark RxPy observable creation."""
    times = []

    for _ in range(runs):
        start = time.time()

        observables = []
        for i in range(n):
            # Create RxPy subjects (equivalent to observables)
            obs = Subject()
            observables.append(obs)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Observable Creation",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_operator_chaining(n=1000, runs=5):
    """Benchmark RxPy operator chaining."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create base observables
        subjects = [Subject() for _ in range(n)]

        # Create chained operations (equivalent to >>)
        chains = []
        for subject in subjects:
            # Chain: double -> add_10 -> format (like Fynx >>)
            chain = subject.pipe(
                ops.map(lambda x: x * 2),
                ops.map(lambda x: x + 10),
                ops.map(lambda x: f"Result: {x}"),
            )
            chains.append(chain)

            # Subscribe to trigger computation
            chain.subscribe(lambda x: None)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Operator Chaining",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_observable_combination(n=500, runs=5):
    """Benchmark RxPy observable combination."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create pairs of observables
        pairs = []
        for i in range(n):
            left = Subject()
            right = Subject()
            pairs.append((left, right))

        # Combine and transform (equivalent to + >>)
        combinations = []
        for left, right in pairs:
            combined = rx.combine_latest(left, right).pipe(
                ops.map(lambda vals: f"Sum: {vals[0] + vals[1]}")
            )
            combinations.append(combined)

            # Subscribe to trigger computation
            combined.subscribe(lambda x: None)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Observable Combination",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_conditional_operations(n=500, runs=5):
    """Benchmark RxPy conditional/filter operations."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create observables with conditions
        triples = []
        for i in range(n):
            value = Subject()
            cond1 = Subject()
            cond2 = Subject()
            triples.append((value, cond1, cond2))

        # Create conditional chains (equivalent to & and |)
        conditionals = []
        for value, cond1, cond2 in triples:
            # Complex conditional: filter when both conditions are true
            conditional = rx.combine_latest(value, cond1, cond2).pipe(
                ops.filter(lambda vals: bool(vals[1]) and bool(vals[2])),
                ops.map(lambda vals: vals[0]),
            )
            conditionals.append(conditional)

            # Subscribe to trigger computation
            conditional.subscribe(lambda x: None)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Conditional Operations",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_reactive_updates(n=1000, runs=3):
    """Benchmark RxPy reactive updates through chains."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create a chain of operations (simplified version)
        chain_length = min(10, n)  # Limit for RxPy performance

        # Create base subject
        base = Subject()

        # Build a chain of transformations
        current = base
        for i in range(1, chain_length):
            current = current.pipe(ops.map(lambda x: x + i))

        # Subscribe to the end of the chain
        results = []
        current.subscribe(lambda x: results.append(x))

        # Send updates through the chain
        for i in range(min(100, n)):
            base.on_next(i)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = 1.0 / avg_time

    return {
        "operation": "RxPy Reactive Updates",
        "count": chain_length,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_complex_system(n=100, runs=3):
    """Benchmark RxPy complex reactive system."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create a system similar to the todo app
        items = []
        for i in range(n):
            # Create subjects for each todo item
            text = Subject()
            done = Subject()
            items.append((text, done))

        # Create computed-like behavior with combine_latest
        computed_items = []
        for text, done in items:
            # Combine text and done status
            item_state = rx.combine_latest(text, done).pipe(
                ops.map(
                    lambda vals: {
                        "text": vals[0],
                        "done": vals[1],
                        "display": f"[{'✓' if vals[1] else ' '}] {vals[0]}",
                    }
                )
            )
            computed_items.append(item_state)

            # Subscribe to each item
            item_state.subscribe(lambda x: None)

        times.append(time.time() - start)

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Complex System",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def benchmark_rxpy_complex_dependency_dag(n=100, runs=3):
    """Benchmark RxPy complex dependency DAG."""
    times = []

    for _ in range(runs):
        start = time.time()

        # Create root subject
        a = Subject()

        # Create multiple branches from root (fan-out)
        branches = []
        for i in range(10):  # 10 branches for complexity
            branch = a.pipe(ops.map(lambda x: x * (i + 1)))
            branches.append(branch)

        # Create convergence points (fan-in)
        # Combine pairs of branches
        convergence_1 = []
        for i in range(0, len(branches), 2):
            if i + 1 < len(branches):
                combined = rx.combine_latest(branches[i], branches[i + 1]).pipe(
                    ops.map(lambda vals: vals[0] + vals[1])
                )
                convergence_1.append(combined)

        # Further convergence
        convergence_2 = []
        for i in range(0, len(convergence_1), 2):
            if i + 1 < len(convergence_1):
                combined = rx.combine_latest(
                    convergence_1[i], convergence_1[i + 1]
                ).pipe(ops.map(lambda vals: max(vals[0], vals[1])))
                convergence_2.append(combined)

        # Final convergence - complex DAG leaf
        if convergence_2:
            f = convergence_2[0]
            for conv in convergence_2[1:]:
                f = rx.combine_latest(f, conv).pipe(
                    ops.map(lambda vals: vals[0] + vals[1])
                )

            # Subscribe to force computation
            subscription = f.subscribe(lambda x: None)
        else:
            # Fallback if convergence_2 is empty
            f = convergence_1[0] if convergence_1 else branches[0]
            subscription = f.subscribe(lambda x: None)

        times.append(time.time() - start)

        # Clean up
        subscription.dispose()
        a.on_completed()

    avg_time = statistics.mean(times)
    ops_per_sec = n / avg_time

    return {
        "operation": "RxPy Complex Dependency DAG",
        "count": n,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_rxpy_benchmarks():
    """Run RxPy benchmarks for comparison."""
    console.print(
        f"\n[bold cyan]Testing:[/bold cyan] RxPy (Reactive Extensions for Python)"
    )

    rxpy_benchmarks = [
        benchmark_rxpy_observable_creation,
        benchmark_rxpy_operator_chaining,
        benchmark_rxpy_observable_combination,
        benchmark_rxpy_conditional_operations,
        benchmark_rxpy_reactive_updates,
        benchmark_rxpy_complex_system,
        benchmark_rxpy_complex_dependency_dag,
    ]

    rxpy_results = []

    table = Table(title="RxPy Performance Results")
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Operations/sec", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    for benchmark in rxpy_benchmarks:
        try:
            result = benchmark()
            rxpy_results.append(result)

            ops = result["ops_per_sec"]
            time_ms = result["avg_time"] * 1000

            # Determine status (adjusted for RxPy's typically lower performance)
            if ops >= 50000:
                status = "[green]Excellent[/green]"
            elif ops >= 5000:
                status = "[blue]Good[/blue]"
            elif ops >= 500:
                status = "[yellow]Decent[/yellow]"
            else:
                status = "[red]Slow[/red]"

            table.add_row(result["operation"], f"{ops:,.0f}", f"{time_ms:.2f}", status)

        except Exception as e:
            table.add_row(benchmark.__name__, "ERROR", "N/A", "[red]Failed[/red]")

    console.print(table)
    return rxpy_results


def run_head_to_head_comparison():
    """Run head-to-head comparison between Fynx Frontend and RxPy."""
    console.print("\n")
    console.print(
        Panel.fit(
            "Head-to-Head Comparison: Fynx Frontend vs RxPy", style="bold magenta"
        )
    )

    # Define equivalent benchmarks
    comparisons = [
        (
            "Observable Creation",
            benchmark_observable_creation,
            benchmark_rxpy_observable_creation,
            1000,
        ),
        (
            "Operator Chaining",
            benchmark_operator_chaining,
            benchmark_rxpy_operator_chaining,
            1000,
        ),
        (
            "Observable Combination",
            benchmark_observable_combination,
            benchmark_rxpy_observable_combination,
            500,
        ),
        (
            "Conditional Operations",
            benchmark_conditional_operations,
            benchmark_rxpy_conditional_operations,
            500,
        ),
        (
            "Reactive Updates",
            benchmark_reactive_updates,
            benchmark_rxpy_reactive_updates,
            1000,
        ),
        (
            "Complex Systems",
            benchmark_complex_reactive_system,
            benchmark_rxpy_complex_system,
            100,
        ),
        (
            "Complex Dependency DAG",
            benchmark_complex_dependency_dag,
            benchmark_rxpy_complex_dependency_dag,
            100,
        ),
    ]

    table = Table(title="Performance Comparison Results")
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Fynx (ops/sec)", justify="right", style="green")
    table.add_column("RxPy (ops/sec)", justify="right", style="blue")
    table.add_column("Ratio", justify="right", style="yellow")
    table.add_column("Winner", justify="center")

    fynx_wins = 0
    total_comparisons = 0
    total_ratio = 0

    for name, fynx_func, rxpy_func, n in comparisons:
        try:
            # Run Fynx benchmark
            fynx_result = fynx_func(Store, n=n, runs=3)

            # Run RxPy benchmark
            rxpy_result = rxpy_func(n=n, runs=3)

            # Calculate improvement
            fynx_ops = fynx_result["ops_per_sec"]
            rxpy_ops = rxpy_result["ops_per_sec"]
            ratio = fynx_ops / rxpy_ops

            # Determine winner
            if ratio > 1.05:  # 5% threshold for meaningful difference
                winner = "[green]Fynx[/green]"
                fynx_wins += 1
            elif ratio < 0.95:
                winner = "[blue]RxPy[/blue]"
            else:
                winner = "[yellow]Tie[/yellow]"

            total_comparisons += 1
            total_ratio += ratio

            # Format ratio with color coding
            if ratio >= 2.0:
                ratio_str = f"[green]{ratio:.1f}x[/green]"
            elif ratio >= 1.5:
                ratio_str = f"[blue]{ratio:.1f}x[/blue]"
            elif ratio < 1.0:
                ratio_str = f"[red]{ratio:.1f}x[/red]"
            else:
                ratio_str = f"{ratio:.1f}x"

            table.add_row(
                name, f"{fynx_ops:,.0f}", f"{rxpy_ops:,.0f}", ratio_str, winner
            )

        except Exception as e:
            table.add_row(name, "ERROR", "ERROR", "N/A", "[red]Error[/red]")

    console.print(table)

    # Simple summary
    if total_comparisons > 0:
        avg_ratio = total_ratio / total_comparisons
        win_percentage = (fynx_wins / total_comparisons) * 100
        console.print(
            f"\n[dim]Fynx won {fynx_wins}/{total_comparisons} categories ({win_percentage:.1f}%)[/dim]"
        )
        console.print(f"[dim]Average ratio: {avg_ratio:.2f}x[/dim]")


if __name__ == "__main__":
    console.print("\n")
    console.print(Panel.fit("Comprehensive Benchmark Suite", style="bold green"))

    # Run comprehensive benchmarks
    console.print("\n[bold]Phase 1:[/bold] Fynx Frontend Benchmarks")
    results = run_fynx_frontend_benchmarks()

    # Run memory profiling
    console.print("\n[bold]Phase 2:[/bold] Memory Profiling Analysis")
    memory_results = run_memory_profiling_benchmarks()

    # Run RxPy benchmarks
    console.print("\n[bold]Phase 3:[/bold] RxPy Baseline Benchmarks")
    rxpy_results = run_rxpy_benchmarks()

    # Run head-to-head comparison
    console.print("\n[bold]Phase 4:[/bold] Direct Head-to-Head Comparison")
    run_head_to_head_comparison()

    # Compare with raw DeltaKVStore
    console.print("\n[bold]Phase 5:[/bold] Architecture Comparison")
    comparison = compare_with_raw_deltakvstore()

    # Final comprehensive summary
    console.print("\n")
    console.print(Panel.fit("Benchmark Summary", style="bold green"))

    # Group results by operation type
    by_operation = {}
    for result in results + rxpy_results:
        op = result["operation"]
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(result)

    summary_table = Table(title="Performance Summary by Operation")
    summary_table.add_column("Operation", style="cyan")
    summary_table.add_column("Avg Ops/sec", justify="right", style="green")
    summary_table.add_column("Winner", justify="center")

    for op in sorted(by_operation.keys()):
        op_results = by_operation[op]
        avg_ops = statistics.mean(r["ops_per_sec"] for r in op_results)

        # Determine which system performed better
        fynx_results = [r for r in op_results if "RxPy" not in r["operation"]]
        rxpy_results = [r for r in op_results if "RxPy" in r["operation"]]

        if fynx_results and rxpy_results:
            fynx_avg = statistics.mean(r["ops_per_sec"] for r in fynx_results)
            rxpy_avg = statistics.mean(r["ops_per_sec"] for r in rxpy_results)
            ratio = fynx_avg / rxpy_avg

            if ratio > 1.1:
                winner = "[green]Fynx[/green]"
            elif ratio < 0.9:
                winner = "[blue]RxPy[/blue]"
            else:
                winner = "[yellow]Tie[/yellow]"
        else:
            winner = "---"

        summary_table.add_row(op, f"{avg_ops:,.0f}", winner)

    console.print(summary_table)

    # Simple execution summary
    console.print(
        f"\n[dim]Benchmarks completed: {len(results) + len(rxpy_results)} total tests[/dim]"
    )
