"""
Fynx Frontend Benchmark Suite
=============================

Comprehensive benchmarks for the Fynx-style frontend with DeltaKVStore backend.
Tests observable operations, operator chaining, and reactive performance.
"""

import statistics
import time

import rx
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rx import operators as ops
from rx.subject import Subject

from frontend import Observable, Store, reactive
from prototype import DeltaKVStore

console = Console()


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
    """Benchmark combining observables with + operator."""
    times = []

    for _ in range(runs):
        start = time.time()

        class TestStore(store_class):
            pass

        store = TestStore()

        # Create pairs of observables
        for i in range(n):
            setattr(store, f"left_{i}", store.observable(f"left_{i}", i))
            setattr(store, f"right_{i}", store.observable(f"right_{i}", i * 2))

        # Combine and transform
        for i in range(n):
            left = getattr(store, f"left_{i}")
            right = getattr(store, f"right_{i}")
            combined = (left + right) >> (lambda a, b: f"Sum: {a + b}")

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
        for _ in range(min(100, n)):  # Multiple updates
            update_start = time.time()
            base.value = base.value + 1
            _ = observables[-1].value  # Force recomputation
            update_times.append(time.time() - update_start)

        times.append(statistics.mean(update_times))

    avg_time = statistics.mean(times)
    ops_per_sec = 1.0 / avg_time

    return {
        "operation": "Reactive Updates (Chain)",
        "count": chain_length,
        "avg_time": avg_time,
        "ops_per_sec": ops_per_sec,
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

    # Frontend approach
    start = time.time()
    store = Store()
    for i in range(n):
        obs = store.observable(f"obs_{i}", i)
        computed = obs >> (lambda x: x * 2)
    frontend_time = time.time() - start

    # Raw DeltaKVStore approach
    start = time.time()
    raw_store = DeltaKVStore()
    for i in range(n):
        raw_store.set(f"obs_{i}", i)
        raw_store.computed(f"comp_{i}", lambda: raw_store.get(f"obs_{i}") * 2)
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
            500,
        ),
        (
            "Observable Combination",
            benchmark_observable_combination,
            benchmark_rxpy_observable_combination,
            200,
        ),
        (
            "Conditional Operations",
            benchmark_conditional_operations,
            benchmark_rxpy_conditional_operations,
            200,
        ),
        (
            "Reactive Updates",
            benchmark_reactive_updates,
            benchmark_rxpy_reactive_updates,
            10,
        ),
        (
            "Complex Systems",
            benchmark_complex_reactive_system,
            benchmark_rxpy_complex_system,
            50,
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

    # Run RxPy benchmarks
    console.print("\n[bold]Phase 2:[/bold] RxPy Baseline Benchmarks")
    rxpy_results = run_rxpy_benchmarks()

    # Run head-to-head comparison
    console.print("\n[bold]Phase 3:[/bold] Direct Head-to-Head Comparison")
    run_head_to_head_comparison()

    # Compare with raw DeltaKVStore
    console.print("\n[bold]Phase 4:[/bold] Architecture Comparison")
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
