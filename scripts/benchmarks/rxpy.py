"""
Comprehensive FynX vs RxPY Performance Benchmark Suite

This benchmark suite covers:
1. Core RxPY operations (creation, transformation, combination)
2. Advanced RxPY operators (filtering, buffering, time-based)
3. FynX strong suits (conditional reactivity, dependency tracking, batching)
4. Memory efficiency and GC pressure
5. Complex reactive patterns
"""

import argparse

# Ensure we use the local fynx package, not the installed one
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

T = TypeVar("T")

import rx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rx import operators as ops
from rx.subject import BehaviorSubject, ReplaySubject, Subject
from utils import (
    CONFIG,
    REGISTRY,
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkProfiler,
    BenchmarkRegistry,
    benchmark,
    run_adaptive_benchmark,
)

from fynx import observable, reactive, transaction

# ============================================================================
# CORE OPERATIONS - Basic RxPY Features
# ============================================================================


@benchmark("Observable Creation", category="Core Operations", detailed_profiling=True)
def bench_creation_fynx(n):
    return [observable(i) for i in range(n)]


@benchmark("Observable Creation", library="rxpy", category="Core Operations")
def bench_creation_rxpy(n):
    subjects = []
    for i in range(n):
        subject = Subject()
        subject.on_next(i)
        subjects.append(subject)
    return subjects


@benchmark("Individual Updates", category="Core Operations")
def bench_updates_fynx(n):
    observables = [observable(0) for _ in range(n)]
    for i, obs in enumerate(observables):
        obs.set(i * 2)
    return observables


@benchmark("Individual Updates", library="rxpy", category="Core Operations")
def bench_updates_rxpy(n):
    subjects = [Subject() for _ in range(n)]
    for i, subject in enumerate(subjects):
        subject.on_next(i * 2)
    return subjects


@benchmark("BehaviorSubject Creation", category="Core Operations")
def bench_behavior_fynx(n):
    return [observable(i) for i in range(n)]


@benchmark("BehaviorSubject Creation", library="rxpy", category="Core Operations")
def bench_behavior_rxpy(n):
    subjects = []
    for i in range(n):
        subject = BehaviorSubject(i)
        subjects.append(subject)
    return subjects


# ============================================================================
# OPTIMIZED RXPY VERSIONS - Using RxPY best practices
# ============================================================================


@benchmark(
    "Map Transformation",
    library="rxpy",
    category="Transformations",
    operations_counter=lambda x: x,
)
def bench_map_rxpy_optimized(n):
    base = Subject()
    # Use share() for multicasting to avoid redundant computations
    mapped = base.pipe(ops.map(lambda x: x * 2)).pipe(ops.share())

    result = []
    mapped.subscribe(result.append)

    for i in range(n):
        base.on_next(i)
    return n


@benchmark(
    "Conditional Reactivity",
    library="rxpy-optimized",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_conditional_rxpy_optimized(n):
    value = Subject()
    is_active = BehaviorSubject(False)

    # Use withLatestFrom instead of combineLatest for better performance
    filtered = value.pipe(
        ops.with_latest_from(is_active),
        ops.filter(lambda pair: pair[1]),
        ops.map(lambda pair: pair[0]),
    )

    results = []
    filtered.subscribe(results.append)

    for i in range(n):
        value.on_next(i)
        is_active.on_next(i % 2 == 0)

    return n


# ============================================================================
# TRANSFORMATION OPERATORS
# ============================================================================


@benchmark(
    "Map Transformation",
    category="Transformations",
    operations_counter=lambda x: x,
    detailed_profiling=True,
)
def bench_map_fynx(n):
    base = observable(0)
    mapped = base >> (lambda x: x * 2)

    count = 0

    def increment(_):
        nonlocal count
        count += 1

    mapped.subscribe(increment)
    for i in range(n):
        base.set(i)
    return count


@benchmark(
    "Base Observable Set", category="Transformations", operations_counter=lambda x: x
)
def bench_base_set_fynx(n):
    """Benchmark just the base observable set operation without computed observables."""
    base = observable(0)
    for i in range(n):
        base.set(i)
    return n


@benchmark(
    "Map Transformation",
    library="rxpy",
    category="Transformations",
    operations_counter=lambda x: x,
)
def bench_map_rxpy(n):
    base = Subject()
    mapped = base.pipe(ops.map(lambda x: x * 2))

    result = []
    mapped.subscribe(result.append)

    for i in range(n):
        base.on_next(i)
    return n


@benchmark(
    "Chain Propagation", category="Transformations", operations_counter=lambda x: x
)
def bench_chain_fynx(n):
    # Limit depth to prevent excessive computation
    depth = min(n, 400)
    base = observable(1)
    current = base
    for i in range(depth):
        current = current >> (lambda x, i=i: x + i)

    _ = current.value
    base.set(2)
    _ = current.value
    return depth


@benchmark(
    "Chain Propagation",
    library="rxpy",
    category="Transformations",
    operations_counter=lambda x: x,
)
def bench_chain_rxpy(n):
    # Use same depth limit as FynX
    depth = min(n, 400)
    base = Subject()
    current = base
    for i in range(depth):
        current = current.pipe(ops.map(lambda x, i=i: x + i))

    result = None

    def capture(x):
        nonlocal result
        result = x

    current.subscribe(capture)
    base.on_next(2)
    return depth


@benchmark(
    "FlatMap Transformation", category="Transformations", operations_counter=lambda x: x
)
def bench_flatmap_fynx(n):
    base = observable(0)
    # Simulate flatmap: each input produces a stream of 2 values
    results = []

    # For each base value, we "flatten" it to multiple derived values
    @reactive(base)
    def flatten(val):
        # Simulate producing 2 values per input (like flatMap(x => [x*2, x*2+1]))
        results.append(val * 2)
        results.append(val * 2 + 1)

    for i in range(n):
        base.set(i)

    return len(results)


@benchmark(
    "FlatMap Transformation",
    library="rxpy",
    category="Transformations",
    operations_counter=lambda x: x,
)
def bench_flatmap_rxpy(n):
    base = Subject()
    flattened = base.pipe(ops.flat_map(lambda x: rx.from_iterable([x * 2, x * 2 + 1])))

    results = []
    flattened.subscribe(results.append)

    for i in range(n):
        base.on_next(i)

    return len(results)


@benchmark(
    "Scan Accumulation", category="Transformations", operations_counter=lambda x: x
)
def bench_scan_fynx(n):
    base = observable(0)
    accumulated = base.scan(lambda acc, x: acc + x, 0)

    for i in range(
        1, n + 1
    ):  # Start from 1 to match RxPY (which accumulates 1+2+...+n)
        base.set(i)

    return accumulated.value


@benchmark(
    "Scan Accumulation",
    library="rxpy",
    category="Transformations",
    operations_counter=lambda x: x,
)
def bench_scan_rxpy(n):
    base = Subject()
    accumulated = base.pipe(ops.scan(lambda acc, x: acc + x, 0))

    result = []
    accumulated.subscribe(result.append)

    for i in range(n):
        base.on_next(i)

    return n


# ============================================================================
# FILTERING OPERATORS
# ============================================================================


@benchmark("Filter Operation", category="Filtering", operations_counter=lambda x: x)
def bench_filter_fynx(n):
    base = observable(0)
    filtered = base & (lambda x: x % 2 == 0)

    results = []
    filtered.subscribe(results.append)

    for i in range(n):
        base.set(i)

    return n


@benchmark(
    "Filter Operation",
    library="rxpy",
    category="Filtering",
    operations_counter=lambda x: x,
)
def bench_filter_rxpy(n):
    base = Subject()
    filtered = base.pipe(ops.filter(lambda x: x % 2 == 0))

    results = []
    filtered.subscribe(results.append)

    for i in range(n):
        base.on_next(i)

    return n


@benchmark(
    "Distinct Until Changed", category="Filtering", operations_counter=lambda x: x
)
def bench_distinct_fynx(n):
    base = observable(0)
    last_value = observable(None)

    results = []

    @reactive(base)
    def track_distinct(val):
        if val != last_value.value:
            last_value.set(val)
            results.append(val)

    for i in range(n):
        base.set(i % 10)  # Repeat values

    return len(results)


@benchmark(
    "Distinct Until Changed",
    library="rxpy",
    category="Filtering",
    operations_counter=lambda x: x,
)
def bench_distinct_rxpy(n):
    base = Subject()
    distinct = base.pipe(ops.distinct_until_changed())

    results = []
    distinct.subscribe(results.append)

    for i in range(n):
        base.on_next(i % 10)

    return len(results)


@benchmark("Take/Limit Operation", category="Filtering", operations_counter=lambda x: x)
def bench_take_fynx(n):
    base = observable(0)
    counter = observable(0)
    limit = min(n // 2, 100)

    results = []

    def collect_limited(val):
        if len(results) < limit:
            results.append(val)

    # Create conditional observable that only emits when under limit
    limited = base & (counter >> (lambda c: c < limit))

    @reactive(limited)
    def increment_counter(val):
        counter.set(counter.value + 1)
        collect_limited(val)

    for i in range(n):
        base.set(i)

    return len(results)


@benchmark(
    "Take/Limit Operation",
    library="rxpy",
    category="Filtering",
    operations_counter=lambda x: x,
)
def bench_take_rxpy(n):
    base = Subject()
    limit = min(n // 2, 100)
    taken = base.pipe(ops.take(limit))

    results = []
    taken.subscribe(results.append)

    for i in range(n):
        base.on_next(i)

    return len(results)


# ============================================================================
# COMBINATION OPERATORS
# ============================================================================


@benchmark("Reactive Fan-out", category="Combination", operations_counter=lambda x: x)
def bench_fanout_fynx(n):
    base = observable(42)
    dependents = [base.then(lambda x, i=i: x + i) for i in range(n)]
    base.set(100)
    return n


@benchmark(
    "Reactive Fan-out",
    library="rxpy",
    category="Combination",
    operations_counter=lambda x: x,
)
def bench_fanout_rxpy(n):
    base = Subject()
    dependents = [base.pipe(ops.map(lambda x, i=i: x + i)) for i in range(n)]

    # Subscribe all
    for dep in dependents:
        dep.subscribe(lambda x: None)

    base.on_next(100)
    return n


@benchmark("Stream Merge", category="Combination", operations_counter=lambda x: x)
def bench_merge_fynx(n):
    obs1 = observable(1)
    obs2 = observable(2)
    merged = obs1.alongside(obs2)

    for i in range(n):
        obs1.set(i)
        obs2.set(i * 2)
    return n


@benchmark(
    "Stream Merge",
    library="rxpy",
    category="Combination",
    operations_counter=lambda x: x,
)
def bench_merge_rxpy(n):
    obs1 = Subject()
    obs2 = Subject()
    merged = rx.merge(obs1, obs2)

    results = []
    merged.subscribe(results.append)

    for i in range(n):
        obs1.on_next(i)
        obs2.on_next(i * 2)
    return n


@benchmark("Stream Zip", category="Combination", operations_counter=lambda x: x)
def bench_zip_fynx(n):
    # FynX doesn't have zip, test StreamMerge performance instead
    obs1 = observable(0)
    obs2 = observable(0)
    merged = obs1 + obs2

    results = []
    merged.subscribe(results.append)

    for i in range(n):
        obs1.set(i)
        obs2.set(i * 2)
    return len(results)


@benchmark(
    "Stream Zip", library="rxpy", category="Combination", operations_counter=lambda x: x
)
def bench_zip_rxpy(n):
    obs1 = Subject()
    obs2 = Subject()
    zipped = rx.zip(obs1, obs2)

    results = []
    zipped.subscribe(results.append)

    for i in range(n):
        obs1.on_next(i)
        obs2.on_next(i * 2)
    return len(results)


@benchmark(
    "CombineLatest",
    category="Combination",
    operations_counter=lambda x: x,
    detailed_profiling=True,
)
def bench_combine_fynx(n):
    obs1 = observable(1)
    obs2 = observable(2)
    combined = obs1 + obs2

    results = []
    combined.subscribe(results.append)

    for i in range(n):
        obs1.set(i)
        obs2.set(i * 2)

    return n


@benchmark(
    "CombineLatest",
    library="rxpy",
    category="Combination",
    operations_counter=lambda x: x,
)
def bench_combine_rxpy(n):
    obs1 = Subject()
    obs2 = Subject()
    combined = rx.combine_latest(obs1, obs2)

    results = []
    combined.subscribe(results.append)

    for i in range(n):
        obs1.on_next(i)
        obs2.on_next(i * 2)

    return n


# ============================================================================
# FYNX STRONG SUITS - Conditional Reactivity
# ============================================================================


@benchmark(
    "Conditional Reactivity",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_conditional_fynx(n):
    value = observable(0)
    is_active = observable(False)

    # Conditional observable - only emits when active
    filtered = value & is_active

    results = []
    filtered.subscribe(results.append)

    for i in range(n):
        value.set(i)
        is_active.set(i % 2 == 0)

    return n


@benchmark(
    "Conditional Reactivity",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_conditional_rxpy(n):
    value = Subject()
    is_active = BehaviorSubject(False)

    # Simulate conditional with withLatestFrom + filter
    filtered = value.pipe(
        ops.with_latest_from(is_active),
        ops.filter(lambda pair: pair[1]),
        ops.map(lambda pair: pair[0]),
    )

    results = []
    filtered.subscribe(results.append)

    for i in range(n):
        value.on_next(i)
        is_active.on_next(i % 2 == 0)

    return n


@benchmark(
    "Complex Conditional Logic",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_complex_conditional_fynx(n):
    temp = observable(20)
    humidity = observable(50)
    is_daytime = observable(True)

    # Complex condition: high temp AND high humidity AND daytime
    is_extreme = (
        (temp >> (lambda t: t > 30)) & (humidity >> (lambda h: h > 70)) & is_daytime
    )

    alert_count = observable(0)

    @reactive(is_extreme)
    def track_alerts(val):
        if val:
            alert_count.set(alert_count.value + 1)

    for i in range(n):
        temp.set(20 + (i % 20))
        humidity.set(50 + (i % 30))
        is_daytime.set(i % 3 != 0)

    return n


@benchmark(
    "Complex Conditional Logic",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_complex_conditional_rxpy(n):
    temp = BehaviorSubject(20)
    humidity = BehaviorSubject(50)
    is_daytime = BehaviorSubject(True)

    is_extreme = rx.combine_latest(
        temp.pipe(ops.map(lambda t: t > 30)),
        humidity.pipe(ops.map(lambda h: h > 70)),
        is_daytime,
    ).pipe(ops.map(lambda vals: all(vals)))

    alert_count = [0]

    def track_alerts(val):
        if val:
            alert_count[0] += 1

    is_extreme.subscribe(track_alerts)

    for i in range(n):
        temp.on_next(20 + (i % 20))
        humidity.on_next(50 + (i % 30))
        is_daytime.on_next(i % 3 != 0)

    return n


# ============================================================================
# FYNX STRONG SUITS - Dependency Tracking
# ============================================================================


@benchmark(
    "Diamond Dependency", category="FynX Strong Suits", operations_counter=lambda x: x
)
def bench_diamond_fynx(n):
    base = observable(1)
    left = base >> (lambda x: x * 2)
    right = base >> (lambda x: x + 10)
    combined = left + right

    result = combined >> (lambda l, r: l + r)

    for i in range(n):
        base.set(i)
        _ = result.value

    return n


@benchmark(
    "Diamond Dependency",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_diamond_rxpy(n):
    base = Subject()
    left = base.pipe(ops.map(lambda x: x * 2))
    right = base.pipe(ops.map(lambda x: x + 10))
    combined = rx.combine_latest(left, right)

    result = combined.pipe(ops.map(lambda lr: lr[0] + lr[1]))

    results = []
    result.subscribe(results.append)

    for i in range(n):
        base.on_next(i)

    return n


@benchmark(
    "Deep Dependency Tree",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
    detailed_profiling=True,
)
def bench_deep_tree_fynx(n):
    depth = min(n, 15)  # Limit depth to prevent OOM
    base = observable(1)

    # Build tree
    nodes = [[base]]
    for level in range(depth):
        new_level = []
        for node in nodes[-1]:
            left = node >> (lambda x: x * 2)
            right = node >> (lambda x: x + 1)
            new_level.extend([left, right])
        nodes.append(new_level)

    # Update and read all leaves
    base.set(42)
    total = sum(leaf.value for leaf in nodes[-1])

    return depth


@benchmark(
    "Deep Dependency Tree",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_deep_tree_rxpy(n):
    depth = min(n, 20)  # RxPY struggles with deep trees
    base = Subject()

    # Build tree
    nodes = [[base]]
    for level in range(depth):
        new_level = []
        for node in nodes[-1]:
            left = node.pipe(ops.map(lambda x: x * 2))
            right = node.pipe(ops.map(lambda x: x + 1))
            new_level.extend([left, right])
        nodes.append(new_level)

    # Subscribe to leaves
    results = []
    for leaf in nodes[-1]:
        leaf.subscribe(results.append)

    base.on_next(42)

    return depth


# ============================================================================
# FYNX STRONG SUITS - Batching and Transactions
# ============================================================================


@benchmark(
    "Batch Updates", category="FynX Strong Suits", operations_counter=lambda x: x
)
def bench_batch_fynx(n):
    observables = [observable(i) for i in range(100)]
    # Use alongside() to create efficient StreamMerge instead of chained +
    combined = observables[0].alongside(*observables[1:])
    final = combined >> (lambda *args: sum(args))

    for i in range(n):
        with transaction():
            for obs in observables:
                obs.set(obs.value + 1)
        _ = final.value

    return n


@benchmark(
    "Batch Updates",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_batch_rxpy(n):
    subjects = [BehaviorSubject(i) for i in range(100)]

    combined = rx.combine_latest(*subjects)
    final = combined.pipe(ops.map(lambda args: sum(args)))

    results = []
    final.subscribe(results.append)

    for i in range(n):
        # RxPY doesn't have built-in batching like FynX transactions
        # Each update triggers intermediate calculations (glitches)
        for subject in subjects:
            subject.on_next(subject.value + 1)

    return n


@benchmark(
    "Glitch-Free Updates", category="FynX Strong Suits", operations_counter=lambda x: x
)
def bench_glitch_free_fynx(n):
    x = observable(1)
    y = x >> (lambda v: v * 2)
    z = x >> (lambda v: v + 1)
    result = y + z

    final = result >> (lambda a, b: a + b)

    update_count = [0]

    @reactive(final)
    def count_updates(val):
        update_count[0] += 1

    for i in range(n):
        with transaction():
            x.set(i)

    return update_count[0]


@benchmark(
    "Glitch-Free Updates",
    library="rxpy",
    category="FynX Strong Suits",
    operations_counter=lambda x: x,
)
def bench_glitch_free_rxpy(n):
    x = BehaviorSubject(1)
    y = x.pipe(ops.map(lambda v: v * 2))
    z = x.pipe(ops.map(lambda v: v + 1))

    result = rx.combine_latest(y, z)
    final = result.pipe(ops.map(lambda ab: ab[0] + ab[1]))

    update_count = [0]

    def count_updates(val):
        update_count[0] += 1

    final.subscribe(count_updates)

    for i in range(n):
        x.on_next(i)

    return update_count[0]


# ============================================================================
# MEMORY EFFICIENCY
# ============================================================================


@benchmark("Memory Pressure - Many Observables", category="Memory Efficiency")
def bench_memory_many_fynx(n):
    observables = [observable(i) for i in range(n)]
    # Access all values to ensure materialization
    _ = [obs.value for obs in observables]
    return observables


@benchmark(
    "Memory Pressure - Many Observables", library="rxpy", category="Memory Efficiency"
)
def bench_memory_many_rxpy(n):
    subjects = [BehaviorSubject(i) for i in range(n)]
    # Subscribe to ensure materialization like FynX
    for subject in subjects:
        subject.subscribe(lambda x: None)
    return subjects


@benchmark("Memory Pressure - Deep Chains", category="Memory Efficiency")
def bench_memory_chains_fynx(n):
    chains = []
    for _ in range(n):
        base = observable(0)
        current = base
        for _ in range(10):
            current = current >> (lambda x: x + 1)
        chains.append(current)
        _ = current.value  # Materialize
    return chains


@benchmark(
    "Memory Pressure - Deep Chains", library="rxpy", category="Memory Efficiency"
)
def bench_memory_chains_rxpy(n):
    chains = []
    for _ in range(n):
        base = Subject()
        current = base
        for _ in range(10):
            current = current.pipe(ops.map(lambda x: x + 1))

        # Subscribe to ensure materialization
        result = []
        current.subscribe(result.append)
        base.on_next(0)
        chains.append(current)
    return chains


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="FynX vs RxPY Performance Comparison")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--benchmarks", nargs="+", help="Run specific benchmarks")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile")
    parser.add_argument("--time-limit", type=float, help="Time limit per benchmark")
    parser.add_argument("--iterations", type=int, help="Number of iterations")
    parser.add_argument("--category", help="Run only benchmarks in this category")

    args = parser.parse_args()

    if args.list:
        console = Console()
        console.print("\n[bold]Available Benchmarks:[/bold]")

        categories = {}
        for name in REGISTRY.list_benchmarks():
            category = REGISTRY.get_category(name)
            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        for category in sorted(categories.keys()):
            console.print(f"\n[cyan]{category}:[/cyan]")
            for name in sorted(categories[category]):
                console.print(f"  • {name}")
        return

    # Update config
    if args.time_limit:
        CONFIG.time_limit = args.time_limit
    if args.iterations:
        CONFIG.num_iterations = args.iterations
    if args.profile:
        CONFIG.profile_enabled = True

    # Filter by category if specified
    benchmark_names = args.benchmarks
    if args.category:
        all_benchmarks = REGISTRY.list_benchmarks()
        benchmark_names = [
            name
            for name in all_benchmarks
            if REGISTRY.get_category(name) == args.category
        ]

    # Run comparison
    comparison = BenchmarkComparison(CONFIG)
    comparison.run(benchmark_names=benchmark_names, registry=REGISTRY)


if __name__ == "__main__":
    main()
