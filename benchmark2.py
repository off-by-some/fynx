#!/usr/bin/env python3
"""Fixed-size FynX benchmarks with optional RxPY comparisons.

This benchmark favors repeatability over theatrics:
- fixed workload sizes
- construction timed separately from update/propagation
- correctness checks outside timed regions where practical
- fresh worker processes for repetitions
- optional normal-GC and GC-disabled timing modes
- memory measured while graphs are still live

RxPY comparisons are skipped unless ``reactivex`` is installed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from fynx import observable


DEFAULT_PROCESSES = 5
DEFAULT_SAMPLES = 15
WARMUP_UPDATES = 200

FYNX_CREATION_SIZES = [1_000, 10_000, 100_000]
FYNX_UPDATE_SIZES = [1_000, 10_000, 100_000]
FYNX_CHAIN_SIZES = [10, 100, 1_000, 10_000]
FYNX_FANOUT_SIZES = [10, 100, 1_000, 10_000]
FYNX_DIAMOND_SIZES = [10, 100, 1_000]
FYNX_DYNAMIC_SWITCHES = [1_000, 10_000]

COMPARE_EVENT_SIZES = [10_000, 100_000, 1_000_000]
COMPARE_CHAIN_DEPTHS = [1, 10, 100, 1_000]
COMPARE_FANOUT_SIZES = [1, 10, 100, 1_000, 10_000]

QUICK_SCALE = {
    "creation": [1_000],
    "updates": [1_000],
    "chain": [10, 100],
    "fanout": [10, 100],
    "diamond": [10],
    "dynamic": [1_000],
    "compare_events": [10_000],
    "compare_chain": [1, 10],
    "compare_fanout": [1, 10],
}


def ns() -> int:
    return time.perf_counter_ns()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "min": min(values),
        "median": median,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mad": statistics.median(deviations),
    }


def rss_bytes() -> int:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024
    except Exception:
        return 0


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "pid": os.getpid(),
    }


@dataclass
class WorkerResult:
    workload: str
    library: str
    size: int
    variant: str
    unit: str
    samples_ns: List[int]
    correct: bool
    metrics: Dict[str, Any]
    skipped: Optional[str] = None


class CounterSink:
    __slots__ = ("count", "last")

    def __init__(self) -> None:
        self.count = 0
        self.last = None

    def __call__(self, value: Any) -> None:
        self.count += 1
        self.last = value


def maybe_disable_gc(mode: str) -> bool:
    was_enabled = gc.isenabled()
    if mode == "disabled" and was_enabled:
        gc.disable()
    return was_enabled


def restore_gc(mode: str, was_enabled: bool) -> None:
    if mode == "disabled" and was_enabled:
        gc.enable()


def time_samples(
    sample_count: int,
    gc_mode: str,
    sample: Callable[[int], int],
) -> List[int]:
    timings = []
    for index in range(sample_count):
        was_enabled = maybe_disable_gc(gc_mode)
        try:
            start = ns()
            sample(index)
            timings.append(ns() - start)
        finally:
            restore_gc(gc_mode, was_enabled)
    return timings


def memory_for_build(
    build: Callable[[], Any], live_count: int
) -> Tuple[Any, Dict[str, Any]]:
    gc.collect()
    rss_before = rss_bytes()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    obj = build()
    current, peak = tracemalloc.get_traced_memory()
    rss_after = rss_bytes()
    traced_delta = max(0, current - before_current)
    metrics = {
        "traced_bytes": traced_delta,
        "peak_traced_bytes": peak,
        "bytes_per_live_node": traced_delta / live_count if live_count else 0,
        "rss_before": rss_before,
        "rss_after": rss_after,
        "rss_delta": max(0, rss_after - rss_before),
    }
    tracemalloc.stop()
    return obj, metrics


def fynx_creation(size: int, samples: int, gc_mode: str) -> WorkerResult:
    created, memory = memory_for_build(
        lambda: [observable(i) for i in range(size)],
        size,
    )
    assert len(created) == size

    def sample(index: int) -> int:
        values = [observable(index + i) for i in range(size)]
        assert len(values) == size
        return len(values)

    timings = time_samples(samples, gc_mode, sample)
    del created
    gc.collect()
    memory["rss_after_teardown"] = rss_bytes()
    return WorkerResult(
        "creation", "fynx", size, "independent", "build", timings, True, memory
    )


def fynx_updates(
    size: int, samples: int, gc_mode: str, subscribed: bool
) -> WorkerResult:
    values = [observable(0) for _ in range(size)]
    sinks = []
    if subscribed:
        sinks = [CounterSink() for _ in range(size)]
        for value, sink in zip(values, sinks):
            value.subscribe(sink)

    for index, value in enumerate(values[: min(size, WARMUP_UPDATES)]):
        value.set(-(index + 1))

    latency_samples = []

    def sample(sample_index: int) -> int:
        base_value = sample_index * size + 1
        for offset, value in enumerate(values):
            if len(latency_samples) < 2048:
                start = ns()
                value.set(base_value + offset)
                latency_samples.append(ns() - start)
            else:
                value.set(base_value + offset)
        return size

    timings = time_samples(samples, gc_mode, sample)
    if subscribed:
        expected_min = samples * size
        assert sum(sink.count for sink in sinks) >= expected_min
    assert values[-1].value == samples * size
    return WorkerResult(
        "updates",
        "fynx",
        size,
        "subscribed" if subscribed else "unsubscribed",
        "update-batch",
        timings,
        True,
        {"per_update_latency_ns": summarize([float(v) for v in latency_samples])},
    )


def fynx_chain(size: int, samples: int, gc_mode: str) -> WorkerResult:
    source = observable(0)
    current = source
    for _ in range(size):
        current = current >> (lambda x: x + 1)
    sink = CounterSink()
    current.subscribe(sink)

    for value in range(WARMUP_UPDATES):
        source.set(-(value + 1))
    sink.count = 0

    def sample(index: int) -> int:
        source.set(index + 1)
        return 1

    timings = time_samples(samples, gc_mode, sample)
    assert sink.count == samples
    assert sink.last == samples + size
    return WorkerResult(
        "chain",
        "fynx",
        size,
        "final-subscriber",
        "source-set-to-final-callback",
        timings,
        True,
        {"ns_per_edge_median": statistics.median(timings) / max(size, 1)},
    )


def fynx_fanout(size: int, samples: int, gc_mode: str) -> WorkerResult:
    source = observable(0)
    sinks = [CounterSink() for _ in range(size)]
    dependents = []
    for index, sink in enumerate(sinks):
        dep = source >> (lambda x, offset=index: x + offset)
        dep.subscribe(sink)
        dependents.append(dep)

    for value in range(WARMUP_UPDATES):
        source.set(-(value + 1))
    for sink in sinks:
        sink.count = 0

    def sample(index: int) -> int:
        source.set(index + 1)
        return size

    timings = time_samples(samples, gc_mode, sample)
    assert sum(sink.count for sink in sinks) == samples * size
    assert sinks[-1].last == samples + size - 1
    return WorkerResult(
        "fanout",
        "fynx",
        size,
        "computed-dependent-subscribers",
        "source-set-to-all-callbacks",
        timings,
        True,
        {"ns_per_dependent_median": statistics.median(timings) / max(size, 1)},
    )


def fynx_diamonds(size: int, samples: int, gc_mode: str) -> WorkerResult:
    source = observable(0)
    recompute_counts = {"a": 0, "b": 0, "c": 0}
    sinks = []

    def count_a(value):
        recompute_counts["a"] += 1
        return value + 1

    def count_b(value):
        recompute_counts["b"] += 1
        return value * 2

    def count_c(left, right):
        recompute_counts["c"] += 1
        return left + right

    for _ in range(size):
        left = source >> count_a
        right = source >> count_b
        converged = (left + right) >> count_c
        sink = CounterSink()
        converged.subscribe(sink)
        sinks.append(sink)

    for value in range(WARMUP_UPDATES):
        source.set(-(value + 1))
    recompute_counts = {"a": 0, "b": 0, "c": 0}
    for sink in sinks:
        sink.count = 0

    def sample(index: int) -> int:
        source.set(index + 1)
        return size

    timings = time_samples(samples, gc_mode, sample)
    assert sum(sink.count for sink in sinks) == samples * size
    source_updates = samples
    expected_intermediate_recomputes = source_updates * size * 2
    expected_terminal_recomputes = source_updates * size
    observed_intermediate_recomputes = recompute_counts["a"] + recompute_counts["b"]
    observed_terminal_recomputes = recompute_counts["c"]
    return WorkerResult(
        "diamond",
        "fynx",
        size,
        "many-diamonds",
        "source-set-to-converged-callbacks",
        timings,
        True,
        {
            "diamonds": size,
            "semantic_nodes": 1 + (size * 3),
            "runtime_nodes": 1 + (size * 4),
            "source_updates": source_updates,
            "timed_phase": "propagation only",
            "graph_build_timed": False,
            "recompute_counts": recompute_counts,
            "expected_intermediate_recomputes": expected_intermediate_recomputes,
            "observed_intermediate_recomputes": observed_intermediate_recomputes,
            "expected_terminal_recomputes": expected_terminal_recomputes,
            "observed_terminal_recomputes": observed_terminal_recomputes,
            "duplicate_terminal_recomputes": max(
                0, observed_terminal_recomputes - expected_terminal_recomputes
            ),
            "single_recompute_per_diamond": observed_terminal_recomputes
            == expected_terminal_recomputes,
        },
    )


def fynx_dynamic(size: int, samples: int, gc_mode: str) -> WorkerResult:
    source = observable(5)
    use_left = observable(True)
    left_limit = observable(10)
    right_limit = observable(3)
    recomputes = {"predicate": 0}

    def under_active_limit(value):
        recomputes["predicate"] += 1
        return value < (left_limit.value if use_left.value else right_limit.value)

    selected = source & under_active_limit
    sink = CounterSink()
    selected.subscribe(sink)

    def do_switches(base: int) -> None:
        for offset in range(size):
            use_left.set((base + offset) % 2 == 0)

    do_switches(0)
    sink.count = 0
    recomputes["predicate"] = 0

    def sample(index: int) -> int:
        do_switches(index * size)
        return size

    timings = time_samples(samples, gc_mode, sample)
    active_count = sink.count

    # Untimed correctness check: updating the inactive branch must not recompute.
    before = recomputes["predicate"]
    if use_left.value:
        right_limit.set(100)
    else:
        left_limit.set(100)
    inactive_delta = recomputes["predicate"] - before
    assert inactive_delta == 0

    gc.collect()
    return WorkerResult(
        "dynamic",
        "fynx",
        size,
        "toggle-selected-dependency",
        "switch-batch",
        timings,
        True,
        {
            "predicate_recomputes": recomputes["predicate"],
            "inactive_update_recomputes": inactive_delta,
            "subscriber_notifications": active_count,
            "timed_phase": "condition-toggle-propagation",
            "rss_after_gc": rss_bytes(),
        },
    )


class ReactiveAdapter(Protocol):
    name: str

    def source(self, initial: int) -> Any: ...

    def map(self, source: Any, fn: Callable[[Any], Any]) -> Any: ...

    def scan(self, source: Any, fn: Callable[[Any, Any], Any], seed: Any) -> Any: ...

    def subscribe(self, source: Any, callback: Callable[[Any], None]) -> Any: ...

    def emit(self, source: Any, value: Any) -> None: ...

    def dispose(self) -> None: ...


class FynxAdapter:
    name = "fynx"

    def __init__(self) -> None:
        self._disposables = []

    def source(self, initial: int) -> Any:
        return observable(initial)

    def map(self, source: Any, fn: Callable[[Any], Any]) -> Any:
        return source >> fn

    def scan(self, source: Any, fn: Callable[[Any, Any], Any], seed: Any) -> Any:
        total = observable(seed)
        state = seed

        def update(value):
            nonlocal state
            state = fn(state, value)
            total.set(state)

        source.subscribe(update)
        self._disposables.append((source, update))
        return total

    def subscribe(self, source: Any, callback: Callable[[Any], None]) -> Any:
        source.subscribe(callback)
        self._disposables.append((source, callback))
        return callback

    def emit(self, source: Any, value: Any) -> None:
        source.set(value)

    def dispose(self) -> None:
        for source, callback in self._disposables:
            try:
                source.unsubscribe(callback)
            except Exception:
                pass
        self._disposables.clear()


class RxPYAdapter:
    name = "rxpy"

    def __init__(self) -> None:
        try:
            import reactivex
            from reactivex import operators as ops
            from reactivex.subject import Subject
        except Exception as exc:
            raise RuntimeError("reactivex is not installed") from exc

        self._reactivex = reactivex
        self._ops = ops
        self._subject_type = Subject
        self._disposables = []

    def source(self, initial: int) -> Any:
        return self._subject_type()

    def map(self, source: Any, fn: Callable[[Any], Any]) -> Any:
        return source.pipe(self._ops.map(fn))

    def scan(self, source: Any, fn: Callable[[Any, Any], Any], seed: Any) -> Any:
        return source.pipe(self._ops.scan(fn, seed=seed))

    def subscribe(self, source: Any, callback: Callable[[Any], None]) -> Any:
        disposable = source.subscribe(callback)
        self._disposables.append(disposable)
        return disposable

    def emit(self, source: Any, value: Any) -> None:
        source.on_next(value)

    def dispose(self) -> None:
        for disposable in self._disposables:
            disposable.dispose()
        self._disposables.clear()


def adapter_for(name: str) -> ReactiveAdapter:
    if name == "fynx":
        return FynxAdapter()
    if name == "rxpy":
        return RxPYAdapter()
    raise ValueError(f"unknown adapter: {name}")


def compare_map(library: str, size: int, samples: int, gc_mode: str) -> WorkerResult:
    try:
        adapter = adapter_for(library)
    except RuntimeError as exc:
        return WorkerResult(
            "compare-map", library, size, "map", "events", [], False, {}, str(exc)
        )

    source = adapter.source(0)
    mapped = adapter.map(source, lambda x: x * 2)
    sink = CounterSink()
    adapter.subscribe(mapped, sink)

    for value in range(WARMUP_UPDATES):
        adapter.emit(source, -(value + 1))
    sink.count = 0

    def sample(index: int) -> int:
        base = index * size
        for offset in range(size):
            adapter.emit(source, base + offset + 1)
        return size

    timings = time_samples(samples, gc_mode, sample)
    assert sink.count == samples * size
    assert sink.last == (samples * size) * 2
    adapter.dispose()
    return WorkerResult(
        "compare-map", library, size, "source-map-sink", "events", timings, True, {}
    )


def compare_chain(library: str, size: int, samples: int, gc_mode: str) -> WorkerResult:
    try:
        adapter = adapter_for(library)
    except RuntimeError as exc:
        return WorkerResult(
            "compare-chain", library, size, "chain", "emission", [], False, {}, str(exc)
        )

    source = adapter.source(0)
    current = source
    for _ in range(size):
        current = adapter.map(current, lambda x: x + 1)
    sink = CounterSink()
    adapter.subscribe(current, sink)

    for value in range(WARMUP_UPDATES):
        adapter.emit(source, -(value + 1))
    sink.count = 0

    def sample(index: int) -> int:
        adapter.emit(source, index + 1)
        return 1

    timings = time_samples(samples, gc_mode, sample)
    assert sink.count == samples
    assert sink.last == samples + size
    adapter.dispose()
    return WorkerResult(
        "compare-chain", library, size, "map-chain", "emission", timings, True, {}
    )


def compare_fanout(library: str, size: int, samples: int, gc_mode: str) -> WorkerResult:
    try:
        adapter = adapter_for(library)
    except RuntimeError as exc:
        return WorkerResult(
            "compare-fanout",
            library,
            size,
            "fanout",
            "emission",
            [],
            False,
            {},
            str(exc),
        )

    source = adapter.source(0)
    sinks = [CounterSink() for _ in range(size)]
    for sink in sinks:
        adapter.subscribe(source, sink)

    for value in range(WARMUP_UPDATES):
        adapter.emit(source, -(value + 1))
    for sink in sinks:
        sink.count = 0

    def sample(index: int) -> int:
        adapter.emit(source, index + 1)
        return size

    timings = time_samples(samples, gc_mode, sample)
    assert sum(sink.count for sink in sinks) == samples * size
    assert all(sink.last == samples for sink in sinks)
    adapter.dispose()
    return WorkerResult(
        "compare-fanout",
        library,
        size,
        "source-to-sinks",
        "emission",
        timings,
        True,
        {},
    )


def compare_accumulate(
    library: str, size: int, samples: int, gc_mode: str
) -> WorkerResult:
    try:
        adapter = adapter_for(library)
    except RuntimeError as exc:
        return WorkerResult(
            "compare-accumulate",
            library,
            size,
            "scan",
            "events",
            [],
            False,
            {},
            str(exc),
        )

    source = adapter.source(0)
    total = adapter.scan(source, lambda acc, value: acc + value, 0)
    sink = CounterSink()
    adapter.subscribe(total, sink)

    for value in range(WARMUP_UPDATES):
        adapter.emit(source, 0)
    sink.count = 0

    def sample(index: int) -> int:
        for value in range(1, size + 1):
            adapter.emit(source, value)
        return size

    timings = time_samples(samples, gc_mode, sample)
    expected_single = size * (size + 1) // 2
    assert sink.last == expected_single * samples
    adapter.dispose()
    return WorkerResult(
        "compare-accumulate",
        library,
        size,
        "running-total",
        "events",
        timings,
        True,
        {},
    )


def run_worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    workload = spec["workload"]
    size = int(spec["size"])
    samples = int(spec["samples"])
    gc_mode = spec["gc_mode"]
    library = spec.get("library", "fynx")
    variant = spec.get("variant", "")

    if workload == "creation":
        result = fynx_creation(size, samples, gc_mode)
    elif workload == "updates":
        result = fynx_updates(
            size, samples, gc_mode, subscribed=variant == "subscribed"
        )
    elif workload == "chain":
        result = fynx_chain(size, samples, gc_mode)
    elif workload == "fanout":
        result = fynx_fanout(size, samples, gc_mode)
    elif workload == "diamond":
        result = fynx_diamonds(size, samples, gc_mode)
    elif workload == "dynamic":
        result = fynx_dynamic(size, samples, gc_mode)
    elif workload == "compare-map":
        result = compare_map(library, size, samples, gc_mode)
    elif workload == "compare-chain":
        result = compare_chain(library, size, samples, gc_mode)
    elif workload == "compare-fanout":
        result = compare_fanout(library, size, samples, gc_mode)
    elif workload == "compare-accumulate":
        result = compare_accumulate(library, size, samples, gc_mode)
    else:
        raise ValueError(f"unknown workload: {workload}")

    output = {
        "workload": result.workload,
        "library": result.library,
        "size": result.size,
        "variant": result.variant,
        "unit": result.unit,
        "samples_ns": result.samples_ns,
        "correct": result.correct,
        "metrics": result.metrics,
        "skipped": result.skipped,
        "environment": environment(),
    }
    return output


def failed_worker_result(spec: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    return {
        "workload": spec.get("workload", "unknown"),
        "library": spec.get("library", "fynx"),
        "size": int(spec.get("size", 0)),
        "variant": spec.get("variant", "worker-error"),
        "unit": spec.get("unit", ""),
        "samples_ns": [],
        "correct": False,
        "metrics": {"error_type": type(exc).__name__},
        "skipped": f"worker failed: {type(exc).__name__}: {exc}",
        "environment": environment(),
    }


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    sizes = QUICK_SCALE if args.quick else {}

    def choose(name: str, default: List[int]) -> List[int]:
        return sizes.get(name, default)

    specs: List[Dict[str, Any]] = []
    wanted = set(args.workload or [])
    include_all = not wanted

    def add(workload: str, size_list: Iterable[int], **extra: Any) -> None:
        if include_all or workload in wanted:
            for size in size_list:
                specs.append(
                    {
                        "workload": workload,
                        "size": size,
                        "samples": args.samples,
                        "gc_mode": args.gc_mode,
                        **extra,
                    }
                )

    add("creation", choose("creation", FYNX_CREATION_SIZES))
    for variant in ["unsubscribed", "subscribed"]:
        add("updates", choose("updates", FYNX_UPDATE_SIZES), variant=variant)
    add("chain", choose("chain", FYNX_CHAIN_SIZES))
    add("fanout", choose("fanout", FYNX_FANOUT_SIZES))
    add("diamond", choose("diamond", FYNX_DIAMOND_SIZES))
    add("dynamic", choose("dynamic", FYNX_DYNAMIC_SWITCHES))

    if args.compare:
        comparison_specs: List[Dict[str, Any]] = []

        def add_compare(workload: str, size_list: Iterable[int]) -> None:
            if not include_all and workload not in wanted:
                return
            for size in size_list:
                for library in ["fynx", "rxpy"]:
                    comparison_specs.append(
                        {
                            "workload": workload,
                            "size": size,
                            "samples": args.samples,
                            "gc_mode": args.gc_mode,
                            "library": library,
                        }
                    )

        add_compare("compare-map", choose("compare_events", COMPARE_EVENT_SIZES))
        add_compare("compare-chain", choose("compare_chain", COMPARE_CHAIN_DEPTHS))
        add_compare("compare-fanout", choose("compare_fanout", COMPARE_FANOUT_SIZES))
        add_compare("compare-accumulate", choose("compare_events", COMPARE_EVENT_SIZES))

        if args.order == "grouped":
            comparison_specs.sort(
                key=lambda spec: (
                    0 if spec["library"] == "fynx" else 1,
                    spec["workload"],
                    spec["size"],
                )
            )
        elif args.order == "random":
            random.Random(args.seed).shuffle(comparison_specs)

        specs.extend(comparison_specs)

    return specs


def run_spec_in_process(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    encoded = json.dumps(spec)
    for _ in range(spec["processes"]):
        try:
            completed = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", encoded],
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(json.loads(completed.stdout))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            results.append(failed_worker_result(spec, exc))
    return results


def aggregate(process_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = process_results[0]
    all_samples = [
        float(sample)
        for result in process_results
        for sample in result.get("samples_ns", [])
    ]
    summary = summarize(all_samples)
    events_per_sample = max(1, int(first["size"]))
    if first["workload"] in {"chain", "compare-chain"}:
        events_per_sample = 1
    if first["workload"] in {"fanout", "diamond", "compare-fanout"}:
        events_per_sample = max(1, int(first["size"]))
    rate = (
        (events_per_sample * 1_000_000_000) / summary["median"]
        if summary and summary["median"]
        else 0
    )
    metrics = dict(first.get("metrics", {}))
    metrics["process_repetitions"] = len(process_results)
    metrics["samples_per_process"] = len(first.get("samples_ns", []))
    metrics["worker_pids"] = [
        result.get("environment", {}).get("pid") for result in process_results
    ]

    if first["workload"] == "diamond":
        recompute_counts = {"a": 0, "b": 0, "c": 0}
        for result in process_results:
            counts = result.get("metrics", {}).get("recompute_counts", {})
            for key in recompute_counts:
                recompute_counts[key] += int(counts.get(key, 0))
        process_count = len(process_results)
        samples_per_process = len(first.get("samples_ns", []))
        size = int(first["size"])
        expected_intermediate = process_count * samples_per_process * size * 2
        expected_terminal = process_count * samples_per_process * size
        observed_intermediate = recompute_counts["a"] + recompute_counts["b"]
        observed_terminal = recompute_counts["c"]
        metrics.update(
            {
                "source_updates": process_count * samples_per_process,
                "recompute_counts": recompute_counts,
                "expected_intermediate_recomputes": expected_intermediate,
                "observed_intermediate_recomputes": observed_intermediate,
                "expected_terminal_recomputes": expected_terminal,
                "observed_terminal_recomputes": observed_terminal,
                "duplicate_terminal_recomputes": max(
                    0, observed_terminal - expected_terminal
                ),
                "single_recompute_per_diamond": observed_terminal == expected_terminal,
            }
        )

    return {
        "workload": first["workload"],
        "library": first["library"],
        "variant": first["variant"],
        "size": first["size"],
        "unit": first["unit"],
        "correct": all(result["correct"] for result in process_results),
        "skipped": first.get("skipped"),
        "summary_ns": summary,
        "rate_per_sec": rate,
        "metrics": metrics,
    }


def fmt_ns(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}s"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f}us"
    return f"{value:.0f}ns"


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\nEnvironment")
    print(json.dumps(environment(), indent=2, sort_keys=True))
    if results:
        metrics = results[0].get("metrics", {})
        print(
            "\nWorker isolation: "
            f"{metrics.get('process_repetitions', 0)} fresh process(es) per row, "
            f"{metrics.get('samples_per_process', 0)} sample(s) per process"
        )
    print("\nResults")
    header = (
        f"{'Workload':<20} {'Lib':<6} {'Variant':<28} {'Size':>10} "
        f"{'Median':>12} {'p95':>12} {'Rate/sec':>14} {'OK':>4}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.get("skipped"):
            print(
                f"{result['workload']:<20} {result['library']:<6} "
                f"{result['variant']:<28} {result['size']:>10} skipped: {result['skipped']}"
            )
            continue
        summary = result["summary_ns"]
        print(
            f"{result['workload']:<20} {result['library']:<6} "
            f"{result['variant']:<28} {result['size']:>10} "
            f"{fmt_ns(summary['median']):>12} {fmt_ns(summary['p95']):>12} "
            f"{result['rate_per_sec']:>14,.0f} {str(result['correct']):>4}"
        )
        metrics = result.get("metrics", {})
        if result["workload"] == "creation" and metrics:
            print(
                f"{'':<20} {'':<6} bytes/live={metrics.get('bytes_per_live_node', 0):.1f} "
                f"rss_delta={metrics.get('rss_delta', 0)}"
            )
        if "per_update_latency_ns" in metrics:
            lat = metrics["per_update_latency_ns"]
            print(
                f"{'':<20} {'':<6} per-update p50={fmt_ns(lat.get('median', float('nan')))} "
                f"p95={fmt_ns(lat.get('p95', float('nan')))} "
                f"p99={fmt_ns(lat.get('p99', float('nan')))}"
            )
        if result["workload"] == "diamond" and metrics:
            print(
                f"{'':<20} {'':<6} diamonds={metrics.get('diamonds')} "
                f"semantic_nodes={metrics.get('semantic_nodes')} "
                f"runtime_nodes={metrics.get('runtime_nodes')} "
                f"source_updates={metrics.get('source_updates')} "
                f"timed_phase={metrics.get('timed_phase')}"
            )
            print(
                f"{'':<20} {'':<6} terminal_recomputes "
                f"expected={metrics.get('expected_terminal_recomputes')} "
                f"observed={metrics.get('observed_terminal_recomputes')} "
                f"duplicates={metrics.get('duplicate_terminal_recomputes')} "
                f"intermediate_observed={metrics.get('observed_intermediate_recomputes')}"
            )
            print(
                f"{'':<20} {'':<6} single_recompute_per_diamond="
                f"{metrics.get('single_recompute_per_diamond')}"
            )

    comparison_rows = [
        result
        for result in results
        if result["workload"].startswith("compare-") and not result.get("skipped")
    ]
    by_case: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for result in comparison_rows:
        by_case.setdefault((result["workload"], result["size"]), {})[
            result["library"]
        ] = result

    paired = [
        (case, libraries["fynx"], libraries["rxpy"])
        for case, libraries in by_case.items()
        if "fynx" in libraries and "rxpy" in libraries
    ]
    if paired:
        print("\nComparison Ratios")
        ratio_header = (
            f"{'Workload':<20} {'Size':>10} {'FynX':>12} {'RxPY':>12} "
            f"{'RxPY/FynX':>10} {'Winner':>8}"
        )
        print(ratio_header)
        print("-" * len(ratio_header))
        for (workload, size), fynx_result, rxpy_result in paired:
            fynx_median = fynx_result["summary_ns"]["median"]
            rxpy_median = rxpy_result["summary_ns"]["median"]
            ratio = rxpy_median / fynx_median if fynx_median else float("nan")
            winner = "fynx" if ratio >= 1 else "rxpy"
            print(
                f"{workload:<20} {size:>10} {fmt_ns(fynx_median):>12} "
                f"{fmt_ns(rxpy_median):>12} {ratio:>9.2f}x {winner:>8}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run small smoke sizes")
    parser.add_argument(
        "--compare", action="store_true", help="Include optional RxPY comparisons"
    )
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--order",
        choices=["interleaved", "grouped", "random"],
        default="interleaved",
        help="Ordering for paired comparison cases",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --order random")
    parser.add_argument(
        "--gc-mode",
        choices=["normal", "disabled"],
        default="normal",
        help="Disable GC only during timed sections when set to disabled",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=[
            "creation",
            "updates",
            "chain",
            "fanout",
            "diamond",
            "dynamic",
            "compare-map",
            "compare-chain",
            "compare-fanout",
            "compare-accumulate",
        ],
        help="Limit to one or more workloads",
    )
    parser.add_argument("--json", action="store_true", help="Emit aggregate JSON")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        spec = json.loads(args.worker)
        try:
            print(json.dumps(run_worker(spec)))
        except Exception as exc:
            print(json.dumps(failed_worker_result(spec, exc)))
        return

    specs = build_specs(args)
    for spec in specs:
        spec["processes"] = args.processes

    aggregates = []
    for index, spec in enumerate(specs, start=1):
        label = f"{spec['workload']} {spec.get('library', 'fynx')} {spec.get('variant', '')} {spec['size']}"
        print(f"[{index}/{len(specs)}] {label}", file=sys.stderr)
        aggregates.append(aggregate(run_spec_in_process(spec)))

    if args.json:
        print(
            json.dumps({"environment": environment(), "results": aggregates}, indent=2)
        )
    else:
        print_results(aggregates)


if __name__ == "__main__":
    main()
