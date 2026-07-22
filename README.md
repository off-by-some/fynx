# FynX

<p align="center">
  <img src="https://github.com/off-by-some/fynx/raw/main/docs/images/banner.svg" alt="FynX Logo" style="border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); max-width: 100%; height: auto;">
</p>

<p align="center">
  <a href="#quick-start" style="text-decoration: none;">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/quick-start.svg" width="180" alt="Quick Start"/>
  </a>
  <a style="display: inline-block; width: 20px;"></a>
  <a href="https://off-by-some.github.io/fynx/" style="text-decoration: none;">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/read-docs.svg" width="180" alt="Read the Docs"/>
  </a>
  <a style="display: inline-block; width: 20px;"></a>
  <a href="https://github.com/off-by-some/fynx/blob/main/examples/" style="text-decoration: none;">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/code-examples.svg" width="180" alt="Examples"/>
  </a>
  <a style="display: inline-block; width: 20px;"></a>
  <a href="https://github.com/off-by-some/fynx/issues" style="text-decoration: none;">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/get-support.svg" width="180" alt="Support"/>
  </a>
</p>

<p align="center" style="margin-bottom: 0">
  <a href="https://pypi.org/project/fynx/">
    <img src="https://img.shields.io/pypi/v/fynx.svg?color=4169E1&label=PyPI" alt="PyPI Version">
  </a>
  <a href="https://github.com/off-by-some/fynx/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/off-by-some/fynx/test.yml?branch=main&label=CI&color=2E8B57" alt="Build Status">
  </a>
  <a href="https://codecov.io/github/off-by-some/fynx">
    <img src="https://codecov.io/github/off-by-some/fynx/graph/badge.svg?token=NX2QHA8V8L" alt="Coverage status">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-FF6B35.svg" alt="License: MIT">
  </a>
  <a href="https://off-by-some.github.io/fynx/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-8A2BE2" alt="Documentation">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/pypi/pyversions/fynx.svg?label=Python&color=1E90FF" alt="Python Versions">
  </a>
</p>

<p align="center"><strong>Reactive incremental computation for Python</strong></p>
<p align="center"><i>FynX (/fɪŋks/) = Functional Yielding Observable Networks</i></p>

**FynX** is a reactive incremental-computation library for Python, designed for application state and any other problem where results must remain synchronized with inputs that change over time. You describe a graph of changing values and the computations derived from them; when an input changes, FynX recomputes the affected part of the graph, keeps unobserved work lazy, and notifies subscribers where a result meets the outside world.

Inspired by [MobX](https://github.com/mobxjs/mobx) and functional reactive programming, FynX works with ordinary Python functions and has zero core dependencies. It can be introduced around a single computation without requiring the rest of the project to adopt a new architecture.

## Quick Start

```bash
pip install fynx
```

```python
from fynx import observable

item_count = observable(1)
price_per_item = observable(10.0)


def calculate_total(count, price):
    return count * price


total_price = (
    item_count + price_per_item
).then(calculate_total)


def print_total(total):
    print(f"Cart Total: ${total:.2f}")


total_price.subscribe(print_total)

item_count.set(3)          # Cart Total: $30.00
price_per_item.set(12.50)  # Cart Total: $37.50
```

This cart is deliberately small, but it illustrates the problem FynX is built to solve.

State rarely becomes difficult all at once. It usually arrives one harmless variable at a time. A count contributes to a total; the total controls a button; the button starts work that changes several values elsewhere. Each relationship is simple while the chain remains visible. Trouble begins when its links become scattered across setters, callbacks, effects, and update handlers. Changing one value then means remembering every part of the program that learned to depend on it.

FynX keeps those relationships in the computations themselves. In the example above, `item_count` and `price_per_item` are changing inputs, while `total_price` records the computation that combines them. Neither input needs to know that a total exists, because FynX can reach it by following the dependencies already present in the graph.

That's the pattern in miniature: derived values name their inputs, conditions can be composed and reused, and reactions are reserved for I/O and other work outside the graph. Application state is one natural use for this model, but the same shape of problem appears in data pipelines, validation, monitoring, configuration, and anywhere a result should continue to follow inputs that change over time.

The operator spelling is equivalent:

```python
total_price = (
    item_count + price_per_item
) >> calculate_total
```

At this size, the graph saves little more than a handwritten recalculation. Its value becomes clearer when the same inputs feed several results, when a derived value is consumed in another module, or when a larger pipeline should reconsider only the work downstream of what changed.

The goal is simple: six months later, the relationship should still be visible on the page rather than hidden in the order callbacks happen to run.

## Why Write the Dependencies Down?

Many reactive systems discover dependencies by recording which values are read while a computation runs. That can be pleasantly terse, especially while the computation is local and synchronous, but it also means that the graph depends on details of execution: which helper was called, which branch ran, or whether a read happened before later work began.

FynX takes the more literal approach of treating a transform's arguments as its dependency list. A discounted price that depends on both `price` and `discount` therefore brings both values into the expression:

```python
price = observable(100.0)
discount = observable(0.1)

# FynX rejects this hidden dependency with TransformPurityError.
# discounted = price >> (
#     lambda current_price:
#         current_price * (1 - discount.value)
# )

discounted = (price + discount) >> (
    lambda current_price, current_discount:
        current_price * (1 - current_discount)
)
```

The second expression asks for a little more syntax, although it also survives more kinds of change. A reader can see both inputs beside the value they define, the type checker can follow the same path, and extracting part of the calculation into a helper does not quietly add or remove an edge. When a transform reaches sideways into another observable, FynX raises `TransformPurityError` rather than allowing the written graph and the running graph to disagree.

Once dependencies are written this way, the next problem is mostly one of vocabulary: how do we turn individual values into a graph without introducing a different construct for every shape the graph might take?

## From Values to a Graph

An [observable](https://off-by-some.github.io/fynx/tutorial/observables/) is the smallest part of the graph: a changing input with a current value. It can remain standalone, as in the cart above, or move into a [Store](https://off-by-some.github.io/fynx/tutorial/stores/) when several inputs need a shared namespace, store-level reactions, or serialization. That changes how the values are organized, but not how computations are built from them:

```python
from fynx import Store, observable

counter = observable(0)
counter.set(1)


class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)


AppState.username = "off-by-some"
```

Whether an observable lives alone or inside a store, FynX connects it with the same six operators:

| Operator | Method | What it does | Example |
|---|---|---|---|
| `>>` | [`.then()`](https://off-by-some.github.io/fynx/tutorial/derived-observables/) | Transform a value | `price >> format_price` |
| `+` | [`.alongside()`](https://off-by-some.github.io/fynx/tutorial/derived-observables/) | Combine values into a tuple | `(first + last) >> join_name` |
| `&` | [`.all()`](https://off-by-some.github.io/fynx/tutorial/conditionals/) | Combine boolean conditions with AND | `authenticated & connected` |
| `\|` | [`.either()`](https://off-by-some.github.io/fynx/tutorial/conditionals/) | Combine boolean conditions with OR | `is_error \| is_warning` |
| `~` | [`.negate()`](https://off-by-some.github.io/fynx/tutorial/conditionals/) | Invert a boolean condition | `~is_loading` |
| `@` | [`.requiring()`](https://off-by-some.github.io/fynx/tutorial/conditionals/) | Pass a value while a condition holds | `file @ (valid & ~processing)` |

The symbolic and method forms build the same graph, so the choice can follow the surrounding code:

```python
full_name = (first_name + last_name) >> join_name
ready = authenticated & connected & ~is_loading

full_name = first_name.alongside(last_name).then(join_name)
ready = authenticated.all(connected).all(is_loading.negate())
```

Transforms remain ordinary Python functions. They can be tested on plain values, then lifted into a changing computation by placing them after `>>` or `.then()`:

```python
def double(value):
    return value * 2


def add_ten(value):
    return value + 10


def format_result(value):
    return f"Result: {value}"


result = counter.then(double).then(add_ten).then(format_result)

# Equivalent:
# result = counter >> double >> add_ten >> format_result
```

A function of one value gives us a chain. When a calculation needs several values at once, `+` gathers their current values and passes them to the next transform as separate arguments:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")


def join_names(first, last):
    return f"{first} {last}"


full_name = (User.first_name + User.last_name) >> join_names

summary = (a + b + c) >> (
    lambda first, second, third:
        make_summary(first, second, third)
)
```

Products stay flat as they grow, so `a + b + c` arrives as three arguments rather than `((a, b), c)`. They are also read-only, since their value is determined by the sources above them:

```python
merged = User.first_name + User.last_name

# Raises ValueError: computed observables are read-only.
# merged.set(("Jane", "Smith"))
```

Transforms and products describe values that are always available whenever their sources are. Applications eventually need a second kind of relationship, because a file may exist without being ready to preview, a form may contain values without being valid, and a payload may be current without being safe to send.

## When a Value Should Be Available

FynX represents those decisions as observable booleans. Since a condition is itself a value, it can be named once, combined with other conditions, displayed in a UI, or reused to control several parts of the graph:

```python
is_logged_in = observable(False)
has_data = observable(False)
is_loading = observable(True)

ready_to_sync = is_logged_in & has_data & ~is_loading
```

Sometimes the boolean is the result we need. At other times, the useful result is still the original value, provided that the condition currently holds. The `@` operator creates that gate:

```python
uploaded_file = observable(None)
is_processing = observable(False)
is_error = observable(False)
is_warning = observable(True)


def is_valid_file(file):
    return file is not None


is_valid = uploaded_file >> is_valid_file
ready = is_valid & ~is_processing

preview_ready = uploaded_file @ ready
needs_attention = is_error | is_warning
```

While the file is missing, invalid, or still being processed, `preview_ready` is inactive. Once `ready` becomes true, the uploaded file passes through without being converted into a boolean, which is why gating preserves the source type.

At this point the graph can express what a value is and whether it is currently available. It still has not sent a request, redrawn a dashboard, or written a line to a log. Those actions belong somewhere else, because allowing them to masquerade as derivations would put timing and mutation back into the relationships we have just made explicit.

## Where Effects Belong

FynX uses [`@reactive`](https://off-by-some.github.io/fynx/tutorial/using-reactive/) and `.subscribe()` at the places where a value leaves the graph and affects the rest of the program:

```python
from fynx import reactive


@reactive(user_count)
def update_dashboard(count):
    render_ui(f"Users: {count}")


@reactive(data_stream)
def sync_to_server(data):
    api.post("/sync", data)


@reactive(error_log)
def log_errors(error):
    print(f"Error: {error}")
```

The values reaching those effects may already have passed through several transforms, and a reusable condition can serve as a reaction source in the same way as any other observable:

```python
doubled = user_count >> (lambda count: count * 2)
formatted = doubled >> (lambda count: f"{count:,} users")

formatted.subscribe(
    lambda value: print(f"New value: {value}")
)


@reactive(ready_to_sync)
def sync_when_ready(should_sync):
    if should_sync:
        perform_sync()
```

That split between `.then()`-style derivation and `@reactive`-style consumption is worth keeping even under pressure to shortcut it. If an effect computes a value and writes it back into reactive state, part of the dependency graph disappears into imperative code, where a later effect can depend on the order in which callbacks happen to run. In FynX, transforms derive values and effects consume them.

A reaction attached to an active observable or store runs immediately, then runs again as its source changes. An inactive gate postpones that first call until it opens. During cleanup, `.unsubscribe()` stops the automatic calls without turning the decorated function into a special object that cannot be used normally:

```python
@reactive(data_stream)
def process_data(data):
    handle_data(data)


process_data.unsubscribe()
process_data(data)
```

This division between derivation and effects is useful to readers, but it also gives the type checker and runtime a stable structure to work with. Neither has to infer which reads were dependencies or whether an intermediate callback was secretly producing more state. [Best Practices](https://off-by-some.github.io/fynx/tutorial/best-practices/#anti-patterns-to-avoid) collects the common ways that split gets blurred in practice, and how to avoid it.

## What the Runtime Can Rely On

That same stable structure carries into the type system. FynX ships inline type information through `py.typed`, allowing the shape of a graph to remain visible as it is built:

```python
from fynx import Observable

height: Observable[float] = Observable("height", 1.8)
weight: Observable[float] = Observable("weight", 75.0)

bmi_data = height + weight
# MergedObservable[float, float]
# value type: tuple[float, float]

bmi = bmi_data >> calculate_bmi
# Observable[float]
```

The annotations follow the runtime closely. Products remain flat and arrive as unpacked arguments, `&`, `|`, and `~` produce `Observable[bool]`, and `@` keeps the type of the value passing through it. Store fields are [typed descriptors](https://off-by-some.github.io/fynx/reference/observable-descriptors/) as well, so `total = observable(0)` appears as `ObservableValue[int]` on class access while the same call outside a store produces `Observable[int]`.

The same explicit structure allows FynX to make internal changes without changing the public result. Pure transform chains can be fused, repeated ordered products can reuse a live product node, and unobserved derived values can stay lazy until they are read. FynX is designed around explicit algebraic laws and tested against them: functors, products, Boolean algebra, and pullback-like gates supply the blueprint, while the implementation and tests are what keep the mutable Python runtime honest.

For transforms, the relevant laws are:

$$
\mathcal{O}(\mathrm{id}) = \mathrm{id}
\qquad
\mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f
$$

Propagation and caching follow the same kind of contract. Subscriptions mark the parts of the graph that must stay eager for notifications, while version-based invalidation lets the rest recompute on demand. FynX currently uses a synchronous, single-threaded propagation model; async support is being left until its ordering and concurrency rules can be stated as deliberately as the rest of the API.

[Mathematical Foundations](https://off-by-some.github.io/fynx/mathematical/mathematical-foundations/) develops these contracts in detail. Once the runtime is allowed to fuse, share, cache, and order work this way, however, there is still a practical question left: whether the machinery saved around each function is enough to matter in actual measurements.

## What It Costs

The runtime choices above are meant to remove work around the user's functions: fewer separately notified nodes in a fused chain, less repeated work when products can be shared, and no eager upkeep for values nobody is observing. The benchmark suite checks whether those savings survive contact with the implementation, while also measuring the places where FynX still has to do unavoidable work, such as notifying every dependent in a fan-out.

A full comparison run uses five fresh worker processes for each row and collects fifteen samples from each process. Correctness checks run outside the timed region, and the comparison rows are limited to synchronous workloads that FynX and RxPY can both express without changing the problem:

```bash
# Faster development run
./scripts/benchmark --quick

# Full suite, including comparable RxPY rows
./scripts/benchmark --compare
```

The run below used Python 3.10.19 on Apple Silicon under macOS 26.1. Its results are mixed in a useful way. Simple map dispatch is nearly even, and RxPY remains a little faster at running accumulation. A one-step chain is also slightly cheaper in RxPY. As chains deepen or one source fans out to more sinks, FynX begins to spend less time in the reactive machinery around the functions:

| Workload | Size | FynX median | RxPY median | RxPY / FynX |
|---|---:|---:|---:|---:|
| Source → map → sink | 10K events | 12.021 ms | 12.126 ms | 1.01x |
| Source → map → sink | 1M events | 1.179 s | 1.217 s | 1.03x |
| Transform chain | 1 map | 1.62 µs | 1.50 µs | 0.92x |
| Transform chain | 10 maps | 2.04 µs | 3.67 µs | 1.80x |
| Transform chain | 100 maps | 6.00 µs | 26.5 µs | 4.42x |
| Fan-out | 1 sink | 1.04 µs | 1.29 µs | 1.24x |
| Fan-out | 100 sinks | 11.2 µs | 21.3 µs | 1.91x |
| Fan-out | 10K sinks | 981 µs | 2.174 ms | 2.22x |
| Running accumulation | 1M events | 1.399 s | 1.345 s | 0.96x |

A ratio above `1.00x` means the FynX row completed sooner. At a chain depth of 1,000, FynX completed in 50.9 µs while the RxPY worker hit Python's recursion limit; that row was skipped rather than converted into a timing claim.

The comparison rows cover only the overlap between the libraries. FynX's own suite continues into construction, subscription cost, deep propagation, converging graphs, and conditions whose dependencies change at runtime:

| Workload | Size | Median | What the run showed |
|---|---:|---:|---|
| Observable creation | 100K observables | 651.174 ms | 154K creations/s; about 1,780 live bytes per observable |
| Updates, no subscribers | 100K updates | 66.204 ms | 1.5M updates/s; sampled p50 latency 667 ns |
| Updates, one subscriber | 100K updates | 89.281 ms | 1.1M updates/s; sampled p50 latency 917 ns |
| Fused chain propagation | depth 10K | 522 µs | final value delivered at about 2K source updates/s |
| Derived fan-out | 10K dependents | 6.794 ms | about 1.5M dependent deliveries/s |
| Diamond convergence | 1K diamonds | 23.615 ms | 75,000 expected terminal recomputes, 75,000 observed, 0 duplicates |
| Dynamic conditions | 10K branch changes | 106.664 ms | about 94K dependency switches/s with stale-edge cleanup |

The memory measurement is intentionally a live-object measurement rather than an allocation trace that disappears at the end of a loop. At 100,000 observables, the process reported an RSS increase of 318.08 MiB alongside roughly 1,780 traced bytes per surviving observable. The diamond audit is similarly concerned with behavior rather than speed alone: across 1,000 diamonds and 75 source updates, every terminal recomputed exactly once per update, with no duplicate convergence work.

None of the rows renders a dashboard or validates a form; they measure the machinery those applications sit on, which is where the examples become more useful.

## From the Pieces to an Application

The [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/) directory begins with individual observables and operators, then works toward programs where several parts of the graph are active at once:

| File | What it shows |
|---|---|
| [`basics.py`](https://github.com/off-by-some/fynx/blob/main/examples/basics.py) | observables, subscriptions, stores, reactions, and conditions |
| [`cart_checkout.py`](https://github.com/off-by-some/fynx/blob/main/examples/cart_checkout.py) | a shopping cart with a reactive total |
| [`advanced_user_profile.py`](https://github.com/off-by-some/fynx/blob/main/examples/advanced_user_profile.py) | validation, notifications, persistence, and derived state |
| [`streamlit/store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/store.py) | a `StreamlitStore` synchronized with session state |
| [`streamlit/todo_app.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py) | a complete reactive Streamlit todo app |
| [`streamlit/todo_store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_store.py) | filtering, computed values, and bulk todo operations |

Because FynX does not need to own the application around it, the same graph can sit inside Streamlit, FastAPI, Flask, or a plain script. The [documentation](https://off-by-some.github.io/fynx/) starts with [tutorials](https://off-by-some.github.io/fynx/tutorial/observables/) and continues into the [API reference](https://off-by-some.github.io/fynx/reference/api/) and [mathematical details](https://off-by-some.github.io/fynx/mathematical/mathematical-foundations/), so readers can stop at whichever layer answers the question that brought them there.

## Project Health and Contributing

The graph laws and runtime behavior described above are covered by the test suite, with coverage published through Codecov in several views:

| Sunburst | Grid | Icicle |
|---|---|---|
| <img src="https://codecov.io/github/off-by-some/fynx/graphs/sunburst.svg?token=NX2QHA8V8L" alt="Sunburst coverage diagram" height="200"/><br>*Coverage arranged from the project outward through folders and files.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/tree.svg?token=NX2QHA8V8L" alt="Grid coverage diagram" height="200"/><br>*Files shown as blocks sized by statement count.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/icicle.svg?token=NX2QHA8V8L" alt="Icicle coverage diagram" height="200"/><br>*Project, folders, and files shown in descending layers.* |

FynX uses **Poetry** for dependency management and **pytest** for testing. A fresh checkout can be prepared and checked with:

```bash
poetry install --with dev --with test
poetry run pre-commit install
poetry run pytest

poetry run pre-commit run --all-files
poetry run pytest --cov=fynx
./scripts/lint.sh
./scripts/lint.sh --fix
```

From there, contributions follow the usual fork-and-branch workflow. A focused branch such as `feature/amazing-feature` should include the tests that describe its change, along with a pull request explaining what the new behavior adds and how it fits the contracts already described.

<br>

## Found It Useful?

A [star on the repository](https://github.com/off-by-some/fynx) helps other Python developers find FynX. ⭐

***

<br>

<p align="center">
  <strong>FynX</strong> — Functional Yielding Observable Networks
</p>

<p align="center">
  <a href="https://github.com/off-by-some/fynx/blob/main/LICENSE">License</a> •
  <a href="https://github.com/off-by-some/fynx/blob/main/CONTRIBUTING.md">Contributing</a> •
  <a href="https://github.com/off-by-some/fynx/blob/main/CODE_OF_CONDUCT.md">Code of Conduct</a>
</p>

<p align="center">
  <em>Crafted with ❤️ by <a href="https://github.com/off-by-some">Cassidy Bridges</a></em>
</p>

<p align="center">
  © 2025-2026 Cassidy Bridges • MIT Licensed
</p>

<br>

***
