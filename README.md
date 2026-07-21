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
  <a href="https://codecov.io/github/off-by-some/fynx" >
    <img src="https://codecov.io/github/off-by-some/fynx/graph/badge.svg?token=NX2QHA8V8L"/>
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

<p align="center" style=""><i>FynX ( /fɪŋks/ ) = Functional Yielding Observable Networks</i></p>

**FynX** makes state management in Python feel inevitable rather than effortful. Inspired by [MobX](https://github.com/mobxjs/mobx) and functional reactive programming, the library turns your data reactive with minimal ceremony—declare relationships once, and updates cascade automatically through your entire application.

Whether you're building real-time [Streamlit](https://streamlit.io/) dashboards, data pipelines, or interactive applications, FynX ensures that when one value changes, everything depending on it updates instantly. No stale state. No forgotten dependencies. No manual synchronization.

**Define relationships once. Updates flow by necessity.**

## Quick Start

```bash
pip install fynx
```

```python
from fynx import Store, observable

class CartStore(Store):
    item_count = observable(1)
    price_per_item = observable(10.0)

# Define transformation function
def calculate_total(count, price):
    return count * price

# Reactive computation using .then()
total_price = (CartStore.item_count + CartStore.price_per_item).then(calculate_total)
# total_price = (CartStore.item_count + CartStore.price_per_item) >> calculate_total # Equivalent!

def print_total(total):
    print(f"Cart Total: ${total:.2f}")

total_price.subscribe(print_total)

# Automatic updates
CartStore.item_count = 3          # Cart Total: $30.00
CartStore.price_per_item = 12.50  # Cart Total: $37.50
```

This example captures the core promise: declare what should be true, and FynX ensures it remains true. For complete tutorials and patterns, see the [full documentation](https://off-by-some.github.io/fynx/) or explore [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/).

## Where FynX Applies

FynX works wherever values change over time and other computations depend on those changes. The reactive model scales naturally across domains:

* **Streamlit dashboards** with interdependent widgets ([see example](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py))
* **Data pipelines** where downstream computations recalculate when upstream data arrives
* **Analytics systems** visualizing live, streaming data
* **Form validation** with complex interdependencies between fields
* **Real-time applications** where manual state coordination becomes unwieldy
* **ETL processes** with dynamic transformation chains
* **Monitoring systems** reacting to threshold crossings and composite conditions
* **Configuration systems** where derived settings update when base parameters change

The common thread: data flows through transformations, and multiple parts of your system need to stay synchronized. FynX handles the tedious work of tracking dependencies and triggering updates. You focus on *what* relationships should hold; the library ensures they do.

This breadth isn't accidental. The universal properties underlying FynX apply to any scenario involving time-varying values and compositional transformations—which describes a surprisingly large fraction of software.


## The Five Reactive Operators

FynX provides five composable operators that form a complete algebra for reactive programming. You can use either the symbolic operators (`>>`, `+`, `&`, `|`, `~`) or their natural language method equivalents (`.then()`, `.alongside()`, `.requiring()`, `.either()`, `.negate()`):

| Operator | Method | Operation | Purpose | Example |
|----------|--------|-----------|---------|---------|
| `>>` | `.then()` | Transform | Apply functions to values | `price >> (lambda p: f"${p:.2f}")` |
| `+` | `.alongside()` | Combine | Merge observables into read-only tuples | `(first + last) >> join` |
| `&` | `.requiring()` | Filter | Gate reactivity based on conditions | `file & valid & ~processing` |
| `\|` | `.either()` | Logical OR | Combine boolean conditions | `is_error \| is_warning` |
| `~` | `.negate()` | Negate | Invert boolean conditions | `~is_loading` |

Each operation creates a new observable. Chain them to build sophisticated reactive systems from simple parts. These operators correspond to precise mathematical structures—functors, products, pullbacks—that guarantee correct behavior under composition.

## The Mathematical Guarantee

You don't need to understand category theory to use FynX, but it's what makes FynX reliable: the reactive behavior isn't just validated by examples—it's guaranteed by mathematical necessity. Every reactive program you construct will work correctly because FynX is built on universal properties from category theory (detailed in the [**Mathematical Foundations**](https://off-by-some.github.io/fynx/mathematical/mathematical-foundations/)). These aren't abstract concepts for their own sake; they're implementation principles that ensure correctness and enable powerful optimizations.

FynX satisfies specific universal properties from category theory, guaranteeing correctness:
* **Functoriality**: Transformations with `>>` preserve composition. Your chains work exactly as expected, regardless of how you compose them.
* **Products**: Combining observables with `+` creates proper categorical products. No matter how complex your combinations, the structure stays coherent.
* **Pullbacks**: Filtering with `&` constructs mathematical pullbacks. Stack conditions freely—the meaning never changes.


The functoriality property guarantees that lifted functions preserve composition:

$$
\mathcal{O}(\mathrm{id}) = \mathrm{id} \quad \mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f
$$

In practice, this means complex reactive systems composed from simple parts behave predictably under all transformations. You describe what relationships should exist; FynX guarantees they hold.

These same categorical structures enable FynX's automatic optimizer. Composition laws prove `obs >> f >> g >> h` can safely fuse into a single operation. Product properties allow sharing common computations. Pullback semantics let filters combine without changing meaning. The theory doesn't just ensure correctness—it shows exactly which optimizations are safe.

Think of it like an impossibly thorough test suite: one covering not just the cases you wrote, but every case that could theoretically exist. (We also ship with [conventional tests](./tests/), naturally.)

## Performance

While the theory guarantees correctness, implementation determines speed. FynX's benchmark suite uses fixed-size workloads, fresh worker processes, correctness checks outside timed regions, and optional RxPY comparisons for the subset both libraries can express synchronously.

```bash
# Run fixed-size FynX stress tests
python benchmark2.py --quick

# Include RxPY comparisons when reactivex is installed
python benchmark2.py --quick --compare
```

Sample comparison output from `python benchmark2.py --quick --compare` on one Apple Silicon / Python 3.10 run:

```
compare-map          FynX  9.186ms   RxPY 11.501ms   1.25x
compare-chain 1      FynX  1.250us   RxPY 1.500us    1.20x
compare-chain 10     FynX  1.750us   RxPY 3.542us    2.02x
compare-fanout 1     FynX  916ns     RxPY 1.250us    1.36x
compare-fanout 10    FynX  2.208us   RxPY 3.000us    1.36x
compare-accumulate   FynX 10.042ms   RxPY 12.756ms   1.27x
```

The headline comparison is deliberately limited to equivalent synchronous workloads: source-map-sink, transformation chains, fan-out to lightweight sinks, and running accumulation. FynX-only workloads separately measure observable creation, subscribed and unsubscribed updates, chain propagation, fan-out, diamond convergence, and dynamic condition dependency switching.

| Workload | What It Measures |
|----------|------------------|
| **Observable Creation** | Construction time, traced bytes per live observable, RSS delta |
| **Independent Updates** | Setter throughput with no subscribers and with one subscriber |
| **Chain Propagation** | Source mutation to final observer delivery through fixed-depth chains |
| **Fan-out** | One source notifying many lightweight dependents |
| **Diamond Graphs** | Converging-node recomputation counts and duplicate recomputation checks |
| **Dynamic Conditions** | Callable condition dependency switching and stale subscription cleanup |

The benchmark reports exactly what happened, rather than translating lightweight dependents into UI claims. That makes results easier to compare across commits, Python versions, and machines.

The optimizer details are covered in the [**Mathematical Foundations**](https://off-by-some.github.io/fynx/mathematical/mathematical-foundations/) documentation, which explains how FynX achieves this performance through a categorical graph optimizer that automatically applies proven rewrite rules based on functoriality, products, and pullbacks.

## Observables

[Observables](https://off-by-some.github.io/fynx/reference/observable/) form the foundation—reactive values that notify dependents automatically when they change. Create them standalone or organize them into [Stores](https://off-by-some.github.io/fynx/tutorial/stores/):

```python
from fynx import observable, Store

# Standalone observable
counter = observable(0)
counter.set(1)  # Triggers reactive updates

# Store-based observables (recommended for organization)
class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)

AppState.username = "off-by-some"  # Normal assignment, reactive behavior
```

Stores provide structure for related state and enable features like store-level reactions and serialization. With observables established, you compose them using FynX's five fundamental operators.


## Transforming Data with `>>` or `.then()`

The `>>` operator (or `.then()` method) transforms observables through functions. Chain multiple transformations to build [derived observables](https://off-by-some.github.io/fynx/generation/markdown/derived-observables/):

```python
# Define transformation functions
def double(x):
    return x * 2

def add_ten(x):
    return x + 10

def format_result(x):
    return f"Result: {x}"

# Chain transformations using .then()
result_method = (counter
    .then(double)
    .then(add_ten)
    .then(format_result))

# Alternative syntax using >> operator
result_operator = (counter
    >> double
    >> add_ten
    >> format_result)
```

Each transformation creates a new observable that recalculates when its source changes. Transform functions are pure: they receive plain values as arguments and may not read or mutate observables from inside the function. If you need another reactive input, combine it first with `+` or `.alongside()`.

```python
price = observable(100.0)
discount = observable(0.1)

# Good: both reactive inputs are explicit
discounted = (price + discount) >> (lambda p, d: p * (1 - d))

# Error: discount.value is a hidden dependency inside the transform
# discounted = price >> (lambda p: p * (1 - discount.value))
```

This chaining works predictably because `>>` implements functorial mapping—structure preservation under transformation.

## Combining Observables with `+` or `.alongside()`

Use `+` (or `.alongside()`) to combine multiple observables into reactive tuples.
Merged observables are read-only computed observables that derive their value from their source observables:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

# Define transformation function
def join_names(first, last):
    return f"{first} {last}"

# Combine and transform using .then()
full_name_method = (User.first_name + User.last_name).then(join_names)

# Alternative using >> operator
full_name_operator = (User.first_name + User.last_name) >> join_names

# Merged observables are read-only
merged = User.first_name + User.last_name
# merged.set(("Jane", "Smith"))  # Raises ValueError: Computed observables are read-only
```

When any combined observable changes, downstream values recalculate automatically. This operator constructs categorical products, ensuring combination remains symmetric and associative regardless of nesting.

## Filtering with `&`, `.requiring()`, `~`, `.negate()`, `|`, and `.either()`

The `&` operator (or `.requiring()`) filters observables to emit only when [conditions](https://off-by-some.github.io/fynx/generation/markdown/conditionals/) are met. Use `~` (or `.negate()`) to invert, and `|` (or `.either()`) for logical OR conditions:

```python
uploaded_file = observable(None)
is_processing = observable(False)
is_error = observable(False)
is_warning = observable(True)

# Define validation function
def is_valid_file(f):
    return f is not None

# Conditional observables using .then()
is_valid_method = uploaded_file.then(is_valid_file)
is_valid_operator = uploaded_file >> is_valid_file

# Filter using & operator (or .requiring() method)
preview_ready_method = uploaded_file.requiring(is_valid_method).requiring(is_processing.negate())
preview_ready_operator = uploaded_file & is_valid_operator & (~is_processing)

# Logical OR using | operator (or .either() method)
needs_attention = is_error | is_warning
# Alternative: needs_attention = is_error.either(is_warning)
```

The `preview_ready` observable emits only when all conditions align—file exists, it's valid, and processing is inactive. The `needs_attention` observable is a total boolean value: `True` when any error or warning condition is true, and `False` otherwise. Filtering emerges from pullback constructions through `&`; `|` and `~` build boolean conditions that can feed those gates.

## Reacting to Changes

React to observable changes using the [`@reactive`](https://off-by-some.github.io/fynx/generation/markdown/using-reactive/) decorator or subscriptions.

**The fundamental principle**: `@reactive` is for side effects only—UI updates, logging, network calls, and other operations that interact with the outside world. For deriving new values from existing data, use the `>>` operator instead. This separation keeps your reactive system predictable and maintainable.

**Important note on timing**: Reactive functions run immediately when attached to active observables or stores, then run again when dependencies change. If a conditional observable is inactive at setup time, the reaction waits until the condition becomes active.

```python
from fynx import reactive

# GOOD: Side effects with @reactive
@reactive(user_count)
def update_dashboard(count):
    render_ui(f"Users: {count}")  # Side effect: UI update

@reactive(data_stream)
def sync_to_server(data):
    api.post('/sync', data)  # Side effect: network I/O

@reactive(error_log)
def log_errors(error):
    print(f"Error: {error}")  # Side effect: logging

# GOOD: Data transformations with >> operator
doubled = count >> (lambda x: x * 2)  # Pure transformation
formatted = doubled >> (lambda x: f"${x:.2f}")  # Pure transformation

# Inline subscriptions for dynamic behavior
observable.subscribe(lambda x: print(f"New value: {x}"))

# Conditional reactions using boolean operators
is_logged_in = observable(False)
has_data = observable(False)
is_loading = observable(True)

# React only when logged in AND has data AND NOT loading
@reactive(is_logged_in & has_data & ~is_loading)
def sync_when_ready(should_sync):
    if should_sync:
        perform_sync()  # Side effect: network operation

# Multiple observables via derived state
first_name = observable("Alice")
last_name = observable("Smith")

# Derive first, then react
full_name = (first_name + last_name) >> (lambda f, l: f"{f} {l}")

@reactive(full_name)
def update_greeting(name):
    display_message(f"Hello, {name}!")  # Side effect: UI update
```

**Lifecycle management**: Use `.unsubscribe()` to stop reactive behavior when cleaning up components or changing modes. After unsubscribing, the function returns to normal, non-reactive behavior and can be called manually again.

```python
@reactive(data_stream)
def process_data(data):
    handle_data(data)

# Later, during cleanup
process_data.unsubscribe()  # Stops reacting to changes
```

**Remember**: Use `@reactive` for side effects at your application's boundaries—where your pure reactive data flow meets the outside world. Use `>>`, `+`, `&`, `|`, and `~` for data transformations and computations. Inside a `>>` / `.then()` transform, use only the arguments FynX passes to your function; combine extra inputs first with `+` / `.alongside()`. This "functional core, reactive shell" pattern is what makes reactive systems both powerful and maintainable.

## Additional Examples

Explore the [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/) directory for demonstrations across use cases:

| File | Description |
|------|-------------|
| [`basics.py`](https://github.com/off-by-some/fynx/blob/main/examples/basics.py) | Core concepts: observables, subscriptions, computed properties, stores, reactive decorators, conditional logic |
| [`cart_checkout.py`](https://github.com/off-by-some/fynx/blob/main/examples/cart_checkout.py) | Shopping cart with reactive total calculation |
| [`advanced_user_profile.py`](https://github.com/off-by-some/fynx/blob/main/examples/advanced_user_profile.py) | Complex reactive system with validation, notifications, persistence, and sophisticated computed properties |
| [`streamlit/store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/store.py) | Custom StreamlitStore with automatic session state synchronization |
| [`streamlit/todo_app.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py) | Complete reactive todo list with Streamlit UI, real-time updates, and automatic persistence |
| [`streamlit/todo_store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_store.py) | Todo store with computed properties, filtering, and bulk operations |

These examples demonstrate how FynX's composable primitives scale from simple to sophisticated. The consistency across scales follows from the mathematical foundations.

## Design Philosophy

Deep mathematics should enable simpler code, not complicate it. FynX grounds itself in category theory precisely because those abstractions—functors, products, pullbacks—capture the essence of composition without the accidents of implementation. Users benefit from mathematical rigor whether they recognize the theory or not.

The interface reflects this. Observables feel like ordinary values—read them, write them, pass them around. Reactivity works behind the scenes, tracking dependencies through categorical structure without requiring explicit wiring. Method chaining flows naturally: `observable(42).subscribe(print)` reads as plain description, not ceremony. The `>>` operator transforms, `+` combines, `&` filters, `|` creates OR conditions, `~` negates—each produces new observables ready for further composition. Complex reactive systems emerge from simple, reusable pieces.

FynX offers multiple APIs because different contexts call for different styles. Use decorators when conciseness matters, direct calls when you need explicit control, context managers when reactions should be scoped. The library adapts to your preferred way of working.

The library remains framework agnostic by design. FynX has zero dependencies in its core and integrates cleanly with Streamlit, FastAPI, Flask, or any Python environment. Whether you're building web applications, data pipelines, or desktop software, the reactive primitives fit naturally without forcing architectural changes.

One current limitation: FynX operates single-threaded. Async support is planned as the concurrency model matures.

## Test Coverage

FynX maintains comprehensive test coverage tracked through Codecov:

| Sunburst Diagram | Grid Diagram | Icicle Diagram |
|---|---|---|
| <img src="https://codecov.io/github/off-by-some/fynx/graphs/sunburst.svg?token=NX2QHA8V8L" alt="Sunburst Coverage Diagram" height="200"/><br>*Inner circle represents the entire project, radiating outward through folders and files. Size and color indicate statement count and coverage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/tree.svg?token=NX2QHA8V8L" alt="Grid Coverage Diagram" height="200"/><br>*Each block represents a file. Size and color indicate statement count and coverage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/icicle.svg?token=NX2QHA8V8L" alt="Icicle Coverage Diagram" height="200"/><br>*Top section represents the entire project, with folders and files below. Size and color indicate statement count and coverage.* |

## Contributing

Contributions to FynX are welcome. This project uses **Poetry** for dependency management and **pytest** for testing.

> To learn more about the vision for version 1.0, see the [**1.0 Product Specification**](https://github.com/off-by-some/fynx/blob/main/docs/1.0_TODO.md).

### Getting Started

```bash
poetry install --with dev --with test
poetry run pre-commit install
poetry run pytest
```

Pre-commit hooks run automatically on each commit, checking code formatting and style. Run them manually across all files with `poetry run pre-commit run --all-files`.

### Development Workflow

* **Test your changes**: `poetry run pytest --cov=fynx`
* **Check linting**: `./scripts/lint.sh`
* **Auto-fix formatting**: `./scripts/lint.sh --fix`
* **Fork and create feature branch**: `feature/amazing-feature`
* **Add tests and ensure they pass**
* **Submit PR** with clear description of changes

<br>

## 🌟 Love FynX?

Support the evolution of reactive programming by [**starring the repository**](https://github.com/off-by-some/fynx) ⭐

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
