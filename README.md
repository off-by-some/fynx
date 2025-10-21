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

<p align="center" style=""><i>FynX ("Finks") = Functional Yielding Observable Networks</i></p>

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

You don't need to understand category theory at all to use FynX, though it is what makes FynX different: the reactive behavior doesn't just work for the examples you see—it works by mathematical necessity for any reactive program you could construct. FynX is built on solid mathematical foundations from category theory. These aren't abstractions for their own sake—they're implementation principles that guarantee correctness and enable powerful optimizations.

You don't need to understand category theory to use FynX, but it's what makes FynX reliable: the reactive behavior isn't just validated by examples—it's guaranteed by mathematical necessity. Every reactive program you construct will work correctly because FynX is built on universal properties from category theory (detailed in the [**Mathematical Foundations**](https://off-by-some.github.io/fynx/generation/markdown/mathematical-foundations/)). These aren't abstract concepts for their own sake; they're implementation principles that ensure correctness and enable powerful optimizations.

FynX satisfies specific universal properties from category theory, guaranteeing correctness:
* **Functoriality**: Transformations with `>>` preserve composition. Your chains work exactly as expected, regardless of how you compose them.
* **Products**: Combining observables with `+` creates proper categorical products. No matter how complex your combinations, the structure stays coherent.
* **Pullbacks**: Filtering with `&` constructs mathematical pullbacks. Stack conditions freely—the meaning never changes.


The functoriality property guarantees that lifted functions preserve composition:

$$
\mathcal{O}(\mathrm{id}) = \mathrm{id} \quad \mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f
$$

The mathematics works beneath the surface, ensuring that complex reactive systems composed from simple parts behave predictably under all transformations. Write declarative code describing relationships, and the universal properties guarantee those relationships hold.

These same categorical structures also enable FynX's automatic optimizer—composition laws prove that `obs >> f >> g >> h` can safely fuse into a single operation, product properties allow sharing common computations, and pullback semantics let filters combine without changing meaning. The theory doesn't just ensure correctness; it shows exactly which performance optimizations preserve your program's semantics.

Think of it as a particularly thorough test suite—one that covers not just the cases you thought to write, but every possible case that could theoretically exist (but yes, we've got the ["real deal" tests](./tests/test_readme.py) if you want to live dangerously).

## Performance

While the theory guarantees correctness; implementation determines speed. FynX delivers both—and the mathematics directly enables the performance.

```bash
# Run the full benchmark suite
poetry install --with benchmark
poetry run python scripts/benchmark.py
```

Below is a sample of the output from the above command:

```
FynX Benchmark Configuration:
  TIME_LIMIT_SECONDS: 1.0
  STARTING_N: 10
  SCALE_FACTOR: 1.5

╭───────────────────────── 🎯 FynX Benchmarks ────────────────────────────────────╮
│                    FynX Performance Benchmark Suite                             │
╰─────────────────────────────────────────────────────────────────────────────────╯

Running Observable Creation benchmark...
✓ Observable Creation: 793,506 ops/sec (810325 items)
Running Individual Updates benchmark...
✓ Individual Updates: 352,902 ops/sec (360145 items)
Running Chain Propagation benchmark...
✓ Chain Propagation: 1,640 ops/sec (2776 links, 609μs latency)
Running Reactive Fan-out benchmark...
✓ Reactive Fan-out: 47,115 ops/sec (47427 items)

╭─────────────────────── 🎯 Real-World Performance Translation ───────────────────╮
│ ✓ Can handle ~47,427 UI components reacting to single state change              │
│ ✓ Supports component trees up to 2,776 levels deep                              │
│ ✓ Processes 353K+ state updates per second                                      │
│ ✓ Creates 794K+ observable objects per second                                   │
│ ✓ Average propagation latency: 609μs per dependency link                        │
╰──────────────────────────────────────────────────────────────────────────────── ╯
```

The library processes over **353,000 state updates per second** and handles reactive graphs with **47,000+ dependent components** updating from a single source change using nothing but pure Python. This speed isn't accidental—it emerges from a categorical optimizer that rewrites your reactive graphs using proven algebraic transformations.

| Benchmark | Performance | Real-World Translation |
|-----------|-------------|------------------------|
| **Observable Creation** | 794K ops/sec | Create 790,000+ observable objects per second |
| **Individual Updates** | 353K ops/sec | Process 350,000+ state changes per second |
| **Chain Propagation** | 1.6K ops/sec | Support dependency chains 2,776 levels deep |
| **Reactive Fan-out** | 47K ops/sec | Update 47,000+ UI components from single state change |

Latency remains sub-microsecond for individual updates and averages 609μs per dependency link for complex chain propagation. This predictability matters—reactive systems shouldn't stutter when graphs grow large.

The optimizer details are covered in the [**Mathematical Foundations**](https://off-by-some.github.io/fynx/generation/markdown/mathematical-foundations/) documentation, which explains how FynX achieves this performance through a categorical graph optimizer that automatically applies proven rewrite rules based on functoriality, products, and pullbacks.

## Observables

[Observables](https://off-by-some.github.io/fynx/generation/markdown/observables/) form the foundation—reactive values that notify dependents automatically when they change. Create them standalone or organize them into [Stores](https://off-by-some.github.io/fynx/generation/markdown/stores/):

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

Each transformation creates a new observable that recalculates when its source changes. This chaining works predictably because `>>` implements functorial mapping—structure preservation under transformation.

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

The `preview_ready` observable emits only when all conditions align—file exists, it's valid, and processing is inactive. The `needs_attention` observable emits when any error or warning condition is true. This filtering emerges from pullback constructions that create a "smart gate" filtering to the fiber where all conditions are True.

## Reacting to Changes

React to observable changes using the [`@reactive`](https://off-by-some.github.io/fynx/generation/markdown/using-reactive/) decorator or subscriptions.

**The fundamental principle**: `@reactive` is for side effects only—UI updates, logging, network calls, and other operations that interact with the outside world. For deriving new values from existing data, use the `>>` operator instead. This separation keeps your reactive system predictable and maintainable.

**Important note on timing**: Reactive functions don't fire immediately when created—they only fire when their dependencies *change*. This follows from FynX's pullback semantics in category theory. If you need initialization logic, handle it separately before setting up the reaction.

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

**Remember**: Use `@reactive` for side effects at your application's boundaries—where your pure reactive data flow meets the outside world. Use `>>`, `+`, `&`, `|`, and `~` for all data transformations and computations. This "functional core, reactive shell" pattern is what makes reactive systems both powerful and maintainable.

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
  © 2025 Cassidy Bridges • MIT Licensed
</p>

<br>

***
