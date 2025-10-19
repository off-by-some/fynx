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

**FynX** eliminates state management complexity in Python applications. Inspired by [MobX](https://github.com/mobxjs/mobx) and functional reactive programming, FynX makes your data reactive with zero boilerplate—just declare relationships once, and watch automatic updates cascade through your entire application.

Stop wrestling with manual state synchronization. Whether you're building real-time [Streamlit](https://streamlit.io/) dashboards, complex data pipelines, or interactive applications, FynX ensures that when one value changes, everything that depends on it updates instantly and predictably. No stale UI. No forgotten dependencies. No synchronization headaches.

**Define relationships once. Updates flow automatically. Your application stays in sync—effortlessly.**

## Quick Start

```bash
pip install fynx
```

```python
from fynx import Store, observable

class CartStore(Store):
    item_count = observable(1)
    price_per_item = observable(10.0)

# Reactive computation
total_price = (CartStore.item_count | CartStore.price_per_item) >> (lambda count, price: count * price)
total_price.subscribe(lambda total: print(f"Cart Total: ${total:.2f}"))

# Automatic updates
CartStore.item_count = 3      # Cart Total: $30.00
CartStore.price_per_item = 12.50  # Cart Total: $37.50
```

For the complete tutorial and advanced examples, see the [full documentation](https://off-by-some.github.io/fynx/) or explore [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/).

## Observables

[Observables](https://off-by-some.github.io/fynx/generation/markdown/observables/) are the heart of FynX—these reactive values automatically notify their dependents when they change. Create them standalone or organize them into [Stores](https://off-by-some.github.io/fynx/generation/markdown/stores/) for better structure:

```python
from fynx import observable, Store

# Standalone observable
counter = observable(0)
counter.set(1)  # Automatically triggers reactive updates

# Store-based observables (recommended)
class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)

AppState.username = "off-by-some"  # Normal assignment, reactive behavior
```

Stores provide organizational structure for related state and unlock powerful features like store-level reactions and serialization.

***

## Transforming Data with `>>`

The `>>` operator transforms observables through functions. Chain multiple transformations to build elegant [derived observables](https://off-by-some.github.io/fynx/generation/markdown/derived-observables/):

```python
from fynx import computed

# Inline transformations with >>
result = (counter
    >> (lambda x: x * 2)
    >> (lambda x: x + 10)
    >> (lambda x: f"Result: {x}"))

# Reusable transformations with computed
doubled = computed(lambda x: x * 2, counter)
```

Each transformation creates a new observable that recalculates automatically when its source changes.

***

## Combining Observables with `|`

Use the `|` operator to combine multiple observables into reactive tuples:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

# Combine and transform in a single expression
full_name = (User.first_name | User.last_name) >> (lambda f, l: f"{f} {l}")
```

When *any* combined observable changes, downstream values recalculate automatically.

> **Note:** The `|` operator will transition to `@` in a future release to support logical OR operations.

***

## Filtering with `&` and `~`

The `&` operator filters observables to emit only when [conditions](https://off-by-some.github.io/fynx/generation/markdown/conditionals/) are met. Use `~` to negate conditions:

```python
uploaded_file = observable(None)
is_processing = observable(False)

# Create conditional observables
is_valid = uploaded_file >> (lambda f: f is not None)
preview_ready = uploaded_file & is_valid & (~is_processing)
```

The `preview_ready` observable only emits when a file exists, it's valid, *and* processing is inactive. All conditions must align before downstream execution—perfect for complex business logic.

***

## Reacting to Changes

React to observable changes using the [`@reactive`](https://off-by-some.github.io/fynx/generation/markdown/using-reactive/) decorator, subscriptions, or the [`@watch`](https://off-by-some.github.io/fynx/generation/markdown/using-watch/) pattern:

```python
from fynx import reactive, watch

# Dedicated reaction functions
@reactive(observable)
def handle_change(value):
    print(f"Changed: {value}")

# Inline reactions with subscriptions
observable.subscribe(lambda x: print(f"New value: {x}"))

# Conditional reactions
condition1 = observable(True)
condition2 = observable(False)

@watch(condition1 & condition2)
def on_conditions_met():
    print("All conditions satisfied!")
```

Choose the pattern that fits your use case—FynX adapts to your preferred style.

***

## The Four Reactive Operators

FynX provides four composable operators that form a complete algebra for reactive programming:

| Operator | Operation | Purpose | Example |
|----------|-----------|---------|---------|
| `>>` | Transform | Apply functions to values | `price >> (lambda p: f"${p:.2f}")` |
| `\|` | Combine | Merge observables into tuples | `(first \| last) >> join` |
| `&` | Filter | Gate based on conditions | `file & valid & ~processing` |
| `~` | Negate | Invert boolean conditions | `~is_loading` |

**Each operation creates a new observable.** Chain them infinitely to build sophisticated reactive systems from simple, composable parts.

***

## Where FynX Shines

FynX excels when data flows through transformations and multiple components need to stay in sync:

* **Streamlit dashboards** where widgets depend on shared state ([see example](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py))
* **Data pipelines** where computed values must recalculate when inputs change
* **Analytics dashboards** that visualize live, streaming data
* **Complex forms** with interdependent validation rules
* **Real-time applications** where state coordination becomes unwieldy

The library frees you from the tedious work of tracking dependencies and triggering updates. Instead of thinking about *when* to update state, you focus purely on *what* relationships should hold. The rest happens automatically.

## Complete Example

Here's how these concepts compose into a practical file upload system. Notice how complex reactive logic emerges naturally from simple building blocks:

```python
from fynx import Store, observable, reactive

class FileUpload(Store):
    uploaded_file = observable(None)
    is_processing = observable(False)
    progress = observable(0)

# Derive conditions from state
is_valid = FileUpload.uploaded_file >> (lambda f: f is not None)
is_complete = FileUpload.progress >> (lambda p: p >= 100)

# Compose conditions to control preview visibility
ready_for_preview = FileUpload.uploaded_file & is_valid & (~FileUpload.is_processing)

@reactive(ready_for_preview)
def show_file_preview(file):
    print(f"Preview: {file}")

# Watch the reactive graph in action
FileUpload.uploaded_file = "document.pdf"  # → Preview: document.pdf

FileUpload.is_processing = True
FileUpload.uploaded_file = "image.jpg"     # → No preview (processing active)

FileUpload.is_processing = False           # → Preview: image.jpg
```

The preview function triggers automatically, but only when all conditions align. You never manually check whether to show the preview—the reactive graph coordinates everything for you.

## Additional Examples

Explore the [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/) directory for demonstrations of FynX's capabilities:

| File | Description |
|------|-------------|
| [`basics.py`](https://github.com/off-by-some/fynx/blob/main/examples/basics.py) | Core FynX concepts: observables, subscriptions, computed properties, stores, reactive decorators, and conditional logic |
| [`cart_checkout.py`](https://github.com/off-by-some/fynx/blob/main/examples/cart_checkout.py) | Shopping cart with reactive total calculation using merged observables and subscriptions |
| [`advanced_user_profile.py`](https://github.com/off-by-some/fynx/blob/main/examples/advanced_user_profile.py) | Complex reactive system demonstrating validation, notifications, state persistence, and sophisticated computed properties |
| [`streamlit/store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/store.py) | Custom StreamlitStore implementation with automatic session state synchronization |
| [`streamlit/todo_app.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py) | Complete reactive todo list application with Streamlit UI, showcasing real-time updates and automatic persistence |
| [`streamlit/todo_store.py`](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_store.py) | Todo list store with computed properties, filtering, and bulk operations |

## The Mathematical Foundation

Time-varying values have structure. When you create an `Observable<T>` in FynX, you're working with something that behaves like a continuous function from time to values—formally, $\mathcal{T} \to T$ where $\mathcal{T}$ represents the temporal domain. This might seem like a simple wrapper, but observables possess a deeper mathematical character: they form what category theorists call an endofunctor $\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$ on Python's type system.

Functors are transformation preservers. If you have a function $f: A \to B$ that transforms regular values, a functor lifts that transformation to work on structured values. The `>>` operator in FynX implements exactly this lifting—it takes ordinary functions and makes them work on observables. This isn't arbitrary cleverness; functors must satisfy two laws that guarantee predictable behavior:

$$\mathcal{O}(\mathrm{id}) = \mathrm{id} \qquad \mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f$$

The first law says that doing nothing to an observable still does nothing—identity maps to identity. The second says that composing two functions and then lifting the result is identical to lifting each function separately and composing them. These laws mean that no matter how you chain transformations with `>>`, the order of operations is preserved exactly as you'd expect from ordinary function composition.

Consider what happens when you combine observables with the `|` operator. Mathematically, you're constructing a Cartesian product in the observable category: $\mathcal{O}(A) \times \mathcal{O}(B) \cong \mathcal{O}(A \times B)$. This isomorphism reveals something elegant—combining two separate time-varying values produces a single time-varying tuple, and these two perspectives are equivalent. The structure you get from pairing observables is the same structure you'd get from observing pairs directly. This property, called the product structure, ensures that combining observables remains symmetric and associative regardless of nesting order.

Filtering introduces a different categorical construction. When you use the `&` operator with a predicate $p: A \to \mathbb{B}$, FynX constructs what's known as a pullback—a universal way of selecting subobjects. The predicate maps values to the boolean domain $\mathbb{B}$, and we're essentially pulling back along the "true" morphism:

$$
\mathcal{O}(A) \xrightarrow{\mathcal{O}(p)} \mathcal{O}(\mathbb{B}) \xrightarrow{\text{true}} \mathbb{B}
$$

Pullbacks guarantee that combining filters with `&` behaves associatively and commutatively. Stack conditions in any order, nest them however you like—the semantics remain consistent because they're derived from a universal construction.

The categorical perspective matters because it provides proofs, not just patterns. When you chain operations in FynX, you're not hoping the library handles edge cases correctly—the mathematics guarantees it must. Functoriality ensures that structure-preserving transformations in your domain remain structure-preserving when lifted to observables. The product and pullback constructions come with universal properties that dictate precisely how composition must behave. There are no special cases to memorize, no gotchas lurking in complex reactive graphs.

Changes propagate through your system transparently because the underlying category theory proves they must propagate correctly. FynX tracks dependencies and manages updates automatically, but that automation isn't heuristic—it follows necessarily from the categorical structure. You write declarative code describing what relationships should hold, and the functor laws, product isomorphisms, and pullback universality ensure those relationships are maintained under all transformations.

This is the power of building on mathematical foundations. The theory isn't window dressing or academic indulgence—it's the reason you can compose observables fearlessly. Category theory gives FynX its correctness guarantees, turning reactive programming from a collection of patterns into a rigorous calculus with laws you can depend on.

## Design Philosophy

FynX embraces a conviction: deep mathematics should enable simpler code, not complicate it. The library grounds itself in category theory precisely because those abstractions—functors, products, pullbacks—capture the essence of composition without the accidents of implementation. Users benefit from mathematical rigor whether they recognize the theory or not.

The interface reflects this philosophy. Observables feel like ordinary values—read them, write them, pass them around. Reactivity works behind the scenes, tracking dependencies through the categorical structure without requiring explicit wiring. Method chaining flows naturally: `observable(42).subscribe(print)` reads as plain description, not ceremony. The `>>` operator transforms, `|` combines, `&` filters—each operation produces new observables ready for further composition. Complex reactive systems emerge from simple, reusable pieces, much as mathematicians build elaborate structures from fundamental morphisms.

FynX offers multiple APIs because different contexts call for different styles. Use decorators when conciseness matters, direct calls when you need explicit control, context managers when reactions should be scoped. The library adapts to your preferred way of working rather than enforcing a single paradigm.

The library remains framework agnostic by design. FynX has zero dependencies in its core and integrates cleanly with Streamlit, FastAPI, Flask, or any Python environment. Whether you're building web applications, data pipelines, or desktop software, the reactive primitives fit naturally into your existing stack without forcing architectural changes.

One current limitation: FynX operates single-threaded. Async support is planned for future releases as the concurrency model matures.

## Test Coverage

FynX maintains comprehensive test coverage tracked through Codecov. Here are visual representations of our current coverage:

| Sunburst Diagram | Grid Diagram | Icicle Diagram |
|---|---|---|
| <img src="https://codecov.io/github/off-by-some/fynx/graphs/sunburst.svg?token=NX2QHA8V8L" alt="Sunburst Coverage Diagram" height="200"/><br>*The inner-most circle represents the entire project, with folders and files radiating outward. Size and color represent statement count and coverage percentage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/tree.svg?token=NX2QHA8V8L" alt="Grid Coverage Diagram" height="200"/><br>*Each block represents a file. Size and color indicate statement count and coverage percentage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/icicle.svg?token=NX2QHA8V8L" alt="Icicle Coverage Diagram" height="200"/><br>*The top section represents the entire project, with folders and files below. Size and color represent statement count and coverage percentage.* |

## Contributing

Contributions to FynX are always welcome! This project uses **Poetry** for dependency management and **pytest** for testing.

> To learn more about the vision and goals for version 1.0, see the [**1.0 Product Specification**](https://github.com/off-by-some/fynx/blob/main/docs/1.0_TODO.md).

### Getting Started

```bash
poetry install --with dev --with test
poetry run pre-commit install
poetry run pytest
```

The pre-commit hooks run automatically on each commit, checking code formatting and style. You can also run them manually across all files with `poetry run pre-commit run --all-files`.

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

<br>
<br>
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
  <em>Architected with ❤️ by <a href="https://github.com/off-by-some">Cassidy Bridges</a></em>
</p>

<p align="center">
  © 2025 Cassidy Bridges • MIT Licensed
</p>
