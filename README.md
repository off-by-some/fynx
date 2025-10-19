# FynX

<p align="center">
  <img src="https://github.com/off-by-some/fynx/raw/main/docs/images/banner.svg" alt="FynX Logo" style="border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); max-width: 100%; height: auto;">
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

<p align="center">
  <a href="#quick-start">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/quick-start.svg" width="180" alt="Quick Start"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://off-by-some.github.io/fynx/">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/read-docs.svg" width="180" alt="Read the Docs"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/off-by-some/fynx/blob/main/examples/">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/code-examples.svg" width="180" alt="Examples"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/off-by-some/fynx/issues">
    <img src="https://raw.githubusercontent.com/off-by-some/fynx/main/docs/images/get-support.svg" width="180" alt="Support"/>
  </a>
</p>

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

If you're curious about the theory powering FynX, the core insight is that observables form a functor in the category-theoretic sense. An `Observable<T>` represents a time-varying value—formally, a continuous function $\mathcal{T} \to T$ where $\mathcal{T}$ denotes the temporal domain. This construction naturally forms an endofunctor $\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$ on the category of Python types.

The `>>` operator implements functorial mapping, satisfying the functor laws. For any morphism $f: A \to B$, we get a lifted morphism $\mathcal{O}(f): \mathcal{O}(A) \to \mathcal{O}(B)$, ensuring that:

$$\mathcal{O}(\mathrm{id}) = \mathrm{id} \qquad \mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f$$

The `|` operator constructs Cartesian products in the observable category, giving us $\mathcal{O}(A) \times \mathcal{O}(B) \cong \mathcal{O}(A \times B)$. This isomorphism means combining observables is equivalent to observing tuples—a property that ensures composition remains well-behaved.

The `&` operator forms filtered subobjects through pullbacks. For a predicate $p: A \to \mathbb{B}$ (where $\mathbb{B}$ is the boolean domain), we construct a monomorphism representing the subset where $p$ holds true:

$$
\mathcal{O}(A) \xrightarrow{\mathcal{O}(p)} \mathcal{O}(\mathbb{B}) \xrightarrow{\text{true}} \mathbb{B}
$$

This isn't merely academic terminology. These mathematical properties guarantee that reactive graphs compose predictably through universal constructions. Functoriality ensures transformations preserve structure: if $f$ and $g$ compose in the base category, their lifted versions $\mathcal{O}(f)$ and $\mathcal{O}(g)$ compose identically in the observable category. The pullback construction for filtering ensures that combining filters behaves associatively and commutatively—no matter how you nest your conditions with `&`, the semantics remain consistent.

Category theory provides formal proof that FynX's behavior is correct and composable. The functor laws guarantee that chaining transformations never produces unexpected behavior. The product structure ensures that combining observables remains symmetric and associative. These aren't implementation details—they're mathematical guarantees that follow from the categorical structure itself.

**The practical benefit?** Changes flow through your reactive graph transparently because the mathematics proves they must. FynX handles all dependency tracking and propagation automatically, and the categorical foundation ensures there are no edge cases or surprising interactions. You describe what you want declaratively, and the underlying mathematics—specifically the universal properties of functors, products, and pullbacks—ensures it behaves correctly in all circumstances.

## Design Philosophy

FynX embodies a simple principle: **mathematical rigor shouldn't compromise usability**. The library builds on category theory but exposes that power through Pythonic interfaces. Observables behave like normal values—you read and write them naturally—while reactivity happens behind the scenes. Method chaining flows intuitively: `observable(42).subscribe(print)` reads like plain English.

**Composability** runs through every aspect of the design. Transform with `>>`, combine with `|`, filter with `&`. Each operation produces new observables that you can transform further. Complex reactive systems emerge from simple, reusable pieces. This compositional approach mirrors how mathematicians think about functions and morphisms, but you don't need to know category theory to benefit from its guarantees.

**Multiple APIs for multiple contexts.** FynX offers decorators for convenience, direct calls for control, and context managers for scoped reactions. The library adapts to your style rather than forcing one approach.

**Framework agnostic.** FynX works with Streamlit, FastAPI, Flask, or any Python framework. The core library has zero dependencies and integrates cleanly with existing tools. Whether you're building web applications, data pipelines, or desktop software, FynX fits naturally into your stack.

> **Note:** FynX is currently single-threaded; async support is planned for future releases.

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
