# FynX

<p align="center">
  <img src="docs/images/banner.svg" alt="FynX Logo">
</p>
<p align="center">
  <a href="https://pypi.org/project/fynx/">
    <img src="https://img.shields.io/pypi/v/fynx.svg?color=4169E1&label=PyPI" alt="PyPI Version">
  </a>
  <a href="https://github.com/off-by-some/fynx/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/off-by-some/fynx/test.yml?branch=main&label=CI&color=2E8B57" alt="Build Status">
  </a>
  <a href="https://codecov.io/github/off-by-some/fynx" >
    <img src="https://codecov.io/github/off-by-some/fynx/graph/badge.svg?token=NX2QHA8V8L"/>
  </a>
    <a href="https://off-by-some.github.io/fynx/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-8A2BE2" alt="Documentation">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/pypi/pyversions/fynx.svg?label=Python&color=1E90FF" alt="Python Versions">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-FF6B35.svg" alt="License: MIT">
  </a>
</p>

<p align="center"><i>FynX ("Finks") = Functional Yielding Observable Networks</i></p>

***

**FynX** eliminates state management complexity in Python applications. Inspired by [MobX](https://github.com/mobxjs/mobx) and functional reactive programming, FynX makes your data reactive with zero boilerplate—just declare relationships once, and watch automatic updates cascade through your entire application.

Stop wrestling with manual state synchronization. Whether you're building real-time [Streamlit](https://streamlit.io/) dashboards, complex data pipelines, or interactive applications, FynX ensures that when one value changes, everything that depends on it updates instantly and predictably. No stale UI. No forgotten dependencies. No synchronization headaches.

**Define relationships once. Updates flow automatically. Your application stays in sync—effortlessly.**

## Table of Contents

- [Quick Start](#quick-start)
- [Where FynX Shines](#where-fynx-shines)
- [Core Concepts](#core-concepts)
  - [Understanding Observables](#understanding-observables)
  - [Transforming Data](#transforming-data)
  - [Combining Observables](#combining-observables)
  - [Filtering with Conditions](#filtering-with-conditions)
  - [Reacting to Changes](#reacting-to-changes)
- [The Reactive Operators](#the-reactive-operators)
- [A Complete Example](#a-complete-example)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [The Mathematical Foundation](#the-mathematical-foundation)
- [Design Philosophy](#design-philosophy)
- [Test Coverage](#test-coverage)
- [API Reference](https://off-by-some.github.io/fynx/)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

Install FynX with a single command:

```bash
pip install fynx
```

Here's reactive state management in 15 lines. Build a shopping cart where the total price updates automatically whenever quantities or prices change:

```python
from fynx import Store, observable

class CartStore(Store):
    item_count = observable(1)
    price_per_item = observable(10.0)

def update_ui(total: float):
    print(f">>> Cart Total: ${total:.2f}")

# Combine observables and compute total price
combined = CartStore.item_count | CartStore.price_per_item
total_price = combined >> (lambda count, price: count * price)
total_price.subscribe(update_ui)

print("=" * 50)

# Change the cart—total updates automatically
CartStore.item_count = 2        # >>> Cart Total: $20.00
CartStore.price_per_item = 15   # >>> Cart Total: $30.00
```

That's the essence of FynX. Define your relationships once, and the library ensures everything stays synchronized. You never write update code again—just describe what should be true, and FynX makes it so.

**Ready to dive deeper?** Continue reading, or jump to [Examples](#examples) to see FynX in real applications.

## Where FynX Shines

FynX excels when data flows through transformations and multiple components need to stay in sync:

- **Streamlit dashboards** where widgets depend on shared state ([see example](./examples/streamlit/todo_app.py))
- **Data pipelines** where computed values must recalculate when inputs change
- **Analytics dashboards** that visualize live, streaming data
- **Complex forms** with interdependent validation rules
- **Real-time applications** where state coordination becomes unwieldy

The library frees you from the tedious work of tracking dependencies and triggering updates. Instead of thinking about *when* to update state, you focus purely on *what* relationships should hold. The rest happens automatically.

## Core Concepts

### Understanding Observables

Observables are reactive values that automatically notify dependents when they change. Create them standalone or organize them in Stores:

```python
from fynx import observable, Store

# Standalone observable
counter = observable(0)
counter.set(1)  # Triggers reactive updates automatically

# Store-based observables (recommended for organization)
class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)

AppState.username = "off-by-some"  # Normal assignment, reactive behind the scenes
```

Stores provide structure for related state and enable powerful features like store-level reactions and serialization.

> **Learn more:** For comprehensive details on all observable types and their features, see the [full documentation](https://off-by-some.github.io/fynx/).

### Transforming Data

Transform observables with the `>>` operator or `computed` decorator. Chain transformations to create data pipelines:

```python
from fynx import computed

# Using >> operator for inline transformations
result = (counter
    >> (lambda x: x * 2)
    >> (lambda x: x + 10)
    >> (lambda x: f"Result: {x}"))

# Using computed decorator for reusable transformations
doubled = computed(lambda x: x * 2, counter)
```

Every transformation creates a new observable that automatically recalculates when its source changes.

### Combining Observables

Combine multiple observables with the `|` operator to create reactive tuples:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

# Combine and transform in one expression
full_name = (User.first_name | User.last_name) >> (lambda first, last: f"{first} {last}")
```

Changes to *any* combined observable trigger automatic recalculation of downstream values.

> **Note:** The `|` operator will transition to `@` in a future release to support OR operations.

### Filtering with Conditions

The `&` operator filters observables so they only emit values when conditions are met. Use `~` to negate conditions:

```python
uploaded_file = observable(None)
is_processing = observable(False)

# Create conditional observables
is_valid = uploaded_file >> (lambda f: f is not None)
preview_ready = uploaded_file & is_valid & (~is_processing)
```

The `preview_ready` observable only has a value when a file exists, it's valid, *and* processing isn't active. All three conditions must align before anything downstream executes—perfect for complex business logic.

### Reacting to Changes

React to changes using decorators, subscriptions, or context managers:

```python
from fynx import reactive, watch

# Decorator for dedicated reaction functions
@reactive(observable)
def handle_change(value):
    print(f"Changed: {value}")

# Subscriptions for inline reactions
observable.subscribe(lambda x: print(f"New value: {x}"))

# Conditional reactions with @watch
@watch(lambda: condition1, lambda: condition2)
def on_conditions_met():
    print("All conditions satisfied!")
```

Choose the pattern that fits your use case—FynX adapts to your style.

## The Reactive Operators

FynX provides four core operators that compose into sophisticated reactive systems:

| Operator | Purpose | Example |
|----------|---------|---------|
| `>>` | **Transform** values through functions | `total_price >> (lambda t: f"${t:.2f}")` |
| `\|` | **Combine** multiple observables into tuples | `(first \| last) >> (lambda f, l: f"{f} {l}")` |
| `&` | **Filter** based on boolean conditions | `file & is_valid & (~is_processing)` |
| `~` | **Negate** conditions | `~is_processing` |

**Each operation creates a new observable that you can chain indefinitely**—transform, combine, and filter with complete freedom. Together, these operators form a complete algebra for reactive data flow.

## A Complete Example

Here's how these pieces fit together in a practical file upload system. Notice how complex reactive logic emerges naturally from simple compositions:

```python
from fynx import Store, observable, reactive

class FileUpload(Store):
    uploaded_file = observable(None)
    is_processing = observable(False)
    progress = observable(0)

# Derive conditions from state
is_valid = FileUpload.uploaded_file >> (lambda f: f is not None)
is_complete = FileUpload.progress >> (lambda p: p >= 100)

# Combine conditions to control when preview shows
ready_for_preview = FileUpload.uploaded_file & is_valid & (~FileUpload.is_processing)

@reactive(ready_for_preview)
def show_file_preview(file):
    print(f"Preview: {file}")

# Watch the reactive graph in action
FileUpload.uploaded_file = "document.pdf"  # Preview: document.pdf

FileUpload.is_processing = True
FileUpload.uploaded_file = "image.jpg"     # No preview (processing active)

FileUpload.is_processing = False           # Preview: image.jpg
```

The preview function triggers automatically, but only when all conditions align. You never manually check whether to show the preview—the reactive graph handles that coordination.

## Advanced Features

FynX supports sophisticated patterns for complex applications:

**Store-level reactions** give you snapshots of all observables whenever anything changes—perfect for logging or persistence:

```python
@reactive(UserProfile)
def on_any_change(snapshot):
    print(f"Profile updated: {snapshot.first_name} {snapshot.last_name}")
```

**State serialization** enables persistence across sessions:

```python
# Save state
state_dict = UserProfile.to_dict()

# Restore state later
UserProfile.load_state(state_dict)
```

**Custom stores** let you integrate with any framework. The Streamlit example shows how to sync with session state automatically.

## Examples

Explore the [`examples/`](./examples/) directory for demonstrations of FynX's capabilities:

| File | Description |
|------|-------------|
| [`basics.py`](./examples/basics.py) | Core FynX concepts: observables, subscriptions, computed properties, stores, reactive decorators, and conditional logic |
| [`cart_checkout.py`](./examples/cart_checkout.py) | Shopping cart with reactive total calculation using merged observables and subscriptions |
| [`advanced_user_profile.py`](./examples/advanced_user_profile.py) | Complex reactive system demonstrating validation, notifications, state persistence, and sophisticated computed properties |
| [`streamlit/store.py`](./examples/streamlit/store.py) | Custom StreamlitStore implementation with automatic session state synchronization |
| [`streamlit/todo_app.py`](./examples/streamlit/todo_app.py) | Complete reactive todo list application with Streamlit UI, showcasing real-time updates and automatic persistence |
| [`streamlit/todo_store.py`](./examples/streamlit/todo_store.py) | Todo list store with computed properties, filtering, and bulk operations |

## The Mathematical Foundation

If you're curious about the theory powering FynX, the core insight is that observables form a functor in the category-theoretic sense. An `Observable<T>` represents a time-varying value—formally, a continuous function $\mathcal{T} \to T$ where $\mathcal{T}$ denotes the temporal domain. This construction naturally forms an endofunctor $\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$ on the category of Python types.

The `>>` operator implements functorial mapping, satisfying the functor laws. For any morphism $f: A \to B$, we get a lifted morphism $\mathcal{O}(f): \mathcal{O}(A) \to \mathcal{O}(B)$, ensuring that:

$$
\begin{align*}
\mathcal{O}(\mathrm{id}_A) &= \mathrm{id}_{\mathcal{O}A} \\
\mathcal{O}(g \circ f) &= \mathcal{O}g \circ \mathcal{O}f
\end{align*}
$$

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

## Test Coverage

FynX maintains comprehensive test coverage tracked through Codecov. Here are visual representations of our current coverage:

| Sunburst Diagram | Grid Diagram | Icicle Diagram |
|---|---|---|
| <img src="https://codecov.io/github/off-by-some/fynx/graphs/sunburst.svg?token=NX2QHA8V8L" alt="Sunburst Coverage Diagram" height="200"/><br>*The inner-most circle represents the entire project, with folders and files radiating outward. Size and color represent statement count and coverage percentage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/tree.svg?token=NX2QHA8V8L" alt="Grid Coverage Diagram" height="200"/><br>*Each block represents a file. Size and color indicate statement count and coverage percentage.* | <img src="https://codecov.io/github/off-by-some/fynx/graphs/icicle.svg?token=NX2QHA8V8L" alt="Icicle Coverage Diagram" height="200"/><br>*The top section represents the entire project, with folders and files below. Size and color represent statement count and coverage percentage.* |

## Contributing

Contributions to FynX are always welcome! This project uses **Poetry** for dependency management and **pytest** for testing.

> To learn more about the vision and goals for version 1.0, see the [**1.0 Product Specification**](./docs/1.0_TODO.md).

### Getting Started

```bash
poetry install --with dev --with test
poetry run pre-commit install
poetry run pytest
```

The pre-commit hooks run automatically on each commit, checking code formatting and style. You can also run them manually across all files with `poetry run pre-commit run --all-files`.

### Testing and Linting

Verify your changes pass all tests and coverage requirements:

```bash
poetry run pytest --cov=fynx
```

Check for linting issues:

```bash
./scripts/lint.sh
```

Automatically fix formatting and import problems:

```bash
./scripts/lint.sh --fix
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch with a descriptive name: `feature/amazing-feature`
3. Make your changes and add comprehensive tests
4. Ensure the test suite passes
5. Submit a pull request with a clear description of what you've changed and why

## License

FynX is licensed under the MIT License. See the [LICENSE](./LICENSE) file for complete details.

---

<p align="center">
  <a href="https://github.com/off-by-some/fynx">
    <img src="https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F-red?style=for-the-badge" alt="Made with ❤️">
  </a>
</p>
