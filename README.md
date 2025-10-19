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

# Reactive computation
total_price = (CartStore.item_count | CartStore.price_per_item) >> (lambda count, price: count * price)
total_price.subscribe(lambda total: print(f"Cart Total: ${total:.2f}"))

# Automatic updates
CartStore.item_count = 3          # Cart Total: $30.00
CartStore.price_per_item = 12.50  # Cart Total: $37.50
```

This example captures the core promise: declare what should be true, and FynX ensures it remains true. For complete tutorials and patterns, see the [full documentation](https://off-by-some.github.io/fynx/) or explore [`examples/`](https://github.com/off-by-some/fynx/tree/main/examples/).



## Where FynX Applies

FynX works wherever values change over time and other computations depend on those changes. The reactive model scales naturally across domains:

- **Streamlit dashboards** with interdependent widgets ([see example](https://github.com/off-by-some/fynx/blob/main/examples/streamlit/todo_app.py))
- **Data pipelines** where downstream computations recalculate when upstream data arrives
- **Analytics systems** visualizing live, streaming data
- **Form validation** with complex interdependencies between fields
- **Real-time applications** where manual state coordination becomes unwieldy
- **ETL processes** with dynamic transformation chains
- **Monitoring systems** reacting to threshold crossings and composite conditions
- **Configuration systems** where derived settings update when base parameters change

The common thread: data flows through transformations, and multiple parts of your system need to stay synchronized. FynX handles the tedious work of tracking dependencies and triggering updates. You focus on *what* relationships should hold; the library ensures they do.

This breadth isn't accidental. The universal properties underlying FynX apply to any scenario involving time-varying values and compositional transformations—which describes a surprisingly large fraction of software.


## The Mathematical Guarantee

Here's what makes FynX different: the reactive behavior doesn't just work for the examples you see—it works by mathematical necessity for any reactive program you could construct.

FynX satisfies specific universal properties from category theory. These aren't abstractions for their own sake; they're implementation principles that guarantee correctness:

- **Functoriality**: Any function you lift with `>>` preserves composition exactly. Chain transformations freely—the order of operations is guaranteed.
- **Products**: Combining observables with `|` creates proper categorical products. No matter how you nest combinations, the structure remains coherent.
- **Pullbacks**: Filtering with `&` constructs mathematical pullbacks. Stack conditions in any order—the semantics stay consistent.

You don't need to understand category theory to use FynX. The mathematics works beneath the surface, ensuring that complex reactive systems composed from simple parts behave predictably under all transformations. Write declarative code describing relationships, and the universal properties guarantee those relationships hold.

Think of it as a particularly thorough test suite—one that covers not just the cases you thought to write, but every possible case that could theoretically exist (but yes, we’ve got the [“real deal” tests](./tests/test_readme.py) if you want to live dangerously).

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

Stores provide structure for related state and enable features like store-level reactions and serialization. With observables established, you compose them using FynX's four fundamental operators.

## The Four Reactive Operators

FynX provides four composable operators that form a complete algebra for reactive programming:

| Operator | Operation | Purpose | Example |
|----------|-----------|---------|---------|
| `>>` | Transform | Apply functions to values | `price >> (lambda p: f"${p:.2f}")` |
| `\|` | Combine | Merge observables into tuples | `(first \| last) >> join` |
| `&` | Filter | Gate based on conditions | `file & valid & ~processing` |
| `~` | Negate | Invert boolean conditions | `~is_loading` |

Each operation creates a new observable. Chain them to build sophisticated reactive systems from simple parts. These operators correspond to precise mathematical structures—functors, products, pullbacks—that guarantee correct behavior under composition.

## Transforming Data with `>>`

The `>>` operator transforms observables through functions. Chain multiple transformations to build [derived observables](https://off-by-some.github.io/fynx/generation/markdown/derived-observables/):

```python
from fynx import computed

# Inline transformations
result = (counter
    >> (lambda x: x * 2)
    >> (lambda x: x + 10)
    >> (lambda x: f"Result: {x}"))

# Reusable transformations
doubled = computed(lambda x: x * 2, counter)
```

Each transformation creates a new observable that recalculates when its source changes. This chaining works predictably because `>>` implements functorial mapping—structure preservation under transformation.

## Combining Observables with `|`

Use `|` to combine multiple observables into reactive tuples:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

# Combine and transform
full_name = (User.first_name | User.last_name) >> (lambda f, l: f"{f} {l}")
```

When any combined observable changes, downstream values recalculate automatically. This operator constructs categorical products, ensuring combination remains symmetric and associative regardless of nesting.

> **Note:** The `|` operator will transition to `@` in a future release to support logical OR operations.

## Filtering with `&` and `~`

The `&` operator filters observables to emit only when [conditions](https://off-by-some.github.io/fynx/generation/markdown/conditionals/) are met. Use `~` to negate:

```python
uploaded_file = observable(None)
is_processing = observable(False)

# Conditional observables
is_valid = uploaded_file >> (lambda f: f is not None)
preview_ready = uploaded_file & is_valid & (~is_processing)
```

The `preview_ready` observable emits only when all conditions align—file exists, it's valid, and processing is inactive. This filtering emerges from pullback constructions, guaranteeing consistent semantics no matter how you stack conditions.

## Reacting to Changes

React to observable changes using the [`@reactive`](https://off-by-some.github.io/fynx/generation/markdown/using-reactive/) decorator, subscriptions, or the [`@watch`](https://off-by-some.github.io/fynx/generation/markdown/using-watch/) pattern:

```python
from fynx import reactive, watch

# Dedicated reaction functions
@reactive(observable)
def handle_change(value):
    print(f"Changed: {value}")

# Inline reactions
observable.subscribe(lambda x: print(f"New value: {x}"))

# Conditional reactions
condition1 = observable(True)
condition2 = observable(False)

@watch(condition1 & condition2)
def on_conditions_met():
    print("All conditions satisfied!")
```

Choose the pattern that fits your context. These reactions fire automatically because the dependency graph tracks relationships through the categorical structure underlying observables.

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

## The Mathematical Foundation

Time-varying values have structure. When you create an `Observable<T>` in FynX, you're working with something that behaves like a continuous function from time to values—formally, $\mathcal{T} \to T$ where $\mathcal{T}$ represents the temporal domain. Observables possess deeper mathematical character: they form what category theorists call an endofunctor $\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$ on Python's type system.

Functors preserve transformations. If you have a function $f: A \to B$ that transforms regular values, a functor lifts that transformation to work on structured values. The `>>` operator implements this lifting—it takes ordinary functions and makes them work on observables.

```python
# Regular function on values
def double(x): return x * 2
def add_ten(x): return x + 10

value = 5
result = add_ten(double(value))  # 20

# Same composition, lifted to observables
obs = observable(5)
obs_result = obs >> double >> add_ten  # Observable(20)
```

Functors must satisfy two laws that guarantee predictable behavior:

$$\mathcal{O}(\mathrm{id}) = \mathrm{id} \qquad \mathcal{O}(g \circ f) = \mathcal{O}g \circ \mathcal{O}f$$

The first law: doing nothing to an observable still does nothing. The second: composing two functions then lifting is identical to lifting each separately and composing.

```python
# Identity law: O(id) = id
obs = observable(42)
assert (obs >> (lambda x: x)).value == obs.value

# Composition law: O(g ∘ f) = O(g) ∘ O(f)
composed = obs >> (lambda x: add_ten(double(x)))
chained = obs >> double >> add_ten
assert composed.value == chained.value  # Both are 94
```

These laws mean that no matter how you chain transformations with `>>`, the order of operations preserves exactly as you'd expect from ordinary function composition.

When you combine observables with `|`, you're constructing a Cartesian product in the observable category: $\mathcal{O}(A) \times \mathcal{O}(B) \cong \mathcal{O}(A \times B)$. This isomorphism is elegant—combining two separate time-varying values produces a single time-varying tuple, and these perspectives are equivalent.

```python
first_name = observable("Jane")
last_name = observable("Doe")

# Product creates a tuple observable
full_name = (first_name | last_name) >> (lambda f, l: f"{f} {l}")

first_name.set("John")  # full_name automatically becomes "John Doe"
```

The product structure ensures that combining observables remains symmetric and associative regardless of nesting order.

```python
a = observable(1)
b = observable(2)
c = observable(3)

# Associativity: (a | b) | c ≅ a | (b | c)
left_assoc = (a | b) | c   # ((1, 2), 3)
right_assoc = a | (b | c)  # (1, (2, 3))

# Both represent the same product structure, just different tuple nesting
# The mathematical product A×B×C is unique up to isomorphism
```

Filtering introduces pullbacks. When you use `&` with a predicate $p: A \to \mathbb{B}$, FynX constructs a universal way of selecting subobjects. The predicate maps values to the boolean domain $\mathbb{B}$, pulling back along the "true" morphism:

$$
\mathcal{O}(A) \xrightarrow{\mathcal{O}(p)} \mathcal{O}(\mathbb{B}) \xrightarrow{\text{true}} \mathbb{B}
$$

```python
data = observable(42)
is_positive = data >> (lambda x: x > 0)
is_even = data >> (lambda x: x % 2 == 0)

# Pullback: only emits when both conditions hold
filtered = data & is_positive & is_even

data.set(42)   # filtered emits: 42 (positive and even)
data.set(-4)   # filtered doesn't emit (not positive)
data.set(7)    # filtered doesn't emit (not even)
```

Pullbacks guarantee that combining filters with `&` behaves associatively and commutatively. Stack conditions in any order—the semantics remain consistent because they derive from a universal construction.

```python
# Commutativity: a & b ≡ b & a
filter1 = data & is_positive & is_even
filter2 = data & is_even & is_positive

# Both represent the same pullback
data.set(42)
assert filter1.value == filter2.value  # Both emit 42
```

The categorical perspective provides proofs, not just patterns. When you chain operations in FynX, you're not hoping the library handles edge cases correctly—the mathematics guarantees it must.

```python
# Complex composition: all laws hold automatically
price = observable(100.0)
quantity = observable(3)
discount = observable(0.1)
is_valid = quantity >> (lambda q: q > 0)

# Functor laws + product structure + pullback semantics = correct composition
total = ((price | quantity) >> (lambda p, q: p * q)) & is_valid
discounted = total >> (lambda t: t * (1 - discount.value))

quantity.set(5)  # Everything updates correctly by mathematical necessity
```

Functoriality ensures that structure-preserving transformations in your domain remain structure-preserving when lifted to observables. Product and pullback constructions come with universal properties that dictate precisely how composition behaves. No special cases. No hidden gotchas.

Changes propagate through your system correctly because the underlying category theory proves they must. FynX tracks dependencies and manages updates automatically, but that automation isn't heuristic—it follows necessarily from the categorical structure. You write declarative code describing relationships, and the functor laws, product isomorphisms, and pullback universality ensure those relationships maintain under all transformations.

This is the power of building on mathematical foundations. The theory isn't ornamentation—it's why you can compose observables fearlessly. Category theory gives FynX its correctness guarantees, turning reactive programming from a collection of patterns into a rigorous calculus with laws you can depend on.


## Design Philosophy

Deep mathematics should enable simpler code, not complicate it. FynX grounds itself in category theory precisely because those abstractions—functors, products, pullbacks—capture the essence of composition without the accidents of implementation. Users benefit from mathematical rigor whether they recognize the theory or not.

The interface reflects this. Observables feel like ordinary values—read them, write them, pass them around. Reactivity works behind the scenes, tracking dependencies through categorical structure without requiring explicit wiring. Method chaining flows naturally: `observable(42).subscribe(print)` reads as plain description, not ceremony. The `>>` operator transforms, `|` combines, `&` filters—each produces new observables ready for further composition. Complex reactive systems emerge from simple, reusable pieces.

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

---
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

---
