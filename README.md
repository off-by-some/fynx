# Fynx

<p align="center">
  <img src="https://github.com/off-by-some/fynx/raw/main/docs/fynx.png" alt="Fynx Logo" height="400px">
</p>

<p align="center">
  <a href="https://github.com/off-by-some/fynx/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/off-by-some/fynx/test.yml?branch=main&label=CI" alt="Build Status">
  </a>
  <a href="https://codecov.io/github/off-by-some/fynx" >
    <img src="https://codecov.io/github/off-by-some/fynx/graph/badge.svg?token=NX2QHA8V8L"/>
  </a>
  <a href="https://pypi.org/project/fynx/">
    <img src="https://img.shields.io/pypi/v/fynx.svg?color=4b8bbe&label=PyPI" alt="PyPI Version">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/pypi/pyversions/fynx.svg?label=Python&color=3776AB" alt="Python Versions">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

---

**Fynx** is a lightweight reactive state management library for Python that makes your data flow effortlessly. Inspired by [MobX](https://github.com/mobxjs/mobx) and similar [functional reactive programming](https://wiki.haskell.org/Functional_Reactive_Programming) frameworks, Fynx transforms plain Python objects into reactive observables that update automatically. You define how your data relates—Fynx handles the synchronization with zero boilerplate.

Whether you're building interactive Streamlit dashboards or crafting data-driven UIs, Fynx ensures that changes propagate smoothly, predictably, and elegantly through your application. When one value changes, everything that depends on it updates automatically. No manual wiring, no stale state, no hassle.

## Getting Started

Install Fynx with a single command:

```bash
pip install fynx
```

Here's what reactive state management looks like in practice.

For example, imagine you're building a shopping cart and want to display the total price. With Fynx, the calculation happens automatically whenever the cart contents change:

```python
from fynx import Store, observable

# Define a store for a shopping cart
class CartStore(Store):
    item_count = observable(1)
    price_per_item = observable(10.0)

def update_ui(total: float):
    print(f">>> Cart Total: ${total:.2f}")

# Link item_count and price_per_item to auto-calculate total_price
combined_observables = CartStore.item_count | CartStore.price_per_item

# The >> operator takes any observable and passes the value(s) to the right.
total_price = combined_observables >> (lambda count, price: count * price)
total_price.subscribe(update_ui) # Subscribe and update the UI when it changes

print("=" * 50)

# Now whenever we change the cart state, total_price updates automatically,
# and the UI is updated accordingly.
CartStore.item_count = 2
CartStore.price_per_item = 15

# ==================================================
# >>> Cart Total: $20.00
# >>> Cart Total: $30.00
```
That's the essence of Fynx. Define your relationships once, and the library ensures everything stays synchronized. You never write update code again—just describe what should be true, and Fynx makes it so.

## Where Fynx Shines

Fynx excels in scenarios where data flows through transformations and multiple components need to stay in sync. Consider Streamlit applications where widgets depend on shared state, or data pipelines where computed values must recalculate when their inputs change. Analytics dashboards that visualize live data, forms with interdependent validation rules, or any system where state coordination becomes complex—these are Fynx's natural habitat.

The library frees you from the tedious work of tracking dependencies and triggering updates. Instead of thinking about *when* to update state, you focus purely on *what* relationships should hold. The rest happens automatically.

## Understanding Observables

At the heart of Fynx lies the [observable](https://medium.com/@davidjtomczyk/introduction-to-observable-85a5122bf260): a value that changes over time and notifies interested parties when it does. You can create standalone observables or organize them into Stores for related state:

```python
from fynx import observable, Store

counter = observable(0)

# You can set standalone observable values with .set(),
counter.set(1)

class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)

# Stores allow us to treat them as normal values
AppState.username = "off-by-some"
```

Observables work like normal Python values—you read and write them naturally—but they carry reactive superpowers beneath the surface.

## Transforming Data

The `>>` operator lets you transform observables declaratively. Each transformation creates a new derived observable that automatically recalculates when its source changes:

```python
doubled = counter >> (lambda x: x * 2)

result = (counter
    >> (lambda x: x * 2)
    >> (lambda x: x + 10)
    >> (lambda x: f"Result: {x}"))
```

If syntactic sugar isn't quite your thing, you can use `computed` instead:

```python
from fynx import computed

# Simple transformation
doubled = computed(lambda x: x * 2, counter)

# Chained transformations (step-by-step)
step1 = computed(lambda x: x * 2, counter)
step2 = computed(lambda x: x + 10, step1)
result = computed(lambda x: f"Result: {x}", step2)
```

No matter how you write them, these transformations compose naturally. You can chain them indefinitely, and Fynx ensures the data flows correctly through each stage.

## Combining Multiple Streams

When you need to work with multiple observables together, the `|` operator combines them into reactive tuples. Change any component, and the combined observable updates:

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

full_name_parts = User.first_name | User.last_name
full_name = full_name_parts >> (lambda first, last: f"{first} {last}")
```

Now whenever either the first or last name changes, the full name recalculates automatically. This pattern scales elegantly to any number of observables.

## Filtering with Conditions

The `&` operator filters observables so they only emit values when conditions are met. Use `~` to negate conditions. This becomes powerful when building reactive systems with complex business logic:

```python
uploaded_file = observable(None)
is_processing = observable(False)

is_valid = uploaded_file >> (lambda f: f is not None)
preview_ready = uploaded_file & is_valid & (~is_processing)
```

The `preview_ready` observable only has a value when a file exists, it's valid, and processing isn't active. All three conditions must align before anything downstream executes.

## Reacting to Changes

Fynx offers multiple ways to react to observable changes, letting you choose the style that fits each situation. Decorators provide clean syntax for dedicated reaction functions:

```python
@reactive(preview_ready)
def show_preview(file):
    print(f"Showing: {file}")
```

Subscriptions work well for inline reactions:

```python
full_name.subscribe(lambda name: print(f"Name: {name}"))
```

Context managers create scoped reactions that clean up automatically:

```python
with full_name_parts as react:
    react(lambda first, last: print(f"Changed to: {first} {last}"))
```

For reactions that should only trigger when specific conditions become true, the `watch` decorator monitors multiple conditions simultaneously:

```python
@watch(lambda: User.age.value >= 18, lambda: User.email.value.endswith('.com'))
def process_eligible_user():
    print("Eligible user detected!")
```

## The Reactive Operators

Fynx provides four core operators that compose into sophisticated reactive systems. The `>>` operator transforms values through functions. The `|` operator combines multiple observables into tuples. The `&` operator filters based on boolean conditions. The `~` operator negates those conditions. Together, these operators form a complete algebra for reactive data flow.

Consider `total_price >> (lambda t: f"${t:.2f}")` for transformations, `(first | last) >> (lambda f, l: f"{f} {l}")` for combinations, `file & is_valid & (~is_processing)` for conditional filtering, and `~(is_processing)` for negation. Each operation produces a new observable that you can transform, combine, or filter further.

## A Complete Example

Here's how these pieces fit together in a practical file upload system. Notice how complex reactive logic emerges naturally from simple compositions:

```python
from fynx import Store, observable, reactive

class FileUpload(Store):
    uploaded_file = observable(None)
    is_processing = observable(False)
    progress = observable(0)

is_valid = FileUpload.uploaded_file >> (lambda f: f is not None)
is_complete = FileUpload.progress >> (lambda p: p >= 100)

ready_for_preview = FileUpload.uploaded_file & is_valid & (~FileUpload.is_processing)

@reactive(ready_for_preview)
def show_file_preview(file):
    print(f"Preview: {file}")

FileUpload.uploaded_file = "document.pdf"  # Preview: document.pdf

FileUpload.is_processing = True
FileUpload.uploaded_file = "image.jpg"     # No preview (processing active)

FileUpload.is_processing = False           # Preview: image.jpg
```

The preview function triggers automatically, but only when all conditions align. You never manually check whether to show the preview—the reactive graph handles that coordination.

## Going Deeper

Fynx supports more sophisticated patterns for complex applications. Store-level reactions give you snapshots of all observables whenever anything changes, perfect for logging or persistence:

```python
@reactive(UserProfile)
def on_any_change(snapshot):
    print(f"Profile updated: {snapshot.first_name} {snapshot.last_name}")
```

Stores also provide serialization for state persistence. Save your entire reactive state to a dictionary and restore it later:

```python
state_dict = UserProfile.to_dict()
UserProfile.load_state(state_dict)
```

For more examples showing how observables compose into sophisticated reactive systems, explore the [`examples/`](./examples/) directory. The UserProfile example demonstrates store-level reactions, multiple subscription patterns, and building complex transformations from simpler components. Additional examples cover reactive forms, real-time dashboards, and integration with popular frameworks.

## The Mathematical Foundation

If you're curious about the theory powering Fynx, the core insight is that observables form a functor in the category-theoretic sense. An `Observable<T>` represents a time-varying value—formally, a continuous function $\mathcal{T} \to T$ where $\mathcal{T}$ denotes the temporal domain. This construction naturally forms an endofunctor $\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$ on the category of Python types.

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

Category theory provides formal proof that Fynx's behavior is correct and composable. The functor laws guarantee that chaining transformations never produces unexpected behavior. The product structure ensures that combining observables remains symmetric and associative. These aren't implementation details—they're mathematical guarantees that follow from the categorical structure itself.

But what are the practical benefits of this? Ultimately, it's that changes flow through your reactive graph transparently because the mathematics proves they must. Fynx handles all dependency tracking and propagation automatically, and the categorical foundation ensures there are no edge cases or surprising interactions. You describe what you want declaratively, and the underlying mathematics—specifically the universal properties of functors, products, and pullbacks—ensures it behaves correctly in all circumstances.


## Design Philosophy

Fynx embodies a simple principle: mathematical rigor shouldn't compromise usability. The library builds on category theory but exposes that power through Pythonic interfaces. Observables behave like normal values—you read and write them naturally—while reactivity happens behind the scenes. Method chaining flows naturally: `observable(42).subscribe(print)` reads like plain English.

Composability runs through every aspect of the design. Transform with `>>`, combine with `|`, filter with `&`. Each operation produces new observables that you can transform further. Complex reactive systems emerge from simple, reusable pieces. This compositional approach mirrors how mathematicians think about functions and morphisms, but you don't need to know category theory to benefit from its guarantees.

Fynx offers multiple APIs because different situations call for different tools. Use decorators for convenience, direct calls for control, or context managers for scoped reactions. The library adapts to your style rather than forcing one approach.

Framework agnosticism matters. Fynx works with Streamlit, FastAPI, Flask, or any Python framework. The core library has zero dependencies and integrates cleanly with existing tools. Whether you're building web applications, data pipelines, or desktop software, Fynx fits naturally into your stack.

## API Reference

**Core Functions**

`observable(initial_value)` creates a reactive value that notifies subscribers when changed. This forms the foundation of reactive state in Fynx.

`reactive(observable)` provides a decorator that reacts to observable changes, executing the decorated function whenever the observable emits a new value.

`watch(*conditions, callback)` monitors specific conditions and triggers callbacks when those conditions become true, enabling reactive logic based on boolean expressions.

**Classes**

`Store` serves as a base class for organizing related observables with built-in serialization support. Stores provide structure for complex state management and enable persistence patterns.

## Contributing

We welcome contributions to Fynx. The project uses Poetry for dependency management and pytest for testing. To get started:

```bash
poetry install --with dev --with test
poetry run pre-commit install
poetry run pytest
```

The pre-commit hooks run automatically on each commit, checking code formatting and style. You can also run them manually across all files with `poetry run pre-commit run --all-files`.

To verify your changes pass all tests and coverage requirements, run `poetry run pytest --cov=fynx`. The linting script at `./scripts/lint.sh` checks for issues, and `./scripts/lint.sh --fix` automatically fixes formatting and import problems.


When contributing, fork the repository and create a feature branch with a descriptive name like `feature/amazing-feature`. Make your changes, add comprehensive tests, and ensure the test suite passes. Submit a pull request with a clear description of what you've changed and why.

### Visualizing Current Test Coverage

Fynx maintains comprehensive test coverage tracked through Codecov. Here are visual representations of our current coverage:

#### Sunburst Diagram
<p align="center">
  <img src="https://codecov.io/github/off-by-some/fynx/graphs/sunburst.svg?token=NX2QHA8V8L" alt="Sunburst Coverage Diagram"/>
</p>
*The inner-most circle represents the entire project, with folders and files radiating outward. Size and color represent statement count and coverage percentage.*

#### Grid Diagram
<p align="center">
  <img src="https://codecov.io/github/off-by-some/fynx/graphs/tree.svg?token=NX2QHA8V8L" alt="Grid Coverage Diagram"/>
</p>
*Each block represents a file. Size and color indicate statement count and coverage percentage.*

#### Icicle Diagram
<p align="center">
  <img src="https://codecov.io/github/off-by-some/fynx/graphs/icicle.svg?token=NX2QHA8V8L" alt="Icicle Coverage Diagram"/>
</p>

*The top section represents the entire project, with folders and files below. Size and color represent statement count and coverage percentage.*

## License

Fynx is licensed under the MIT License. See the [LICENSE](./LICENSE) file for complete details.
