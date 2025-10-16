# Fynx

<p align="center">
  <img src="https://github.com/off-by-some/fynx/raw/main/docs/fynx.png" alt="Fynx Logo"  height="400px">
</p>

<p align="center">
  <a href="https://badge.fury.io/py/fynx">
    <img src="https://badge.fury.io/py/fynx.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/off-by-some/fynx/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/off-by-some/fynx/test.yml" alt="Tests">
  </a>
  <a href="https://codecov.io/github/off-by-some/fynx">
    <img src="https://codecov.io/github/off-by-some/fynx/coverage.svg?branch=main" alt="Coverage">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

---


**Fynx** is a lightweight reactive runtime for Python — a system where state isn’t just stored, it *flows*. 

Inspired by [MobX](https://github.com/mobxjs/mobx) and [functional reactive programming](https://en.wikipedia.org/wiki/Functional_reactive_programming), Fynx turns plain Python objects into living, observable structures. You don’t “manage” updates; you describe relationships, and Fynx takes care of the rest. When one value changes, everything depending on it simply reconfigures itself — transparently, predictably, beautifully.

## Table of Contents

- [Quickstart](#quickstart)
- [Use Cases](#use-cases)
- [Core Concepts](#core-concepts)
  - [Observables](#observables)
  - [Transformations](#transformations)
  - [Combining Observables](#combining-observables)
  - [Conditions and Reactions](#conditions-and-reactions)
  - [Reactive Operators Summary](#reactive-operators-summary)
- [A Complete Example](#a-complete-example)
- [Advanced Usage](#advanced-usage)
- [The Mathematical Foundation](#the-mathematical-foundation)
- [Design Philosophy](#design-philosophy)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Quickstart

You can install fynx via pip

```bash
pip install fynx
```

Below is a quick overview of fynx, more detailed examples are available below.

```python
from fynx import observable

# Create a reactive value
temperature = observable(20)

# Transform it with a simple pipeline
fahrenheit = temperature >> (lambda c: c * 9/5 + 32)
fahrenheit.subscribe(lambda f: print(f"Temperature: {f}°F"))

# Changes propagate automatically
temperature.value = 25  # prints: Temperature: 77.0°F
temperature.value = 30  # prints: Temperature: 86.0°F
```

```text
Temperature: 77.0°F
Temperature: 86.0°F
```

That's it. No manual updates, no callbacks to wire up—just describe what you want, and Fynx keeps everything in sync.

Try the complete interactive example in [`examples/example.py`](examples/example.py) to explore all Fynx features.

## Use Cases

Fynx shines when data flows through transformations:

- **Streamlit and Python frontends** — reactive UIs that update automatically as state changes
- **Data pipelines** — computed values that recalculate when their dependencies change
- **Reactive analytics** — dashboards and visualizations that stay synchronized with live data
- **State synchronization** — any application where multiple values need to track each other

Fynx lets you stop thinking about when to update state and focus purely on what relationships should hold.

## Core Concepts

### Observables

An observable is a value that changes over time. Create them with `observable(initial_value)` or as class attributes in a Store. They work like normal Python values but notify subscribers when they change.

```python
from fynx import observable, Store

# Standalone observable
counter = observable(0)

# Or as part of a Store
class AppState(Store):
    username = observable("")
    is_logged_in = observable(False)
```

#### Transformations

Use the `>>` operator to transform observables. The transformation creates a new derived observable that automatically recalculates when the source changes.

```python
# Transform a single observable
doubled = counter >> (lambda x: x * 2)

# Chain multiple transformations
result = counter >> (lambda x: x * 2) >> (lambda x: x + 10) >> (lambda x: f"Result: {x}")
```

#### Combining Observables

Combine multiple observables with the `|` operator to create reactive tuples. When any component changes, the combined observable updates.

```python
class User(Store):
    first_name = observable("John")
    last_name = observable("Doe")

# Combine observables
full_name_parts = User.first_name | User.last_name

# Transform the combination
full_name = full_name_parts >> (lambda first, last: f"{first} {last}")
```

### Conditions and Reactions

#### Filtering with Conditions

Filter observables with the `&` operator—values only emit when boolean conditions are true. Use `~` for negation.

```python
uploaded_file = observable(None)
is_processing = observable(False)

# Only show preview when file exists and we're not processing
is_valid = uploaded_file >> (lambda f: f is not None)
preview_ready = uploaded_file & is_valid & (~is_processing)
```

#### Reacting to Changes

React to changes using decorators, subscriptions, or context managers:

```python
# Decorator style
@reactive(preview_ready)
def show_preview(file):
    print(f"Showing: {file}")

# Subscription style
full_name.subscribe(lambda name: print(f"Name: {name}"))

# Context manager style
with full_name_parts as react:
    react(lambda first, last: print(f"Changed to: {first} {last}"))
```

For conditional reactions that trigger only when specific conditions are met:

```python
@watch(lambda: User.age.value >= 18, lambda: User.email.value.endswith('.com'))
def process_eligible_user():
    print("Eligible user detected!")
```

#### Reactive Operators Summary

| Operator | Purpose | Example |
|----------|---------|---------|
| `>>` | Transform/map values | `temp >> (lambda c: c * 9/5 + 32)` |
| `\|` | Combine multiple observables | `(first \| last) >> (lambda f, l: f"{f} {l}")` |
| `&` | Filter by conditions | `file & is_valid & (~is_processing)` |
| `~` | Negate conditions | `~(is_processing)` |

## A Complete Example

Here's a file upload system that shows how Fynx makes complex reactive logic elegant:

```python
from fynx import Store, observable, reactive

class FileUpload(Store):
    uploaded_file = observable(None)
    is_processing = observable(False)
    progress = observable(0)

# Derived state emerges naturally
is_valid = FileUpload.uploaded_file >> (lambda f: f is not None)
is_complete = FileUpload.progress >> (lambda p: p >= 100)

# Combine conditions intuitively
ready_for_preview = FileUpload.uploaded_file & is_valid & (~FileUpload.is_processing)

# React only when all conditions align
@reactive(ready_for_preview)
def show_file_preview(file):
    print(f"Preview: {file}")

# Everything updates automatically
FileUpload.uploaded_file = "document.pdf"  # Preview shown!

FileUpload.is_processing = True
FileUpload.uploaded_file = "image.jpg"     # No preview (processing)

FileUpload.is_processing = False           # Preview shown again!
```

```text
Preview: document.pdf
Preview: image.jpg
```

## Advanced Usage

For more sophisticated examples showing how observables compose into complex reactive systems, see the [UserProfile example](examples/advanced_user_profile.py) demonstrating store-level reactions, multiple subscription patterns, and building transformations from simpler components.

Store-level reactions give you snapshots of all observables whenever anything changes:

```python
@reactive(UserProfile)
def on_any_change(snapshot):
    print(f"Profile updated: {snapshot.first_name} {snapshot.last_name}")
```

Stores also support serialization for state persistence:

```python
# Save state
state_dict = UserProfile.to_dict()

# Restore state
UserProfile.load_state(state_dict)
```

See more examples in [`examples/`](examples/) — from reactive forms to real-time dashboards.

## The Mathematical Foundation

If you're curious about the theory behind Fynx, here's the core insight: observables are functors in the category-theoretic sense. An `Observable<T>` represents a time-varying value—essentially a function from Time → T. The `>>` operator is functorial mapping, `|` creates Cartesian products, and `&` forms filtered subobjects.

This isn't just academic terminology. These mathematical properties guarantee that reactive graphs compose predictably, that transformations preserve structure, and that side effects propagate consistently. You describe what you want declaratively, and the category theory ensures it behaves correctly.

The practical benefit? Changes flow through your reactive graph transparently. Fynx handles all dependency tracking and propagation automatically, so you never have to manually wire up update logic.

## Design Philosophy

**Mathematical rigor, Pythonic simplicity.** Built on category theory, but easy to use—observables behave like normal values, ensuring composability and zero boilerplate. Read and write them naturally: reactivity happens behind the scenes with fluent method chaining like `observable(42).subscribe(print)`.

**Composability everywhere.** Transform with `>>`, combine with `|`, filter with `&`. Every operation creates new observables you can transform further. Complex reactive systems emerge from simple, reusable pieces.

**Choose your style.** Use decorators for convenience, direct calls for control, or context managers for scoped reactions. Fynx provides multiple APIs so you can pick what fits each situation.

**Framework-agnostic.** Use Fynx with Streamlit, FastAPI, Flask, or any Python framework. The core library has no dependencies and integrates cleanly with your existing tools.

## API Reference

### Core Functions

- **`observable(initial_value)`** — Create a reactive value that notifies subscribers when changed
- **`reactive(observable)`** — Decorator to react to observable changes
- **`watch(*conditions, callback)`** — Watch for specific conditions and trigger callbacks

### Classes

- **`Store`** — Base class for organizing related observables with serialization support

## Contributing

Contributions are welcome! This project uses Poetry for dependency management and pytest for testing.

```bash
# Install dependencies
poetry install --with dev --with test

# Run tests
poetry run pytest

# Run the full test suite with coverage
poetry run pytest --cov=fynx

# Activate virtual environment
poetry shell
```

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure tests pass (`poetry run pytest`)
5. Submit a pull request

## License

Licensed under MIT — see [LICENSE](LICENSE) for details.