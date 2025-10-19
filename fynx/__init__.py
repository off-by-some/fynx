"""
FynX - Python Reactive State Management Library
===============================================

FynX is a lightweight, transparent reactive state management library for Python,
inspired by MobX. It enables reactive programming patterns where state changes
automatically propagate through your application, eliminating the need for manual
state synchronization.

Core Concepts
-------------

**Observables**: Values that can be watched for changes. When an observable value
changes, all dependent computations and reactions automatically update.

**Computed Values**: Derived values that automatically recalculate when their
dependencies change, with built-in memoization for performance.

**Reactions**: Functions that run automatically when their observed dependencies
change, enabling side effects like UI updates or API calls.

**Stores**: Classes that group related observables together with convenient
subscription and state management methods.

Key Features
-------------

- **Transparent Reactivity**: No special syntax needed - just use regular Python objects
- **Automatic Dependency Tracking**: Observables automatically track what depends on them
- **Lazy Evaluation**: Computed values only recalculate when needed
- **Type Safety**: Full type hint support for better IDE experience and static analysis
- **Memory Efficient**: Automatic cleanup of unused reactive contexts
- **Composable**: Easy to combine and nest reactive components

Quick Example
-------------

```python
from fynx import Store, observable, reactive

# Create a reactive store
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)

    # Computed property using the >> operator
    greeting = (name | age) >> (lambda n, a: f"Hello, {n}! You are {a} years old.")

# React to changes
@reactive(UserStore.name, UserStore.age)
def on_user_change(name, age):
    print(f"User updated: {name}, {age}")

# Changes trigger reactions automatically
UserStore.name = "Bob"  # Prints: User updated: Bob, 30
UserStore.age = 31      # Prints: User updated: Bob, 31
```

For more examples and detailed documentation, see the README.md file.
"""

__version__ = "0.0.7"
__author__ = "Cassidy Bridges"
__email__ = "cassidybridges@gmail.com"

from .computed import computed
from .observable import (
    ConditionalObservable,
    MergedObservable,
    Observable,
    ReactiveContext,
)
from .observable import SubscriptableDescriptor as Subscriptable
from .reactive import reactive
from .store import Store, observable
from .watch import watch

__all__ = [
    "Observable",
    "Store",
    "Subscriptable",
    "MergedObservable",
    "ConditionalObservable",
    "ReactiveContext",
    "observable",
    "reactive",
    "computed",
    "watch",
]
