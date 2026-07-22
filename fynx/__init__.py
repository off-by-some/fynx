"""
FynX - reactive state management for Python
============================================

FynX is a reactive state library inspired by MobX. Observables hold values;
computed values and reactions derive from them and update automatically when
their dependencies change, so state doesn't need to be synchronized by hand.

- Observables hold a value and notify watchers when it changes.
- Computed values derive from observables and recompute (with memoization)
  only when their inputs change.
- Reactions are functions that run as a side effect when their observed
  dependencies change.
- Stores are classes that group related observables with subscription helpers.

Example
-------

```python
from fynx import Store, observable, reactive

# Create a reactive store
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)

    # Computed property using the >> operator
    greeting = (name + age) >> (lambda n, a: f"Hello, {n}! You are {a} years old.")

# React to changes
@reactive(UserStore.name, UserStore.age)
def on_user_change(name, age):
    print(f"User updated: {name}, {age}")

# Changes trigger reactions automatically
UserStore.name = "Bob"  # Prints: User updated: Bob, 30
UserStore.age = 31      # Prints: User updated: Bob, 31
```

See README.md for more examples and detailed documentation.
"""

__version__ = "0.2.1"
__author__ = "Cassidy Bridges"
__email__ = "cassidybridges@gmail.com"

from .observable import (
    ConditionalNeverMet,
    ConditionalObservable,
    MergedObservable,
    Observable,
    ReactiveContext,
    SubscriptableDescriptor,
    TransformPurityError,
)
from .reactive import ReactiveFunctionWasCalled, reactive
from .store import Store, StoreSnapshot, observable
from .types import SessionValue, StoreState, StoreStateMapping, Subscriber

Subscriptable = SubscriptableDescriptor

__all__ = [
    "Observable",
    "Store",
    "StoreSnapshot",
    "Subscriptable",
    "MergedObservable",
    "ConditionalObservable",
    "ConditionalNeverMet",
    "TransformPurityError",
    "ReactiveContext",
    "ReactiveFunctionWasCalled",
    "SessionValue",
    "StoreState",
    "StoreStateMapping",
    "Subscriber",
    "observable",
    "reactive",
]
