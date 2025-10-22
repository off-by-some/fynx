"""
FynX Observable Module - Core Reactive Classes
==============================================

This module contains the fundamental building blocks of FynX's reactive programming
system. It provides classes and utilities for creating observable values, managing
reactive dependencies, and composing complex reactive behaviors.

Understanding Observables
-------------------------

At its core, an observable is a value that can notify others when it changes. Unlike
regular variables, observables automatically track what depends on them and update
those dependencies when their value changes.

The reactive programming paradigm in FynX enables automatic propagation of state
changes through your application, eliminating the need for manual synchronization
of dependent values and side effects.

Core Classes
-------------

**Observable**: The foundation of reactivity - a value that automatically notifies
subscribers when it changes. Observables behave like regular values but provide
reactive capabilities.

**ReactiveContext**: Manages the execution context for reactive functions, handling
automatic dependency tracking and coordinating updates when dependencies change.

**MergedObservable**: Combines multiple observables into a single reactive unit using
the `+` operator. Useful for coordinating related values and passing them as a group
to computed functions.

**ConditionalObservable**: Filters reactive streams based on boolean conditions using
the `&` operator. Only emits values when all specified conditions are met.

**ComputedObservable**: A read-only observable that derives its value from other
observables. Computed values automatically recalculate when their dependencies change.

**ObservableValue**: A transparent wrapper that makes reactive attributes behave like
regular values while providing access to observable methods.

**SubscriptableDescriptor**: Descriptor class for creating observable attributes in
Store classes and other contexts where class-level observables are needed.

Reactive Operators
------------------

FynX provides intuitive operators for composing reactive behaviors:

**Merge (`+`)**: Combines observables into tuples for coordinated updates:
```python
point = x + y  # Creates (x.value, y.value) that updates when either changes
```

**Transform (`>>`)**: Applies functions to create computed values:
```python
doubled = counter >> (lambda x: x * 2)  # Computed value
```

**Filter (`&`)**: Creates conditional observables that only emit when conditions are met:
```python
valid_data = data & is_valid  # Only emits when is_valid is True
```

**Negate (`~`)**: Creates boolean observables with inverted logic:
```python
is_not_loading = ~is_loading  # True when is_loading is False
```

Key Concepts
------------

**Dependency Tracking**: Observables automatically track which reactive contexts
depend on them during execution, enabling precise and efficient updates.

**Transparent Behavior**: Reactive classes behave like their underlying values in
most contexts, making them easy to integrate into existing code.

**Lazy Evaluation**: Computed values only recalculate when accessed and dependencies
have changed, improving performance.

**Automatic Cleanup**: Reactive contexts are automatically cleaned up when no longer
needed, preventing memory leaks.

**Type Safety**: Full generic type support ensures type-safe reactive programming
with excellent IDE support and static analysis.

**Circular Dependency Detection**: FynX automatically detects and prevents circular
dependencies at runtime.

Practical Examples
------------------

### Basic Observable Usage
```python
from fynx.observable import Observable

# Create an observable value
counter = Observable("counter", 0)
print(counter.value)  # 0

# Subscribe to changes
def on_change():
    print(f"Counter changed to: {counter.value}")

counter.subscribe(on_change)
counter.set(5)  # Prints: "Counter changed to: 5"
```

### Merging Observables
```python
# Create multiple observables
width = Observable("width", 10)
height = Observable("height", 20)

# Merge them into a single reactive unit
dimensions = width + height
print(dimensions.value)  # (10, 20)

# Changes to either update the merged observable
width.set(15)
print(dimensions.value)  # (15, 20)
```

### Conditional Observables
```python
temperature = Observable("temp", 20)
is_heating_on = Observable("heating", False)

# Only emit temperature when heating is on
heating_temp = temperature & is_heating_on

def activate_heating(temp):
    print(f"Maintaining temperature at {temp}°C")

heating_temp.subscribe(activate_heating)

temperature.set(22)     # No output (heating is off)
is_heating_on.set(True) # Prints: "Maintaining temperature at 22°C"
temperature.set(25)     # Prints: "Maintaining temperature at 25°C"
```

### Computed Values
```python
# Create computed observables using .then()
area = dimensions.then(lambda w, h: w * h)
print(area.value)  # 300

width.set(20)
print(area.value)  # 400 (automatically recalculated)
```

Performance Considerations
--------------------------

- **Lazy Evaluation**: Computed values only recalculate when accessed
- **Dependency Tracking**: Only tracks actual dependencies, not speculative ones
- **Memory Management**: Automatic cleanup of unused reactive contexts
- **Efficient Updates**: Only notifies observers when values actually change

Common Patterns
---------------

- **State Synchronization**: Use observables to keep UI and data in sync
- **Derived State**: Use computed values for calculated properties
- **Event Filtering**: Use conditional observables for selective reactivity
- **Data Composition**: Use merged observables for related value coordination

See Also
--------

- `fynx.computed`: For creating derived values from observables
- `fynx.store`: For grouping observables into reactive state containers
- `fynx.watch`: For conditional reactive functions
- `fynx.reactive`: For reactive decorators and subscriptions
"""

from fynx.observable.computed import (
    ComputedObservable,
    ConditionalNeverMet,
    ConditionalNotMet,
    ConditionalObservable,
    MergedObservable,
)
from fynx.observable.core.abstract.operations import (
    OperationsMixin,
    OperatorMixin,
    TupleMixin,
)
from fynx.observable.core.context import ReactiveContext
from fynx.observable.core.context import ReactiveContext as ReactiveContextInterface
from fynx.observable.core.observable import Observable
from fynx.observable.core.value import ObservableValue, SubscriptableDescriptor
from fynx.observable.types.protocols.conditional_protocol import Conditional
from fynx.observable.types.protocols.merged_protocol import Mergeable
from fynx.observable.types.protocols.observable_protocol import (
    Observable as ObservableInterface,
)

__all__ = [
    "Observable",
    "ComputedObservable",
    "MergedObservable",
    "ConditionalObservable",
    "ConditionalNeverMet",
    "ReactiveContext",
    "ObservableValue",
    "SubscriptableDescriptor",
    "OperationsMixin",
    # Protocols
    "ObservableInterface",
    "Mergeable",
    "Conditional",
    "ReactiveContextInterface",
    # Operator mixins
    "OperatorMixin",
    "TupleMixin",
]
