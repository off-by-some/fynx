"""
Fynx Computed - Computed Observable Utilities
============================================

This module provides the `computed` function for creating derived observables
whose values are automatically calculated from other observables.

Computed observables enable:
- **Derived State**: Create values that depend on other values
- **Automatic Updates**: Values recalculate when dependencies change
- **Memoization**: Avoid redundant computations
- **Composition**: Build complex derived state from simple observables

The `computed` function implements the functorial map operation over observables,
transforming observable values through pure functions while maintaining reactivity.

Key Concepts:
- **Pure Functions**: Computed functions should be pure (no side effects)
- **Automatic Dependencies**: Dependencies are tracked automatically
- **Lazy Evaluation**: Computed values only update when accessed and dependencies changed

Example:
    ```python
    from fynx import observable, computed

    # Base observables
    width = observable(10)
    height = observable(20)

    # Computed observable
    area = computed(lambda w, h: w * h, width | height)
    print(area.value)  # 200

    # Changes propagate automatically
    width.set(15)
    print(area.value)  # 300
    ```
"""

from typing import Callable

from .observable import MergedObservable, Observable


def computed(func: Callable, observable) -> Observable:
    """
    Create a computed observable that derives its value from other observables.

    The `computed` function creates a new observable whose value is automatically
    calculated by applying the given function to the values of the input observable(s).
    When the input observable(s) change, the computed observable automatically updates.

    This implements the functorial map operation over observables, allowing you to
    transform observable values through pure functions while preserving reactivity.

    Args:
        func: A pure function that computes the derived value. For merged observables,
              the function receives individual values as separate arguments. For single
              observables, it receives the single value.
        observable: The source observable(s) to compute from. Can be a single Observable
                   or a MergedObservable (created with the `|` operator).

    Returns:
        A new Observable containing the computed values. The observable will
        automatically update whenever the source observable(s) change.

    Examples:
        ```python
        from fynx import observable, computed

        # Single observable computation
        counter = observable(5)
        doubled = computed(lambda x: x * 2, counter)
        print(doubled.value)  # 10

        counter.set(7)
        print(doubled.value)  # 14

        # Merged observable computation
        width = observable(10)
        height = observable(20)
        dimensions = width | height

        area = computed(lambda w, h: w * h, dimensions)
        print(area.value)  # 200

        # More complex computation
        person = observable({"name": "Alice", "age": 30})
        greeting = computed(
            lambda p: f"Hello {p['name']}, you are {p['age']} years old!",
            person
        )
        print(greeting.value)  # "Hello Alice, you are 30 years old!"
        ```

    Note:
        Computed functions should be pure (no side effects) and relatively fast,
        as they may be called frequently when dependencies change.

    See Also:
        observable: Create basic observables
        Observable: The returned observable type
        MergedObservable: For combining multiple observables
    """
    if isinstance(observable, MergedObservable):
        # For merged observables, apply func to the tuple values
        merged_computed_obs: Observable = Observable("computed", None)

        def update_merged_computed():
            values = tuple(obs.value for obs in observable._source_observables)
            result = func(*values)
            merged_computed_obs.set(result)

        # Initial computation
        update_merged_computed()

        # Subscribe to changes in the source observable
        observable.subscribe(lambda *args: update_merged_computed())

        return merged_computed_obs
    else:
        # For single observables
        single_computed_obs: Observable = Observable("computed", None)

        def update_single_computed():
            result = func(observable.value)
            single_computed_obs.set(result)

        # Initial computation
        update_single_computed()

        # Subscribe to changes
        observable.subscribe(lambda val: update_single_computed())

        return single_computed_obs
