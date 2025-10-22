"""
FynX Observable Protocols - Type-Safe Interface Definitions
==========================================================

This module defines Protocol-based interfaces for observables, providing
better type safety than ABCs while avoiding circular import issues.

Protocols are structural types that define interfaces without requiring
inheritance. They enable duck typing with full type safety and can be
used with isinstance() checks when marked with @runtime_checkable.

Key Features:
- Automatic unwrapping of nested Observable values
- Value access with dependency tracking
- Change notification to observers
- Reactive operations and operator overloading
- Transparent behavior in most contexts

Auto-Unwrapping:
All observables automatically unwrap nested Observable values:
- Observable(Observable(1)) → 1
- (Observable(1), Observable(2)) → (1, 2)
- Observable((Observable(1), Observable(2))) → (1, 2)

Key Benefits:
- No circular imports (protocols don't import concrete implementations)
- Better type safety than ABCs
- Runtime isinstance() support with @runtime_checkable
- Structural subtyping (duck typing with type safety)
- Clean separation of interface from implementation
- Automatic value unwrapping for ergonomic API
"""

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, Set, Tuple
from typing import TypeVar
from typing import TypeVar as TypeVarAlias
from typing import Union, overload, runtime_checkable

# Import mixin protocols from operations
from ..protocols.operations_protocol import ReactiveOperations, TupleBehavior

# Import TransparentValue and ObservableValue from the new location
from ..protocols.value_protocol import TransparentValue
from ..value.value import ObservableValue

# Import common types
from .common_types import (
    ComputedFactory,
    Condition,
    ConditionalFactory,
    ConditionFunction,
    MergedFactory,
    ObservableFactory,
    T,
    TransformFunction,
    TypeChecker,
    U,
)

# ============================================================================
# CORE OBSERVABLE PROTOCOL
# ============================================================================


@runtime_checkable
class Observable(Protocol[T]):
    """
    Protocol defining the core observable interface.

    This is the fundamental interface that all reactive values must implement.
    It defines the essential reactive behavior: value access with dependency
    tracking, change notification, and basic lifecycle management.

    Key Features:
    - Value access with automatic dependency tracking
    - Automatic unwrapping of nested Observable values
    - Change notification to observers
    - Subscription management
    - Transparent behavior in most contexts
    - Reactive operations (then, alongside, requiring, negate, either)
    - Operator overloading (+, >>, &, ~, |)

    Auto-Unwrapping:
    The value property automatically unwraps nested Observable values:
    - Observable(Observable(1)) → 1
    - (Observable(1), Observable(2)) → (1, 2)
    - Observable((Observable(1), Observable(2))) → (1, 2)

    Example:
        ```python
        def process_observable(obs: Observable[int]) -> None:
            print(f"Value: {obs.value}")  # Automatically unwrapped
            obs.subscribe(lambda v: print(f"Changed to: {v}"))

            # All observables support reactive operations
            doubled = obs.then(lambda x: x * 2)  # Natural language
            doubled = obs >> (lambda x: x * 2)   # Operator syntax

            # Auto-unwrapping works everywhere
            nested = observable(observable(5))
            print(nested.value)  # Prints: 5 (not Observable(5))
        ```
    """

    # Core properties
    key: str

    @property
    def value(self) -> T:
        """
        Get the observable's value with automatic unwrapping of nested Observables.

        This property automatically unwraps nested Observable values, making
        the API more ergonomic. Users always get the raw values rather than
        Observable wrappers.

        Returns:
            The unwrapped value, with all nested Observables resolved.

        Example:
            ```python
            # Simple case
            obs = observable(42)
            print(obs.value)  # 42

            # Nested observables are unwrapped
            nested = observable(observable(42))
            print(nested.value)  # 42 (not Observable(42))

            # Tuples with observables are unwrapped
            tuple_obs = observable((observable(1), observable(2)))
            print(tuple_obs.value)  # (1, 2)
            ```
        """
        ...

    # Core methods
    def set(self, value: Optional[T]) -> None:
        """
        Update the observable's value and notify all observers if the value changed.

        Args:
            value: The new value to store in the observable.
        """
        ...

    def subscribe(self, func: Callable[[T], None]) -> "Observable[T]":
        """
        Subscribe a function to react to value changes.

        Args:
            func: A callable that accepts one argument (the new value).

        Returns:
            This observable instance for method chaining.
        """
        ...

    def unsubscribe(self, func: Callable[[T], None]) -> None:
        """
        Unsubscribe a function from this observable.

        Args:
            func: The function to unsubscribe from this observable.
        """
        ...

    def add_observer(self, observer: Callable[[], None]) -> None:
        """
        Add a low-level observer function that will be called when the value changes.

        Args:
            observer: A callable that takes no arguments and will be called
                     whenever the observable's value changes.
        """
        ...

    def remove_observer(self, observer: Callable[[], None]) -> None:
        """
        Remove an observer function.

        Args:
            observer: The observer function to remove.
        """
        ...

    # Magic methods for transparent behavior
    def __bool__(self) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ============================================================================
# DERIVED OBSERVABLE PROTOCOL (Base for Computed, Merged, Conditional)
# ============================================================================


@runtime_checkable
class DerivedObservable(Observable[T], Protocol[T]):
    """
    Protocol for derived observables that extend the base observable with additional capabilities.

    Derived observables are read-only reactive values that automatically
    calculate their value based on other observables or conditions. They
    provide derived state without manual synchronization.

    Key Features:
    - Read-only (cannot be set directly via .set())
    - Automatic recalculation when dependencies change
    - Lazy evaluation (only computes when accessed)
    - Extends base Observable with derived-specific methods

    Example:
        ```python
        def process_derived_value(derived: DerivedObservable[float]) -> None:
            print(f"Derived value: {derived.value}")
            # derived.set(5) would raise ValueError - derived values are read-only
        ```
    """

    def _set_computed_value(self, value: Optional[T]) -> None:
        """
        Internal method for updating derived observable values.

        Warning: This method should only be called by the FynX framework internals.
        Direct use may break reactive relationships and is not supported.

        Args:
            value: The new derived value calculated from dependencies.
        """
        ...


# ============================================================================
# MERGEABLE PROTOCOL (Inherits from DerivedObservable)
# ============================================================================


@runtime_checkable
class Mergeable(DerivedObservable[T], TupleBehavior[T], Protocol[T]):
    """
    Protocol for observables that can be merged with others using the + operator.

    Merged observables combine multiple source observables into tuples,
    updating when any of their components change. This enables coordinated
    reactive updates across related values. This protocol composes
    DerivedObservable with TupleBehavior for tuple-specific functionality.

    Key Features:
    - Combines multiple observables into tuples
    - Updates when any source observable changes
    - Supports chaining with + operator
    - Tuple-like behavior for accessing combined values
    - Read-only (inherits from DerivedObservable)
    - Inherits all DerivedObservable capabilities

    Example:
        ```python
        def process_coordinates(coords: Mergeable[tuple[int, int]]) -> None:
            x, y = coords[0], coords[1]  # Tuple indexing
            for coord in coords:        # Tuple iteration
                print(coord)
        ```
    """

    _source_observables: List[Observable]


# ============================================================================
# CONDITIONAL PROTOCOL (Inherits from DerivedObservable)
# ============================================================================


@runtime_checkable
class Conditional(DerivedObservable[T], Protocol[T]):
    """
    Protocol for observables that filter values based on boolean conditions.

    Conditional observables only emit values when ALL specified conditions
    are True. They act as smart gates that filter reactive streams based
    on complex boolean logic. This protocol composes DerivedObservableProtocol
    with ReactiveOperatorsProtocol for full functionality.

    Key Features:
    - Filters values based on conditions
    - Only emits when all conditions are satisfied
    - Supports chaining with & operator
    - Provides active/inactive state information
    - Read-only (inherits from DerivedObservable)
    - Inherits all DerivedObservable capabilities

    Example:
        ```python
        def process_valid_data(data: ConditionalProtocol[str]) -> None:
            if data.is_active:
                print(f"Valid data: {data.value}")
            else:
                print("Data is not valid")
        ```
    """

    is_active: bool


@runtime_checkable
class Computed(DerivedObservable[T], Protocol[T]):
    """
    Protocol for computed observables that derive their values from other observables.

    Computed observables are read-only reactive values that automatically
    calculate their value based on other observables. They provide derived
    state without manual synchronization. Inherits reactive operations
    from the base Observable protocol.

    Key Features:
    - Read-only (cannot be set directly)
    - Automatic recalculation when dependencies change
    - Lazy evaluation (only computes when accessed)
    - Supports transformation chains with .then()
    - Inherits all DerivedObservable capabilities

    Example:
        ```python
        def process_computed_value(computed: Computed[float]) -> None:
            print(f"Computed value: {computed.value}")
            # computed.set(5) would raise ValueError - computed values are read-only
        ```
    """

    _computation_func: Optional[Callable[[], T]]
    _source_observable: Optional[Observable]

    def then(self, func: Callable[[T], U]) -> "Computed[U]":
        """
        Transform this observable's value with the given function.

        Args:
            func: A pure function to apply to the observable's value.

        Returns:
            A new computed Observable containing the transformed value.
        """
        ...


# ============================================================================
# DESCRIPTOR PROTOCOLS
# ============================================================================

# ObservableValue is now imported from ..value.value as a concrete implementation


@runtime_checkable
class SubscriptableDescriptor(Protocol[T]):
    """
    Protocol for descriptors that create reactive class attributes.

    This protocol enables Store classes to define attributes that behave
    like regular Python attributes while providing full reactive capabilities.
    """

    attr_name: Optional[str]
    _initial_value: Optional[T]
    _original_observable: Optional[Observable[T]]
    _owner_class: Optional[type]

    def __set_name__(self, owner: type, name: str) -> None: ...
    def __get__(self, instance: Optional[object], owner: Optional[type]) -> Any: ...
    def __set__(self, instance: Optional[object], value: Optional[T]) -> None: ...
