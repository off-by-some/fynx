"""
FynX Primitives Context - Reactive Context Protocol and Implementation
====================================================================

This module defines the ReactiveContext protocol and implementation for the primitives package,
providing the interface and concrete implementation for reactive execution contexts that manage dependency tracking.

The ReactiveContext protocol provides the interface for reactive execution contexts
that track which observables are accessed during function execution and automatically
set up observers to re-run the function when dependencies change.

Key Features:
- Automatic dependency tracking
- Observer management
- Lifecycle management
- Nested context support

Key Benefits:
- No circular imports (protocols don't import concrete implementations)
- Better type safety than ABCs
- Runtime isinstance() support with @runtime_checkable
- Structural subtyping (duck typing with type safety)
- Clean separation of interface from implementation
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)

# Import cycle detector for dependency graph management
from ...util.cycle_detector import IncrementalTopoSort

# Import common types
from ..types.common_types import T, U

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from ..protocols.observable_protocol import Observable

# ============================================================================
# REACTIVE CONTEXT PROTOCOL
# ============================================================================


@runtime_checkable
class ReactiveContext(Protocol):
    """
    Protocol for reactive execution contexts that manage dependency tracking.

    Reactive contexts track which observables are accessed during function
    execution and automatically set up observers to re-run the function
    when dependencies change.

    Key Features:
    - Automatic dependency tracking
    - Observer management
    - Lifecycle management
    - Nested context support

    Example:
        ```python
        def use_reactive_context(ctx: ReactiveContext) -> None:
            ctx.run()  # Execute and track dependencies
            ctx.dispose()  # Clean up
        ```
    """

    func: Callable[..., Any]
    original_func: Optional[Callable[..., Any]]
    subscribed_observable: Optional["Observable[Any]"]
    dependencies: Set["Observable[Any]"]
    is_running: bool

    def run(self, value: Any = None) -> None:
        """
        Execute the reactive function and track its dependencies.

        Args:
            value: Optional value to pass to the function.
        """
        ...

    def add_dependency(self, observable: "Observable[Any]") -> None:
        """
        Add an observable as a dependency of this context.

        Args:
            observable: The observable to track as a dependency.
        """
        ...

    def dispose(self) -> None:
        """
        Clean up the reactive context and remove all observers.
        """
        ...


# ============================================================================
# REACTIVE CONTEXT IMPLEMENTATION
# ============================================================================


class ReactiveContextImpl(ReactiveContext):
    """
    Execution context for reactive functions with automatic dependency tracking.

    ReactiveContext manages the lifecycle of reactive functions (computations and reactions).
    It automatically tracks which observables are accessed during execution and sets up
    the necessary observers to re-run the function when any dependency changes.

    Key Responsibilities:
    - Track observable dependencies during function execution
    - Coordinate re-execution when dependencies change
    - Manage observer registration and cleanup
    - Handle merged observables and complex dependency relationships
    - Detect circular dependencies using incremental topological sort

    The context uses a stack-based approach to handle nested reactive functions,
    ensuring that dependencies are tracked correctly even in complex scenarios.

    Attributes:
        func (Callable): The reactive function to execute
        original_func (Callable): The original user function (for unsubscribe)
        subscribed_observable (Observable): The observable this context is subscribed to
        dependencies (Set[Observable]): Set of observables accessed during execution
        is_running (bool): Whether the context is currently executing

    Note:
        This class is typically managed automatically by FynX's decorators and
        observable operations. Direct instantiation is usually not needed.

    Example:
        ```python
        # Usually created automatically by @reactive decorator
        context = ReactiveContext(my_function, my_function, some_observable)
        context.run()  # Executes function and tracks dependencies
        ```
    """

    # Global cycle detector shared across all reactive contexts
    _global_cycle_detector: Optional[IncrementalTopoSort] = None

    @classmethod
    def _get_cycle_detector(cls) -> IncrementalTopoSort:
        """Get or create the global cycle detector instance."""
        if cls._global_cycle_detector is None:
            cls._global_cycle_detector = IncrementalTopoSort()
        return cls._global_cycle_detector

    def __init__(
        self,
        func: Callable,
        original_func: Optional[Callable] = None,
        subscribed_observable: Optional["Observable"] = None,
    ) -> None:
        self.func = func
        self.original_func = (
            original_func or func
        )  # Store the original user function for unsubscribe
        self.subscribed_observable = (
            subscribed_observable  # The observable this context is subscribed to
        )
        self.dependencies: Set["Observable"] = set()
        self.is_running = False
        # For merged observables, we need to remove the observer from the merged observable,
        # not from the automatically tracked source observables
        self._observer_to_remove_from = subscribed_observable
        # For store subscriptions, keep track of all store observables
        self._store_observables: Optional[List["Observable"]] = None

    def run(self, value=None) -> None:
        """Run the reactive function, tracking dependencies."""
        # Import here to avoid circular imports
        from .observable import Observable

        old_context = Observable._current_context
        Observable._current_context = self

        try:
            self.is_running = True
            self.dependencies.clear()  # Clear old dependencies
            self.func()
        finally:
            self.is_running = False
            Observable._current_context = old_context

    def add_dependency(self, observable: "Observable") -> None:
        """Add an observable as a dependency of this context with cycle detection."""
        # Don't add self as dependency (prevents self-cycles)
        if observable is self.subscribed_observable:
            return

        # Only add if not already a dependency to avoid redundant observer registration
        if observable not in self.dependencies:
            self.dependencies.add(observable)
            observable.subscribe(self.run)

            # Add dependency edge to global cycle detector
            # Edge: observable -> subscribed_observable (subscribed_observable depends on observable)
            if self.subscribed_observable is not None:
                cycle_detector = self._get_cycle_detector()
                try:
                    cycle_detector.add_edge(observable, self.subscribed_observable)
                except ValueError as e:
                    # Cycle detected - clean up the subscription we just added
                    observable.unsubscribe(self.run)
                    self.dependencies.remove(observable)
                    raise RuntimeError(f"Circular dependency detected: {e}") from e

    def dispose(self) -> None:
        """Stop the reactive computation and remove all observers."""
        # Clean up cycle detector edges
        cycle_detector = self._get_cycle_detector()
        if self.subscribed_observable is not None:
            for dependency in self.dependencies:
                cycle_detector.remove_edge(dependency, self.subscribed_observable)

        if self._observer_to_remove_from is not None:
            # For single observables or merged observables
            self._observer_to_remove_from.unsubscribe(self.run)
        elif (
            hasattr(self, "_store_observables") and self._store_observables is not None
        ):
            # For store-level subscriptions, remove from all store observables
            for observable in self._store_observables:
                observable.unsubscribe(self.run)

        self.dependencies.clear()


# Export ReactiveContextImpl as ReactiveContext for easier importing
ReactiveContext = ReactiveContextImpl
