"""
Enhanced DerivedValue - Elegant State Management
================================================

Key improvements:
1. Unified state machine for all derived observables
2. Template method pattern for computation flow
3. Separation of concerns (computation, notification, state)
4. Centralized circular dependency detection
5. Performance optimizations with minimal complexity
"""

from abc import abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, ContextManager, Optional, TypeVar

from .observable import BaseObservable

T = TypeVar("T")


class ComputationState(Enum):
    """Clear state machine for derived observable lifecycle."""

    CLEAN = "clean"  # Value is up-to-date
    DIRTY = "dirty"  # Needs recomputation
    COMPUTING = "computing"  # Currently computing
    INACTIVE = "inactive"  # Cannot compute (e.g., conditions not met)


class ComputationContext:
    """
    Centralized context for tracking computation scope.

    Replaces ad-hoc _computed_from attribute manipulation.
    Uses context managers for automatic cleanup.
    """

    def __init__(self):
        self._stack = []

    @contextmanager
    def computing(self, observable: "DerivedValue") -> ContextManager:
        """Track computation scope for circular dependency detection."""
        self._stack.append(observable)
        try:
            # Check for cycles
            if len(self._stack) != len(set(self._stack)):
                cycle = self._find_cycle()
                raise RuntimeError(f"Circular dependency detected: {cycle}")
            yield
        finally:
            self._stack.pop()

    def _find_cycle(self):
        """Find the cycle in the computation stack."""
        seen = {}
        cycle = []
        for i, obs in enumerate(self._stack):
            if obs in seen:
                cycle = self._stack[seen[obs] :]
                break
            seen[obs] = i
        return " -> ".join(str(o) for o in cycle)

    def is_computing(self, observable: "DerivedValue") -> bool:
        """Check if observable is currently being computed."""
        return observable in self._stack


class DerivedValue(BaseObservable[T]):
    """
    Elegant base class for read-only derived observables.

    Uses template method pattern with clear extension points:
    - _compute_value() - What to compute
    - _should_compute() - When to compute
    - _should_notify() - When to notify observers
    - _can_access_value() - When value access is allowed

    All state management is centralized in this base class.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        initial_value: Optional[T] = None,
        source_observable: Optional["BaseObservable"] = None,
        source_observables: Optional[list] = None,
    ):
        super().__init__(key, initial_value)

        # Source tracking - simplified
        self._source_observable = source_observable
        self._source_observables = source_observables or []

        # Unified state machine
        self._state = ComputationState.DIRTY

        # Performance optimization: cache computed value version
        self._value_version = 0

        # Per-instance cycle detection
        self._computation_context = ComputationContext()

        # Set up dependency subscription
        if source_observable is not None:
            self._setup_source_observers()

    # ============================================================
    # Template Method Pattern - Core Computation Flow
    # ============================================================

    def _evaluate_and_update(self) -> None:
        """
        Template method for computation flow.

        This is the ONLY method that coordinates computation.
        Subclasses override hooks, not this method.
        """
        # Guard: Already computing (prevent recursion)
        if self._state == ComputationState.COMPUTING:
            return

        # Guard: No need to compute
        if not self._should_compute():
            return

        # Transition to computing state
        self._state = ComputationState.COMPUTING

        try:
            # Use per-instance computation context for cycle detection
            with self._computation_context.computing(self):
                new_value = self._compute_value()

            # Check if we should update and notify
            if self._should_notify(new_value):
                self._update_and_notify(new_value)
            else:
                # Update without notification
                self._value_wrapper._value = new_value
                self._state = ComputationState.CLEAN

        except Exception as e:
            # Graceful error handling
            self._handle_computation_error(e)

    def _update_and_notify(self, new_value: T) -> None:
        """Update value and notify observers atomically."""
        old_value = self._value_wrapper._value
        self._value_wrapper._value = new_value
        self._state = ComputationState.CLEAN
        self._value_version += 1

        # Notify observers
        self._notify_observers(new_value)

    # ============================================================
    # Extension Points - Subclasses Override These
    # ============================================================

    @abstractmethod
    def _compute_value(self) -> T:
        """
        Compute the derived value.

        Called within a computation context for cycle detection.
        Should be pure - no side effects.
        """
        pass

    def _should_compute(self) -> bool:
        """
        Determine if computation is needed.

        Default: Only compute when dirty.
        Override for lazy evaluation or conditional logic.
        """
        return self._state == ComputationState.DIRTY

    def _should_notify(self, new_value: T) -> bool:
        """
        Determine if observers should be notified.

        Default: Notify if value changed.
        Override for filtering (e.g., ConditionalObservable).
        """
        return self._value_wrapper._value != new_value

    def _can_access_value(self) -> bool:
        """
        Determine if value can be accessed.

        Default: Always accessible.
        Override for conditional access (e.g., ConditionalObservable).
        """
        return True

    def _handle_computation_error(self, error: Exception) -> None:
        """
        Handle computation errors gracefully.

        Default: Log and keep previous value.
        Override for custom error handling.
        """
        # Re-raise circular dependency errors as they should bubble up
        if "Circular dependency detected" in str(error):
            raise error

        import logging

        logging.error(f"Computation error in {self}: {error}")
        self._state = ComputationState.DIRTY  # Allow retry

    # ============================================================
    # Source Change Handling - Simplified
    # ============================================================

    def _on_source_change(self, value: Any) -> None:
        """
        Handle source changes uniformly.

        Mark dirty and evaluate immediately (push-based).
        For pull-based (lazy), override _should_compute().
        """
        self._mark_dirty()
        self._evaluate_and_update()

    def _mark_dirty(self) -> None:
        """Mark that recomputation is needed."""
        if self._state != ComputationState.COMPUTING:
            self._state = ComputationState.DIRTY

    def _setup_source_observers(self) -> None:
        """
        Subscribe to source changes.

        Override for multiple sources (e.g., MergedObservable).
        """
        if self._source_observable is not None:
            self._source_observable.subscribe(self._on_source_change)

    # ============================================================
    # Value Access - With Protocol
    # ============================================================

    @property
    def value(self) -> T:
        """
        Get current value with lazy evaluation.

        Respects _can_access_value() for conditional access.
        """
        # Track dependency in reactive context
        if BaseObservable._current_context is not None:
            BaseObservable._current_context.add_dependency(self)

        # Check access permission
        if not self._can_access_value():
            return self._get_fallback_value()

        # Evaluate if needed (lazy evaluation)
        self._evaluate_and_update()

        return self._value_wrapper._value

    def _get_fallback_value(self) -> Optional[T]:
        """
        Fallback when value cannot be accessed.

        Default: Return None
        Override to raise exceptions (ConditionalObservable)
        """
        return None

    # ============================================================
    # Read-Only Enforcement
    # ============================================================

    def set(self, value: Optional[T]) -> None:
        """Prevent direct modification of derived values."""
        raise ValueError(
            f"{self.__class__.__name__} is read-only. "
            f"Update source observables instead."
        )

    def _set_computed_value(self, value: T) -> None:
        """
        Internal method for framework to update values.

        Bypasses read-only restriction.
        Used by reactive operators.
        """
        self._value_wrapper._value = value
        self._state = ComputationState.CLEAN
        self._value_version += 1
        self._notify_observers(value)
