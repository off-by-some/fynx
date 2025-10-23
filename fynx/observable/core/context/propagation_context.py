"""
FynX PropagationContext - Topologically-Ordered Change Propagation
=================================================================

Key improvement: Process notifications in topological order to ensure
dependencies are updated before dependents.

This eliminates race conditions in complex dependency graphs while
maintaining O(N log N) performance.
"""

import threading
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Set, Tuple


class PropagationContext:
    """
    Manages topologically-ordered change propagation.

    Instead of processing notifications in FIFO order, this implementation:
    1. Collects all pending notifications
    2. Sorts observables by topological order
    3. Processes notifications depth-first (dependencies before dependents)

    This ensures that when ConditionalObservable evaluates conditions,
    the condition observables have already been updated.
    """

    _local = threading.local()

    @classmethod
    def _get_state(cls) -> dict:
        """Get thread-local propagation state."""
        if not hasattr(cls._local, "state"):
            cls._local.state = {
                "is_propagating": False,
                # Map observable -> list of (observer, value) pairs
                "pending": defaultdict(list),
            }
        return cls._local.state

    @classmethod
    def _enqueue_notification(
        cls, observer: Callable, observable: Any, value: Any
    ) -> None:
        """
        Enqueue a notification to be processed.

        Groups notifications by observable for efficient topological sorting.
        """
        state = cls._get_state()
        state["pending"][observable].append((observer, value))

    @classmethod
    def _process_notifications(cls) -> None:
        """
        Process all pending notifications in topological order.

        Algorithm:
        1. Collect all observables with pending notifications
        2. Sort them using topological order from cycle detector
        3. Process each observable's notifications
        4. Check _should_notify_observers() before calling observer

        Performance: O(N log N) where N = number of observables with pending notifications
        """
        state = cls._get_state()

        # Prevent re-entrance
        if state["is_propagating"]:
            return

        state["is_propagating"] = True
        try:
            while state["pending"]:
                # Get all observables with pending notifications
                pending_observables = list(state["pending"].keys())

                # Sort by topological order (dependencies first)
                sorted_observables = cls._topological_sort(pending_observables)

                # Process notifications for each observable in order
                for observable in sorted_observables:
                    # Get and clear pending notifications for this observable
                    notifications = state["pending"].pop(observable, [])

                    # Check if observable should notify
                    should_notify = (
                        not hasattr(observable, "_should_notify_observers")
                        or observable._should_notify_observers()
                    )

                    if not should_notify:
                        continue

                    # Call all observers with their values
                    for observer, value in notifications:
                        observer(value)
        finally:
            state["is_propagating"] = False
            # Clear any remaining pending (shouldn't happen, but defensive)
            state["pending"].clear()

    @classmethod
    def _topological_sort(cls, observables: list) -> list:
        """
        Sort observables in topological order using the global cycle detector.

        Observables with no dependencies come first, followed by those that
        depend on them, and so on.

        Args:
            observables: List of observables to sort

        Returns:
            Sorted list with dependencies before dependents
        """
        # Get the global cycle detector
        from fynx.observable.core.context import ReactiveContextImpl

        cycle_detector = ReactiveContextImpl._get_cycle_detector()

        # If no cycle detector or no nodes, return original order
        if cycle_detector is None or not hasattr(cycle_detector, "_nodes"):
            return observables

        # Build a mapping of observable -> topological level
        # Lower levels = fewer dependencies (should be processed first)
        levels = cls._compute_levels(observables, cycle_detector)

        # Sort by level (ascending), then by insertion order for stability
        return sorted(
            observables, key=lambda obs: (levels.get(obs, float("inf")), id(obs))
        )

    @classmethod
    def _compute_levels(cls, observables: list, cycle_detector) -> Dict[Any, int]:
        """
        Compute topological levels for observables.

        Level 0: No dependencies
        Level 1: Depends only on level 0
        Level N: Depends on observables up to level N-1

        This is a simplified topological sort that assigns levels.
        """
        levels = {}
        observable_set = set(observables)

        # Helper to get incoming edges (what does this observable depend on?)
        def get_dependencies(obs) -> Set:
            """Get observables that this observable depends on."""
            if not hasattr(cycle_detector, "_adjacency_list"):
                return set()

            deps = set()
            # In the cycle detector, edge (A -> B) means B depends on A
            # So we need to find all A where (A -> obs) exists
            for source, targets in cycle_detector._adjacency_list.items():
                if obs in targets and source in observable_set:
                    deps.add(source)
            return deps

        # BFS to compute levels
        queue = deque()

        # Find observables with no dependencies in our set
        for obs in observables:
            deps = get_dependencies(obs)
            if not deps:
                levels[obs] = 0
                queue.append(obs)

        # Process queue: assign level based on max dependency level + 1
        while queue:
            current = queue.popleft()
            current_level = levels[current]

            # Find observables that depend on current
            if hasattr(cycle_detector, "_adjacency_list"):
                dependents = cycle_detector._adjacency_list.get(current, set())

                for dependent in dependents:
                    if dependent not in observable_set:
                        continue

                    # Check if all dependencies of dependent have been processed
                    dependent_deps = get_dependencies(dependent)
                    if all(dep in levels for dep in dependent_deps):
                        # Assign level = max(dependency levels) + 1
                        dependent_level = (
                            max((levels[dep] for dep in dependent_deps), default=-1) + 1
                        )

                        if dependent not in levels:
                            levels[dependent] = dependent_level
                            queue.append(dependent)

        return levels

    @classmethod
    def _reset_state(cls) -> None:
        """Reset the propagation state for testing."""
        cls._local.__dict__.clear()
