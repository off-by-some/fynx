"""
FynX PropagationContext - Breadth-First Change Propagation Context
================================================================

This module provides the PropagationContext class for managing breadth-first
change propagation to prevent stack overflow in reactive systems.

PropagationContext implements a breadth-first notification system that queues
observers and processes them in batches, preventing deep recursion that could
lead to stack overflow in complex reactive dependency graphs.
"""

import threading
from collections import deque
from typing import Any, Callable


class PropagationContext:
    """Manages breadth-first change propagation to prevent stack overflow."""

    _local = threading.local()

    @classmethod
    def _get_state(cls) -> dict:
        if not hasattr(cls._local, "state"):
            cls._local.state = {"is_propagating": False, "pending": deque()}
        return cls._local.state

    @classmethod
    def _enqueue_notification(
        cls, observer: Callable, observable: Any, value: Any
    ) -> None:
        cls._get_state()["pending"].append((observer, observable, value))

    @classmethod
    def _process_notifications(cls) -> None:
        state = cls._get_state()
        if state["is_propagating"]:
            return

        state["is_propagating"] = True
        try:
            while state["pending"]:
                observer, observable, value = state["pending"].popleft()
                # Check if observable should notify before calling observer
                if (
                    hasattr(observable, "_should_notify_observers")
                    and not observable._should_notify_observers()
                ):
                    continue
                observer(value)
        finally:
            state["is_propagating"] = False

    @classmethod
    def _reset_state(cls) -> None:
        """Reset the propagation state for testing."""
        cls._local.__dict__.clear()
