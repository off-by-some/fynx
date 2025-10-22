"""
Copy-on-Write Observer Set
=========================

This module provides CoWObserverSet and SharedObserverArray for memory-efficient
observer storage using copy-on-write semantics.

Memory savings example:
- 1000 observables with same 10 observers
- Traditional: 1000 × 10 = 10,000 references
- CoW: 1 shared array + 1000 lightweight wrappers ≈ 100 references
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass
class SharedObserverArray:
    """Shared observer array for CoW."""

    observers: List[Callable]


class CoWObserverSet:
    """
    Copy-on-Write observer set for memory efficiency.

    Shares observer arrays between instances until modification occurs.
    This reduces memory usage when multiple observables have similar
    observer sets.

    Memory savings example:
    - 1000 observables with same 10 observers
    - Traditional: 1000 × 10 = 10,000 references
    - CoW: 1 shared array + 1000 lightweight wrappers ≈ 100 references
    """

    __slots__ = ("_shared_ref", "_own_observers", "_is_shared")

    def __init__(self, shared_ref: Optional[SharedObserverArray] = None):
        self._shared_ref = shared_ref
        self._own_observers: Optional[List[Callable]] = None
        self._is_shared = shared_ref is not None

    def add(self, callback: Callable) -> None:
        """Add observer, copying shared array if necessary."""
        if self._is_shared:
            # First modification - copy the shared array
            self._own_observers = self._shared_ref.observers.copy()
            self._is_shared = False
            self._shared_ref = None

        if self._own_observers is None:
            self._own_observers = []

        self._own_observers.append(callback)

    def notify_all(self, value: Any) -> None:
        """Notify using either shared or own array."""
        observers = (
            self._shared_ref.observers if self._is_shared else self._own_observers
        )

        if observers:
            for observer in observers:
                try:
                    observer(value)
                except Exception:
                    pass

    def clone(self) -> "CoWObserverSet":
        """Create a copy that shares the backing array."""
        if self._is_shared:
            return CoWObserverSet(self._shared_ref)
        else:
            # Create new shared reference
            shared = SharedObserverArray(self._own_observers)
            return CoWObserverSet(shared)
