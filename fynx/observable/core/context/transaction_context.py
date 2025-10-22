"""
FynX TransactionContext - Transaction Context for Batching Observable Updates
===========================================================================

This module provides the TransactionContext class for batching observable updates
and emitting single notification on commit.

TransactionContext enables efficient batching of multiple observable updates
by deferring notifications until the transaction is committed, preventing
intermediate notifications and improving performance.
"""

import threading
from typing import Any, Callable, Optional

from .propagation_context import PropagationContext


class TransactionContext:
    """Batches observable updates and emits single notification on commit."""

    _local = threading.local()

    @classmethod
    def _get_active(cls) -> list:
        if not hasattr(cls._local, "active"):
            cls._local.active = []
        return cls._local.active

    def __init__(self, observable: "BaseObservable"):
        self.observable = observable
        self._is_outermost = False

    def __enter__(self):
        active = self._get_active()
        self._is_outermost = not active
        active.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        active = self._get_active()
        active.pop()
        if self._is_outermost and not active:
            PropagationContext._process_notifications()

    @classmethod
    def _reset_state(cls) -> None:
        """Reset the transaction state for testing."""
        cls._local.__dict__.clear()
