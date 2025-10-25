"""
FynX - Functional Yielding Observable Networks

A high-performance reactive key-value store that uses delta-based change detection
and only propagates changes to affected nodes.
"""

# Import DeltaKVStore from observable.py (the backend implementation)
from .delta_kv_store import DeltaKVStore

# Import core classes from frontend.py (the Observable API)
from .frontend import (
    ConditionalNeverMet,
    ConditionalNotMet,
    ConditionalObservable,
    DerivedObservable,
    MergedObservable,
    NegatedObservable,
    Observable,
    OrObservable,
    ReactiveContext,
    Store,
    observable,
    reactive,
)

# Import Store and related classes from store.py (the Store API)
from .store import (
    SessionValue,
)
from .store import Store as StoreClass
from .store import (
    StoreMeta,
    StoreSnapshot,
    Subscriptable,
)
from .store import SubscriptableDescriptor as StoreSubscriptableDescriptor


# Create a simple exception class for reactive functions
class ReactiveFunctionWasCalled(Exception):
    """Exception raised when a reactive function is called during testing."""

    pass


# Export all the main classes and functions
__all__ = [
    # Core observables from frontend.py
    "Observable",
    "DerivedObservable",
    "MergedObservable",
    "ConditionalObservable",
    "OrObservable",
    "NegatedObservable",
    # Store and related from store.py
    "Store",
    "StoreMeta",
    "StoreSnapshot",
    "SessionValue",
    "Subscriptable",
    "StoreSubscriptableDescriptor",
    # Reactive system from frontend.py
    "reactive",
    "ReactiveContext",
    # Factory function from frontend.py
    "observable",
    # Exception classes from frontend.py
    "ConditionalNeverMet",
    "ConditionalNotMet",
    # Backend implementation from observable.py
    "DeltaKVStore",
    # Exceptions
    "ReactiveFunctionWasCalled",
]
