"""
FynX - Functional Yielding Observable Networks

A high-performance reactive key-value store that uses delta-based change detection
and only propagates changes to affected nodes.
"""

# Import DeltaKVStore from observable.py (the backend implementation)
from .delta_kv_store import DeltaKVStore

# Import core classes from observable.py (the Observable API)
from .observable import Observable, observable, reactive, transaction

# Import Store classes from store.py
from .store import Store, StoreMeta
from .store import observable as store_observable

# Import Store and related classes from store.py (the Store API)
# from .store import (
#     SessionValue,
# )
# from .store import Store as StoreClass
# from .store import (
#     StoreMeta,
#     StoreSnapshot,
#     Subscriptable,
# )
# from .store import SubscriptableDescriptor as StoreSubscriptableDescriptor


# Create a simple exception class for reactive functions
class ReactiveFunctionWasCalled(Exception):
    """Exception raised when a reactive function is called during testing."""

    pass


# Export all the main classes and functions
__all__ = [
    # Core observables from observable.py
    "Observable",
    "DerivedObservable",
    "MergedObservable",
    "ConditionalObservable",
    "OrObservable",
    "NegatedObservable",
    # Store classes from store.py
    "Store",
    "StoreMeta",
    # Reactive system from observable.py
    "reactive",
    "ReactiveContext",
    # Factory functions
    "observable",  # Global observable function
    "store_observable",  # Store-specific observable descriptor
    "transaction",
    # Exception classes from observable.py
    "ConditionalNeverMet",
    "ConditionalNotMet",
    # Backend implementation from observable.py
    "DeltaKVStore",
    # Exceptions
    "ReactiveFunctionWasCalled",
]
