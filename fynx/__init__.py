"""
FynX - Functional Yielding Observable Networks

A high-performance reactive key-value store that uses delta-based change detection
and only propagates changes to affected nodes.
"""

# Import ReactiveStore from delta_kv_store.py (the backend implementation)
from .delta_kv_store import (
    ChangeSignificanceTester,
    ChangeType,
    CircularDependencyError,
    ComputationError,
    Delta,
    GTCPMetrics,
    MorphismType,
    ReactiveStore,
)

# Import core classes from observable.py (the Observable API)
from .observable import (
    NULL_EVENT,
    ComputedObservable,
    ConditionalNeverMet,
    ConditionalObservable,
    ConditionNotMet,
    Observable,
    ReactiveFunctionError,
    SimpleMapObservable,
    StreamMerge,
    _reset_global_store,
    get_global_store,
)
from .observable import observable as global_observable  # Exceptions; Sentinel
from .observable import (
    reactive,
    transaction,
)

# Import Store from store.py (the metaclass-based Store with automatic attribute access)
from .store import Store

# Backward compatibility alias
observable = global_observable

# Import Store classes from store.py
from .store import (
    StoreMeta,
    StoreSnapshot,
    Subscriptable,
)
from .store import observable as store_observable


# Create a simple exception class for reactive functions
class ReactiveFunctionWasCalled(Exception):
    """Exception raised when a reactive function is called during testing."""

    pass


# Export all the main classes and functions
__all__ = [
    # Core observables from observable.py
    "Observable",
    "SimpleMapObservable",
    "ComputedObservable",
    "ConditionalObservable",
    "StreamMerge",
    # Store classes
    "Store",
    "StoreMeta",
    "StoreSnapshot",
    "Subscriptable",
    # Reactive system from observable.py
    "reactive",
    # Factory functions
    "observable",  # Global observable function (backward compatibility)
    "global_observable",  # Global observable function from observable.py
    "store_observable",  # Store-specific observable function from store.py
    "transaction",
    "get_global_store",
    # Exception classes from observable.py
    "ConditionalNeverMet",
    "ConditionNotMet",
    "ReactiveFunctionError",
    # Backend implementation
    "ReactiveStore",
    "ChangeType",
    "Delta",
    "CircularDependencyError",
    "ComputationError",
    "ChangeSignificanceTester",
    "MorphismType",
    "GTCPMetrics",
    # Sentinel
    "NULL_EVENT",
    # Testing utilities (internal use)
    "_reset_global_store",
    # Exceptions
    "ReactiveFunctionWasCalled",
]
