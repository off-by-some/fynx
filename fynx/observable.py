"""
Hyper-Efficient Delta-Based Key-Value Store with Subscriptions
=============================================================

A high-performance reactive key-value store that uses delta-based change detection
and only propagates changes to affected nodes, based on principles from Self-Adjusting
Computation (SAC) and Differential Dataflow (DD).

Key Features:
- Delta-based change detection (O(affected) complexity)
- Automatic dependency tracking with DAG
- Topological change propagation
- Lazy evaluation for computed values
- Hyper-efficient data structures
- Subscription system for key changes

Mathematical Foundation:
- Self-Adjusting Computation (SAC): Dynamic dependency graphs with trace stability
- Differential Dataflow (DD): Delta collections <Data, Time, Delta>
- O(affected) complexity bounds for optimal incremental computation
"""

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

# Import DeltaKVStore classes from dedicated modules
from .computed_values import (
    HierarchicalComputedValue,
    OptimizedComputedValue,
    StandardComputedValue,
)
from .delta_kv_store import (
    BatchContext,
    ChangeType,
    CircularDependencyError,
    ComputedValue,
    Delta,
    DeltaKVStore,
    DependencyGraph,
)
from .math_optimizations import (
    DynamicSpectralSparsifier,
    IncrementalStatisticsTracker,
    OrthogonalDeltaMerger,
)
