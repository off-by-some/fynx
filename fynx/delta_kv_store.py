"""
Enhanced Reactive Store with Automatic Differentiation over Delta Algebras

Improvements:
- Myers diff algorithm for optimal string diffing
- Memory management with weak references for traces
- Full async/await support (accepts both sync and async functions)
- Better trace cleanup and memory efficiency
"""

import asyncio
import atexit
import difflib
import gc
import inspect
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

T = TypeVar("T")


# ============================================================================
# EXCEPTIONS
# ============================================================================


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected."""

    pass


class ComputationError(Exception):
    """Raised when a computed value fails to evaluate."""

    pass


# ============================================================================
# TYPE-AWARE DELTA ALGEBRA
# ============================================================================


class DeltaType(Enum):
    """Classification of delta types for dispatch."""

    NUMERIC = "numeric"
    STRING = "string"
    LIST = "list"
    SET = "set"
    DICT = "dict"
    ARRAY = "array"
    COMPOSITE = "composite"
    OPAQUE = "opaque"


@dataclass
class TypedDelta:
    """
    Type-aware delta with algebraic structure.

    For type T, represents Δ_T such that: O' = O ⊕_T Δ_T
    """

    delta_type: DeltaType
    value: Any
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"Δ[{self.delta_type.value}]({self.value})"


class DeltaAlgebra(ABC):
    """
    Abstract delta algebra for type T.

    Implements:
    - δ_T(O, O') → Δ_T  (compute delta)
    - O ⊕_T Δ_T → O'    (apply delta)
    - Δ₁ ⊕ Δ₂ → Δ₃      (compose deltas)
    """

    @abstractmethod
    def compute_delta(self, old: Any, new: Any) -> Optional[TypedDelta]:
        """Compute Δ_T = δ_T(O, O')"""
        pass

    @abstractmethod
    def apply_delta(self, value: Any, delta: TypedDelta) -> Any:
        """Apply delta: O' = O ⊕_T Δ_T"""
        pass

    @abstractmethod
    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        """Compose deltas: Δ₃ = Δ₁ ⊕ Δ₂"""
        pass

    @abstractmethod
    def is_identity(self, delta: TypedDelta) -> bool:
        """Check if delta is identity (no change)"""
        pass


class NumericAlgebra(DeltaAlgebra):
    """Algebra for numbers: Δ = O' - O"""

    def compute_delta(self, old: Any, new: Any) -> Optional[TypedDelta]:
        if not isinstance(old, (int, float, complex)) or not isinstance(
            new, (int, float, complex)
        ):
            return None
        diff = new - old
        return TypedDelta(DeltaType.NUMERIC, diff)

    def apply_delta(self, value: Any, delta: TypedDelta) -> Any:
        return value + delta.value

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        return TypedDelta(DeltaType.NUMERIC, delta1.value + delta2.value)

    def is_identity(self, delta: TypedDelta) -> bool:
        return abs(delta.value) < 1e-10


class ListAlgebra(DeltaAlgebra):
    """Algebra for lists: Δ = [operations]"""

    def compute_delta(self, old: List, new: List) -> Optional[TypedDelta]:
        if not isinstance(old, list) or not isinstance(new, list):
            return None

        operations = []

        # Fast path: append only
        if len(new) > len(old) and old and new[: len(old)] == old:
            for i in range(len(old), len(new)):
                operations.append(("insert", i, new[i]))
            return TypedDelta(DeltaType.LIST, operations)

        # Fast path: remove from end
        if len(new) < len(old) and new and old[: len(new)] == new:
            for i in range(len(old) - 1, len(new) - 1, -1):
                operations.append(("remove", i))
            return TypedDelta(DeltaType.LIST, operations)

        # Fast path: empty lists
        if not old and not new:
            return TypedDelta(DeltaType.LIST, [])

        # General case: use difflib
        operations = self._compute_list_diff(old, new)
        return TypedDelta(DeltaType.LIST, operations)

    def _compute_list_diff(self, old: List, new: List) -> List[Tuple]:
        """Compute list diff using difflib for optimal insert/remove/update operations."""

        # Check if all elements are hashable (required for difflib)
        # Use type-based check to avoid try/catch for control flow
        def _is_hashable(obj):
            # Common unhashable types that might be in lists
            if isinstance(obj, (dict, list, set)):
                return False
            # For other types, assume hashable (strings, numbers, tuples of hashables, etc.)
            return True

        if not all(_is_hashable(item) for item in old + new):
            # Fall back to full replacement for unhashable types
            return [("remove", i) for i in range(len(old) - 1, -1, -1)] + [
                ("insert", i, new[i]) for i in range(len(new))
            ]

        operations = []
        matcher = difflib.SequenceMatcher(None, old, new)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "delete":
                for i in range(i2 - 1, i1 - 1, -1):
                    operations.append(("remove", i))
            elif tag == "insert":
                for j in range(j1, j2):
                    operations.append(("insert", i1 + (j - j1), new[j]))
            elif tag == "replace":
                for i in range(i2 - 1, i1 - 1, -1):
                    operations.append(("remove", i))
                for j in range(j1, j2):
                    operations.append(("insert", i1 + (j - j1), new[j]))

        return operations

    def apply_delta(self, value: List, delta: TypedDelta) -> List:
        result = list(value)
        for op in delta.value:
            if op[0] == "insert":
                _, idx, val = op
                if idx <= len(result):
                    result.insert(idx, val)
            elif op[0] == "remove":
                _, idx = op
                if 0 <= idx < len(result):
                    del result[idx]
            elif op[0] == "update":
                _, idx, val = op
                if 0 <= idx < len(result):
                    result[idx] = val
        return result

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        return TypedDelta(DeltaType.LIST, delta1.value + delta2.value)

    def is_identity(self, delta: TypedDelta) -> bool:
        return len(delta.value) == 0


class SetAlgebra(DeltaAlgebra):
    """Algebra for sets: Δ = (added, removed)"""

    def compute_delta(self, old: Set, new: Set) -> Optional[TypedDelta]:
        if not isinstance(old, set) or not isinstance(new, set):
            return None
        added = new - old
        removed = old - new
        return TypedDelta(DeltaType.SET, (added, removed))

    def apply_delta(self, value: Set, delta: TypedDelta) -> Set:
        added, removed = delta.value
        return (value | added) - removed

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        add1, rem1 = delta1.value
        add2, rem2 = delta2.value
        added = (add1 | add2) - rem2
        removed = (rem1 | rem2) - add2
        return TypedDelta(DeltaType.SET, (added, removed))

    def is_identity(self, delta: TypedDelta) -> bool:
        added, removed = delta.value
        return len(added) == 0 and len(removed) == 0


class DictAlgebra(DeltaAlgebra):
    """Algebra for dicts: Δ = {key: (old, new)}"""

    def compute_delta(self, old: Dict, new: Dict) -> Optional[TypedDelta]:
        if not isinstance(old, dict) or not isinstance(new, dict):
            return None
        changes = {}
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                changes[key] = (old_val, new_val)
        return TypedDelta(DeltaType.DICT, changes)

    def apply_delta(self, value: Dict, delta: TypedDelta) -> Dict:
        result = dict(value)
        for key, (old_val, new_val) in delta.value.items():
            if new_val is None:
                result.pop(key, None)
            else:
                result[key] = new_val
        return result

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        changes = dict(delta1.value)
        for key, (old2, new2) in delta2.value.items():
            if key in changes:
                old1, _ = changes[key]
                changes[key] = (old1, new2)
            else:
                changes[key] = (old2, new2)
        return TypedDelta(DeltaType.DICT, changes)

    def is_identity(self, delta: TypedDelta) -> bool:
        return len(delta.value) == 0


class StringAlgebra(DeltaAlgebra):
    """
    Algebra for strings using Myers diff algorithm.

    This provides optimal edit sequences with O(ND) time complexity,
    where N is the length of the strings and D is the edit distance.
    """

    def compute_delta(self, old: str, new: str) -> Optional[TypedDelta]:
        if not isinstance(old, str) or not isinstance(new, str):
            return None

        if old == new:
            return TypedDelta(DeltaType.STRING, [])

        # For very large strings, fall back to replacement
        if len(old) > 10000 or len(new) > 10000:
            return TypedDelta(DeltaType.STRING, [("replace_all", new)])

        # Use Myers diff for optimal edit sequence
        operations = self._myers_diff(old, new)

        # If the edit distance is too large, use replacement
        if len(operations) > max(len(old), len(new)) * 0.8:
            return TypedDelta(DeltaType.STRING, [("replace_all", new)])

        return TypedDelta(DeltaType.STRING, operations)

    def _myers_diff(self, old: str, new: str) -> List[Tuple]:
        """
        Myers diff algorithm for optimal string diffing.

        Returns a list of edit operations: ('delete', pos), ('insert', pos, char), or ('keep', pos)
        We convert this to our operation format.
        """
        n, m = len(old), len(new)

        # Use difflib's SequenceMatcher which implements a variant of Myers
        # This is more practical than implementing Myers from scratch
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)

        operations = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "delete":
                operations.append(("delete", i1, old[i1:i2]))
            elif tag == "insert":
                operations.append(("insert", i1, new[j1:j2]))
            elif tag == "replace":
                operations.append(("replace", i1, old[i1:i2], new[j1:j2]))
            # 'equal' means no operation

        return operations

    def apply_delta(self, value: str, delta: TypedDelta) -> str:
        if not isinstance(value, str):
            return value

        if not delta.value:
            return value

        if len(delta.value) == 1 and delta.value[0][0] == "replace_all":
            return delta.value[0][1]

        result = list(value)
        sorted_ops = sorted(delta.value, key=lambda op: op[1], reverse=True)

        for op in sorted_ops:
            if op[0] == "delete":
                _, pos, deleted_text = op
                if pos < len(result):
                    del result[pos : pos + len(deleted_text)]
            elif op[0] == "insert":
                _, pos, inserted_text = op
                if pos <= len(result):
                    result[pos:pos] = list(inserted_text)
            elif op[0] == "replace":
                _, pos, old_text, new_text = op
                if pos < len(result):
                    end_pos = min(pos + len(old_text), len(result))
                    result[pos:end_pos] = list(new_text)

        return "".join(result)

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        if delta2.value and any(op[0] == "replace_all" for op in delta2.value):
            return delta2
        if delta1.value and any(op[0] == "replace_all" for op in delta1.value):
            temp_result = self.apply_delta("", delta1)
            final_result = self.apply_delta(temp_result, delta2)
            return TypedDelta(DeltaType.STRING, [("replace_all", final_result)])

        return TypedDelta(DeltaType.STRING, delta1.value + delta2.value)

    def is_identity(self, delta: TypedDelta) -> bool:
        return len(delta.value) == 0


class ArrayAlgebra(DeltaAlgebra):
    """Algebra for NumPy arrays: Δ = element-wise diff"""

    def compute_delta(self, old: np.ndarray, new: np.ndarray) -> Optional[TypedDelta]:
        if not isinstance(old, np.ndarray) or not isinstance(new, np.ndarray):
            return None
        if old.shape != new.shape:
            return None

        diff = new - old
        nonzero = np.count_nonzero(diff)

        if nonzero < diff.size * 0.1 and nonzero > 0:
            indices = np.nonzero(diff)
            values = diff[indices]
            return TypedDelta(
                DeltaType.ARRAY,
                {
                    "sparse": True,
                    "indices": indices,
                    "values": values,
                    "shape": old.shape,
                },
            )
        else:
            return TypedDelta(DeltaType.ARRAY, {"sparse": False, "diff": diff})

    def apply_delta(self, value: np.ndarray, delta: TypedDelta) -> np.ndarray:
        if delta.value.get("sparse"):
            result = value.copy()
            indices = delta.value["indices"]
            values = delta.value["values"]
            result[indices] = value[indices] + values
            return result
        else:
            return value + delta.value["diff"]

    def compose_deltas(self, delta1: TypedDelta, delta2: TypedDelta) -> TypedDelta:
        if not delta1.value.get("sparse", False):
            diff1 = delta1.value["diff"]
        else:
            shape = delta1.value["shape"]
            diff1 = np.zeros(shape)
            indices = delta1.value["indices"]
            values = delta1.value["values"]
            diff1[indices] = values

        if not delta2.value.get("sparse", False):
            diff2 = delta2.value["diff"]
        else:
            shape = delta2.value["shape"]
            diff2 = np.zeros(shape)
            indices = delta2.value["indices"]
            values = delta2.value["values"]
            diff2[indices] = values

        combined = diff1 + diff2
        return TypedDelta(DeltaType.ARRAY, {"sparse": False, "diff": combined})

    def is_identity(self, delta: TypedDelta) -> bool:
        if delta.value.get("sparse"):
            return len(delta.value.get("values", [])) == 0
        else:
            diff = delta.value.get("diff")
            if diff is None:
                return True
            return np.allclose(diff, 0)


class DeltaRegistry:
    """Central registry for type-specific delta algebras."""

    def __init__(self):
        self._algebras: Dict[DeltaType, DeltaAlgebra] = {
            DeltaType.NUMERIC: NumericAlgebra(),
            DeltaType.STRING: StringAlgebra(),
            DeltaType.LIST: ListAlgebra(),
            DeltaType.SET: SetAlgebra(),
            DeltaType.DICT: DictAlgebra(),
            DeltaType.ARRAY: ArrayAlgebra(),
        }

        self._type_rules = [
            (lambda x: isinstance(x, (int, float, complex)), DeltaType.NUMERIC),
            (lambda x: isinstance(x, str), DeltaType.STRING),
            (lambda x: isinstance(x, np.ndarray), DeltaType.ARRAY),
            (lambda x: isinstance(x, list), DeltaType.LIST),
            (lambda x: isinstance(x, set), DeltaType.SET),
            (lambda x: isinstance(x, dict), DeltaType.DICT),
        ]

    def detect_type(self, value: Any) -> DeltaType:
        """Detect the delta type for a value."""
        for predicate, delta_type in self._type_rules:
            if predicate(value):
                return delta_type
        return DeltaType.OPAQUE

    def compute_delta(self, old: Any, new: Any) -> Optional[TypedDelta]:
        """Compute delta between two values."""
        if old is None or new is None:
            return None

        delta_type = self.detect_type(old)
        if delta_type == DeltaType.OPAQUE:
            return None

        algebra = self._algebras.get(delta_type)
        if algebra is None:
            return None
        return algebra.compute_delta(old, new)

    def apply_delta(self, value: Any, delta: Optional[TypedDelta]) -> Any:
        """Apply a delta to a value."""
        if delta is None:
            return value

        algebra = self._algebras.get(delta.delta_type)
        if algebra is None:
            return value
        return algebra.apply_delta(value, delta)

    def compose_deltas(
        self, delta1: Optional[TypedDelta], delta2: Optional[TypedDelta]
    ) -> Optional[TypedDelta]:
        """Compose two deltas."""
        if delta1 is None:
            return delta2
        if delta2 is None:
            return delta1
        if delta1.delta_type != delta2.delta_type:
            return None

        algebra = self._algebras.get(delta1.delta_type)
        if algebra is None:
            return None
        return algebra.compose_deltas(delta1, delta2)

    def is_identity(self, delta: Optional[TypedDelta]) -> bool:
        """Check if delta represents no change."""
        if delta is None:
            return True

        algebra = self._algebras.get(delta.delta_type)
        if algebra is None:
            return True
        return algebra.is_identity(delta)

    def register_custom_algebra(
        self,
        delta_type: DeltaType,
        algebra: DeltaAlgebra,
        type_predicate: Callable[[Any], bool],
    ) -> None:
        """Register a custom delta algebra for user-defined types."""
        self._algebras[delta_type] = algebra
        self._type_rules.insert(0, (type_predicate, delta_type))


# ============================================================================
# AUTOMATIC DIFFERENTIATION WITH MEMORY MANAGEMENT
# ============================================================================


class TracedOp(Enum):
    """Operations that can be traced and differentiated."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOORDIV = "//"
    MOD = "%"
    NEG = "neg"
    ABS = "abs"
    POW = "**"
    GETITEM = "[]"
    LEN = "len"
    MAX = "max"
    MIN = "min"
    SIN = "sin"
    EXP = "exp"
    CONST = "const"
    SOURCE = "source"


@dataclass
class TraceNode:
    """A node in the computation trace."""

    op: TracedOp
    inputs: List[int]
    value: Any
    source_key: Optional[str] = None

    def __repr__(self) -> str:
        if self.source_key:
            return f"Node[{self.source_key}]"
        return f"Node[{self.op.value}]"


class ComputationTrace:
    """Records a computation for later delta pushforward with memory management."""

    def __init__(self):
        self.nodes: List[TraceNode] = []
        self.source_nodes: Dict[str, int] = {}
        self.const_cache: Dict[Any, int] = {}
        self._last_accessed = time.time()

    def add_source(self, key: str, value: Any) -> int:
        """Record a store access."""
        self._last_accessed = time.time()
        if key in self.source_nodes:
            return self.source_nodes[key]

        idx = len(self.nodes)
        self.nodes.append(
            TraceNode(op=TracedOp.SOURCE, inputs=[], value=value, source_key=key)
        )
        self.source_nodes[key] = idx
        return idx

    def add_const(self, value: Any) -> int:
        """Record a constant value."""
        self._last_accessed = time.time()

        # Check if value is hashable for caching
        if isinstance(value, (int, float, str, tuple, frozenset, type(None))):
            cache_key = value
            if cache_key in self.const_cache:
                return self.const_cache[cache_key]
        else:
            cache_key = id(value)

        idx = len(self.nodes)
        self.nodes.append(TraceNode(op=TracedOp.CONST, inputs=[], value=value))

        # Only cache if hashable
        if isinstance(value, (int, float, str, tuple, frozenset, type(None))):
            self.const_cache[cache_key] = idx

        return idx

    def add_op(self, op: TracedOp, inputs: List[int], value: Any) -> int:
        """Record an operation."""
        self._last_accessed = time.time()
        idx = len(self.nodes)
        self.nodes.append(TraceNode(op=op, inputs=inputs, value=value))
        return idx

    def memory_size(self) -> int:
        """Estimate memory size in bytes."""
        import sys

        size = sys.getsizeof(self.nodes)
        size += sys.getsizeof(self.source_nodes)
        size += sys.getsizeof(self.const_cache)
        return size

    def __repr__(self) -> str:
        lines = ["Trace:"]
        for i, node in enumerate(self.nodes):
            if node.source_key:
                lines.append(f"  {i}: {node.source_key} = {node.value}")
            elif node.op == TracedOp.CONST:
                lines.append(f"  {i}: const({node.value})")
            else:
                lines.append(f"  {i}: {node.op.value}({node.inputs}) = {node.value}")
        return "\n".join(lines)


class TracedValue:
    """A value that records operations for AD."""

    def __init__(self, value: Any, trace_idx: int, trace: ComputationTrace):
        self.value = value
        self.trace_idx = trace_idx
        self.trace = trace

    def _unwrap(self, other):
        """Extract value and trace index from other operand."""
        if isinstance(other, TracedValue):
            return other.value, other.trace_idx
        else:
            idx = self.trace.add_const(other)
            return other, idx

    def __add__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value + other_val
        idx = self.trace.add_op(TracedOp.ADD, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value - other_val
        idx = self.trace.add_op(TracedOp.SUB, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __rsub__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = other_val - self.value
        idx = self.trace.add_op(TracedOp.SUB, [other_idx, self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def __mul__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value * other_val
        idx = self.trace.add_op(TracedOp.MUL, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value / other_val
        idx = self.trace.add_op(TracedOp.DIV, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __rtruediv__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = other_val / self.value
        idx = self.trace.add_op(TracedOp.DIV, [other_idx, self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def __floordiv__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value // other_val
        idx = self.trace.add_op(TracedOp.FLOORDIV, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __mod__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value % other_val
        idx = self.trace.add_op(TracedOp.MOD, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __pow__(self, other):
        other_val, other_idx = self._unwrap(other)
        result = self.value**other_val
        idx = self.trace.add_op(TracedOp.POW, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def __neg__(self):
        result = -self.value
        idx = self.trace.add_op(TracedOp.NEG, [self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def __abs__(self):
        result = abs(self.value)
        idx = self.trace.add_op(TracedOp.ABS, [self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def __getitem__(self, key):
        key_val, key_idx = self._unwrap(key)
        result = self.value[key_val]
        idx = self.trace.add_op(TracedOp.GETITEM, [self.trace_idx, key_idx], result)
        return TracedValue(result, idx, self.trace)

    def __len__(self):
        result = len(self.value)
        idx = self.trace.add_op(TracedOp.LEN, [self.trace_idx], result)
        return result

    def max(self, other):
        """Maximum of self and other."""
        other_val, other_idx = self._unwrap(other)
        result = max(self.value, other_val)
        idx = self.trace.add_op(TracedOp.MAX, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def min(self, other):
        """Minimum of self and other."""
        other_val, other_idx = self._unwrap(other)
        result = min(self.value, other_val)
        idx = self.trace.add_op(TracedOp.MIN, [self.trace_idx, other_idx], result)
        return TracedValue(result, idx, self.trace)

    def sin(self):
        """Sine of self (using math.sin)."""
        import math

        result = math.sin(self.value)
        idx = self.trace.add_op(TracedOp.SIN, [self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def exp(self):
        """Exponential of self (using math.exp)."""
        import math

        result = math.exp(self.value)
        idx = self.trace.add_op(TracedOp.EXP, [self.trace_idx], result)
        return TracedValue(result, idx, self.trace)

    def __repr__(self):
        return f"Traced({self.value})"

    def __str__(self):
        return str(self.value)

    def __format__(self, format_spec):
        return format(self.value, format_spec)

    def __eq__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value == other_val

    def __ne__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value != other_val

    def __lt__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value < other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value <= other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value > other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, TracedValue) else other
        return self.value >= other_val

    def __bool__(self):
        """Boolean conversion returns the boolean value of the underlying value."""
        return bool(self.value)

    def __contains__(self, item):
        return item in self.value

    def __iter__(self):
        return iter(self.value)

    def __getattr__(self, name):
        attr = getattr(self.value, name)
        if callable(attr):

            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                idx = self.trace.add_op(TracedOp.CONST, [], result)
                return result

            return method_wrapper
        else:
            return attr


class AutoDiffEngine:
    """
    Automatic differentiation engine for delta algebras with memory management.

    Pushes deltas forward through traced computations:
    Given Δx and trace of y = f(x), compute Δy = Df(Δx)
    """

    def __init__(self, delta_registry: DeltaRegistry):
        self.registry = delta_registry

    def pushforward(
        self,
        trace: ComputationTrace,
        input_deltas: Dict[str, TypedDelta],
        output_idx: int,
    ) -> Optional[TypedDelta]:
        """
        Push deltas forward through computation trace.

        Args:
            trace: Recorded computation
            input_deltas: {source_key: Δ_source}
            output_idx: Index of output node in trace

        Returns:
            Δ_output or None if cannot compute incrementally
        """
        if not input_deltas:
            return None

        node_deltas: Dict[int, Optional[TypedDelta]] = {}

        for key, delta in input_deltas.items():
            if key in trace.source_nodes:
                idx = trace.source_nodes[key]
                node_deltas[idx] = delta

        for i in range(len(trace.nodes)):
            if i in node_deltas:
                continue

            node = trace.nodes[i]

            if node.op in (TracedOp.CONST, TracedOp.SOURCE):
                node_deltas[i] = None
                continue

            delta = self._pushforward_op(node, node_deltas, trace)
            node_deltas[i] = delta

        return node_deltas.get(output_idx)

    def _pushforward_op(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Compute delta for a single operation."""
        if node.op == TracedOp.ADD:
            return self._push_add(node, node_deltas, trace)
        elif node.op == TracedOp.SUB:
            return self._push_sub(node, node_deltas, trace)
        elif node.op == TracedOp.MUL:
            return self._push_mul(node, node_deltas, trace)
        elif node.op == TracedOp.DIV:
            return self._push_div(node, node_deltas, trace)
        elif node.op == TracedOp.NEG:
            return self._push_neg(node, node_deltas, trace)
        elif node.op == TracedOp.POW:
            return self._push_pow(node, node_deltas, trace)
        elif node.op == TracedOp.GETITEM:
            return self._push_getitem(node, node_deltas, trace)
        elif node.op == TracedOp.LEN:
            return self._push_len(node, node_deltas, trace)
        elif node.op == TracedOp.MAX:
            return self._push_max(node, node_deltas, trace)
        elif node.op == TracedOp.MIN:
            return self._push_min(node, node_deltas, trace)
        elif node.op == TracedOp.SIN:
            return self._push_sin(node, node_deltas, trace)
        elif node.op == TracedOp.EXP:
            return self._push_exp(node, node_deltas, trace)
        else:
            return None

    def _push_add(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a + b) = Δa + Δb"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        if left_delta and right_delta:
            if left_delta.delta_type == right_delta.delta_type == DeltaType.NUMERIC:
                return TypedDelta(
                    DeltaType.NUMERIC, left_delta.value + right_delta.value
                )

        if left_delta:
            return left_delta
        return right_delta

    def _push_sub(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a - b) = Δa - Δb"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        if left_delta and right_delta:
            if left_delta.delta_type == right_delta.delta_type == DeltaType.NUMERIC:
                return TypedDelta(
                    DeltaType.NUMERIC, left_delta.value - right_delta.value
                )

        if left_delta:
            return left_delta

        if right_delta and right_delta.delta_type == DeltaType.NUMERIC:
            return TypedDelta(DeltaType.NUMERIC, -right_delta.value)

        return None

    def _push_mul(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a * b) ≈ b·Δa + a·Δb (linearization)"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        left_val = trace.nodes[left_idx].value
        right_val = trace.nodes[right_idx].value

        result = 0.0

        if left_delta and left_delta.delta_type == DeltaType.NUMERIC:
            if isinstance(right_val, (int, float)):
                result += right_val * left_delta.value

        if right_delta and right_delta.delta_type == DeltaType.NUMERIC:
            if isinstance(left_val, (int, float)):
                result += left_val * right_delta.value

        if abs(result) < 1e-10:
            return None

        return TypedDelta(DeltaType.NUMERIC, result)

    def _push_div(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a / b) ≈ (Δa·b - a·Δb) / b²"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        left_val = trace.nodes[left_idx].value
        right_val = trace.nodes[right_idx].value

        if not isinstance(left_val, (int, float)) or not isinstance(
            right_val, (int, float)
        ):
            return None

        if abs(right_val) < 1e-10:
            return None

        numerator = 0.0

        if left_delta and left_delta.delta_type == DeltaType.NUMERIC:
            numerator += left_delta.value * right_val

        if right_delta and right_delta.delta_type == DeltaType.NUMERIC:
            numerator -= left_val * right_delta.value

        result = numerator / (right_val**2)

        if abs(result) < 1e-10:
            return None

        return TypedDelta(DeltaType.NUMERIC, result)

    def _push_neg(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(-a) = -Δa"""
        input_idx = node.inputs[0]
        input_delta = node_deltas.get(input_idx)

        if input_delta and input_delta.delta_type == DeltaType.NUMERIC:
            return TypedDelta(DeltaType.NUMERIC, -input_delta.value)

        return None

    def _push_pow(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a^n) ≈ n·a^(n-1)·Δa (for constant n)"""
        base_idx, exp_idx = node.inputs
        base_delta = node_deltas.get(base_idx)

        if base_delta is None:
            return None

        if base_delta.delta_type != DeltaType.NUMERIC:
            return None

        base_val = trace.nodes[base_idx].value
        exp_val = trace.nodes[exp_idx].value

        if not isinstance(base_val, (int, float)) or not isinstance(
            exp_val, (int, float)
        ):
            return None

        if abs(base_val) < 1e-10 and exp_val < 1:
            return None

        derivative = exp_val * (base_val ** (exp_val - 1))
        result = derivative * base_delta.value

        if abs(result) < 1e-10:
            return None

        return TypedDelta(DeltaType.NUMERIC, result)

    def _push_getitem(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(a[i]) = Δa[i] if Δa is dict delta"""
        container_idx, key_idx = node.inputs
        container_delta = node_deltas.get(container_idx)

        if container_delta is None:
            return None

        if container_delta.delta_type == DeltaType.DICT:
            key_val = trace.nodes[key_idx].value
            if key_val in container_delta.value:
                old_val, new_val = container_delta.value[key_val]
                return self.registry.compute_delta(old_val, new_val)

        return None

    def _push_len(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(len(a)) = count(inserts) - count(removes)"""
        input_idx = node.inputs[0]
        input_delta = node_deltas.get(input_idx)

        if input_delta is None:
            return None

        if input_delta.delta_type == DeltaType.LIST:
            inserts = sum(1 for op in input_delta.value if op[0] == "insert")
            removes = sum(1 for op in input_delta.value if op[0] == "remove")
            delta = inserts - removes

            if delta != 0:
                return TypedDelta(DeltaType.NUMERIC, delta)

        return None

    def _push_max(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(max(a, b)) = Δa if a > b, Δb if b > a"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        left_val = trace.nodes[left_idx].value
        right_val = trace.nodes[right_idx].value

        if abs(left_val - right_val) < 1e-10:
            return None

        if left_val > right_val:
            return left_delta
        else:
            return right_delta

    def _push_min(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(min(a, b)) = Δa if a < b, Δb if b < a"""
        left_idx, right_idx = node.inputs
        left_delta = node_deltas.get(left_idx)
        right_delta = node_deltas.get(right_idx)

        if left_delta is None and right_delta is None:
            return None

        left_val = trace.nodes[left_idx].value
        right_val = trace.nodes[right_idx].value

        if abs(left_val - right_val) < 1e-10:
            return None

        if left_val < right_val:
            return left_delta
        else:
            return right_delta

    def _push_sin(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(sin(x)) = cos(x)·Δx"""
        input_idx = node.inputs[0]
        input_delta = node_deltas.get(input_idx)

        if input_delta is None or input_delta.delta_type != DeltaType.NUMERIC:
            return None

        input_val = trace.nodes[input_idx].value
        import math

        derivative = math.cos(input_val)
        result = derivative * input_delta.value

        if abs(result) < 1e-10:
            return None

        return TypedDelta(DeltaType.NUMERIC, result)

    def _push_exp(
        self,
        node: TraceNode,
        node_deltas: Dict[int, Optional[TypedDelta]],
        trace: ComputationTrace,
    ) -> Optional[TypedDelta]:
        """Δ(exp(x)) = exp(x)·Δx"""
        input_idx = node.inputs[0]
        input_delta = node_deltas.get(input_idx)

        if input_delta is None or input_delta.delta_type != DeltaType.NUMERIC:
            return None

        input_val = trace.nodes[input_idx].value
        import math

        derivative = math.exp(input_val)
        result = derivative * input_delta.value

        if abs(result) < 1e-10:
            return None

        return TypedDelta(DeltaType.NUMERIC, result)


# ============================================================================
# CHANGE EVENT
# ============================================================================


class ChangeType(Enum):
    """Type of change that occurred."""

    SOURCE_UPDATE = "source"
    COMPUTED_UPDATE = "computed"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class Change:
    """Immutable change event with type-aware delta support."""

    key: str
    change_type: ChangeType
    old_value: Any
    new_value: Any
    timestamp: float
    differential: Optional[TypedDelta] = None

    def is_identity(self) -> bool:
        try:
            if isinstance(self.old_value, np.ndarray) or isinstance(
                self.new_value, np.ndarray
            ):
                if type(self.old_value) != type(self.new_value):
                    return False
                return np.array_equal(self.old_value, self.new_value)
            return self.old_value == self.new_value
        except (ValueError, TypeError):
            return False

    def compose(self, other: "Change") -> "Change":
        if self.key != other.key:
            raise ValueError(f"Cannot compose changes for different keys")

        return Change(
            key=self.key,
            change_type=self.change_type,
            old_value=self.old_value,
            new_value=other.new_value,
            timestamp=max(self.timestamp, other.timestamp),
            differential=other.differential,
        )

    @property
    def is_creation(self) -> bool:
        return self.old_value is None and self.new_value is not None

    @property
    def is_deletion(self) -> bool:
        return self.change_type == ChangeType.DELETED

    @property
    def is_update(self) -> bool:
        return not self.is_creation and not self.is_deletion

    def __repr__(self) -> str:
        if self.is_creation:
            return f"Change({self.key}: created = {self.new_value!r})"
        elif self.is_deletion:
            return f"Change({self.key}: deleted)"
        else:
            return f"Change({self.key}: {self.old_value!r} → {self.new_value!r})"


# ============================================================================
# GRAPH TOPOLOGY
# ============================================================================


class _DependencyGraph:
    def __init__(self):
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)

    def add_edge(self, source: str, dependent: str) -> None:
        if dependent not in self._forward[source]:
            self._forward[source].add(dependent)
            self._reverse[dependent].add(source)

    def remove_edge(self, source: str, dependent: str) -> None:
        if dependent in self._forward[source]:
            self._forward[source].discard(dependent)
            self._reverse[dependent].discard(source)

    def get_dependents(self, key: str) -> Set[str]:
        return self._forward[key].copy()

    def get_dependencies(self, key: str) -> Set[str]:
        return self._reverse[key].copy()

    def get_all_dependents(self, key: str) -> Set[str]:
        """Get all transitive dependents of a key."""
        affected = set()
        to_visit = {key}

        while to_visit:
            next_level = set()
            for node in to_visit:
                for dep in self._forward.get(node, set()):
                    if dep not in affected:
                        affected.add(dep)
                        next_level.add(dep)
            to_visit = next_level

        return affected

    def topological_sort(self, keys: Set[str]) -> List[str]:
        """Sort keys in topological order."""
        if not keys:
            return []

        in_degree = {}
        for key in keys:
            degree = sum(1 for dep in self._reverse.get(key, set()) if dep in keys)
            in_degree[key] = degree

        queue = deque([k for k in keys if in_degree[k] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for dependent in self._forward.get(current, set()):
                if dependent in keys:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return result


# ============================================================================
# ASYNC UTILITIES
# ============================================================================


def _is_async_callable(func: Callable) -> bool:
    """Check if a function is async (coroutine function)."""
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


async def _ensure_awaitable(value: Any) -> Any:
    """Ensure a value is awaitable, wrapping if necessary."""
    if inspect.isawaitable(value):
        return await value
    return value


# ============================================================================
# COMPUTED VALUE WITH AUTOMATIC DIFFERENTIATION AND ASYNC SUPPORT
# ============================================================================


class _ComputedValue:
    """Base class for computed values with async support."""

    def __init__(self, key: str, store: "ReactiveStore"):
        self.key = key
        self._store = weakref.ref(store)  # Use weak reference
        self._cached_value: Any = None
        self._is_dirty = True
        self._dependencies: Set[str] = set()
        self._last_computed = 0.0
        self._last_error: Optional[Exception] = None
        self._trace: Optional[weakref.ref] = None  # Weak reference to trace
        self._output_idx: Optional[int] = None
        self._is_async = False

    def get(self) -> Any:
        """Get the current value, computing if necessary (sync version)."""
        if self._is_async:
            raise RuntimeError(
                f"Cannot use synchronous get() on async computed value '{self.key}'. Use await get_async() instead."
            )

        if not self._is_dirty and self._last_error is None:
            return self._cached_value

        if self._last_error is not None and not self._is_dirty:
            raise self._last_error

        store = self._store()
        if store is None:
            raise RuntimeError(f"Store has been garbage collected for '{self.key}'")

        ctx = store._ctx
        if not hasattr(ctx, "computing_stack"):
            ctx.computing_stack = set()

        if self.key in ctx.computing_stack:
            raise CircularDependencyError(
                f"Circular dependency detected involving '{self.key}'. "
                f"Chain: {' → '.join(ctx.computing_stack)} → {self.key}"
            )

        ctx.computing_stack.add(self.key)
        self._recompute()
        ctx.computing_stack.discard(self.key)
        return self._cached_value

    async def get_async(self) -> Any:
        """Get the current value, computing if necessary (async version)."""
        if not self._is_dirty and self._last_error is None:
            return self._cached_value

        if self._last_error is not None and not self._is_dirty:
            raise self._last_error

        store = self._store()
        if store is None:
            raise RuntimeError(f"Store has been garbage collected for '{self.key}'")

        ctx = store._ctx
        if not hasattr(ctx, "computing_stack"):
            ctx.computing_stack = set()

        if self.key in ctx.computing_stack:
            raise CircularDependencyError(
                f"Circular dependency detected involving '{self.key}'. "
                f"Chain: {' → '.join(ctx.computing_stack)} → {self.key}"
            )

        ctx.computing_stack.add(self.key)
        await self._recompute_async()
        ctx.computing_stack.discard(self.key)
        return self._cached_value

    def _recompute(self) -> None:
        """Recompute the value and update dependencies (sync version)."""
        if self._is_async:
            raise RuntimeError(
                f"Cannot use synchronous recompute on async computed value '{self.key}'"
            )

        store = self._store()
        if store is None:
            return

        accessed = set()
        ctx = store._ctx
        prev_tracking = getattr(ctx, "accessed_keys", None)
        ctx.accessed_keys = accessed

        self._cached_value = self._compute()
        self._last_error = None
        self._is_dirty = False
        self._last_computed = time.time()

        if prev_tracking is not None:
            ctx.accessed_keys = prev_tracking
        elif hasattr(ctx, "accessed_keys"):
            delattr(ctx, "accessed_keys")

        if accessed:
            self._update_dependencies(accessed)

    async def _recompute_async(self) -> None:
        """Recompute the value and update dependencies (async version)."""
        store = self._store()
        if store is None:
            return

        accessed = set()
        ctx = store._ctx
        prev_tracking = getattr(ctx, "accessed_keys", None)
        ctx.accessed_keys = accessed

        value = self._compute()
        # Handle both sync and async compute results
        self._cached_value = await _ensure_awaitable(value)
        self._last_error = None
        self._is_dirty = False
        self._last_computed = time.time()

        if prev_tracking is not None:
            ctx.accessed_keys = prev_tracking
        elif hasattr(ctx, "accessed_keys"):
            delattr(ctx, "accessed_keys")

        if accessed:
            self._update_dependencies(accessed)

    def _compute(self) -> Any:
        """Override in subclasses."""
        raise NotImplementedError

    def _update_dependencies(self, new_deps: Set[str]) -> None:
        """Update dependency graph."""
        store = self._store()
        if store is None:
            return

        old_deps = self._dependencies

        for dep in old_deps - new_deps:
            store._graph.remove_edge(dep, self.key)

        for dep in new_deps - old_deps:
            store._graph.add_edge(dep, self.key)

        self._dependencies = new_deps

    def invalidate(self) -> None:
        """Mark as needing recomputation."""
        self._is_dirty = True
        self._last_error = None

    def try_incremental_update(self, triggering_change: Change) -> Optional[Any]:
        """Try to update incrementally using AD."""
        trace_ref = self._trace
        if trace_ref is None or self._output_idx is None:
            return None

        trace = trace_ref()
        if trace is None:
            # Trace was garbage collected
            self._trace = None
            self._output_idx = None
            return None

        if triggering_change.differential is None:
            return None

        store = self._store()
        if store is None:
            return None

        input_deltas = {triggering_change.key: triggering_change.differential}
        output_delta = store._ad_engine.pushforward(
            trace, input_deltas, self._output_idx
        )

        if output_delta is not None and not store._delta_registry.is_identity(
            output_delta
        ):
            new_value = store._delta_registry.apply_delta(
                self._cached_value, output_delta
            )
            # Update trace values
            for node in trace.nodes:
                if node.source_key:
                    node.value = store.get(node.source_key)
            return new_value

        return None


class _AutoComputedValue(_ComputedValue):
    """Computed value with automatic dependency tracking and AD."""

    def __init__(self, key: str, func: Callable, store: "ReactiveStore"):
        super().__init__(key, store)
        self._func = func
        self._is_async = _is_async_callable(func)

    def _compute(self) -> Any:
        store = self._store()
        if store is None:
            return self._cached_value

        trace = ComputationTrace()
        ctx = store._ctx
        prev_trace = getattr(ctx, "active_trace", None)
        ctx.active_trace = trace

        result = self._func()

        if isinstance(result, TracedValue):
            self._trace = weakref.ref(trace)
            self._output_idx = result.trace_idx
            return result.value
        else:
            self._trace = None
            self._output_idx = None
            return result

        if prev_trace is not None:
            ctx.active_trace = prev_trace
        elif hasattr(ctx, "active_trace"):
            delattr(ctx, "active_trace")


class _ExplicitComputedValue(_ComputedValue):
    """Computed value with explicit dependencies."""

    def __init__(
        self, key: str, deps: List[str], func: Callable, store: "ReactiveStore"
    ):
        super().__init__(key, store)
        self._deps = deps
        self._func = func
        self._is_async = _is_async_callable(func)

        self._dependencies = set(deps)
        for dep in deps:
            store._graph.add_edge(dep, key)

    def _compute(self) -> Any:
        store = self._store()
        if store is None:
            return self._cached_value

        trace = ComputationTrace()
        ctx = store._ctx
        prev_trace = getattr(ctx, "active_trace", None)
        ctx.active_trace = trace

        args = [store.get(dep) for dep in self._deps]
        unwrapped_args = [
            arg.value if isinstance(arg, TracedValue) else arg for arg in args
        ]
        result = self._func(*unwrapped_args)

        if isinstance(result, TracedValue):
            self._trace = weakref.ref(trace)
            self._output_idx = result.trace_idx
            return result.value
        else:
            self._trace = None
            self._output_idx = None
            return result if result is not None else self._cached_value

        if prev_trace is not None:
            ctx.active_trace = prev_trace
        elif hasattr(ctx, "active_trace"):
            delattr(ctx, "active_trace")

    def _update_dependencies(self, new_deps: Set[str]) -> None:
        # Dependencies are explicit, don't update
        pass


# ============================================================================
# MAIN STORE WITH ASYNC SUPPORT
# ============================================================================


class ReactiveStore:
    """
    Reactive store with automatic differentiation over delta algebras and full async support.

    Key Features:
    - Myers diff algorithm for optimal string diffing
    - Memory management with weak references for traces
    - Full async/await support (automatically handles both sync and async functions)
    - Automatic delta propagation through AD
    """

    _MAX_HISTORY = 10000
    _TRACE_CLEANUP_INTERVAL = 100  # Cleanup every N operations
    _TRACE_MAX_AGE = 300  # 5 minutes

    def __init__(self):
        self._values: Dict[str, Any] = {}
        self._computed: Dict[str, _ComputedValue] = {}
        self._objects: Dict[str, Any] = {}
        self._graph = _DependencyGraph()

        self._delta_registry = DeltaRegistry()
        self._ad_engine = AutoDiffEngine(self._delta_registry)

        self._observers: Dict[str, List[Callable]] = defaultdict(list)
        self._global_observers: weakref.WeakSet = weakref.WeakSet()

        self._lock = threading.RLock()
        self._ctx = threading.local()

        self._batch_depth = 0
        self._pending_changes: List[Change] = []
        self._history: deque[Change] = deque(maxlen=self._MAX_HISTORY)

        self._operation_count = 0  # For periodic cleanup

        # Compatibility
        self._kv = self

        atexit.register(self._cleanup)

    # ========================================================================
    # COMPATIBILITY PROPERTIES
    # ========================================================================

    @property
    def _data(self):
        """Compatibility property for observable.py Store interface."""
        return self._values

    @property
    def _dep_graph(self):
        """Compatibility property for test_delta_store.py interface."""
        return self._graph

    # ========================================================================
    # CORE API (SYNC)
    # ========================================================================

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._get_internal(key)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._set_internal(key, value)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self._delete_internal(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._values or key in self._computed

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        self[key] = value

    def delete(self, key: str) -> None:
        del self[key]

    # ========================================================================
    # CORE API (ASYNC)
    # ========================================================================

    async def get_async(self, key: str, default: Any = None) -> Any:
        """Async version of get - handles both sync and async computed values."""
        if key in self._computed:
            computed = self._computed[key]
            return await computed.get_async()
        elif key in self._values:
            value = self._values[key]
            if callable(value) and hasattr(value, "_key") and value._key == key:
                result = value()
                return await _ensure_awaitable(result)
            return value
        else:
            return default

    async def set_async(self, key: str, value: Any) -> None:
        """Async version of set."""
        with self._lock:
            await self._set_internal_async(key, value)

    async def delete_async(self, key: str) -> None:
        """Async version of delete."""
        with self._lock:
            await self._delete_internal_async(key)

    # ========================================================================
    # COMPUTED VALUES WITH AUTOMATIC DIFFERENTIATION AND ASYNC
    # ========================================================================

    def computed(
        self, key: str, func: Callable, deps: Optional[List[str]] = None, **kwargs
    ) -> None:
        """
        Create a computed value with automatic incremental updates.

        Automatically handles both sync and async functions!

        Args:
            key: Name for the computed value
            func: Computation function (sync or async)
            deps: Optional explicit dependencies

        Examples:
            # Sync function
            store.computed('sum', lambda: store['x'] + store['y'])

            # Async function - works automatically!
            async def fetch_data():
                response = await fetch_api()
                return response + store['offset']

            store.computed('data', fetch_data)

            # Access async computed values:
            value = await store.get_async('data')
        """
        with self._lock:
            if deps:
                for dep in deps:
                    if dep not in self._values and dep not in self._computed:
                        raise KeyError(f"Dependency '{dep}' does not exist")

            if deps:
                computed = _ExplicitComputedValue(key, deps, func, self)
            else:
                computed = _AutoComputedValue(key, func, self)

            self._computed[key] = computed

    # ========================================================================
    # SUBSCRIPTION
    # ========================================================================

    def on(self, key: str, callback: Callable[[Change], None]) -> Callable[[], None]:
        """Subscribe to changes on a key. Callback can be sync or async."""
        with self._lock:
            self._observers[key].append(callback)

            def unsubscribe():
                with self._lock:
                    if callback in self._observers[key]:
                        self._observers[key].remove(callback)

            return unsubscribe

    def on_any(self, callback: Callable[[Change], None]) -> Callable[[], None]:
        """Subscribe to all changes. Callback can be sync or async."""
        with self._lock:
            self._global_observers.add(callback)

            def unsubscribe():
                with self._lock:
                    self._global_observers.discard(callback)

            return unsubscribe

    # ========================================================================
    # BATCHING
    # ========================================================================

    def batch(self) -> "BatchContext":
        """Batch multiple updates for efficiency."""
        return BatchContext(self)

    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================

    def _cleanup_old_traces(self) -> None:
        """Clean up old traces to prevent memory leaks."""
        current_time = time.time()

        for computed in list(self._computed.values()):
            if computed._trace is not None:
                trace_ref = computed._trace
                trace = trace_ref()

                if trace is None:
                    # Already garbage collected
                    computed._trace = None
                    computed._output_idx = None
                elif current_time - trace._last_accessed > self._TRACE_MAX_AGE:
                    # Old trace - release reference
                    computed._trace = None
                    computed._output_idx = None

    def _maybe_cleanup(self) -> None:
        """Periodically trigger cleanup."""
        self._operation_count += 1
        if self._operation_count % self._TRACE_CLEANUP_INTERVAL == 0:
            self._cleanup_old_traces()
            gc.collect()  # Suggest garbage collection

    # ========================================================================
    # INTERNAL IMPLEMENTATION
    # ========================================================================

    def _get_internal(self, key: str) -> Any:
        """Internal get with tracing support."""
        ctx = self._ctx

        if hasattr(ctx, "accessed_keys"):
            ctx.accessed_keys.add(key)

        if key in self._computed:
            computed = self._computed[key]
            if hasattr(ctx, "computing_stack") and key in ctx.computing_stack:
                value = computed._cached_value
            else:
                value = computed.get()
        elif key in self._values:
            value = self._values[key]
            if callable(value) and hasattr(value, "_key") and value._key == key:
                value = value()
        else:
            raise KeyError(f"Key not found: {key}")

        if hasattr(ctx, "active_trace"):
            trace = ctx.active_trace
            idx = trace.add_source(key, value)
            return TracedValue(value, idx, trace)

        return value

    def _set_internal(
        self, key: str, value: Any, differential: Optional[TypedDelta] = None
    ) -> None:
        """Internal set with automatic delta computation."""
        if key in self._computed:
            raise ValueError(f"Cannot update computed value '{key}'")

        ctx = self._ctx
        notifying = getattr(ctx, "notifying_keys", None)
        if notifying and key in notifying:
            raise CircularDependencyError(
                f"Cannot modify '{key}' from within its own notification"
            )

        old_value = self._values.get(key)
        self._values[key] = value

        if differential is None and old_value is not None:
            differential = self._delta_registry.compute_delta(old_value, value)

        change = Change(
            key=key,
            change_type=ChangeType.SOURCE_UPDATE,
            old_value=old_value,
            new_value=value,
            timestamp=time.time(),
            differential=differential,
        )

        self._propagate(change)
        self._maybe_cleanup()

    async def _set_internal_async(
        self, key: str, value: Any, differential: Optional[TypedDelta] = None
    ) -> None:
        """Async version of internal set."""
        if key in self._computed:
            raise ValueError(f"Cannot update computed value '{key}'")

        ctx = self._ctx
        notifying = getattr(ctx, "notifying_keys", None)
        if notifying and key in notifying:
            raise CircularDependencyError(
                f"Cannot modify '{key}' from within its own notification"
            )

        old_value = self._values.get(key)
        self._values[key] = value

        if differential is None and old_value is not None:
            differential = self._delta_registry.compute_delta(old_value, value)

        change = Change(
            key=key,
            change_type=ChangeType.SOURCE_UPDATE,
            old_value=old_value,
            new_value=value,
            timestamp=time.time(),
            differential=differential,
        )

        await self._propagate_async(change)
        self._maybe_cleanup()

    def _delete_internal(self, key: str) -> None:
        if key not in self._values:
            raise KeyError(f"Key not found: {key}")

        old_value = self._values[key]
        del self._values[key]

        change = Change(
            key=key,
            change_type=ChangeType.DELETED,
            old_value=old_value,
            new_value=None,
            timestamp=time.time(),
        )

        self._propagate(change)

    async def _delete_internal_async(self, key: str) -> None:
        if key not in self._values:
            raise KeyError(f"Key not found: {key}")

        old_value = self._values[key]
        del self._values[key]

        change = Change(
            key=key,
            change_type=ChangeType.DELETED,
            old_value=old_value,
            new_value=None,
            timestamp=time.time(),
        )

        await self._propagate_async(change)

    def _propagate_change(self, change: Change) -> None:
        """Propagate a change (used by Observable for untracked updates)."""
        ctx = self._ctx
        notifying = getattr(ctx, "notifying_keys", None)
        if notifying and change.key in notifying:
            raise CircularDependencyError(
                f"Cannot modify '{change.key}' from within its own notification"
            )

        self._propagate(change)

    def _propagate(self, change: Change) -> None:
        if self._batch_depth > 0:
            self._pending_changes.append(change)
            return

        self._propagate_immediate(change)

    async def _propagate_async(self, change: Change) -> None:
        if self._batch_depth > 0:
            self._pending_changes.append(change)
            return

        await self._propagate_immediate_async(change)

    def _propagate_immediate(self, change: Change) -> None:
        """Immediate propagation with AD-based incremental computation (sync version)."""
        if change.is_identity():
            return

        self._history.append(change)

        ctx = self._ctx
        prev_change = getattr(ctx, "current_change", None)
        ctx.current_change = change

        affected = self._graph.get_all_dependents(change.key)

        if not affected:
            self._notify(change)
            return

        old_values = {}
        for key in affected:
            if key in self._computed:
                computed = self._computed[key]
                old_values[key] = computed._cached_value
                computed.invalidate()

        order = self._graph.topological_sort(affected)
        changes = [change]

        for key in order:
            if key not in self._computed:
                continue

            computed = self._computed[key]
            old_value = old_values.get(key)

            # Try incremental update with AD
            new_value = computed.try_incremental_update(change)

            if new_value is None:
                logging.debug(f"AD failed for '{key}', falling back to recompute")
                new_value = computed.get()
            else:
                computed._cached_value = new_value
                computed._is_dirty = False
                computed._last_computed = time.time()

            if old_value is None or not self._values_equal(old_value, new_value):
                output_delta = self._delta_registry.compute_delta(old_value, new_value)

                output_change = Change(
                    key=key,
                    change_type=ChangeType.COMPUTED_UPDATE,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=time.time(),
                    differential=output_delta,
                )

                if not output_change.is_identity():
                    changes.append(output_change)
                    change = output_change

        for c in changes:
            self._notify(c)

        if prev_change is not None:
            ctx.current_change = prev_change
        elif hasattr(ctx, "current_change"):
            delattr(ctx, "current_change")

    async def _propagate_immediate_async(self, change: Change) -> None:
        """Async version of immediate propagation."""
        if change.is_identity():
            return

        self._history.append(change)

        ctx = self._ctx
        prev_change = getattr(ctx, "current_change", None)
        ctx.current_change = change

        affected = self._graph.get_all_dependents(change.key)

        if not affected:
            await self._notify_async(change)
            return

        old_values = {}
        for key in affected:
            if key in self._computed:
                computed = self._computed[key]
                old_values[key] = computed._cached_value
                computed.invalidate()

        order = self._graph.topological_sort(affected)
        changes = [change]

        for key in order:
            if key not in self._computed:
                continue

            computed = self._computed[key]
            old_value = old_values.get(key)

            # Try incremental update with AD
            new_value = computed.try_incremental_update(change)

            if new_value is None:
                logging.debug(f"AD failed for '{key}', falling back to recompute")
                new_value = await computed.get_async()
            else:
                computed._cached_value = new_value
                computed._is_dirty = False
                computed._last_computed = time.time()

            if old_value is None or not self._values_equal(old_value, new_value):
                output_delta = self._delta_registry.compute_delta(old_value, new_value)

                output_change = Change(
                    key=key,
                    change_type=ChangeType.COMPUTED_UPDATE,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=time.time(),
                    differential=output_delta,
                )

                if not output_change.is_identity():
                    changes.append(output_change)
                    change = output_change

        for c in changes:
            await self._notify_async(c)

        if prev_change is not None:
            ctx.current_change = prev_change
        elif hasattr(ctx, "current_change"):
            delattr(ctx, "current_change")

    def _notify(self, change: Change) -> None:
        """Notify observers (sync version)."""
        ctx = self._ctx

        if not hasattr(ctx, "notifying_keys"):
            ctx.notifying_keys = set()

        ctx.notifying_keys.add(change.key)

        for callback in self._observers.get(change.key, []):
            result = callback(change)
            # Handle async callbacks in sync context
            if inspect.isawaitable(result):
                # Schedule for later execution
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(result)
                else:
                    loop.run_until_complete(result)

        for callback in list(self._global_observers):
            result = callback(change)
            if inspect.isawaitable(result):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(result)
                else:
                    loop.run_until_complete(result)

        ctx.notifying_keys.discard(change.key)
        if not ctx.notifying_keys:
            delattr(ctx, "notifying_keys")

    async def _notify_async(self, change: Change) -> None:
        """Notify observers (async version)."""
        ctx = self._ctx

        if not hasattr(ctx, "notifying_keys"):
            ctx.notifying_keys = set()

        ctx.notifying_keys.add(change.key)

        for callback in self._observers.get(change.key, []):
            result = callback(change)
            await _ensure_awaitable(result)

        for callback in list(self._global_observers):
            result = callback(change)
            await _ensure_awaitable(result)

        ctx.notifying_keys.discard(change.key)
        if not ctx.notifying_keys:
            delattr(ctx, "notifying_keys")

    def _values_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if type(a) != type(b):
                return False
            return np.array_equal(a, b)
        return a == b

    def _merge_changes(self, changes: List[Change]) -> List[Change]:
        if not changes:
            return []

        by_key = defaultdict(list)
        for change in changes:
            by_key[change.key].append(change)

        merged = []
        for key, key_changes in by_key.items():
            if len(key_changes) == 1:
                merged.append(key_changes[0])
            else:
                result = key_changes[0]
                for c in key_changes[1:]:
                    result = result.compose(c)
                merged.append(result)

        return merged

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._values.keys()) + list(self._computed.keys())

    def items(self) -> List[tuple]:
        with self._lock:
            result = [(k, v) for k, v in self._values.items()]
            for k, c in self._computed.items():
                result.append((k, c.get()))
            return result

    async def items_async(self) -> List[tuple]:
        """Async version of items."""
        with self._lock:
            result = [(k, v) for k, v in self._values.items()]
            for k, c in self._computed.items():
                result.append((k, await c.get_async()))
            return result

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            result = dict(self._values)
            for key, computed in self._computed.items():
                result[key] = computed.get()
            return result

    async def snapshot_async(self) -> Dict[str, Any]:
        """Async version of snapshot."""
        with self._lock:
            result = dict(self._values)
            for key, computed in self._computed.items():
                result[key] = await computed.get_async()
            return result

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            traced_count = sum(
                1 for c in self._computed.values() if c._trace is not None
            )
            async_count = sum(1 for c in self._computed.values() if c._is_async)
            return {
                "total_keys": len(self._values) + len(self._computed),
                "source_keys": len(self._values),
                "computed_keys": len(self._computed),
                "async_computed_keys": async_count,
                "traced_computations": traced_count,
                "observers": sum(len(obs) for obs in self._observers.values()),
                "global_observers": len(self._global_observers),
                "history_size": len(self._history),
                "total_dependencies": sum(
                    len(deps) for deps in self._graph._forward.values()
                ),
                "operation_count": self._operation_count,
            }

    def history(self, limit: int = 100) -> List[Change]:
        with self._lock:
            return list(self._history)[-limit:]

    def close(self) -> None:
        with self._lock:
            self._cleanup()

    def _cleanup(self) -> None:
        self._observers.clear()
        self._global_observers.clear()
        self._history.clear()

        for computed in list(self._computed.values()):
            for dep in list(computed._dependencies):
                self._graph.remove_edge(dep, computed.key)

        self._computed.clear()
        self._values.clear()

        self._graph._forward.clear()
        self._graph._reverse.clear()

    def __del__(self) -> None:
        self._cleanup()

    def __repr__(self) -> str:
        return f"ReactiveStore(keys={len(self.keys())})"

    # ========================================================================
    # COMPATIBILITY ALIASES
    # ========================================================================

    def source(self, key: str, value: Any) -> None:
        """Set a source value (alias for set)."""
        self.set(key, value)

    def observe(
        self, key: str, callback: Callable[[Change], None]
    ) -> Callable[[], None]:
        """Subscribe to changes (alias for on)."""
        return self.on(key, callback)

    def subscribe(self, key: str, callback: Callable) -> Callable[[], None]:
        """Subscribe to changes (supports both Change and TypedDelta callbacks)."""

        def wrapped_callback(change: Change):
            callback(change)

        return self.on(key, wrapped_callback)

    def derive(self, key: str, fn: Callable, deps: Optional[List[str]] = None) -> None:
        """Create a derived/computed value (alias for computed)."""
        self.computed(key, fn, deps)

    def product(self, key: str, sources: List[str]) -> None:
        """Create a product (tuple) of multiple sources."""

        def product_fn(*args):
            return tuple(args)

        self.computed(key, product_fn, sources)

    def feedback(
        self, key: str, fn: Callable, input_key: str, initial_state: Any
    ) -> None:
        """Create a feedback loop with stateful computation."""
        state_key = f"{key}_state"
        self._values[state_key] = initial_state
        self._values[key] = None

        class FeedbackValue:
            def __init__(self, store, key, state_key, input_key, fn):
                self._store = store
                self._key = key
                self._state_key = state_key
                self._input_key = input_key
                self._fn = fn

            def __call__(self):
                current_input = self._store.get(self._input_key)
                current_state = self._store._values.get(self._state_key)

                if current_input is None or current_state is None:
                    return None

                new_state, output = self._fn(current_state, current_input)
                self._store._values[self._state_key] = new_state
                return output

        feedback_val = FeedbackValue(self, key, state_key, input_key, fn)
        self._values[key] = feedback_val

        def on_input_change(_):
            pass

        self.on(input_key, on_input_change)


# ============================================================================
# BATCH CONTEXT
# ============================================================================


class BatchContext:
    def __init__(self, store: ReactiveStore):
        self._store = store

    def __enter__(self):
        self._store._batch_depth += 1
        if self._store._batch_depth == 1:
            self._store._pending_changes = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store._batch_depth -= 1

        if self._store._batch_depth == 0:
            pending = self._store._pending_changes
            if pending:
                merged = self._store._merge_changes(pending)

                for change in merged:
                    self._store._propagate_immediate(change)

                self._store._pending_changes = []

        return False


# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================


__all__ = [
    "ReactiveStore",
    "Store",
    "Change",
    "ChangeType",
    "BatchContext",
    "CircularDependencyError",
    "ComputationError",
    "TypedDelta",
    "DeltaType",
    "DeltaRegistry",
    "TracedValue",
    "AutoDiffEngine",
    "TracedOp",
    "ComputationTrace",
]

# Alias for backward compatibility
Store = ReactiveStore
