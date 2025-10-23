"""
FynX Instruction Tape Prototype - Optimized Version
"""

import time
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from numba import njit, prange

# Type variable for generic observables
T = TypeVar("T")

# =============================================================================
# Protocol Integration for Cross-System Compatibility
# =============================================================================


@runtime_checkable
class TapeObservable(Protocol):
    """Protocol for tape-compatible observables."""

    def get_tape_index(self) -> int: ...
    def get_graph(self) -> "OptimizedTapeGraph": ...


class AutoRegisterMeta(type):
    """Metaclass that auto-registers classes implementing protocols."""

    def __new__(cls, name, bases, namespace, **kwargs):
        # Get declared protocols
        protocols = namespace.get("__protocols__", [])

        # Create the class
        new_class = super().__new__(cls, name, bases, namespace, **kwargs)

        # Auto-register with protocols
        for protocol in protocols:
            if hasattr(protocol, "_implementations"):
                if new_class not in protocol._implementations:
                    protocol._implementations.add(new_class)
            else:
                protocol._implementations = {new_class}

        return new_class


# Cross-system compatibility example
class MyObservable:
    """Example third-party observable that implements TapeObservable protocol."""

    def __init__(self, tape_index: int, graph: "OptimizedTapeGraph"):
        self._tape_index = tape_index
        self._graph = graph

    def get_tape_index(self) -> int:
        return self._tape_index

    def get_graph(self) -> "OptimizedTapeGraph":
        return self._graph


def process_observable(obs: TapeObservable) -> float:
    """Process any TapeObservable for cross-system compatibility."""
    graph = obs.get_graph()
    value = graph.get_node_value(obs.get_tape_index())
    return value


class ValueType(Enum):
    """Supported value types in the tape system."""

    FLOAT = 0
    INT = 1
    BOOL = 2
    STRING = 3


# =============================================================================
# Optimized Constants and Types
# =============================================================================


class Opcodes:
    """Operation codes for optimized instruction tape."""

    SET = 0  # Set value (input)
    ADD = 1  # Addition
    MUL = 2  # Multiplication
    CONST = 3  # Constant value
    AND = 4  # Logical AND
    OR = 5  # Logical OR
    NOT = 6  # Logical NOT
    CALL = 7  # Function call (for lambda compilation)
    FMADD = 8  # Fused multiply-add: a * b + c


# Structure of Arrays layout for cache efficiency
INSTRUCTION_DTYPE = np.dtype(
    [
        ("op", np.int8),
        ("a", np.int32),  # Operand A (node index)
        ("b_type", np.int8),  # 0=node, 1=immediate
        ("b_node", np.int32),  # Operand B node index
        ("b_imm", np.float32),  # Operand B immediate value
        ("c_imm", np.float32),  # Operand C immediate (for FMADD)
        ("out", np.int32),  # Output index
    ]
)


# =============================================================================
# Optimized Numba-compiled Execution Engine
# =============================================================================


@njit(boundscheck=False, fastmath=True)
def execute_optimized(
    values: np.ndarray, instructions: np.ndarray, num_instructions: int
) -> None:
    """
    Execute instruction tape operations with immediate operands and fused ops.
    """
    for i in range(num_instructions):
        instr = instructions[i]
        op = instr["op"]
        a_val = values[instr["a"]]
        out_idx = instr["out"]

        if op == 1:  # ADD
            if instr["b_type"] == 1:  # immediate
                values[out_idx] = a_val + instr["b_imm"]
            else:  # node
                values[out_idx] = a_val + values[instr["b_node"]]

        elif op == 2:  # MUL
            if instr["b_type"] == 1:  # immediate
                values[out_idx] = a_val * instr["b_imm"]
            else:  # node
                values[out_idx] = a_val * values[instr["b_node"]]

        elif op == 4:  # AND
            if instr["b_type"] == 1:  # immediate
                b_val = instr["b_imm"]
            else:  # node
                b_val = values[instr["b_node"]]
            values[out_idx] = 1.0 if (a_val != 0.0 and b_val != 0.0) else 0.0

        elif op == 5:  # OR
            if instr["b_type"] == 1:  # immediate
                b_val = instr["b_imm"]
            else:  # node
                b_val = values[instr["b_node"]]
            values[out_idx] = 1.0 if (a_val != 0.0 or b_val != 0.0) else 0.0

        elif op == 6:  # NOT
            values[out_idx] = 1.0 if a_val == 0.0 else 0.0

        elif op == 8:  # FMADD (fused multiply-add)
            if instr["b_type"] == 1:  # immediate
                b_val = instr["b_imm"]
            else:  # node
                b_val = values[instr["b_node"]]
            values[out_idx] = a_val * b_val + instr["c_imm"]

        # SET and CONST operations are handled during input setting


@njit(parallel=True, boundscheck=False, fastmath=True)
def execute_parallel_layer(
    values: np.ndarray, instructions: np.ndarray, start_idx: int, end_idx: int
) -> None:
    """
    Execute a layer of independent operations in parallel.
    """
    for i in prange(start_idx, end_idx):
        instr = instructions[i]
        op = instr["op"]
        a_val = values[instr["a"]]
        out_idx = instr["out"]

        if op == 1:  # ADD
            if instr["b_type"] == 1:  # immediate
                values[out_idx] = a_val + instr["b_imm"]
            else:  # node
                values[out_idx] = a_val + values[instr["b_node"]]

        elif op == 2:  # MUL
            if instr["b_type"] == 1:  # immediate
                values[out_idx] = a_val * instr["b_imm"]
            else:  # node
                values[out_idx] = a_val * values[instr["b_node"]]


# =============================================================================
# Optimized Instruction Tape Graph Engine
# =============================================================================


class OptimizedTapeGraph:
    """
    Instruction tape execution engine with optimizations for performance.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.next_node_index = 0
        self.input_indices: List[int] = []

        # Build phase: Python lists for flexibility
        self.instructions: List[tuple] = []
        self.node_values: List[Any] = []
        self.node_types: List[type] = []

        # Compiled phase: NumPy arrays in SoA layout for performance
        self.instruction_array: Optional[np.ndarray] = None
        self.value_array: Optional[np.ndarray] = None
        self.compiled = False
        self.num_instructions = 0

        # Parallel execution support
        self.execution_layers: Optional[List[tuple]] = None  # (start, end) indices

    def _ensure_capacity(self, additional_nodes: int = 1) -> None:
        """Ensure sufficient capacity for additional nodes."""
        required_capacity = self.next_node_index + additional_nodes
        if required_capacity > self.capacity:
            self._grow_arrays(required_capacity)

    def _grow_arrays(self, min_capacity: int) -> None:
        """Grow internal arrays to accommodate at least min_capacity elements."""
        new_capacity = max(min_capacity, self.capacity * 2)

        if self.compiled:
            new_value_array = np.empty(new_capacity, dtype=np.float32)
            new_value_array[: self.capacity] = self.value_array
            self.value_array = new_value_array

            new_instruction_array = np.empty(new_capacity, dtype=INSTRUCTION_DTYPE)
            new_instruction_array[: self.capacity] = self.instruction_array
            self.instruction_array = new_instruction_array

        self.capacity = new_capacity

    def add_input(self, initial_value: Any = 0.0) -> int:
        """Add an input node to the graph."""
        self._ensure_capacity()
        node_index = self.next_node_index
        self.next_node_index += 1

        float_value = (
            float(initial_value) if isinstance(initial_value, (int, float)) else 0.0
        )

        if self.compiled:
            self.value_array[node_index] = float_value
            instruction = (Opcodes.SET, -1, 1, -1, float_value, 0.0, node_index)
            self.instruction_array[self.num_instructions] = instruction
        else:
            self.node_values.append(float_value)
            self.node_types.append(type(initial_value))
            self.instructions.append(
                (Opcodes.SET, -1, 1, -1, float_value, 0.0, node_index)
            )

        self.input_indices.append(node_index)
        return node_index

    def add_constant(self, value: Any) -> int:
        """Add a constant value node (deprecated - use immediate operands instead)."""
        return self.add_input(value)  # For compatibility

    def add_operation(
        self, opcode: int, operand_a: int, operand_b: Union[int, float]
    ) -> int:
        """Add an operation node with immediate operand support."""
        self._ensure_capacity()
        node_index = self.next_node_index
        self.next_node_index += 1

        # Determine if operand_b is immediate or node reference
        if isinstance(operand_b, (int, float)):
            b_type = 1  # immediate
            b_node = -1
            b_imm = float(operand_b)
            c_imm = 0.0
        else:
            b_type = 0  # node
            b_node = operand_b
            b_imm = 0.0
            c_imm = 0.0

        if self.compiled:
            self.value_array[node_index] = 0.0
            instruction = (opcode, operand_a, b_type, b_node, b_imm, c_imm, node_index)
            self.instruction_array[self.num_instructions] = instruction
            self.num_instructions += 1
        else:
            self.node_values.append(0.0)
            self.node_types.append(float)
            self.instructions.append(
                (opcode, operand_a, b_type, b_node, b_imm, c_imm, node_index)
            )
            self.num_instructions += 1

        return node_index

    def add_fused_operation(
        self,
        opcode: int,
        operand_a: int,
        operand_b: Union[int, float],
        operand_c: float,
    ) -> int:
        """Add a fused operation (like FMADD) with immediate operands."""
        self._ensure_capacity()
        node_index = self.next_node_index
        self.next_node_index += 1

        # operand_b can be node or immediate, operand_c is always immediate
        if isinstance(operand_b, (int, float)):
            b_type = 1  # immediate
            b_node = -1
            b_imm = float(operand_b)
        else:
            b_type = 0  # node
            b_node = operand_b
            b_imm = 0.0

        c_imm = float(operand_c)

        if self.compiled:
            self.value_array[node_index] = 0.0
            instruction = (opcode, operand_a, b_type, b_node, b_imm, c_imm, node_index)
            self.instruction_array[self.num_instructions] = instruction
            self.num_instructions += 1
        else:
            self.node_values.append(0.0)
            self.node_types.append(float)
            self.instructions.append(
                (opcode, operand_a, b_type, b_node, b_imm, c_imm, node_index)
            )
            self.num_instructions += 1

        return node_index

    def set_input_value(self, node_index: int, value: Any) -> None:
        """Set the value of an input node (optimized hot path)."""
        if self.compiled:
            self.value_array[node_index] = float(value)
        else:
            self.node_values[node_index] = float(value)

    def _compute_execution_layers(self) -> List[tuple]:
        """Compute independent execution layers for parallelization using topological sort."""
        if not self.instructions:
            return []

        # Build dependency graph: instruction -> list of instructions that depend on it
        num_instructions = len(self.instructions)
        outgoing = [[] for _ in range(num_instructions)]
        in_degree = [0] * num_instructions

        # Map node indices to the instructions that produce them
        node_to_instruction = {}
        for i, instr in enumerate(self.instructions):
            op, a, b_type, b_node, b_imm, c_imm, out = instr
            node_to_instruction[out] = i

        # Build dependency graph
        for i, instr in enumerate(self.instructions):
            op, a, b_type, b_node, b_imm, c_imm, out = instr
            # Dependencies
            if a >= 0 and a in node_to_instruction:
                producer_instr = node_to_instruction[a]
                outgoing[producer_instr].append(i)
                in_degree[i] += 1

            if (
                b_type == 0 and b_node >= 0 and b_node in node_to_instruction
            ):  # node dependency
                producer_instr = node_to_instruction[b_node]
                outgoing[producer_instr].append(i)
                in_degree[i] += 1

        # Kahn's algorithm for topological layers
        layers = []
        queue = [i for i in range(num_instructions) if in_degree[i] == 0]

        while queue:
            current_layer = queue[:]
            layers.append((min(current_layer), max(current_layer) + 1))

            next_queue = []
            for instr_idx in current_layer:
                for dependent in outgoing[instr_idx]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        # Fallback to single layer if topological sort failed
        if not layers:
            layers = [(0, num_instructions)]

        return layers

    def compile(self, enable_parallel: bool = False) -> None:
        """Compile the graph for optimized execution."""
        if self.compiled:
            return

        self._ensure_capacity(max(self.next_node_index, len(self.node_values)))

        self.instruction_array = np.empty(self.capacity, dtype=INSTRUCTION_DTYPE)
        self.instruction_array[: len(self.instructions)] = self.instructions

        self.value_array = np.empty(self.capacity, dtype=np.float32)
        self.value_array[: len(self.node_values)] = self.node_values

        self.num_instructions = len(self.instructions)
        self.compiled = True

        if enable_parallel:
            self.execution_layers = self._compute_execution_layers()

    def execute(self) -> None:
        """Execute the compiled instruction tape with optional parallelization."""
        if not self.compiled:
            self.compile()

        if self.execution_layers and len(self.execution_layers) > 1:
            # Execute layer by layer, with parallelization within layers
            for start_idx, end_idx in self.execution_layers:
                if end_idx - start_idx > 10:  # Only parallelize substantial layers
                    execute_parallel_layer(
                        self.value_array, self.instruction_array, start_idx, end_idx
                    )
                else:
                    # Sequential execution for this layer
                    for i in range(start_idx, end_idx):
                        instr = self.instruction_array[i]
                        op = instr["op"]
                        a_val = self.value_array[instr["a"]]
                        out_idx = instr["out"]

                        if op == 1:  # ADD
                            if instr["b_type"] == 1:  # immediate
                                self.value_array[out_idx] = a_val + instr["b_imm"]
                            else:  # node
                                self.value_array[out_idx] = (
                                    a_val + self.value_array[instr["b_node"]]
                                )

                        elif op == 2:  # MUL
                            if instr["b_type"] == 1:  # immediate
                                self.value_array[out_idx] = a_val * instr["b_imm"]
                            else:  # node
                                self.value_array[out_idx] = (
                                    a_val * self.value_array[instr["b_node"]]
                                )

                        elif op == 8:  # FMADD
                            if instr["b_type"] == 1:  # immediate
                                b_val = instr["b_imm"]
                            else:  # node
                                b_val = self.value_array[instr["b_node"]]
                            self.value_array[out_idx] = a_val * b_val + instr["c_imm"]
        else:
            # Sequential execution for all instructions
            execute_optimized(
                self.value_array, self.instruction_array, self.num_instructions
            )

    def get_node_value(self, node_index: int) -> float:
        """Get the value of a node (optimized)."""
        if self.compiled:
            return float(self.value_array[node_index])
        else:
            return float(self.node_values[node_index])


class Obs(Generic[T], metaclass=AutoRegisterMeta):
    """Reactive observable with tape-based execution."""

    __slots__ = ("i", "graph", "_subscribers", "_cached_value")
    __protocols__ = [TapeObservable]

    def __init__(self, i: int, graph: OptimizedTapeGraph) -> None:
        self.i = i
        self.graph = graph
        self._subscribers: List[Callable[[T], None]] = []
        self._cached_value: Optional[T] = None

    def get_tape_index(self) -> int:
        """Protocol method: Get the tape index of this observable."""
        return self.i

    def get_graph(self) -> "OptimizedTapeGraph":
        """Protocol method: Get the graph this observable belongs to."""
        return self.graph

    @property
    def value(self) -> T:
        """Get the current value with reactive notifications."""
        current_val = self.graph.get_node_value(self.i)
        if current_val != self._cached_value:
            self._cached_value = current_val
            # Notify subscribers of value change
            for subscriber in self._subscribers:
                subscriber(current_val)
        return current_val

    def set(self, value: T) -> None:
        """Set the value of this observable (for input nodes) and trigger reactivity."""
        self.graph.set_input_value(self.i, value)
        # Force re-evaluation and notifications
        _ = self.value

    def subscribe(self, callback: Callable[[T], None]) -> "Obs":
        """Subscribe to value changes."""
        self._subscribers.append(callback)
        # Call immediately with current value
        callback(self.value)
        return self  # For chaining

    # FynX DSL Operators
    def __rshift__(self, f: Callable) -> "Obs":
        """Transform operator (>>): apply function to observable value."""
        result_obs = Obs(self.graph.add_input(0.0), self.graph)

        def update_transform(_value=None):
            result_obs.set(f(self.value))

        self.subscribe(update_transform)
        # Initialize with current value
        update_transform()
        return result_obs

    def then(self, f: Callable) -> "Obs":
        """Alias for >> operator."""
        return self.__rshift__(f)

    def __add__(self, other) -> "Obs":
        """Addition operator with immediate operand support."""
        if isinstance(other, Obs):
            return Obs(
                self.graph.add_operation(Opcodes.ADD, self.i, other.i), self.graph
            )
        else:
            return Obs(
                self.graph.add_operation(Opcodes.ADD, self.i, float(other)), self.graph
            )

    def __and__(self, other) -> "Obs":
        """Filter operator (&): require both conditions."""
        if isinstance(other, Obs):
            result_obs = Obs(self.graph.add_input(False), self.graph)

            def update_filter(_val=None):
                result_obs.set(bool(self.value) and bool(other.value))

            self.subscribe(update_filter)
            other.subscribe(update_filter)
            update_filter()  # Initialize
            return result_obs
        return self

    def requiring(self, other) -> "Obs":
        """Alias for & operator."""
        return self.__and__(other)

    def __or__(self, other) -> "Obs":
        """Logical OR operator (|)."""
        if isinstance(other, Obs):
            result_obs = Obs(self.graph.add_input(False), self.graph)

            def update_or(_val=None):
                result_obs.set(bool(self.value) or bool(other.value))

            self.subscribe(update_or)
            other.subscribe(update_or)
            update_or()  # Initialize
            return result_obs
        return self

    def either(self, other) -> "Obs":
        """Alias for | operator."""
        return self.__or__(other)

    def __invert__(self) -> "Obs":
        """Negate operator (~): invert boolean value."""
        result_obs = Obs(self.graph.add_input(False), self.graph)

        def update_negate(_val=None):
            result_obs.set(not bool(self.value))

        self.subscribe(update_negate)
        update_negate()  # Initialize
        return result_obs

    def negate(self) -> "Obs":
        """Alias for ~ operator."""
        return self.__invert__()

    def __mul__(self, other) -> "Obs":
        """Multiplication operator with immediate operand support."""
        if isinstance(other, Obs):
            return Obs(
                self.graph.add_operation(Opcodes.MUL, self.i, other.i), self.graph
            )
        else:
            return Obs(
                self.graph.add_operation(Opcodes.MUL, self.i, float(other)), self.graph
            )

    def alongside(self, other) -> "Obs":
        """Alias for + operator."""
        return self.__add__(other)

    def fmadd(self, multiplier: Union["Obs", float], addend: float) -> "Obs":
        """Fused multiply-add operation: self * multiplier + addend."""
        if isinstance(multiplier, Obs):
            return Obs(
                self.graph.add_fused_operation(
                    Opcodes.FMADD, self.i, multiplier.i, addend
                ),
                self.graph,
            )
        else:
            return Obs(
                self.graph.add_fused_operation(
                    Opcodes.FMADD, self.i, multiplier, addend
                ),
                self.graph,
            )


# Reactive decorator for side effects
def reactive(*observables):
    """Decorator to create reactive functions that respond to observable changes."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Subscribe to all observables
            for obs in observables:
                obs.subscribe(lambda val: func(val, *args, **kwargs))
            return func

        return wrapper

    return decorator


class Store:
    """Base class for organizing related observables."""

    pass


def observable(
    initial_value: T = 0.0, graph: Optional[OptimizedTapeGraph] = None
) -> "Obs[T]":
    """Create a new observable value in the given graph."""
    if graph is None:
        graph = OptimizedTapeGraph()
    return Obs(graph.add_input(initial_value), graph)


def computed(func: Callable, *dependencies: Obs) -> Obs:
    """
    Create a computed observable from a function and its dependencies.

    Note: This is a simplified implementation. Full computed observables
    would require more sophisticated compilation and dependency tracking.
    """
    raise NotImplementedError(
        "computed() not supported in optimized tape mode - use operator overloading"
    )
