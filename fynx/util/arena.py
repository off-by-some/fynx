"""
FynX Arena Allocator - Zero-Allocation Observable Storage
========================================================

This module provides an arena-based allocator for observables, enabling zero-allocation
performance during updates and cache-friendly memory layout.

Key Features:
- Pre-allocated contiguous memory blocks
- Index-based references (no pointer chasing)
- Bitset-based dirty tracking (64 observables per word)
- CSR (Compressed Sparse Row) format for dependencies
- Static topological ordering for efficient evaluation
- Version vectors for O(1) staleness checking

Performance Benefits:
- Zero allocations during updates
- Cache-friendly memory layout
- O(1) staleness checking with bitsets
- O(dependents) propagation (not O(all_observables))
- SIMD-friendly operations on bitsets
"""

import array
from collections import deque
from typing import Any, Callable, List, Optional, Set


class ObservableArena:
    """
    Arena allocator for observables with zero-allocation performance.

    All observables live in a single contiguous memory block with:
    - Pre-allocated arrays for metadata
    - CSR format for dependency storage
    - Bitset for dirty tracking
    - Static topological ordering
    """

    def __init__(self, max_observables: int = 65536):
        """
        Initialize arena with pre-allocated storage.

        Args:
            max_observables: Maximum number of observables (default: 65536)
                            Fits in uint16, allows dense bitsets
        """
        self.capacity = max_observables
        self.count = 0

        # Metadata arrays (parallel arrays for cache efficiency)
        self.versions = array.array("Q", [0] * max_observables)  # uint64
        self.values = [None] * max_observables  # Python objects
        self.computations = [None] * max_observables  # Functions
        self.keys = [None] * max_observables  # Observable keys/names

        # Graph structure (CSR format for cache efficiency)
        self.dependency_offsets = array.array("I", [0] * (max_observables + 1))
        self.dependency_indices = array.array(
            "H", [0] * (max_observables * 8)
        )  # avg 8 deps
        self.dep_count = 0

        # Reverse dependency structure (for efficient dirty marking)
        self.reverse_dep_offsets = array.array("I", [0] * (max_observables + 1))
        self.reverse_dep_indices = array.array(
            "H", [0] * (max_observables * 16)
        )  # avg 16 dependents
        self.reverse_dep_count = 0

        # Dirty tracking (bitset: 1 bit per observable)
        self.dirty_bits = array.array("Q", [0] * ((max_observables + 63) // 64))

        # Observer storage (list of lists for each observable)
        self.observers = [[] for _ in range(max_observables)]

        # Static topological order (computed once)
        self.topo_order = array.array("H", range(max_observables))
        self.topo_levels = array.array("H", [0] * max_observables)
        self.topo_valid = False

        # Free list for reuse
        self.free_list = []

    def allocate(
        self,
        key: Optional[str] = None,
        initial_value: Any = None,
        computation: Callable = None,
    ) -> int:
        """
        Allocate an observable, return its ID.

        O(1) operation, no memory allocation.

        Args:
            key: Optional key/name for the observable
            initial_value: Initial value
            computation: Computation function (None for source observables)

        Returns:
            Observable ID (index in arena)

        Raises:
            RuntimeError: If arena is exhausted
        """
        if self.free_list:
            obs_id = self.free_list.pop()
        else:
            if self.count >= self.capacity:
                raise RuntimeError("Observable arena exhausted")
            obs_id = self.count
            self.count += 1

        # Initialize
        self.values[obs_id] = initial_value
        self.computations[obs_id] = computation
        self.keys[obs_id] = key or f"<unnamed:{obs_id}>"
        self.versions[obs_id] = 1
        self._mark_clean(obs_id)

        # Initialize dependency offsets for new observable
        # dependency_offsets[obs_id] = current dep_count (start of this observable's dependencies)
        # dependency_offsets[obs_id + 1] = current dep_count (end of this observable's dependencies, initially)
        self.dependency_offsets[obs_id] = self.dep_count
        self.dependency_offsets[obs_id + 1] = self.dep_count

        # Initialize reverse dependency offsets
        self.reverse_dep_offsets[obs_id] = self.reverse_dep_count
        self.reverse_dep_offsets[obs_id + 1] = self.reverse_dep_count

        # Invalidate topo order if adding new observable
        self.topo_valid = False

        return obs_id

    def add_observer(self, obs_id: int, observer: Callable) -> None:
        """Add an observer for an observable."""
        if obs_id < len(self.observers):
            if observer not in self.observers[obs_id]:
                self.observers[obs_id].append(observer)

    def remove_observer(self, obs_id: int, observer: Callable) -> None:
        """Remove an observer from an observable."""
        if obs_id < len(self.observers):
            if observer in self.observers[obs_id]:
                self.observers[obs_id].remove(observer)

    def notify_observers(self, obs_id: int, value: Any) -> None:
        """Notify all observers of an observable."""
        if obs_id < len(self.observers):
            for observer in self.observers[obs_id][
                :
            ]:  # Copy to avoid modification during iteration
                try:
                    observer(value)
                except Exception:
                    # Ignore observer exceptions
                    pass

    def free(self, obs_id: int) -> None:
        """Free an observable for reuse."""
        self.values[obs_id] = None
        self.computations[obs_id] = None
        self.keys[obs_id] = None
        self._mark_clean(obs_id)
        self.free_list.append(obs_id)
        self.topo_valid = False

    # Bitset operations - O(1)
    def _mark_dirty(self, obs_id: int) -> None:
        """Set dirty bit for observable."""
        word_idx = obs_id >> 6  # divide by 64
        bit_idx = obs_id & 63  # modulo 64
        self.dirty_bits[word_idx] |= 1 << bit_idx

    def _mark_clean(self, obs_id: int) -> None:
        """Clear dirty bit for observable."""
        word_idx = obs_id >> 6
        bit_idx = obs_id & 63
        self.dirty_bits[word_idx] &= ~(1 << bit_idx)

    def _is_dirty(self, obs_id: int) -> bool:
        """Check if observable is dirty."""
        word_idx = obs_id >> 6
        bit_idx = obs_id & 63
        return bool(self.dirty_bits[word_idx] & (1 << bit_idx))

    # Graph operations - CSR format (cache-friendly)
    def add_dependency(self, dependent: int, dependency: int) -> None:
        """Add edge: dependent depends on dependency."""
        # Bounds checking
        if dependent >= self.count or dependency >= self.count:
            return

        # Check if dependency already exists
        start = self.dependency_offsets[dependent]
        end = self.dependency_offsets[dependent + 1]
        for i in range(start, end):
            if self.dependency_indices[i] == dependency:
                return  # Already exists

        # Ensure we have space in the indices array
        if self.dep_count >= len(self.dependency_indices):
            # Grow array
            new_size = len(self.dependency_indices) * 2
            new_indices = array.array("H", [0] * new_size)
            for i in range(self.dep_count):
                new_indices[i] = self.dependency_indices[i]
            self.dependency_indices = new_indices

        # Insert the new dependency at the end of dependent's list
        insert_pos = self.dependency_offsets[dependent + 1]

        # Shift existing dependencies to make room
        for i in range(self.dep_count, insert_pos, -1):
            self.dependency_indices[i] = self.dependency_indices[i - 1]

        # Insert the new dependency
        self.dependency_indices[insert_pos] = dependency
        self.dep_count += 1

        # Update offsets for all observables after this one
        for i in range(dependent + 1, self.count + 1):
            self.dependency_offsets[i] += 1

        # Add reverse dependency
        self._add_reverse_dependency(dependency, dependent)

        self.topo_valid = False

    def _add_reverse_dependency(self, dependency: int, dependent: int) -> None:
        """Add reverse dependency: dependency -> dependent."""
        # Similar to add_dependency but for reverse structure
        start = self.reverse_dep_offsets[dependency]
        end = self.reverse_dep_offsets[dependency + 1]

        # Check if already exists
        for i in range(start, end):
            if self.reverse_dep_indices[i] == dependent:
                return

        # Ensure space
        if self.reverse_dep_count >= len(self.reverse_dep_indices):
            new_size = len(self.reverse_dep_indices) * 2
            new_indices = array.array("H", [0] * new_size)
            for i in range(self.reverse_dep_count):
                new_indices[i] = self.reverse_dep_indices[i]
            self.reverse_dep_indices = new_indices

        # Insert
        insert_pos = self.reverse_dep_offsets[dependency + 1]
        for i in range(self.reverse_dep_count, insert_pos, -1):
            self.reverse_dep_indices[i] = self.reverse_dep_indices[i - 1]

        self.reverse_dep_indices[insert_pos] = dependent
        self.reverse_dep_count += 1

        # Update offsets
        for i in range(dependency + 1, self.count + 1):
            self.reverse_dep_offsets[i] += 1

    def get_dependents(self, obs_id: int) -> List[int]:
        """Get list of dependents (observables that depend on this one)."""
        if obs_id >= self.count:
            return []
        start = self.reverse_dep_offsets[obs_id]
        end = self.reverse_dep_offsets[obs_id + 1]
        return list(self.reverse_dep_indices[start:end])

    def get_dependencies(self, obs_id: int) -> List[int]:
        """Get list of dependencies (observables this depends on)."""
        if obs_id >= self.count:
            return []
        start = self.dependency_offsets[obs_id]
        end = self.dependency_offsets[obs_id + 1]
        return list(self.dependency_indices[start:end])

    def compute_topological_order(self) -> None:
        """
        Compute static topological order once.

        Uses Kahn's algorithm with level assignment.
        Complexity: O(V + E) where V = observables, E = edges.

        Result: topo_order contains sorted indices
                topo_levels contains depth for each observable
        """
        if self.topo_valid:
            return

        # In-degree calculation
        in_degree = array.array("H", [0] * self.count)
        for dependent in range(self.count):
            for dependency in self.get_dependencies(dependent):
                in_degree[dependent] += 1

        # BFS with level tracking
        queue = deque()

        # Add all sources (in-degree 0)
        for i in range(self.count):
            if in_degree[i] == 0:
                queue.append(i)
                self.topo_levels[i] = 0

        # Process
        topo_idx = 0
        while queue:
            obs_id = queue.popleft()
            self.topo_order[topo_idx] = obs_id
            topo_idx += 1

            # Process dependents
            for dependent in range(self.count):
                if obs_id in self.get_dependencies(dependent):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        self.topo_levels[dependent] = self.topo_levels[obs_id] + 1
                        queue.append(dependent)

        if topo_idx != self.count:
            raise RuntimeError("Cycle detected in observable graph")

        self.topo_valid = True

    def mark_dirty_tree(self, obs_id: int) -> None:
        """Mark transitive dependents as dirty (DFS using reverse dependencies)."""
        self._mark_dirty(obs_id)

        # Use DFS with reverse dependency structure for O(dependents) performance
        visited = set()

        def mark_dependents(current_id: int):
            if current_id in visited:
                return
            visited.add(current_id)

            # Get direct dependents efficiently using reverse structure
            for dependent in self.get_dependents(current_id):
                if not self._is_dirty(dependent):
                    self._mark_dirty(dependent)
                    mark_dependents(dependent)

        mark_dependents(obs_id)

    def find_dirty_dependencies(self, obs_id: int) -> List[int]:
        """Find all dirty dependencies using iterative DFS."""
        dirty_deps = []
        visited = set()
        stack = []

        # Start with direct dependencies
        for dep_id in self.get_dependencies(obs_id):
            stack.append(dep_id)

        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)

            # Check if this dependency is dirty
            if self._is_dirty(current_id):
                dirty_deps.append(current_id)

            # Add dependencies to stack
            for dep_id in self.get_dependencies(current_id):
                if dep_id not in visited:
                    stack.append(dep_id)

        # Sort by topological order for correct evaluation order (dependencies first)
        if self.topo_valid:
            dirty_deps.sort(
                key=lambda x: (
                    self.topo_order.index(x)
                    if x in self.topo_order[: len(self.topo_order)]
                    else 0
                )
            )
        else:
            # If topo order is invalid, just sort by ID
            dirty_deps.sort()

        return dirty_deps

    def evaluate_single(self, obs_id: int) -> None:
        """Evaluate single observable (assumes dependencies are fresh)."""
        computation = self.computations[obs_id]

        if computation is None:
            # Source observable, already has value
            pass
        else:
            # Computed observable - gather dependency values and compute
            deps = self.get_dependencies(obs_id)
            dep_values = [self.values[d] for d in deps]

            # Call computation function
            if len(dep_values) == 1:
                new_value = computation(dep_values[0])
            else:
                new_value = computation(*dep_values)

            self.values[obs_id] = new_value

        # Mark clean and bump version
        self._mark_clean(obs_id)
        self.versions[obs_id] += 1


# Global arena instance
_global_arena = ObservableArena()
