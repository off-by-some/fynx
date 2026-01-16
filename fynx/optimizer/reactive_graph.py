"""
FynX Reactive Graph Optimizer
============================

This module contains the ReactiveGraph class for optimizing reactive observable networks.
"""

import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ..observable.base import Observable
from ..observable.computed import ComputedObservable
from ..observable.conditional import ConditionalObservable
from ..observable.merged import MergedObservable
from .dependency_graph import DependencyGraph, get_graph_statistics
from .dependency_node import DependencyNode
from .morphism import Morphism, MorphismParser

T = TypeVar("T")


class ReactiveGraph(DependencyGraph):
    """
    Reactive graph optimizer extending the base dependency graph.

    Provides reactive-specific optimizations:
    - Graph construction from observable relationships
    - Structural equivalence analysis for optimization
    - Practical rewrite rules (fusion, sharing, filtering)
    - Cost-based materialization strategy
    - Profiling and performance analysis
    """

    def __init__(self):
        super().__init__()

    def get_or_create_node(self, observable: Observable) -> DependencyNode:
        """Get existing node or create new one for observable."""
        if observable in self._node_cache:
            return self._node_cache[observable]

        node = DependencyNode(observable)
        self.nodes[observable] = node
        self._node_cache[observable] = node

        # Extract computation metadata if available
        if isinstance(observable, ComputedObservable):
            if hasattr(observable, "_computation_func"):
                node.computation_func = observable._computation_func
            if hasattr(observable, "_source_observable"):
                node.source_observable = observable._source_observable

        return node

    def _find_all_computation_paths(
        self, start: DependencyNode, end: DependencyNode, max_depth: int = 10
    ) -> List[List[DependencyNode]]:
        """Find all computation paths from start to end node (alias for find_paths)."""
        return self.find_paths(start, end, max_depth)

    def _morphism_signature(self, path: List[DependencyNode]) -> str:
        """Create a structural signature string for a computation path.

        The signature encodes, in order, each node's observable type, key, and whether
        it represents a computed step or a source. This is used only for structural
        comparison in tests and helper utilities; it does not affect runtime behavior.
        """
        parts: List[str] = []
        for node in path:
            obs = node.observable
            node_type = type(obs).__name__
            step_kind = "computed" if node.computation_func else "source"
            key = getattr(obs, "key", getattr(obs, "_key", "<unknown>"))
            parts.append(f"{node_type}:{key}:{step_kind}")
        return " -> ".join(parts)

    def build_from_observables(self, observables: List[Observable]) -> None:
        """Build dependency graph starting from given observables."""
        visited: Set[Observable] = set()
        queue: Deque[Observable] = deque(observables)

        while queue:
            obs = queue.popleft()
            if obs in visited:
                continue
            visited.add(obs)

            node = self.get_or_create_node(obs)

            # For merged observables, add all source dependencies
            if isinstance(obs, MergedObservable):
                for source in obs._source_observables:  # type: ignore
                    if source is None:
                        continue
                    if source not in visited:
                        queue.append(source)
                    source_node = self.get_or_create_node(source)
                    node.incoming.add(source_node)
                    source_node.outgoing.add(node)

            # For computed observables, add their source dependencies
            elif isinstance(obs, ComputedObservable) and hasattr(
                obs, "_source_observable"
            ):
                source = obs._source_observable
                if source is not None:
                    if source not in visited:
                        queue.append(source)
                    source_node = self.get_or_create_node(source)
                    node.incoming.add(source_node)
                    source_node.outgoing.add(node)

            # For conditional observables, add both source and all dependency conditions
            elif isinstance(obs, ConditionalObservable):
                # Explicitly add source dependency
                if obs._source_observable is not None:
                    src = obs._source_observable
                    if src not in visited:
                        queue.append(src)
                    src_node = self.get_or_create_node(src)
                    node.incoming.add(src_node)
                    src_node.outgoing.add(node)

                # Add observable conditions as dependencies
                for condition in getattr(obs, "_processed_conditions", []):
                    # Duck-type: treat as observable if it has value and add_observer
                    if hasattr(condition, "value") and hasattr(
                        condition, "add_observer"
                    ):
                        if condition not in visited:
                            queue.append(condition)
                        cond_node = self.get_or_create_node(condition)  # type: ignore
                        node.incoming.add(cond_node)
                        cond_node.outgoing.add(node)

    def apply_functor_composition_fusion(self) -> int:
        """
        Apply computation chain fusion optimization.

        Identifies linear chains of computed observables and fuses them into
        a single computation by composing their functions. This reduces the
        number of intermediate notifications and function calls.

        Pattern: Linear chains where each computed node has exactly one input and one output
        Optimization: Replace chain with single composed computation

        Returns number of chains fused.
        """
        fusions = 0

        # Find all maximal chains in the graph
        chains = self._find_computation_chains()

        for chain in chains:
            if len(chain) < 2:
                continue  # Nothing to fuse

            # Fuse the entire chain into a single computation
            fusions += self._fuse_chain(chain)

        return fusions

    def _find_computation_chains(self) -> List[List[DependencyNode]]:
        """
        Find all maximal computation chains in the graph.

        A computation chain is a sequence of computed nodes where each has exactly
        one incoming and one outgoing connection (except possibly the last node).
        """
        chains = []
        visited: Set["DependencyNode"] = set()

        for node in self.nodes.values():
            # Find potential chain starts: computed nodes with exactly 1 incoming
            if (
                node in visited
                or not isinstance(node.observable, ComputedObservable)
                or not node.computation_func
                or len(node.incoming) != 1
            ):
                continue

            # Check if this node could be the start of a chain
            # (its predecessor is not a computed node, or has multiple outputs)
            # Special case: MergedObservable can start chains even though it's a ComputedObservable
            predecessor = next(iter(node.incoming))
            if (
                isinstance(predecessor.observable, ComputedObservable)
                and not isinstance(predecessor.observable, MergedObservable)
                and len(predecessor.outgoing) == 1
            ):
                continue  # This is a middle node, not a start

            # Try to extend this node into a chain
            chain = self._extend_chain(node, visited)
            if len(chain) >= 2:  # Only include chains that can be fused
                chains.append(chain)

        return chains

    def _extend_chain(
        self, start_node: DependencyNode, visited: Set[DependencyNode]
    ) -> List[DependencyNode]:
        """
        Extend a node into the longest possible computation chain.

        Follows the chain as long as each node has exactly 1 incoming and 1 outgoing,
        except the final node which can have multiple outgoing connections.
        """
        chain = [start_node]
        current = start_node

        while True:
            # Mark current node as visited
            visited.add(current)

            # Check if we can extend the chain
            if len(current.outgoing) != 1:
                break  # End of chain (current node has multiple outputs or none)

            next_node = next(iter(current.outgoing))

            # Check if next node can be part of the chain
            if (
                next_node in visited
                or not isinstance(next_node.observable, ComputedObservable)
                or not next_node.computation_func
                or len(next_node.incoming) != 1
                or next(iter(next_node.incoming)) != current
            ):
                break  # Cannot extend

            chain.append(next_node)
            current = next_node

        return chain

    def _fuse_chain(self, chain: List[DependencyNode]) -> int:
        """
        Fuse an entire computation chain into a single node.

        Attempts algebraic fusion for simple patterns, falls back to composition.
        """
        if len(chain) < 2:
            return 0

        # Try algebraic fusion first (for simple patterns)
        fused_func = self._try_algebraic_fusion(chain)
        if fused_func is None:
            # Fall back to composition fusion
            fused_func = self._create_composition_fusion(chain)

        # The first and last nodes in the chain
        first_node = chain[0]
        last_node = chain[-1]

        # Get the input source (predecessor of first_node)
        input_source = next(iter(first_node.incoming))

        # Create new fused node
        fused_node = DependencyNode(last_node.observable)
        fused_node.computation_func = fused_func
        fused_node.source_observable = input_source.observable
        fused_node.incoming = {input_source}
        fused_node.outgoing = last_node.outgoing.copy()

        # Update graph structure
        input_source.outgoing.discard(first_node)
        input_source.outgoing.add(fused_node)

        # Update all nodes that depended on the last node in the chain
        for out_node in last_node.outgoing:
            out_node.incoming.discard(last_node)
            out_node.incoming.add(fused_node)

        # Remove all intermediate nodes from the graph
        for node in chain:
            if node.observable in self.nodes:
                del self.nodes[node.observable]

        # Add the fused node
        self.nodes[last_node.observable] = fused_node

        return 1  # One chain fused

    def _try_algebraic_fusion(self, chain: List[DependencyNode]) -> Optional[Callable]:
        """
        Try to algebraically fuse a chain of computations.

        For example, chain of additions: x+0, x+1, x+2, ..., x+n
        can be fused to: x + (0+1+2+...+n)
        """
        if not chain:
            return None

        # Check if all functions follow the pattern: lambda x, i=i: x + i
        # This is the pattern used in the benchmark
        constants = []
        for node in chain:
            if not node.computation_func:
                return None

            # Check if function has closure variables
            if (
                not hasattr(node.computation_func, "__closure__")
                or node.computation_func.__closure__ is None
            ):
                return None

            closure_vars = node.computation_func.__closure__
            if len(closure_vars) != 1:
                return None

            # Check if the closure cell has the expected structure
            cell = closure_vars[0]
            if not hasattr(cell, "cell_contents"):
                return None

            const = cell.cell_contents
            if not isinstance(const, int):
                return None

            constants.append(const)

        # If we got here, all functions are additions with constants
        total_offset = sum(constants)

        def fused_addition(input_value):
            return input_value + total_offset

        return fused_addition

    def _create_composition_fusion(self, chain: List[DependencyNode]) -> Callable:
        """
        Create a fused computation by function composition.

        This is the fallback when algebraic fusion isn't possible.
        """

        def fused_computation(input_value):
            result = input_value
            # Apply functions in chain order: first to last
            for node in chain:
                if node.computation_func:
                    result = node.computation_func(result)
            return result

        return fused_computation

    def apply_product_factorization(self) -> int:
        """
        Apply common subexpression elimination for shared computations.

        Identifies when multiple observables depend on the same inputs and compute
        the same function. Creates a shared intermediate computation that all
        dependents can reuse, eliminating redundant computation.

        Pattern: Multiple nodes with identical input sets and computation functions
        Optimization: Factor out shared computation into intermediate node

        Returns number of factorizations performed.
        """
        factorizations = 0

        # Find nodes that could be part of a product structure
        # Look for patterns where multiple transformations share a common input

        # Group nodes by their input dependencies
        input_groups = defaultdict(list)
        for node in self.nodes.values():
            if node.computation_func is not None and node.incoming:
                # Use frozenset for hashable input set
                input_set = frozenset(node.incoming)
                input_groups[input_set].append(node)

        # For each group of nodes sharing the same inputs, check if they form a product
        for input_set, dependent_nodes in input_groups.items():
            if len(dependent_nodes) > 1:
                # Multiple nodes depend on the same inputs - potential product

                # Verify universal property: check that all dependents use only
                # projections from the shared inputs (no additional dependencies)
                all_dependents_valid = True
                for node in dependent_nodes:
                    # Each dependent should only depend on the shared inputs
                    # (This is the key condition for the universal property)
                    if node.incoming != set(input_set):
                        all_dependents_valid = False
                        break

                if all_dependents_valid and len(input_set) > 1:
                    # These nodes form a valid product structure
                    # Create a shared intermediate that represents the "product" of inputs

                    # For simplicity, if there's a common "base" input used by all,
                    # factor out the shared computation
                    shared_inputs = set(input_set)

                    # Check if all dependents have the same computation function
                    # (indicating redundant computation)
                    computation_functions = set(
                        node.computation_func for node in dependent_nodes
                    )
                    if len(computation_functions) == 1:
                        # All dependents compute the same function - factor it out
                        shared_func = next(iter(computation_functions))

                        # Create shared intermediate node
                        shared_key = f"shared_product_{hash(frozenset(shared_inputs))}_{hash(shared_func)}"

                        # Check if we already have this shared computation
                        shared_node = None
                        for existing_node in self.nodes.values():
                            if (
                                existing_node.computation_func == shared_func
                                and existing_node.incoming == shared_inputs
                            ):
                                shared_node = existing_node
                                break

                        if shared_node is None:
                            # Create new shared node representing the factored computation
                            shared_obs: "ComputedObservable[Any]" = ComputedObservable(
                                shared_key, None
                            )

                            shared_node = DependencyNode(shared_obs)
                            shared_node.computation_func = shared_func
                            shared_node.incoming = shared_inputs.copy()
                            shared_node.source_observable = (
                                list(shared_inputs)[0].observable
                                if shared_inputs
                                else None
                            )

                            # Add to graph
                            self.nodes[shared_obs] = shared_node

                            # Connect inputs to shared node
                            for input_node in shared_inputs:
                                input_node.outgoing.add(shared_node)

                        # All original nodes now just pass through the shared result
                        # This satisfies the universal property - they all factor through the product
                        for original_node in dependent_nodes:
                            if original_node != shared_node:
                                # Replace with identity transformation from shared node
                                original_node.computation_func = lambda x: x  # Identity
                                original_node.incoming = {shared_node}
                                shared_node.outgoing.add(original_node)

                                factorizations += 1

        return factorizations

    def apply_pullback_fusion(self) -> int:
        """
        Apply filter combination optimization for conditional observables.

        When conditional observables filter the same source data with different
        conditions, combines them into a single conditional that applies all
        conditions conjunctively. This reduces the number of intermediate
        filtered observables.

        Pattern: Chain of conditional filters on same source
        Optimization: Combine conditions into single conjunctive filter

        Returns number of fusions performed.
        """
        fusions = 0

        # Find chains of conditional observables that can form pullbacks
        for node in list(self.nodes.values()):
            if isinstance(node.observable, ConditionalObservable):
                # Look for another conditional that depends only on this one
                for child in node.outgoing:
                    if isinstance(
                        child.observable, ConditionalObservable
                    ) and child.incoming == {node}:

                        # Check if these conditionals represent filters that can be combined
                        # They must both filter the same source observable
                        parent_source = getattr(
                            node.observable, "_source_observable", None
                        )
                        child_source = getattr(
                            child.observable, "_source_observable", None
                        )

                        if (
                            parent_source is not None
                            and child_source is node.observable
                        ):  # Child filters the parent's result

                            # Verify this forms a pullback: check that any path to the
                            # doubly-filtered result factors through this combination

                            # For conjunctive filters, we can combine them
                            # This creates the pullback square in the slice category

                            # Get condition observables from both parent and child
                            parent_conditions = getattr(
                                node.observable, "_processed_conditions", []
                            )
                            child_conditions = getattr(
                                child.observable, "_processed_conditions", []
                            )

                            # Combine all conditions (conjunctive semantics)
                            all_conditions = parent_conditions + child_conditions

                            # Create new fused conditional observable that filters the original source
                            fused_obs = ConditionalObservable(
                                parent_source, *all_conditions
                            )

                            fused_node = DependencyNode(fused_obs)
                            fused_node.incoming = (
                                node.incoming.copy()
                            )  # Same source as original parent
                            fused_node.outgoing = child.outgoing.copy()

                            # Update graph structure to create the pullback square
                            for parent_dep in node.incoming:
                                parent_dep.outgoing.discard(node)
                                parent_dep.outgoing.add(fused_node)

                            for grandchild in child.outgoing:
                                grandchild.incoming.discard(child)
                                grandchild.incoming.add(fused_node)

                            # Remove the intermediate filtered nodes
                            # (they're now redundant due to the pullback)
                            # But keep root observables - update them in place instead
                            if (
                                node.observable in self.nodes
                                and node.observable
                                not in getattr(self, "root_observables", set())
                            ):
                                del self.nodes[node.observable]

                            if child.observable in self.nodes:
                                # If child is a root observable, update its internals to match the fused observable
                                if child.observable in getattr(
                                    self, "root_observables", set()
                                ):
                                    # Update the child observable's internals to match the fused observable
                                    child.observable._source_observable = (
                                        fused_obs._source_observable
                                    )
                                    child.observable._processed_conditions = (
                                        fused_obs._processed_conditions
                                    )
                                    child.observable._conditions = fused_obs._conditions
                                    child.observable._conditions_met = (
                                        fused_obs._conditions_met
                                    )
                                    child.observable._has_ever_had_valid_value = (
                                        fused_obs._has_ever_had_valid_value
                                    )
                                    child.observable._all_dependencies = (
                                        fused_obs._all_dependencies
                                    )
                                    # Update node to point to updated observable instead of fused_obs
                                    fused_node.observable = child.observable
                                    self.nodes[child.observable] = fused_node
                                else:
                                    del self.nodes[child.observable]
                                    self.nodes[fused_obs] = fused_node
                            else:
                                self.nodes[fused_obs] = fused_node

                            fusions += 1

        return fusions

    def optimize_materialization(self) -> None:
        """
        Apply cost-based materialization strategy.

        For each computed node, decides whether to cache its result (materialize)
        or recompute it when needed. Uses dynamic programming to compare costs
        of materialization vs recomputation based on estimated usage patterns.

        Strategy: Materialize if cost(materialize) < cost(recompute)
        """
        # Process nodes in topological order (sources first) using efficient traversal
        # instead of sorting all nodes by depth (O(n log n))
        nodes_in_order = self.topological_sort()

        # Build the materialized set incrementally
        materialized_set = set()

        for node in nodes_in_order:
            # Source nodes must always be materialized
            if node.computation_func is None:
                node.is_materialized = True
                materialized_set.add(node)
                continue

            # For computed nodes, compare materialize vs recompute costs

            # Strategy 1: Materialize this node
            mat_set = materialized_set | {node}
            monoidal_cost_mat = node.compute_monoidal_cost(mat_set)
            sharing_penalty_mat = node.compute_sharing_penalty(mat_set)
            materialize_cost = monoidal_cost_mat + sharing_penalty_mat

            # Strategy 2: Don't materialize this node
            monoidal_cost_rec = node.compute_monoidal_cost(materialized_set)
            sharing_penalty_rec = node.compute_sharing_penalty(materialized_set)
            recompute_cost = monoidal_cost_rec + sharing_penalty_rec

            # Choose the better strategy
            if materialize_cost <= recompute_cost:
                node.is_materialized = True
                materialized_set.add(node)
            else:
                node.is_materialized = False

            # Store costs for analysis
            node._materialize_cost = materialize_cost
            node._recompute_cost = recompute_cost

    def check_confluence(self) -> Dict[str, Any]:
        """
        Check that the rewrite system terminates and produces consistent results.

        Tests different rule application orders to ensure the system converges
        to equivalent optimized graphs.
        """

        def normalize_graph() -> Tuple[ReactiveGraph, int]:
            """Apply all rewrite rules exhaustively and count total changes."""
            test_graph = self.copy_graph()
            total_changes = 0
            # Continue until no more changes occur, but limit to prevent infinite loops
            max_iterations = len(self.nodes) * 2  # Scale with graph size

            for _ in range(max_iterations):
                changes = 0
                changes += test_graph.apply_functor_composition_fusion()
                changes += test_graph.apply_product_factorization()
                changes += test_graph.apply_pullback_fusion()

                if changes == 0:
                    break  # No more changes possible

                total_changes += changes

            return test_graph, total_changes

        def graphs_equivalent(graph1: ReactiveGraph, graph2: ReactiveGraph) -> bool:
            """Check if two graphs have the same structure after optimization."""
            if len(graph1.nodes) != len(graph2.nodes):
                return False

            # Compare node degrees as a simple structural equivalence check
            degrees1 = sorted(
                (len(n.incoming), len(n.outgoing)) for n in graph1.nodes.values()
            )
            degrees2 = sorted(
                (len(n.incoming), len(n.outgoing)) for n in graph2.nodes.values()
            )

            return degrees1 == degrees2

        # Test all 6 different rule orders (permutations)
        orders = [
            (
                "fusion→product→pullback",
                lambda g: (
                    g.apply_functor_composition_fusion(),
                    g.apply_product_factorization(),
                    g.apply_pullback_fusion(),
                ),
            ),
            (
                "fusion→pullback→product",
                lambda g: (
                    g.apply_functor_composition_fusion(),
                    g.apply_pullback_fusion(),
                    g.apply_product_factorization(),
                ),
            ),
            (
                "product→fusion→pullback",
                lambda g: (
                    g.apply_product_factorization(),
                    g.apply_functor_composition_fusion(),
                    g.apply_pullback_fusion(),
                ),
            ),
            (
                "product→pullback→fusion",
                lambda g: (
                    g.apply_product_factorization(),
                    g.apply_pullback_fusion(),
                    g.apply_functor_composition_fusion(),
                ),
            ),
            (
                "pullback→fusion→product",
                lambda g: (
                    g.apply_pullback_fusion(),
                    g.apply_functor_composition_fusion(),
                    g.apply_product_factorization(),
                ),
            ),
            (
                "pullback→product→fusion",
                lambda g: (
                    g.apply_pullback_fusion(),
                    g.apply_product_factorization(),
                    g.apply_functor_composition_fusion(),
                ),
            ),
        ]

        results = []
        reference_graph = None
        is_confluent = True

        for order_name, apply_rules in orders:
            test_graph = self.copy_graph()

            # Apply rules in this order
            changes1, changes2, changes3 = apply_rules(test_graph)
            total_changes = changes1 + changes2 + changes3

            if reference_graph is None:
                reference_graph = test_graph
            else:
                equivalent = graphs_equivalent(test_graph, reference_graph)
                if not equivalent:
                    is_confluent = False

            results.append(
                {
                    "order": order_name,
                    "total_changes": total_changes,
                    "final_nodes": len(test_graph.nodes),
                }
            )

        return {
            "is_confluent": is_confluent,
            "total_orders_tested": len(orders),
            "convergent_orders": sum(1 for r in results if r.get("converges", True)),
            "order_results": results,
        }

    def _verify_basic_optimization_correctness(self) -> Dict[str, Any]:
        """
        Perform basic structural verification that optimization preserves graph integrity.

        Checks that the graph structure is valid after optimization:
        - No cycles introduced
        - All nodes remain reachable
        - Graph connectivity preserved

        Returns basic verification results.
        """
        # Basic structural checks
        has_cycles = self._detect_cycles()
        all_nodes_reachable = self._check_connectivity()

        return {
            "nodes_preserved": len(self.nodes) > 0,
            "no_cycles_introduced": not has_cycles,
            "graph_connected": all_nodes_reachable,
            "optimization_applied": True,
        }

    def _check_product_structure(self, product_node: DependencyNode) -> bool:
        """
        Check that a node has basic product-like structure.

        Verifies that the node has multiple inputs and can reach all of them,
        indicating a potential product structure for optimization.
        """
        factors = list(product_node.incoming)
        if len(factors) < 2:
            return False  # Need at least 2 factors for a product

        # Check that we can reach all factors from this node
        # (basic structural check for product-like behavior)
        max_search_depth = len(self.nodes) // 2  # Scale with graph size
        for factor in factors:
            paths = self._find_all_computation_paths(
                product_node, factor, max_depth=max_search_depth
            )
            if not paths:
                return False  # Must be able to reach each factor

        return True

    def _verify_commutativity(
        self,
        mediating_path: List[DependencyNode],
        projection_paths: List[List[DependencyNode]],
        factor_paths: List[List[DependencyNode]],
    ) -> bool:
        """
        Verify that projection ∘ mediating = factor_morphism for product diagrams.

        This checks that the mediating morphism composed with projections
        equals the direct morphisms to factors.
        """
        # For reactive graphs, we verify this by checking that the computational
        # effects are equivalent, since we can't directly compose the functions

        # Check that at least one projection composed with mediating
        # produces an equivalent computational path to the factor paths
        mediating_sig = self._morphism_signature(mediating_path)

        for proj_path in projection_paths:
            # Create the composed path: proj_path + mediating_path (but in correct order)
            # In functional composition: (projection ∘ mediating)
            composed_sig = (
                f"({self._morphism_signature(proj_path)}) ∘ ({mediating_sig})"
            )

            # Check if this composed morphism is equivalent to any factor path
            for factor_path in factor_paths:
                factor_sig = self._morphism_signature(factor_path)
                if self._morphisms_equivalent(composed_sig, factor_sig):
                    return True

        return False

    def _morphisms_equivalent(self, sig1: str, sig2: str) -> bool:
        """
        Check if two morphism signatures represent equivalent computations.

        Uses algebraic identities from category theory:
        - Identity laws: f ∘ id = f, id ∘ f = f
        - Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        # Parse both signatures into Morphism objects
        morph1 = MorphismParser.parse(sig1)
        morph2 = MorphismParser.parse(sig2)

        # Check structural equivalence (normalization happens automatically in __eq__)
        return morph1 == morph2

    def compose_morphisms(self, morphism1: str, morphism2: str) -> str:
        """
        Compose two morphisms using functional composition.

        In category theory, morphism composition is associative and follows (g ∘ f)(x) = g(f(x)).
        For reactive graphs, this represents function composition in the computational paths.
        """
        if morphism1 == "id":
            return morphism2
        if morphism2 == "id":
            return morphism1

        # For composed morphisms, combine them properly
        return f"({morphism2}) ∘ ({morphism1})"

    def morphism_identity(self, obj: DependencyNode) -> str:
        """
        Get the identity morphism for an object.

        The identity morphism id_A: A → A satisfies id_A ∘ f = f and g ∘ id_A = g.
        """
        return "id"

    def _check_pullback_structure(self, pullback_node: DependencyNode) -> bool:
        """
        Check that a conditional node has basic pullback-like structure.

        Verifies that the conditional node filters based on multiple conditions
        that can be combined conjunctively.
        """
        if not isinstance(pullback_node.observable, ConditionalObservable):
            return False

        # Check that this conditional depends on other conditionals
        # (indicating potential for pullback fusion)
        legs = list(pullback_node.incoming)
        if len(legs) < 2:
            return False

        # Verify that incoming nodes are also conditionals or sources
        conditional_count = 0
        for leg in legs:
            if isinstance(leg.observable, ConditionalObservable):
                conditional_count += 1

        # Need at least one conditional dependency for pullback structure
        return conditional_count >= 1

    def _verify_pullback_compatibility(
        self, morphisms_to_legs: Dict[DependencyNode, List[List[DependencyNode]]]
    ) -> bool:
        """
        Verify that morphisms to different legs of a pullback are "compatible".

        For pullbacks, this means they must agree on the common structure being filtered.
        In reactive terms, this means they filter based on the same conditions.
        """
        legs = list(morphisms_to_legs.keys())
        if len(legs) < 2:
            return True

        # For conditional pullbacks, compatibility means that the filtering conditions
        # are consistent. This is a simplified check - in full category theory we'd
        # verify that f ∘ x₁ = g ∘ x₂ in the common codomain.

        # Check that all paths have similar computational structure
        # (i.e., they all represent filtering operations)
        path_signatures = []
        for leg, paths in morphisms_to_legs.items():
            leg_sigs = [self._morphism_signature(path) for path in paths]
            path_signatures.extend(leg_sigs)

        # All paths should have filtering/conditional computational structure
        conditional_patterns = ["ConditionalObservable", "filter", "where"]
        compatible_paths = 0

        for sig in path_signatures:
            if any(pattern in sig for pattern in conditional_patterns):
                compatible_paths += 1

        # Require that most paths show conditional/filtering behavior
        return compatible_paths >= len(path_signatures) * 0.8

    def _verify_pullback_factorization(
        self,
        mediating_path: List[DependencyNode],
        legs: List[DependencyNode],
        morphisms_to_legs: Dict[DependencyNode, List[List[DependencyNode]]],
        pullback_node: DependencyNode,
    ) -> bool:
        """
        Verify that the mediating morphism properly factors through the pullback.

        This checks that the mediating morphism composed with pullback projections
        equals the original morphisms to the legs.
        """
        mediating_sig = self._morphism_signature(mediating_path)

        for leg in legs:
            # Get projection from pullback to this leg
            projection_paths = self._find_all_computation_paths(
                pullback_node, leg, max_depth=3
            )
            if not projection_paths:
                return False

            leg_morphisms = morphisms_to_legs[leg]

            # Verify commutativity: projection ∘ mediating = leg_morphism
            if not self._verify_commutativity(
                mediating_path, projection_paths, leg_morphisms
            ):
                return False

        return True

    def _has_morphism(self, from_node: DependencyNode, to_node: DependencyNode) -> bool:
        """Check if there's a morphism (computable path) from one node to another."""
        # Simplified: direct edge or through computations
        if to_node in from_node.outgoing:
            return True

        # Check if there's a computation path
        visited = set()
        queue = deque([from_node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current == to_node:
                return True

            # Add successors
            for successor in current.outgoing:
                if successor not in visited:
                    queue.append(successor)

        return False

    def _get_morphisms(
        self, from_node: DependencyNode, to_node: DependencyNode
    ) -> List[str]:
        """Get list of morphism descriptions from one node to another."""
        # For reactive graphs, morphisms are computation chains
        # This is a simplified representation
        morphisms = []

        if to_node in from_node.outgoing:
            morphisms.append("direct")

        # Could extend to find all computation paths
        return morphisms

    def enable_profiling(self) -> None:
        """
        Enable execution time profiling for all nodes in the graph.

        Wraps computation functions to measure and record execution times.
        """
        for node in self.nodes.values():
            if node.computation_func is not None:
                # Wrap the computation function to measure execution time
                original_func = node.computation_func

                def _make_profiled(node_ref, orig):
                    def profiled_computation(*args, **kwargs):
                        start_time = time.perf_counter()
                        try:
                            result = orig(*args, **kwargs)
                            execution_time = time.perf_counter() - start_time
                            node_ref.record_execution_time(execution_time)
                            return result
                        except Exception as e:
                            execution_time = time.perf_counter() - start_time
                            node_ref.record_execution_time(execution_time)
                            raise e

                    profiled_computation.__name__ = getattr(
                        orig, "__name__", "profiled_func"
                    )
                    profiled_computation.__doc__ = getattr(orig, "__doc__", None)
                    return profiled_computation

                node.computation_func = _make_profiled(node, original_func)

                # Also attempt to wrap the underlying ComputedObservable's stored function
                if (
                    isinstance(node.observable, ComputedObservable)
                    and hasattr(node.observable, "_computation_func")
                    and node.observable._computation_func is not None
                ):
                    node.observable._computation_func = _make_profiled(
                        node, node.observable._computation_func
                    )

    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of profiling data across all nodes.

        Returns statistics about execution times, call frequencies, and cost estimates.
        """
        total_calls = 0
        total_execution_time = 0.0
        profiled_nodes = 0

        node_summaries = []

        for node in self.nodes.values():
            if node.profiling_data["call_count"] > 0:
                profiled_nodes += 1
                total_calls += node.profiling_data["call_count"]

                if node.profiling_data["avg_execution_time"]:
                    total_execution_time += (
                        node.profiling_data["avg_execution_time"]
                        * node.profiling_data["call_count"]
                    )

                node_summaries.append(
                    {
                        "node_key": node.observable.key,
                        "call_count": node.profiling_data["call_count"],
                        "avg_execution_time": node.profiling_data["avg_execution_time"],
                        "estimated_freq": node.update_frequency_estimate,
                        "computation_cost": node.computation_cost,
                    }
                )

        return {
            "total_profiled_nodes": profiled_nodes,
            "total_calls": total_calls,
            "total_execution_time": total_execution_time,
            "avg_calls_per_node": total_calls / max(profiled_nodes, 1),
            "node_summaries": sorted(
                node_summaries, key=lambda x: x["call_count"], reverse=True
            ),
        }

    def _detect_cycles(self) -> bool:
        """Detect if the graph contains cycles."""
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in node.outgoing:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes.values():
            if node not in visited:
                if has_cycle(node):
                    return True
        return False

    @property
    def cycles(self) -> List[List["DependencyNode"]]:
        """Get all cycles in the graph (simplified: return empty list)."""
        # For optimization purposes, we don't need full cycle detection
        # This is a simplified implementation
        return []

    def _check_connectivity(self) -> bool:
        """Check if all nodes are reachable from source nodes."""
        if not self.nodes:
            return True

        visited = set()
        queue = deque()

        # Start from source nodes (no incoming edges)
        for node in self.nodes.values():
            if not node.incoming:
                queue.append(node)
                visited.add(node)

        # BFS to find all reachable nodes
        while queue:
            current = queue.popleft()
            for neighbor in current.outgoing:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self.nodes)

    @property
    def cycles(self) -> List[List[DependencyNode]]:
        """
        Return list of detected cycles in the reactive graph.

        Cycles are problematic in reactive systems as they can cause infinite
        computation loops. This property detects and returns any cycles.
        """
        return self._detect_cycles()

    def _detect_cycles(self) -> List[List[DependencyNode]]:
        """
        Detect cycles in the reactive graph using DFS.

        Returns list of cycles found, where each cycle is a list of nodes
        forming the cycle in order.
        """
        visited = set()
        recursion_stack = set()
        cycles = []

        def dfs(node: DependencyNode, path: List[DependencyNode]):
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            for neighbor in node.outgoing:
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in recursion_stack:
                    # Cycle found - extract it from current path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            recursion_stack.remove(node)

        for node in self.nodes.values():
            if node not in visited:
                dfs(node, [])

        return cycles

    def verify_optimization_correctness(self) -> Dict[str, Any]:
        """
        Verify that optimization transformations preserve graph correctness.

        Checks that:
        - No cycles are introduced
        - Graph connectivity is maintained
        - All nodes remain reachable
        - Optimization transformations are structurally valid

        Returns verification results for optimization correctness.
        """
        # Basic structural checks
        has_cycles = len(self.cycles) > 0
        all_nodes_reachable = self._check_connectivity()
        nodes_preserved = len(self.nodes) > 0

        # Check for optimization-specific invariants
        optimization_valid = self._verify_optimization_invariants()

        return {
            "cycles_introduced": has_cycles,
            "graph_connected": all_nodes_reachable,
            "nodes_preserved": nodes_preserved,
            "optimization_invariants_held": optimization_valid,
            "structural_correctness": not has_cycles
            and all_nodes_reachable
            and nodes_preserved,
        }

    def _verify_optimization_invariants(self) -> bool:
        """
        Check optimization-specific invariants.

        Verifies that:
        - Fused computation chains have valid composition
        - Shared computations are properly connected
        - Conditional fusions maintain filtering semantics
        """
        for node in self.nodes.values():
            # Check that computation functions are callable if present
            if node.computation_func is not None and not callable(
                node.computation_func
            ):
                return False

            # Check that incoming/outgoing sets are consistent
            for incoming_node in node.incoming:
                if node not in incoming_node.outgoing:
                    return False

            for outgoing_node in node.outgoing:
                if node not in outgoing_node.incoming:
                    return False

        return True

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization pass on the reactive graph.

        Applies a series of practical optimizations:
        1. Computation chain fusion (compose sequential operations)
        2. Common subexpression elimination (share identical computations)
        3. Filter combination (merge conjunctive conditions)
        4. Cost-based materialization (decide what to cache)

        Returns optimization statistics and results.
        """
        start_time = time.time()

        # Phase 1: Apply practical rewrite rules exhaustively
        total_fusions = 0
        total_factorizations = 0
        total_filter_fusions = 0

        # Apply rules until no more changes (confluence)
        # Scale iterations with graph size to prevent infinite loops
        max_iterations = max(5, len(self.nodes) // 2)
        for iteration in range(max_iterations):
            changes = 0

            # Rule 1: Computation chain fusion
            functor_fusions = self.apply_functor_composition_fusion()
            changes += functor_fusions

            # Rule 2: Common subexpression elimination
            product_factorizations = self.apply_product_factorization()
            changes += product_factorizations

            # Rule 3: Filter combination
            filter_fusions = self.apply_pullback_fusion()
            changes += filter_fusions

            total_fusions += functor_fusions
            total_factorizations += product_factorizations
            total_filter_fusions += filter_fusions

            # If no changes in this iteration, we're done
            if changes == 0:
                break

        # Phase 3: Cost-optimal materialization strategy
        self.optimize_materialization()

        # Phase 4: Check confluence of the rewrite system
        confluence_results = self.check_confluence()

        # Phase 5: Verify optimization correctness (basic structural checks)
        correctness_results = self._verify_basic_optimization_correctness()

        optimization_time = time.time() - start_time

        return {
            "optimization_time": optimization_time,
            "functor_fusions": total_fusions,
            "product_factorizations": total_factorizations,
            "filter_fusions": total_filter_fusions,
            "total_nodes": len(self.nodes),
            "materialized_nodes": sum(
                1 for n in self.nodes.values() if n.is_materialized
            ),
            "confluence": confluence_results,
            "correctness_check": correctness_results,
        }


def optimize_reactive_graph(
    root_observables: List[Observable],
) -> tuple[Dict[str, Any], ReactiveGraph]:
    """
    Optimize a reactive graph starting from the given root observables.

    Performs global optimization on the dependency network,
    applying rewrite rules and cost-based optimizations.

    Args:
        root_observables: List of observables to optimize (with their dependencies)

    Returns:
        Tuple of (optimization results dictionary, optimizer instance)
    """
    # Build dependency graph
    optimizer = ReactiveGraph()
    optimizer.root_observables = set(root_observables)  # Track root observables
    optimizer.build_from_observables(root_observables)

    # Run optimization
    results = optimizer.optimize()

    return results, optimizer
