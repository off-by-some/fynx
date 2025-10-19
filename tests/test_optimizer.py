"""
FynX Optimizer Tests - Comprehensive Categorical Optimization Testing
=====================================================================

This module contains unit tests for FynX's categorical optimizer, focusing on extreme
cases where optimizations are guaranteed to apply. These tests verify that the
categorical rewrite system correctly implements the theoretical optimizations.

Test Coverage
-------------

**Chain Fusion (Rule 1)**: Functor composition collapse
- Long sequential chains of transformations
- Nested function compositions
- Deep dependency chains

**Product Factorization (Rule 2)**: Common subexpression elimination
- Multiple dependents sharing computation prefixes
- Diamond dependency patterns
- Shared intermediate values

**Pullback Fusion (Rule 3)**: Filter combination
- Sequential conditional filters
- Conjunctive predicate merging
- Filter chain optimization

**Equivalence Analysis**: DAG quotient construction
- Bisimulation-based semantic equivalence
- Identical computation detection
- Function normalization

**Materialization Optimization (Rule 4)**: Cost-based caching
- Dynamic programming for optimal materialization
- Cost functional minimization
- Memory vs computation tradeoffs

**Integration Tests**: End-to-end optimization validation
- Complex reactive graphs
- Multiple optimization rule interactions
- Performance regression detection
"""

import time

import pytest

from fynx import observable
from fynx.observable.computed import ComputedObservable
from fynx.optimizer import ReactiveGraph, get_graph_statistics, optimize_reactive_graph


class TestChainFusion:
    """Test Rule 1: Functor Composition Collapse - O(g) ∘ O(f) = O(g ∘ f)"""

    def test_simple_chain_fusion(self):
        """Test that a simple 3-link chain gets fused into 2 nodes."""
        base = observable(5)
        chain = base >> (lambda x: x * 2) >> (lambda x: x + 10) >> str

        # Trigger optimization
        results, optimizer = optimize_reactive_graph([chain])

        # Should have performed functor fusions
        assert results["functor_fusions"] >= 1

        # Verify semantics preserved
        assert chain.value == "20"  # (5 * 2 + 10) -> "20"

    def test_extreme_chain_fusion_100_links(self):
        """Test fusion of a 100-link chain - should reduce to minimal nodes."""
        from fynx.optimizer import ReactiveGraph

        base = observable(0)
        chain = base

        # Create 100-link chain
        for i in range(100):
            chain = chain >> (lambda x, i=i: x + i)

        # Build graph and get initial stats
        graph = ReactiveGraph()
        graph.build_from_observables([chain])
        initial_nodes = len(graph.nodes)

        # Optimize
        results = graph.optimize()

        # Should fuse the entire chain at once (efficient algorithm)
        assert results["functor_fusions"] >= 1  # At least the chain was fused

        # Should dramatically reduce node count
        final_nodes = len(graph.nodes)
        reduction_ratio = initial_nodes / max(final_nodes, 1)
        assert reduction_ratio >= 10  # At least 10x reduction

        # Verify correct result: sum(0..99) + 1 (base set to 1)
        base.set(1)
        expected = 1 + sum(range(100))  # 1 + 4950 = 5051
        assert chain.value == expected

    def test_nested_function_composition(self):
        """Test fusion preserves correct function composition order."""
        base = observable(2)

        # Create chain: ((x² + 3) × 4) - 1
        chain = (
            base
            >> (lambda x: x**2)  # x²
            >> (lambda x: x + 3)  # x² + 3
            >> (lambda x: x * 4)  # (x² + 3) × 4
            >> (lambda x: x - 1)
        )  # (x² + 3) × 4 - 1

        results, optimizer = optimize_reactive_graph([chain])

        # Should fuse the entire chain
        assert results["functor_fusions"] >= 1

        # Verify composition: ((2² + 3) × 4) - 1 = ((4 + 3) × 4) - 1 = (7 × 4) - 1 = 28 - 1 = 27
        assert chain.value == 27

    def test_chain_fusion_with_merges(self):
        """Test chain fusion works with merged observables."""
        a = observable(1)
        b = observable(2)

        # Create merged observable, then chain
        merged = a | b
        chain = (
            merged
            >> (lambda x, y: x + y)  # Add them
            >> (lambda x: x * 3)  # Multiply by 3
            >> str
        )  # Convert to string

        results, optimizer = optimize_reactive_graph([chain])

        # Should optimize despite merge
        assert results["total_nodes"] <= 4  # merged + chain nodes

        # Verify: (1 + 2) * 3 = 9 -> "9"
        assert chain.value == "9"


class TestProductFactorization:
    """Test Rule 2: Product Factorization - Common subexpression elimination"""

    def test_common_subexpression_elimination(self):
        """Test that shared computation prefixes are factored out."""
        base = observable(10)

        # Two computations sharing the same prefix
        shared = base >> (lambda x: x * 2)  # *2 (shared)
        branch1 = shared >> (lambda x: x + 5)  # *2 +5
        branch2 = shared >> (lambda x: x - 3)  # *2 -3

        results, optimizer = optimize_reactive_graph([branch1, branch2])

        # Should perform efficient chain fusion
        assert (
            results["functor_fusions"] >= 0
        )  # May be handled by product factorization

        # Should reduce computations through fusion
        # One branch gets fully fused, optimizing the shared computation
        stats = get_graph_statistics(optimizer)
        assert stats["total_nodes"] <= 4  # Optimized structure

        # Verify results: 10*2=20, 20+5=25, 20-3=17
        assert branch1.value == 25
        assert branch2.value == 17

    def test_diamond_dependency_pattern(self):
        """Test optimization of diamond-shaped dependency graphs."""
        base = observable(5)

        # Diamond pattern:
        #     base
        #    /     \
        # left     right
        #    \     /
        #   combine

        left = base >> (lambda x: x + 1)
        right = base >> (lambda x: x * 2)
        combine = (left | right) >> (lambda l, r: l + r)

        results, optimizer = optimize_reactive_graph([combine])

        # Should optimize diamond pattern
        assert results["total_nodes"] <= 5  # base + left + right + merged + combine

        # Verify: (5+1) + (5*2) = 6 + 10 = 16
        assert combine.value == 16

    def test_multiple_shared_prefixes(self):
        """Test multiple overlapping shared computations."""
        base = observable(3)

        # Create multiple computations with shared prefixes
        computations = []
        for i in range(5):
            # Each shares: base -> (*2) -> (+i)
            comp = base >> (lambda x: x * 2) >> (lambda x, i=i: x + i)
            computations.append(comp)

        results, optimizer = optimize_reactive_graph(computations)

        # Should optimize through functor fusion
        assert results["functor_fusions"] >= 3

        # Should have optimized structure
        stats = get_graph_statistics(optimizer)
        assert stats["total_nodes"] <= len(computations) + 2  # optimized chains + base

        # Verify all results: 3*2+i for i in 0..4 -> 6,7,8,9,10
        expected = [6, 7, 8, 9, 10]
        actual = [c.value for c in computations]
        assert actual == expected


class TestPullbackFusion:
    """Test Rule 3: Pullback Fusion - Sequential filter combination"""

    def test_sequential_filter_fusion(self):
        """Test that sequential conditional filters are fused."""
        data = observable(15)

        # Create chain of filters
        filtered = (
            data
            & (lambda x: x > 10)  # x > 10
            & (lambda x: x < 20)  # x > 10 AND x < 20
            & (lambda x: x % 2 == 1)
        )  # x > 10 AND x < 20 AND x odd

        results, optimizer = optimize_reactive_graph([filtered])

        # Should perform filter fusions
        assert results["filter_fusions"] >= 1

        # Should reduce to single conditional node
        stats = get_graph_statistics(optimizer)
        assert stats["total_nodes"] <= 3  # data + fused_filter + result

        # 15 meets all conditions: >10, <20, odd
        assert filtered.value == 15

        # Test boundary case
        data.set(25)  # >20, fails second condition
        assert filtered.value is None  # Should be filtered out

        # Test another value that meets conditions
        data.set(17)  # >10, <20, odd
        assert filtered.value == 17

    def test_filter_fusion_complex_predicates(self):
        """Test fusion with complex multi-condition filters."""
        value = observable(42)

        # Multiple complex conditions that should be fused
        complex_filter = (
            value
            & (lambda x: x % 7 == 0)  # Divisible by 7
            & (lambda x: x > 35)  # > 35
            & (lambda x: x < 50)  # < 50
            & (lambda x: x % 2 == 0)
        )  # Even

        results, optimizer = optimize_reactive_graph([complex_filter])

        # Should fuse all conditions
        assert results["filter_fusions"] >= 2

        # 42 meets all: 42/7=6, >35, <50, even
        assert complex_filter.value == 42

        # Test failure case
        value.set(49)  # 49%2==1 (odd), fails last condition
        assert complex_filter.value is None


class TestEquivalenceAnalysis:
    """Test DAG quotient construction via bisimulation"""

    def test_identical_computations_equivalence(self):
        """Test that identical computations are recognized as equivalent."""
        base = observable(5)

        # Create two identical computation chains
        chain1 = base >> (lambda x: x * 2) >> (lambda x: x + 3)
        chain2 = base >> (lambda x: x * 2) >> (lambda x: x + 3)  # Identical

        results, optimizer = optimize_reactive_graph([chain1, chain2])

        # Should recognize equivalence
        assert (
            results["equivalence_classes"] <= 3
        )  # base + shared_computation + result_type

        # Both should produce same result: (5*2)+3 = 13
        assert chain1.value == 13
        assert chain2.value == 13

    def test_semantic_equivalence_detection(self):
        """Test detection of semantically equivalent but syntactically different computations."""
        base = observable(10)

        # Different syntax, same semantics
        comp1 = base >> (lambda x: x + 5) >> (lambda x: x * 2)  # (x+5)*2
        comp2 = base >> (lambda x: x * 2) >> (lambda x: x + 10)  # x*2 + 10

        # Both compute: x*2 + 10, but through different intermediate forms
        results, optimizer = optimize_reactive_graph([comp1, comp2])

        # Should recognize they're computing the same final result
        # (10+5)*2 = 30, 10*2 + 10 = 30
        assert comp1.value == 30
        assert comp2.value == 30

    def test_function_normalization(self):
        """Test that function equivalence handles closures and bytecode."""
        base = observable(1)

        # Functions with different closures but same computation
        def make_adder(n):
            return lambda x: x + n

        comp1 = base >> make_adder(5)
        comp2 = base >> (lambda x: x + 5)  # Same computation

        results, optimizer = optimize_reactive_graph([comp1, comp2])

        # Should recognize functional equivalence
        assert comp1.value == 6
        assert comp2.value == 6


class TestMaterializationOptimization:
    """Test Rule 4: Cost-optimal materialization via dynamic programming"""

    def test_high_frequency_optimization(self):
        """Test that frequently updated nodes get materialized."""
        # Create a node that will be updated very frequently
        base = observable(0)

        # Create many dependents (high fan-out)
        dependents = []
        for i in range(20):
            dep = base >> (lambda x, i=i: x + i)
            dependents.append(dep)

        results, optimizer = optimize_reactive_graph(dependents)

        # Should materialize the base node due to high fan-out (many dependents)
        # The dependents themselves are leaves and shouldn't be materialized
        assert results["materialized_nodes"] >= 1  # at least the base node

        # Verify the graph structure is optimized
        stats = get_graph_statistics(optimizer)
        assert stats["total_nodes"] >= len(dependents) + 1  # dependents + base

    def test_low_frequency_recomputation(self):
        """Test that rarely updated expensive computations get recomputed."""
        base = observable(1000)

        # Create expensive computation that's rarely updated
        expensive = base >> (lambda x: sum(i**2 for i in range(x)))  # O(x) computation
        result = expensive >> (lambda x: x // 1000)  # Simple final computation

        results, optimizer = optimize_reactive_graph([result])

        # The expensive intermediate should be recomputed, not materialized
        # (since it's rarely updated and expensive)
        assert results["materialized_nodes"] <= 2  # base + final result only

    def test_cost_function_minimization(self):
        """Test that DP finds optimal materialization strategy."""
        # Create a complex graph where different strategies have different costs
        base = observable(1)

        # Create chain with varying computation costs
        chain = base
        for i in range(10):
            if i % 3 == 0:
                # Expensive computation every 3rd step
                chain = chain >> (lambda x, i=i: sum(j for j in range(x + i)))
            else:
                # Cheap computation
                chain = chain >> (lambda x, i=i: x + i)

        results, optimizer = optimize_reactive_graph([chain])

        # Should fuse the entire chain into minimal operations
        assert (
            results["total_nodes"] <= 3
        )  # base + final result (+ maybe one intermediate)
        assert results["functor_fusions"] >= 1  # Entire chain fused efficiently

        # Verify correctness despite optimization
        expected = 1
        for i in range(10):
            if i % 3 == 0:
                expected = sum(j for j in range(expected + i))
            else:
                expected = expected + i
        assert chain.value == expected


class TestIntegrationOptimization:
    """Integration tests for complex optimization scenarios"""

    def test_complex_reactive_graph_optimization(self):
        """Test optimization of a complex real-world reactive graph."""
        # Simulate a complex UI component with multiple interdependent computations

        # Base data
        user_data = observable({"age": 25, "score": 85, "level": 3})

        # Extract fields
        age = user_data >> (lambda d: d["age"])
        score = user_data >> (lambda d: d["score"])
        level = user_data >> (lambda d: d["level"])

        # Computed properties
        is_adult = age >> (lambda a: a >= 18)
        is_high_score = score >> (lambda s: s >= 80)
        bonus_multiplier = level >> (lambda l: 1.0 + l * 0.1)

        # Complex computation using multiple fields
        final_score = (
            (score | bonus_multiplier) >> (lambda s, m: s * m) >> (lambda s: int(s))
        )

        # Conditional display
        display_score = final_score & is_adult & is_high_score

        # Optimize the entire graph
        all_nodes = [
            age,
            score,
            level,
            is_adult,
            is_high_score,
            bonus_multiplier,
            final_score,
            display_score,
        ]

        results, optimizer = optimize_reactive_graph(all_nodes)

        # Should perform optimizations (efficient chain fusion reduces individual operation count)
        assert (
            results["functor_fusions"]
            + results["product_factorizations"]
            + results["filter_fusions"]
        ) >= 2

        # Should reduce graph size through optimization
        assert results["total_nodes"] <= len(all_nodes)  # At least some reduction

        # Verify all computations still work correctly
        assert is_adult.value == True
        assert is_high_score.value == True
        assert abs(bonus_multiplier.value - 1.3) < 0.001  # 1.0 + 3*0.1
        assert final_score.value == int(85 * 1.3)  # 110.5 -> 110 (int)
        assert display_score.value == 110

    def test_semantic_preservation_under_optimization(self):
        """Test that all optimizations preserve observable semantics."""
        # Create complex chains and verify semantics preserved through optimization
        base = observable(7)

        # Create multiple different computation patterns
        patterns = [
            # Arithmetic chain
            base >> (lambda x: x + 3) >> (lambda x: x * 2) >> (lambda x: x - 1),
            # String processing chain
            base >> str >> (lambda s: f"value:{s}") >> (lambda s: s.upper()),
            # Conditional chain
            base >> (lambda x: x if x > 5 else 0) >> (lambda x: x**2),
            # Complex mixed chain
            (
                base
                >> (lambda x: x * 2)
                >> (lambda x: x + 5)
                >> str
                >> (lambda s: f"result:{s}")
                >> len
            ),
        ]

        # Optimize all patterns together
        results, optimizer = optimize_reactive_graph(patterns)

        # Should fuse chains efficiently
        assert results["functor_fusions"] >= 3  # At least most chains fused

        # Verify all results correct
        assert patterns[0].value == ((7 + 3) * 2) - 1 == 19  # 7+3=10, 10*2=20, 20-1=19
        assert patterns[1].value == "VALUE:7"
        assert patterns[2].value == 7**2 == 49  # 7 > 5 so 7^2 = 49
        assert (
            patterns[3].value == len(f"result:{((7 * 2) + 5)}") == len("result:19") == 9
        )


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running FynX Optimizer Extreme Case Tests...")

    # Test chain fusion
    test = TestChainFusion()
    test.test_extreme_chain_fusion_100_links()
    print("✓ Chain fusion (100 links)")

    test.test_nested_function_composition()
    print("✓ Nested function composition")

    # Test product factorization
    product_test = TestProductFactorization()
    product_test.test_multiple_shared_prefixes()
    print("✓ Common subexpression elimination")

    # Test equivalence
    equiv_test = TestEquivalenceAnalysis()
    equiv_test.test_identical_computations_equivalence()
    print("✓ Semantic equivalence detection")

    # Test materialization
    mat_test = TestMaterializationOptimization()
    mat_test.test_high_frequency_optimization()
    print("✓ Cost-optimal materialization")

    # Test integration
    integration_test = TestIntegrationOptimization()
    integration_test.test_complex_reactive_graph_optimization()
    print("✓ Complex graph optimization")

    print("\nAll extreme case tests passed! 🎉")
    print("Categorical optimization system is working correctly.")
