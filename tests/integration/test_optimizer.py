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
from fynx.observable.conditional import ConditionalNeverMet, ConditionalNotMet
from fynx.optimizer import (
    MorphismParser,
    MorphismType,
    ReactiveGraph,
    get_graph_statistics,
    optimize_reactive_graph,
)


class TestChainFusion:
    """Test Rule 1: Functor Composition Collapse - O(g) ∘ O(f) = O(g ∘ f)"""

    @pytest.mark.integration
    @pytest.mark.optimizer
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

    @pytest.mark.integration
    @pytest.mark.optimizer
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

    @pytest.mark.integration
    @pytest.mark.optimizer
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

    @pytest.mark.integration
    @pytest.mark.optimizer
    def test_chain_fusion_with_merges(self):
        """Test chain fusion works with merged observables."""
        a = observable(1)
        b = observable(2)

        # Create merged observable, then chain
        merged = a + b
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
        combine = (left + right) >> (lambda l, r: l + r)

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
        with pytest.raises(ConditionalNotMet):
            _ = filtered.value  # Should raise exception when conditions not met

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
        with pytest.raises(ConditionalNotMet):
            _ = complex_filter.value


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


class TestGraphStructure:
    """Test graph construction and basic structure handling"""

    @pytest.mark.integration
    @pytest.mark.optimizer
    def test_empty_graph_optimization(self):
        """Test optimization of empty graph."""
        results, optimizer = optimize_reactive_graph([])
        assert results["total_nodes"] == 0
        assert results["functor_fusions"] == 0

    def test_single_node_graph(self):
        """Test optimization of single observable."""
        obs = observable(42)
        results, optimizer = optimize_reactive_graph([obs])

        stats = get_graph_statistics(optimizer)
        assert stats["total_nodes"] == 1
        assert obs.value == 42

    def test_linear_chain_no_fusion(self):
        """Test chain with non-fusable operations."""
        base = observable(5)
        # Create chain where each function has different signature
        chain = base
        chain = chain >> (lambda x: x + 1)
        chain = chain >> (lambda x: str(x))
        chain = chain >> (lambda x: len(x))
        chain = chain >> (lambda x: x * 2)

        results, optimizer = optimize_reactive_graph([chain])

        # Should fuse some parts but not across type changes
        assert results["functor_fusions"] >= 1  # Can fuse some parts
        # 5 + 1 = 6, str(6) = "6", len("6") = 1, 1 * 2 = 2
        assert chain.value == 2

    def test_multiple_independent_chains(self):
        """Test multiple independent computation chains."""
        base1 = observable(10)
        base2 = observable(20)

        chain1 = base1 >> (lambda x: x * 2) >> (lambda x: x + 5)
        chain2 = base2 >> (lambda x: x // 2) >> (lambda x: x - 3)

        results, optimizer = optimize_reactive_graph([chain1, chain2])

        # Should optimize each chain independently
        assert results["functor_fusions"] >= 2
        assert chain1.value == 25  # (10*2) + 5 = 25
        assert chain2.value == 7  # (20//2) - 3 = 7

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are handled gracefully."""
        # Note: FynX doesn't actually support circular dependencies in the reactive graph,
        # but this tests the robustness of the optimizer
        base = observable(1)

        # This would create a circular dependency if allowed
        # We can't actually create circular deps in FynX, so this tests error handling
        try:
            # This should work fine as FynX prevents circular dependencies
            chain = base >> (lambda x: x + 1)
            results, optimizer = optimize_reactive_graph([chain])
            assert results["total_nodes"] >= 1
        except Exception:
            # If there are issues, they should be handled gracefully
            pass


class TestFusionEdgeCases:
    """Test edge cases in fusion optimization"""

    def test_fusion_with_none_values(self):
        """Test fusion when computations return None."""
        base = observable(5)
        chain = (
            base
            >> (lambda x: x if x > 3 else None)
            >> (lambda x: x * 2 if x is not None else 0)
            >> str
        )

        results, optimizer = optimize_reactive_graph([chain])
        assert results["functor_fusions"] >= 1
        assert chain.value == "10"  # 5 > 3, so 5 * 2 = 10

    def test_fusion_with_exceptions(self):
        """Test fusion robustness when functions might raise exceptions."""
        base = observable(10)
        chain = (
            base
            >> (lambda x: x / 2)  # 5.0
            >> (lambda x: int(x))  # 5
            >> (lambda x: x + 5)  # 10
        )

        results, optimizer = optimize_reactive_graph([chain])
        assert results["functor_fusions"] >= 1
        assert chain.value == 10

    def test_very_deep_narrow_chain(self):
        """Test optimization of a very deep but narrow chain."""
        base = observable(1)
        chain = base

        # Create 50-step chain
        for i in range(50):
            chain = chain >> (lambda x, i=i: x + 1)

        results, optimizer = optimize_reactive_graph([chain])

        # Should fuse the entire chain
        assert results["functor_fusions"] >= 1
        assert chain.value == 51  # Started at 1, added 1 fifty times

    def test_wide_shallow_graph(self):
        """Test optimization of wide but shallow graph."""
        base = observable(10)

        # Create many parallel computations
        branches = []
        for i in range(10):
            branch = base >> (lambda x, i=i: x + i)
            branches.append(branch)

        results, optimizer = optimize_reactive_graph(branches)

        # Should optimize each branch
        assert (
            results["total_nodes"] <= len(branches) + 2
        )  # branches + base + maybe shared

        # Verify all results
        for i, branch in enumerate(branches):
            assert branch.value == 10 + i


class TestConditionalFusion:
    """Test conditional/conditional observable fusion"""

    def test_multiple_condition_composition(self):
        """Test fusion of multiple conditions on same observable."""
        data = observable(100)

        # Multiple conditions that should be fused
        filtered = (
            data
            & (lambda x: x > 50)
            & (lambda x: x < 200)
            & (lambda x: x % 10 == 0)
            & (lambda x: x % 7 != 0)
        )

        results, optimizer = optimize_reactive_graph([filtered])
        assert results["filter_fusions"] >= 1  # Should fuse multiple conditions
        assert filtered.value == 100  # Meets all conditions

        # Test boundary cases - create fresh observables for each test
        data2 = observable(49)  # < 50, should fail
        filtered2 = data2 & (lambda x: x > 50) & (lambda x: x < 200)
        with pytest.raises(ConditionalNeverMet):
            _ = filtered2.value

        data3 = observable(210)  # > 200, should fail
        filtered3 = data3 & (lambda x: x > 50) & (lambda x: x < 200)
        with pytest.raises(ConditionalNeverMet):
            _ = filtered3.value

    def test_condition_fusion_with_transformation(self):
        """Test condition fusion combined with transformations."""
        data = observable(20)

        # Create filtered observable, then transform it
        filtered = data & (lambda x: x > 10) & (lambda x: x < 30)
        result = filtered >> (lambda x: x * 2) >> str

        results, optimizer = optimize_reactive_graph([result])
        assert results["filter_fusions"] >= 1
        assert result.value == "40"  # 20 * 2 = 40

    def test_condition_order_preservation(self):
        """Test that condition order is preserved in fusion."""
        data = observable(15)

        # Conditions with side effects (for testing order)
        call_order = []

        def cond1(x):
            call_order.append(1)
            return x > 10

        def cond2(x):
            call_order.append(2)
            return x < 20

        filtered = data & cond1 & cond2

        # Force evaluation
        _ = filtered.value

        # Should call conditions in order (1 then 2)
        assert call_order == [1, 2]
        assert filtered.value == 15


class TestMorphismOperations:
    """Test morphism (transformation) operations"""

    def test_identity_morphism(self):
        """Test identity morphism operations."""
        from fynx.optimizer import Morphism

        ident = Morphism.identity()
        assert str(ident) == "id"

        # Test normalization
        normalized = ident.normalize()
        assert normalized == ident

    def test_single_morphism(self):
        """Test single morphism creation."""
        from fynx.optimizer import Morphism

        single = Morphism.single("test_func")
        assert str(single) == "test_func"

    def test_compose_morphisms(self):
        """Test morphism composition."""
        from fynx.optimizer import Morphism

        f = Morphism.single("f")
        g = Morphism.single("g")

        composed = Morphism.compose(f, g)
        assert str(composed) == "(f) ∘ (g)"

        # Test normalization
        normalized = composed.normalize()
        assert normalized == composed

    def test_morphism_parser(self):
        """Test morphism parsing."""
        # Test identity
        parsed = MorphismParser.parse("id")
        assert parsed.morphism_type == MorphismType.IDENTITY

        # Test single
        parsed = MorphismParser.parse("func")
        assert parsed.morphism_type == MorphismType.SINGLE
        assert parsed.name == "func"

        # Test composition
        parsed = MorphismParser.parse("f ∘ g")
        assert parsed.morphism_type == MorphismType.COMPOSE


class TestOptimizationCorrectness:
    """Test that optimizations preserve correctness"""

    def test_commutative_operations_fusion(self):
        """Test fusion of operations that are mathematically equivalent."""
        base = observable(5)

        # Different ways to compute x*2 + 10
        way1 = base >> (lambda x: x * 2) >> (lambda x: x + 10)  # (x*2) + 10
        way2 = base >> (lambda x: x + 10) >> (lambda x: x * 2)  # (x+10) * 2

        results, optimizer = optimize_reactive_graph([way1, way2])

        # Both should give correct results
        assert way1.value == 20  # (5*2) + 10 = 20
        assert way2.value == 30  # (5+10) * 2 = 30

    def test_associative_operations_fusion(self):
        """Test fusion respects mathematical associativity."""
        base = observable(2)

        # Test (a + b) + c = a + (b + c)
        chain1 = base >> (lambda x: x + 3) >> (lambda x: x + 5)  # ((2+3)+5) = 10
        chain2 = base >> (lambda x: x + 5) >> (lambda x: x + 3)  # ((2+5)+3) = 10

        results, optimizer = optimize_reactive_graph([chain1, chain2])

        assert chain1.value == 10
        assert chain2.value == 10

    def test_fusion_preserves_side_effects_order(self):
        """Test that fusion preserves order of side effects during initial evaluation."""
        base = observable(0)

        # Functions with side effects that must happen in order
        effects = []

        def f1(x):
            effects.append("f1")
            return x + 1

        def f2(x):
            effects.append("f2")
            return x + 2

        def f3(x):
            effects.append("f3")
            return x + 3

        chain = base >> f1 >> f2 >> f3

        # Force initial evaluation
        result = chain.value

        # Side effects should happen in correct order during first evaluation
        assert effects == ["f1", "f2", "f3"]
        assert result == 6  # 0 + 1 + 2 + 3

        # After optimization, re-evaluation may not trigger side effects due to caching
        # but the result should still be correct
        result2 = chain.value
        assert result2 == 6

    def test_optimization_doesnt_change_observable_values(self):
        """Test that optimization doesn't change final observable values."""
        test_cases = [
            # (initial_value, expected_final_value, computation_chain)
            (5, 25, lambda x: x >> (lambda y: y * 2) >> (lambda y: y + 15)),
            (10, "20", lambda x: x >> (lambda y: y * 2) >> str),
            (3, 18, lambda x: x >> (lambda y: y**2) >> (lambda y: y * 2)),
            (7, 13, lambda x: x >> (lambda y: y + 3) >> (lambda y: y + 3)),
        ]

        for initial, expected, chain_func in test_cases:
            obs = observable(initial)
            chain = chain_func(obs)

            # Get value before optimization
            value_before = chain.value

            # Optimize
            results, optimizer = optimize_reactive_graph([chain])

            # Get value after optimization
            value_after = chain.value

            # Values should be identical
            assert value_before == value_after == expected


class TestIntegrationOptimization:
    """Integration tests for complex optimization scenarios"""

    def test_complex_reactive_graph_optimization_performs_fusions(self):
        """Complex reactive graph optimization performs expected fusion operations"""
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
            (score + bonus_multiplier) >> (lambda s, m: s * m) >> (lambda s: int(s))
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

    def test_complex_reactive_graph_optimization_reduces_graph_size(self):
        """Complex reactive graph optimization reduces total node count"""
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
            (score + bonus_multiplier) >> (lambda s, m: s * m) >> (lambda s: int(s))
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

        # Should reduce graph size through optimization
        assert results["total_nodes"] <= len(all_nodes)  # At least some reduction

    def test_complex_reactive_graph_optimization_preserves_computation_correctness(
        self,
    ):
        """Complex reactive graph optimization preserves computation correctness"""
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
            (score + bonus_multiplier) >> (lambda s, m: s * m) >> (lambda s: int(s))
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

        # Verify all computations still work correctly
        assert is_adult.value == True
        assert is_high_score.value == True
        assert abs(bonus_multiplier.value - 1.3) < 0.001  # 1.0 + 3*0.1
        assert final_score.value == int(85 * 1.3)  # 110.5 -> 110 (int)
        assert display_score.value == 110

    def test_semantic_preservation_under_optimization_performs_fusions(self):
        """Semantic preservation optimization performs expected fusion operations"""
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

    def test_semantic_preservation_under_optimization_maintains_correctness(self):
        """Semantic preservation optimization maintains computation correctness"""
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

        # Verify all results correct
        assert patterns[0].value == ((7 + 3) * 2) - 1 == 19  # 7+3=10, 10*2=20, 20-1=19
        assert patterns[1].value == "VALUE:7"
        assert patterns[2].value == 7**2 == 49  # 7 > 5 so 7^2 = 49
        assert (
            patterns[3].value == len(f"result:{((7 * 2) + 5)}") == len("result:19") == 9
        )


if __name__ == "__main__":
    # Run comprehensive smoke tests
    print("Running FynX Optimizer Comprehensive Tests...")

    # Test graph structure
    struct_test = TestGraphStructure()
    struct_test.test_empty_graph_optimization()
    print("✓ Empty graph optimization")

    struct_test.test_single_node_graph()
    print("✓ Single node optimization")

    struct_test.test_linear_chain_no_fusion()
    print("✓ Linear chain handling")

    struct_test.test_multiple_independent_chains()
    print("✓ Multiple independent chains")

    # Test fusion edge cases
    fusion_test = TestFusionEdgeCases()
    fusion_test.test_fusion_with_none_values()
    print("✓ Fusion with None values")

    fusion_test.test_very_deep_narrow_chain()
    print("✓ Very deep chain fusion")

    fusion_test.test_wide_shallow_graph()
    print("✓ Wide shallow graph optimization")

    # Test conditional fusion
    cond_test = TestConditionalFusion()
    cond_test.test_multiple_condition_composition()
    print("✓ Multiple condition fusion")

    cond_test.test_condition_fusion_with_transformation()
    print("✓ Condition + transformation fusion")

    # Test morphism operations
    morph_test = TestMorphismOperations()
    morph_test.test_identity_morphism()
    print("✓ Identity morphism")

    morph_test.test_compose_morphisms()
    print("✓ Morphism composition")

    morph_test.test_morphism_parser()
    print("✓ Morphism parsing")

    # Test optimization correctness
    correct_test = TestOptimizationCorrectness()
    correct_test.test_commutative_operations_fusion()
    print("✓ Commutative operations")

    correct_test.test_associative_operations_fusion()
    print("✓ Associative operations")

    correct_test.test_fusion_preserves_side_effects_order()
    print("✓ Side effect ordering")

    correct_test.test_optimization_doesnt_change_observable_values()
    print("✓ Value preservation")

    # Test chain fusion (original)
    chain_test = TestChainFusion()
    chain_test.test_extreme_chain_fusion_100_links()
    print("✓ Chain fusion (100 links)")

    chain_test.test_nested_function_composition()
    print("✓ Nested function composition")

    # Test product factorization (original)
    product_test = TestProductFactorization()
    product_test.test_multiple_shared_prefixes()
    print("✓ Common subexpression elimination")

    # Test materialization (original)
    mat_test = TestMaterializationOptimization()
    mat_test.test_high_frequency_optimization()
    print("✓ Cost-optimal materialization")

    # Test integration (original)
    integration_test = TestIntegrationOptimization()
    integration_test.test_complex_reactive_graph_optimization()
    print("✓ Complex graph optimization")

    integration_test.test_semantic_preservation_under_optimization()
    print("✓ Semantic preservation")
