"""
Tests for delta algebras and automatic differentiation.
"""

import math

import numpy as np
import pytest

from fynx.delta_kv_store import (
    ArrayAlgebra,
    AutoDiffEngine,
    ComputationTrace,
    DeltaAlgebra,
    DeltaRegistry,
    DeltaType,
    DictAlgebra,
    ListAlgebra,
    NumericAlgebra,
    ReactiveStore,
    SetAlgebra,
    StringAlgebra,
    TracedOp,
    TracedValue,
    TypedDelta,
)


class TestDeltaAlgebras:
    """Test all built-in delta algebras comprehensively."""

    def test_numeric_algebra_all_types(self):
        algebra = NumericAlgebra()

        # Test all numeric types
        test_cases = [
            # (old, new, expected_delta)
            (5, 7, 2),
            (5.0, 7.0, 2.0),
            (5.5, 7.5, 2.0),
            (1 + 2j, 3 + 4j, 2 + 2j),
            (0, 0, 0),
            (-5, 5, 10),
            (100, 50, -50),
        ]

        for old, new, expected_delta in test_cases:
            delta = algebra.compute_delta(old, new)
            assert delta.delta_type == DeltaType.NUMERIC
            assert delta.value == expected_delta

            # Apply delta
            result = algebra.apply_delta(old, delta)
            assert result == new

            # Compose deltas
            if expected_delta != 0:
                delta2 = algebra.compute_delta(new, new + expected_delta)
                composed = algebra.compose_deltas(delta, delta2)
                assert composed.value == expected_delta * 2

        # Identity checks
        assert algebra.is_identity(TypedDelta(DeltaType.NUMERIC, 0))
        assert algebra.is_identity(TypedDelta(DeltaType.NUMERIC, 1e-15))
        assert algebra.is_identity(TypedDelta(DeltaType.NUMERIC, -1e-15))
        assert not algebra.is_identity(TypedDelta(DeltaType.NUMERIC, 0.1))

        # Non-numeric types should return None
        assert algebra.compute_delta("hello", "world") is None
        assert algebra.compute_delta([1, 2], [3, 4]) is None

    def test_string_algebra_comprehensive(self):
        algebra = StringAlgebra()

        # Test various string operations
        test_cases = [
            # Basic operations
            ("hello", "hello world", "append"),
            ("hello world", "hello", "remove"),
            ("abc", "axc", "insert"),
            ("axc", "abc", "delete"),
            ("hello", "HELLO", "case change"),
            ("", "hello", "from empty"),
            ("hello", "", "to empty"),
            ("test", "test", "identical"),
            # Complex operations
            ("The quick brown fox", "The slow brown fox", "word replacement"),
            ("abc123def", "abc456def", "number replacement"),
            ("hello\nworld", "hello\nthere", "multiline"),
            ("café", "cafe", "unicode normalization"),
            ("🚀", "🚀🚀", "emoji"),
            ("hello", "world", "complex replacement"),
        ]

        for old, new, description in test_cases:
            delta = algebra.compute_delta(old, new)
            assert delta.delta_type == DeltaType.STRING

            # Apply delta
            result = algebra.apply_delta(old, delta)
            assert result == new, f"Failed for {description}: {old} -> {new}"

        # Identity checks
        assert algebra.is_identity(algebra.compute_delta("test", "test"))
        assert algebra.is_identity(algebra.compute_delta("", ""))

        # Compose deltas
        delta1 = algebra.compute_delta("hello", "world")
        delta2 = algebra.compute_delta("world", "world!")
        composed = algebra.compose_deltas(delta1, delta2)
        intermediate = algebra.apply_delta("hello", delta1)
        final = algebra.apply_delta(intermediate, delta2)
        composed_result = algebra.apply_delta("hello", composed)
        assert composed_result == final

        # Non-string types should return None
        assert algebra.compute_delta(123, 456) is None
        assert algebra.compute_delta([1, 2], [3, 4]) is None

    def test_list_algebra_basic(self):
        algebra = ListAlgebra()

        # Test basic list operations that work well with difflib
        test_cases = [
            ([1, 2, 3], [1, 2, 4, 3], "insert middle"),
            ([1, 2, 3], [1, 2], "remove end"),
            ([1, 2, 3], [4, 2, 3], "replace start"),
            ([1, 2, 3], [1, 2, 3, 4], "append"),
            ([1, 2, 3], [0, 1, 2, 3], "prepend"),
            ([], [1, 2, 3], "from empty"),
            ([1, 2, 3], [], "to empty"),
            ([1, 2, 3], [1, 2, 3], "identical"),
            ([1, 2, 3, 4, 5], [1, 2, 6, 3, 4, 5], "insert middle"),
            ([1, 2, 3, 4, 5], [1, 3, 4, 5], "remove middle"),
        ]

        for old_list, new_list, description in test_cases:
            delta = algebra.compute_delta(old_list, new_list)
            assert delta.delta_type == DeltaType.LIST

            # Apply delta
            result = algebra.apply_delta(old_list, delta)
            assert result == new_list, f"Failed for {description}"

        # Identity checks
        assert algebra.is_identity(algebra.compute_delta([1, 2, 3], [1, 2, 3]))
        assert algebra.is_identity(algebra.compute_delta([], []))

        # Non-list types should return None
        assert algebra.compute_delta("hello", "world") is None
        assert algebra.compute_delta(123, 456) is None

    def test_dict_algebra_comprehensive(self):
        algebra = DictAlgebra()

        # Test various dict operations
        test_cases = [
            # Basic operations
            ({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}, "add and modify"),
            ({"a": 1, "b": 2, "c": 3}, {"a": 1}, "remove keys"),
            ({"a": 1, "b": 2}, {"a": 10, "b": 20}, "modify values"),
            ({}, {"a": 1, "b": 2}, "from empty"),
            ({"a": 1, "b": 2}, {}, "to empty"),
            ({"a": 1}, {"a": 1}, "identical"),
            # Complex operations
            (
                {"x": 1, "y": 2, "z": 3},
                {"x": 1, "y": 20, "z": 3, "w": 4},
                "mixed operations",
            ),
            ({"a": {"nested": 1}}, {"a": {"nested": 2}}, "nested dict"),
            ({"items": [1, 2, 3]}, {"items": [1, 2, 4]}, "list values"),
            # Edge cases
            ({"": "empty"}, {"": "not_empty"}, "empty string key"),
            (
                {1: "one", "1": "str_one"},
                {1: "one", "1": "modified"},
                "int/str key confusion",
            ),
        ]

        for old_dict, new_dict, description in test_cases:
            delta = algebra.compute_delta(old_dict, new_dict)
            assert delta.delta_type == DeltaType.DICT

            # Apply delta
            result = algebra.apply_delta(old_dict, delta)
            assert result == new_dict, f"Failed for {description}"

        # Identity checks
        assert algebra.is_identity(algebra.compute_delta({"x": 1}, {"x": 1}))
        assert algebra.is_identity(algebra.compute_delta({}, {}))

        # Compose deltas
        delta1 = algebra.compute_delta({"a": 1}, {"a": 1, "b": 2})
        delta2 = algebra.compute_delta({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
        composed = algebra.compose_deltas(delta1, delta2)
        intermediate = algebra.apply_delta({"a": 1}, delta1)
        final = algebra.apply_delta(intermediate, delta2)
        composed_result = algebra.apply_delta({"a": 1}, composed)
        assert composed_result == final

        # Non-dict types should return None
        assert algebra.compute_delta([1, 2], [3, 4]) is None
        assert algebra.compute_delta("hello", "world") is None

    def test_set_algebra_comprehensive(self):
        algebra = SetAlgebra()

        # Test various set operations
        test_cases = [
            # Basic operations
            ({1, 2, 3}, {1, 2, 4, 5}, "add and remove"),
            ({1, 2, 3, 4}, {1, 2}, "remove elements"),
            ({1, 2}, {1, 2, 3, 4}, "add elements"),
            ({1, 2, 3}, {4, 5, 6}, "complete replacement"),
            (set(), {1, 2, 3}, "from empty"),
            ({1, 2, 3}, set(), "to empty"),
            ({1, 2, 3}, {1, 2, 3}, "identical"),
            # Complex operations
            ({1, 2, 3, 4, 5}, {1, 2, 6, 7}, "mixed operations"),
            ({"a", "b", "c"}, {"a", "b", "d", "e"}, "string elements"),
            # Mixed types
            ({1, "a", 2.5}, {1, "a", 2.5, True}, "mixed types"),
            ({None, 0, ""}, {None, 0, "", False}, "falsy values"),
            # Large sets
            (set(range(100)), set(range(50, 150)), "large set changes"),
        ]

        for old_set, new_set, description in test_cases:
            delta = algebra.compute_delta(old_set, new_set)
            assert delta.delta_type == DeltaType.SET

            # Apply delta
            result = algebra.apply_delta(old_set, delta)
            assert result == new_set, f"Failed for {description}"

        # Identity checks
        assert algebra.is_identity(algebra.compute_delta({1, 2, 3}, {1, 2, 3}))
        assert algebra.is_identity(algebra.compute_delta(set(), set()))

        # Compose deltas
        delta1 = algebra.compute_delta({1, 2}, {1, 2, 3})
        delta2 = algebra.compute_delta({1, 2, 3}, {1, 2, 3, 4})
        composed = algebra.compose_deltas(delta1, delta2)
        intermediate = algebra.apply_delta({1, 2}, delta1)
        final = algebra.apply_delta(intermediate, delta2)
        composed_result = algebra.apply_delta({1, 2}, composed)
        assert composed_result == final

        # Check delta structure
        delta = algebra.compute_delta({1, 2, 3}, {1, 2, 4, 5})
        added, removed = delta.value
        assert added == {4, 5}
        assert removed == {3}

        # Non-set types should return None
        assert algebra.compute_delta([1, 2], [3, 4]) is None
        assert algebra.compute_delta("hello", "world") is None

    def test_array_algebra_basic(self):
        algebra = ArrayAlgebra()

        # Test basic array operations (same shape only)
        test_cases = [
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]), "float64 modify"),
            (np.array([1, 2, 3]), np.array([1, 2, 4]), "int32 modify"),
            (np.array([[1, 2], [3, 4]]), np.array([[1, 5], [3, 4]]), "2d modify"),
            (np.array([1.0]), np.array([1.0]), "single element"),
            (np.array([0.0, 0.0]), np.array([0.0, 0.0]), "zeros"),
        ]

        for old_array, new_array, description in test_cases:
            delta = algebra.compute_delta(old_array, new_array)
            if old_array.shape == new_array.shape:  # Only test same shape
                assert delta.delta_type == DeltaType.ARRAY

                # Apply delta
                result = algebra.apply_delta(old_array, delta)
                np.testing.assert_array_equal(
                    result, new_array, f"Failed for {description}"
                )

        # Identity checks
        same_array = np.array([1.0, 2.0])
        assert algebra.is_identity(algebra.compute_delta(same_array, same_array))

        # Different shapes should return None
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert algebra.compute_delta(arr1, arr2) is None

        # Non-array types should return None
        assert algebra.compute_delta([1, 2], [3, 4]) is None
        assert algebra.compute_delta("hello", "world") is None


class TestDeltaRegistry:
    """Test the delta registry and type detection."""

    def test_type_detection_comprehensive(self):
        registry = DeltaRegistry()

        # Test all built-in types
        test_cases = [
            # (value, expected_type)
            (42, DeltaType.NUMERIC),
            (42.0, DeltaType.NUMERIC),
            (42 + 0j, DeltaType.NUMERIC),
            ("hello", DeltaType.STRING),
            ("", DeltaType.STRING),
            ([1, 2, 3], DeltaType.LIST),
            ([], DeltaType.LIST),
            ({1, 2, 3}, DeltaType.SET),
            (set(), DeltaType.SET),
            ({"a": 1}, DeltaType.DICT),
            ({}, DeltaType.DICT),
            (np.array([1, 2]), DeltaType.ARRAY),
            (np.array([]), DeltaType.ARRAY),
            # Opaque types (custom objects)
            (object(), DeltaType.OPAQUE),
            (lambda x: x, DeltaType.OPAQUE),
            (Exception("test"), DeltaType.OPAQUE),
        ]

        for value, expected_type in test_cases:
            detected_type = registry.detect_type(value)
            assert (
                detected_type == expected_type
            ), f"Failed for {value!r}: expected {expected_type}, got {detected_type}"

    def test_delta_computation_for_all_types(self):
        registry = DeltaRegistry()

        # Test that deltas work for all supported types
        test_cases = [
            (5, 10, DeltaType.NUMERIC),
            ("hello", "world", DeltaType.STRING),
            ([1, 2], [1, 2, 3], DeltaType.LIST),
            ({1, 2}, {1, 2, 3}, DeltaType.SET),
            ({"a": 1}, {"a": 1, "b": 2}, DeltaType.DICT),
            (np.array([1.0, 2.0]), np.array([1.0, 3.0]), DeltaType.ARRAY),
        ]

        for old_val, new_val, expected_type in test_cases:
            delta = registry.compute_delta(old_val, new_val)
            assert delta is not None
            assert delta.delta_type == expected_type

            # Apply delta
            result = registry.apply_delta(old_val, delta)
            if expected_type == DeltaType.ARRAY:
                np.testing.assert_array_equal(result, new_val)
            else:
                assert result == new_val

    def test_opaque_types_no_delta(self):
        registry = DeltaRegistry()

        # Opaque types should not have deltas computed
        obj1 = object()
        obj2 = object()

        delta = registry.compute_delta(obj1, obj2)
        assert delta is None

        # Applying None delta should return original value
        result = registry.apply_delta(obj1, None)
        assert result is obj1

    def test_composite_types_custom_algebras(self):
        registry = DeltaRegistry()

        # Test custom composite algebra
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __eq__(self, other):
                return (
                    isinstance(other, Point) and self.x == other.x and self.y == other.y
                )

            def __repr__(self):
                return f"Point({self.x}, {self.y})"

        class CompositeAlgebra(DeltaAlgebra):
            def compute_delta(self, old, new):
                if isinstance(old, Point) and isinstance(new, Point):
                    dx = new.x - old.x
                    dy = new.y - old.y
                    return TypedDelta(
                        DeltaType.COMPOSITE, {"type": "point", "dx": dx, "dy": dy}
                    )
                return None

            def apply_delta(self, value, delta):
                if isinstance(value, Point) and delta.value.get("type") == "point":
                    return Point(
                        value.x + delta.value["dx"], value.y + delta.value["dy"]
                    )
                return value

            def compose_deltas(self, delta1, delta2):
                if (
                    delta1.value.get("type") == "point"
                    and delta2.value.get("type") == "point"
                ):
                    return TypedDelta(
                        DeltaType.COMPOSITE,
                        {
                            "type": "point",
                            "dx": delta1.value["dx"] + delta2.value["dx"],
                            "dy": delta1.value["dy"] + delta2.value["dy"],
                        },
                    )
                return delta2  # fallback

            def is_identity(self, delta):
                if delta.value.get("type") == "point":
                    return delta.value["dx"] == 0 and delta.value["dy"] == 0
                return True

        # Register custom algebra
        registry.register_custom_algebra(
            DeltaType.COMPOSITE, CompositeAlgebra(), lambda x: isinstance(x, Point)
        )

        # Test Point type detection and delta computation
        point = Point(1, 2)
        assert registry.detect_type(point) == DeltaType.COMPOSITE

        p1 = Point(0, 0)
        p2 = Point(3, 4)
        delta = registry.compute_delta(p1, p2)
        assert delta is not None
        assert delta.delta_type == DeltaType.COMPOSITE
        assert delta.value == {"type": "point", "dx": 3, "dy": 4}

        # Apply delta
        result = registry.apply_delta(p1, delta)
        assert result == p2

        # Test identity
        same_point = Point(1, 2)
        identity_delta = registry.compute_delta(same_point, same_point)
        assert identity_delta is not None
        assert registry._algebras[DeltaType.COMPOSITE].is_identity(identity_delta)

        # Test compose deltas
        p_start = Point(0, 0)
        p_mid = Point(1, 1)
        p_end = Point(2, 2)

        delta1 = registry.compute_delta(p_start, p_mid)
        delta2 = registry.compute_delta(p_mid, p_end)
        composed = registry._algebras[DeltaType.COMPOSITE].compose_deltas(
            delta1, delta2
        )

        result = registry.apply_delta(p_start, composed)
        assert result == p_end


class TestAutomaticDifferentiation:
    """Test automatic differentiation engine."""

    def test_basic_arithmetic_ad(self):
        registry = DeltaRegistry()
        ad_engine = AutoDiffEngine(registry)

        # Create a simple trace: x -> 2*x
        trace = ComputationTrace()
        x_idx = trace.add_source("x", 5.0)
        const_idx = trace.add_const(2.0)
        mul_idx = trace.add_op(TracedOp.MUL, [const_idx, x_idx], 10.0)

        # Test pushforward: Δx = 3, should give Δ(2*x) = 2*3 = 6
        input_deltas = {"x": TypedDelta(DeltaType.NUMERIC, 3.0)}
        output_delta = ad_engine.pushforward(trace, input_deltas, mul_idx)

        assert output_delta is not None
        assert output_delta.delta_type == DeltaType.NUMERIC
        assert abs(output_delta.value - 6.0) < 1e-10

    def test_complex_computation_ad(self):
        registry = DeltaRegistry()
        ad_engine = AutoDiffEngine(registry)

        # Create trace: z = 2*x + 3*y
        trace = ComputationTrace()
        x_idx = trace.add_source("x", 1.0)
        y_idx = trace.add_source("y", 2.0)

        const2_idx = trace.add_const(2.0)
        const3_idx = trace.add_const(3.0)

        mul1_idx = trace.add_op(TracedOp.MUL, [const2_idx, x_idx], 2.0)
        mul2_idx = trace.add_op(TracedOp.MUL, [const3_idx, y_idx], 6.0)
        add_idx = trace.add_op(TracedOp.ADD, [mul1_idx, mul2_idx], 8.0)

        # Test pushforward: Δx = 1, Δy = 1, should give Δz = 2*1 + 3*1 = 5
        input_deltas = {
            "x": TypedDelta(DeltaType.NUMERIC, 1.0),
            "y": TypedDelta(DeltaType.NUMERIC, 1.0),
        }
        output_delta = ad_engine.pushforward(trace, input_deltas, add_idx)

        assert output_delta is not None
        assert output_delta.delta_type == DeltaType.NUMERIC
        assert abs(output_delta.value - 5.0) < 1e-10

    def test_max_min_operations(self):
        registry = DeltaRegistry()
        ad_engine = AutoDiffEngine(registry)

        # Test max operation
        trace = ComputationTrace()
        a_idx = trace.add_source("a", 3.0)
        b_idx = trace.add_source("b", 1.0)
        max_idx = trace.add_op(TracedOp.MAX, [a_idx, b_idx], 3.0)

        # Δa = 2, a > b, so Δmax = Δa = 2
        input_deltas = {"a": TypedDelta(DeltaType.NUMERIC, 2.0)}
        output_delta = ad_engine.pushforward(trace, input_deltas, max_idx)
        assert output_delta.value == 2.0

    def test_trigonometric_operations(self):
        registry = DeltaRegistry()
        ad_engine = AutoDiffEngine(registry)

        # Test sin operation: d/dx sin(x) = cos(x)
        trace = ComputationTrace()
        x_idx = trace.add_source("x", 0.0)  # sin(0) = 0, cos(0) = 1
        sin_idx = trace.add_op(TracedOp.SIN, [x_idx], 0.0)

        input_deltas = {"x": TypedDelta(DeltaType.NUMERIC, 1.0)}
        output_delta = ad_engine.pushforward(trace, input_deltas, sin_idx)
        # cos(0) = 1, so Δsin = 1 * Δx = 1
        assert output_delta is not None
        assert abs(output_delta.value - 1.0) < 1e-10

    def test_exponential_operation(self):
        registry = DeltaRegistry()
        ad_engine = AutoDiffEngine(registry)

        # Test exp operation: d/dx exp(x) = exp(x)
        trace = ComputationTrace()
        x_idx = trace.add_source("x", 0.0)  # exp(0) = 1
        exp_idx = trace.add_op(TracedOp.EXP, [x_idx], 1.0)

        input_deltas = {"x": TypedDelta(DeltaType.NUMERIC, 1.0)}
        output_delta = ad_engine.pushforward(trace, input_deltas, exp_idx)
        # d/dx exp(0) = exp(0) = 1
        assert abs(output_delta.value - 1.0) < 1e-10


class TestReactiveStoreIntegration:
    """Integration tests with the full ReactiveStore."""

    def test_string_operations_in_store(self):
        store = ReactiveStore()

        store["text"] = "hello"
        store.computed("upper", lambda: store["text"].upper())

        assert store["upper"] == "HELLO"

        store["text"] = "world"
        assert store["upper"] == "WORLD"

    def test_mathematical_operations_in_store(self):
        store = ReactiveStore()

        store["x"] = 1.0
        store["y"] = 2.0

        # Test max operation
        store.computed("max_val", lambda: max(store["x"], store["y"]))
        assert store["max_val"] == 2.0

        store["x"] = 3.0
        assert store["max_val"] == 3.0

        # Test sin operation
        store.computed("sin_x", lambda: store["x"].sin())
        expected_sin = math.sin(3.0)
        assert abs(store["sin_x"] - expected_sin) < 1e-10

    def test_list_diff_in_store(self):
        store = ReactiveStore()

        store["items"] = [1, 2, 3, 4, 5]
        store.computed("length", lambda: len(store["items"]))

        assert store["length"] == 5

        # Modify list - should use efficient diff
        store["items"] = [1, 2, 3, 6, 4, 5]  # Insert 6 at position 3
        assert store["length"] == 6

        store["items"] = [1, 3, 6, 4, 5]  # Remove element at position 1
        assert store["length"] == 5

    def test_ad_fallback_logging(self, caplog):
        store = ReactiveStore()

        store["x"] = 1.0
        # Create a simple computation that uses AD
        store.computed("double", lambda: store["x"] * 2)

        assert store["double"] == 2.0

        # Change x - should use AD successfully
        store["x"] = 3.0
        assert store["double"] == 6.0

        # Create a more complex computation
        store["y"] = 1.0
        store.computed("complex", lambda: store["x"] + store["y"])
        assert store["complex"] == 4.0

        store["y"] = 2.0
        assert store["complex"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__])
