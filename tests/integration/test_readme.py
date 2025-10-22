"""
Tests for README.md examples to ensure they work as documented.

This test file verifies that all code examples in the README.md actually execute
correctly and produce the expected results. It serves as both documentation
verification and a regression test suite.
"""

import pytest

from fynx import Store, observable, reactive


class TestReadmeExamples:
    """Test all code examples from the README.md file."""

    def test_quick_start_example(self):
        """Test the Quick Start example with CartStore."""

        class CartStore(Store):
            item_count = observable(1)
            price_per_item = observable(10.0)

        # Reactive computation
        total_price = (CartStore.item_count + CartStore.price_per_item) >> (
            lambda t: t[0] * t[1]
        )

        # Test initial state
        assert total_price.value == 10.0  # 1 * 10.0

        # Track subscription calls
        call_log = []
        total_price.subscribe(lambda total: call_log.append(total))

        # Automatic updates
        CartStore.item_count = 3  # Should trigger subscription with 30.00
        assert total_price.value == 30.0
        assert call_log == [30.0]

        CartStore.price_per_item = 12.50  # Should trigger subscription with 37.50
        assert total_price.value == 37.5
        assert call_log == [30.0, 37.5]

    def test_observables_example(self):
        """Test the Observables section examples."""
        # Standalone observable
        counter = observable(0)
        counter.set(1)  # Triggers reactive updates
        assert counter.value == 1

        # Store-based observables
        class AppState(Store):
            username = observable("")
            is_logged_in = observable(False)

        AppState.username = "off-by-some"  # Normal assignment, reactive behavior
        assert AppState.username == "off-by-some"
        assert AppState.is_logged_in.value is False

    def test_transforming_data_with_rshift(self):
        """Test the >> operator examples."""
        # Inline transformations
        counter = observable(5)
        result = (
            counter
            >> (lambda x: x * 2)  # 10
            >> (lambda x: x + 10)  # 20
            >> (lambda x: f"Result: {x}")
        )  # "Result: 20"

        assert result.value == "Result: 20"

        # Reusable transformations
        doubled = counter.then(lambda x: x * 2)
        assert doubled.value == 10

    def test_combining_observables_with_or(self):
        """Test the + operator examples."""

        class User(Store):
            first_name = observable("John")
            last_name = observable("Doe")

        # Combine and transform
        full_name = (User.first_name + User.last_name) >> (lambda t: f"{t[0]} {t[1]}")
        assert full_name.value == "John Doe"

        User.first_name = "Jane"
        assert full_name.value == "Jane Doe"

    def test_filtering_with_and_not(self):
        """Test the & and ~ operator examples."""
        uploaded_file = observable(None)
        is_processing = observable(False)

        # Conditional observables
        is_valid = uploaded_file >> (lambda f: f is not None)
        preview_ready = uploaded_file & is_valid & (~is_processing)

        # Initially conditions never met, so accessing value raises ConditionalNeverMet
        from fynx.observable.conditional import ConditionalNeverMet

        with pytest.raises(ConditionalNeverMet):
            _ = preview_ready.value

        # Set file - now all conditions are met, so preview_ready should be "file.txt"
        uploaded_file.set("file.txt")
        assert is_valid.value is True  # File is now valid
        assert preview_ready.value == "file.txt"  # All conditions met

        # Set processing to True - should block
        is_processing.set(True)

        # Accessing value when conditions are unmet should raise ConditionalNotMet
        from fynx.observable.conditional import ConditionalNotMet

        with pytest.raises(
            ConditionalNotMet, match="Conditions are not currently satisfied"
        ):
            _ = preview_ready.value

        # Clear processing - should unblock
        is_processing.set(False)
        assert preview_ready.value == "file.txt"

    def test_reacting_to_changes(self):
        """Test the reaction examples."""
        # Dedicated reaction functions
        obs = observable(0)
        call_log = []

        @reactive(obs)
        def handle_change(value):
            call_log.append(f"Changed: {value}")

        # Reactive decorator runs immediately on the current value
        assert call_log == ["Changed: 0"]

        obs.set(5)
        obs.set(10)
        assert call_log == ["Changed: 0", "Changed: 5", "Changed: 10"]

        # Inline reactions - use fresh observable to avoid reactive decorator interference
        obs2 = observable(0)
        call_log.clear()

        obs2.subscribe(lambda x: call_log.append(f"New value: {x}"))
        obs2.set(15)
        # Note: subscribe doesn't trigger immediately, only on changes
        assert call_log == ["New value: 15"]

        # Conditional reactions
        condition1 = observable(True)
        condition2 = observable(False)
        watch_log = []

        @reactive(condition1 & condition2)
        def on_conditions_met(value):
            watch_log.append("All conditions satisfied!")

        # Initially conditions not met (condition2 is False)
        assert len(watch_log) == 0

        # Make condition2 true - should trigger
        condition2.set(True)
        assert watch_log == ["All conditions satisfied!"]

    def test_functor_examples(self):
        """Test the mathematical foundation functor examples."""

        # Regular function on values
        def double(x):
            return x * 2

        def add_ten(x):
            return x + 10

        value = 5
        result = add_ten(double(value))  # 20
        assert result == 20

        # Same composition, lifted to observables
        obs = observable(5)
        obs_result = obs >> double >> add_ten  # Observable(20)
        assert obs_result.value == 20

    def test_functor_laws(self):
        """Test the functor laws: identity and composition."""
        # Identity law: O(id) = id
        obs = observable(42)
        identity_result = (obs >> (lambda x: x)).value
        assert identity_result == obs.value

        # Composition law: O(g ∘ f) = O(g) ∘ O(f)
        def double(x):
            return x * 2

        def add_ten(x):
            return x + 10

        obs2 = observable(5)
        composed = obs2 >> (lambda x: add_ten(double(x)))
        chained = obs2 >> double >> add_ten
        assert composed.value == chained.value == 20

    def test_product_examples(self):
        """Test the product examples from mathematical foundation."""
        first_name = observable("Jane")
        last_name = observable("Doe")

        # Product creates a tuple observable
        full_name = (first_name + last_name) >> (lambda t: f"{t[0]} {t[1]}")
        assert full_name.value == "Jane Doe"

        first_name.set("John")  # full_name automatically becomes "John Doe"
        assert full_name.value == "John Doe"

    def test_product_associativity(self):
        """Test product associativity."""
        a = observable(1)
        b = observable(2)
        c = observable(3)

        # Associativity: (a + b) + c ≅ a + (b + c)
        left_assoc = (a + b) + c  # ((1, 2), 3) -> (1, 2, 3)
        right_assoc = a + (b + c)  # (1, (2, 3)) -> (1, 2, 3)

        # Both should flatten to the same tuple
        assert left_assoc.value == (1, 2, 3)
        assert right_assoc.value == (1, 2, 3)

    def test_pullback_examples(self):
        """Test the pullback examples."""
        data = observable(42)
        is_positive = data >> (lambda x: x > 0)
        is_even = data >> (lambda x: x % 2 == 0)

        # Pullback: only emits when both conditions hold
        filtered = data & is_positive & is_even

        # Initial value should be 42 (positive and even)
        assert filtered.value == 42

        # Test different values
        data.set(42)  # positive and even
        assert filtered.value == 42

        data.set(-4)  # not positive

        # Accessing value when conditions are unmet should raise ConditionalNotMet
        from fynx.observable.conditional import ConditionalNotMet

        with pytest.raises(
            ConditionalNotMet, match="Conditions are not currently satisfied"
        ):
            _ = filtered.value

        data.set(7)  # not even

        # Still throws error
        with pytest.raises(
            ConditionalNotMet, match="Conditions are not currently satisfied"
        ):
            _ = filtered.value

        data.set(10)  # positive and even
        assert filtered.value == 10

    def test_pullback_commutativity(self):
        """Test pullback commutativity."""
        data = observable(42)
        is_positive = data >> (lambda x: x > 0)
        is_even = data >> (lambda x: x % 2 == 0)

        # Commutativity: a & b ≡ b & a
        filter1 = data & is_positive & is_even
        filter2 = data & is_even & is_positive

        # Both should behave identically
        assert filter1.value == filter2.value == 42

        data.set(42)
        assert filter1.value == filter2.value == 42

    def test_complex_composition(self):
        """Test the complex composition example."""
        # Complex composition: all laws hold automatically
        price = observable(100.0)
        quantity = observable(3)
        discount = observable(0.1)
        is_valid = quantity >> (lambda q: q > 0)

        total = ((price + quantity) >> (lambda t: t[0] * t[1])) & is_valid
        discounted = total >> (
            lambda t: t * (1 - discount.value) if t is not None else 0
        )

        # Initial calculations
        assert total.value == 300.0  # 100 * 3
        assert discounted.value == 270.0  # 300 * (1 - 0.1)

        quantity.set(5)  # Everything updates correctly by mathematical necessity

        assert total.value == 500.0  # 100 * 5
        assert discounted.value == 450.0  # 500 * (1 - 0.1)

        # Verify the math
        expected = 100.0 * 5 * (1 - 0.1)  # 450.0
        assert abs(discounted.value - expected) < 0.001
