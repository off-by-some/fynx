"""
Comprehensive tests for FynX derived observables functionality.

This test suite validates .then() and >> operators for creating derived observables,
ensuring they work exactly as described in the derived-observables.md documentation.
"""

import pytest

from fynx import observable


class TestBasicTransformation:
    """Tests for basic .then() and >> functionality."""

    def test_then_method_creates_derived_observable(self):
        """then() creates a new observable with transformed values."""
        source = observable(5)

        def double(x):
            return x * 2

        derived = source.then(double)

        # Initial value
        assert derived.value == 10

        # Updates when source changes
        source.set(7)
        assert derived.value == 14

    def test_shift_operator_creates_derived_observable(self):
        """>> operator creates a new observable with transformed values."""
        source = observable(5)

        def double(x):
            return x * 2

        derived = source >> double

        # Initial value
        assert derived.value == 10

        # Updates when source changes
        source.set(7)
        assert derived.value == 14

    def test_then_and_shift_are_equivalent(self):
        """then() and >> produce identical results."""
        source = observable([1, 2, 3])

        def sum_list(lst):
            return sum(lst)

        derived_then = source.then(sum_list)
        derived_shift = source >> sum_list

        assert derived_then.value == derived_shift.value == 6

        source.set([4, 5, 6])
        assert derived_then.value == derived_shift.value == 15


class TestChainingTransformations:
    """Tests for chaining multiple transformations."""

    def test_chaining_with_then_method(self):
        """Can chain multiple then() calls."""
        numbers = observable([1, 2, 3])

        def sum_numbers(nums):
            return sum(nums)

        def format_total(total):
            return f"Total: {total}"

        # Chain transformations
        total = numbers.then(sum_numbers)
        description = total.then(format_total)

        assert description.value == "Total: 6"

        numbers.set([4, 5, 6])
        assert description.value == "Total: 15"

    def test_chaining_with_shift_operator(self):
        """Can chain multiple >> operators."""
        numbers = observable([1, 2, 3])

        def sum_numbers(nums):
            return sum(nums)

        def format_total(total):
            return f"Total: {total}"

        # Chain with >>
        description = numbers >> sum_numbers >> format_total

        assert description.value == "Total: 6"

        numbers.set([4, 5, 6])
        assert description.value == "Total: 15"

    def test_mixed_chaining_then_and_shift(self):
        """Can mix then() and >> in chains."""
        numbers = observable([1, 2, 3])

        def sum_numbers(nums):
            return sum(nums)

        def format_total(total):
            return f"Total: {total}"

        # Mix both approaches
        total = numbers.then(sum_numbers)  # then()
        description = total >> format_total  # then >>

        assert description.value == "Total: 6"

    def test_complex_transformation_pipeline(self):
        """Multi-step transformation pipeline from documentation."""
        raw_data = observable([1, 2, 3, None, 4, None])

        def filter_none(data):
            return [x for x in data if x is not None]

        def filter_positive(clean):
            return [x for x in clean if x > 0]

        def sum_values(filtered):
            return sum(filtered)

        def format_result(total):
            return f"Total: {total}"

        # Create pipeline
        result = (
            raw_data >> filter_none >> filter_positive >> sum_values >> format_result
        )

        assert result.value == "Total: 10"  # 1+2+3+4

        raw_data.set([5, None, -1, 10])
        assert result.value == "Total: 15"  # 5+10


class TestMultipleObservableTransformations:
    """Tests for transforming multiple observables using +."""

    def test_single_observable_transformation(self):
        """Transform single observable."""
        name = observable("alice")

        def create_greeting(n):
            return f"Hello, {n.title()}!"

        greeting = name.then(create_greeting)
        assert greeting.value == "Hello, Alice!"

        name.set("bob")
        assert greeting.value == "Hello, Bob!"

    def test_multiple_observables_with_then(self):
        """Transform multiple observables using + and then()."""
        first = observable("John")
        last = observable("Doe")

        def combine_names(first_name, last_name):
            return f"{first_name} {last_name}"

        full_name = (first + last).then(combine_names)
        assert full_name.value == "John Doe"

        first.set("Jane")
        assert full_name.value == "Jane Doe"

    def test_multiple_observables_with_shift(self):
        """Transform multiple observables using + and >>."""
        first = observable("John")
        last = observable("Doe")

        def combine_names(first_name, last_name):
            return f"{first_name} {last_name}"

        full_name = (first + last) >> combine_names
        assert full_name.value == "John Doe"

    def test_shopping_cart_example_from_docs(self):
        """Complete shopping cart example from documentation."""
        cart_items = observable([{"name": "Widget", "price": 10, "quantity": 2}])
        tax_rate = observable(0.08)
        shipping_threshold = observable(50)

        def calculate_subtotal(items):
            return sum(item["price"] * item["quantity"] for item in items)

        def calculate_tax(subtotal):
            return subtotal * tax_rate.value

        def calculate_shipping(subtotal):
            return 0 if subtotal >= shipping_threshold.value else 5.99

        def calculate_total(subtotal, tax, shipping):
            return subtotal + tax + shipping

        # Create derived observables
        subtotal = cart_items.then(calculate_subtotal)
        tax = subtotal.then(calculate_tax)
        shipping = subtotal.then(calculate_shipping)
        total = (subtotal + tax + shipping).then(calculate_total)

        # Initial values
        assert subtotal.value == 20.0
        assert tax.value == 1.6
        assert shipping.value == 5.99
        assert abs(total.value - 27.59) < 0.01

        # Add item and verify updates
        cart_items.set(
            cart_items.value + [{"name": "Gadget", "price": 15, "quantity": 1}]
        )

        assert subtotal.value == 35.0  # 20 + 15
        assert abs(tax.value - 2.8) < 0.01  # 35 * 0.08
        assert shipping.value == 5.99  # Still under threshold
        assert abs(total.value - 43.79) < 0.01  # 35 + 2.8 + 5.99


class TestReturnValues:
    """Tests for different return types from transformation functions."""

    def test_return_primitive_types(self):
        """Transformations can return primitives."""
        source = observable(42)

        doubled = source >> (lambda x: x * 2)
        as_string = source >> str
        as_bool = source >> bool

        assert doubled.value == 84
        assert as_string.value == "42"
        assert as_bool.value == True

    def test_return_collections(self):
        """Transformations can return lists, dicts, etc."""
        data = observable({"users": [{"name": "Alice"}, {"name": "Bob"}]})

        def extract_user_count(d):
            return len(d["users"])

        def extract_user_names(d):
            return [u["name"] for u in d["users"]]

        user_count = data.then(extract_user_count)
        user_names = data.then(extract_user_names)

        assert user_count.value == 2
        assert user_names.value == ["Alice", "Bob"]

    def test_return_observables(self):
        """Transformations can return new observables."""
        data = observable({"users": [{"name": "Alice"}, {"name": "Bob"}]})

        def create_count_observable(d):
            return observable(len(d["users"]))

        user_count_obs = data.then(create_count_observable)

        # The result is an observable
        assert hasattr(user_count_obs.value, "value")
        assert user_count_obs.value.value == 2


class TestPerformanceCharacteristics:
    """Tests for lazy evaluation and memoization."""

    def test_virtual_computed_recompute_on_each_access(self):
        """Virtual computed values recompute on each access (no caching)."""
        source = observable([10, 20, 30])

        call_count = [0]  # Use list to modify from inner function

        def expensive_sum(data):
            call_count[0] += 1
            return sum(data)

        derived = source >> expensive_sum

        # Each access recomputes (virtual ComputedObservable don't cache)
        assert derived.value == 60
        assert call_count[0] == 1

        assert derived.value == 60  # Recomputes
        assert call_count[0] == 2

        # Change dependency
        source.set([5, 10, 15])
        assert derived.value == 30
        assert call_count[0] == 3  # Incremented

    def test_virtual_computed_always_recompute(self):
        """Virtual computed values always recompute on access."""
        source = observable(5)

        transform_calls = []

        def track_transform(x):
            transform_calls.append(x)
            return x * 2

        derived = source >> track_transform

        # Access triggers computation
        assert derived.value == 10
        assert transform_calls == [5]

        # Virtual ComputedObservable don't cache - always recompute
        assert derived.value == 10
        assert transform_calls == [5, 5]  # Called again

        # Different value triggers recomputation
        source.set(7)
        assert derived.value == 14
        assert transform_calls == [5, 5, 7]


class TestErrorHandling:
    """Tests for error handling in transformations."""

    def test_transformation_errors_thrown_on_access(self):
        """Errors in transformation functions are thrown when value is accessed."""
        data = observable({"value": 42})

        def access_missing_key(d):
            return d["missing_key"] * 2  # KeyError

        # Transformation is created successfully
        derived_then = data.then(access_missing_key)
        derived_shift = data >> access_missing_key

        # Error occurs when accessing value
        with pytest.raises(KeyError):
            derived_then.value

        with pytest.raises(KeyError):
            derived_shift.value

    def test_handle_errors_by_preprocessing(self):
        """Handle errors by preprocessing data to safe format."""
        data = observable({"value": None})

        def safe_double(d):
            value = d.get("value")
            if value is None:
                return 0
            return value * 2

        derived = data >> safe_double
        assert derived.value == 0

        data.set({"value": 5})
        assert derived.value == 10


class TestExternalDependencies:
    """Tests for external state dependencies."""

    def test_external_variables_are_accessed_fresh(self):
        """External variables are accessed fresh on each computation."""
        external_multiplier = 2
        counter = observable(0)

        def multiply_by_external(c):
            return c * external_multiplier

        doubled = counter >> multiply_by_external

        counter.set(5)
        assert doubled.value == 10  # external_multiplier = 2

        # Changing external variable affects result
        external_multiplier = 3
        assert doubled.value == 15  # Now uses external_multiplier = 3


class TestBestPractices:
    """Tests demonstrating best practices from documentation."""

    def test_pure_functions_produce_predictable_results(self):
        """Pure functions give same output for same input."""
        source = observable("alice")

        def to_uppercase(n):
            return n.upper()

        uppercase = source >> to_uppercase
        assert uppercase.value == "ALICE"

        source.set("alice")  # Same input
        assert uppercase.value == "ALICE"  # Same output

    def test_handle_edge_cases_gracefully(self):
        """Transformations should handle edge cases."""
        numbers = observable([])

        def safe_average(nums):
            return sum(nums) / len(nums) if nums else 0

        average = numbers >> safe_average
        assert average.value == 0

        numbers.set([1, 2, 3])
        assert average.value == 2.0

    def test_descriptive_transformation_names(self):
        """Use descriptive names for transformations."""
        birth_date = observable(None)  # Would be datetime in real code

        def calculate_age(date):
            # Simplified for testing
            return 25 if date is not None else 0

        def is_adult(age):
            return age >= 18

        user_age = birth_date >> calculate_age
        is_adult_obs = user_age >> is_adult

        assert user_age.value == 0
        assert is_adult_obs.value == False

        # Set a birth date (simulated)
        birth_date.set("some_date")
        assert user_age.value == 25
        assert is_adult_obs.value == True

    def test_avoid_deep_nesting(self):
        """Break complex transformations into steps."""
        api_response = observable({"user": {"age": 25}})

        def extract_user_data(response):
            return response["user"]

        def extract_user_age(user_data):
            return user_data["age"]

        def is_adult(age):
            return age >= 18

        # Step by step
        user_data = api_response >> extract_user_data
        user_age = user_data >> extract_user_age
        is_adult_obs = user_age >> is_adult

        assert user_data.value == {"age": 25}
        assert user_age.value == 25
        assert is_adult_obs.value == True


class TestCommonPatterns:
    """Tests for common transformation patterns from documentation."""

    def test_data_validation_pattern(self):
        """Data validation transformation pattern."""
        email = observable("user@")

        def validate_email(e):
            return "@" in e and "." in e.split("@")[1]

        def email_feedback(valid):
            return "Valid" if valid else "Invalid"

        is_valid = email >> validate_email
        feedback = is_valid >> email_feedback

        assert is_valid.value == False
        assert feedback.value == "Invalid"

        email.set("user@example.com")
        assert is_valid.value == True
        assert feedback.value == "Valid"

    def test_data_formatting_pattern(self):
        """Data formatting transformation pattern."""
        price = observable(29.99)

        def format_price(p):
            return f"${p:.2f}"

        formatted = price >> format_price
        assert formatted.value == "$29.99"

        price.set(19.95)
        assert formatted.value == "$19.95"

    def test_collection_operations(self):
        """Collection transformation patterns."""
        items = observable([1, 2, 3, 4, 5])

        def filter_evens(lst):
            return [x for x in lst if x % 2 == 0]

        def double_items(lst):
            return [x * 2 for x in lst]

        def sum_items(lst):
            return sum(lst)

        # Filter
        evens = items >> filter_evens
        assert evens.value == [2, 4]

        # Map
        doubled = items >> double_items
        assert doubled.value == [2, 4, 6, 8, 10]

        # Reduce
        total = items >> sum_items
        assert total.value == 15

    def test_state_derivation_pattern(self):
        """State derivation transformation pattern."""
        app_state = observable("loading")

        def is_loading_state(s):
            return s == "loading"

        def is_error_state(s):
            return s == "error"

        def is_ready_state(s):
            return s == "ready"

        is_loading = app_state >> is_loading_state
        is_error = app_state >> is_error_state
        is_ready = app_state >> is_ready_state

        # Initial state
        assert is_loading.value == True
        assert is_error.value == False
        assert is_ready.value == False

        # Change to error
        app_state.set("error")
        assert is_loading.value == False
        assert is_error.value == True
        assert is_ready.value == False

        # Change to ready
        app_state.set("ready")
        assert is_loading.value == False
        assert is_error.value == False
        assert is_ready.value == True


class TestIntegrationWithOtherOperators:
    """Tests for using .then() and >> with other FynX operators."""

    def test_with_plus_operator(self):
        """Use transformations with + for multi-input."""
        prices = observable([10, 20, 30])
        discount_rate = observable(0.1)

        def calculate_discounted_total(prices, rate):
            return sum(price * (1 - rate) for price in prices)

        discounted_total = (prices + discount_rate) >> calculate_discounted_total

        # 10*0.9 + 20*0.9 + 30*0.9 = 9 + 18 + 27 = 54
        assert discounted_total.value == 54.0

    def test_with_conditionals(self):
        """Use transformations with conditional operators."""
        total = observable(45)

        def is_expensive(t):
            return t > 50

        def format_expensive_message(is_exp):
            return "High-value order" if is_exp else "Regular order"

        is_expensive_obs = total >> is_expensive
        message = is_expensive_obs >> format_expensive_message

        assert is_expensive_obs.value == False
        assert message.value == "Regular order"

        total.set(60)
        assert is_expensive_obs.value == True
        assert message.value == "High-value order"


class TestFusionPerformance:
    """Tests for fusion optimization to prevent recursion errors in long chains."""

    def test_long_chain_no_recursion_error(self):
        """Long transformation chains should not cause recursion errors due to fusion."""
        # This test ensures fusion prevents stack overflow in long chains
        base = observable(0)
        current = base

        # Create a chain of 1000 transformations (would cause recursion without fusion)
        chain_length = 1000
        for i in range(chain_length):
            current = current >> (lambda x, i=i: x + 1)

        # Access value should work without recursion error
        result = current.value
        assert result == chain_length

        # Change source and verify propagation still works
        base.set(10)
        result_after_change = current.value
        assert result_after_change == 10 + chain_length

    def test_deeply_nested_chains_maintain_correctness(self):
        """Deep chains should maintain mathematical correctness."""
        base = observable(1)
        current = base

        # Create chain: x -> x*2 -> x+1 -> x*3 -> x-1 -> ... -> final result
        transformations = [
            lambda x: x * 2,  # 1 -> 2
            lambda x: x + 1,  # 2 -> 3
            lambda x: x * 3,  # 3 -> 9
            lambda x: x - 1,  # 9 -> 8
            lambda x: x + 10,  # 8 -> 18
        ]

        expected_values = [1, 2, 3, 9, 8, 18]

        for i, transform in enumerate(transformations):
            current = current >> transform
            assert current.value == expected_values[i + 1]

        # Verify final result
        assert current.value == 18

        # Change source and verify all transformations reapply correctly
        base.set(2)
        # 2 -> 4 -> 5 -> 15 -> 14 -> 24
        assert current.value == 24
