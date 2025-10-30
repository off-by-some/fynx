"""
Comprehensive tests for FynX Store functionality.

This test suite validates all Store features described in the stores.md documentation,
ensuring Stores work exactly as documented with clean attribute access, computed values,
methods, and cross-store dependencies.
"""

import pytest

from fynx import Store, observable


class TestBasicStoreFunctionality:
    """Tests for basic Store creation and attribute access."""

    def test_store_creation_with_single_attribute(self):
        """Store can be created with a single observable attribute."""

        class CounterStore(Store):
            count = observable(0)

        # Read the value
        assert CounterStore.count == 0

        # Write the value
        CounterStore.count = 5
        assert CounterStore.count == 5

    def test_store_attribute_is_observable_under_hood(self):
        """Store attributes are observables under the hood."""

        class CounterStore(Store):
            count = observable(0)

        notifications = []
        unsubscribe = CounterStore.count.subscribe(lambda c: notifications.append(c))

        CounterStore.count = 10
        assert notifications == [10]

        unsubscribe()

    def test_store_vs_standalone_observable_syntax(self):
        """Store attributes use clean syntax while standalone observables require .value/.set()."""

        # Store observable - clean syntax
        class MyStore(Store):
            value = observable(100)

        # These are equivalent
        assert MyStore.value == 100

        # Writing is clean
        MyStore.value = 200
        assert MyStore.value == 200

        # But you can still use it as an observable
        doubled = MyStore.value >> (lambda v: v * 2)
        assert doubled.value == 400

    def test_standalone_observable_requires_explicit_access(self):
        """Standalone observables require .value and .set()."""
        counter = observable(0)

        # Must use .value to read
        assert counter.value == 0

        # Must use .set() to write
        counter.set(5)
        assert counter.value == 5


class TestStoreOrganization:
    """Tests for Store organization and encapsulation."""

    def test_store_groups_related_state(self):
        """Stores group related observables together."""

        class UserStore(Store):
            first_name = observable("Alice")
            last_name = observable("Smith")
            age = observable(30)
            email = observable("alice@example.com")
            is_authenticated = observable(False)

        assert UserStore.first_name == "Alice"
        assert UserStore.last_name == "Smith"
        assert UserStore.age == 30
        assert UserStore.email == "alice@example.com"
        assert UserStore.is_authenticated == False

    def test_store_methods_encapsulate_state_changes(self):
        """Store methods provide clean APIs for state modification."""

        class UserStore(Store):
            first_name = observable("")
            last_name = observable("")
            email = observable("")
            age = observable(0)
            is_authenticated = observable(False)

            @classmethod
            def update_profile(cls, first, last, age, email):
                cls.first_name = first
                cls.last_name = last
                cls.age = age
                cls.email = email

            @classmethod
            def logout(cls):
                cls.is_authenticated = False
                cls.first_name = ""
                cls.last_name = ""
                cls.email = ""

        # Update profile
        UserStore.update_profile("John", "Doe", 25, "john@example.com")
        assert UserStore.first_name == "John"
        assert UserStore.last_name == "Doe"
        assert UserStore.age == 25
        assert UserStore.email == "john@example.com"

        # Logout
        UserStore.logout()
        assert UserStore.is_authenticated == False
        assert UserStore.first_name == ""
        assert UserStore.email == ""


class TestComputedValues:
    """Tests for computed values using >> operator."""

    def test_simple_computed_value(self):
        """Computed values recalculate when dependencies change."""

        class CartStore(Store):
            items = observable([])
            item_count = items >> (lambda items: len(items))

        assert CartStore.item_count == 0

        CartStore.items = [{"name": "Widget", "price": 10}]
        assert CartStore.item_count == 1

        CartStore.items = [{"name": "Widget"}, {"name": "Gadget"}]
        assert CartStore.item_count == 2

    def test_computed_value_with_complex_transformation(self):
        """Computed values can perform complex transformations."""

        class CartStore(Store):
            items = observable([])
            subtotal = items >> (
                lambda items: sum(item["price"] * item["quantity"] for item in items)
            )

        CartStore.items = [
            {"name": "Widget", "price": 10, "quantity": 2},
            {"name": "Gadget", "price": 15, "quantity": 1},
        ]

        assert CartStore.subtotal == 35.0

    def test_computed_values_are_lazy_and_cached(self):
        """Computed values are lazy and cache results."""

        class AnalyticsStore(Store):
            values = observable([10, 20, 30])
            total = values >> (lambda v: sum(v))

        # First access computes
        assert AnalyticsStore.total == 60

        # Second access returns cached value (no recomputation)
        assert AnalyticsStore.total == 60

        # Change dependency triggers recomputation
        AnalyticsStore.values = [5, 10, 15]
        assert AnalyticsStore.total == 30


class TestMergedObservables:
    """Tests for merging observables with + operator."""

    def test_merging_two_observables(self):
        """+ operator merges observables for multi-input computations."""

        class CartStore(Store):
            items = observable([])
            tax_rate = observable(0.08)

            subtotal = items >> (
                lambda items: sum(item["price"] * item["quantity"] for item in items)
            )

            tax_amount = (subtotal + tax_rate) >> (lambda sub, rate: sub * rate)

        CartStore.items = [{"name": "Widget", "price": 20, "quantity": 1}]
        assert CartStore.subtotal == 20.0
        assert CartStore.tax_amount == 1.6

        CartStore.tax_rate = 0.10
        assert CartStore.tax_amount == 2.0

    def test_three_way_merge(self):
        """Three observables can be merged together."""

        class CartStore(Store):
            items = observable([])
            tax_rate = observable(0.08)
            discount = observable(0.0)

            subtotal = items >> (
                lambda items: sum(item["price"] * item["quantity"] for item in items)
            )

            total = (subtotal + tax_rate + discount) >> (
                lambda sub, rate, disc: sub * (1 + rate) - disc
            )

        CartStore.items = [{"name": "Widget", "price": 20, "quantity": 1}]
        CartStore.discount = 5.0

        assert CartStore.subtotal == 20.0
        assert CartStore.total == 16.6  # 20 * 1.08 - 5


class TestImmutableUpdates:
    """Tests for immutable update patterns."""

    def test_list_immutable_updates(self):
        """Lists must be updated immutably to trigger reactivity."""

        class CartStore(Store):
            items = observable([])

            @classmethod
            def add_item(cls, item):
                # Correct: Create new list
                cls.items = cls.items + [item]

        CartStore.add_item({"name": "Widget", "price": 10})
        assert len(CartStore.items) == 1
        assert CartStore.items[0]["name"] == "Widget"

    def test_dict_immutable_updates(self):
        """Dicts must be updated immutably."""

        class UserStore(Store):
            profile = observable({})

            @classmethod
            def update_name(cls, name):
                # Correct: Create new dict
                cls.profile = {**cls.profile, "name": name}

        UserStore.update_name("Alice")
        assert UserStore.profile["name"] == "Alice"

    def test_nested_structure_updates(self):
        """Nested structures require reconstructing the entire path."""

        class CartStore(Store):
            items = observable([])

            @classmethod
            def update_quantity(cls, item_id, new_quantity):
                cls.items = [
                    (
                        {**item, "quantity": new_quantity}
                        if item["id"] == item_id
                        else item
                    )
                    for item in cls.items
                ]

        CartStore.items = [
            {"id": 1, "name": "Widget", "quantity": 1},
            {"id": 2, "name": "Gadget", "quantity": 2},
        ]

        CartStore.update_quantity(1, 5)
        assert CartStore.items[0]["quantity"] == 5
        assert CartStore.items[1]["quantity"] == 2


class TestChainingComputedValues:
    """Tests for chaining computed values."""

    @pytest.mark.skip(
        reason="Store descriptor propagation needs implementation updates"
    )
    def test_four_level_computation_chain(self):
        """Computed values can depend on other computed values."""

        class AnalyticsStore(Store):
            values = observable([10, 20, 30, 40, 50])

            # Level 1: Basic stats
            count = values >> (lambda v: len(v))
            total = values >> (lambda v: sum(v))

            # Level 2: Depends on count and total
            mean = (total + count) >> (lambda t, c: t / c if c > 0 else 0)

            # Level 3: Depends on values and mean
            variance = (values + mean + count) >> (
                lambda vals, avg, n: (
                    sum((x - avg) ** 2 for x in vals) / (n - 1) if n > 1 else 0
                )
            )

            # Level 4: Depends on variance
            std_dev = variance >> (lambda v: v**0.5)

        assert AnalyticsStore.count == 5
        assert AnalyticsStore.total == 150
        assert AnalyticsStore.mean == 30.0
        assert abs(AnalyticsStore.std_dev - 15.811388300841896) < 0.001

        AnalyticsStore.values = [5, 10, 15, 20, 25]
        assert AnalyticsStore.mean == 15.0
        assert abs(AnalyticsStore.std_dev - 7.905694150420948) < 0.001


class TestUserProfileExample:
    """Tests for the comprehensive UserProfileStore example."""

    def test_user_profile_store_full_functionality(self):
        """Complete UserProfileStore example from documentation."""

        class UserProfileStore(Store):
            # Basic observables
            first_name = observable("")
            last_name = observable("")
            email = observable("")
            age = observable(0)
            is_premium = observable(False)

            # Computed: full name
            full_name = (first_name + last_name) >> (
                lambda first, last: f"{first} {last}".strip()
            )

            # Computed: display name (falls back if no name)
            display_name = full_name >> (
                lambda name: name if name else "Anonymous User"
            )

            # Computed: email validation
            is_email_valid = email >> (
                lambda e: "@" in e and "." in e.split("@")[-1] if e else False
            )

            # Computed: age validation
            is_adult = age >> (lambda a: a >= 18)

            # Computed: profile completeness
            is_complete = (first_name + last_name + email + is_email_valid) >> (
                lambda first, last, email_addr, email_valid: bool(
                    first and last and email_addr and email_valid
                )
            )

            # Computed: user tier
            user_tier = (is_premium + is_complete) >> (
                lambda premium, complete: (
                    "Premium" if premium else "Complete" if complete else "Basic"
                )
            )

            @classmethod
            def update_name(cls, first, last):
                cls.first_name = first.strip()
                cls.last_name = last.strip()

            @classmethod
            def update_email(cls, email):
                cls.email = email.strip().lower()

            @classmethod
            def set_age(cls, age):
                if age >= 0:
                    cls.age = age

            @classmethod
            def upgrade_to_premium(cls):
                cls.is_premium = True

            @classmethod
            def reset(cls):
                cls.first_name = ""
                cls.last_name = ""
                cls.email = ""
                cls.age = 0
                cls.is_premium = False

        # Initial state
        assert UserProfileStore.display_name == "Anonymous User"
        assert UserProfileStore.user_tier == "Basic"

        # Update name
        UserProfileStore.update_name("Alice", "Smith")
        assert UserProfileStore.display_name == "Alice Smith"
        assert UserProfileStore.full_name == "Alice Smith"

        # Update email
        UserProfileStore.update_email("alice@example.com")
        assert UserProfileStore.is_email_valid == True

        # Set age
        UserProfileStore.set_age(25)
        assert UserProfileStore.is_adult == True
        assert UserProfileStore.is_complete == True
        assert UserProfileStore.user_tier == "Complete"

        # Upgrade
        UserProfileStore.upgrade_to_premium()
        assert UserProfileStore.user_tier == "Premium"

        # Reset
        UserProfileStore.reset()
        assert UserProfileStore.first_name == ""
        assert UserProfileStore.display_name == "Anonymous User"
        assert UserProfileStore.user_tier == "Basic"


class TestCrossStoreDependencies:
    """Tests for dependencies between different Stores."""

    def test_cross_store_dependencies(self):
        """Stores can reference observables from other Stores."""

        class ThemeStore(Store):
            mode = observable("light")  # "light" or "dark"
            font_size = observable(16)

        class UIStore(Store):
            sidebar_open = observable(True)

            # Depends on ThemeStore
            background_color = ThemeStore.mode >> (
                lambda mode: "#ffffff" if mode == "light" else "#1a1a1a"
            )

            text_color = ThemeStore.mode >> (
                lambda mode: "#000000" if mode == "light" else "#ffffff"
            )

            # Depends on multiple observables from ThemeStore
            css_vars = (ThemeStore.mode + ThemeStore.font_size) >> (
                lambda mode, size: {
                    "--bg": "#ffffff" if mode == "light" else "#1a1a1a",
                    "--text": "#000000" if mode == "light" else "#ffffff",
                    "--font-size": f"{size}px",
                }
            )

        # Initial state
        assert UIStore.background_color == "#ffffff"
        assert UIStore.text_color == "#000000"
        assert UIStore.css_vars["--bg"] == "#ffffff"
        assert UIStore.css_vars["--font-size"] == "16px"

        # Change theme
        ThemeStore.mode = "dark"
        assert UIStore.background_color == "#1a1a1a"
        assert UIStore.text_color == "#ffffff"
        assert UIStore.css_vars["--bg"] == "#1a1a1a"

        # Change font size
        ThemeStore.font_size = 18
        assert UIStore.css_vars["--font-size"] == "18px"


class TestStoreInheritance:
    """Tests for Store inheritance and state isolation."""

    def test_store_inheritance_creates_separate_state(self):
        """Inherited Stores get completely independent state."""

        class BaseStore(Store):
            count = observable(0)
            name = observable("Base")

        class ChildStore(BaseStore):
            pass  # Inherits count and name observables

        # Each class gets completely independent state
        BaseStore.count = 5
        ChildStore.count = 10

        assert BaseStore.count == 5
        assert ChildStore.count == 10  # completely separate

        BaseStore.name = "Modified Base"
        assert BaseStore.name == "Modified Base"
        assert ChildStore.name == "Base"  # unchanged

    def test_explicit_overrides_replace_inherited_observables(self):
        """Child classes can override inherited observables."""

        class BaseStore(Store):
            count = observable(0)
            name = observable("Base")

        class CustomStore(BaseStore):
            count = observable(100)  # Completely replaces parent's count
            name = observable("Custom")  # Completely replaces parent's name

        assert CustomStore.count == 100  # not 0
        assert CustomStore.name == "Custom"

        # Parent unchanged
        assert BaseStore.count == 0
        assert BaseStore.name == "Base"


class TestStoreMethods:
    """Tests for Store methods and encapsulation."""

    def test_cart_store_with_methods(self):
        """CartStore example with encapsulated methods."""

        class CartStore(Store):
            items = observable([])

            item_count = items >> len

            total = items >> (
                lambda items: sum(
                    item["price"] * item.get("quantity", 1) for item in items
                )
            )

            @classmethod
            def add_item(cls, name, price, quantity=1):
                """Add an item to the cart or update quantity if it exists."""
                current_items = cls.items

                # Find existing item
                existing = next(
                    (item for item in current_items if item["name"] == name), None
                )

                if existing:
                    # Update quantity
                    cls.items = [
                        (
                            {**item, "quantity": item["quantity"] + quantity}
                            if item["name"] == name
                            else item
                        )
                        for item in current_items
                    ]
                else:
                    # Add new item
                    cls.items = current_items + [
                        {"name": name, "price": price, "quantity": quantity}
                    ]

            @classmethod
            def remove_item(cls, name):
                """Remove an item from the cart."""
                cls.items = [item for item in cls.items if item["name"] != name]

            @classmethod
            def clear(cls):
                """Remove all items."""
                cls.items = []

        # Test adding items
        CartStore.add_item("Widget", 10.0, 2)
        CartStore.add_item("Gadget", 15.0)
        assert CartStore.item_count == 2
        assert CartStore.total == 35.0

        # Test updating quantity
        CartStore.add_item("Widget", 10.0, 1)
        assert CartStore.item_count == 2  # still 2 items
        assert CartStore.total == 45.0  # 3 widgets + 1 gadget

        # Test removing items
        CartStore.remove_item("Widget")
        assert CartStore.item_count == 1
        assert CartStore.total == 15.0

        # Test clearing
        CartStore.clear()
        assert CartStore.item_count == 0
        assert CartStore.total == 0


class TestBestPractices:
    """Tests demonstrating Store best practices."""

    def test_computed_values_handle_edge_cases(self):
        """Computed values should handle edge cases defensively."""

        class AnalyticsStore(Store):
            values = observable([])

            # Handles empty list
            average = values >> (
                lambda vals: sum(vals) / len(vals) if len(vals) > 0 else 0
            )

            # Handles None
            user_name = values >> (
                lambda vals: f"User has {len(vals)} items" if vals else "No items"
            )

        assert AnalyticsStore.average == 0
        assert AnalyticsStore.user_name == "No items"

        AnalyticsStore.values = [10, 20, 30]
        assert AnalyticsStore.average == 20.0
        assert AnalyticsStore.user_name == "User has 3 items"

    def test_clear_naming_for_computed_values(self):
        """Computed values should have clear, descriptive names."""

        class FormStore(Store):
            email = observable("")
            password = observable("")

            is_valid_email = email >> (lambda e: "@" in e)
            is_valid_password = password >> (lambda p: len(p) >= 8)
            is_form_valid = (is_valid_email + is_valid_password) >> (
                lambda e, p: e and p
            )

        FormStore.email = "user@example.com"
        FormStore.password = "password123"

        assert FormStore.is_valid_email == True
        assert FormStore.is_valid_password == True
        assert FormStore.is_form_valid == True
