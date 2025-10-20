"""Integration tests for reactive system interactions between components."""

import pytest

from fynx import Store, observable


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
@pytest.mark.operators
def test_diamond_dependency_maintains_invariant_relationships(diamond_dependency):
    """Diamond dependency pattern maintains correct relationships when source changes"""
    # Arrange - diamond_dependency fixture provides (source, path_a, path_b, combined)
    source, path_a, path_b, combined = diamond_dependency

    # Act - Change source value
    source.set(20)

    # Assert - Combined should reflect both transformation paths
    assert combined.value == 65  # (20 + 5) + (20 * 2)
    assert path_a.value == 25  # 20 + 5
    assert path_b.value == 40  # 20 * 2


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
def test_temperature_monitor_store_updates_dependents(temperature_monitor):
    """Temperature monitor store correctly updates computed values"""
    # Arrange - temperature_monitor fixture provides TemperatureMonitor class
    TemperatureMonitor = temperature_monitor

    # Act - Set celsius temperature
    TemperatureMonitor.celsius = 100.0

    # Assert - Fahrenheit should be computed correctly
    assert TemperatureMonitor.fahrenheit.value == 212.0


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
def test_user_profile_store_maintains_full_name(user_profile):
    """User profile store maintains full name as first/last names change"""
    # Arrange - user_profile fixture provides UserProfile class
    UserProfile = user_profile

    # Act - Set names
    UserProfile.first_name = "John"
    UserProfile.last_name = "Doe"

    # Assert - Full name is computed correctly
    assert UserProfile.full_name.value == "John Doe"

    # Act - Change first name
    UserProfile.first_name = "Jane"

    # Assert - Full name updates
    assert UserProfile.full_name.value == "Jane Doe"


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
@pytest.mark.operators
def test_counter_with_bounds_checking_initial_state_is_valid(counter_with_limits):
    """Counter with bounds checking starts in valid state"""
    # Arrange - counter_with_limits fixture provides (counter, min_val, max_val, is_valid)
    counter, min_val, max_val, is_valid = counter_with_limits

    # Act & Assert - Initial state should be valid
    assert is_valid.value is True


def test_counter_with_bounds_checking_validates_within_bounds(counter_with_limits):
    """Counter with bounds checking validates values within bounds"""
    # Arrange - counter_with_limits fixture provides (counter, min_val, max_val, is_valid)
    counter, min_val, max_val, is_valid = counter_with_limits

    # Act - Set counter within bounds
    counter.set(50)

    # Assert - Should be valid
    assert is_valid.value is True


def test_counter_with_bounds_checking_rejects_below_minimum(counter_with_limits):
    """Counter with bounds checking rejects values below minimum"""
    # Arrange - counter_with_limits fixture provides (counter, min_val, max_val, is_valid)
    counter, min_val, max_val, is_valid = counter_with_limits

    # Act - Set counter below minimum
    counter.set(-5)

    # Assert - Should be invalid
    assert is_valid.value is False


def test_counter_with_bounds_checking_accepts_values_back_within_bounds(
    counter_with_limits,
):
    """Counter with bounds checking accepts values when set back within bounds"""
    # Arrange - counter_with_limits fixture provides (counter, min_val, max_val, is_valid)
    counter, min_val, max_val, is_valid = counter_with_limits

    # Act - Set counter back within bounds
    counter.set(25)

    # Assert - Should be valid
    assert is_valid.value is True


def test_counter_with_bounds_checking_handles_dynamic_bounds_changes(
    counter_with_limits,
):
    """Counter with bounds checking handles dynamic bounds changes"""
    # Arrange - counter_with_limits fixture provides (counter, min_val, max_val, is_valid)
    counter, min_val, max_val, is_valid = counter_with_limits

    # Act - Set counter to 25 first
    counter.set(25)

    # Act - Change bounds dynamically
    min_val.set(10)

    # Assert - 25 is still within new bounds (10 <= 25 <= 100)
    assert is_valid.value is True

    # Act - Change max bound and set counter
    max_val.set(50)
    counter.set(30)

    # Assert - Should be valid
    assert is_valid.value is True


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
@pytest.mark.operators
def test_reactive_filter_chain_processes_values_correctly(reactive_filter_chain):
    """Reactive filter chain correctly filters and transforms values"""
    # Arrange - reactive_filter_chain fixture provides (source, predicate, multiplier, result)
    source, predicate, multiplier, result = reactive_filter_chain

    # Act - Set positive value (should pass filter)
    source.set(5)

    # Assert - Value passes filter and gets transformed
    assert result.value == 10  # 5 * 2

    # Act - Set negative value (should be filtered out)
    source.set(-3)

    # Assert - Negative value filtered, result becomes 0
    assert result.value == 0

    # Act - Change multiplier and test positive value again
    multiplier.set(3)
    source.set(4)

    # Assert - New multiplier applied to filtered value
    assert result.value == 12  # 4 * 3


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
@pytest.mark.memory
def test_reactive_chain_memory_cleanup(memory_test_chain, no_leaks):
    """Reactive chains can be created and cleaned up without memory leaks"""
    # Arrange - memory_test_chain fixture provides (source, chain_elements)
    source, chain_elements = memory_test_chain

    # Act - Create and destroy multiple reactive chains
    def create_and_destroy_chains():
        for i in range(100):
            # Create temporary chains that reference the source
            temp_chain = source >> (lambda x: x + i) >> (lambda x: f"result_{x}")
            # Simulate use
            source.set(i)
            # Cleanup - delete the chain
            del temp_chain

    # Assert - No memory leaks during chain creation/destruction
    no_leaks(create_and_destroy_chains, "Observable")


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
def test_store_subscription_notifies_on_any_observable_change():
    """Store subscription system notifies subscribers of any observable changes"""

    class ShoppingCart(Store):
        item_count = observable(0)
        total_price = observable(0.0)
        discount_percent = observable(0.0)

        final_price = (total_price + discount_percent).then(
            lambda price, discount: price * (1 - discount / 100)
        )

    # Arrange
    cart = ShoppingCart()
    notifications = []

    def track_changes(snapshot):
        notifications.append(
            {
                "item_count": snapshot.item_count,
                "total_price": snapshot.total_price,
                "discount_percent": snapshot.discount_percent,
                "final_price": snapshot.final_price,
            }
        )

    ShoppingCart.subscribe(track_changes)

    # Act - Change different observables
    cart.item_count = 3
    cart.total_price = 29.97
    cart.discount_percent = 10.0

    # Assert - All changes were tracked
    assert len(notifications) >= 3

    # Check that final notification reflects all changes
    final_notification = notifications[-1]
    assert final_notification["item_count"] == 3
    assert final_notification["total_price"] == 29.97
    assert final_notification["discount_percent"] == 10.0
    assert final_notification["final_price"] == 26.973  # 29.97 * (1 - 0.10)


@pytest.mark.integration
@pytest.mark.store
@pytest.mark.observable
@pytest.mark.operators
def test_complex_reactive_workflow_with_multiple_stores():
    """Complex workflow involving multiple stores with interdependent computations"""

    class InventoryStore(Store):
        widgets = observable(100)
        gadgets = observable(50)

        total_items = (widgets + gadgets).then(lambda w, g: w + g)

    class PricingStore(Store):
        base_price_per_item = observable(10.0)
        bulk_discount_threshold = observable(100)

        def __init__(self):
            super().__init__()
            # Connect to inventory changes
            self._inventory_subscription = None

        def connect_inventory(self, inventory_store):
            """Connect pricing calculations to inventory changes"""
            self.inventory_total = inventory_store.total_items
            self.effective_price = (
                self.base_price_per_item
                + self.inventory_total
                + self.bulk_discount_threshold
            ).then(
                lambda base, total, threshold: (
                    base * 0.9 if total >= threshold else base
                )
            )

    # Arrange
    inventory = InventoryStore()
    pricing = PricingStore()
    pricing.connect_inventory(inventory)

    # Act & Assert - Initial state (above bulk threshold)
    assert inventory.total_items.value == 150
    assert pricing.effective_price.value == 9.0  # Bulk discount applied

    # Act - Reduce inventory below threshold
    inventory.widgets = 40
    inventory.gadgets = 30

    # Assert - Price remains full (below threshold)
    assert inventory.total_items.value == 70
    assert pricing.effective_price.value == 10.0

    # Act - Increase inventory above threshold
    inventory.widgets = 80

    # Assert - Discount applied
    assert inventory.total_items.value == 110
    assert pricing.effective_price.value == 9.0  # 10% discount applied
