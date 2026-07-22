"""Focused tests for the algebraic runtime guarantees."""

import gc
import weakref

import pytest

from fynx import Store, observable, reactive
from fynx.observable.base import Observable, TransformPurityError
from fynx.observable.merged import MergedObservable


@pytest.mark.unit
@pytest.mark.observable
def test_diamond_convergence_recomputes_once_per_source_update():
    """A converging node in a diamond graph recomputes once after stabilization."""
    source = Observable("source", 1)
    calls = {"left": 0, "right": 0, "joined": 0}

    def left_fn(value):
        calls["left"] += 1
        return value + 1

    def right_fn(value):
        calls["right"] += 1
        return value * 2

    def joined_fn(left, right):
        calls["joined"] += 1
        return left + right

    left = source >> left_fn
    right = source >> right_fn
    joined = (left + right) >> joined_fn
    received = []
    joined.subscribe(received.append)

    calls.update({"left": 0, "right": 0, "joined": 0})

    source.set(2)

    assert joined.value == 7
    assert received == [7]
    assert calls == {"left": 1, "right": 1, "joined": 1}


@pytest.mark.unit
@pytest.mark.observable
def test_boolean_diamond_with_merged_guard_notifies_once_per_source_update():
    """A boolean AND with plain and product-derived inputs emits once after convergence."""
    items = observable(0)
    subtotal = items >> (lambda count: count * 10)
    zero = items >> (lambda count: 0)
    total = (subtotal + zero) >> (
        lambda current_subtotal, current_zero: current_subtotal - current_zero
    )
    has_items = items >> (lambda count: count > 0)
    total_positive = total >> (lambda current_total: current_total > 0)
    received = []

    ready = has_items & total_positive
    ready.subscribe(received.append)
    items.set(2)

    assert received == [True]


@pytest.mark.unit
@pytest.mark.observable
def test_nested_boolean_shared_guards_notify_once_per_source_update():
    """Nested boolean ANDs with shared upstream guards emit once after stabilization."""
    items = observable(0)
    subtotal = items >> (lambda count: count * 10)
    discount = (subtotal + items) >> (lambda current_subtotal, count: count * 0)
    total = (subtotal + discount) >> (
        lambda current_subtotal, current_discount: current_subtotal - current_discount
    )
    has_items = items >> (lambda count: count > 0)
    total_positive = total >> (lambda current_total: current_total > 0)
    not_free = discount >> (lambda current_discount: current_discount == 0)
    received = []

    ready = (has_items & total_positive) & not_free
    ready.subscribe(received.append)
    items.set(2)

    assert received == [True]


@pytest.mark.unit
@pytest.mark.observable
def test_reference_api_checkout_example_enables_once_when_cart_becomes_valid():
    """The reference checkout example does not duplicate eligibility effects."""

    class ShoppingCartStore(Store):
        items = observable([])
        discount_code = observable(None)

    subtotal = ShoppingCartStore.items >> (
        lambda items: sum(item["price"] * item["quantity"] for item in items)
    )
    discount_amount = (ShoppingCartStore.items + ShoppingCartStore.discount_code) >> (
        lambda items, code: (
            sum(item["price"] * item["quantity"] for item in items) * 0.20
            if code == "SAVE20"
            else 0.0
        )
    )
    total = (subtotal + discount_amount) >> (lambda sub, disc: sub - disc)
    has_items = ShoppingCartStore.items >> (lambda items: len(items) > 0)
    total_positive = total >> (lambda current_total: current_total > 0)
    can_checkout = has_items & total_positive
    enabled = []

    @reactive(can_checkout)
    def enable_checkout_button(can_checkout_val):
        if can_checkout_val:
            enabled.append(can_checkout_val)

    ShoppingCartStore.items.set([{"price": 20.0, "quantity": 2}])

    assert enabled == [True]


@pytest.mark.unit
@pytest.mark.observable
def test_order_core_conditional_transform_can_start_inactive():
    """Identity transforms over inactive gates preserve the gate instead of crashing."""

    class OrderCore(Store):
        items = observable([])
        address = observable("")
        payment = observable("")
        is_processing = observable(False)

        has_items = items >> (lambda current_items: len(current_items) > 0)
        has_address = address >> bool
        has_payment = payment >> bool
        can_checkout = (has_items @ (has_address & has_payment & ~is_processing)) >> (
            lambda allowed: allowed
        )

    enabled = []
    OrderCore.can_checkout.subscribe(enabled.append)
    OrderCore.items.set(["widget"])
    OrderCore.address.set("123 Main")
    OrderCore.payment.set("card")

    assert enabled == [True]


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_reads_with_hint():
    """Transforms are pure maps over their explicit input values."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    def apply_discount(value):
        return value * (1 - discount.value)

    with pytest.raises(TransformPurityError) as error:
        price >> apply_discount

    message = str(error.value)
    assert "inside a transform" in message
    assert "pass every reactive input explicitly" in message
    assert "price + discount" in message


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_reads_through_helper():
    """The runtime guard catches helpers that static certification cannot inspect."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    def current_discount():
        return discount.value

    with pytest.raises(TransformPurityError, match="inside a transform"):
        price >> (lambda value: value * (1 - current_discount()))


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_defaults():
    """Default arguments cannot smuggle observable reads into a transform."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        price >> (lambda value, hidden=discount: value * (1 - hidden.value))


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_hidden_observable_mutations_with_hint():
    """Transform functions cannot smuggle side effects into the reactive graph."""
    source = Observable("source", 1)
    target = Observable("target", 0)

    def mutate_target(value):
        target.set(value)
        return value

    with pytest.raises(TransformPurityError) as error:
        source >> mutate_target

    message = str(error.value)
    assert "inside a transform" in message
    assert "Move side effects and mutations" in message
    assert target.value == 0


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_transparent_observable_reads():
    """Value-like Observable operations are still reads inside transforms."""
    source = Observable("source", 1)
    flag = Observable("flag", True)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: value if flag else 0)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: f"{value}:{flag}")


@pytest.mark.unit
@pytest.mark.observable
def test_transform_rejects_transparent_store_value_reads():
    """Store descriptors keep their friendly syntax outside transforms only."""

    class Flags(Store):
        enabled = observable(True)

    source = observable(1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        source >> (lambda value: value if Flags.enabled else 0)


@pytest.mark.unit
@pytest.mark.observable
def test_transform_accepts_explicit_product_dependencies():
    """Multi-input transforms stay expressive by combining inputs first."""
    price = Observable("price", 100.0)
    discount = Observable("discount", 0.1)

    discounted = (price + discount) >> (
        lambda current_price, current_discount: current_price * (1 - current_discount)
    )

    assert discounted.value == 90.0

    discount.set(0.2)
    assert discounted.value == 80.0

    price.set(50.0)
    assert discounted.value == 40.0


@pytest.mark.unit
@pytest.mark.observable
def test_transform_chains_are_fused_by_construction():
    """Chained transforms compose onto the original source."""
    source = Observable("source", 2)

    transformed = source >> (lambda value: value + 3) >> (lambda value: value * 10)

    assert transformed._source_observable is source
    assert transformed.value == 50

    source.set(4)
    assert transformed.value == 70


@pytest.mark.unit
@pytest.mark.observable
def test_products_are_canonical_for_ordered_sources():
    """The same ordered product expression reuses the same product node."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    city = Observable("city", "London")

    assert (first + last) is (first + last)
    assert ((first + last) + city) is (first + (last + city))
    assert (last + first) is not (first + last)


@pytest.mark.unit
@pytest.mark.observable
def test_product_cache_does_not_keep_unobserved_products_alive():
    """Canonical products are weakly cached, so unused product nodes can collect."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last
    product_ref = weakref.ref(product)

    del product
    gc.collect()

    assert product_ref() is None


@pytest.mark.unit
@pytest.mark.observable
def test_product_cache_recreates_collected_products():
    """A collected canonical product is recreated cleanly on the next expression."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product_ref = weakref.ref(first + last)

    gc.collect()
    product = first + last

    assert product_ref() is None
    assert product.value == ("Ada", "Lovelace")


@pytest.mark.unit
@pytest.mark.observable
def test_product_cache_does_not_keep_source_graph_alive():
    """The weak product cache does not retain otherwise unreachable sources."""

    def build_product_refs():
        first = Observable("first", "Ada")
        last = Observable("last", "Lovelace")
        product = first + last
        return weakref.ref(first), weakref.ref(last), weakref.ref(product)

    first_ref, last_ref, product_ref = build_product_refs()
    gc.collect()

    assert first_ref() is None
    assert last_ref() is None
    assert product_ref() is None


@pytest.mark.unit
@pytest.mark.observable
def test_canonical_product_subscriptions_are_shared_by_identity():
    """Callbacks attach to the canonical product, whichever expression creates it."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last
    alias = first + last
    received = []

    product.subscribe(
        lambda first_name, last_name: received.append((first_name, last_name))
    )
    last.set("Byron")

    assert alias is product
    assert alias.value == ("Ada", "Byron")
    assert received == [("Ada", "Byron")]


@pytest.mark.unit
@pytest.mark.observable
def test_canonical_product_unsubscribe_removes_source_observers():
    """Removing the final product subscriber returns sources to lazy-only tracking."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last

    def callback(first_name: str, last_name: str) -> None:
        pass

    product.subscribe(callback)
    assert product._source_observer in first._observers

    (first + last).unsubscribe(callback)

    assert product._source_observer not in first._observers
    assert product._source_observer not in last._observers


@pytest.mark.unit
@pytest.mark.observable
def test_canonical_product_subscriber_can_unsubscribe_itself_while_notifying():
    """A product subscriber can remove itself during delivery without lingering."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last
    received = []

    def callback(first_name: str, last_name: str) -> None:
        received.append((first_name, last_name))
        product.unsubscribe(callback)

    product.subscribe(callback)
    last.set("Byron")
    last.set("King")

    assert received == [("Ada", "Byron")]
    assert product._source_observer not in first._observers
    assert product._source_observer not in last._observers


@pytest.mark.unit
@pytest.mark.observable
def test_canonical_product_alias_unsubscribe_preserves_remaining_subscriber():
    """Unsubscribing through one alias keeps other callbacks on the shared product."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last
    alias = first + last
    first_received = []
    second_received = []

    def first_callback(first_name: str, last_name: str) -> None:
        first_received.append((first_name, last_name))

    def second_callback(first_name: str, last_name: str) -> None:
        second_received.append((first_name, last_name))

    product.subscribe(first_callback)
    alias.subscribe(second_callback)
    alias.unsubscribe(first_callback)
    last.set("Byron")

    assert first_received == []
    assert second_received == [("Ada", "Byron")]
    assert product._source_observer in first._observers


@pytest.mark.unit
@pytest.mark.observable
def test_canonical_product_reconstruction_during_notification_reuses_node():
    """Rebuilding a product inside its callback returns the notifying node."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last
    observed_aliases = []

    def callback(first_name: str, last_name: str) -> None:
        alias = first + last
        observed_aliases.append((alias is product, alias.value))

    product.subscribe(callback)
    last.set("Byron")

    assert observed_aliases == [(True, ("Ada", "Byron"))]


@pytest.mark.unit
@pytest.mark.observable
def test_observed_product_cycle_collects_with_unreachable_sources():
    """Observed products and their source observer cycles collect as a unit."""

    def build_refs():
        first = Observable("first", "Ada")
        last = Observable("last", "Lovelace")
        product = first + last

        def callback(first_name: str, last_name: str) -> None:
            pass

        product.subscribe(callback)
        return weakref.ref(first), weakref.ref(last), weakref.ref(product)

    first_ref, last_ref, product_ref = build_refs()
    gc.collect()

    assert first_ref() is None
    assert last_ref() is None
    assert product_ref() is None


@pytest.mark.unit
@pytest.mark.observable
def test_product_exception_after_self_unsubscribe_leaves_lazy_state_clean():
    """A callback can disappear and raise without leaving stale source observers."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last

    def callback(first_name: str, last_name: str) -> None:
        product.unsubscribe(callback)
        raise RuntimeError("boom")

    product.subscribe(callback)

    with pytest.raises(RuntimeError, match="boom"):
        last.set("Byron")

    assert product._is_notifying is False
    assert Observable._notification_scheduled is False
    assert product._source_observer not in first._observers
    assert product._source_observer not in last._observers

    last.set("King")
    assert product.value == ("Ada", "King")


@pytest.mark.unit
@pytest.mark.observable
def test_direct_merged_constructor_does_not_enter_canonical_cache():
    """Only public product construction through +/.alongside() is canonicalized."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")

    direct = MergedObservable(first, last)
    canonical = first + last

    assert direct is not canonical
    assert direct.value == canonical.value


@pytest.mark.unit
@pytest.mark.observable
def test_unobserved_computed_values_recompute_lazily_from_source_versions():
    """Unobserved derived values are invalidated by source version, then read lazily."""
    source = Observable("source", 1)
    calls = {"count": 0}

    def double(value):
        calls["count"] += 1
        return value * 2

    doubled = source >> double
    assert calls["count"] == 1

    source.set(2)
    assert calls["count"] == 1

    assert doubled.value == 4
    assert calls["count"] == 2


@pytest.mark.unit
@pytest.mark.observable
def test_subscribers_create_and_remove_the_eager_frontier():
    """Observed derived values maintain themselves; unobserved values return to lazy mode."""
    source = Observable("source", 1)
    calls = {"count": 0}
    received = []

    def double(value):
        calls["count"] += 1
        return value * 2

    doubled = source >> double
    callback = received.append
    doubled.subscribe(callback)

    source.set(2)
    assert received == [4]
    assert calls["count"] == 2

    doubled.unsubscribe(callback)
    source.set(3)
    assert calls["count"] == 2
    assert doubled.value == 6
    assert calls["count"] == 3


@pytest.mark.unit
@pytest.mark.observable
def test_products_are_lazy_until_observed_and_versioned_when_read():
    """Products do not subscribe upstream until demanded by a subscriber."""
    first = Observable("first", "Ada")
    last = Observable("last", "Lovelace")
    product = first + last

    assert product._dependencies_active is False

    first.set("Augusta")
    assert product._dependencies_active is False
    assert product.value == ("Augusta", "Lovelace")

    received = []
    callback = lambda *values: received.append(values)
    product.subscribe(callback)
    assert product._dependencies_active is True

    last.set("King")
    assert received == [("Augusta", "King")]

    product.unsubscribe(callback)
    assert product._dependencies_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_callable_condition_tracks_external_observable_dependencies():
    """Callable pullbacks subscribe to observables read inside predicates."""
    source = Observable("source", 5)
    limit = Observable("limit", 10)
    filtered = source @ (lambda value: value < limit.value)

    assert filtered.is_active is True

    limit.set(3)
    assert filtered.is_active is False

    limit.set(6)
    assert filtered.is_active is True

    source.set(7)
    assert filtered.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_callable_condition_switches_dynamic_dependencies():
    """Predicate dependencies move when a conditional predicate changes branches."""
    source = Observable("source", 5)
    use_left = Observable("use_left", True)
    left_limit = Observable("left_limit", 10)
    right_limit = Observable("right_limit", 3)

    filtered = source @ (
        lambda value: value
        < (left_limit.value if use_left.value else right_limit.value)
    )

    assert filtered.is_active is True

    right_limit.set(10)
    assert filtered.is_active is True

    use_left.set(False)
    assert filtered.is_active is True

    right_limit.set(4)
    assert filtered.is_active is False

    left_limit.set(1)
    assert filtered.is_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_callable_condition_unsubscribes_from_stale_branch_dependencies():
    """Predicate branch switching removes stale dependency notifications."""
    source = Observable("source", 5)
    use_left = Observable("use_left", True)
    left_limit = Observable("left_limit", 10)
    right_limit = Observable("right_limit", 10)
    received = []

    filtered = source @ (
        lambda value: value
        < (left_limit.value if use_left.value else right_limit.value)
    )
    filtered.subscribe(received.append)
    use_left.set(False)
    left_limit.set(1)
    right_limit.set(4)

    assert received == []
