from fynx import Store, observable


# Define a store for a shopping cart
class CartStore(Store):
    item_count = observable(1)
    price_per_item = observable(10.0)


def update_ui(total: float):
    print(f">>> Cart Total: ${total:.2f}")


# Link item_count and price_per_item to auto-calculate total_price
combined_observables = CartStore.item_count | CartStore.price_per_item

# The >> operator takes any observable and passes the value(s) to the right.
total_price = combined_observables >> (lambda count, price: count * price)
total_price.subscribe(update_ui)  # Subscribe and update the UI when it changes

print("=" * 50)

# Now whenever we change the cart state, total_price updates automatically,
# and the UI is updated accordingly.
CartStore.item_count = 2
CartStore.price_per_item = 15  #

# ==================================================
# >>> Cart Total: $20.00
# >>> Cart Total: $30.00
