#!/usr/bin/env python3
import sys

sys.path.insert(0, "/home/fox/Workspace/fynx")

import prototype

# Monkey patch to add debug
original_setattr = prototype.Store.__setattr__


@classmethod
def debug_setattr(cls, name, value):
    print(f"Store.__setattr__(cls={cls.__name__}, name={name}, value={value})")
    print(f'  hasattr _observables: {hasattr(cls, "_observables")}')
    if hasattr(cls, "_observables"):
        print(f"  _observables: {list(cls._observables.keys())}")
        print(f"  name in _observables: {name in cls._observables}")
        if name in cls._observables:
            print(f"  calling set on observable with reg {cls._observables[name]._r}")
    result = original_setattr(name, value)
    return result


prototype.Store.__setattr__ = debug_setattr

# Run the Store part
print("Creating Store...")


class CartStore(prototype.Store):
    item_count = prototype.observable(1)
    price_per_item = prototype.observable(10.0)


print("CartStore._observables:", CartStore._observables)


def calculate_total(count_price_tuple):
    count, price = count_price_tuple
    return count * price


cart_total = (CartStore.item_count + CartStore.price_per_item) >> calculate_total

print(f"Initial total: ${cart_total.value:.2f}")

print("Setting item_count = 3...")
CartStore.item_count = 3
print(f"After 3 items: ${cart_total.value:.2f}")

print("Setting price_per_item = 12.50...")
CartStore.price_per_item = 12.50
print(f"After price change: ${cart_total.value:.2f}")
