# @reactive

`@reactive` subscribes a function to one or more observables. It runs once immediately with the current value(s), then again on every subsequent change, until `.unsubscribe()` is called.

## Basic Usage

```python
from fynx import reactive, observable

count = observable(0)

@reactive(count)
def log_count(value):
    print(f"Count: {value}")
# Prints immediately: "Count: 0"

count.set(5)   # Prints: "Count: 5"
count.set(10)  # Prints: "Count: 10"
```

## Decorator Forms

`@reactive` accepts a Store class, a single observable, or multiple observables:

```python
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)

# Store: the function receives a StoreSnapshot
@reactive(UserStore)
def on_user_change(snapshot):
    print(snapshot.name, snapshot.age)

# Single observable: the function receives its value
@reactive(UserStore.name)
def on_name_change(name):
    print(name)

# Multiple observables: one argument per observable
@reactive(UserStore.name, UserStore.age)
def on_either_change(name, age):
    print(name, age)
```

## Execution Timing

`@reactive` fires immediately on decoration whenever it can, then again on every later qualifying change:

* **Store targets** fire immediately with a `StoreSnapshot` of the current state.
* **Observable, computed, or merged targets** fire immediately with the current value.
* **Conditional targets** (built with `@` / `.requiring()`) fire immediately only if the gate is already active at decoration time; otherwise the function waits until the gate opens.

```python
ready = observable(True)

@reactive(ready)
def on_ready(value):
    print(f"Ready: {value}")
# Prints immediately: "Ready: True"

ready.set(False)  # Prints: "Ready: False"
```

## Manual Calls Are Blocked While Subscribed

```python
@reactive(count)
def log_count(value):
    print(f"Count: {value}")

log_count(10)  # Raises fynx.reactive.ReactiveFunctionWasCalled
```

`.unsubscribe()` releases the function back to normal, callable behavior:

```python
log_count.unsubscribe()
log_count(15)  # Works normally now: prints "Count: 15"
```

## Self-Mutation Raises, It Doesn't Loop

A reaction that sets the observable it watches doesn't run forever. The decoration itself runs once immediately, before the subscription is registered, so that first call succeeds quietly - but any later external change raises:

```python
count = observable(0)

@reactive(count)
def increment_forever(value):
    count.set(value + 1)

count.set(5)
# Raises RuntimeError: Circular dependency detected in reactive computation!
```

## Store-Level Reactions

Reacting to a Store class receives a snapshot rather than individual values, and fires on any attribute change:

```python
class UserStore(Store):
    name = observable("Alice")
    age = observable(30)

@reactive(UserStore)
def sync_to_server(snapshot):
    api.post('/user/update', {'name': snapshot.name, 'age': snapshot.age})

UserStore.name = "Bob"  # Triggers sync_to_server
UserStore.age = 31      # Also triggers sync_to_server
```

## Key Properties

* **Eager**: Fires immediately on decoration when its target is active, then again on every later qualifying change
* **Exclusive**: Raises `ReactiveFunctionWasCalled` if called manually while still subscribed
* **Reversible**: `.unsubscribe()` returns the function to normal, callable behavior
* **Side-effect-only by convention**: use `.then()` / `.alongside()` / `.all()` / `.requiring()` to derive values; reserve `@reactive` for effects that leave the reactive graph (I/O, logging, UI updates)

See [Using @reactive](../tutorial/using-reactive.md) for the full walkthrough and [Best Practices](../tutorial/best-practices.md) for anti-patterns and gotchas.

::: fynx.reactive
