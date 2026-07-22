# Mathematical Foundations of FynX

## Introduction

Picture a temperature sensor in your application. Right now it reads 72°F. A moment from now, it might read 73°F. An hour from now, perhaps 68°F. This simple scenario—a value that changes over time—appears constantly in programming. User input changes. API responses arrive. Database queries complete. The world is full of values that vary.

Most reactive programming libraries handle this straightforwardly enough. You create an observable, transform it a few times, connect some UI elements, and things work. Then you add another layer. Then another. Somewhere around the fifth or sixth transformation, you notice something odd. An update happens in an unexpected order. You add a `console.log` to debug it, and suddenly everything works perfectly. You remove the log. It breaks again.

These aren't always bugs in your code. Often they're symptoms of systems whose rules are hard to state. When a reactive library does not say exactly what a transformation, combination, or filter means, every new feature becomes harder to reason about.

FynX takes a different path. Its API is designed around a small algebra of observables: transform with `>>`, combine with `+`, filter with `&`, choose with `|`, and negate with `~`. Category theory gives useful names for these patterns and, more importantly, tells us which rewrites are safe under FynX's semantic contract.

Let's explore how.

---

## What We Mean by "Same"

Before talking about functors or products, we need one plain-English definition.

Two FynX expressions are equivalent when, for the same sequence of source updates, they expose the same public values and active/inactive states through the public API. They do not have to be the same Python object. They do not have to allocate the same number of intermediate nodes. They do not promise the same private evaluation counts or memory layout.

That distinction matters. This expression:

```python
obs >> (lambda x: x)
```

creates a derived observable. It is not `obs` by object identity. But as a value-level expression, it reflects the same values as `obs` whenever the source changes.

FynX's algebra relies on a few practical assumptions:

- Transform functions used with `>>` / `.then()` are pure. They may only use the plain argument values FynX passes in.
- Extra observable inputs must be made explicit with `+` / `.alongside()`.
- Transform functions may not read `.value` from observables or call `.set()` on observables. FynX raises `TransformPurityError` for those cases.
- Side effects belong at the boundary: `.subscribe()` or `@reactive`.
- FynX's core propagation model is synchronous and state-like: an observable always has a current value unless a conditional gate is inactive.

With that contract in place, the mathematical structures below are more than decoration. They explain why the API composes cleanly and why certain optimizations preserve the behavior users can observe.

FynX now leans into that contract directly in the runtime:

- Pure transform chains are fused as they are built. `obs >> f >> g` keeps the same value behavior as mapping through `f` and then `g`, but it is represented as one composed transform over `obs` when no observed boundary says otherwise.
- Ordered products are canonical while live. Repeating `a + b` gives the same product node, while `b + a` remains a different ordered tuple.
- Derived values use source versions for invalidation. Unobserved nodes stay lazy and refresh only when read after an input version changes.
- Subscribers create the eager frontier. A node with observers maintains itself enough to deliver notifications; a node without observers returns to lazy cached behavior.

---

## Observables as Functors

### A Value That Changes

Consider that temperature sensor again. Mathematically, we can think of it as a function from time to temperatures:

$$\mathcal{T} \to \text{Temperature}$$

At time $t_1$, it returns 72°F. At time $t_2$, it returns 73°F. The sensor isn't storing a single value—it's representing an entire temporal function.

Now suppose you have a function that converts Celsius to Fahrenheit. It's a simple function on ordinary numbers:

```python
def celsius_to_fahrenheit(c: float) -> float:
    return c * 9/5 + 32
```

But your sensor is an observable, not a number. What does it mean to apply this function to something that varies over time? The answer turns out to be elegant: you lift the function to work on the entire temporal function at once.

```python
temp_c: Observable[float] = observable(20.0)
temp_f: Observable[float] = temp_c >> celsius_to_fahrenheit
```

The `>>` operator performs this lifting automatically. Give it a function on values, and it produces a function on observables. Whenever `temp_c` changes, `temp_f` updates accordingly, always maintaining the conversion relationship.

### The Structure Beneath

This lifting operation isn't arbitrary. It has mathematical structure—specifically, it defines what category theorists call a functor. The functor $\mathcal{O}$ takes types to observable types:

$$\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$$

And it takes functions between types to lifted functions between observable types. But there's more. For this to be a proper functor, certain laws must hold.

The first law says: if you lift the identity function (which does nothing), you get the identity on observables (which also does nothing):

$$\mathcal{O}(\text{id}_A) = \text{id}_{\mathcal{O}(A)}$$

In code: `obs >> (lambda x: x)` has the same value behavior as `obs`. It may be a different Python object, and internally it may have a different subscription or caching structure, but it represents the same changing value.

This works because transforms are pure. The identity transform cannot mutate state, read hidden observables, or perform effects at surprising times. FynX enforces the most important observable-specific part of that rule at runtime: reading or mutating an observable from inside a transform raises `TransformPurityError`.

The second law is more interesting. If you have two functions $f$ and $g$, then composing them first and then lifting should give the same result as lifting each separately and then composing:

$$\mathcal{O}(g \circ f) = \mathcal{O}(g) \circ \mathcal{O}(f)$$

What this means in practice is beautiful: these two expressions are mathematically equivalent:

```python
result = obs >> (lambda x: g(f(x)))
result = obs >> f >> g
```

You can split function chains or merge them freely as value-level expressions. The order of functions is preserved, and the public result is the same under the transform-purity contract.

### Why This Matters

Here's where theory meets practice in an unexpected way. Because the composition law holds, FynX can optimize your code automatically. When you write:

```python
obs >> f >> g >> h
```

FynX can internally transform this into:

```python
obs >> (lambda x: h(g(f(x))))
```

This is not a benchmark-specific trick. Under the transform contract above, the two expressions have the same public value behavior. The mathematics tells us why this rewrite is safe for pure transformations.

This is why FynX stays efficient even with deep transformation chains. Fusion removes intermediate reactive-node overhead such as dispatch, subscriptions, and graph traversal. It does not magically remove the work of arbitrary Python functions: evaluating `n` meaningful transformations is still `O(n)` in the number of functions. The win is that the reactive machinery does much less work around those functions.

The elegance here is worth pausing to appreciate. You write code naturally, chaining transformations as makes sense to you. The mathematical structure gives FynX a disciplined way to keep that code predictable while trimming unnecessary machinery.

---

## Products: Combining Independent Values

### When Two Become One

Consider a simple scenario: you're building a form with first name and last name fields. Each field is independent—users can type into either one at any time. But at some point, you need both values together. Maybe to display a full name. Maybe to validate that both fields are filled. Maybe to submit them to a server.

You want to combine two separate time-varying values into a single time-varying pair:

```python
first_name: Observable[str] = observable("Jane")
last_name: Observable[str] = observable("Doe")

full_name_data: Observable[tuple[str, str]] = first_name + last_name
```

The `+` operator creates this combination. But what exactly has happened here? Category theory gives us a precise answer: we've constructed a product.

For FynX's state-like observables, the product satisfies an elegant relationship:

$$\mathcal{O}(A) \times \mathcal{O}(B) \cong \mathcal{O}(A \times B)$$

This says that combining two current-value observables produces an observable of current-value pairs. In FynX, `a + b` means "the latest value of `a` alongside the latest value of `b`." When either source changes, the pair reflects the new current snapshot.

This is not the only possible meaning of "combine" in reactive programming. Stream libraries often distinguish `zip`, `combine_latest`, and other event-combination operators. FynX chooses the state-like product semantics because ordinary application state usually needs the current values together.

### The Universal Property

Products come with something called a universal property. The term sounds abstract, but the idea is practical: the product is the "most general" way to capture "both A and B together."

Formally, once this product has been selected, any computation that consumes the paired current values can be expressed through the product and a downstream transform. That is the practical meaning of the universal property here: build the pair once, then map from that pair into whatever shape you need.

In practice, this manifests in a useful way. Consider:

```python
product = first_name + last_name

full_name = product >> (lambda first, last: f"{first} {last}")
initials = product >> (lambda first, last: f"{first[0]}.{last[0]}.")
display = product >> (lambda first, last: f"{last}, {first}")
```

All three computations need both names. The universal property tells us that using a shared product is semantically valid: there is one canonical current pair of first and last name. FynX makes this concrete for ordered products: while a product node is live, repeating the same ordered source list reuses that node.

### Symmetry and Associativity

Products have nice algebraic properties. They're symmetric—order doesn't affect the structure:

```python
ab = first_name + last_name    # Observable[tuple[str, str]]
ba = last_name + first_name    # Observable[tuple[str, str]]
```

These are isomorphic. Same information, different tuple order. The structure is preserved.

They're also associative—grouping doesn't matter:

```python
city = observable("New York")

left = (first_name + last_name) + city
right = first_name + (last_name + city)
```

In FynX these groupings flatten to the same ordered source list, so both values are `(first, last, city)`. Changes to any source are reflected in the next read and delivered to subscribers.

These properties aren't just mathematical curiosities. They let the runtime share products by construction instead of rediscovering the same pair over and over.

---

## Pullbacks: The Mathematics of Filtering

### Starting with Code

Suppose you're building a data dashboard. You have a sensor reading that changes over time, but you only want to display readings that meet certain quality criteria:

```python
sensor_reading: Observable[float] = observable(42.5)

# Define quality conditions over the current value
is_in_range = sensor_reading >> (lambda x: 0 <= x <= 100)
has_signal = sensor_reading >> (lambda x: x is not None)

# Stability is a temporal idea, so make the previous reading explicit.
previous_reading: Observable[float] = observable(40.0)
is_stable = (sensor_reading + previous_reading) >> (
    lambda current, previous: abs(current - previous) < 5
)

# Only show readings that pass all quality checks
valid_reading = sensor_reading & is_in_range & is_stable & has_signal
```

The `valid_reading` observable only has a value when all three conditions are satisfied. Set `sensor_reading` to 42.5, and if all checks pass, `valid_reading` emits 42.5. Set it to 150, and `valid_reading` goes silent—it's no longer active because the range check fails.

This pattern appears everywhere in reactive systems:
- A "Submit" button that's only enabled when a form is valid
- ETL pipelines that only process records matching data quality rules
- Real-time dashboards that filter out anomalous data points
- Shopping carts that only apply discounts when eligibility conditions are met

Let's understand what's happening here through an analogy, then see the mathematical structure underneath.

### The Airport Security Analogy

Think of it like airport security. Your data stream is passengers trying to board a flight. Each condition is a checkpoint:

```
Passenger Data
     ↓
[Passport Control] ← is_in_range check
     ↓
[Security Screening] ← is_stable check
     ↓
[Boarding Pass Validation] ← has_signal check
     ↓
   Gate
     ↓
  Flight (valid_reading)
```

Only passengers who pass *all* checkpoints make it to the gate. If you fail at any checkpoint, you don't continue. The gate only opens when all checkpoints have been cleared.

When you write `data & condition1 & condition2`, you're building this chain of checkpoints. The conditional observable is the gate at the end—it only opens when all conditions are satisfied.

```python
sensor_reading.set(42.5)  # All conditions pass → gate opens
sensor_reading.set(150)   # Fails range check → gate closes
sensor_reading.set(-10)   # Fails range check → gate stays closed
sensor_reading.set(43.0)  # Within range and within 5 of previous_reading → gate opens again
```

### More Practical Examples

**Example 1: Streamlit Dashboard with Live Filtering**

```python
# User controls
selected_category = observable("Electronics")
min_price = observable(0.0)
max_price = observable(1000.0)
in_stock_only = observable(True)

# Product list from database (populated elsewhere)
products = observable([])

# Keep every reactive input explicit
filtered_products = (
    products
    + selected_category
    + min_price
    + max_price
    + in_stock_only
) >> (
    lambda items, category, min_p, max_p, stock_only: [
        product
        for product in items
        if product.category == category
        and min_p <= product.price <= max_p
        and (not stock_only or product.stock > 0)
    ]
)

# Dashboard automatically updates as users adjust filters
display(filtered_products)
```

As users move sliders or toggle checkboxes, `filtered_products` reactively updates—showing only items that pass all active filters. The dependencies are visible in the expression itself, which keeps the graph easy to inspect and optimize.

**Example 2: ETL Pipeline with Data Quality Gates**

```python
# Raw data ingestion
raw_record = observable({"id": "r-1", "timestamp": 100, "value": 42})
last_processed_time = observable(90)
processed_ids = observable(set())

# Data quality conditions
has_required_fields = raw_record >> (lambda r: all(k in r for k in ['id', 'timestamp', 'value']))
timestamp_valid = (raw_record + last_processed_time) >> (lambda r, last: r['timestamp'] > last)
value_in_bounds = raw_record >> (lambda r: -1000 <= r['value'] <= 1000)
no_duplicates = (raw_record + processed_ids) >> (lambda r, seen: r['id'] not in seen)

# Only process records that pass quality gates
processable_record = raw_record & has_required_fields & timestamp_valid & value_in_bounds & no_duplicates

# Transform only valid records
transformed = processable_record >> transform_logic
```

The ETL pipeline automatically filters out malformed, duplicate, or out-of-bounds records. Each record either passes all quality gates and gets processed, or fails at least one gate.

**Example 3: Form Validation State Machine**

```python
# Form fields
username = observable("")
email = observable("")
password = observable("")
age = observable(0)
form_state = username + email + password + age

# Validation conditions
username_valid = username >> (lambda u: len(u) >= 3 and u.isalnum())
email_valid = email >> (lambda e: "@" in e and "." in e.split("@")[-1])
password_strong = password >> (lambda p: len(p) >= 8 and any(c.isupper() for c in p) and any(c.islower() for c in p))
age_appropriate = age >> (lambda a: 13 <= a <= 120)

# Form only submittable when all validations pass
can_submit = form_state & username_valid & email_valid & password_strong & age_appropriate

# Button state automatically reflects validation
submit_button.enabled = can_submit.is_active
```

The form exists in one of two states: submittable (all validations pass) or not submittable (at least one validation fails). The conditional observable tracks this state automatically, and UI components bind directly to it.

### The Categorical Structure

Now let's see the mathematical structure that makes this work. The pattern we've been using corresponds to what category theorists call a pullback.

In general, a pullback is the limit of a cospan diagram. Given two morphisms pointing to a common object:

```
    X ---f--→ Z ←--g--- Y
```

The pullback is a new object (often written $X \times_Z Y$) along with projections making this diagram commute:

```
    X ×_Z Y ----→ Y
       +          +
       +          + g
       ↓          ↓
       X ---f--→  Z
```

The pullback consists of pairs $(x, y)$ where $f(x) = g(y)$—elements that "agree" when mapped to the common codomain.

FynX applies this construction in a specialized, state-like way. A condition may be a boolean observable derived from the same source, or from a product of several sources. The key requirement is that at any moment it has a current truth value.

```
Source Observable ---c_i--→ Observable[Bool]
```

We're interested in values where all conditions map to `True`. Visually, for a single condition:

```
    ConditionalObservable ----→ {True}
            +                      +
            + π                    +
            ↓                      ↓
    Source Observable ---c--→ Observable[Bool]
```

This is a pullback-like fiber over `True`: the conditional observable exposes the source value only while the condition is true.

For multiple conditions over one shared state, we're taking the intersection of multiple such fibers:

$$\text{ConditionalObservable}(s, c_1, \ldots, c_n) \cong \{ x \in s \mid c_1(x) \wedge c_2(x) \wedge \cdots \wedge c_n(x) \}$$

When conditions come from several observables, you can think of the shared state as the product of those inputs first, then the conditions select the part of that state where every check is true. That keeps the model honest: a password check, a confirmation check, and a terms checkbox are not all predicates on an email string. They are predicates on the form state as a whole.

### Special Properties

Here's something interesting. General pullbacks in category theory don't necessarily commute or associate. FynX's boolean guard sets have a narrower, useful property under the public semantics described above, and there's a specific reason why.

All our conditions map to the same codomain ($\mathbb{B}$). They all filter to the same fiber (`True`). And logical AND is both commutative and associative. These structural properties of Boolean algebra transfer to our pullback construction:

**Guard-order commutativity**: for a fixed source, the order of independent pure guards does not affect the resulting conditional observable:
$$G(G(\text{data}, c_1), c_2) \equiv G(G(\text{data}, c_2), c_1)$$

This does **not** mean `&` is globally commutative. `data & is_ready` means "expose `data` while `is_ready` is true"; `is_ready & data` would expose the boolean readiness value while `data` is truthy. The source and guard roles are intentionally asymmetric.

**Guard-chain associativity**: grouping a sequence of guards does not change the resulting gate:
$$G(G(G(\text{data}, c_1), c_2), c_3) \equiv G(\text{data}, c_1, c_2, c_3)$$

This is specific to FynX's Boolean filtering contract. In the general category-theoretic setting, these properties don't always hold. In FynX, pure boolean conditions can be combined because `and` over truth values is associative and commutative, and the conditional observable exposes the source only when all conditions are active.

### States and Transitions

A conditional observable exists in one of three states. It might have never been active—conditions have never been satisfied. It might be currently active—all conditions are satisfied right now. Or it might be inactive—conditions were satisfied before, but aren't currently.

The implementation respects these states precisely:

```python
@property
def value(self):
    if self.is_active:
        return self._value
    elif self._has_ever_had_valid_value:
        raise ConditionalNotMet("Conditions not currently satisfied")
    else:
        raise ConditionalNeverMet("Conditions never satisfied")
```

You can only access the value when you're in the fiber—when all conditions hold. The distinction between "never active" and "inactive after being active" is extra temporal bookkeeping layered on top of the basic fiber idea. It gives better runtime errors without changing the simple rule: no active condition, no exposed value.

### Optimization Through Structure

Because pure boolean conditions compose with `and`, FynX can fuse compatible conditional chains:

$$\text{obs} \& c_1 \& c_2 \& c_3 \equiv \text{obs} \& (c_1 \wedge c_2 \wedge c_3)$$

Instead of three separate conditional observables checking conditions in sequence, compatible conditions can be represented as one conditional with a combined predicate. This preserves public behavior when the conditions are pure, synchronous, and no one observes the intermediate conditional nodes as separate values.

The mathematics doesn't make every filter optimization automatically valid. It tells us the assumptions under which the optimization is valid.

---

## Composition: Building Complex Systems

### Combining the Pieces

The real power emerges when you compose these structures together. Consider form validation—a common pattern where correctness matters:

```python
email = observable("")
password = observable("")
confirmation = observable("")
terms_accepted = observable(False)

# Product: the form state as one current-value tuple
form_state = email + password + confirmation + terms_accepted

# Functors: lift validation functions
is_valid_email = email >> (lambda e: "@" in e and e.count("@") == 1 and not e.startswith("@") and not e.endswith("@") and all(part for part in e.split("@")[1].split(".") if part))
is_strong_password = password >> (lambda p: len(p) >= 8)

# Product + functor: compare related fields
passwords_match = (password + confirmation) >> (lambda p, c: p == c)

# Pullback-like gate: expose form_state only when all conditions hold
valid_form = form_state & is_valid_email & is_strong_password & passwords_match & terms_accepted

# Functor: create submission payload when valid
submission = valid_form >> (lambda state: {
    "email": state[0],
    "password": state[1]
})
```

This pipeline has multiple source observables, derived validations, a product representing the form state, a pullback-like gate filtering to valid states, and a final transformation. Each piece uses one of the structures we've discussed.

The composition works because each structure has a clear contract. Functors transform explicit values. Products combine current values in a well-defined tuple. Gates expose a value only while their boolean conditions are true. You do not need to know the category-theory names to use these patterns, but the names help explain why the pieces fit together.

### The Categorical View

When you write complex reactive graphs like this, you're relying on a small set of rules rather than a pile of special cases. Functoriality explains why pure transformations compose without surprises. Products explain why related current values can be bundled and reused. Pullback-like gates explain why conditional values are available only when their conditions hold.

This is what mathematical foundations provide here: not a machine-checked proof of every line of Python, but a disciplined model that constrains the API, guides the implementation, and gives tests clear laws to check.

---

## Runtime Representation: Algebra as Architecture

FynX achieves strong performance in fixed-size synchronous benchmarks while preserving the algebraic semantics described above. The benchmark suite reports concrete graph operations—source updates, chain propagation, fan-out notification, diamond convergence, and dynamic condition switching—rather than translating lightweight dependents into UI claims.

There is no separate optimizer module in the normal runtime. The important algebraic choices are represented directly by the observable nodes themselves. That keeps the system easier to inspect: the fast representation is the ordinary representation.

### Functor Composition Fusion

The composition law explains why sequential pure transformations can fuse:

$$\text{obs} \gg f \gg g \gg h \rightarrow \text{obs} \gg (h \circ g \circ f)$$

Instead of forcing every `>>` in a chain to behave like a separately notified runtime node, FynX represents compatible chains as composed functions over the original source. This is why deep chains stay efficient: arbitrary functions still run in order, but the reactive graph does less bookkeeping.

The functor laws explain why this is semantics-preserving for pure transforms when intermediate nodes are not being observed as distinct effect boundaries. If a node is observed, that subscriber becomes an effect boundary and FynX maintains enough eager state to deliver the promised notification.

### Canonical Products

The universal property of products explains why multiple computations needing the same product may share it:

```python
result1 = (a + b) >> f
result2 = (a + b) >> g
result3 = (a + b) >> h
```

Because repeated products with the same ordered sources have equivalent current-value semantics, FynX canonicalizes them while live. The product interpretation supports that sharing, and the ordered source list gives the runtime a simple, concrete key for recognizing it. This is not a separate rewrite pass; it is how `+` / `.alongside()` constructs products.

Unobserved products remain lazy. They store a cached tuple and a version signature for their sources; if a source changes, the product refreshes on the next read. Subscribed products attach to their sources so they can notify observers immediately when the current tuple changes.

### Version-Based Invalidation

Every observable has a version. Derived values remember the versions of their inputs. A read checks that signature; if an input version changed, the derived value recomputes once and updates its own version if the public value changed.

This gives FynX lazy caching without a mode switch. Unobserved values do not subscribe upstream just to stay warm. They become fresh when you ask for them.

### Subscriber Frontier

Subscribers are the effect boundary. If a derived observable has subscribers, FynX activates the dependencies needed to deliver notifications. If it has no subscribers, it can return to lazy, version-checked behavior.

This is the materialization boundary in the current design. It is not a global min-cut optimizer, and it does not claim global optimality. It is a simple rule that follows actual demand: observed nodes are maintained for effects; unobserved nodes are cached and recomputed on read.

### Boolean Gates

The guard-order commutativity and guard-chain associativity of FynX's boolean gates explain why sequential filters compose predictably:

$$\text{obs} \& c_1 \& c_2 \& c_3 \equiv \text{obs} \& (c_1 \wedge c_2 \wedge c_3)$$

FynX still represents conditions as explicit gates rather than hiding them in an optimizer pass. The algebraic structure explains why stacking pure boolean conditions preserves public behavior.

### Why This Is Sound

These aren't benchmark tricks discovered through profiling. The representation choices come from the algebraic structure and are guarded by FynX's public semantics. Composition collapse follows from functor laws. Product sharing follows from product semantics. Boolean gate composition follows from Boolean algebra. Version invalidation preserves the current-value contract while avoiding unnecessary work.

---

## Dependency Graphs and Update Order

### The Hidden DAG

Behind every FynX program is a directed acyclic graph. Nodes are observables. Edges represent dependencies—if B depends on A, there's an edge from A to B.

This graph structure is fundamental to how reactivity works. When A changes, everything reachable from A through outgoing edges needs to update. The acyclic property is crucial—if the graph had cycles, you'd have circular dependencies, and the system couldn't compute a stable state.

FynX guards against the cycles that can arise during reactive execution, especially when a computation or reaction tries to mutate something it currently depends on:

```python
if transform_is_running:
    raise TransformPurityError("Move mutations to .subscribe() or @reactive")

if current_context and observable in current_context.dependencies:
    raise RuntimeError("Circular dependency detected")
```

In everyday use, the most important rule is simple: derived values should derive; effects and mutations belong at the edge of the system.

### Topological Order

When multiple observables change, the update order matters profoundly. Consider:

```python
a = observable(1)
b = a >> (lambda x: x * 2)
c = b >> (lambda x: x + 10)
```

If `a` changes to 5, both `b` and `c` need updating. But `c` depends on `b`, so we must update `b` first. Otherwise `c` would read a stale value and compute incorrectly.

FynX processes pending notifications in dependency order—dependencies before dependents. Conceptually, this looks like Kahn's algorithm over the pending part of the graph:

```python
def order_notifications(pending):
    incoming_count = count_pending_dependencies(pending)
    ready = nodes_with_no_pending_dependencies(incoming_count)
    ordered = []

    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for dependent in pending_dependents(node):
            incoming_count[dependent] -= 1
            if incoming_count[dependent] == 0:
                ready.append(dependent)

    return ordered
```

This ensures that when an observable evaluates, all its dependencies have already been updated. It's not an optimization—it's a correctness requirement.

### Batched Processing

FynX uses stabilization passes for efficiency. A root mutation begins notification, and any dependent notifications queued during that propagation are drained together before the system returns to idle:

```python
_pending_notifications: Set["Observable"] = set()

def schedule_notification(observable):
    _pending_notifications.add(observable)
    if not _notification_scheduled:
        process_notifications()
```

This is not the same as a user-visible transaction block where arbitrary sequential `set()` calls are delayed until later. Separate top-level `set()` calls usually propagate synchronously. The batching boundary is the current stabilization pass: changes that arise while propagation is already happening are collected, ordered by dependency edges, and drained together.

---

## Performance in Practice

### The Numbers

FynX's benchmark suite measures fundamental operations with fixed sizes and fresh worker processes. A sample quick comparison run on Apple Silicon / Python 3.10 reported:

| Workload | FynX median | RxPY median | Ratio |
|----------|------------:|------------:|------:|
| Source → map → sink, 10K events | 9.186 ms | 11.501 ms | 1.25x |
| One-map chain | 1.250 us | 1.500 us | 1.20x |
| Ten-map chain | 1.750 us | 3.542 us | 2.02x |
| Fan-out to 1 sink | 916 ns | 1.250 us | 1.36x |
| Fan-out to 10 sinks | 2.208 us | 3.000 us | 1.36x |
| Running accumulation, 10K events | 10.042 ms | 12.756 ms | 1.27x |

The FynX-only rows additionally report construction memory, update latency percentiles, diamond recomputation counts, and dynamic condition dependency cleanup.

### Scaling Properties

The algebraic structure suggests specific scaling goals, and the implementation is tuned around those goals:

**Linear in dependencies**: Each dependency adds constant cost, as it must.

**Lower overhead in deep chains**: Fusion removes intermediate reactive-node overhead. Evaluating `n` arbitrary transformations is still linear in the number of transformations, but graph propagation can avoid behaving like `n` separately subscribed nodes.

**Linear in fan-out**: Each dependent still needs a notification. Product sharing avoids redundant upstream work, so the per-dependent cost stays low and predictable.

**Stabilization-pass amortization**: Notifications queued during propagation share ordering and drain costs within that pass.

These properties are engineering consequences of the algebraic design. The benchmark suite measures whether the implementation is actually achieving them.

### Reading Benchmark Results

Treat the benchmark results as measurements of exactly the structures named in the output: lightweight observables, transformations, subscribers, fan-out dependents, and diamond graphs. A UI can be built on those primitives, but the benchmark does not claim to render or update UI components.

---

## Related Work and Context

FynX builds on established concepts in functional reactive programming and category theory. Functors appear prominently in languages like Haskell and libraries like Scala Cats. Product types are fundamental to type theory. Pullbacks have been applied to constraint systems and relational databases.

FynX's contribution is applying these structures specifically to reactive observables in Python, with automatic optimization guided by categorical properties. The mathematical foundation isn't novel, but its application to this domain and language is distinctive.

---

## Conclusion

We've explored how category theory provides a useful foundation for reactive programming—not as abstract theory, but as practical structure. Functors explain why pure transformations compose predictably. Products explain how current values combine. Pullback-like fibers explain conditional observables. And these same structures reveal optimization opportunities.

The mathematical foundations serve three purposes: they give users a simple mental model, they show where optimizations should preserve semantics, and they give the implementation concrete laws to test.

Understanding these foundations isn't required to use FynX effectively. The `>>`, `+`, `&`, `|`, and `~` operators work intuitively without knowing category theory. But the mathematics explains why the library behaves as it does, why certain design decisions were made, and why some optimizations are valid under FynX's public contract.

Category theory transforms reactive programming from a collection of patterns into a structured system with explicit laws. The elegance is that complexity becomes composability. Theory becomes tool. And mathematics guides engineering toward correctness without forcing users to become mathematicians first.
