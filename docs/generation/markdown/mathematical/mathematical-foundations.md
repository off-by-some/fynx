# Mathematical Foundations of FynX

## Introduction

Picture a temperature sensor in your application. Right now it reads 72°F. A moment from now, it might read 73°F. An hour from now, perhaps 68°F. This simple scenario—a value that changes over time—appears constantly in programming: user input, API responses, database queries, anything that arrives on its own schedule.

Most reactive programming libraries handle this straightforwardly enough. You create an observable, transform it a few times, connect some UI elements, and things work. Then you add another layer. Then another. Somewhere around the fifth or sixth transformation, you notice something odd. An update happens in an unexpected order. You add a log to debug it, and suddenly the behavior is easier to explain. You remove the log. The confusion comes back.

That kind of confusion rarely means a reactive system forgot to think about semantics. It usually means the system chose a different answer than the one the application author had in mind: dynamic dependency capture, scheduler-mediated effects, asynchronous tracking boundaries, or batched delivery. Those are legitimate design choices. They are also choices you need to understand before you can reason locally about a program.

FynX commits to a small, explicit algebra instead of leaving that meaning implicit: transform with `>>`, combine with `+`, combine booleans with `&` / `|` / `~`, and gate values with `@`. Category theory supplies names for these patterns, and because those names come with laws attached, they also supply a way to know which rewrites are safe under FynX's semantic contract.

---

## Where Reactive Systems Diverge

That contract is one particular set of answers. Existing reactive systems answer the same underlying questions differently: which reads become dependencies, when updates are batched, whether two effects triggered by the same change run in a defined order relative to each other, and whether effects may safely write back to reactive state. Those choices are reasonable - each solves a real problem for the library that made it - but they mean two visually similar programs in two different systems can obey different rules underneath.

These differences show up as documented behavior in the libraries themselves, not as edge cases someone had to go looking for:

**Logging can misrepresent what was tracked.**

MobX tracks whichever observable properties are actually read while a tracked function runs:

```js
autorun(() => {
  console.log(message);
});

message.updateTitle("Hello world");
```

`console.log(message)` logs the object, but doesn't read `message.title`, so this `autorun` never reads that property and won't rerun when it changes. Browsers can make this more confusing rather than less: some consoles format a logged object lazily, so the printed value can show the *new* title even though the reaction never tracked it. The `console.log` you added to understand the bug can show you a value the reaction never actually saw.

**Two reactions triggered by the same change have no guaranteed order relative to each other.**

MobX's documentation is explicit that reaction order isn't part of its contract:

```js
reaction(() => state.ready, () => persistState());
reaction(() => state.ready, () => notifyServer());
```

Both reactions fire when `state.ready` changes; MobX does not guarantee which runs first. If `notifyServer` assumes `persistState` already completed, that assumption exists in the application, not in the reactive graph.

**Dependency tracking stops at the first `await`.**

Vue's `watchEffect`, Svelte's `$effect`, and MobX's tracked functions all register a dependency only for reads that happen synchronously. A read after that point doesn't count, no matter how visibly it appears in the function:

```js
watchEffect(async () => {
  const user = currentUser.value;   // tracked
  await loadProfile(user);
  console.log(theme.value);         // read, but not tracked
});
```

A later change to `theme` won't rerun this effect, even though `theme.value` is visibly read inside it. Moving that same line above the `await` would track it. The dependency graph reflects when a value was read, not only whether it was read.

**Effects that write derived state hide part of the graph.**

Both Solid and Angular's signal guides recommend against computing a derived value by writing to a signal inside an effect:

```js
// Discouraged
createEffect(() => {
  setTotal(price() * quantity());
});
```

Both recommend a memo or computed value instead - because once an effect is allowed to write the state another effect reads, part of the dependency graph moves out of the declared dependencies and into whatever order the effects happen to run in.

**Batching changes which updates an effect actually sees.**

Vue watchers can run batched (the default) or synchronously:

```js
watchEffect(cb);                    // one call per flush
watchEffect(cb, { flush: "sync" }); // one call per mutation
```

Pushing 1,000 items onto an array one at a time, a batched watcher sees one settled update; a synchronous one sees a thousand. Same data, same code, a different number of observed events, decided by a scheduling flag.

These systems choose dynamic dependency capture and scheduler-mediated effects. FynX chooses explicit graph construction and synchronous state semantics. Dependencies come from composing observables with `+` and `>>`, so a transform can only derive a value from the arguments FynX actually hands it - reading or writing reactive state on the side isn't just discouraged, it raises `TransformPurityError`. That same explicitness carries through to propagation and effects: `@` / `.requiring()` is what restricts propagation, and `@reactive` / `.subscribe()` are the places effects belong.

That still leaves a separate question open: which internal rewrites - fusing a chain, reusing a product node - is FynX itself allowed to make without changing what a program does? Answering that precisely starts with agreeing on what "the same" even means for two expressions.

---

## What We Mean by "Same"

Before talking about functors or products, we need one plain-English definition of what "the same" means for two FynX expressions.

Two FynX expressions are equivalent when, for the same sequence of source updates, they expose the same public values and active/inactive states through the public API. They do not have to be the same Python object. They do not have to allocate the same number of intermediate nodes. They do not promise the same private evaluation counts or memory layout.

This expression:

```python
obs >> (lambda x: x)
```

creates a derived observable that is a different Python object from `obs` - but by the definition above, it's still "the same" as `obs`: it produces the same values, updating whenever `obs`'s source does.

Making that kind of claim - that two things behaving differently underneath are still "the same" - depends on a few practical assumptions holding:

- Transform functions used with `>>` / `.then()` are written as pure value functions in the semantic model. They may only use the plain argument values FynX passes in.
- Extra observable inputs must be made explicit with `+` / `.alongside()`.
- Transform functions may not read `.value` from observables or call `.set()` on observables. FynX raises `TransformPurityError` for those cases.
- Side effects belong at the boundary: `.subscribe()` or `@reactive`.
- FynX's core propagation model is synchronous and state-like: an observable always has a current value unless a conditional gate is inactive.

There is an important distinction here. `TransformPurityError` enforces **reactive isolation**: no hidden observable reads, and no reactive mutations, inside transforms. It does not prove full referential transparency for arbitrary Python functions. A transform can still close over ordinary mutable Python state, perform I/O, raise exceptions, or call an effectful helper unless you choose not to write it that way. The mathematical laws below describe the well-behaved sublanguage: deterministic, total, observationally pure value functions used as FynX transforms. FynX enforces the reactive part of that contract; full functional purity remains a programmer obligation.

With that contract in place, the sections below make each structure precise, starting with the simplest: `>>` as a functor. [Runtime Representation](#runtime-representation-algebra-as-architecture), near the end, comes back to show exactly how the runtime leans on these same laws to fuse transforms, reuse products, and cache lazily - for now, the assumptions above are what everything that follows is built on.

---

## Observables as Functors

### A Value That Changes

Consider that temperature sensor again. Mathematically, we can think of it as a function from time to temperatures:

$$\mathcal{T} \to \text{Temperature}$$

At time $t_1$, it returns 72°F. At time $t_2$, it returns 73°F. What the sensor actually represents is a temporal function - a rule that produces a temperature for each moment, not one stored number.

Now suppose you have a function that converts Celsius to Fahrenheit. It's a simple function on ordinary numbers:

```python
def celsius_to_fahrenheit(c: float) -> float:
    return c * 9/5 + 32
```

Your sensor, though, is an observable rather than a plain number, so applying `celsius_to_fahrenheit` to it means lifting the function to work on the entire temporal function at once.

```python
temp_c: Observable[float] = observable(20.0)
temp_f: Observable[float] = temp_c >> celsius_to_fahrenheit
```

The `>>` operator performs this lifting automatically. Give it a function on values, and it produces a function on observables. Whenever `temp_c` changes, `temp_f` updates accordingly, always maintaining the conversion relationship.

### The Structure Beneath

This lifting operation has real mathematical structure behind it: category theorists call it a functor. As explanatory shorthand, write $\mathcal{O}$ for the operation that takes a value type to its observable type:

$$\mathcal{O}: \mathbf{Type} \to \mathbf{Type}$$

Read this in the restricted semantic category described above: objects are Python value types as FynX observes them, and morphisms are deterministic, total, observationally pure value functions accepted as transforms. Ordinary Python functions are more general than that - they can be partial, effectful, exception-raising, or dependent on mutable external state - so the functor laws are not claims about every callable Python will let you pass. They are the contract for the pure transform sublanguage in which FynX's rewrites are valid.

Within that category, $\mathcal{O}$ takes functions between types to lifted functions between observable types. To behave as a functor in that restricted category, rather than as an arbitrary lifting, a couple of laws have to hold.

The first law says: if you lift the identity function (which does nothing), you get the identity on observables (which also does nothing):

$$\mathcal{O}(\text{id}_A) = \text{id}_{\mathcal{O}(A)}$$

In code, this is the `obs >> (lambda x: x)` example from [What We Mean by "Same"](#what-we-mean-by-same) - a different Python object, possibly a different subscription or caching structure internally, but the same changing value. That it works out this way isn't a coincidence; it's what the identity law promises.

The law only holds in the pure transform model - the identity transform can't mutate state, read hidden observables, or run effects at surprising times. FynX enforces the reactive part of that model at runtime: reading or mutating an observable from inside a transform raises `TransformPurityError`. It remains your responsibility to keep the ordinary Python body of the transform deterministic and side-effect-free if you want the categorical laws to apply fully.

A second law covers composition. If you have two functions $f$ and $g$, then composing them first and then lifting should give the same result as lifting each separately and then composing:

$$\mathcal{O}(g \circ f) = \mathcal{O}(g) \circ \mathcal{O}(f)$$

In practice, these two expressions are mathematically equivalent:

```python
result = obs >> (lambda x: g(f(x)))
result = obs >> f >> g
```

Because they're equivalent, you can split function chains or merge them freely as value-level expressions - the order of functions is preserved, and the public result is the same under the transform-purity contract.

### Why This Matters

Because the composition law holds, FynX can optimize your code automatically. When you write:

```python
obs >> f >> g >> h
```

FynX can internally transform this into:

```python
obs >> (lambda x: h(g(f(x))))
```

Under the transform contract above, the two expressions have the same public value behavior, so the rewrite is safe for any pure transformation, not a fusion built for one specific benchmark.

This is why FynX stays efficient even with deep transformation chains. Fusion removes intermediate reactive-node overhead such as dispatch, subscriptions, and graph traversal. It does not magically remove the work of arbitrary Python functions: evaluating `n` meaningful transformations is still `O(n)` in the number of functions. The win is that the reactive machinery does much less work around those functions.

None of that internal bookkeeping changes how you write the code, though: you chain transformations however makes sense to you, and the mathematical structure is what lets FynX keep that predictable while trimming unnecessary machinery underneath it.

A functor lifts a function of one argument. Plenty of real computations need more than one observable at once - that's the next structure.

---

## Products: Combining Independent Values

### When Two Become One

Consider a simple scenario: you're building a form with first name and last name fields. Each field is independent - users can type into either one at any time. But at some point you need both values together: to display a full name, validate that both fields are filled, or submit them to a server.

You want to combine two separate time-varying values into a single time-varying pair:

```python
first_name: Observable[str] = observable("Jane")
last_name: Observable[str] = observable("Doe")

full_name_data: Observable[tuple[str, str]] = first_name + last_name
```

The `+` operator creates this combination, and category theory has a name for it: a product.

For FynX's state-like observables, the product satisfies this relationship:

$$\mathcal{O}(A) \times \mathcal{O}(B) \cong \mathcal{O}(A \times B)$$

This says that combining two current-value observables produces an observable of current-value pairs. In FynX, `a + b` means "the latest value of `a` alongside the latest value of `b`." When either source changes, the pair reflects the new current snapshot.

This is not the only possible meaning of "combine" in reactive programming. Stream libraries often distinguish `zip`, `combine_latest`, and other event-combination operators. FynX chooses the state-like product semantics because ordinary application state usually needs the current values together.

### The Universal Property

Products come with a universal property: the product is the most general way to capture "both A and B together," so any computation that needs the paired values can be expressed as that shared product plus a downstream transform. Build the pair once; map from it into whatever shape you need.

In practice, that means one product can feed several transforms at once:

```python
product = first_name + last_name

full_name = product >> (lambda first, last: f"{first} {last}")
initials = product >> (lambda first, last: f"{first[0]}.{last[0]}.")
display = product >> (lambda first, last: f"{last}, {first}")
```

All three computations need both names, and at any moment there's exactly one current `(first, last)` pair - so sharing a single product across all three is valid. FynX takes advantage of that directly: while a product node is live, repeating the same ordered source list reuses that node instead of building a new one.

### Symmetry and Associativity

Products are symmetric - order doesn't affect the structure:

```python
ab = first_name + last_name    # Observable[tuple[str, str]]
ba = last_name + first_name    # Observable[tuple[str, str]]
```

These are isomorphic: same information, different tuple order.

They're also associative—grouping doesn't matter:

```python
city = observable("New York")

left = (first_name + last_name) + city
right = first_name + (last_name + city)
```

In FynX these groupings flatten to the same ordered source list, so both values are `(first, last, city)`. Because they collapse to the same list underneath, changes to any source are reflected identically in the next read, regardless of which grouping you happened to write.

Associativity is what lets differently grouped products normalize to the same flattened ordered source sequence. Symmetry says `a + b` and `b + a` contain equivalent information up to swapping positions, but they are not the same ordered product in FynX because tuple order is observable. The implementation therefore uses the narrower, concrete rule: products with the same flattened ordered source sequence may share a live node.

Functors and products both describe how values get computed. Neither one says anything about whether a value should be visible at all - that's what the next structure is for.

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
quality_ok = is_in_range & is_stable & has_signal
valid_reading = sensor_reading @ quality_ok
```

The `valid_reading` observable only has a value when all three conditions are satisfied. Set `sensor_reading` to 42.5, and if all checks pass, `valid_reading` emits 42.5. Set it to 150, and `valid_reading` goes silent—it's no longer active because the range check fails.

This pattern appears everywhere in reactive systems:
- A "Submit" button that's only enabled when a form is valid
- ETL pipelines that only process records matching data quality rules
- Real-time dashboards that filter out anomalous data points
- Shopping carts that only apply discounts when eligibility conditions are met

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

Only passengers who clear every checkpoint reach the gate - fail any one, and you don't continue.

Back in code, that's `data @ (condition1 & condition2)`: `&` builds the boolean condition, and `@` builds the gate that opens once it's `True`.

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

As users move sliders or toggle checkboxes, `filtered_products` reactively updates - showing only items that pass all active filters, because every one of those filters is named right there in the expression, not hidden behind a callback that happens to run later.

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
quality_ok = has_required_fields & timestamp_valid & value_in_bounds & no_duplicates
processable_record = raw_record @ quality_ok

# Transform only valid records
transformed = processable_record >> transform_logic
```

The ETL pipeline automatically filters out malformed, duplicate, or out-of-bounds records.

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
can_submit = username_valid & email_valid & password_strong & age_appropriate
valid_form = form_state @ can_submit

# Button state automatically reflects validation
submit_button.enabled = can_submit.value
```

The form is submittable exactly when every validation passes. The conditional observable tracks that automatically, and UI components bind directly to it.

### The Categorical Structure

The pattern we've been using is best understood as a pullback-like gate, or a fiber over `True`.

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

For an ordinary predicate $p : X \to \mathbb{B}$, the subset of values where $p$ is true can be represented as the pullback of $p$ along the inclusion $\{\text{True}\} \hookrightarrow \mathbb{B}$. That is the precise categorical idea FynX borrows.

FynX applies it in a specialized, state-like runtime. A condition may be a boolean observable derived from the gated source, or from a product of several sources. A conditional observable also carries temporal state: it can be active now, inactive now, or never active so far. Those runtime details are more than the simple set-theoretic pullback; operationally, `@` creates a partial current-value observable that exposes the source value only while the guard's current value is truthy.

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

This is the pullback intuition specialized to FynX's current-value semantics: the conditional observable exposes the source value only while the condition is true.

For multiple pure conditions over one shared state, we're taking the intersection of multiple such fibers:

$$\text{ConditionalObservable}(s, c_1, \ldots, c_n) \cong \{ x \in s \mid c_1(x) \wedge c_2(x) \wedge \cdots \wedge c_n(x) \}$$

When conditions come from several observables, the clean mathematical picture first builds the shared product state, then treats the conditions as predicates on that product. A password check, a confirmation check, and a terms checkbox, for example, are predicates on the form state as a whole, not on any single field. FynX lets you write the ergonomic expression directly, while the explicit product-and-guard interpretation is the one to keep in mind when reasoning about the laws.

### Special Properties

General pullbacks in category theory don't necessarily commute or associate, but FynX separates the commutative Boolean part from the asymmetric gate.

The Boolean part is the easy one: all our conditions map to the same codomain ($\mathbb{B}$), and logical AND is both commutative and associative. These structural properties live directly in `&` / `.all()`:

$$c_1 \& c_2 \equiv c_2 \& c_1$$

The gate, by contrast, remains intentionally asymmetric:

$$\text{data} @ c$$

means "expose `data` while `c` is true." Since Boolean conditions are ordinary observables, you can build the guard first and then apply the gate:

$$\text{data} @ (c_1 \& c_2 \& c_3)$$

### States and Transitions

A conditional observable exists in one of three states. It might have never been active—conditions have never been satisfied. It might be currently active—all conditions are satisfied right now. Or it might be inactive—conditions were satisfied before, but aren't currently.

The implementation represents these states explicitly:

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

You can only access the value when all conditions hold. Distinguishing "never active" from "inactive after being active" is extra temporal bookkeeping on top of the fiber-over-`True` intuition, added because it produces better runtime error messages and clearer behavior when a gate has not produced a value yet.

### Optimization Through Structure

Because pure boolean conditions compose with `&`, FynX can represent compatible conditional chains through one combined guard:

$$\text{obs} @ c_1 @ c_2 @ c_3 \equiv \text{obs} @ (c_1 \& c_2 \& c_3)$$

Instead of three separate conditional observables checking conditions in sequence, compatible conditions can be represented as one conditional with a combined predicate. This preserves public behavior when the conditions are pure, synchronous, and no one observes the intermediate conditional nodes as separate values.

---

## Composition: Building Complex Systems

### Combining the Pieces

Composing these structures together looks like this, for a form validation example:

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
form_ready = is_valid_email & is_strong_password & passwords_match & terms_accepted
valid_form = form_state @ form_ready

# Functor: create submission payload when valid
submission = valid_form >> (lambda state: {
    "email": state[0],
    "password": state[1]
})
```

This pipeline has multiple source observables, derived validations, a product representing the form state, a pullback-like gate filtering to valid states, and a final transformation. Each piece uses one of the structures we've discussed.

The composition works because each structure has a clear contract. Functors transform explicit values. Products combine current values in a well-defined tuple. Gates expose a value only while their boolean conditions are true. You could describe all three contracts without ever saying "functor" or "pullback" - the names aren't required to use these patterns. They're only there to explain why the pieces click together instead of just happening to work.

### The Categorical View

Naming them earns its keep at the system level, though. Writing reactive graphs this way means relying on three general rules instead of accumulating a special case for every new combination of transform, merge, and filter - that's the actual payoff.

This model constrains the API, guides the implementation, and gives the test suite concrete laws to check against - at the level of the algebra, not line-by-line proofs of the Python.

---

## Runtime Representation: Algebra as Architecture

Back in [What We Mean by "Same"](#what-we-mean-by-same), FynX's practical assumptions - purity, explicit inputs, effects only at the boundary - were stated as a contract. This is what keeping that contract buys: pure transform chains fuse as they're built, repeated products reuse the same node instead of rebuilding it, derived values stay lazy until something actually reads them, and a node only pays for eager delivery once something subscribes to it. The next four sections walk through each of those in turn.

That absence of ceremony is itself part of the design: there is no separate optimizer module doing this work after the fact. The important algebraic choices are represented directly by the observable nodes themselves, so the fast representation is the ordinary representation.

None of that is worth much without measurement, though. FynX achieves strong performance in fixed-size synchronous benchmarks while preserving the algebraic semantics described above. The benchmark suite reports concrete graph operations - source updates, chain propagation, fan-out notification, diamond convergence, dynamic condition switching - and stops there. It makes no claim about what any of that means for a rendered UI.

### Functor Composition Fusion

The composition law explains why sequential pure transformations can fuse:

$$\text{obs} \gg f \gg g \gg h \rightarrow \text{obs} \gg (h \circ g \circ f)$$

Instead of forcing every `>>` in a chain to behave like a separately notified runtime node, FynX represents compatible chains as composed functions over the original source. This is why deep chains stay efficient: arbitrary functions still run in order, but the reactive graph does less bookkeeping.

The functor laws explain why this is semantics-preserving: fusing `f` and `g` into one step changes nothing about the pure math. It only changes whether anything can observe `f`'s intermediate result on its own. Call a node that's actually observed like that an **effect boundary**. Up to that point, the chain is just math waiting to be composed; past it, FynX has to actually keep state around to deliver a real notification. Fusion is only valid between two nodes when neither one is an effect boundary.

### Canonical Products

The universal property of products explains why multiple computations needing the same product may share it:

```python
result1 = (a + b) >> f
result2 = (a + b) >> g
result3 = (a + b) >> h
```

Because repeated products with the same ordered sources have equivalent current-value semantics, FynX canonicalizes them while live: the ordered source list works as a concrete key, and `+` / `.alongside()` check it at construction time. There's no separate rewrite pass that comes along afterward and deduplicates things.

Unobserved products remain lazy. They store a cached tuple and a version signature for their sources; if a source changes, the product refreshes on the next read. Subscribed products, by contrast, attach to their sources directly, so they can notify observers immediately when the current tuple changes.

### Version-Based Invalidation

Every observable has a version, and derived values go a step further: they remember the versions of their inputs too. A read checks that signature; if an input version changed, the derived value recomputes once and updates its own version if the public value changed.

This gives FynX lazy caching without a mode switch. In practice, that means unobserved values don't subscribe upstream just to stay warm - they become fresh only when you ask for them.

### The Effect Boundary in Practice

A derived observable becomes an effect boundary the moment it gets a subscriber: FynX then activates whatever dependencies it needs to keep delivering notifications. Drop to zero subscribers, and it returns to lazy, version-checked behavior instead.

That's the whole rule: maintain what's observed, cache and recompute on read what isn't. It's driven by actual demand, not by a search for some theoretically optimal set of nodes to keep warm.

### Boolean Gates

The Boolean algebra of FynX's conditions is why guards compose predictably:

$$\text{obs} @ c_1 @ c_2 @ c_3 \equiv \text{obs} @ (c_1 \& c_2 \& c_3)$$

That predictability comes directly from the representation: FynX represents conditions as explicit gates in the graph itself, so stacking pure boolean conditions preserves public behavior without needing an optimizer pass to prove it after the fact.

### Why This Is Sound

Each of these representation choices comes from the algebraic structure above, not from profiling: transform fusion follows from functor laws, product sharing from product semantics, boolean gate composition from Boolean algebra, and version invalidation from the current-value contract.

Fusion, product sharing, gating, and invalidation are all operations on nodes and edges. It's worth making that structure explicit.

---

## Dependency Graphs and Update Order

### The Derivation DAG

Behind FynX's derived values is a directed acyclic graph. Nodes are observables. Edges represent derivation dependencies - if B depends on A, there's an edge from A to B.

When A changes, everything reachable from A through outgoing edges needs to update. If the graph had cycles, that update would have no defined stopping point - A's change would eventually feed back into A again.

That's exactly what FynX guards against in the derivation graph: transforms may not mutate reactive state, and circular dependency checks catch cases where reactive execution tries to feed directly back into what it currently depends on.

```python
if transform_is_running:
    raise TransformPurityError("Move mutations to .subscribe() or @reactive")

if current_context and observable in current_context.dependencies:
    raise RuntimeError("Circular dependency detected")
```

The complete application-level **effect graph** is a different thing. A subscription is allowed to perform effects, and effects may attempt to write observable state:

```python
a.subscribe(lambda value: b.set(value + 1))
b.subscribe(lambda value: a.set(value - 1))
```

That kind of feedback is not a pure derivation anymore. It lives at the effect boundary, where the runtime must reject circular updates, stabilize them, or leave the responsibility with the application. FynX's purity boundary makes the split visible: derived values are meant to form a DAG; effects are where feedback can enter and where runtime safeguards matter.

### Topological Order

When multiple observables change, the update order matters. Consider:

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

This is what guarantees no pending dependent is processed before its pending dependency.

That is not the same as promising one global total order for independent effects. If two sibling reactions both depend on the same source but not on each other, a topological order may legally run either sibling first:

```text
      source
      /    \
effect A  effect B
```

FynX's contract is dependency order, not arbitrary sibling order. If one effect must happen before another, represent that relationship explicitly in the graph or sequence it outside the reactive callback.

### Batched Processing

FynX uses stabilization passes for efficiency. A root mutation begins notification, and any dependent notifications queued during that propagation are drained together before the system returns to idle:

```python
_pending_notifications: Set["Observable"] = set()

def schedule_notification(observable):
    _pending_notifications.add(observable)
    if not _notification_scheduled:
        process_notifications()
```

This is not the same as a user-visible transaction block where arbitrary sequential `set()` calls are delayed until later - separate top-level `set()` calls usually propagate synchronously, each in its own pass. The batching only kicks in for changes that arise while a propagation is already underway: those are collected, ordered by dependency edges, and drained together within that same stabilization pass.

---

## Performance in Practice

Fusion, canonical products, lazy invalidation, and ordered batching all make a performance claim on top of the correctness one. Here's what that claim measures out to in practice.

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

Those numbers are one snapshot, taken on one machine. What should hold more generally follows from the same algebraic structure discussed earlier, and the implementation is tuned to hit these targets:

**Linear in dependencies**: Each dependency adds constant cost, as it must.

**Lower overhead in deep chains**: Fusion removes intermediate reactive-node overhead. Evaluating `n` arbitrary transformations is still linear in the number of transformations, but graph propagation can avoid behaving like `n` separately subscribed nodes.

**Linear in fan-out**: Each dependent still needs a notification. Product sharing avoids redundant upstream work, so the per-dependent cost stays low and predictable.

**Stabilization-pass amortization**: Notifications queued during propagation share ordering and drain costs within that pass.

These properties are engineering consequences of the algebraic design. The benchmark suite measures whether the implementation is actually achieving them.

### Reading Benchmark Results

Treat the benchmark results as measurements of exactly the structures named in the output: lightweight observables, transformations, subscribers, fan-out dependents, and diamond graphs. A UI can be built on those primitives, but the benchmark does not claim to render or update UI components.

---

## Related Work and Context

The numbers above come from a specific set of structural choices - functors, products, and pullback-like fibers - and FynX didn't invent any of them: functors appear prominently in languages like Haskell and libraries like Scala Cats, product types are fundamental to type theory, and pullbacks have been applied to constraint systems and relational databases.

What FynX does with that existing mathematics - applying it to state-like reactive observables in Python, and using it to guide automatic runtime representation choices - is the specific contribution here.

---

## Conclusion

That contribution comes down to three things: functors explain why pure transformations compose predictably, products explain how current values combine, and pullback-like fibers explain conditional observables. Each of these also justifies a specific runtime optimization: transform fusion, product sharing, and conditional-chain fusion respectively.

You don't need any of this to use FynX: `>>`, `+`, `&`, `|`, `~`, and `@` work intuitively on their own. It's here for the times you want to know why a particular optimization is safe, or why a design decision landed the way it did. The test suite then checks these laws over concrete, constrained cases, which is the honest complement to the mathematical model rather than a machine-checked proof of the whole implementation.
